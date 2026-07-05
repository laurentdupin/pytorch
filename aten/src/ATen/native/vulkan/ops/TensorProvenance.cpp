#include <ATen/native/vulkan/ops/TensorProvenance.h>

#ifdef USE_VULKAN_API

#include <ATen/Functions.h>
#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/ops/TensorState.h>
#include <ATen/native/vulkan/ops/VulkanValueTrace.h>
#include <ATen/native/vulkan/planning/ReplayTensorState.h>
#include <c10/core/ScalarType.h>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

struct ProvenanceRegistry final {
  std::mutex mutex;
  uint64_t next_sequence{1};
  std::unordered_map<uint64_t, VulkanTensorProvenanceRecord> by_storage;
};

ProvenanceRegistry& provenance_registry() {
  static ProvenanceRegistry registry;
  return registry;
}

uint64_t provenance_key(const VulkanTensorStateDesc& state) {
  uint64_t key = state.storage_id;
  key ^= state.view_id + 0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2);
  key ^= state.logical_desc_hash + 0x9e3779b97f4a7c15ULL + (key << 6) +
      (key >> 2);
  key ^= state.generation + 0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2);
  return key;
}

uint64_t packed_weight_source_key(const VulkanTensorStateDesc& state) {
  uint64_t key = state.storage_id;
  key ^= state.logical_desc_hash + 0x9e3779b97f4a7c15ULL + (key << 6) +
      (key >> 2);
  key ^= state.generation + 0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2);
  return key;
}

void set_contract_provenance(
    VulkanTensorProvenanceRecord& record,
    const TensorContractProvenance* contract_provenance) {
  if (contract_provenance == nullptr) {
    return;
  }
  if (
      contract_provenance->contract_name != nullptr &&
      contract_provenance->contract_name[0] != '\0') {
    record.contract_name = contract_provenance->contract_name;
  }
  if (
      contract_provenance->contract_family != nullptr &&
      contract_provenance->contract_family[0] != '\0') {
    record.contract_family = contract_provenance->contract_family;
  }
  if (
      contract_provenance->contract_tuple_id != nullptr &&
      contract_provenance->contract_tuple_id[0] != '\0') {
    record.contract_tuple_id = contract_provenance->contract_tuple_id;
  }
  if (
      contract_provenance->contract_materialization_policy != nullptr &&
      contract_provenance->contract_materialization_policy[0] != '\0') {
    record.contract_materialization_policy =
        contract_provenance->contract_materialization_policy;
  }
}

void append_contract_provenance(
    std::ostringstream& stream,
    const VulkanTensorProvenanceRecord& record) {
  if (!record.contract_name.empty()) {
    stream << " contract_name=" << record.contract_name;
  }
  if (!record.contract_family.empty()) {
    stream << " contract_family=" << record.contract_family;
  }
  if (!record.contract_tuple_id.empty()) {
    stream << " contract_tuple_id=" << record.contract_tuple_id;
  }
  if (!record.contract_materialization_policy.empty()) {
    stream << " contract_materialization_policy="
           << record.contract_materialization_policy;
  }
  if (record.has_integer_value_bounds) {
    stream << " integer_value_min=" << record.integer_value_min
           << " integer_value_max=" << record.integer_value_max
           << " integer_value_bounds_source="
           << record.integer_value_bounds_source;
  }
}

std::string describe_known_writer_locked(
    const VulkanTensorStateDesc& state,
    const ProvenanceRegistry& registry) {
  const auto it = registry.by_storage.find(provenance_key(state));
  if (it == registry.by_storage.end()) {
    return "writer=<unknown>";
  }

  const VulkanTensorProvenanceRecord& record = it->second;
  std::ostringstream stream;
  stream << "writer_op=" << record.writer_op
         << " route=" << record.route
         << " sequence=" << record.sequence
         << " storage_id=0x" << std::hex << record.storage_id
         << " view_id=0x" << record.view_id
         << " record_generation=" << std::dec << record.generation
         << " current_generation=" << state.generation
         << " record_logical_hash=0x" << std::hex
         << record.logical_desc_hash
         << " current_logical_hash=0x" << state.logical_desc_hash
         << std::dec;
  append_contract_provenance(stream, record);
  if (
      record.generation != state.generation ||
      record.logical_desc_hash != state.logical_desc_hash) {
    stream << " stale_descriptor=1";
  }
  return stream.str();
}

std::string describe_input_writers_locked(
    const VulkanTensorProvenanceRecord& record) {
  std::ostringstream stream;
  for (size_t idx = 0; idx < record.input_writers.size(); ++idx) {
    if (idx != 0u) {
      stream << " | ";
    }
    stream << "input" << idx << '{' << record.input_writers[idx] << '}';
  }
  return stream.str();
}

uint64_t root_input_key_locked(
    const VulkanTensorStateDesc& state,
    const ProvenanceRegistry& registry) {
  const auto it = registry.by_storage.find(provenance_key(state));
  if (it != registry.by_storage.end() && it->second.root_input_key != 0u) {
    return it->second.root_input_key;
  }
  return packed_weight_source_key(state);
}

Tensor finite_check_source(const Tensor& snapshot) {
  if (
      snapshot.scalar_type() == kHalf ||
      snapshot.scalar_type() == kBFloat16) {
    return snapshot.to(kFloat);
  }
  return snapshot;
}

bool finite_after_write_enabled() {
  const char* env = std::getenv("PYTORCH_VULKAN_CHECK_FINITE_AFTER_WRITE");
  if (!env || env[0] == '\0') {
    return false;
  }
  const std::string value(env);
  return value != "0" && value != "false" && value != "False" &&
      value != "FALSE";
}

bool& finite_after_write_guard() {
  static thread_local bool active = false;
  return active;
}

} // namespace

void record_tensor_provenance(
    const Tensor& output,
    const char* op_name,
    const char* route_name,
    ArrayRef<Tensor> inputs,
    const bool clear_replay_stamp,
    const bool check_finite_after_write,
    const TensorContractProvenance* contract_provenance) {
  if (!output.defined()) {
    return;
  }

  const VulkanTensorStateDesc output_state = inspect_tensor_state(output);
  if (output_state.storage_id == 0u) {
    return;
  }
  if (clear_replay_stamp) {
    utils::clear_replay_tensor_stamp(output);
  }

  const char* writer_op = op_name && op_name[0] != '\0'
      ? op_name
      : "<unknown>";

  std::vector<std::string> deferred_input_states;
  uint64_t deferred_vulkan_input_count = 0;
  uint64_t deferred_missing_input_lease_count = 0;
  if (api::vulkan_deferred_region_plan_logging_enabled()) {
    deferred_input_states.reserve(inputs.size());
    for (const Tensor& input : inputs) {
      const VulkanTensorStateDesc input_state = inspect_tensor_state(input);
      deferred_input_states.emplace_back(describe_tensor_state(input_state));
      if (input_state.storage_id != 0u) {
        ++deferred_vulkan_input_count;
      } else {
        ++deferred_missing_input_lease_count;
      }
    }
  }

  {
    ProvenanceRegistry& registry = provenance_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);

    VulkanTensorProvenanceRecord record;
    record.sequence = registry.next_sequence++;
    record.storage_id = output_state.storage_id;
    record.view_id = output_state.view_id;
    record.generation = output_state.generation;
    record.logical_desc_hash = output_state.logical_desc_hash;
    record.writer_op = writer_op;
    record.route =
        route_name && route_name[0] != '\0' ? route_name : "<unspecified>";
    set_contract_provenance(record, contract_provenance);
    record.output_state = describe_tensor_state(output_state);
    record.input_states.reserve(inputs.size());
    record.input_writers.reserve(inputs.size());
    record.input_state_keys.reserve(inputs.size());
    for (const Tensor& input : inputs) {
      const VulkanTensorStateDesc input_state = inspect_tensor_state(input);
      record.input_states.emplace_back(describe_tensor_state(input_state));
      record.input_writers.emplace_back(
          describe_known_writer_locked(input_state, registry));
      record.input_state_keys.emplace_back(
          root_input_key_locked(input_state, registry));
    }
    if (!record.input_state_keys.empty()) {
      record.root_input_key = record.input_state_keys.front();
    }

    registry.by_storage[provenance_key(output_state)] = std::move(record);
  }

  api::note_vulkan_deferred_region_tensor_write(
      writer_op,
      route_name,
      output_state.storage_id,
      output_state.view_id,
      output_state.generation,
      output_state.logical_desc_hash,
      output_state.storage_offset,
      output_state.buffer_length,
      output_state.is_view,
      describe_tensor_state(output_state),
      deferred_input_states,
      deferred_vulkan_input_count,
      deferred_missing_input_lease_count);

  record_tensor_value_write(output, writer_op, route_name, inputs);

  bool& finite_guard = finite_after_write_guard();
  if (check_finite_after_write && finite_after_write_enabled() && !finite_guard) {
    finite_guard = true;
    try {
      check_tensor_finite(output, writer_op);
      finite_guard = false;
    } catch (...) {
      finite_guard = false;
      throw;
    }
  }
}

void record_tensor_write(
    const Tensor& output,
    const char* op_name,
    const char* route_name,
    ArrayRef<Tensor> inputs,
    const TensorContractProvenance* contract_provenance) {
  record_tensor_provenance(
      output,
      op_name,
      route_name,
      inputs,
      /*clear_replay_stamp=*/true,
      /*check_finite_after_write=*/true,
      contract_provenance);
}

Tensor record_tensor_write_and_return(
    Tensor output,
    const char* op_name,
    const char* route_name,
    ArrayRef<Tensor> inputs,
    const TensorContractProvenance* contract_provenance) {
  record_tensor_write(
      output,
      op_name,
      route_name,
      inputs,
      contract_provenance);
  return output;
}

void record_tensor_alias(
    const Tensor& output,
    const Tensor& base,
    const char* op_name,
    const char* route_name) {
  record_tensor_provenance(
      output,
      op_name,
      route_name,
      {base},
      /*clear_replay_stamp=*/false,
      /*check_finite_after_write=*/false,
      nullptr);
}

Tensor record_tensor_alias_and_return(
    Tensor output,
    const Tensor& base,
    const char* op_name,
    const char* route_name) {
  record_tensor_alias(output, base, op_name, route_name);
  return output;
}

void record_tensor_integer_value_bounds(
    const Tensor& tensor,
    const int64_t min_value,
    const int64_t max_value,
    const char* proof_source) {
  if (!tensor.defined()) {
    return;
  }
  const VulkanTensorStateDesc state = inspect_tensor_state(tensor);
  if (state.storage_id == 0u) {
    return;
  }
  ProvenanceRegistry& registry = provenance_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);

  auto it = registry.by_storage.find(provenance_key(state));
  if (it == registry.by_storage.end()) {
    VulkanTensorProvenanceRecord record;
    record.sequence = registry.next_sequence++;
    record.storage_id = state.storage_id;
    record.view_id = state.view_id;
    record.generation = state.generation;
    record.logical_desc_hash = state.logical_desc_hash;
    record.writer_op = "aten::copy_";
    record.route = "cpu_to_vulkan";
    it = registry.by_storage.emplace(provenance_key(state), std::move(record))
             .first;
  }
  VulkanTensorProvenanceRecord& record = it->second;
  if (
      record.generation != state.generation ||
      record.logical_desc_hash != state.logical_desc_hash) {
    return;
  }
  record.has_integer_value_bounds = true;
  record.integer_value_min = min_value;
  record.integer_value_max = max_value;
  record.integer_value_bounds_source =
      proof_source && proof_source[0] != '\0' ? proof_source : "unknown";
}

bool tensor_integer_values_in_range(
    const Tensor& tensor,
    const int64_t min_inclusive,
    const int64_t max_exclusive) {
  if (!tensor.defined()) {
    return false;
  }
  const VulkanTensorStateDesc state = inspect_tensor_state(tensor);
  if (state.storage_id == 0u) {
    return false;
  }
  ProvenanceRegistry& registry = provenance_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);

  const auto it = registry.by_storage.find(provenance_key(state));
  if (it == registry.by_storage.end()) {
    return false;
  }
  const VulkanTensorProvenanceRecord& record = it->second;
  return record.has_integer_value_bounds &&
      record.generation == state.generation &&
      record.logical_desc_hash == state.logical_desc_hash &&
      record.integer_value_min >= min_inclusive &&
      record.integer_value_max < max_exclusive;
}

std::string describe_tensor_provenance(const Tensor& tensor) {
  const VulkanTensorStateDesc state = inspect_tensor_state(tensor);
  ProvenanceRegistry& registry = provenance_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);

  const auto it = registry.by_storage.find(provenance_key(state));
  if (it == registry.by_storage.end()) {
    std::ostringstream stream;
    stream << "tensor_provenance{writer=<unknown> state={"
           << describe_tensor_state(state) << "}}";
    return stream.str();
  }

  const VulkanTensorProvenanceRecord& record = it->second;
  std::ostringstream stream;
  stream << "tensor_provenance{writer_op=" << record.writer_op
         << " route=" << record.route
         << " sequence=" << record.sequence
         << " storage_id=0x" << std::hex << record.storage_id
         << " view_id=0x" << record.view_id
         << " record_generation=" << std::dec << record.generation
         << " current_generation=" << state.generation
         << " record_logical_hash=0x" << std::hex
         << record.logical_desc_hash
         << " current_logical_hash=0x" << state.logical_desc_hash
         << std::dec;
  append_contract_provenance(stream, record);
  if (
      record.generation != state.generation ||
      record.logical_desc_hash != state.logical_desc_hash) {
    stream << " stale_descriptor=1";
  }
  stream << " output_state={" << record.output_state << '}';
  const std::string input_writers = describe_input_writers_locked(record);
  if (!input_writers.empty()) {
    stream << " input_writers=[" << input_writers << ']';
  }
  stream << " current_state={" << describe_tensor_state(state) << "}}";
  return stream.str();
}

std::string tensor_provenance_writer(const Tensor& tensor) {
  const VulkanTensorStateDesc state = inspect_tensor_state(tensor);
  ProvenanceRegistry& registry = provenance_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);

  const auto it = registry.by_storage.find(provenance_key(state));
  if (it == registry.by_storage.end()) {
    return "unknown";
  }
  return it->second.writer_op.empty() ? "unknown" : it->second.writer_op;
}

std::string tensor_provenance_route(const Tensor& tensor) {
  const VulkanTensorStateDesc state = inspect_tensor_state(tensor);
  ProvenanceRegistry& registry = provenance_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);

  const auto it = registry.by_storage.find(provenance_key(state));
  if (it == registry.by_storage.end()) {
    return "unknown";
  }
  return it->second.route.empty() ? "unknown" : it->second.route;
}

uint64_t tensor_provenance_first_input_key(const Tensor& tensor) {
  const VulkanTensorStateDesc state = inspect_tensor_state(tensor);
  if (state.storage_id == 0u) {
    return 0u;
  }
  ProvenanceRegistry& registry = provenance_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);

  const auto it = registry.by_storage.find(provenance_key(state));
  if (it == registry.by_storage.end()) {
    return packed_weight_source_key(state);
  }
  if (it->second.root_input_key != 0u) {
    return it->second.root_input_key;
  }
  if (!it->second.input_state_keys.empty()) {
    return it->second.input_state_keys.front();
  }
  return packed_weight_source_key(state);
}

bool check_tensor_finite(const Tensor& tensor, const char* consumer_op) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return true;
  }
  if (!c10::isFloatingType(tensor.scalar_type())) {
    return true;
  }

  const Tensor snapshot = tensor.is_vulkan() ? tensor.cpu() : tensor.cpu();
  const Tensor finite_source = finite_check_source(snapshot);
  const Tensor nonfinite_mask =
      at::logical_not(at::isfinite(finite_source)).to(kLong);
  const int64_t nonfinite_count = nonfinite_mask.sum().item<int64_t>();
  const int64_t checked_numel = finite_source.numel();
  if (nonfinite_count == 0) {
    return true;
  }

  std::ostringstream detail;
  detail << "nonfinite_count=" << nonfinite_count
         << " numel=" << checked_numel
         << " state={" << describe_tensor_state(tensor) << "} "
         << describe_tensor_provenance(tensor);
  const char* op_name =
      consumer_op && consumer_op[0] != '\0' ? consumer_op
                                            : "vulkan_prepack::check_tensor_finite";
  api::fail_vulkan(
      api::VulkanFailureClass::KernelIncorrect,
      op_name,
      "NonFiniteTensor",
      detail.str());
  return false;
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
