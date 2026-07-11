#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/api/Sync.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/TensorState.h>
#include <ATen/native/vulkan/planning/TransitionPlanner.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <atomic>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

std::atomic<uint64_t>& cpu_fallback_counter() {
  static std::atomic<uint64_t> counter{0};
  return counter;
}

std::atomic<uint64_t>& sync_readback_counter() {
  static std::atomic<uint64_t> counter{0};
  return counter;
}

std::atomic<uint64_t>& deferred_value_creation_counter() {
  static std::atomic<uint64_t> counter{0};
  return counter;
}

struct VulkanGraphExecutionScope final {
  int64_t token = 0;
  uint64_t cpu_fallback_count = 0;
  uint64_t sync_readback_count = 0;
  uint64_t deferred_value_creation_count = 0;
};

std::vector<VulkanGraphExecutionScope>& graph_execution_scopes_tls() {
  thread_local std::vector<VulkanGraphExecutionScope> scopes;
  return scopes;
}

int64_t& graph_execution_scope_next_token_tls() {
  thread_local int64_t token = 0;
  return token;
}

void record_vulkan_graph_execution_scope_event(
    const VulkanCpuFallbackKind kind) {
  for (VulkanGraphExecutionScope& scope : graph_execution_scopes_tls()) {
    if (kind == VulkanCpuFallbackKind::SyncReadback) {
      ++scope.sync_readback_count;
    } else {
      ++scope.cpu_fallback_count;
    }
  }
}

struct VulkanFallbackPhaseCounters final {
  std::atomic<uint64_t> unknown{0};
  std::atomic<uint64_t> model_setup{0};
  std::atomic<uint64_t> owner_context_create{0};
  std::atomic<uint64_t> owner_forward{0};
  std::atomic<uint64_t> decoder_setup{0};
  std::atomic<uint64_t> positional_embedding_setup{0};
  std::atomic<uint64_t> readback{0};
  std::atomic<uint64_t> test_harness{0};
};

VulkanFallbackPhaseCounters& fallback_phase_counters() {
  static VulkanFallbackPhaseCounters counters;
  return counters;
}

VulkanFallbackPhaseCounters& timed_fallback_phase_counters() {
  static VulkanFallbackPhaseCounters counters;
  return counters;
}

VulkanFallbackPhase& fallback_phase_tls() {
  thread_local VulkanFallbackPhase phase = VulkanFallbackPhase::Unknown;
  return phase;
}

bool& timed_region_tls() {
  thread_local bool enabled = false;
  return enabled;
}

std::atomic<uint64_t>& fallback_phase_counter(
    VulkanFallbackPhaseCounters& counters,
    const VulkanFallbackPhase phase) {
  switch (phase) {
    case VulkanFallbackPhase::ModelSetup:
      return counters.model_setup;
    case VulkanFallbackPhase::OwnerContextCreate:
      return counters.owner_context_create;
    case VulkanFallbackPhase::OwnerForward:
      return counters.owner_forward;
    case VulkanFallbackPhase::DecoderSetup:
      return counters.decoder_setup;
    case VulkanFallbackPhase::PositionalEmbeddingSetup:
      return counters.positional_embedding_setup;
    case VulkanFallbackPhase::Readback:
      return counters.readback;
    case VulkanFallbackPhase::TestHarness:
      return counters.test_harness;
    case VulkanFallbackPhase::Unknown:
      return counters.unknown;
  }
  return counters.unknown;
}

bool env_flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  const std::string s(value);
  return !(s.empty() || s == "0" || s == "false" || s == "False" ||
           s == "FALSE" || s == "off" || s == "OFF");
}

std::string env_string(const char* name) {
  const char* value = std::getenv(name);
  return value ? std::string(value) : std::string();
}

bool cpu_fallback_log_enabled(const VulkanCpuFallbackKind kind) {
  return env_flag_enabled("PYTORCH_VULKAN_LOG_CPU_FALLBACK") ||
      env_flag_enabled("PYTORCH_VULKAN_LOG_FALLBACK") ||
      (kind == VulkanCpuFallbackKind::SyncReadback &&
       env_flag_enabled("PYTORCH_VULKAN_LOG_SYNC_READBACK"));
}

bool cpu_fallback_warn_enabled(const VulkanCpuFallbackKind kind) {
  const std::string policy = env_string("PYTORCH_VULKAN_CPU_FALLBACK");
  return policy == "warn" || policy == "WARN" || cpu_fallback_log_enabled(kind);
}

bool cpu_fallback_error_enabled(const VulkanCpuFallbackKind kind) {
  if (env_flag_enabled("PYTORCH_VULKAN_NO_CPU_FALLBACK")) {
    return true;
  }
  if (
      kind == VulkanCpuFallbackKind::SyncReadback &&
      env_flag_enabled("PYTORCH_VULKAN_FAIL_ON_SYNC_READBACK")) {
    return true;
  }
  const std::string policy = env_string("PYTORCH_VULKAN_CPU_FALLBACK");
  return policy == "error" || policy == "ERROR" || policy == "fail" ||
      policy == "FAIL";
}

const char* fallback_kind_name(const VulkanCpuFallbackKind kind) {
  switch (kind) {
    case VulkanCpuFallbackKind::OpFallback:
      return "cpu_fallback";
    case VulkanCpuFallbackKind::SyncReadback:
      return "sync_readback";
  }
  return "cpu_fallback";
}

const char* phase_name(const VulkanFallbackPhase phase) {
  switch (phase) {
    case VulkanFallbackPhase::Unknown:
      return "unknown";
    case VulkanFallbackPhase::ModelSetup:
      return "model_setup";
    case VulkanFallbackPhase::OwnerContextCreate:
      return "owner_context_create";
    case VulkanFallbackPhase::OwnerForward:
      return "owner_forward";
    case VulkanFallbackPhase::DecoderSetup:
      return "decoder_setup";
    case VulkanFallbackPhase::PositionalEmbeddingSetup:
      return "positional_embedding_setup";
    case VulkanFallbackPhase::Readback:
      return "readback";
    case VulkanFallbackPhase::TestHarness:
      return "test_harness";
  }
  return "unknown";
}

std::string tensor_detail(const Tensor& tensor) {
  std::ostringstream out;
  out << "device=" << tensor.device() << " dtype=" << tensor.scalar_type()
      << " sizes=" << tensor.sizes();
  if (tensor.is_vulkan()) {
    out << " layout=" << tensor.layout();
  }
  return out.str();
}

std::string fallback_detail(
    const VulkanCpuFallbackKind kind,
    ArrayRef<Tensor> tensors) {
  std::ostringstream out;
  out << "kind=" << fallback_kind_name(kind)
      << " phase=" << phase_name(fallback_phase_tls())
      << " inside_timed_forward=" << (timed_region_tls() ? 1 : 0);
  for (const auto i : c10::irange(tensors.size())) {
    out << " tensor" << i << "={" << tensor_detail(tensors[i]) << '}';
  }
  return out.str();
}

bool has_vulkan_tensor(ArrayRef<Tensor> tensors) {
  for (const Tensor& tensor : tensors) {
    if (tensor.defined() && tensor.is_vulkan()) {
      return true;
    }
  }
  return false;
}

int64_t fallback_tensor_bytes(ArrayRef<Tensor> tensors) {
  int64_t bytes = 0;
  bool found = false;
  for (const Tensor& tensor : tensors) {
    if (tensor.defined() && tensor.is_vulkan()) {
      bytes += static_cast<int64_t>(tensor.nbytes());
      found = true;
    }
  }
  return found ? bytes : -1;
}

std::string format_int_array(const IntArrayRef values) {
  std::ostringstream out;
  out << '[';
  for (const auto i : c10::irange(values.size())) {
    if (i != 0) {
      out << ',';
    }
    out << values[i];
  }
  out << ']';
  return out.str();
}

const Tensor* first_vulkan_tensor(ArrayRef<Tensor> tensors) {
  for (const Tensor& tensor : tensors) {
    if (tensor.defined() && tensor.is_vulkan()) {
      return &tensor;
    }
  }
  return nullptr;
}

bool is_conv_weight_layout_repack_transition(
    const char* op_name,
    const char* reason) {
  return op_name != nullptr && reason != nullptr &&
      std::strcmp(op_name, "vulkan_prepack::conv2d_context") == 0 &&
      std::strcmp(reason, "vulkan_weight_cpu_materialization") == 0;
}

bool is_small_control_dtype(const ScalarType dtype, const bool allow_float) {
  return dtype == kBool || dtype == kLong || dtype == kInt ||
      (allow_float && dtype == kFloat);
}

bool is_small_control_tensor(const Tensor& tensor, const bool allow_float) {
  return tensor.defined() && tensor.is_vulkan() &&
      is_small_control_dtype(tensor.scalar_type(), allow_float) &&
      tensor.dim() <= 4 && tensor.numel() <= 16;
}

bool all_vulkan_tensors_are_small_control(
    ArrayRef<Tensor> tensors,
    const bool allow_float) {
  bool found_vulkan_tensor = false;
  for (const Tensor& tensor : tensors) {
    if (!tensor.defined() || !tensor.is_vulkan()) {
      continue;
    }
    found_vulkan_tensor = true;
    if (!is_small_control_tensor(tensor, allow_float)) {
      return false;
    }
  }
  return found_vulkan_tensor;
}

bool is_small_control_tensor_fallback(
    const char* op_name,
    const char* reason,
    ArrayRef<Tensor> tensors,
    const VulkanCpuFallbackKind kind) {
  if (kind != VulkanCpuFallbackKind::OpFallback || op_name == nullptr ||
      reason == nullptr) {
    return false;
  }
  const bool is_control_comparison =
      std::strcmp(op_name, "aten::comparison") == 0;
  if (!all_vulkan_tensors_are_small_control(tensors, is_control_comparison)) {
    return false;
  }
  return std::strcmp(op_name, "aten::binary_op") == 0 ||
      is_control_comparison ||
      std::strcmp(op_name, "aten::cat") == 0 ||
      std::strcmp(op_name, "aten::isin.Tensor_Tensor") == 0 ||
      std::strcmp(op_name, "aten::all") == 0 ||
      std::strcmp(op_name, "aten::any") == 0 ||
      std::strcmp(op_name, "aten::max") == 0 ||
      std::strcmp(op_name, "aten::masked_fill") == 0 ||
      std::strcmp(op_name, "aten::fill_.Scalar") == 0 ||
      std::strcmp(op_name, "aten::to") == 0 ||
      std::strcmp(reason, "bool_not_cpu_fallback") == 0 ||
      std::strcmp(reason, "bool_or_cpu_fallback") == 0 ||
      std::strcmp(reason, "bool_and_cpu_fallback") == 0 ||
      std::strcmp(reason, "small_control_tensor_cpu_fallback") == 0;
}

bool is_small_control_scalar_extraction(
    const char* op_name,
    ArrayRef<Tensor> tensors,
    const VulkanCpuFallbackKind kind) {
  if (kind != VulkanCpuFallbackKind::SyncReadback || op_name == nullptr ||
      std::strcmp(op_name, "aten::_local_scalar_dense") != 0 ||
      !all_vulkan_tensors_are_small_control(tensors, false)) {
    return false;
  }
  for (const Tensor& tensor : tensors) {
    if (tensor.defined() && tensor.is_vulkan() && tensor.numel() != 1) {
      return false;
    }
  }
  return true;
}

void log_fallback_transition(
    const char* op_name,
    const char* reason,
    ArrayRef<Tensor> tensors,
    const VulkanCpuFallbackKind kind) {
  if (!utils::transition_logging_enabled()) {
    return;
  }
  const Tensor* source_tensor = first_vulkan_tensor(tensors);
  std::string source_dtype;
  std::string source_sizes;
  std::string source_strides;
  std::string source_layout;
  const char* source_storage = nullptr;
  if (source_tensor != nullptr) {
    std::ostringstream dtype_out;
    dtype_out << source_tensor->scalar_type();
    source_dtype = dtype_out.str();
    source_sizes = format_int_array(source_tensor->sizes());
    source_strides = format_int_array(source_tensor->strides());
    source_layout = describe_tensor_state(*source_tensor);
    source_storage = "vulkan_tensor";
  }
  const bool conv_weight_layout_repack =
      is_conv_weight_layout_repack_transition(op_name, reason);
  const bool small_control_tensor_fallback =
      is_small_control_tensor_fallback(op_name, reason, tensors, kind);
  const bool small_control_scalar_extraction =
      is_small_control_scalar_extraction(op_name, tensors, kind);
  const char* producer_contract = nullptr;
  const char* consumer_contract = nullptr;
  if (conv_weight_layout_repack) {
    producer_contract = "ConvWeightLayoutRepackTransitionContract";
    consumer_contract = "LegacyConv2DWeightCPURepack";
  } else if (small_control_scalar_extraction) {
    producer_contract = "SmallControlScalarExtractionContract";
    consumer_contract = "PythonControlPlaneScalarConsumer";
  } else if (small_control_tensor_fallback) {
    producer_contract = "SmallControlTensorFallbackContract";
    consumer_contract = "PythonControlPlaneTensorConsumer";
  }
  const char* destination_layout = conv_weight_layout_repack
      ? "legacy_shader_packed_conv_weight"
      : nullptr;
  const char* destination_storage =
      conv_weight_layout_repack ? "TEXTURE_2D" : nullptr;
  std::string detail_string;
  if (conv_weight_layout_repack) {
    detail_string =
        "packer_path=pack_weights;actual_values_required=1;"
        "explicit_unpack_preserved=1;pickle_unpack_preserved=1";
  } else if (small_control_scalar_extraction) {
    detail_string =
        "control_tensor=1;scalar_extraction=1;behavior_neutral=1;"
        "native_kernel_unsupported=1;"
        "host_residency_contract=SmallControlHostResidencyContract.v0;"
        "host_scalar_boundary_preserved=1;host_residency_authorized=0;"
        "host_residency_top_blocker=python_scalar_boundary_required";
  } else if (small_control_tensor_fallback) {
    detail_string =
        "control_tensor=1;small_tensor=1;behavior_neutral=1;"
        "native_kernel_unsupported=1;"
        "host_residency_contract=SmallControlHostResidencyContract.v0;"
        "host_result_reuploaded_to_vulkan=1;host_residency_authorized=0;"
        "host_residency_top_blocker=consumer_chain_proof_missing";
  }
  const char* detail = detail_string.empty() ? nullptr : detail_string.c_str();
  utils::log_vulkan_transition(utils::VulkanTransitionRequest{
      phase_name(fallback_phase_tls()),
      utils::TransitionReason::FallbackMaterialization,
      utils::TransitionKind::Fallback,
      fallback_tensor_bytes(tensors),
      kind == VulkanCpuFallbackKind::SyncReadback,
      kind == VulkanCpuFallbackKind::SyncReadback,
      kind == VulkanCpuFallbackKind::SyncReadback,
      kind == VulkanCpuFallbackKind::SyncReadback,
      op_name ? op_name : "unknown",
      reason ? reason : fallback_kind_name(kind),
      producer_contract,
      consumer_contract,
      {source_dtype.empty() ? nullptr : source_dtype.c_str(),
       source_sizes.empty() ? nullptr : source_sizes.c_str(),
       source_strides.empty() ? nullptr : source_strides.c_str()},
      {source_layout.empty() ? nullptr : source_layout.c_str(),
       source_storage,
       nullptr,
       nullptr},
      {},
      {destination_layout, destination_storage, nullptr, nullptr},
      detail,
  });
}

} // namespace

uint64_t vulkan_cpu_fallback_count() {
  return cpu_fallback_counter().load(std::memory_order_relaxed);
}

uint64_t vulkan_sync_readback_count() {
  return sync_readback_counter().load(std::memory_order_relaxed);
}

uint64_t vulkan_deferred_value_creation_count() {
  return deferred_value_creation_counter().load(std::memory_order_relaxed);
}

int64_t begin_vulkan_graph_execution_scope() {
  int64_t& next_token = graph_execution_scope_next_token_tls();
  TORCH_CHECK(
      next_token < std::numeric_limits<int64_t>::max(),
      "Vulkan graph execution scope token overflow");
  ++next_token;
  graph_execution_scopes_tls().push_back(VulkanGraphExecutionScope{next_token});
  return next_token;
}

std::vector<int64_t> end_vulkan_graph_execution_scope(const int64_t token) {
  std::vector<VulkanGraphExecutionScope>& scopes = graph_execution_scopes_tls();
  TORCH_CHECK(
      !scopes.empty(),
      "Vulkan graph execution scope end without a matching begin");
  const VulkanGraphExecutionScope& scope = scopes.back();
  TORCH_CHECK(
      scope.token == token,
      "Vulkan graph execution scopes must end in LIFO order: expected token ",
      scope.token,
      ", got ",
      token);
  const std::vector<int64_t> counts = {
      static_cast<int64_t>(scope.cpu_fallback_count),
      static_cast<int64_t>(scope.sync_readback_count),
      static_cast<int64_t>(scope.deferred_value_creation_count)};
  scopes.pop_back();
  return counts;
}

bool vulkan_graph_execution_scope_active() {
  return !graph_execution_scopes_tls().empty();
}

void guard_vulkan_deferred_value_registration(const char* producer) {
  std::vector<VulkanGraphExecutionScope>& scopes = graph_execution_scopes_tls();
  TORCH_CHECK(
      scopes.empty(),
      "Vulkan graph execution cannot register a deferred value from ",
      producer ? producer : "an unnamed producer");
  deferred_value_creation_counter().fetch_add(1, std::memory_order_relaxed);
}

void reset_vulkan_fallback_counters() {
  cpu_fallback_counter().store(0, std::memory_order_relaxed);
  sync_readback_counter().store(0, std::memory_order_relaxed);
}

std::vector<int64_t> vulkan_fallback_phase_counters_snapshot() {
  const auto& counters = fallback_phase_counters();
  return {
      static_cast<int64_t>(counters.unknown.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.model_setup.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.owner_context_create.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.owner_forward.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.decoder_setup.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.positional_embedding_setup.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.readback.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.test_harness.load(std::memory_order_relaxed)),
  };
}

std::vector<int64_t> vulkan_timed_fallback_phase_counters_snapshot() {
  const auto& counters = timed_fallback_phase_counters();
  return {
      static_cast<int64_t>(counters.unknown.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.model_setup.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.owner_context_create.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.owner_forward.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.decoder_setup.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.positional_embedding_setup.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.readback.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.test_harness.load(std::memory_order_relaxed)),
  };
}

void reset_vulkan_fallback_phase_counters() {
  auto& counters = fallback_phase_counters();
  counters.unknown.store(0, std::memory_order_relaxed);
  counters.model_setup.store(0, std::memory_order_relaxed);
  counters.owner_context_create.store(0, std::memory_order_relaxed);
  counters.owner_forward.store(0, std::memory_order_relaxed);
  counters.decoder_setup.store(0, std::memory_order_relaxed);
  counters.positional_embedding_setup.store(0, std::memory_order_relaxed);
  counters.readback.store(0, std::memory_order_relaxed);
  counters.test_harness.store(0, std::memory_order_relaxed);

  auto& timed_counters = timed_fallback_phase_counters();
  timed_counters.unknown.store(0, std::memory_order_relaxed);
  timed_counters.model_setup.store(0, std::memory_order_relaxed);
  timed_counters.owner_context_create.store(0, std::memory_order_relaxed);
  timed_counters.owner_forward.store(0, std::memory_order_relaxed);
  timed_counters.decoder_setup.store(0, std::memory_order_relaxed);
  timed_counters.positional_embedding_setup.store(0, std::memory_order_relaxed);
  timed_counters.readback.store(0, std::memory_order_relaxed);
  timed_counters.test_harness.store(0, std::memory_order_relaxed);
}

void set_vulkan_fallback_phase(const VulkanFallbackPhase phase) {
  fallback_phase_tls() = phase;
}

void set_vulkan_benchmark_timed_region(const bool enabled) {
  timed_region_tls() = enabled;
}

VulkanFallbackPhase current_vulkan_fallback_phase() {
  return fallback_phase_tls();
}

bool current_vulkan_benchmark_timed_region() {
  return timed_region_tls();
}

const char* vulkan_fallback_phase_name(const VulkanFallbackPhase phase) {
  return phase_name(phase);
}

void report_vulkan_cpu_fallback(
    const char* op_name,
    const char* reason,
    ArrayRef<Tensor> tensors,
    const VulkanCpuFallbackKind kind) {
  if (!tensors.empty() && !has_vulkan_tensor(tensors)) {
    return;
  }
  api::flush_vulkan_lazy_chain_boundary(
      kind == VulkanCpuFallbackKind::SyncReadback ? "sync_readback"
                                                  : "cpu_fallback",
      reason ? reason : fallback_kind_name(kind));

  if (kind == VulkanCpuFallbackKind::SyncReadback) {
    sync_readback_counter().fetch_add(1, std::memory_order_relaxed);
    api::vulkan_sync_counters().fallback_sync_readback_count.fetch_add(
        1, std::memory_order_relaxed);
  } else {
    cpu_fallback_counter().fetch_add(1, std::memory_order_relaxed);
  }
  record_vulkan_graph_execution_scope_event(kind);
  fallback_phase_counter(fallback_phase_counters(), fallback_phase_tls())
      .fetch_add(1, std::memory_order_relaxed);
  if (timed_region_tls()) {
    fallback_phase_counter(timed_fallback_phase_counters(), fallback_phase_tls())
        .fetch_add(1, std::memory_order_relaxed);
  }
  log_fallback_transition(op_name, reason, tensors, kind);

  const std::string detail = fallback_detail(kind, tensors);
  const std::string message = api::format_vulkan_failure(
      api::VulkanFailureClass::Unsupported, op_name, reason, detail);
  if (cpu_fallback_warn_enabled(kind)) {
    TORCH_WARN(message);
  }
  if (cpu_fallback_log_enabled(kind)) {
    api::log_vulkan_failure(
        api::VulkanFailureClass::Unsupported, op_name, reason, detail);
  }
  if (cpu_fallback_error_enabled(kind)) {
    api::fail_vulkan(
        api::VulkanFailureClass::Unsupported, op_name, reason, detail);
  }
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
