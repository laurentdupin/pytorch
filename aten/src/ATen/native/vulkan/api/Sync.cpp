#include <ATen/native/vulkan/api/Sync.h>

#ifdef USE_VULKAN_API

#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <tuple>

namespace at {
namespace native {
namespace vulkan {
namespace api {

namespace {

thread_local VulkanVisionStackPhase g_vision_stack_phase =
    VulkanVisionStackPhase::Unknown;
thread_local int64_t g_vision_stack_block_index = -1;
thread_local std::vector<int64_t> g_vision_stack_capture_indices;
thread_local VulkanSubmitPhase g_submit_phase = VulkanSubmitPhase::Unknown;
thread_local VulkanRetiredResourceKind g_retired_resource_kind =
    VulkanRetiredResourceKind::Unknown;
thread_local VulkanRetiredResourceRole g_retired_resource_role =
    VulkanRetiredResourceRole::Unknown;
thread_local std::vector<VulkanStackLastUseProof> g_stack_last_use_proofs;
thread_local uint64_t g_stack_dispatch_dependency_scope_id = 0u;
thread_local uint64_t g_stack_dispatch_dependency_position = 0u;
std::atomic<uint64_t> g_next_stack_dispatch_dependency_scope_id{1u};

void maybe_write_stack_region_dependency_graph_dump();

struct RetiredResourceAggregateKey final {
  VulkanRetiredResourceKind kind = VulkanRetiredResourceKind::Unknown;
  VulkanRetiredResourceRole role = VulkanRetiredResourceRole::Unknown;
  VulkanSubmitPhase phase = VulkanSubmitPhase::Unknown;
  VulkanRetireCallSite callsite = VulkanRetireCallSite::Unknown;
  VulkanVisionStackPhase stack_phase = VulkanVisionStackPhase::Unknown;
  int64_t block_index = -1;
  VulkanStackTensorLifetimeClass lifetime =
      VulkanStackTensorLifetimeClass::Unknown;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  int64_t dtype = -1;
  bool direct_buffer = false;
  bool buffer_storage = false;
  bool image_storage = false;
  bool escapes_stack = false;
  bool requested_intermediate = false;
  bool final_output = false;
  bool alias_or_view = false;
  bool has_last_use_proof = false;
  VulkanVisionStackPhase expected_consumer_phase =
      VulkanVisionStackPhase::Unknown;
  int64_t expected_consumer_block_index = -1;
  bool final_consumer_before_stack_submit = false;
  bool internal_non_escaping = false;
  bool aliases_runtime_input = false;
  bool aliases_runtime_output = false;
  bool has_stack_provenance = false;

  bool operator<(const RetiredResourceAggregateKey& other) const {
    return std::tie(
               kind,
               role,
               phase,
               callsite,
               stack_phase,
               block_index,
               lifetime,
               shape,
               strides,
               dtype,
               direct_buffer,
               buffer_storage,
               image_storage,
               escapes_stack,
               requested_intermediate,
               final_output,
               alias_or_view,
               has_last_use_proof,
               expected_consumer_phase,
               expected_consumer_block_index,
               final_consumer_before_stack_submit,
               internal_non_escaping,
               aliases_runtime_input,
               aliases_runtime_output,
               has_stack_provenance) <
        std::tie(
               other.kind,
               other.role,
               other.phase,
               other.callsite,
               other.stack_phase,
               other.block_index,
               other.lifetime,
               other.shape,
               other.strides,
               other.dtype,
               other.direct_buffer,
               other.buffer_storage,
               other.image_storage,
               other.escapes_stack,
               other.requested_intermediate,
               other.final_output,
               other.alias_or_view,
               other.has_last_use_proof,
               other.expected_consumer_phase,
               other.expected_consumer_block_index,
               other.final_consumer_before_stack_submit,
               other.internal_non_escaping,
               other.aliases_runtime_input,
               other.aliases_runtime_output,
               other.has_stack_provenance);
  }
};

struct RetiredResourceAggregateValue final {
  uint64_t count = 0u;
  uint64_t bytes = 0u;
  uint64_t queue_submit_count = 0u;
  uint64_t blocking_wait_count = 0u;
  uint64_t poll_only_count = 0u;
};

struct StackTempLifetimeSafetyKey final {
  VulkanRetiredResourceRole role = VulkanRetiredResourceRole::Unknown;
  VulkanStackTempLifetimeSafety safety =
      VulkanStackTempLifetimeSafety::Unknown;
  VulkanSubmitPhase phase = VulkanSubmitPhase::Unknown;
  VulkanRetireCallSite callsite = VulkanRetireCallSite::Unknown;
  VulkanVisionStackPhase stack_phase = VulkanVisionStackPhase::Unknown;
  int64_t block_index = -1;
  VulkanStackTensorLifetimeClass lifetime =
      VulkanStackTensorLifetimeClass::Unknown;
  std::vector<int64_t> shape;
  int64_t dtype = -1;
  bool has_last_use_proof = false;
  VulkanVisionStackPhase expected_consumer_phase =
      VulkanVisionStackPhase::Unknown;
  int64_t expected_consumer_block_index = -1;
  bool final_consumer_before_stack_submit = false;
  bool internal_non_escaping = false;
  bool escapes_stack = false;
  bool requested_intermediate = false;
  bool final_output = false;
  bool alias_or_view = false;
  bool aliases_runtime_input = false;
  bool aliases_runtime_output = false;
  bool has_stack_provenance = false;

  bool operator<(const StackTempLifetimeSafetyKey& other) const {
    return std::tie(
               role,
               safety,
               phase,
               callsite,
               stack_phase,
               block_index,
               lifetime,
               shape,
               dtype,
               has_last_use_proof,
               expected_consumer_phase,
               expected_consumer_block_index,
               final_consumer_before_stack_submit,
               internal_non_escaping,
               escapes_stack,
               requested_intermediate,
               final_output,
               alias_or_view,
               aliases_runtime_input,
               aliases_runtime_output,
               has_stack_provenance) <
        std::tie(
               other.role,
               other.safety,
               other.phase,
               other.callsite,
               other.stack_phase,
               other.block_index,
               other.lifetime,
               other.shape,
               other.dtype,
               other.has_last_use_proof,
               other.expected_consumer_phase,
               other.expected_consumer_block_index,
               other.final_consumer_before_stack_submit,
               other.internal_non_escaping,
               other.escapes_stack,
               other.requested_intermediate,
               other.final_output,
               other.alias_or_view,
               other.aliases_runtime_input,
               other.aliases_runtime_output,
               other.has_stack_provenance);
  }
};

struct StackTempLifetimeSafetyValue final {
  uint64_t count = 0u;
  uint64_t bytes = 0u;
  uint64_t queue_submit_count = 0u;
  uint64_t blocking_wait_count = 0u;
  uint64_t poll_only_count = 0u;
};

struct StackScratchArenaLifetimeKey final {
  uint64_t arena_id = 0u;
  uint64_t generation = 0u;
  VulkanSubmitPhase phase = VulkanSubmitPhase::Unknown;
  VulkanRetireCallSite callsite = VulkanRetireCallSite::Unknown;
  VulkanVisionStackPhase first_producer_phase = VulkanVisionStackPhase::Unknown;
  int64_t first_producer_block = -1;
  VulkanVisionStackPhase last_consumer_phase = VulkanVisionStackPhase::Unknown;
  int64_t last_consumer_block = -1;
  bool submitted_with_stack_timeline = false;
  bool escapes_stack = false;
  bool aliases_runtime_input = false;
  bool aliases_runtime_output = false;
  bool safe_to_retire_on_stack_submit = false;

  bool operator<(const StackScratchArenaLifetimeKey& other) const {
    return std::tie(
               arena_id,
               generation,
               phase,
               callsite,
               first_producer_phase,
               first_producer_block,
               last_consumer_phase,
               last_consumer_block,
               submitted_with_stack_timeline,
               escapes_stack,
               aliases_runtime_input,
               aliases_runtime_output,
               safe_to_retire_on_stack_submit) <
        std::tie(
               other.arena_id,
               other.generation,
               other.phase,
               other.callsite,
               other.first_producer_phase,
               other.first_producer_block,
               other.last_consumer_phase,
               other.last_consumer_block,
               other.submitted_with_stack_timeline,
               other.escapes_stack,
               other.aliases_runtime_input,
               other.aliases_runtime_output,
               other.safe_to_retire_on_stack_submit);
  }
};

struct StackScratchArenaLifetimeValue final {
  uint64_t count = 0u;
  uint64_t bytes = 0u;
  uint64_t queue_submit_count = 0u;
  uint64_t blocking_wait_count = 0u;
  uint64_t poll_only_count = 0u;
};

constexpr const char* kDryRunProvenStackActivation =
    "proven_stack_activation";
constexpr const char* kDryRunMissingStackActivationProof =
    "missing_stack_activation_proof";
constexpr const char* kDryRunCaptureSensitiveStackActivation =
    "capture_sensitive_stack_activation";
constexpr const char* kDryRunAttentionSubresource =
    "attention_subresource";
constexpr const char* kDryRunAttentionScoreProbabilitySubresource =
    "attention_score_probability_subresource";
constexpr const char*
    kDryRunAttentionBufferGenerationRangeMissingStackProof =
        "attention_buffer_generation_range_missing_stack_proof";
constexpr const char*
    kDryRunAttentionRawGenerationRangeMissingStackProof =
        "attention_raw_generation_range_missing_stack_proof";
constexpr const char*
    kDryRunAttentionScoreProbabilityRangeMissingAliasEscapeProof =
        "attention_score_probability_range_missing_alias_escape_proof";
constexpr const char*
    kDryRunAttentionRawAuxiliaryRangeMissingAliasEscapeProof =
        "attention_raw_auxiliary_range_missing_alias_escape_proof";
constexpr const char*
    kDryRunAttentionScoreProbabilityRangeNonEscapeLastConsumer =
        "attention_score_probability_range_non_escape_last_consumer";
constexpr const char*
    kDryRunAttentionRawAuxiliaryRangeNonEscapeLastConsumer =
        "attention_raw_auxiliary_range_non_escape_last_consumer";
constexpr const char* kDryRunAttentionProvenanceMissingLastUse =
    "attention_provenance_missing_last_use";
constexpr const char* kDryRunAttentionUnknownSubresource =
    "attention_unknown_subresource";
constexpr const char* kDryRunLayerNormStatBuffer =
    "layernorm_stat_buffer";
constexpr const char* kDryRunLayerNormInternalStatBuffer =
    "layernorm_internal_stat_buffer";
constexpr const char* kDryRunMetadataUniform = "metadata_uniform";
constexpr const char* kDryRunRawNoProvenance = "raw_no_provenance";
constexpr const char* kDryRunNonStackSetupStagingPending =
    "non_stack_setup_staging_pending";
constexpr const char* kDryRunUnscopedRawBufferNoStackProof =
    "unscoped_raw_buffer_no_stack_proof";
constexpr const char* kDryRunStackInternalRawMissingGeneration =
    "stack_internal_raw_missing_generation";
constexpr const char* kDryRunStackInternalRawGenerationRange =
    "stack_internal_raw_generation_range";
constexpr const char*
    kDryRunStackInternalTempRawGenerationRangeMissingLastConsumer =
        "stack_internal_temp_raw_generation_range_missing_last_consumer";
constexpr const char*
    kDryRunStackQkvOutputRawGenerationRangeNonEscapeLastConsumer =
        "stack_qkv_output_raw_generation_range_non_escape_last_consumer";
constexpr const char*
    kDryRunStackProjOutputRawGenerationRangeNonEscapeLastConsumer =
        "stack_proj_output_raw_generation_range_non_escape_last_consumer";
constexpr const char*
    kDryRunStackResidual1OutputRawGenerationRangeNonEscapeLastConsumer =
        "stack_residual1_output_raw_generation_range_non_escape_last_consumer";
constexpr const char* kDryRunTrulyUnknownRawResource =
    "truly_unknown_raw_resource";
constexpr const char* kDryRunHostVisibleOrRequestedOutput =
    "host_visible_or_requested_output";
constexpr const char* kDryRunAllocatorOrScratchBacking =
    "allocator_or_scratch_backing";

bool stack_shapes_match(
    const std::vector<int64_t>& lhs,
    const std::vector<int64_t>& rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (lhs.size() + 1 == rhs.size() && rhs.front() == 1) {
    return std::equal(lhs.begin(), lhs.end(), rhs.begin() + 1);
  }
  if (rhs.size() + 1 == lhs.size() && lhs.front() == 1) {
    return std::equal(rhs.begin(), rhs.end(), lhs.begin() + 1);
  }
  return false;
}

const VulkanStackLastUseProof* find_stack_last_use_proof(
    const VulkanVisionStackPhase phase,
    const int64_t block_index,
    const VulkanRetiredResourceRole role,
    const std::vector<int64_t>& shape,
    const int64_t dtype) {
  for (const VulkanStackLastUseProof& proof : g_stack_last_use_proofs) {
    if (
        proof.producer_phase == phase &&
        proof.producer_block_index == block_index &&
        proof.producer_role == role && proof.dtype == dtype &&
        stack_shapes_match(shape, proof.shape)) {
      return &proof;
    }
  }
  return nullptr;
}

std::mutex& stack_aggregate_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<std::string, uint64_t>& stack_dispatch_aggregate() {
  static std::map<std::string, uint64_t> aggregate;
  return aggregate;
}

struct StackAllocationValue final {
  uint64_t count = 0u;
  uint64_t bytes = 0u;
  uint64_t peak_live_estimate_bytes = 0u;
};

std::map<std::string, StackAllocationValue>& stack_allocation_aggregate() {
  static std::map<std::string, StackAllocationValue> aggregate;
  return aggregate;
}

struct StackDispatchDependencyDispatchValue final {
  uint64_t count = 0u;
  uint64_t first_position = 0u;
  uint64_t last_position = 0u;
};

struct StackDispatchDependencyDryRunValue final {
  uint64_t count = 0u;
  uint64_t bytes = 0u;
  uint64_t queue_submit_count = 0u;
  uint64_t fully_proven_count = 0u;
};

std::map<std::string, StackDispatchDependencyDispatchValue>&
stack_dispatch_dependency_dispatch_rows() {
  static std::map<std::string, StackDispatchDependencyDispatchValue> rows;
  return rows;
}

std::map<std::string, StackDispatchDependencyDryRunValue>&
stack_dispatch_dependency_dry_run_rows() {
  static std::map<std::string, StackDispatchDependencyDryRunValue> rows;
  return rows;
}

std::string stack_dispatch_dependency_dispatch_key(
    const uint64_t scope_id,
    const VulkanVisionStackPhase phase,
    const int64_t block,
    const char* const shader_name) {
  std::ostringstream key;
  key << "dispatch=1"
      << " scope_id=" << scope_id
      << " phase=" << vision_stack_phase_name(phase)
      << " block=" << block
      << " shader=" << (shader_name && shader_name[0] ? shader_name : "unknown");
  return key.str();
}

const StackDispatchDependencyDispatchValue* find_stack_dispatch_observation(
    const uint64_t scope_id,
    const VulkanVisionStackPhase phase,
    const int64_t block) {
  const std::string prefix =
      "dispatch=1 scope_id=" + std::to_string(scope_id) +
      " phase=" + vision_stack_phase_name(phase) +
      " block=" + std::to_string(block) + " ";
  for (const auto& item : stack_dispatch_dependency_dispatch_rows()) {
    if (item.first.find(prefix) == 0u) {
      return &item.second;
    }
  }
  return nullptr;
}

const char* stack_dispatch_op_label(const VulkanVisionStackPhase phase) {
  switch (phase) {
    case VulkanVisionStackPhase::Norm1:
      return "vision_block.norm1";
    case VulkanVisionStackPhase::QkvLinear:
      return "vision_block.qkv_linear";
    case VulkanVisionStackPhase::QkvTransform:
      return "vision_block.qkv_transform";
    case VulkanVisionStackPhase::Attention:
      return "vision_block.attention";
    case VulkanVisionStackPhase::ProjLinear:
      return "vision_block.proj_linear";
    case VulkanVisionStackPhase::Residual1:
      return "vision_block.residual1";
    case VulkanVisionStackPhase::Norm2:
      return "vision_block.norm2";
    case VulkanVisionStackPhase::Fc1Gelu:
      return "vision_block.fc1_gelu";
    case VulkanVisionStackPhase::Fc2:
      return "vision_block.fc2";
    case VulkanVisionStackPhase::Residual2:
      return "vision_block.residual2";
    case VulkanVisionStackPhase::IntermediateCapture:
      return "vision_stack.intermediate_capture";
    default:
      return "unknown";
  }
}

const char* stack_dispatch_dependency_kind(
    const VulkanVisionStackPhase consumer_phase) {
  if (consumer_phase == VulkanVisionStackPhase::IntermediateCapture) {
    return "write_to_capture_read";
  }
  return "compute_shader_write_to_compute_shader_read";
}

std::string stack_dispatch_dependency_reject_reason(
    const bool residual2_candidate,
    const bool has_allocation_generation,
    const bool has_byte_range,
    const bool has_formal_last_use_proof,
    const bool producer_dispatch_observed,
    const bool consumer_dispatch_observed,
    const VulkanVisionStackPhase consumer_phase) {
  if (!residual2_candidate) {
    return "not_residual2_buffer_edge";
  }
  if (!has_allocation_generation) {
    return "missing_allocation_generation";
  }
  if (!has_byte_range) {
    return "missing_byte_range";
  }
  if (!has_formal_last_use_proof) {
    return "missing_formal_last_use_proof";
  }
  if (!producer_dispatch_observed) {
    return "missing_producer_dispatch";
  }
  if (!consumer_dispatch_observed) {
    return consumer_phase == VulkanVisionStackPhase::IntermediateCapture
        ? "capture_consumer_has_no_dispatch"
        : "missing_consumer_dispatch";
  }
  return "none";
}

std::string format_sizes(const std::vector<int64_t>& values) {
  std::ostringstream stream;
  stream << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << ',';
    }
    stream << values[i];
  }
  stream << ']';
  return stream.str();
}

const char* stack_region_dependency_graph_path() {
  const char* env = std::getenv("PYTORCH_VULKAN_STACK_DEP_GRAPH");
  return (env && *env) ? env : nullptr;
}

std::mutex& stack_region_dependency_graph_dump_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::string json_escape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char c : value) {
    switch (c) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += c;
        break;
    }
  }
  return escaped;
}

void append_json_comma(std::ostream& out, bool& first) {
  if (!first) {
    out << ',';
  }
  first = false;
}

void append_json_string(
    std::ostream& out,
    const char* key,
    const std::string& value,
    bool& first) {
  append_json_comma(out, first);
  out << '"' << key << "\":\"" << json_escape(value) << '"';
}

void append_json_u64(
    std::ostream& out,
    const char* key,
    const uint64_t value,
    bool& first) {
  append_json_comma(out, first);
  out << '"' << key << "\":" << value;
}

void append_json_bool(
    std::ostream& out,
    const char* key,
    const bool value,
    bool& first) {
  append_json_comma(out, first);
  out << '"' << key << "\":" << (value ? "true" : "false");
}

std::map<std::string, std::string> parse_space_separated_fields(
    const std::string& row) {
  std::map<std::string, std::string> fields;
  std::istringstream stream(row);
  std::string token;
  while (stream >> token) {
    const size_t equals = token.find('=');
    if (equals == std::string::npos || equals == 0u) {
      continue;
    }
    fields[token.substr(0, equals)] = token.substr(equals + 1u);
  }
  return fields;
}

uint64_t parsed_u64(
    const std::map<std::string, std::string>& fields,
    const char* key) {
  const auto it = fields.find(key);
  if (it == fields.end()) {
    return 0u;
  }
  try {
    return static_cast<uint64_t>(std::stoull(it->second));
  } catch (...) {
    return 0u;
  }
}

void append_json_fields_object(
    std::ostream& out,
    const std::map<std::string, std::string>& fields) {
  bool first = true;
  out << '{';
  for (const auto& field : fields) {
    append_json_string(out, field.first.c_str(), field.second, first);
  }
  out << '}';
}

void append_json_string_array(
    std::ostream& out,
    const char* key,
    const std::vector<std::string>& values,
    bool& first) {
  append_json_comma(out, first);
  out << '"' << key << "\":[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out << ',';
    }
    out << '"' << json_escape(values[i]) << '"';
  }
  out << ']';
}

std::vector<std::string> missing_dependency_metadata_fields(
    const std::map<std::string, std::string>& fields) {
  std::vector<std::string> missing;
  const auto has_true = [&fields](const char* key) {
    const auto it = fields.find(key);
    return it != fields.end() && it->second == "1";
  };
  if (!has_true("allocation_has_generation")) {
    missing.emplace_back("allocation_generation");
  }
  if (!has_true("allocation_has_byte_range")) {
    missing.emplace_back("byte_range");
  }
  if (!has_true("formal_last_use_proof")) {
    missing.emplace_back("formal_last_use_proof");
  }
  if (!has_true("producer_dispatch_observed")) {
    missing.emplace_back("producer_dispatch");
  }
  if (!has_true("consumer_dispatch_observed")) {
    missing.emplace_back("consumer_dispatch");
  }
  if (!has_true("descriptor_binding_known")) {
    missing.emplace_back("descriptor_binding");
  }
  return missing;
}

std::string field_or(
    const std::map<std::string, std::string>& fields,
    const char* key,
    const char* fallback) {
  const auto it = fields.find(key);
  return it == fields.end() ? std::string(fallback) : it->second;
}

bool boundary_has_planned_non_capture_norm1_consumer(
    const std::map<std::string, std::string>& fields) {
  return field_or(fields, "consumer_dispatch_planned", "0") == "1" &&
      field_or(fields, "consumer_dispatch_proof", "missing") ==
      "planned_non_capture_residual2_to_norm1";
}

std::vector<std::string> boundary_complete_dependency_missing_fields(
    const std::map<std::string, std::string>& fields) {
  std::vector<std::string> missing = missing_dependency_metadata_fields(fields);
  if (boundary_has_planned_non_capture_norm1_consumer(fields)) {
    missing.erase(
        std::remove(
            missing.begin(), missing.end(), std::string("consumer_dispatch")),
        missing.end());
  }
  return missing;
}

std::string stage_for_access(const std::string& access) {
  if (access.find("shader") != std::string::npos) {
    return "compute_shader";
  }
  return "unknown_stage";
}

std::string dependency_node_id(
    const std::map<std::string, std::string>& fields,
    const char* dispatch_position_key,
    const char* op_key,
    const char* phase_key,
    const char* block_key) {
  std::ostringstream stream;
  stream << "scope=" << field_or(fields, "scope_id", "unknown") << ":cmd="
         << field_or(fields, "command_buffer_sequence", "unknown")
         << ":pos=" << field_or(fields, dispatch_position_key, "unknown")
         << ":op=" << field_or(fields, op_key, "unknown")
         << ":phase=" << field_or(fields, phase_key, "unknown")
         << ":block=" << field_or(fields, block_key, "unknown");
  return stream.str();
}

std::string barrier_plan_rejection_reason(
    const std::map<std::string, std::string>& fields) {
  const std::vector<std::string> missing =
      missing_dependency_metadata_fields(fields);
  if (!missing.empty()) {
    return "missing_" + missing.front();
  }
  if (field_or(fields, "queue_submit", "0") != "1") {
    return "not_a_phase_boundary_submit_edge";
  }
  if (field_or(fields, "dependency_kind", "unknown") == "unknown") {
    return "missing_dependency_kind";
  }
  return "none";
}

bool barrier_plan_record_is_plannable(
    const std::map<std::string, std::string>& fields) {
  return barrier_plan_rejection_reason(fields) == "none";
}

std::string boundary_key_for_dependency(
    const std::map<std::string, std::string>& fields) {
  std::ostringstream stream;
  stream << field_or(fields, "callsite", "unknown") << ":scope="
         << field_or(fields, "scope_id", "unknown") << ":"
         << field_or(fields, "producer_phase", "unknown") << "@"
         << field_or(fields, "producer_block", "unknown") << "->"
         << field_or(fields, "consumer_phase", "unknown") << "@"
         << field_or(fields, "consumer_block", "unknown");
  return stream.str();
}

void append_graph_row_object(
    std::ostream& out,
    const std::string& row,
    const char* kind);

void append_barrier_plan_record(
    std::ostream& out,
    const std::string& row,
    const size_t index) {
  const auto fields = parse_space_separated_fields(row);
  const bool plannable = barrier_plan_record_is_plannable(fields);
  const std::string producer_access = field_or(fields, "producer_access", "unknown");
  const std::string consumer_access = field_or(fields, "consumer_access", "unknown");
  bool first = true;
  out << '{';
  append_json_string(
      out,
      "plan_record_id",
      "barrier_plan_edge_" + std::to_string(index),
      first);
  append_json_string(
      out,
      "producer_dispatch_node_id",
      dependency_node_id(
          fields,
          "producer_dispatch_first_position",
          "producer_op",
          "producer_phase",
          "producer_block"),
      first);
  append_json_string(
      out,
      "consumer_dispatch_node_id",
      dependency_node_id(
          fields,
          "consumer_dispatch_first_position",
          "consumer_op",
          "consumer_phase",
          "consumer_block"),
      first);
  append_json_string(
      out, "producer_dispatch_position",
      field_or(fields, "producer_dispatch_first_position", "unknown"), first);
  append_json_string(
      out, "consumer_dispatch_position",
      field_or(fields, "consumer_dispatch_first_position", "unknown"), first);
  append_json_string(
      out, "command_buffer_sequence",
      field_or(fields, "command_buffer_sequence", "unknown"), first);
  append_json_string(
      out, "allocation_id", field_or(fields, "allocation_id", "unknown"), first);
  append_json_string(
      out,
      "allocation_generation",
      field_or(fields, "allocation_generation", "unknown"),
      first);
  append_json_string(
      out, "byte_offset", field_or(fields, "byte_offset", "unknown"), first);
  append_json_string(
      out, "byte_range", field_or(fields, "byte_range", "unknown"), first);
  append_json_string(
      out,
      "dependency_kind",
      field_or(fields, "dependency_kind", "unknown"),
      first);
  append_json_string(out, "src_stage", stage_for_access(producer_access), first);
  append_json_string(out, "src_access", producer_access, first);
  append_json_string(out, "dst_stage", stage_for_access(consumer_access), first);
  append_json_string(out, "dst_access", consumer_access, first);
  append_json_string(
      out,
      "descriptor_binding",
      field_or(fields, "consumer_descriptor_binding", "unknown"),
      first);
  append_json_string(
      out,
      "planned_barrier_location",
      plannable ? "before_consumer_dispatch" : "not_plannable",
      first);
  append_json_bool(out, "plannable", plannable, first);
  append_json_bool(
      out,
      "could_theoretically_replace_phase_boundary_submit",
      plannable && field_or(fields, "queue_submit", "0") == "1",
      first);
  append_json_bool(out, "actual_barrier_inserted", false, first);
  append_json_bool(out, "actual_submit_removed", false, first);
  append_json_string(
      out,
      "boundary_key",
      boundary_key_for_dependency(fields),
      first);
  append_json_string(
      out,
      "rejection_reason",
      plannable ? "none" : barrier_plan_rejection_reason(fields),
      first);
  append_json_string_array(
      out,
      "missing_metadata_fields",
      missing_dependency_metadata_fields(fields),
      first);
  append_json_comma(out, first);
  out << "\"source_edge_fields\":";
  append_json_fields_object(out, fields);
  out << '}';
}

void append_barrier_plan_json(
    std::ostream& out,
    const std::vector<std::string>& dependency_edges,
    bool& first) {
  uint64_t candidate_records = 0u;
  uint64_t plannable_records = 0u;
  uint64_t rejected_records = 0u;
  uint64_t phase_boundary_replace_candidate_records = 0u;
  std::map<std::string, uint64_t> rejection_reasons;
  for (const auto& row : dependency_edges) {
    const auto fields = parse_space_separated_fields(row);
    const uint64_t count = parsed_u64(fields, "count");
    const bool plannable = barrier_plan_record_is_plannable(fields);
    const std::string rejection =
        plannable ? "none" : barrier_plan_rejection_reason(fields);
    candidate_records += count;
    if (plannable) {
      plannable_records += count;
      if (field_or(fields, "queue_submit", "0") == "1") {
        phase_boundary_replace_candidate_records += count;
      }
    } else {
      rejected_records += count;
      rejection_reasons[rejection] += count;
    }
  }

  append_json_comma(out, first);
  out << "\"barrier_plan\":{";
  bool plan_first = true;
  append_json_string(out, "schema", "StackRegionBarrierPlan.v0", plan_first);
  append_json_bool(out, "behavior_neutral", true, plan_first);
  append_json_bool(out, "dry_run_only", true, plan_first);
  append_json_string(out, "source_graph_schema", "StackRegionDependencyGraph.v0", plan_first);
  append_json_string(
      out,
      "planning_stage",
      "dispatch_dependency_edge_plan",
      plan_first);
  append_json_u64(out, "candidate_records", candidate_records, plan_first);
  append_json_u64(out, "plannable_records", plannable_records, plan_first);
  append_json_u64(out, "rejected_records", rejected_records, plan_first);
  append_json_u64(
      out,
      "phase_boundary_replace_candidate_records",
      phase_boundary_replace_candidate_records,
      plan_first);
  append_json_u64(out, "barriers_inserted", 0u, plan_first);
  append_json_u64(out, "submits_removed", 0u, plan_first);
  append_json_string(
      out,
      "why_submits_remain_required",
      "barrier plan is per-edge dry-run only; boundary-level required-edge sets are not complete",
      plan_first);
  append_json_comma(out, plan_first);
  out << "\"rejection_reasons\":{";
  bool reject_first = true;
  for (const auto& item : rejection_reasons) {
    append_json_u64(out, item.first.c_str(), item.second, reject_first);
  }
  out << "}";
  append_json_string_array(
      out,
      "missing_boundary_plan_fields",
      {"complete_boundary_required_edge_set",
       "boundary_to_barrier_record_coverage",
       "retire_only_resource_classification",
       "public_host_final_output_blocker_resolution"},
      plan_first);
  append_json_comma(out, plan_first);
  out << "\"records\":[";
  for (size_t i = 0; i < dependency_edges.size(); ++i) {
    if (i > 0) {
      out << ',';
    }
    append_barrier_plan_record(out, dependency_edges[i], i);
  }
  out << "]}";
}

struct BoundaryResourceClassSummary final {
  uint64_t count = 0u;
  uint64_t bytes = 0u;
};

struct BoundaryCompleteProof final {
  std::string boundary_id;
  std::string boundary_phase = "block_entry";
  std::string producer_block;
  std::string consumer_block;
  uint64_t required_edge_records = 0u;
  uint64_t covered_edge_records = 0u;
  uint64_t rejected_edge_records = 0u;
  uint64_t queue_submit_records = 0u;
  uint64_t required_edge_bytes = 0u;
  uint64_t consumer_dispatch_planned_records = 0u;
  uint64_t consumer_dispatch_missing_reduced_records = 0u;
  std::map<std::string, uint64_t> edge_rejection_reasons;
  std::map<std::string, uint64_t> missing_fields;
  std::map<std::string, uint64_t> consumer_dispatch_proofs;
  std::map<std::string, BoundaryResourceClassSummary> retire_only_resources;
  std::map<std::string, BoundaryResourceClassSummary> ordering_required_resources;
  std::map<std::string, BoundaryResourceClassSummary> public_blockers;
  std::map<std::string, uint64_t> boundary_reject_reasons;
  std::vector<std::string> boundary_rows;
};

int64_t parsed_i64(
    const std::map<std::string, std::string>& fields,
    const char* key,
    const int64_t fallback = -1) {
  const auto it = fields.find(key);
  if (it == fields.end()) {
    return fallback;
  }
  try {
    return std::stoll(it->second);
  } catch (...) {
    return fallback;
  }
}

bool signature_resource_class_is_retire_only(const std::string& resource_class) {
  return resource_class == "attention_score_probability_subresource" ||
      resource_class == "layernorm_internal_stat_buffer" ||
      resource_class == "metadata_uniform" ||
      resource_class == "proven_stack_activation";
}

bool signature_resource_class_is_public_blocker(
    const std::string& resource_class) {
  return resource_class.find("host_visible") != std::string::npos ||
      resource_class.find("requested") != std::string::npos ||
      resource_class.find("final_output") != std::string::npos ||
      resource_class.find("public") != std::string::npos;
}

void add_boundary_resource_class(
    std::map<std::string, BoundaryResourceClassSummary>& target,
    const std::string& resource_class,
    const uint64_t count,
    const uint64_t bytes) {
  auto& entry = target[resource_class];
  entry.count += count;
  entry.bytes += bytes;
}

void collect_boundary_signature_resources(
    const std::map<std::string, std::string>& fields,
    BoundaryCompleteProof& proof) {
  const uint64_t boundary_count = std::max<uint64_t>(parsed_u64(fields, "count"), 1u);
  const auto signature = fields.find("signature");
  if (signature == fields.end()) {
    proof.boundary_reject_reasons["missing_boundary_signature"] +=
        parsed_u64(fields, "count");
    return;
  }

  std::istringstream stream(signature->second);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      continue;
    }
    const size_t first_hash = token.find('#');
    const size_t second_hash =
        first_hash == std::string::npos
        ? std::string::npos
        : token.find('#', first_hash + 1u);
    if (first_hash == std::string::npos || second_hash == std::string::npos) {
      proof.boundary_reject_reasons["malformed_boundary_signature"] +=
          parsed_u64(fields, "count");
      continue;
    }
    const std::string resource_class = token.substr(0, first_hash);
    uint64_t resource_count = 0u;
    uint64_t resource_bytes = 0u;
    try {
      resource_count = static_cast<uint64_t>(
          std::stoull(token.substr(first_hash + 1u, second_hash - first_hash - 1u)));
      resource_bytes =
          static_cast<uint64_t>(std::stoull(token.substr(second_hash + 1u)));
    } catch (...) {
      proof.boundary_reject_reasons["malformed_boundary_signature"] +=
          parsed_u64(fields, "count");
      continue;
    }
    resource_count *= boundary_count;
    resource_bytes *= boundary_count;
    if (signature_resource_class_is_public_blocker(resource_class)) {
      add_boundary_resource_class(
          proof.public_blockers, resource_class, resource_count, resource_bytes);
      add_boundary_resource_class(
          proof.ordering_required_resources,
          resource_class,
          resource_count,
          resource_bytes);
    } else if (signature_resource_class_is_retire_only(resource_class)) {
      add_boundary_resource_class(
          proof.retire_only_resources,
          resource_class,
          resource_count,
          resource_bytes);
    } else {
      add_boundary_resource_class(
          proof.ordering_required_resources,
          resource_class,
          resource_count,
          resource_bytes);
    }
  }
}

std::map<std::string, bool> capture_source_blocks_for_dependencies(
    const std::vector<std::string>& dependency_edges) {
  std::map<std::string, bool> capture_source_blocks;
  for (const auto& row : dependency_edges) {
    const auto fields = parse_space_separated_fields(row);
    if (field_or(fields, "consumer_phase", "unknown") ==
        "intermediate_capture") {
      capture_source_blocks[field_or(fields, "producer_block", "unknown")] =
          true;
    }
  }
  return capture_source_blocks;
}

bool is_non_capture_residual2_to_norm1_boundary_candidate(
    const std::map<std::string, std::string>& fields,
    const std::map<std::string, bool>& capture_source_blocks) {
  if (field_or(fields, "producer_phase", "unknown") != "residual2" ||
      field_or(fields, "consumer_phase", "unknown") != "norm1" ||
      field_or(fields, "role", "unknown") != "stack_residual2_output") {
    return false;
  }
  const std::string producer_block = field_or(fields, "producer_block", "unknown");
  if (capture_source_blocks.find(producer_block) != capture_source_blocks.end()) {
    return false;
  }
  const int64_t producer = parsed_i64(fields, "producer_block");
  const int64_t consumer = parsed_i64(fields, "consumer_block");
  return producer >= 0 && consumer == producer + 1;
}

std::string boundary_complete_proof_key(
    const std::map<std::string, std::string>& fields) {
  std::ostringstream stream;
  stream << "residual2_to_norm1:producer_block="
         << field_or(fields, "producer_block", "unknown")
         << ":consumer_block=" << field_or(fields, "consumer_block", "unknown");
  return stream.str();
}

void append_resource_class_summary_object(
    std::ostream& out,
    const std::map<std::string, BoundaryResourceClassSummary>& classes) {
  bool first = true;
  out << '{';
  for (const auto& item : classes) {
    append_json_comma(out, first);
    out << '"' << json_escape(item.first) << "\":{";
    bool class_first = true;
    append_json_u64(out, "count", item.second.count, class_first);
    append_json_u64(out, "bytes", item.second.bytes, class_first);
    out << '}';
  }
  out << '}';
}

void append_u64_map_object(
    std::ostream& out,
    const std::map<std::string, uint64_t>& values) {
  bool first = true;
  out << '{';
  for (const auto& item : values) {
    append_json_u64(out, item.first.c_str(), item.second, first);
  }
  out << '}';
}

bool boundary_complete_proof_is_complete(const BoundaryCompleteProof& proof) {
  return proof.required_edge_records > 0u &&
      proof.required_edge_records == proof.covered_edge_records &&
      proof.rejected_edge_records == 0u && !proof.boundary_rows.empty() &&
      proof.public_blockers.empty() && proof.boundary_reject_reasons.empty() &&
      proof.ordering_required_resources.empty();
}

void append_boundary_complete_proof_record(
    std::ostream& out,
    const BoundaryCompleteProof& proof) {
  const bool complete = boundary_complete_proof_is_complete(proof);
  bool first = true;
  out << '{';
  append_json_string(out, "boundary_id", proof.boundary_id, first);
  append_json_string(out, "boundary_phase", proof.boundary_phase, first);
  append_json_string(out, "producer_block", proof.producer_block, first);
  append_json_string(out, "consumer_block", proof.consumer_block, first);
  append_json_u64(out, "required_edge_records", proof.required_edge_records, first);
  append_json_u64(out, "barrier_plan_covered_edge_records", proof.covered_edge_records, first);
  append_json_u64(out, "rejected_edge_records", proof.rejected_edge_records, first);
  append_json_u64(out, "queue_submit_records", proof.queue_submit_records, first);
  append_json_u64(out, "required_edge_bytes", proof.required_edge_bytes, first);
  append_json_u64(
      out,
      "consumer_dispatch_planned_records",
      proof.consumer_dispatch_planned_records,
      first);
  append_json_u64(
      out,
      "consumer_dispatch_missing_reduced_records",
      proof.consumer_dispatch_missing_reduced_records,
      first);
  append_json_bool(out, "complete", complete, first);
  append_json_bool(out, "behavior_change_allowed", false, first);
  append_json_comma(out, first);
  out << "\"consumer_dispatch_proofs\":";
  append_u64_map_object(out, proof.consumer_dispatch_proofs);
  append_json_comma(out, first);
  out << "\"edge_rejection_reasons\":";
  append_u64_map_object(out, proof.edge_rejection_reasons);
  append_json_comma(out, first);
  out << "\"missing_fields\":";
  append_u64_map_object(out, proof.missing_fields);
  append_json_comma(out, first);
  out << "\"retire_only_resources\":";
  append_resource_class_summary_object(out, proof.retire_only_resources);
  append_json_comma(out, first);
  out << "\"ordering_required_resources\":";
  append_resource_class_summary_object(out, proof.ordering_required_resources);
  append_json_comma(out, first);
  out << "\"public_host_final_requested_blockers\":";
  append_resource_class_summary_object(out, proof.public_blockers);
  append_json_comma(out, first);
  out << "\"boundary_reject_reasons\":";
  append_u64_map_object(out, proof.boundary_reject_reasons);
  append_json_comma(out, first);
  out << "\"phase_boundary_rows\":[";
  for (size_t i = 0; i < proof.boundary_rows.size(); ++i) {
    if (i > 0) {
      out << ',';
    }
    append_graph_row_object(out, proof.boundary_rows[i], "phase_boundary");
  }
  out << "]}";
}

void append_boundary_complete_dependency_proof_json(
    std::ostream& out,
    const std::vector<std::string>& dependency_edges,
    const std::vector<std::string>& boundary_nodes,
    bool& first) {
  const std::map<std::string, bool> capture_source_blocks =
      capture_source_blocks_for_dependencies(dependency_edges);
  std::map<std::string, BoundaryCompleteProof> proofs;
  for (const auto& row : dependency_edges) {
    const auto fields = parse_space_separated_fields(row);
    if (!is_non_capture_residual2_to_norm1_boundary_candidate(
            fields, capture_source_blocks)) {
      continue;
    }
    const std::string key = boundary_complete_proof_key(fields);
    auto& proof = proofs[key];
    proof.boundary_id = key;
    proof.producer_block = field_or(fields, "producer_block", "unknown");
    proof.consumer_block = field_or(fields, "consumer_block", "unknown");
    const uint64_t count = parsed_u64(fields, "count");
    proof.required_edge_records += count;
    proof.queue_submit_records += parsed_u64(fields, "queue_submit");
    proof.required_edge_bytes += parsed_u64(fields, "bytes");
    if (boundary_has_planned_non_capture_norm1_consumer(fields)) {
      proof.consumer_dispatch_planned_records += count;
      proof.consumer_dispatch_proofs
          [field_or(fields, "consumer_dispatch_proof", "missing")] += count;
    } else {
      proof.consumer_dispatch_proofs
          [field_or(fields, "consumer_dispatch_proof", "missing")] += count;
    }
    const bool plannable = barrier_plan_record_is_plannable(fields);
    if (plannable) {
      proof.covered_edge_records += count;
    } else {
      proof.rejected_edge_records += count;
      proof.edge_rejection_reasons[barrier_plan_rejection_reason(fields)] +=
          count;
      const auto strict_missing = missing_dependency_metadata_fields(fields);
      const auto boundary_missing =
          boundary_complete_dependency_missing_fields(fields);
      if (strict_missing.size() > boundary_missing.size()) {
        proof.consumer_dispatch_missing_reduced_records += count;
      }
      for (const auto& missing : boundary_missing) {
        proof.missing_fields[missing] += count;
      }
    }
  }

  for (const auto& row : boundary_nodes) {
    const auto fields = parse_space_separated_fields(row);
    if (field_or(fields, "boundary_stack_phase", "unknown") != "block_entry") {
      continue;
    }
    const std::string consumer_block = field_or(fields, "boundary_block", "unknown");
    for (auto& item : proofs) {
      auto& proof = item.second;
      if (proof.consumer_block != consumer_block) {
        continue;
      }
      proof.boundary_rows.emplace_back(row);
      collect_boundary_signature_resources(fields, proof);
      const std::string budget_reject =
          field_or(fields, "budget_reject", "missing_budget_reject");
      if (budget_reject != "none") {
        proof.boundary_reject_reasons["budget_reject:" + budget_reject] +=
            parsed_u64(fields, "count");
      }
      const std::string blockers = field_or(fields, "blockers", "none");
      if (blockers != "none") {
        proof.boundary_reject_reasons["blockers:" + blockers] +=
            parsed_u64(fields, "count");
      }
    }
  }

  uint64_t candidate_boundaries = 0u;
  uint64_t complete_boundaries = 0u;
  uint64_t required_edge_records = 0u;
  uint64_t covered_edge_records = 0u;
  uint64_t consumer_dispatch_planned_records = 0u;
  uint64_t consumer_dispatch_missing_reduced_records = 0u;
  std::map<std::string, uint64_t> blocker_reasons;
  for (const auto& item : proofs) {
    const auto& proof = item.second;
    ++candidate_boundaries;
    required_edge_records += proof.required_edge_records;
    covered_edge_records += proof.covered_edge_records;
    consumer_dispatch_planned_records += proof.consumer_dispatch_planned_records;
    consumer_dispatch_missing_reduced_records +=
        proof.consumer_dispatch_missing_reduced_records;
    if (boundary_complete_proof_is_complete(proof)) {
      ++complete_boundaries;
    } else {
      if (proof.boundary_rows.empty()) {
        blocker_reasons["missing_phase_boundary_group"] += 1u;
      }
      for (const auto& reason : proof.edge_rejection_reasons) {
        blocker_reasons["edge:" + reason.first] += reason.second;
      }
      for (const auto& reason : proof.boundary_reject_reasons) {
        blocker_reasons["boundary:" + reason.first] += reason.second;
      }
      for (const auto& resource : proof.ordering_required_resources) {
        blocker_reasons["ordering_required_resource:" + resource.first] +=
            resource.second.count;
      }
      for (const auto& resource : proof.public_blockers) {
        blocker_reasons["public_host_final_requested:" + resource.first] +=
            resource.second.count;
      }
    }
  }

  append_json_comma(out, first);
  out << "\"boundary_complete_dependency_proof\":{";
  bool proof_first = true;
  append_json_string(
      out, "schema", "BoundaryCompleteDependencyProof.v0", proof_first);
  append_json_bool(out, "behavior_neutral", true, proof_first);
  append_json_bool(out, "dry_run_only", true, proof_first);
  append_json_string(
      out,
      "target_boundary_class",
      "non_capture_residual2_to_norm1",
      proof_first);
  append_json_u64(out, "candidate_boundaries", candidate_boundaries, proof_first);
  append_json_u64(out, "complete_boundaries", complete_boundaries, proof_first);
  append_json_u64(
      out, "required_edge_records", required_edge_records, proof_first);
  append_json_u64(
      out, "barrier_plan_covered_edge_records", covered_edge_records, proof_first);
  append_json_u64(
      out,
      "consumer_dispatch_planned_records",
      consumer_dispatch_planned_records,
      proof_first);
  append_json_u64(
      out,
      "consumer_dispatch_missing_reduced_records",
      consumer_dispatch_missing_reduced_records,
      proof_first);
  append_json_u64(out, "barriers_inserted", 0u, proof_first);
  append_json_u64(out, "submits_removed", 0u, proof_first);
  append_json_comma(out, proof_first);
  out << "\"blocker_reasons\":";
  append_u64_map_object(out, blocker_reasons);
  append_json_comma(out, proof_first);
  out << "\"records\":[";
  bool first_record = true;
  for (const auto& item : proofs) {
    if (!first_record) {
      out << ',';
    }
    first_record = false;
    append_boundary_complete_proof_record(out, item.second);
  }
  out << "]}";
}

void append_graph_row_object(
    std::ostream& out,
    const std::string& row,
    const char* kind) {
  const auto fields = parse_space_separated_fields(row);
  bool first = true;
  out << '{';
  append_json_string(out, "kind", kind, first);
  append_json_string(out, "raw", row, first);
  append_json_comma(out, first);
  out << "\"fields\":";
  append_json_fields_object(out, fields);
  if (std::string(kind) == "dependency_edge") {
    append_json_string_array(
        out, "missing_metadata_fields", missing_dependency_metadata_fields(fields), first);
  }
  out << '}';
}

void append_graph_array(
    std::ostream& out,
    const char* key,
    const std::vector<std::string>& rows,
    const char* kind,
    bool& first) {
  append_json_comma(out, first);
  out << '"' << key << "\":[";
  for (size_t i = 0; i < rows.size(); ++i) {
    if (i > 0) {
      out << ',';
    }
    append_graph_row_object(out, rows[i], kind);
  }
  out << ']';
}

void split_stack_graph_rows(
    const std::vector<std::string>& rows,
    std::vector<std::string>& dispatch_nodes,
    std::vector<std::string>& dependency_edges,
    std::vector<std::string>& capture_edges) {
  for (const auto& row : rows) {
    if (row.find("dispatch=1") != std::string::npos) {
      dispatch_nodes.emplace_back(row);
      continue;
    }
    if (row.find("stack_dispatch_dependency=1") != std::string::npos) {
      dependency_edges.emplace_back(row);
      if (row.find("consumer_phase=intermediate_capture") != std::string::npos) {
        capture_edges.emplace_back(row);
      }
    }
  }
}

void split_lifetime_graph_rows(
    const std::vector<std::string>& rows,
    std::vector<std::string>& resource_nodes,
    std::vector<std::string>& boundary_nodes) {
  for (const auto& row : rows) {
    if (row.find("resource=1") != std::string::npos) {
      resource_nodes.emplace_back(row);
    } else if (row.find("phase_boundary_group=1") != std::string::npos) {
      boundary_nodes.emplace_back(row);
    }
  }
}

void write_stack_region_dependency_graph_json(std::ostream& out) {
  const std::vector<std::string> dispatch_dependency_rows =
      stack_dispatch_dependency_dry_run_snapshot();
  const std::vector<std::string> allocation_rows =
      stack_allocation_aggregate_snapshot();
  const std::vector<std::string> lifetime_rows =
      stack_subresource_lifetime_dry_run_snapshot();
  const std::vector<std::string> region_rows =
      region_lifetime_submit_attribution_snapshot();

  std::vector<std::string> dispatch_nodes;
  std::vector<std::string> dependency_edges;
  std::vector<std::string> capture_edges;
  split_stack_graph_rows(
      dispatch_dependency_rows, dispatch_nodes, dependency_edges, capture_edges);

  std::vector<std::string> resource_nodes;
  std::vector<std::string> boundary_nodes;
  split_lifetime_graph_rows(lifetime_rows, resource_nodes, boundary_nodes);

  uint64_t fully_proven_edge_records = 0u;
  uint64_t total_dependency_records = 0u;
  uint64_t queue_submit_dependency_records = 0u;
  std::map<std::string, uint64_t> reject_reasons;
  for (const auto& row : dependency_edges) {
    const auto fields = parse_space_separated_fields(row);
    const uint64_t count = parsed_u64(fields, "count");
    total_dependency_records += count;
    queue_submit_dependency_records += parsed_u64(fields, "queue_submit");
    fully_proven_edge_records += parsed_u64(fields, "fully_proven_count");
    const auto it = fields.find("reject_reason");
    reject_reasons[it == fields.end() ? "missing_reject_reason" : it->second] +=
        count;
  }

  uint64_t complete_boundaries = 0u;
  uint64_t queue_submit_boundaries = 0u;
  std::map<std::string, uint64_t> boundary_reject_reasons;
  for (const auto& row : boundary_nodes) {
    const auto fields = parse_space_separated_fields(row);
    const uint64_t count = parsed_u64(fields, "count");
    queue_submit_boundaries += parsed_u64(fields, "queue_submit");
    const auto eligible = fields.find("all_safe_group_eligible");
    if (eligible != fields.end() && eligible->second == "1") {
      complete_boundaries += count;
    }
    const auto reject = fields.find("budget_reject");
    boundary_reject_reasons
        [reject == fields.end() ? "missing_budget_reject" : reject->second] +=
        count;
  }

  bool first = true;
  out << "{";
  append_json_string(out, "schema", "StackRegionDependencyGraph.v0", first);
  append_json_bool(out, "behavior_neutral", true, first);
  append_json_string(
      out, "env_var", "PYTORCH_VULKAN_STACK_DEP_GRAPH", first);
  append_json_comma(out, first);
  out << "\"region\":{";
  bool region_first = true;
  append_json_string(out, "region_id", "missing_region_id", region_first);
  append_json_string(
      out, "stack_context_id", "missing_stack_context_id", region_first);
  append_json_string(
      out, "bridge_session_id", "missing_bridge_session_id", region_first);
  append_json_string_array(
      out,
      "missing_fields",
      {"region_id", "stack_context_id", "bridge_session_id"},
      region_first);
  out << "}";

  append_json_comma(out, first);
  out << "\"summary\":{";
  bool summary_first = true;
  append_json_u64(out, "dispatch_nodes", dispatch_nodes.size(), summary_first);
  append_json_u64(
      out, "dependency_edge_rows", dependency_edges.size(), summary_first);
  append_json_u64(
      out, "dependency_edge_records", total_dependency_records, summary_first);
  append_json_u64(
      out,
      "queue_submit_dependency_records",
      queue_submit_dependency_records,
      summary_first);
  append_json_u64(
      out,
      "fully_proven_dependency_records",
      fully_proven_edge_records,
      summary_first);
  append_json_u64(out, "resource_nodes", resource_nodes.size(), summary_first);
  append_json_u64(out, "allocation_nodes", allocation_rows.size(), summary_first);
  append_json_u64(out, "boundary_nodes", boundary_nodes.size(), summary_first);
  append_json_u64(
      out, "complete_boundary_records", complete_boundaries, summary_first);
  append_json_u64(
      out, "queue_submit_boundary_records", queue_submit_boundaries, summary_first);
  append_json_u64(out, "capture_edges", capture_edges.size(), summary_first);
  append_json_bool(out, "submit_elision_enabled", false, summary_first);
  append_json_string(
      out,
      "current_submit_sync_reason",
      "phase_boundary_submit_required_until_complete_graph_proof",
      summary_first);
  out << "}";

  append_json_comma(out, first);
  out << "\"dependency_reject_reasons\":{";
  bool reject_first = true;
  for (const auto& item : reject_reasons) {
    append_json_u64(out, item.first.c_str(), item.second, reject_first);
  }
  out << "}";

  append_json_comma(out, first);
  out << "\"boundary_reject_reasons\":{";
  bool boundary_reject_first = true;
  for (const auto& item : boundary_reject_reasons) {
    append_json_u64(
        out, item.first.c_str(), item.second, boundary_reject_first);
  }
  out << "}";

  append_graph_array(out, "dispatch_nodes", dispatch_nodes, "dispatch", first);
  append_graph_array(
      out, "dependency_edges", dependency_edges, "dependency_edge", first);
  append_graph_array(out, "capture_edges", capture_edges, "capture_edge", first);
  append_graph_array(out, "resource_nodes", resource_nodes, "resource", first);
  append_graph_array(
      out, "allocation_nodes", allocation_rows, "allocation", first);
  append_graph_array(
      out, "phase_boundary_nodes", boundary_nodes, "phase_boundary", first);
  append_barrier_plan_json(out, dependency_edges, first);
  append_boundary_complete_dependency_proof_json(
      out, dependency_edges, boundary_nodes, first);
  append_graph_array(
      out, "region_lifetime_rows", region_rows, "region_lifetime", first);
  append_json_string_array(
      out,
      "unproven_or_missing_metadata_fields",
      {"region_id",
       "stack_context_id",
       "bridge_session_id",
       "complete_boundary_dependency_set",
       "consumer_dispatch_for_capture_edges_when_not_recorded",
       "boundary_specific_required_edge_set"},
      first);
  out << "}\n";
}

void maybe_write_stack_region_dependency_graph_dump() {
  const char* const path = stack_region_dependency_graph_path();
  if (path == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(stack_region_dependency_graph_dump_mutex());
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    return;
  }
  write_stack_region_dependency_graph_json(out);
}

} // namespace

VulkanSyncCounters& vulkan_sync_counters() {
  static VulkanSyncCounters counters;
  return counters;
}

VulkanSubmitOriginCounters& vulkan_submit_origin_counters() {
  static VulkanSubmitOriginCounters counters;
  return counters;
}

VulkanSubmitOriginPhaseCounters& vulkan_submit_origin_phase_counters() {
  static VulkanSubmitOriginPhaseCounters counters;
  return counters;
}

VulkanRetireDrainCounters& vulkan_retire_drain_counters() {
  static VulkanRetireDrainCounters counters;
  return counters;
}

VulkanStackInternalTempRetireBatchCounters&
stack_internal_temp_retire_batch_counters() {
  static VulkanStackInternalTempRetireBatchCounters counters;
  return counters;
}

VulkanStackRetireDrainBlockerCounters&
stack_retire_drain_blocker_counters() {
  static VulkanStackRetireDrainBlockerCounters counters;
  return counters;
}

VulkanStackSubresourceLifetimeDryRunCounters&
stack_subresource_lifetime_dry_run_counters() {
  static VulkanStackSubresourceLifetimeDryRunCounters counters;
  return counters;
}

std::array<VulkanRetireCallSiteCounter, 27>& retire_call_site_counters() {
  static std::array<VulkanRetireCallSiteCounter, 27> counters;
  return counters;
}

bool is_stack_temp_role(VulkanRetiredResourceRole role);

const char* stack_temp_retire_batch_reject_reason(
    const VulkanStackRetireProvenance& provenance) {
  if (provenance.requested_intermediate || provenance.escapes_stack) {
    return "requested_intermediate";
  }
  if (provenance.final_output) {
    return "final_output";
  }
  if (provenance.alias_or_view) {
    return "alias";
  }
  if (provenance.aliases_runtime_input || provenance.aliases_runtime_output) {
    return "runtime_alias";
  }
  switch (provenance.producer_role) {
    case VulkanRetiredResourceRole::StackFc1GeluOutput:
    case VulkanRetiredResourceRole::StackAttentionOutput:
      break;
    default:
      return "not_target_role";
  }
  if (!provenance.has_last_use_proof) {
    return "missing_proof";
  }
  if (!provenance.internal_non_escaping) {
    return "not_internal_non_escaping";
  }
  if (!provenance.final_consumer_before_stack_submit) {
    return "consumer_after_submit";
  }
  if (provenance.lifetime != VulkanStackTensorLifetimeClass::InternalTemp) {
    return "lifetime";
  }
  return "accepted";
}

std::mutex& retired_resource_aggregate_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<RetiredResourceAggregateKey, RetiredResourceAggregateValue>&
retired_resource_aggregate() {
  static std::map<RetiredResourceAggregateKey, RetiredResourceAggregateValue>
      aggregate;
  return aggregate;
}

std::mutex& stack_temp_lifetime_safety_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<StackTempLifetimeSafetyKey, StackTempLifetimeSafetyValue>&
stack_temp_lifetime_safety_aggregate() {
  static std::map<StackTempLifetimeSafetyKey, StackTempLifetimeSafetyValue>
      aggregate;
  return aggregate;
}

std::mutex& stack_scratch_arena_lifetime_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<StackScratchArenaLifetimeKey, StackScratchArenaLifetimeValue>&
stack_scratch_arena_lifetime_aggregate() {
  static std::map<StackScratchArenaLifetimeKey, StackScratchArenaLifetimeValue>
      aggregate;
  return aggregate;
}

std::mutex& stack_temp_retire_batch_snapshot_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<std::string, StackTempLifetimeSafetyValue>&
stack_temp_retire_batch_decisions() {
  static std::map<std::string, StackTempLifetimeSafetyValue> decisions;
  return decisions;
}

std::mutex& stack_retire_drain_blocker_snapshot_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<std::string, StackTempLifetimeSafetyValue>&
stack_retire_drain_blockers() {
  static std::map<std::string, StackTempLifetimeSafetyValue> blockers;
  return blockers;
}

std::mutex& region_lifetime_submit_attribution_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<std::string, StackTempLifetimeSafetyValue>&
region_lifetime_submit_attribution_rows() {
  static std::map<std::string, StackTempLifetimeSafetyValue> rows;
  return rows;
}

std::mutex& stack_subresource_lifetime_dry_run_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<std::string, StackTempLifetimeSafetyValue>&
stack_subresource_lifetime_dry_run_rows() {
  static std::map<std::string, StackTempLifetimeSafetyValue> rows;
  return rows;
}

void update_peak_atomic(std::atomic<uint64_t>& value, const uint64_t candidate) {
  uint64_t current = value.load(std::memory_order_relaxed);
  while (
      candidate > current &&
      !value.compare_exchange_weak(
          current, candidate, std::memory_order_relaxed)) {
  }
}

bool is_metadata_or_uniform_resource(
    const VulkanRetiredResourceKind kind,
    const VulkanRetiredResourceRole role) {
  switch (role) {
    case VulkanRetiredResourceRole::NativeLayerNormUniform:
    case VulkanRetiredResourceRole::NativeLayerNormMetadata:
    case VulkanRetiredResourceRole::AttentionMetadata:
    case VulkanRetiredResourceRole::LinearMetadata:
    case VulkanRetiredResourceRole::ConvMetadata:
    case VulkanRetiredResourceRole::ResidualAddMetadata:
      return true;
    default:
      break;
  }
  return kind == VulkanRetiredResourceKind::UniformBuffer ||
      kind == VulkanRetiredResourceKind::MetadataBuffer;
}

bool is_layernorm_stat_resource(
    const VulkanRetiredResourceRole role,
    const VulkanStackRetireProvenance& provenance) {
  if (
      role != VulkanRetiredResourceRole::StackNorm1Output &&
      role != VulkanRetiredResourceRole::StackNorm2Output) {
    return false;
  }
  return provenance.shape.size() == 2u && provenance.shape.back() == 1;
}

bool is_layernorm_internal_stat_buffer(
    const VulkanRetiredResourceRole role,
    const VulkanStackRetireProvenance& provenance) {
  if (!is_layernorm_stat_resource(role, provenance)) {
    return false;
  }
  if (
      provenance.producer_role != role ||
      provenance.source != VulkanStackRetireProvenanceSource::TensorAllocation) {
    return false;
  }
  if (
      role == VulkanRetiredResourceRole::StackNorm1Output &&
      provenance.phase != VulkanVisionStackPhase::Norm1) {
    return false;
  }
  if (
      role == VulkanRetiredResourceRole::StackNorm2Output &&
      provenance.phase != VulkanVisionStackPhase::Norm2) {
    return false;
  }
  return provenance.defined && provenance.block_index >= 0 &&
      !provenance.requested_intermediate && !provenance.escapes_stack &&
      !provenance.final_output && !provenance.alias_or_view &&
      !provenance.aliases_runtime_input &&
      !provenance.aliases_runtime_output && provenance.direct_buffer &&
      provenance.buffer_storage && !provenance.image_storage &&
      provenance.shape[0] > 0 && provenance.dtype >= 0;
}

bool is_attention_score_probability_subresource(
    const VulkanRetiredResourceRole role,
    const VulkanStackRetireProvenance& provenance) {
  if (
      role != VulkanRetiredResourceRole::StackAttentionOutput ||
      provenance.producer_role != VulkanRetiredResourceRole::StackAttentionOutput ||
      provenance.phase != VulkanVisionStackPhase::Attention ||
      provenance.source != VulkanStackRetireProvenanceSource::TensorAllocation) {
    return false;
  }
  if (
      !provenance.defined || provenance.requested_intermediate ||
      provenance.escapes_stack || provenance.final_output ||
      provenance.alias_or_view || provenance.aliases_runtime_input ||
      provenance.aliases_runtime_output || !provenance.direct_buffer ||
      !provenance.buffer_storage || provenance.image_storage) {
    return false;
  }
  return provenance.shape.size() == 3u && provenance.shape[1] > 0 &&
      provenance.shape[1] == provenance.shape[2] && provenance.dtype >= 0;
}

bool is_attention_subresource_role(const VulkanRetiredResourceRole role) {
  switch (role) {
    case VulkanRetiredResourceRole::StackQView:
    case VulkanRetiredResourceRole::StackKView:
    case VulkanRetiredResourceRole::StackVView:
    case VulkanRetiredResourceRole::StackAttentionOutput:
      return true;
    default:
      return false;
  }
}

const char* classify_unproven_attention_subresource(
    const VulkanRetiredResourceKind kind,
    const VulkanRetiredResourceRole role,
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof) {
  if (provenance.defined) {
    return kDryRunAttentionProvenanceMissingLastUse;
  }
  if (allocation_proof.has_generation && allocation_proof.has_byte_range) {
    const bool has_stack_scope_evidence =
        role == VulkanRetiredResourceRole::StackAttentionOutput &&
        inside_vision_stack_phase() && current_vision_stack_block_index() >= 0;
    if (
        kind == VulkanRetiredResourceKind::Buffer &&
        allocation_proof.byte_range > 4096u) {
      if (has_stack_scope_evidence) {
        return kDryRunAttentionScoreProbabilityRangeNonEscapeLastConsumer;
      }
      return kDryRunAttentionScoreProbabilityRangeMissingAliasEscapeProof;
    }
    if (kind == VulkanRetiredResourceKind::Unknown) {
      if (has_stack_scope_evidence) {
        return kDryRunAttentionRawAuxiliaryRangeNonEscapeLastConsumer;
      }
      return kDryRunAttentionRawAuxiliaryRangeMissingAliasEscapeProof;
    }
    return kind == VulkanRetiredResourceKind::Buffer
        ? kDryRunAttentionBufferGenerationRangeMissingStackProof
        : kDryRunAttentionRawGenerationRangeMissingStackProof;
  }
  return kDryRunAttentionUnknownSubresource;
}

bool is_attention_score_probability_range_class(
    const char* const resource_class) {
  const std::string key(resource_class);
  return key == kDryRunAttentionScoreProbabilitySubresource ||
      key == kDryRunAttentionScoreProbabilityRangeMissingAliasEscapeProof ||
      key == kDryRunAttentionScoreProbabilityRangeNonEscapeLastConsumer;
}

bool is_attention_raw_auxiliary_range_class(const char* const resource_class) {
  const std::string key(resource_class);
  return key == kDryRunAttentionRawAuxiliaryRangeMissingAliasEscapeProof ||
      key == kDryRunAttentionRawAuxiliaryRangeNonEscapeLastConsumer;
}

bool is_attention_non_escape_last_consumer_class(
    const char* const resource_class) {
  const std::string key(resource_class);
  return key == kDryRunAttentionScoreProbabilityRangeNonEscapeLastConsumer ||
      key == kDryRunAttentionRawAuxiliaryRangeNonEscapeLastConsumer;
}

const char* classify_stack_internal_raw_generation_range(
    const VulkanRetiredResourceRole role,
    const VulkanStackRawResourceAllocationProof& allocation_proof) {
  if (
      !allocation_proof.has_generation || !allocation_proof.has_byte_range ||
      !inside_vision_stack_phase() || current_vision_stack_block_index() < 0) {
    return kDryRunStackInternalRawGenerationRange;
  }
  switch (role) {
    case VulkanRetiredResourceRole::StackInternalTemp:
      return kDryRunStackInternalTempRawGenerationRangeMissingLastConsumer;
    case VulkanRetiredResourceRole::StackQkvOutput:
      return kDryRunStackQkvOutputRawGenerationRangeNonEscapeLastConsumer;
    case VulkanRetiredResourceRole::StackProjOutput:
      return kDryRunStackProjOutputRawGenerationRangeNonEscapeLastConsumer;
    case VulkanRetiredResourceRole::StackResidual1Output:
      return kDryRunStackResidual1OutputRawGenerationRangeNonEscapeLastConsumer;
    default:
      return kDryRunStackInternalRawGenerationRange;
  }
}

bool is_stack_raw_generation_range_evidence_class(
    const char* const resource_class) {
  const std::string key(resource_class);
  return key ==
      kDryRunStackInternalTempRawGenerationRangeMissingLastConsumer ||
      key == kDryRunStackQkvOutputRawGenerationRangeNonEscapeLastConsumer ||
      key == kDryRunStackProjOutputRawGenerationRangeNonEscapeLastConsumer ||
      key ==
      kDryRunStackResidual1OutputRawGenerationRangeNonEscapeLastConsumer;
}

bool is_stack_raw_generation_range_non_escape_last_consumer_class(
    const char* const resource_class) {
  const std::string key(resource_class);
  return key == kDryRunStackQkvOutputRawGenerationRangeNonEscapeLastConsumer ||
      key == kDryRunStackProjOutputRawGenerationRangeNonEscapeLastConsumer ||
      key ==
      kDryRunStackResidual1OutputRawGenerationRangeNonEscapeLastConsumer;
}

const char* stack_raw_last_consumer_for_dry_run(
    const VulkanRetiredResourceRole role,
    const char* const resource_class) {
  const std::string key(resource_class);
  if (key == kDryRunStackQkvOutputRawGenerationRangeNonEscapeLastConsumer) {
    return "attention";
  }
  if (key == kDryRunStackProjOutputRawGenerationRangeNonEscapeLastConsumer) {
    return "residual1";
  }
  if (
      key ==
      kDryRunStackResidual1OutputRawGenerationRangeNonEscapeLastConsumer) {
    return "norm2";
  }
  if (
      key ==
      kDryRunStackInternalTempRawGenerationRangeMissingLastConsumer) {
    return "missing";
  }
  return is_stack_temp_role(role) ? "unknown_stack_consumer"
                                  : "not_stack_raw_resource";
}

const char* attention_substep_for_dry_run(
    const VulkanRetiredResourceKind kind,
    const char* const resource_class) {
  if (is_attention_score_probability_range_class(resource_class)) {
    return "score_probability_buffer";
  }
  if (is_attention_raw_auxiliary_range_class(resource_class)) {
    return "raw_auxiliary";
  }
  if (kind == VulkanRetiredResourceKind::Buffer) {
    return "buffer_generation_range";
  }
  return "unknown_attention_subresource";
}

const char* attention_producer_for_dry_run(const char* const resource_class) {
  if (is_attention_score_probability_range_class(resource_class)) {
    return "qk_score_or_softmax_probability";
  }
  if (is_attention_raw_auxiliary_range_class(resource_class)) {
    return "attention_dispatch_auxiliary";
  }
  return "unknown_attention_producer";
}

const char* attention_last_consumer_for_dry_run(
    const char* const resource_class) {
  if (is_attention_score_probability_range_class(resource_class)) {
    return "softmax_or_value_bmm";
  }
  if (is_attention_raw_auxiliary_range_class(resource_class)) {
    return "attention_dispatch";
  }
  return "unknown_attention_consumer";
}

void note_dry_run_resource_class(
    VulkanStackSubresourceLifetimeDryRunCounters& counters,
    const char* const resource_class,
    const uint64_t bytes) {
  const std::string key(resource_class);
  if (key == kDryRunProvenStackActivation) {
    counters.proven_stack_activation_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.proven_stack_activation_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  } else if (key == kDryRunMissingStackActivationProof) {
    counters.missing_stack_activation_proof_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.missing_stack_activation_proof_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  } else if (key == kDryRunAttentionSubresource) {
    counters.attention_subresource_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.attention_subresource_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  } else if (key == kDryRunAttentionScoreProbabilitySubresource) {
    counters.attention_score_probability_subresource_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.attention_score_probability_subresource_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  } else if (key == kDryRunAttentionBufferGenerationRangeMissingStackProof) {
    counters.attention_buffer_generation_range_missing_stack_proof_count
        .fetch_add(1u, std::memory_order_relaxed);
    counters.attention_buffer_generation_range_missing_stack_proof_bytes
        .fetch_add(bytes, std::memory_order_relaxed);
  } else if (key == kDryRunAttentionRawGenerationRangeMissingStackProof) {
    counters.attention_raw_generation_range_missing_stack_proof_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.attention_raw_generation_range_missing_stack_proof_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  } else if (
      key ==
      kDryRunAttentionScoreProbabilityRangeMissingAliasEscapeProof) {
    counters.attention_score_probability_range_missing_alias_escape_proof_count
        .fetch_add(1u, std::memory_order_relaxed);
    counters.attention_score_probability_range_missing_alias_escape_proof_bytes
        .fetch_add(bytes, std::memory_order_relaxed);
  } else if (
      key == kDryRunAttentionRawAuxiliaryRangeMissingAliasEscapeProof) {
    counters.attention_raw_auxiliary_range_missing_alias_escape_proof_count
        .fetch_add(1u, std::memory_order_relaxed);
    counters.attention_raw_auxiliary_range_missing_alias_escape_proof_bytes
        .fetch_add(bytes, std::memory_order_relaxed);
  } else if (
      key == kDryRunAttentionScoreProbabilityRangeNonEscapeLastConsumer) {
    counters.attention_score_probability_range_non_escape_last_consumer_count
        .fetch_add(1u, std::memory_order_relaxed);
    counters.attention_score_probability_range_non_escape_last_consumer_bytes
        .fetch_add(bytes, std::memory_order_relaxed);
  } else if (
      key == kDryRunAttentionRawAuxiliaryRangeNonEscapeLastConsumer) {
    counters.attention_raw_auxiliary_range_non_escape_last_consumer_count
        .fetch_add(1u, std::memory_order_relaxed);
    counters.attention_raw_auxiliary_range_non_escape_last_consumer_bytes
        .fetch_add(bytes, std::memory_order_relaxed);
  } else if (key == kDryRunAttentionProvenanceMissingLastUse) {
    counters.attention_provenance_missing_last_use_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.attention_provenance_missing_last_use_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  } else if (key == kDryRunAttentionUnknownSubresource) {
    counters.attention_unknown_subresource_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.attention_unknown_subresource_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  } else if (key == kDryRunLayerNormStatBuffer) {
    counters.layernorm_stat_buffer_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.layernorm_stat_buffer_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  } else if (key == kDryRunLayerNormInternalStatBuffer) {
    counters.layernorm_internal_stat_buffer_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.layernorm_internal_stat_buffer_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  } else if (key == kDryRunMetadataUniform) {
    counters.metadata_uniform_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.metadata_uniform_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  } else if (
      key == kDryRunRawNoProvenance ||
      key == kDryRunNonStackSetupStagingPending ||
      key == kDryRunUnscopedRawBufferNoStackProof) {
    counters.raw_no_provenance_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.raw_no_provenance_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  } else if (key == kDryRunStackInternalRawMissingGeneration) {
    counters.stack_internal_raw_missing_generation_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.stack_internal_raw_missing_generation_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  } else if (key == kDryRunStackInternalRawGenerationRange) {
    counters.stack_internal_raw_generation_range_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.stack_internal_raw_generation_range_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  } else if (
      key ==
      kDryRunStackInternalTempRawGenerationRangeMissingLastConsumer) {
    counters.stack_internal_temp_raw_generation_range_missing_last_consumer_count
        .fetch_add(1u, std::memory_order_relaxed);
    counters.stack_internal_temp_raw_generation_range_missing_last_consumer_bytes
        .fetch_add(bytes, std::memory_order_relaxed);
  } else if (
      key == kDryRunStackQkvOutputRawGenerationRangeNonEscapeLastConsumer) {
    counters
        .stack_qkv_output_raw_generation_range_non_escape_last_consumer_count
        .fetch_add(1u, std::memory_order_relaxed);
    counters
        .stack_qkv_output_raw_generation_range_non_escape_last_consumer_bytes
        .fetch_add(bytes, std::memory_order_relaxed);
  } else if (
      key == kDryRunStackProjOutputRawGenerationRangeNonEscapeLastConsumer) {
    counters
        .stack_proj_output_raw_generation_range_non_escape_last_consumer_count
        .fetch_add(1u, std::memory_order_relaxed);
    counters
        .stack_proj_output_raw_generation_range_non_escape_last_consumer_bytes
        .fetch_add(bytes, std::memory_order_relaxed);
  } else if (
      key ==
      kDryRunStackResidual1OutputRawGenerationRangeNonEscapeLastConsumer) {
    counters
        .stack_residual1_output_raw_generation_range_non_escape_last_consumer_count
        .fetch_add(1u, std::memory_order_relaxed);
    counters
        .stack_residual1_output_raw_generation_range_non_escape_last_consumer_bytes
        .fetch_add(bytes, std::memory_order_relaxed);
  } else if (key == kDryRunTrulyUnknownRawResource) {
    counters.truly_unknown_raw_resource_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.truly_unknown_raw_resource_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  } else if (key == kDryRunHostVisibleOrRequestedOutput) {
    counters.host_visible_or_requested_output_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.host_visible_or_requested_output_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  } else if (key == kDryRunAllocatorOrScratchBacking) {
    counters.allocator_or_scratch_backing_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.allocator_or_scratch_backing_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  }
}

bool has_proven_internal_stack_temp_lifetime(
    const VulkanStackRetireProvenance& provenance) {
  return provenance.defined && provenance.has_last_use_proof &&
      provenance.internal_non_escaping &&
      provenance.final_consumer_before_stack_submit &&
      !provenance.requested_intermediate && !provenance.escapes_stack &&
      !provenance.final_output && !provenance.alias_or_view &&
      !provenance.aliases_runtime_input &&
      !provenance.aliases_runtime_output &&
      provenance.lifetime == VulkanStackTensorLifetimeClass::InternalTemp;
}

const char* stack_drain_blocker_reason(
    const VulkanRetiredResourceKind kind,
    const VulkanRetiredResourceRole role,
    const VulkanStackRetireProvenance& provenance,
    const bool qkv_would_batch) {
  if (qkv_would_batch) {
    return "qkv_would_batch";
  }
  if (role == VulkanRetiredResourceRole::StackInternalTemp) {
    return "generic_stack_internal_temp";
  }
  if (provenance.defined) {
    if (provenance.requested_intermediate || provenance.escapes_stack) {
      return "requested_intermediate";
    }
    if (!provenance.has_last_use_proof && is_stack_temp_role(role)) {
      return "missing_proof";
    }
  }
  switch (role) {
    case VulkanRetiredResourceRole::NativeLayerNormUniform:
    case VulkanRetiredResourceRole::NativeLayerNormMetadata:
    case VulkanRetiredResourceRole::AttentionMetadata:
    case VulkanRetiredResourceRole::LinearMetadata:
    case VulkanRetiredResourceRole::ConvMetadata:
    case VulkanRetiredResourceRole::ResidualAddMetadata:
      return "metadata_or_uniform";
    default:
      break;
  }
  if (
      kind == VulkanRetiredResourceKind::UniformBuffer ||
      kind == VulkanRetiredResourceKind::MetadataBuffer) {
    return "metadata_or_uniform";
  }
  return "other_role";
}

const char* stack_provenance_loss_reason(
    const VulkanRetiredResourceRole role,
    const VulkanStackRetireProvenance& provenance) {
  if (!provenance.defined) {
    return "no_stack_provenance";
  }
  if (provenance.has_last_use_proof) {
    return "none";
  }
  if (!is_stack_temp_role(role)) {
    return "not_stack_temp";
  }
  if (
      provenance.source ==
      VulkanStackRetireProvenanceSource::ProgramScratchArenaBackingStorage) {
    return "program_scratch_arena_backing_storage";
  }
  if (provenance.shape.size() == 1u) {
    return "physical_raw_storage_identity";
  }
  if (provenance.dtype < 0) {
    return "missing_dtype";
  }
  return "no_matching_stack_plan_proof";
}

const char* stack_region_lifetime_missing_proof_reason(
    const VulkanRetiredResourceKind kind,
    const VulkanRetiredResourceRole role,
    const char* const resource_class,
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof,
    const bool safe_candidate,
    const bool large_backing,
    const bool formal_last_use_proof) {
  const std::string key(resource_class ? resource_class : "");
  if ((safe_candidate || formal_last_use_proof) && !large_backing) {
    return "none";
  }
  if (large_backing) {
    return "large_backing_excluded";
  }
  if (key == kDryRunHostVisibleOrRequestedOutput) {
    return "host_visible_or_requested_output";
  }
  if (key == kDryRunCaptureSensitiveStackActivation) {
    return "capture_sensitive_output_dependency";
  }
  if (key == kDryRunAllocatorOrScratchBacking) {
    return "allocator_or_scratch_backing";
  }
  if (key == kDryRunStackInternalRawMissingGeneration) {
    return "missing_allocation_generation";
  }
  if (
      key == kDryRunRawNoProvenance ||
      key == kDryRunTrulyUnknownRawResource ||
      key == kDryRunNonStackSetupStagingPending ||
      key == kDryRunUnscopedRawBufferNoStackProof) {
    if (!allocation_proof.has_generation) {
      return "missing_allocation_generation";
    }
    if (!allocation_proof.has_byte_range) {
      return "missing_byte_range";
    }
    if (key == kDryRunNonStackSetupStagingPending) {
      return "non_stack_setup_staging_pending";
    }
    if (key == kDryRunUnscopedRawBufferNoStackProof) {
      return "missing_stack_scope_proof";
    }
    return "truly_unknown_raw_resource";
  }
  if (!allocation_proof.has_byte_range && is_stack_temp_role(role)) {
    return "missing_byte_range";
  }
  if (
      key ==
      kDryRunStackInternalTempRawGenerationRangeMissingLastConsumer) {
    return "missing_last_consumer";
  }
  if (is_stack_raw_generation_range_non_escape_last_consumer_class(
          resource_class)) {
    return "missing_formal_raw_last_use_proof";
  }
  if (key == kDryRunAttentionRawAuxiliaryRangeNonEscapeLastConsumer) {
    return "missing_formal_attention_auxiliary_last_use_proof";
  }
  if (key == kDryRunAttentionScoreProbabilityRangeNonEscapeLastConsumer) {
    return "missing_formal_attention_probability_last_use_proof";
  }
  if (
      key == kDryRunAttentionRawAuxiliaryRangeMissingAliasEscapeProof ||
      key == kDryRunAttentionScoreProbabilityRangeMissingAliasEscapeProof) {
    return "missing_alias_escape_proof";
  }
  if (
      key == kDryRunAttentionBufferGenerationRangeMissingStackProof ||
      key == kDryRunAttentionRawGenerationRangeMissingStackProof) {
    return "missing_stack_scope_proof";
  }
  if (
      key == kDryRunAttentionProvenanceMissingLastUse ||
      (!provenance.has_last_use_proof && is_stack_temp_role(role))) {
    return "missing_last_use_proof";
  }
  if (
      provenance.defined &&
      (!provenance.internal_non_escaping || provenance.alias_or_view ||
       provenance.aliases_runtime_input || provenance.aliases_runtime_output)) {
    return "missing_non_escape_or_alias_proof";
  }
  const char* const provenance_loss =
      stack_provenance_loss_reason(role, provenance);
  if (std::string(provenance_loss) != "none") {
    return provenance_loss;
  }
  (void)kind;
  return "unsafe_resource_class";
}

const char* stack_raw_producer_substep_for_label(
    const std::string& allocation_label);
const char* stack_raw_last_consumer_for_label(
    const std::string& allocation_label);
int64_t stack_raw_block_index_for_label(const std::string& allocation_label);
VulkanVisionStackPhase stack_raw_last_consumer_phase_for_label(
    VulkanRetiredResourceRole role,
    const std::string& allocation_label);
bool stack_subresource_lifetime_dry_run_is_formal_norm2_last_use_label(
    const char* resource_class,
    const std::string& allocation_label);

const char* stack_region_lifetime_last_use_candidate(
    const VulkanRetiredResourceKind kind,
    const VulkanRetiredResourceRole role,
    const char* const resource_class,
    const VulkanStackRetireProvenance& provenance,
    const std::string& allocation_label) {
  if (provenance.has_last_use_proof) {
    return vision_stack_phase_name(provenance.expected_consumer_phase);
  }
  if (stack_subresource_lifetime_dry_run_is_formal_norm2_last_use_label(
          resource_class, allocation_label)) {
    return stack_raw_last_consumer_for_label(allocation_label);
  }
  if (
      is_attention_score_probability_range_class(resource_class) ||
      is_attention_raw_auxiliary_range_class(resource_class)) {
    return attention_last_consumer_for_dry_run(resource_class);
  }
  if (is_stack_raw_generation_range_evidence_class(resource_class)) {
    return stack_raw_last_consumer_for_dry_run(role, resource_class);
  }
  if (kind == VulkanRetiredResourceKind::UniformBuffer) {
    return "descriptor_consumer";
  }
  return "unknown";
}

const char* stack_region_lifetime_producer_substep(
    const VulkanRetiredResourceKind kind,
    const VulkanRetiredResourceRole role,
    const char* const resource_class,
    const VulkanStackRetireProvenance& provenance,
    const std::string& allocation_label) {
  if (stack_subresource_lifetime_dry_run_is_formal_norm2_last_use_label(
          resource_class, allocation_label)) {
    return stack_raw_producer_substep_for_label(allocation_label);
  }
  if (
      is_attention_score_probability_range_class(resource_class) ||
      is_attention_raw_auxiliary_range_class(resource_class)) {
    return attention_producer_for_dry_run(resource_class);
  }
  if (provenance.phase != VulkanVisionStackPhase::Unknown) {
    return vision_stack_phase_name(provenance.phase);
  }
  if (is_metadata_or_uniform_resource(kind, role)) {
    return "metadata_or_uniform";
  }
  return "unknown";
}

int stack_phase_execution_order(const VulkanVisionStackPhase phase) {
  switch (phase) {
    case VulkanVisionStackPhase::BlockEntry:
      return 0;
    case VulkanVisionStackPhase::Norm1:
      return 1;
    case VulkanVisionStackPhase::QkvLinear:
      return 2;
    case VulkanVisionStackPhase::QkvTransform:
      return 3;
    case VulkanVisionStackPhase::Attention:
      return 4;
    case VulkanVisionStackPhase::ProjLinear:
      return 5;
    case VulkanVisionStackPhase::Residual1:
      return 6;
    case VulkanVisionStackPhase::Norm2:
      return 7;
    case VulkanVisionStackPhase::Fc1Gelu:
      return 8;
    case VulkanVisionStackPhase::Fc2:
      return 9;
    case VulkanVisionStackPhase::Residual2:
      return 10;
    case VulkanVisionStackPhase::IntermediateCapture:
      return 11;
    case VulkanVisionStackPhase::StackExit:
      return 12;
    case VulkanVisionStackPhase::StackEntry:
      return -1;
    case VulkanVisionStackPhase::Unknown:
    default:
      return -1;
  }
}

bool stack_phase_has_reached_consumer(
    const VulkanVisionStackPhase current_phase,
    const int64_t current_block,
    const VulkanVisionStackPhase consumer_phase,
    const int64_t consumer_block) {
  if (
      current_phase == VulkanVisionStackPhase::Unknown || current_block < 0 ||
      consumer_phase == VulkanVisionStackPhase::Unknown || consumer_block < 0) {
    return false;
  }
  if (consumer_block < current_block) {
    return true;
  }
  if (consumer_block > current_block) {
    return false;
  }
  const int current_order = stack_phase_execution_order(current_phase);
  const int consumer_order = stack_phase_execution_order(consumer_phase);
  return current_order >= 0 && consumer_order >= 0 &&
      current_order >= consumer_order;
}

bool stack_activation_phase_boundary_carry_candidate(
    const VulkanRetiredResourceRole role,
    const VulkanStackRetireProvenance& provenance) {
  return role == VulkanRetiredResourceRole::StackResidual2Output &&
      provenance.defined && provenance.has_last_use_proof &&
      provenance.lifetime ==
          VulkanStackTensorLifetimeClass::BlockOutputForNextBlock &&
      provenance.phase == VulkanVisionStackPhase::Residual2 &&
      provenance.producer_role == role && provenance.block_index >= 0 &&
      provenance.expected_consumer_phase == VulkanVisionStackPhase::Norm1 &&
      provenance.expected_consumer_block_index == provenance.block_index + 1 &&
      provenance.final_consumer_before_stack_submit &&
      !provenance.escapes_stack && !provenance.requested_intermediate &&
      !provenance.final_output && !provenance.alias_or_view &&
      !provenance.aliases_runtime_input &&
      !provenance.aliases_runtime_output && provenance.direct_buffer &&
      provenance.buffer_storage && !provenance.image_storage &&
      current_vision_stack_phase() == VulkanVisionStackPhase::BlockEntry &&
      current_vision_stack_block_index() ==
      provenance.expected_consumer_block_index;
}

bool stack_activation_phase_boundary_carry_proof(
    const VulkanRetiredResourceRole role,
    const VulkanStackRetireProvenance& provenance) {
  if (!stack_activation_phase_boundary_carry_candidate(role, provenance)) {
    return false;
  }
  if (!vision_stack_capture_dependency_active()) {
    return false;
  }
  if (vision_stack_capture_dependency_reaches_block(
          provenance.expected_consumer_block_index)) {
    return false;
  }
  return true;
}

bool is_stack_temp_role(const VulkanRetiredResourceRole role) {
  switch (role) {
    case VulkanRetiredResourceRole::StackInternalTemp:
    case VulkanRetiredResourceRole::StackNorm1Output:
    case VulkanRetiredResourceRole::StackQkvOutput:
    case VulkanRetiredResourceRole::StackQView:
    case VulkanRetiredResourceRole::StackKView:
    case VulkanRetiredResourceRole::StackVView:
    case VulkanRetiredResourceRole::StackAttentionOutput:
    case VulkanRetiredResourceRole::StackProjOutput:
    case VulkanRetiredResourceRole::StackResidual1Output:
    case VulkanRetiredResourceRole::StackNorm2Output:
    case VulkanRetiredResourceRole::StackFc1GeluOutput:
    case VulkanRetiredResourceRole::StackFc2Output:
    case VulkanRetiredResourceRole::StackResidual2Output:
    case VulkanRetiredResourceRole::StackRequestedOutput:
    case VulkanRetiredResourceRole::StackFinalOutput:
      return true;
    default:
      return false;
  }
}

VulkanStackTempLifetimeSafety classify_stack_temp_lifetime_safety(
    const VulkanRetiredResourceRole role,
    const VulkanStackRetireProvenance& provenance) {
  if (provenance.defined) {
    if (provenance.requested_intermediate ||
        provenance.lifetime ==
            VulkanStackTensorLifetimeClass::RequestedIntermediateOutput) {
      return VulkanStackTempLifetimeSafety::EscapesAsRequestedIntermediate;
    }
    if (provenance.final_output ||
        provenance.lifetime == VulkanStackTensorLifetimeClass::FinalStackOutput) {
      return VulkanStackTempLifetimeSafety::EscapesAsFinalOutput;
    }
    if (provenance.aliases_runtime_input) {
      return VulkanStackTempLifetimeSafety::AliasesRuntimeInput;
    }
    if (provenance.aliases_runtime_output) {
      return VulkanStackTempLifetimeSafety::AliasesRuntimeOutput;
    }
    if (provenance.alias_or_view ||
        provenance.lifetime == VulkanStackTensorLifetimeClass::AliasOrView) {
      return VulkanStackTempLifetimeSafety::UnsafeUnknownConsumer;
    }
    if (provenance.escapes_stack) {
      return VulkanStackTempLifetimeSafety::UnsafeUnknownConsumer;
    }
    if (
        provenance.has_last_use_proof && provenance.internal_non_escaping &&
        provenance.final_consumer_before_stack_submit &&
        provenance.lifetime == VulkanStackTensorLifetimeClass::InternalTemp) {
      return VulkanStackTempLifetimeSafety::SafeToDeferUntilStackSubmit;
    }
    if (
        provenance.lifetime ==
        VulkanStackTensorLifetimeClass::BlockOutputForNextBlock) {
      return VulkanStackTempLifetimeSafety::MustRetireAtPhaseBoundary;
    }
    if (
        provenance.lifetime == VulkanStackTensorLifetimeClass::InternalTemp) {
      return VulkanStackTempLifetimeSafety::UnsafeUnknownConsumer;
    }
  }
  switch (role) {
    case VulkanRetiredResourceRole::StackRequestedOutput:
      return VulkanStackTempLifetimeSafety::EscapesAsRequestedIntermediate;
    case VulkanRetiredResourceRole::StackFinalOutput:
      return VulkanStackTempLifetimeSafety::EscapesAsFinalOutput;
    case VulkanRetiredResourceRole::StackInternalTemp:
    case VulkanRetiredResourceRole::StackNorm1Output:
    case VulkanRetiredResourceRole::StackQkvOutput:
    case VulkanRetiredResourceRole::StackQView:
    case VulkanRetiredResourceRole::StackKView:
    case VulkanRetiredResourceRole::StackVView:
    case VulkanRetiredResourceRole::StackAttentionOutput:
    case VulkanRetiredResourceRole::StackProjOutput:
    case VulkanRetiredResourceRole::StackResidual1Output:
    case VulkanRetiredResourceRole::StackNorm2Output:
    case VulkanRetiredResourceRole::StackFc1GeluOutput:
    case VulkanRetiredResourceRole::StackFc2Output:
    case VulkanRetiredResourceRole::StackResidual2Output:
      return VulkanStackTempLifetimeSafety::UnsafeUnknownConsumer;
    default:
      return VulkanStackTempLifetimeSafety::Unknown;
  }
}

VulkanSubmitPhaseScope::VulkanSubmitPhaseScope(VulkanSubmitPhase phase)
    : previous_(g_submit_phase) {
  g_submit_phase = phase;
}

VulkanSubmitPhaseScope::~VulkanSubmitPhaseScope() {
  g_submit_phase = previous_;
}

VulkanRetiredResourceScope::VulkanRetiredResourceScope(
    VulkanRetiredResourceKind kind,
    VulkanRetiredResourceRole role)
    : previous_kind_(g_retired_resource_kind),
      previous_role_(g_retired_resource_role) {
  g_retired_resource_kind = kind;
  g_retired_resource_role = role;
}

VulkanRetiredResourceScope::~VulkanRetiredResourceScope() {
  g_retired_resource_kind = previous_kind_;
  g_retired_resource_role = previous_role_;
}

VulkanStackLastUseProofScope::VulkanStackLastUseProofScope(
    std::vector<VulkanStackLastUseProof> proofs)
    : previous_(std::move(g_stack_last_use_proofs)) {
  g_stack_last_use_proofs = std::move(proofs);
}

VulkanStackLastUseProofScope::~VulkanStackLastUseProofScope() {
  g_stack_last_use_proofs = std::move(previous_);
}

VulkanVisionStackPhaseScope::VulkanVisionStackPhaseScope(
    VulkanVisionStackPhase phase)
    : previous_(g_vision_stack_phase) {
  g_vision_stack_phase = phase;
}

VulkanVisionStackPhaseScope::~VulkanVisionStackPhaseScope() {
  g_vision_stack_phase = previous_;
}

VulkanVisionStackBlockScope::VulkanVisionStackBlockScope(
    const int64_t block_index)
    : previous_(g_vision_stack_block_index) {
  g_vision_stack_block_index = block_index;
}

VulkanVisionStackBlockScope::~VulkanVisionStackBlockScope() {
  g_vision_stack_block_index = previous_;
}

VulkanVisionStackCaptureScope::VulkanVisionStackCaptureScope(
    std::vector<int64_t> capture_indices)
    : previous_(std::move(g_vision_stack_capture_indices)) {
  g_vision_stack_capture_indices = std::move(capture_indices);
}

VulkanVisionStackCaptureScope::~VulkanVisionStackCaptureScope() {
  g_vision_stack_capture_indices = std::move(previous_);
}

void reset_vulkan_sync_counters() {
  VulkanSyncCounters& counters = vulkan_sync_counters();
  counters.compute_dispatch_count.store(0u, std::memory_order_relaxed);
  counters.submit_compute_job_count.store(0u, std::memory_order_relaxed);
  counters.stream_submit_count.store(0u, std::memory_order_relaxed);
  counters.event_record_count.store(0u, std::memory_order_relaxed);
  counters.event_block_count.store(0u, std::memory_order_relaxed);
  counters.event_wait_count.store(0u, std::memory_order_relaxed);
  counters.retire_poll_count.store(0u, std::memory_order_relaxed);
  counters.retired_resource_count.store(0u, std::memory_order_relaxed);
  counters.queue_wait_idle_count.store(0u, std::memory_order_relaxed);
  counters.forced_sync_count.store(0u, std::memory_order_relaxed);
  counters.fallback_sync_readback_count.store(0u, std::memory_order_relaxed);
  counters.allocation_record_stream_count.store(0u, std::memory_order_relaxed);
  counters.allocation_reuse_deferred_count.store(0u, std::memory_order_relaxed);
  counters.allocation_reuse_after_timeline_count.store(
      0u, std::memory_order_relaxed);
  counters.forced_sync_explicit_synchronize_count.store(
      0u, std::memory_order_relaxed);
  counters.forced_sync_tensor_cpu_readback_count.store(
      0u, std::memory_order_relaxed);
  counters.forced_sync_event_synchronize_count.store(
      0u, std::memory_order_relaxed);
  counters.forced_sync_retire_queue_drain_count.store(
      0u, std::memory_order_relaxed);
  counters.forced_sync_gpu_timestamp_query_reset_count.store(
      0u, std::memory_order_relaxed);
  counters.forced_sync_fallback_policy_readback_count.store(
      0u, std::memory_order_relaxed);
  counters.forced_sync_unknown_count.store(0u, std::memory_order_relaxed);
}

void reset_vulkan_submit_origin_counters() {
  VulkanSubmitOriginCounters& counters = vulkan_submit_origin_counters();
  counters.total_queue_submits.store(0u, std::memory_order_relaxed);
  counters.normal_cmd_submit_frequency.store(0u, std::memory_order_relaxed);
  counters.stack_planned_recording_submit.store(0u, std::memory_order_relaxed);
  counters.pre_stack_flush.store(0u, std::memory_order_relaxed);
  counters.post_stack_flush.store(0u, std::memory_order_relaxed);
  counters.explicit_synchronize.store(0u, std::memory_order_relaxed);
  counters.tensor_cpu_readback.store(0u, std::memory_order_relaxed);
  counters.fallback_readback.store(0u, std::memory_order_relaxed);
  counters.retire_queue_drain.store(0u, std::memory_order_relaxed);
  counters.profiling_timestamp_reset.store(0u, std::memory_order_relaxed);
  counters.profiling_timestamp_readback.store(0u, std::memory_order_relaxed);
  counters.shutdown.store(0u, std::memory_order_relaxed);
  counters.debug_validation.store(0u, std::memory_order_relaxed);
  counters.conv_prepack_upload.store(0u, std::memory_order_relaxed);
  counters.unknown.store(0u, std::memory_order_relaxed);
}

void reset_vulkan_submit_origin_phase_counters() {
  VulkanSubmitOriginPhaseCounters& counters =
      vulkan_submit_origin_phase_counters();
  for (auto& origin_counts : counters.counts) {
    for (auto& count : origin_counts) {
      count.store(0u, std::memory_order_relaxed);
    }
  }
}

void reset_vulkan_retire_drain_counters() {
  VulkanRetireDrainCounters& counters = vulkan_retire_drain_counters();
  counters.total.store(0u, std::memory_order_relaxed);
  counters.queue_submit_count.store(0u, std::memory_order_relaxed);
  counters.blocking_wait_count.store(0u, std::memory_order_relaxed);
  counters.poll_only_count.store(0u, std::memory_order_relaxed);
  counters.pending_resource_count_total.store(0u, std::memory_order_relaxed);
  counters.pending_bytes_total.store(0u, std::memory_order_relaxed);
  counters.explicit_drain.store(0u, std::memory_order_relaxed);
  counters.shutdown.store(0u, std::memory_order_relaxed);
  counters.resource_pressure.store(0u, std::memory_order_relaxed);
  counters.descriptor_pool_pressure.store(0u, std::memory_order_relaxed);
  counters.command_buffer_recycle.store(0u, std::memory_order_relaxed);
  counters.readback_preparation.store(0u, std::memory_order_relaxed);
  counters.synchronize.store(0u, std::memory_order_relaxed);
  counters.stack_scope_end.store(0u, std::memory_order_relaxed);
  counters.decoder_phase.store(0u, std::memory_order_relaxed);
  counters.setup_phase.store(0u, std::memory_order_relaxed);
  counters.debug_validation.store(0u, std::memory_order_relaxed);
  counters.unknown.store(0u, std::memory_order_relaxed);
}

void reset_retire_call_site_counters() {
  for (auto& counter : retire_call_site_counters()) {
    counter.total.store(0u, std::memory_order_relaxed);
    counter.queue_submit_count.store(0u, std::memory_order_relaxed);
    counter.blocking_wait_count.store(0u, std::memory_order_relaxed);
    counter.poll_only_count.store(0u, std::memory_order_relaxed);
    counter.pending_resource_count_total.store(0u, std::memory_order_relaxed);
    counter.pending_bytes_total.store(0u, std::memory_order_relaxed);
  }
}

void reset_retired_resource_aggregate() {
  std::lock_guard<std::mutex> lock(retired_resource_aggregate_mutex());
  retired_resource_aggregate().clear();
}

void reset_stack_temp_lifetime_safety_snapshot() {
  std::lock_guard<std::mutex> lock(stack_temp_lifetime_safety_mutex());
  stack_temp_lifetime_safety_aggregate().clear();
}

void reset_stack_scratch_arena_lifetime_snapshot() {
  std::lock_guard<std::mutex> lock(stack_scratch_arena_lifetime_mutex());
  stack_scratch_arena_lifetime_aggregate().clear();
}

void reset_stack_internal_temp_retire_batch_counters() {
  auto& counters = stack_internal_temp_retire_batch_counters();
  counters.total_attempts.store(0u, std::memory_order_relaxed);
  counters.batch_candidate_count.store(0u, std::memory_order_relaxed);
  counters.batch_candidate_bytes.store(0u, std::memory_order_relaxed);
  counters.batch_accepted_count.store(0u, std::memory_order_relaxed);
  counters.batch_accepted_bytes.store(0u, std::memory_order_relaxed);
  counters.batch_rejected_count.store(0u, std::memory_order_relaxed);
  counters.batch_rejected_bytes.store(0u, std::memory_order_relaxed);
  counters.submitted_batch_count.store(0u, std::memory_order_relaxed);
  counters.submitted_batch_bytes.store(0u, std::memory_order_relaxed);
  counters.rejected_not_target_role.store(0u, std::memory_order_relaxed);
  counters.rejected_missing_proof.store(0u, std::memory_order_relaxed);
  counters.rejected_not_internal_non_escaping.store(
      0u, std::memory_order_relaxed);
  counters.rejected_consumer_after_submit.store(
      0u, std::memory_order_relaxed);
  counters.rejected_requested_intermediate.store(0u, std::memory_order_relaxed);
  counters.rejected_final_output.store(0u, std::memory_order_relaxed);
  counters.rejected_alias.store(0u, std::memory_order_relaxed);
  counters.rejected_runtime_alias.store(0u, std::memory_order_relaxed);
  counters.rejected_lifetime.store(0u, std::memory_order_relaxed);
  counters.rejected_not_stack_recording.store(0u, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(stack_temp_retire_batch_snapshot_mutex());
  stack_temp_retire_batch_decisions().clear();
}

void reset_stack_retire_drain_blocker_counters() {
  auto& counters = stack_retire_drain_blocker_counters();
  counters.total_drains.store(0u, std::memory_order_relaxed);
  counters.queue_submit_drains.store(0u, std::memory_order_relaxed);
  counters.drains_with_old_path_pending.store(0u, std::memory_order_relaxed);
  counters.drains_with_only_already_batched.store(
      0u, std::memory_order_relaxed);
  counters.drains_qkv_would_remove.store(0u, std::memory_order_relaxed);
  counters.drains_blocked_requested_intermediate.store(
      0u, std::memory_order_relaxed);
  counters.drains_blocked_missing_proof.store(0u, std::memory_order_relaxed);
  counters.drains_blocked_generic_stack_internal_temp.store(
      0u, std::memory_order_relaxed);
  counters.drains_blocked_metadata_or_uniform.store(
      0u, std::memory_order_relaxed);
  counters.drains_blocked_other_roles.store(0u, std::memory_order_relaxed);
  counters.old_path_pending_count.store(0u, std::memory_order_relaxed);
  counters.old_path_pending_bytes.store(0u, std::memory_order_relaxed);
  counters.qkv_hypothetical_count.store(0u, std::memory_order_relaxed);
  counters.qkv_hypothetical_bytes.store(0u, std::memory_order_relaxed);
  counters.skipped_no_old_path_pending.store(0u, std::memory_order_relaxed);
  counters.skipped_no_pending_command_work.store(
      0u, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(stack_retire_drain_blocker_snapshot_mutex());
  stack_retire_drain_blockers().clear();
}

void reset_region_lifetime_submit_attribution() {
  std::lock_guard<std::mutex> lock(region_lifetime_submit_attribution_mutex());
  region_lifetime_submit_attribution_rows().clear();
}

void reset_stack_subresource_lifetime_dry_run_counters() {
  auto& counters = stack_subresource_lifetime_dry_run_counters();
  counters.total_groups.store(0u, std::memory_order_relaxed);
  counters.queue_submit_groups.store(0u, std::memory_order_relaxed);
  counters.groups_with_old_path_pending.store(0u, std::memory_order_relaxed);
  counters.all_safe_group_eligible.store(0u, std::memory_order_relaxed);
  counters.would_remove_submit_drains.store(0u, std::memory_order_relaxed);
  counters.actual_removed_submit_drains.store(0u, std::memory_order_relaxed);
  counters.peak_extra_live_bytes_estimate.store(0u, std::memory_order_relaxed);
  counters.skipped_no_old_path_pending.store(0u, std::memory_order_relaxed);
  counters.proven_stack_activation_count.store(0u, std::memory_order_relaxed);
  counters.missing_stack_activation_proof_count.store(
      0u, std::memory_order_relaxed);
  counters.attention_subresource_count.store(0u, std::memory_order_relaxed);
  counters.attention_score_probability_subresource_count.store(
      0u, std::memory_order_relaxed);
  counters.layernorm_stat_buffer_count.store(0u, std::memory_order_relaxed);
  counters.layernorm_internal_stat_buffer_count.store(
      0u, std::memory_order_relaxed);
  counters.metadata_uniform_count.store(0u, std::memory_order_relaxed);
  counters.raw_no_provenance_count.store(0u, std::memory_order_relaxed);
  counters.stack_internal_raw_missing_generation_count.store(
      0u, std::memory_order_relaxed);
  counters.stack_internal_raw_generation_range_count.store(
      0u, std::memory_order_relaxed);
  counters.truly_unknown_raw_resource_count.store(
      0u, std::memory_order_relaxed);
  counters.host_visible_or_requested_output_count.store(
      0u, std::memory_order_relaxed);
  counters.allocator_or_scratch_backing_count.store(
      0u, std::memory_order_relaxed);
  counters.proven_stack_activation_bytes.store(0u, std::memory_order_relaxed);
  counters.missing_stack_activation_proof_bytes.store(
      0u, std::memory_order_relaxed);
  counters.attention_subresource_bytes.store(0u, std::memory_order_relaxed);
  counters.attention_score_probability_subresource_bytes.store(
      0u, std::memory_order_relaxed);
  counters.layernorm_stat_buffer_bytes.store(0u, std::memory_order_relaxed);
  counters.layernorm_internal_stat_buffer_bytes.store(
      0u, std::memory_order_relaxed);
  counters.metadata_uniform_bytes.store(0u, std::memory_order_relaxed);
  counters.raw_no_provenance_bytes.store(0u, std::memory_order_relaxed);
  counters.stack_internal_raw_missing_generation_bytes.store(
      0u, std::memory_order_relaxed);
  counters.stack_internal_raw_generation_range_bytes.store(
      0u, std::memory_order_relaxed);
  counters.truly_unknown_raw_resource_bytes.store(
      0u, std::memory_order_relaxed);
  counters.host_visible_or_requested_output_bytes.store(
      0u, std::memory_order_relaxed);
  counters.allocator_or_scratch_backing_bytes.store(
      0u, std::memory_order_relaxed);
  counters.rejected_unsafe_resource_class.store(0u, std::memory_order_relaxed);
  counters.rejected_over_block_budget.store(0u, std::memory_order_relaxed);
  counters.rejected_over_scope_budget.store(0u, std::memory_order_relaxed);
  counters.rejected_large_backing.store(0u, std::memory_order_relaxed);
  counters.attention_buffer_generation_range_missing_stack_proof_count.store(
      0u, std::memory_order_relaxed);
  counters.attention_raw_generation_range_missing_stack_proof_count.store(
      0u, std::memory_order_relaxed);
  counters.attention_provenance_missing_last_use_count.store(
      0u, std::memory_order_relaxed);
  counters.attention_unknown_subresource_count.store(
      0u, std::memory_order_relaxed);
  counters.attention_buffer_generation_range_missing_stack_proof_bytes.store(
      0u, std::memory_order_relaxed);
  counters.attention_raw_generation_range_missing_stack_proof_bytes.store(
      0u, std::memory_order_relaxed);
  counters.attention_provenance_missing_last_use_bytes.store(
      0u, std::memory_order_relaxed);
  counters.attention_unknown_subresource_bytes.store(
      0u, std::memory_order_relaxed);
  counters.attention_score_probability_range_missing_alias_escape_proof_count
      .store(0u, std::memory_order_relaxed);
  counters.attention_raw_auxiliary_range_missing_alias_escape_proof_count.store(
      0u, std::memory_order_relaxed);
  counters.attention_score_probability_range_missing_alias_escape_proof_bytes
      .store(0u, std::memory_order_relaxed);
  counters.attention_raw_auxiliary_range_missing_alias_escape_proof_bytes.store(
      0u, std::memory_order_relaxed);
  counters.attention_score_probability_range_non_escape_last_consumer_count
      .store(0u, std::memory_order_relaxed);
  counters.attention_raw_auxiliary_range_non_escape_last_consumer_count.store(
      0u, std::memory_order_relaxed);
  counters.attention_score_probability_range_non_escape_last_consumer_bytes
      .store(0u, std::memory_order_relaxed);
  counters.attention_raw_auxiliary_range_non_escape_last_consumer_bytes.store(
      0u, std::memory_order_relaxed);
  counters.stack_internal_temp_raw_generation_range_missing_last_consumer_count
      .store(0u, std::memory_order_relaxed);
  counters.stack_qkv_output_raw_generation_range_non_escape_last_consumer_count
      .store(0u, std::memory_order_relaxed);
  counters.stack_proj_output_raw_generation_range_non_escape_last_consumer_count
      .store(0u, std::memory_order_relaxed);
  counters
      .stack_residual1_output_raw_generation_range_non_escape_last_consumer_count
      .store(0u, std::memory_order_relaxed);
  counters.stack_internal_temp_raw_generation_range_missing_last_consumer_bytes
      .store(0u, std::memory_order_relaxed);
  counters.stack_qkv_output_raw_generation_range_non_escape_last_consumer_bytes
      .store(0u, std::memory_order_relaxed);
  counters.stack_proj_output_raw_generation_range_non_escape_last_consumer_bytes
      .store(0u, std::memory_order_relaxed);
  counters
      .stack_residual1_output_raw_generation_range_non_escape_last_consumer_bytes
      .store(0u, std::memory_order_relaxed);
  counters.phase_boundary_total_groups.store(0u, std::memory_order_relaxed);
  counters.phase_boundary_all_safe_group_eligible.store(
      0u, std::memory_order_relaxed);
  counters.phase_boundary_would_remove_explicit_synchronizes.store(
      0u, std::memory_order_relaxed);
  counters.phase_boundary_actual_removed_explicit_synchronizes.store(
      0u, std::memory_order_relaxed);
  counters.phase_boundary_rejected_unsafe_resource_class.store(
      0u, std::memory_order_relaxed);
  counters.phase_boundary_rejected_over_block_budget.store(
      0u, std::memory_order_relaxed);
  counters.phase_boundary_rejected_over_scope_budget.store(
      0u, std::memory_order_relaxed);
  counters.phase_boundary_rejected_large_backing.store(
      0u, std::memory_order_relaxed);
  counters.phase_boundary_stack_activation_carry_proof_count.store(
      0u, std::memory_order_relaxed);
  counters.phase_boundary_stack_activation_carry_proof_bytes.store(
      0u, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(stack_subresource_lifetime_dry_run_mutex());
  stack_subresource_lifetime_dry_run_rows().clear();
}

void note_vulkan_queue_submit(VulkanSubmitOrigin origin) {
  VulkanSubmitOriginCounters& counters = vulkan_submit_origin_counters();
  counters.total_queue_submits.fetch_add(1u, std::memory_order_relaxed);
  const size_t origin_index = static_cast<size_t>(origin);
  const size_t phase_index = static_cast<size_t>(current_submit_phase());
  if (origin_index < kNumSubmitOrigins && phase_index < kNumSubmitPhases) {
    vulkan_submit_origin_phase_counters()
        .counts[origin_index][phase_index]
        .fetch_add(1u, std::memory_order_relaxed);
  }
  switch (origin) {
    case VulkanSubmitOrigin::NormalCmdSubmitFrequency:
      counters.normal_cmd_submit_frequency.fetch_add(
          1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::StackPlannedRecordingSubmit:
      counters.stack_planned_recording_submit.fetch_add(
          1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::PreStackFlush:
      counters.pre_stack_flush.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::PostStackFlush:
      counters.post_stack_flush.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::ExplicitSynchronize:
      counters.explicit_synchronize.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::TensorCpuReadback:
      counters.tensor_cpu_readback.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::FallbackReadback:
      counters.fallback_readback.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::RetireQueueDrain:
      counters.retire_queue_drain.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::ProfilingTimestampReset:
      counters.profiling_timestamp_reset.fetch_add(
          1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::ProfilingTimestampReadback:
      counters.profiling_timestamp_readback.fetch_add(
          1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::ContextShutdown:
      counters.shutdown.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::DebugValidation:
      counters.debug_validation.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::ConvPrepackUpload:
      counters.conv_prepack_upload.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::Unknown:
    default:
      counters.unknown.fetch_add(1u, std::memory_order_relaxed);
      break;
  }
}

const char* submit_origin_name(const VulkanSubmitOrigin origin) {
  switch (origin) {
    case VulkanSubmitOrigin::NormalCmdSubmitFrequency:
      return "normal_cmd_submit_frequency";
    case VulkanSubmitOrigin::StackPlannedRecordingSubmit:
      return "stack_planned_recording_submit";
    case VulkanSubmitOrigin::PreStackFlush:
      return "pre_stack_flush";
    case VulkanSubmitOrigin::PostStackFlush:
      return "post_stack_flush";
    case VulkanSubmitOrigin::ExplicitSynchronize:
      return "explicit_synchronize";
    case VulkanSubmitOrigin::TensorCpuReadback:
      return "tensor_cpu_readback";
    case VulkanSubmitOrigin::FallbackReadback:
      return "fallback_readback";
    case VulkanSubmitOrigin::RetireQueueDrain:
      return "retire_queue_drain";
    case VulkanSubmitOrigin::ProfilingTimestampReset:
      return "profiling_timestamp_reset";
    case VulkanSubmitOrigin::ProfilingTimestampReadback:
      return "profiling_timestamp_readback";
    case VulkanSubmitOrigin::ContextShutdown:
      return "shutdown";
    case VulkanSubmitOrigin::DebugValidation:
      return "debug_validation";
    case VulkanSubmitOrigin::ConvPrepackUpload:
      return "conv_prepack_upload";
    case VulkanSubmitOrigin::Unknown:
    default:
      return "unknown";
  }
}

const char* submit_phase_name(const VulkanSubmitPhase phase) {
  switch (phase) {
    case VulkanSubmitPhase::ModelSetup:
      return "model_setup";
    case VulkanSubmitPhase::PatchEmbed:
      return "patch_embed";
    case VulkanSubmitPhase::PositionalEmbeddingSetup:
      return "positional_embedding_setup";
    case VulkanSubmitPhase::StackOwner:
      return "stack_owner";
    case VulkanSubmitPhase::StackOwnerNorm:
      return "stack_owner_norm";
    case VulkanSubmitPhase::StackOwnerAttention:
      return "stack_owner_attention";
    case VulkanSubmitPhase::StackOwnerLinear:
      return "stack_owner_linear";
    case VulkanSubmitPhase::StackOwnerResidual:
      return "stack_owner_residual";
    case VulkanSubmitPhase::Decoder:
      return "decoder";
    case VulkanSubmitPhase::DecoderConv:
      return "decoder_conv";
    case VulkanSubmitPhase::DecoderUpsample:
      return "decoder_upsample";
    case VulkanSubmitPhase::DecoderPointwise:
      return "decoder_pointwise";
    case VulkanSubmitPhase::Readback:
      return "readback";
    case VulkanSubmitPhase::ExplicitSynchronize:
      return "explicit_synchronize";
    case VulkanSubmitPhase::Retire:
      return "retire";
    case VulkanSubmitPhase::Profiling:
      return "profiling";
    case VulkanSubmitPhase::Shutdown:
      return "shutdown";
    case VulkanSubmitPhase::TestHarness:
      return "test_harness";
    case VulkanSubmitPhase::Unknown:
    default:
      return "unknown";
  }
}

const char* retire_call_site_name(const VulkanRetireCallSite callsite) {
  switch (callsite) {
    case VulkanRetireCallSite::ContextFlushPending:
      return "context_flush_pending";
    case VulkanRetireCallSite::ContextSubmitFrequency:
      return "context_submit_frequency";
    case VulkanRetireCallSite::ContextExplicitSynchronize:
      return "context_explicit_synchronize";
    case VulkanRetireCallSite::ContextReadback:
      return "context_readback";
    case VulkanRetireCallSite::ContextShutdown:
      return "context_shutdown";
    case VulkanRetireCallSite::StackPlannedRecordingEnd:
      return "stack_planned_recording_end";
    case VulkanRetireCallSite::StackOwnerPhaseBoundary:
      return "stack_owner_phase_boundary";
    case VulkanRetireCallSite::StackOwnerNorm1:
      return "stack_owner_norm1";
    case VulkanRetireCallSite::StackOwnerNorm2:
      return "stack_owner_norm2";
    case VulkanRetireCallSite::StackOwnerAttention:
      return "stack_owner_attention";
    case VulkanRetireCallSite::StackOwnerLinear:
      return "stack_owner_linear";
    case VulkanRetireCallSite::StackOwnerResidual:
      return "stack_owner_residual";
    case VulkanRetireCallSite::NativeLayerNormMetadata:
      return "native_layer_norm_metadata";
    case VulkanRetireCallSite::NativeLayerNormUniform:
      return "native_layer_norm_uniform";
    case VulkanRetireCallSite::AttentionMetadata:
      return "attention_metadata";
    case VulkanRetireCallSite::LinearMetadata:
      return "linear_metadata";
    case VulkanRetireCallSite::ConvMetadata:
      return "conv_metadata";
    case VulkanRetireCallSite::AddResidualMetadata:
      return "add_residual_metadata";
    case VulkanRetireCallSite::DescriptorRecycle:
      return "descriptor_recycle";
    case VulkanRetireCallSite::CommandBufferRecycle:
      return "command_buffer_recycle";
    case VulkanRetireCallSite::StagingBufferRecycle:
      return "staging_buffer_recycle";
    case VulkanRetireCallSite::UniformBufferRecycle:
      return "uniform_buffer_recycle";
    case VulkanRetireCallSite::MetadataBufferRecycle:
      return "metadata_buffer_recycle";
    case VulkanRetireCallSite::BenchmarkReadback:
      return "benchmark_readback";
    case VulkanRetireCallSite::BenchmarkSetup:
      return "benchmark_setup";
    case VulkanRetireCallSite::DebugValidation:
      return "debug_validation";
    case VulkanRetireCallSite::Unknown:
    default:
      return "unknown";
  }
}

const char* retired_resource_kind_name(const VulkanRetiredResourceKind kind) {
  switch (kind) {
    case VulkanRetiredResourceKind::Buffer:
      return "buffer";
    case VulkanRetiredResourceKind::Image:
      return "image";
    case VulkanRetiredResourceKind::UniformBuffer:
      return "uniform_buffer";
    case VulkanRetiredResourceKind::MetadataBuffer:
      return "metadata_buffer";
    case VulkanRetiredResourceKind::DescriptorSet:
      return "descriptor_set";
    case VulkanRetiredResourceKind::DescriptorPool:
      return "descriptor_pool";
    case VulkanRetiredResourceKind::CommandBuffer:
      return "command_buffer";
    case VulkanRetiredResourceKind::StagingBuffer:
      return "staging_buffer";
    case VulkanRetiredResourceKind::QueryBuffer:
      return "query_buffer";
    case VulkanRetiredResourceKind::Other:
      return "other";
    case VulkanRetiredResourceKind::Unknown:
    default:
      return "unknown";
  }
}

const char* retired_resource_role_name(const VulkanRetiredResourceRole role) {
  switch (role) {
    case VulkanRetiredResourceRole::NativeLayerNormUniform:
      return "native_layer_norm_uniform";
    case VulkanRetiredResourceRole::NativeLayerNormMetadata:
      return "native_layer_norm_metadata";
    case VulkanRetiredResourceRole::AttentionMetadata:
      return "attention_metadata";
    case VulkanRetiredResourceRole::LinearMetadata:
      return "linear_metadata";
    case VulkanRetiredResourceRole::ConvMetadata:
      return "conv_metadata";
    case VulkanRetiredResourceRole::ResidualAddMetadata:
      return "residual_add_metadata";
    case VulkanRetiredResourceRole::StackInternalTemp:
      return "stack_internal_temp";
    case VulkanRetiredResourceRole::StackNorm1Output:
      return "stack_norm1_output";
    case VulkanRetiredResourceRole::StackQkvOutput:
      return "stack_qkv_output";
    case VulkanRetiredResourceRole::StackQView:
      return "stack_q_view";
    case VulkanRetiredResourceRole::StackKView:
      return "stack_k_view";
    case VulkanRetiredResourceRole::StackVView:
      return "stack_v_view";
    case VulkanRetiredResourceRole::StackAttentionOutput:
      return "stack_attention_output";
    case VulkanRetiredResourceRole::StackProjOutput:
      return "stack_proj_output";
    case VulkanRetiredResourceRole::StackResidual1Output:
      return "stack_residual1_output";
    case VulkanRetiredResourceRole::StackNorm2Output:
      return "stack_norm2_output";
    case VulkanRetiredResourceRole::StackFc1GeluOutput:
      return "stack_fc1_gelu_output";
    case VulkanRetiredResourceRole::StackFc2Output:
      return "stack_fc2_output";
    case VulkanRetiredResourceRole::StackResidual2Output:
      return "stack_residual2_output";
    case VulkanRetiredResourceRole::StackRequestedOutput:
      return "stack_requested_output";
    case VulkanRetiredResourceRole::StackFinalOutput:
      return "stack_final_output";
    case VulkanRetiredResourceRole::DescriptorRecycle:
      return "descriptor_recycle";
    case VulkanRetiredResourceRole::CommandBufferRecycle:
      return "command_buffer_recycle";
    case VulkanRetiredResourceRole::ReadbackStaging:
      return "readback_staging";
    case VulkanRetiredResourceRole::SetupStaging:
      return "setup_staging";
    case VulkanRetiredResourceRole::Unknown:
    default:
      return "unknown";
  }
}

const char* stack_temp_lifetime_safety_name(
    const VulkanStackTempLifetimeSafety safety) {
  switch (safety) {
    case VulkanStackTempLifetimeSafety::SafeToDeferUntilStackSubmit:
      return "safe_to_defer_until_stack_submit";
    case VulkanStackTempLifetimeSafety::SafeToDeferUntilStackScopeEnd:
      return "safe_to_defer_until_stack_scope_end";
    case VulkanStackTempLifetimeSafety::MustRetireAtPhaseBoundary:
      return "must_retire_at_phase_boundary";
    case VulkanStackTempLifetimeSafety::EscapesAsRequestedIntermediate:
      return "escapes_as_requested_intermediate";
    case VulkanStackTempLifetimeSafety::EscapesAsFinalOutput:
      return "escapes_as_final_output";
    case VulkanStackTempLifetimeSafety::AliasesRuntimeInput:
      return "aliases_runtime_input";
    case VulkanStackTempLifetimeSafety::AliasesRuntimeOutput:
      return "aliases_runtime_output";
    case VulkanStackTempLifetimeSafety::UnsafeUnknownConsumer:
      return "unsafe_unknown_consumer";
    case VulkanStackTempLifetimeSafety::Unknown:
    default:
      return "unknown";
  }
}

const char* stack_retire_provenance_source_name(
    const VulkanStackRetireProvenanceSource source) {
  switch (source) {
    case VulkanStackRetireProvenanceSource::TensorAllocation:
      return "tensor_allocation";
    case VulkanStackRetireProvenanceSource::StorageReallocation:
      return "storage_reallocation";
    case VulkanStackRetireProvenanceSource::ProgramScratchArenaBackingStorage:
      return "program_scratch_arena_backing_storage";
    case VulkanStackRetireProvenanceSource::Unknown:
    default:
      return "unknown";
  }
}

bool is_stack_temp_retired_resource_role(
    const VulkanRetiredResourceRole role) {
  return is_stack_temp_role(role);
}

VulkanStackTempLifetimeSafety stack_retire_lifetime_safety_for_resource(
    const VulkanRetiredResourceRole role,
    const VulkanStackRetireProvenance& provenance) {
  return classify_stack_temp_lifetime_safety(role, provenance);
}

const char* stack_retire_drain_blocker_reason(
    const VulkanRetiredResourceKind kind,
    const VulkanRetiredResourceRole role,
    const VulkanStackRetireProvenance& provenance,
    const bool qkv_would_batch) {
  return stack_drain_blocker_reason(kind, role, provenance, qkv_would_batch);
}

VulkanRetiredResourceRole stack_retired_resource_role_for_phase(
    const VulkanVisionStackPhase phase) {
  switch (phase) {
    case VulkanVisionStackPhase::Norm1:
      return VulkanRetiredResourceRole::StackNorm1Output;
    case VulkanVisionStackPhase::QkvLinear:
      return VulkanRetiredResourceRole::StackQkvOutput;
    case VulkanVisionStackPhase::QkvTransform:
      return VulkanRetiredResourceRole::StackQkvOutput;
    case VulkanVisionStackPhase::Attention:
      return VulkanRetiredResourceRole::StackAttentionOutput;
    case VulkanVisionStackPhase::ProjLinear:
      return VulkanRetiredResourceRole::StackProjOutput;
    case VulkanVisionStackPhase::Residual1:
      return VulkanRetiredResourceRole::StackResidual1Output;
    case VulkanVisionStackPhase::Norm2:
      return VulkanRetiredResourceRole::StackNorm2Output;
    case VulkanVisionStackPhase::Fc1Gelu:
      return VulkanRetiredResourceRole::StackFc1GeluOutput;
    case VulkanVisionStackPhase::Fc2:
      return VulkanRetiredResourceRole::StackFc2Output;
    case VulkanVisionStackPhase::Residual2:
      return VulkanRetiredResourceRole::StackResidual2Output;
    default:
      return VulkanRetiredResourceRole::StackInternalTemp;
  }
}

VulkanStackRetireProvenance current_stack_retire_provenance(
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides,
    const int64_t dtype,
    const bool direct_buffer,
    const bool buffer_storage,
    const bool image_storage,
    const bool alias_or_view,
    const VulkanStackRetireProvenanceSource source) {
  if (!inside_vision_stack_phase()) {
    return {};
  }
  VulkanStackRetireProvenance provenance;
  provenance.defined = true;
  provenance.phase = g_vision_stack_phase;
  provenance.block_index = g_vision_stack_block_index;
  provenance.producer_role =
      stack_retired_resource_role_for_phase(g_vision_stack_phase);
  provenance.source = source;
  provenance.lifetime = alias_or_view
      ? VulkanStackTensorLifetimeClass::AliasOrView
      : VulkanStackTensorLifetimeClass::InternalTemp;
  provenance.shape = shape;
  provenance.strides = strides;
  provenance.dtype = dtype;
  provenance.direct_buffer = direct_buffer;
  provenance.buffer_storage = buffer_storage;
  provenance.image_storage = image_storage;
  provenance.alias_or_view = alias_or_view;
  if (const VulkanStackLastUseProof* proof = find_stack_last_use_proof(
          provenance.phase,
          provenance.block_index,
          provenance.producer_role,
          shape,
          dtype)) {
    provenance.has_last_use_proof = true;
    provenance.expected_consumer_phase = proof->expected_consumer_phase;
    provenance.expected_consumer_block_index =
        proof->expected_consumer_block_index;
    provenance.final_consumer_before_stack_submit =
        proof->final_consumer_before_stack_submit;
    provenance.internal_non_escaping = proof->internal_non_escaping;
    provenance.escapes_stack = proof->escapes_stack;
    provenance.requested_intermediate = proof->requested_intermediate;
    provenance.final_output = proof->final_output;
    provenance.aliases_runtime_input = proof->aliases_runtime_input;
    provenance.aliases_runtime_output = proof->aliases_runtime_output;
    if (proof->escapes_stack || proof->requested_intermediate) {
      provenance.lifetime =
          VulkanStackTensorLifetimeClass::RequestedIntermediateOutput;
    } else if (proof->final_output) {
      provenance.lifetime = VulkanStackTensorLifetimeClass::FinalStackOutput;
    } else if (!proof->internal_non_escaping) {
      provenance.lifetime =
          VulkanStackTensorLifetimeClass::BlockOutputForNextBlock;
    }
  }
  return provenance;
}

VulkanSubmitPhase current_submit_phase() {
  return g_submit_phase;
}

void set_submit_phase(const VulkanSubmitPhase phase) {
  g_submit_phase = phase;
}

void reset_submit_phase() {
  g_submit_phase = VulkanSubmitPhase::Unknown;
}

VulkanRetiredResourceKind current_retired_resource_kind() {
  return g_retired_resource_kind;
}

VulkanRetiredResourceRole current_retired_resource_role() {
  return g_retired_resource_role;
}

std::vector<std::string> submit_origin_phase_snapshot() {
  const auto& counters = vulkan_submit_origin_phase_counters();
  std::vector<std::string> rows;
  for (size_t origin = 0; origin < kNumSubmitOrigins; ++origin) {
    for (size_t phase = 0; phase < kNumSubmitPhases; ++phase) {
      const uint64_t count =
          counters.counts[origin][phase].load(std::memory_order_relaxed);
      if (count == 0u) {
        continue;
      }
      std::ostringstream stream;
      stream << "submit_origin_phase origin="
             << submit_origin_name(static_cast<VulkanSubmitOrigin>(origin))
             << " phase="
             << submit_phase_name(static_cast<VulkanSubmitPhase>(phase))
             << " count=" << count;
      rows.emplace_back(stream.str());
    }
  }
  return rows;
}

std::vector<int64_t> retire_drain_counters_snapshot() {
  const auto& counters = vulkan_retire_drain_counters();
  return {
      static_cast<int64_t>(counters.total.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.queue_submit_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.blocking_wait_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.poll_only_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.pending_resource_count_total.load(
          std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.pending_bytes_total.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.explicit_drain.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.shutdown.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.resource_pressure.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.descriptor_pool_pressure.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.command_buffer_recycle.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.readback_preparation.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.synchronize.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.stack_scope_end.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.decoder_phase.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.setup_phase.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.debug_validation.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.unknown.load(std::memory_order_relaxed)),
  };
}

std::vector<std::string> retire_call_site_counters_snapshot() {
  const auto& counters = retire_call_site_counters();
  std::vector<std::string> rows;
  for (size_t index = 0; index < counters.size(); ++index) {
    const auto& counter = counters[index];
    const uint64_t total = counter.total.load(std::memory_order_relaxed);
    if (total == 0u) {
      continue;
    }
    std::ostringstream stream;
    stream << "retire_call_site callsite="
           << retire_call_site_name(static_cast<VulkanRetireCallSite>(index))
           << " total=" << total << " submit="
           << counter.queue_submit_count.load(std::memory_order_relaxed)
           << " poll="
           << counter.poll_only_count.load(std::memory_order_relaxed)
           << " blocking_wait="
           << counter.blocking_wait_count.load(std::memory_order_relaxed)
           << " pending_resources="
           << counter.pending_resource_count_total.load(
                  std::memory_order_relaxed)
           << " pending_bytes="
           << counter.pending_bytes_total.load(std::memory_order_relaxed);
    rows.emplace_back(stream.str());
  }
  return rows;
}

std::vector<std::string> retired_resource_aggregate_snapshot() {
  std::vector<std::string> rows;
  std::lock_guard<std::mutex> lock(retired_resource_aggregate_mutex());
  for (const auto& entry : retired_resource_aggregate()) {
    const auto& key = entry.first;
    const auto& value = entry.second;
    std::ostringstream stream;
    stream << "retired_resource kind="
           << retired_resource_kind_name(key.kind) << " role="
           << retired_resource_role_name(key.role) << " phase="
           << submit_phase_name(key.phase) << " callsite="
           << retire_call_site_name(key.callsite) << " stack_phase="
           << vision_stack_phase_name(key.stack_phase) << " block="
           << key.block_index << " lifetime="
           << stack_tensor_lifetime_name(key.lifetime) << " shape="
           << format_sizes(key.shape) << " strides="
           << format_sizes(key.strides) << " dtype=" << key.dtype
           << " direct_buffer=" << (key.direct_buffer ? 1 : 0)
           << " buffer_storage=" << (key.buffer_storage ? 1 : 0)
           << " image_storage=" << (key.image_storage ? 1 : 0)
           << " escapes_stack=" << (key.escapes_stack ? 1 : 0)
           << " requested_intermediate="
           << (key.requested_intermediate ? 1 : 0)
           << " final_output=" << (key.final_output ? 1 : 0)
           << " alias_or_view=" << (key.alias_or_view ? 1 : 0)
           << " last_use_proof=" << (key.has_last_use_proof ? 1 : 0)
           << " expected_consumer_phase="
           << vision_stack_phase_name(key.expected_consumer_phase)
           << " expected_consumer_block="
           << key.expected_consumer_block_index
           << " final_consumer_before_stack_submit="
           << (key.final_consumer_before_stack_submit ? 1 : 0)
           << " internal_non_escaping="
           << (key.internal_non_escaping ? 1 : 0)
           << " escapes_stack=" << (key.escapes_stack ? 1 : 0)
           << " requested_intermediate="
           << (key.requested_intermediate ? 1 : 0)
           << " final_output=" << (key.final_output ? 1 : 0)
           << " alias_or_view=" << (key.alias_or_view ? 1 : 0)
           << " aliases_runtime_input="
           << (key.aliases_runtime_input ? 1 : 0)
           << " aliases_runtime_output="
           << (key.aliases_runtime_output ? 1 : 0)
           << " stack_provenance=" << (key.has_stack_provenance ? 1 : 0)
           << " count=" << value.count
           << " bytes=" << value.bytes
           << " queue_submit=" << value.queue_submit_count
           << " blocking_wait=" << value.blocking_wait_count
           << " poll_only=" << value.poll_only_count;
    rows.emplace_back(stream.str());
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

std::vector<std::string> stack_temp_lifetime_safety_snapshot() {
  std::vector<std::string> rows;
  std::lock_guard<std::mutex> lock(stack_temp_lifetime_safety_mutex());
  for (const auto& entry : stack_temp_lifetime_safety_aggregate()) {
    const auto& key = entry.first;
    const auto& value = entry.second;
    std::ostringstream stream;
    stream << "stack_temp_lifetime role="
           << retired_resource_role_name(key.role) << " safety="
           << stack_temp_lifetime_safety_name(key.safety) << " phase="
           << submit_phase_name(key.phase) << " callsite="
           << retire_call_site_name(key.callsite) << " stack_phase="
           << vision_stack_phase_name(key.stack_phase) << " block="
           << key.block_index << " lifetime="
           << stack_tensor_lifetime_name(key.lifetime) << " shape="
           << format_sizes(key.shape) << " dtype=" << key.dtype
           << " last_use_proof=" << (key.has_last_use_proof ? 1 : 0)
           << " expected_consumer_phase="
           << vision_stack_phase_name(key.expected_consumer_phase)
           << " expected_consumer_block="
           << key.expected_consumer_block_index
           << " final_consumer_before_stack_submit="
           << (key.final_consumer_before_stack_submit ? 1 : 0)
           << " internal_non_escaping="
           << (key.internal_non_escaping ? 1 : 0)
           << " stack_provenance=" << (key.has_stack_provenance ? 1 : 0)
           << " count=" << value.count
           << " bytes=" << value.bytes
           << " queue_submit=" << value.queue_submit_count
           << " blocking_wait=" << value.blocking_wait_count
           << " poll_only=" << value.poll_only_count;
    rows.emplace_back(stream.str());
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

std::vector<std::string> stack_scratch_arena_lifetime_snapshot() {
  std::vector<std::string> rows;
  std::lock_guard<std::mutex> lock(stack_scratch_arena_lifetime_mutex());
  for (const auto& entry : stack_scratch_arena_lifetime_aggregate()) {
    const auto& key = entry.first;
    const auto& value = entry.second;
    std::ostringstream stream;
    stream << "stack_scratch_arena_lifetime"
           << " arena_id=" << key.arena_id
           << " generation=" << key.generation
           << " phase=" << submit_phase_name(key.phase)
           << " callsite=" << retire_call_site_name(key.callsite)
           << " first_producer_phase="
           << vision_stack_phase_name(key.first_producer_phase)
           << " first_producer_block=" << key.first_producer_block
           << " last_consumer_phase="
           << vision_stack_phase_name(key.last_consumer_phase)
           << " last_consumer_block=" << key.last_consumer_block
           << " submitted_with_stack_timeline="
           << (key.submitted_with_stack_timeline ? 1 : 0)
           << " escapes_stack=" << (key.escapes_stack ? 1 : 0)
           << " aliases_runtime_input="
           << (key.aliases_runtime_input ? 1 : 0)
           << " aliases_runtime_output="
           << (key.aliases_runtime_output ? 1 : 0)
           << " safe_to_retire_on_stack_submit="
           << (key.safe_to_retire_on_stack_submit ? 1 : 0)
           << " count=" << value.count
           << " bytes=" << value.bytes
           << " queue_submit=" << value.queue_submit_count
           << " blocking_wait=" << value.blocking_wait_count
           << " poll_only=" << value.poll_only_count;
    rows.emplace_back(stream.str());
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

std::vector<int64_t> stack_internal_temp_retire_batch_counters_snapshot() {
  const auto& counters = stack_internal_temp_retire_batch_counters();
  return {
      static_cast<int64_t>(
          counters.total_attempts.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.batch_candidate_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.batch_candidate_bytes.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.batch_accepted_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.batch_accepted_bytes.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.batch_rejected_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.batch_rejected_bytes.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.submitted_batch_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.submitted_batch_bytes.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.rejected_not_target_role.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.rejected_missing_proof.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.rejected_not_internal_non_escaping.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.rejected_consumer_after_submit.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.rejected_requested_intermediate.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.rejected_final_output.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.rejected_alias.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.rejected_runtime_alias.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.rejected_lifetime.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.rejected_not_stack_recording.load(
              std::memory_order_relaxed)),
  };
}

std::vector<std::string> stack_internal_temp_retire_batch_snapshot() {
  std::vector<std::string> rows;
  std::lock_guard<std::mutex> lock(stack_temp_retire_batch_snapshot_mutex());
  for (const auto& entry : stack_temp_retire_batch_decisions()) {
    std::ostringstream stream;
    stream << "stack_internal_temp_retire_batch " << entry.first
           << " count=" << entry.second.count
           << " bytes=" << entry.second.bytes
           << " queue_submit=" << entry.second.queue_submit_count
           << " blocking_wait=" << entry.second.blocking_wait_count
           << " poll_only=" << entry.second.poll_only_count;
    rows.emplace_back(stream.str());
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

std::vector<int64_t> stack_retire_drain_blocker_counters_snapshot() {
  const auto& counters = stack_retire_drain_blocker_counters();
  return {
      static_cast<int64_t>(
          counters.total_drains.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.queue_submit_drains.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.drains_with_old_path_pending.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.drains_with_only_already_batched.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.drains_qkv_would_remove.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.drains_blocked_requested_intermediate.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.drains_blocked_missing_proof.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.drains_blocked_generic_stack_internal_temp.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.drains_blocked_metadata_or_uniform.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.drains_blocked_other_roles.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.old_path_pending_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.old_path_pending_bytes.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.qkv_hypothetical_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.qkv_hypothetical_bytes.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.skipped_no_old_path_pending.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.skipped_no_pending_command_work.load(
              std::memory_order_relaxed)),
  };
}

std::vector<std::string> stack_retire_drain_blocker_snapshot() {
  std::vector<std::string> rows;
  std::lock_guard<std::mutex> lock(stack_retire_drain_blocker_snapshot_mutex());
  for (const auto& entry : stack_retire_drain_blockers()) {
    std::ostringstream stream;
    stream << "stack_retire_drain_blocker " << entry.first
           << " count=" << entry.second.count
           << " bytes=" << entry.second.bytes
           << " queue_submit=" << entry.second.queue_submit_count
           << " blocking_wait=" << entry.second.blocking_wait_count
           << " poll_only=" << entry.second.poll_only_count;
    rows.emplace_back(stream.str());
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

std::vector<std::string> region_lifetime_submit_attribution_snapshot() {
  std::vector<std::string> rows;
  std::lock_guard<std::mutex> lock(region_lifetime_submit_attribution_mutex());
  for (const auto& entry : region_lifetime_submit_attribution_rows()) {
    std::ostringstream stream;
    stream << "region_lifetime_submit_attribution " << entry.first
           << " count=" << entry.second.count
           << " bytes=" << entry.second.bytes
           << " queue_submit=" << entry.second.queue_submit_count
           << " blocking_wait=" << entry.second.blocking_wait_count
           << " poll_only=" << entry.second.poll_only_count;
    rows.emplace_back(stream.str());
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

std::vector<int64_t> stack_subresource_lifetime_dry_run_counters_snapshot() {
  const auto& counters = stack_subresource_lifetime_dry_run_counters();
  return {
      static_cast<int64_t>(
          counters.total_groups.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.queue_submit_groups.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.groups_with_old_path_pending.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.all_safe_group_eligible.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.would_remove_submit_drains.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.actual_removed_submit_drains.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.peak_extra_live_bytes_estimate.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.skipped_no_old_path_pending.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.proven_stack_activation_count.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.missing_stack_activation_proof_count.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_subresource_count.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_score_probability_subresource_count.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.layernorm_stat_buffer_count.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.layernorm_internal_stat_buffer_count.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.metadata_uniform_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.raw_no_provenance_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.stack_internal_raw_missing_generation_count.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.stack_internal_raw_generation_range_count.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.truly_unknown_raw_resource_count.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.host_visible_or_requested_output_count.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.allocator_or_scratch_backing_count.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.proven_stack_activation_bytes.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.missing_stack_activation_proof_bytes.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_subresource_bytes.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_score_probability_subresource_bytes.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.layernorm_stat_buffer_bytes.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.layernorm_internal_stat_buffer_bytes.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.metadata_uniform_bytes.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.raw_no_provenance_bytes.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.stack_internal_raw_missing_generation_bytes.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.stack_internal_raw_generation_range_bytes.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.truly_unknown_raw_resource_bytes.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.host_visible_or_requested_output_bytes.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.allocator_or_scratch_backing_bytes.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.rejected_unsafe_resource_class.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.rejected_over_block_budget.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.rejected_over_scope_budget.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.rejected_large_backing.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_buffer_generation_range_missing_stack_proof_count
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_raw_generation_range_missing_stack_proof_count
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_provenance_missing_last_use_count.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_unknown_subresource_count.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_buffer_generation_range_missing_stack_proof_bytes
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_raw_generation_range_missing_stack_proof_bytes.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_provenance_missing_last_use_bytes.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_unknown_subresource_bytes.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters
              .attention_score_probability_range_missing_alias_escape_proof_count
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_raw_auxiliary_range_missing_alias_escape_proof_count
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters
              .attention_score_probability_range_missing_alias_escape_proof_bytes
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_raw_auxiliary_range_missing_alias_escape_proof_bytes
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters
              .attention_score_probability_range_non_escape_last_consumer_count
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_raw_auxiliary_range_non_escape_last_consumer_count
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters
              .attention_score_probability_range_non_escape_last_consumer_bytes
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_raw_auxiliary_range_non_escape_last_consumer_bytes
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters
              .stack_internal_temp_raw_generation_range_missing_last_consumer_count
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters
              .stack_qkv_output_raw_generation_range_non_escape_last_consumer_count
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters
              .stack_proj_output_raw_generation_range_non_escape_last_consumer_count
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters
              .stack_residual1_output_raw_generation_range_non_escape_last_consumer_count
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters
              .stack_internal_temp_raw_generation_range_missing_last_consumer_bytes
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters
              .stack_qkv_output_raw_generation_range_non_escape_last_consumer_bytes
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters
              .stack_proj_output_raw_generation_range_non_escape_last_consumer_bytes
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters
              .stack_residual1_output_raw_generation_range_non_escape_last_consumer_bytes
              .load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.phase_boundary_total_groups.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.phase_boundary_all_safe_group_eligible.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.phase_boundary_would_remove_explicit_synchronizes.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.phase_boundary_actual_removed_explicit_synchronizes.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.phase_boundary_rejected_unsafe_resource_class.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.phase_boundary_rejected_over_block_budget.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.phase_boundary_rejected_over_scope_budget.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.phase_boundary_rejected_large_backing.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.phase_boundary_stack_activation_carry_proof_count.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.phase_boundary_stack_activation_carry_proof_bytes.load(
              std::memory_order_relaxed)),
  };
}

std::vector<std::string> stack_subresource_lifetime_dry_run_snapshot() {
  std::vector<std::string> rows;
  std::lock_guard<std::mutex> lock(stack_subresource_lifetime_dry_run_mutex());
  for (const auto& entry : stack_subresource_lifetime_dry_run_rows()) {
    std::ostringstream stream;
    stream << "stack_subresource_lifetime_dry_run " << entry.first
           << " count=" << entry.second.count
           << " bytes=" << entry.second.bytes
           << " queue_submit=" << entry.second.queue_submit_count
           << " blocking_wait=" << entry.second.blocking_wait_count
           << " poll_only=" << entry.second.poll_only_count;
    rows.emplace_back(stream.str());
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

void note_vulkan_retire_drain(
    VulkanRetireDrainReason reason,
    VulkanRetireCallSite callsite,
    const bool queue_submit,
    const bool blocking_wait,
    const uint64_t pending_resource_count,
    const uint64_t pending_bytes) {
  auto& counters = vulkan_retire_drain_counters();
  counters.total.fetch_add(1u, std::memory_order_relaxed);
  if (queue_submit) {
    counters.queue_submit_count.fetch_add(1u, std::memory_order_relaxed);
  } else {
    counters.poll_only_count.fetch_add(1u, std::memory_order_relaxed);
  }
  if (blocking_wait) {
    counters.blocking_wait_count.fetch_add(1u, std::memory_order_relaxed);
  }
  counters.pending_resource_count_total.fetch_add(
      pending_resource_count, std::memory_order_relaxed);
  counters.pending_bytes_total.fetch_add(
      pending_bytes, std::memory_order_relaxed);
  const size_t callsite_index = static_cast<size_t>(callsite);
  if (callsite_index < retire_call_site_counters().size()) {
    auto& counter = retire_call_site_counters()[callsite_index];
    counter.total.fetch_add(1u, std::memory_order_relaxed);
    if (queue_submit) {
      counter.queue_submit_count.fetch_add(1u, std::memory_order_relaxed);
    } else {
      counter.poll_only_count.fetch_add(1u, std::memory_order_relaxed);
    }
    if (blocking_wait) {
      counter.blocking_wait_count.fetch_add(1u, std::memory_order_relaxed);
    }
    counter.pending_resource_count_total.fetch_add(
        pending_resource_count, std::memory_order_relaxed);
    counter.pending_bytes_total.fetch_add(
        pending_bytes, std::memory_order_relaxed);
  }
  switch (reason) {
    case VulkanRetireDrainReason::ExplicitDrain:
      counters.explicit_drain.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::Shutdown:
      counters.shutdown.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::ResourcePressure:
      counters.resource_pressure.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::DescriptorPoolPressure:
      counters.descriptor_pool_pressure.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::CommandBufferRecycle:
      counters.command_buffer_recycle.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::ReadbackPreparation:
      counters.readback_preparation.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::Synchronize:
      counters.synchronize.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::StackScopeEnd:
      counters.stack_scope_end.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::DecoderPhase:
      counters.decoder_phase.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::SetupPhase:
      counters.setup_phase.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::DebugValidation:
      counters.debug_validation.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::Unknown:
    default:
      counters.unknown.fetch_add(1u, std::memory_order_relaxed);
      break;
  }
}

bool is_safe_stack_temp_retire_batch_candidate(
    const VulkanStackRetireProvenance& provenance) {
  return std::string(stack_temp_retire_batch_reject_reason(provenance)) ==
      "accepted";
}

bool is_qkv_stack_temp_retire_batch_candidate(
    const VulkanStackRetireProvenance& provenance) {
  return provenance.producer_role == VulkanRetiredResourceRole::StackQkvOutput &&
      has_proven_internal_stack_temp_lifetime(provenance);
}

void note_stack_internal_temp_retire_batch_decision(
    const VulkanStackRetireProvenance& provenance,
    const uint64_t bytes,
    const bool stack_recording_active,
    const bool accepted) {
  if (!provenance.defined || !is_stack_temp_role(provenance.producer_role)) {
    return;
  }

  auto& counters = stack_internal_temp_retire_batch_counters();
  counters.total_attempts.fetch_add(1u, std::memory_order_relaxed);

  const char* reason = stack_temp_retire_batch_reject_reason(provenance);
  bool candidate = std::string(reason) == "accepted";
  if (candidate) {
    counters.batch_candidate_count.fetch_add(1u, std::memory_order_relaxed);
    counters.batch_candidate_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }

  if (accepted) {
    counters.batch_accepted_count.fetch_add(1u, std::memory_order_relaxed);
    counters.batch_accepted_bytes.fetch_add(bytes, std::memory_order_relaxed);
    reason = "accepted";
  } else {
    counters.batch_rejected_count.fetch_add(1u, std::memory_order_relaxed);
    counters.batch_rejected_bytes.fetch_add(bytes, std::memory_order_relaxed);
    if (candidate && !stack_recording_active) {
      reason = "not_stack_recording";
    }
    const std::string reason_string(reason);
    if (reason_string == "not_target_role") {
      counters.rejected_not_target_role.fetch_add(
          1u, std::memory_order_relaxed);
    } else if (reason_string == "missing_proof") {
      counters.rejected_missing_proof.fetch_add(1u, std::memory_order_relaxed);
    } else if (reason_string == "not_internal_non_escaping") {
      counters.rejected_not_internal_non_escaping.fetch_add(
          1u, std::memory_order_relaxed);
    } else if (reason_string == "consumer_after_submit") {
      counters.rejected_consumer_after_submit.fetch_add(
          1u, std::memory_order_relaxed);
    } else if (reason_string == "requested_intermediate") {
      counters.rejected_requested_intermediate.fetch_add(
          1u, std::memory_order_relaxed);
    } else if (reason_string == "final_output") {
      counters.rejected_final_output.fetch_add(1u, std::memory_order_relaxed);
    } else if (reason_string == "alias") {
      counters.rejected_alias.fetch_add(1u, std::memory_order_relaxed);
    } else if (reason_string == "runtime_alias") {
      counters.rejected_runtime_alias.fetch_add(1u, std::memory_order_relaxed);
    } else if (reason_string == "lifetime") {
      counters.rejected_lifetime.fetch_add(1u, std::memory_order_relaxed);
    } else if (reason_string == "not_stack_recording") {
      counters.rejected_not_stack_recording.fetch_add(
          1u, std::memory_order_relaxed);
    }
  }

  std::ostringstream key;
  key << "role=" << retired_resource_role_name(provenance.producer_role)
      << " decision=" << (accepted ? "accepted" : "rejected")
      << " reason=" << reason
      << " phase=" << vision_stack_phase_name(provenance.phase)
      << " block=" << provenance.block_index
      << " expected_consumer_phase="
      << vision_stack_phase_name(provenance.expected_consumer_phase)
      << " expected_consumer_block="
      << provenance.expected_consumer_block_index
      << " shape=" << format_sizes(provenance.shape)
      << " dtype=" << provenance.dtype
      << " last_use_proof=" << (provenance.has_last_use_proof ? 1 : 0)
      << " internal_non_escaping="
      << (provenance.internal_non_escaping ? 1 : 0)
      << " requested_intermediate="
      << (provenance.requested_intermediate ? 1 : 0)
      << " final_output=" << (provenance.final_output ? 1 : 0)
      << " alias_or_view=" << (provenance.alias_or_view ? 1 : 0)
      << " runtime_alias="
      << ((provenance.aliases_runtime_input ||
           provenance.aliases_runtime_output)
              ? 1
              : 0);
  std::lock_guard<std::mutex> lock(stack_temp_retire_batch_snapshot_mutex());
  auto& value = stack_temp_retire_batch_decisions()[key.str()];
  value.count += 1u;
  value.bytes += bytes;
}

void note_stack_internal_temp_retire_batch_submitted(const uint64_t bytes) {
  auto& counters = stack_internal_temp_retire_batch_counters();
  counters.submitted_batch_count.fetch_add(1u, std::memory_order_relaxed);
  counters.submitted_batch_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void note_stack_retire_drain_blocker_resource(
    const VulkanRetiredResourceKind kind,
    const VulkanRetiredResourceRole role,
    const VulkanSubmitPhase phase,
    const VulkanRetireCallSite callsite,
    const uint64_t bytes,
    const bool qkv_would_batch,
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof,
    const std::string& allocation_label) {
  const char* reason =
      stack_drain_blocker_reason(kind, role, provenance, qkv_would_batch);
  const VulkanStackTempLifetimeSafety safety =
      classify_stack_temp_lifetime_safety(role, provenance);
  const char* const resource_class =
      stack_subresource_lifetime_dry_run_resource_class(
          kind, role, provenance, qkv_would_batch, allocation_proof);
  const bool base_safe_candidate =
      stack_subresource_lifetime_dry_run_resource_is_safe(resource_class);
  const bool formal_last_use_proof =
      stack_subresource_lifetime_dry_run_has_formal_stack_owner_last_use_proof(
          kind,
          role,
          resource_class,
          provenance,
          allocation_proof,
          allocation_label,
          callsite);
  const bool safe_candidate = base_safe_candidate || formal_last_use_proof;
  const bool large_backing =
      stack_subresource_lifetime_dry_run_is_large_backing(
          role, bytes, provenance);
  const char* const missing_proof_reason =
      stack_region_lifetime_missing_proof_reason(
          kind,
          role,
          resource_class,
          provenance,
          allocation_proof,
          safe_candidate,
          large_backing,
          formal_last_use_proof);
  std::ostringstream key;
  key << "role=" << retired_resource_role_name(role)
      << " reason=" << reason
      << " safety=" << stack_temp_lifetime_safety_name(safety)
      << " resource_class=" << resource_class
      << " safe_candidate=" << (safe_candidate ? 1 : 0)
      << " formal_last_use_proof=" << (formal_last_use_proof ? 1 : 0)
      << " large_backing=" << (large_backing ? 1 : 0)
      << " missing_proof_reason=" << missing_proof_reason
      << " phase=" << submit_phase_name(phase)
      << " callsite=" << retire_call_site_name(callsite)
      << " stack_phase=" << vision_stack_phase_name(provenance.phase)
      << " block=" << provenance.block_index
      << " shape=" << format_sizes(provenance.shape)
      << " dtype=" << provenance.dtype
      << " qkv_would_batch=" << (qkv_would_batch ? 1 : 0)
      << " last_use_proof="
      << (provenance.has_last_use_proof ? 1 : 0)
      << " requested_intermediate="
      << (provenance.requested_intermediate ? 1 : 0)
      << " final_output=" << (provenance.final_output ? 1 : 0)
      << " alias_or_view=" << (provenance.alias_or_view ? 1 : 0)
      << " stack_provenance=" << (provenance.defined ? 1 : 0)
      << " provenance_source="
      << stack_retire_provenance_source_name(provenance.source)
      << " provenance_source_id=" << provenance.source_identity
      << " provenance_source_generation=" << provenance.source_generation
      << " allocation_id=" << allocation_proof.allocation_id
      << " allocation_generation="
      << allocation_proof.allocation_generation
      << " allocation_has_generation="
      << (allocation_proof.has_generation ? 1 : 0)
      << " allocation_byte_offset=" << allocation_proof.byte_offset
      << " allocation_byte_range=" << allocation_proof.byte_range
      << " allocation_has_byte_range="
      << (allocation_proof.has_byte_range ? 1 : 0)
      << " provenance_loss_reason="
      << stack_provenance_loss_reason(role, provenance);
  std::lock_guard<std::mutex> lock(stack_retire_drain_blocker_snapshot_mutex());
  auto& value = stack_retire_drain_blockers()[key.str()];
  value.count += 1u;
  value.bytes += bytes;
}

void note_stack_retire_drain_blocker_summary(
    const VulkanSubmitPhase phase,
    const VulkanRetireCallSite callsite,
    const bool queue_submit,
    const uint64_t old_path_pending_count,
    const uint64_t old_path_pending_bytes,
    const uint64_t qkv_hypothetical_count,
    const uint64_t qkv_hypothetical_bytes,
    const bool qkv_would_remove_drain,
    const bool only_already_batched,
    const bool blocked_requested_intermediate,
    const bool blocked_missing_proof,
    const bool blocked_generic_stack_internal_temp,
    const bool blocked_metadata_or_uniform,
    const bool blocked_other_roles,
    const bool skipped_no_old_path_pending,
    const bool skipped_no_pending_command_work) {
  auto& counters = stack_retire_drain_blocker_counters();
  counters.total_drains.fetch_add(1u, std::memory_order_relaxed);
  if (queue_submit) {
    counters.queue_submit_drains.fetch_add(1u, std::memory_order_relaxed);
  }
  if (old_path_pending_count > 0u) {
    counters.drains_with_old_path_pending.fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (only_already_batched) {
    counters.drains_with_only_already_batched.fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (qkv_would_remove_drain) {
    counters.drains_qkv_would_remove.fetch_add(1u, std::memory_order_relaxed);
  }
  if (blocked_requested_intermediate) {
    counters.drains_blocked_requested_intermediate.fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (blocked_missing_proof) {
    counters.drains_blocked_missing_proof.fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (blocked_generic_stack_internal_temp) {
    counters.drains_blocked_generic_stack_internal_temp.fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (blocked_metadata_or_uniform) {
    counters.drains_blocked_metadata_or_uniform.fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (blocked_other_roles) {
    counters.drains_blocked_other_roles.fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (skipped_no_old_path_pending) {
    counters.skipped_no_old_path_pending.fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (skipped_no_pending_command_work) {
    counters.skipped_no_pending_command_work.fetch_add(
        1u, std::memory_order_relaxed);
  }
  counters.old_path_pending_count.fetch_add(
      old_path_pending_count, std::memory_order_relaxed);
  counters.old_path_pending_bytes.fetch_add(
      old_path_pending_bytes, std::memory_order_relaxed);
  counters.qkv_hypothetical_count.fetch_add(
      qkv_hypothetical_count, std::memory_order_relaxed);
  counters.qkv_hypothetical_bytes.fetch_add(
      qkv_hypothetical_bytes, std::memory_order_relaxed);

  std::ostringstream key;
  key << "summary=1 phase=" << submit_phase_name(phase)
      << " callsite=" << retire_call_site_name(callsite)
      << " queue_submit=" << (queue_submit ? 1 : 0)
      << " old_path_pending=" << old_path_pending_count
      << " qkv_hypothetical=" << qkv_hypothetical_count
      << " qkv_would_remove_drain="
      << (qkv_would_remove_drain ? 1 : 0)
      << " only_already_batched=" << (only_already_batched ? 1 : 0)
      << " blocked_requested_intermediate="
      << (blocked_requested_intermediate ? 1 : 0)
      << " blocked_missing_proof=" << (blocked_missing_proof ? 1 : 0)
      << " blocked_generic_stack_internal_temp="
      << (blocked_generic_stack_internal_temp ? 1 : 0)
      << " blocked_metadata_or_uniform="
      << (blocked_metadata_or_uniform ? 1 : 0)
      << " blocked_other_roles=" << (blocked_other_roles ? 1 : 0)
      << " skipped_no_old_path_pending="
      << (skipped_no_old_path_pending ? 1 : 0)
      << " skipped_no_pending_command_work="
      << (skipped_no_pending_command_work ? 1 : 0);
  std::lock_guard<std::mutex> lock(stack_retire_drain_blocker_snapshot_mutex());
  auto& value = stack_retire_drain_blockers()[key.str()];
  value.count += 1u;
  value.bytes += old_path_pending_bytes;
  if (queue_submit) {
    value.queue_submit_count += 1u;
  } else {
    value.poll_only_count += 1u;
  }
}

void note_stack_retire_drain_copresent_group(
    const VulkanSubmitPhase phase,
    const VulkanRetireCallSite callsite,
    const bool queue_submit,
    const uint64_t old_path_pending_count,
    const uint64_t old_path_pending_bytes,
    const uint64_t qkv_hypothetical_count,
    const bool qkv_would_remove_drain,
    const bool skipped_no_old_path_pending,
    const std::string& signature,
    const std::string& blockers) {
  std::ostringstream key;
  key << "copresent_group=1 phase=" << submit_phase_name(phase)
      << " callsite=" << retire_call_site_name(callsite)
      << " queue_submit=" << (queue_submit ? 1 : 0)
      << " old_path_pending=" << old_path_pending_count
      << " qkv_hypothetical=" << qkv_hypothetical_count
      << " qkv_would_remove_drain="
      << (qkv_would_remove_drain ? 1 : 0)
      << " skipped_no_old_path_pending="
      << (skipped_no_old_path_pending ? 1 : 0)
      << " blockers=" << blockers
      << " signature=" << signature;
  std::lock_guard<std::mutex> lock(stack_retire_drain_blocker_snapshot_mutex());
  auto& value = stack_retire_drain_blockers()[key.str()];
  value.count += 1u;
  value.bytes += old_path_pending_bytes;
  if (queue_submit) {
    value.queue_submit_count += 1u;
  } else {
    value.poll_only_count += 1u;
  }
}

void note_region_lifetime_submit_attribution_group(
    const VulkanSubmitOrigin origin,
    const VulkanSubmitPhase phase,
    const VulkanRetireCallSite callsite,
    const bool queue_submit,
    const bool had_pending_work,
    const uint64_t pending_resource_count,
    const uint64_t pending_bytes,
    const std::string& signature,
    const std::string& blockers) {
  std::ostringstream key;
  key << "group=1 origin=" << submit_origin_name(origin)
      << " phase=" << submit_phase_name(phase)
      << " callsite=" << retire_call_site_name(callsite)
      << " queue_submit=" << (queue_submit ? 1 : 0)
      << " had_pending_work=" << (had_pending_work ? 1 : 0)
      << " pending_resources=" << pending_resource_count
      << " blockers=" << blockers
      << " signature=" << signature;
  std::lock_guard<std::mutex> lock(region_lifetime_submit_attribution_mutex());
  auto& value = region_lifetime_submit_attribution_rows()[key.str()];
  value.count += 1u;
  value.bytes += pending_bytes;
  if (queue_submit) {
    value.queue_submit_count += 1u;
  } else {
    value.poll_only_count += 1u;
  }
}

void note_region_lifetime_submit_attribution_resource(
    const VulkanSubmitOrigin origin,
    const VulkanSubmitPhase phase,
    const VulkanRetireCallSite callsite,
    const VulkanRetiredResourceKind kind,
    const VulkanRetiredResourceRole role,
    const uint64_t bytes,
    const char* const reason,
    const VulkanStackTempLifetimeSafety safety,
    const bool queue_submit,
    const bool had_pending_work,
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof,
    const std::string& allocation_label) {
  const bool qkv_would_batch =
      is_qkv_stack_temp_retire_batch_candidate(provenance);
  const char* const resource_class =
      stack_subresource_lifetime_dry_run_resource_class(
          kind, role, provenance, qkv_would_batch, allocation_proof);
  const bool base_safe_candidate =
      stack_subresource_lifetime_dry_run_resource_is_safe(resource_class);
  const bool formal_last_use_proof =
      stack_subresource_lifetime_dry_run_has_formal_stack_owner_last_use_proof(
          kind,
          role,
          resource_class,
          provenance,
          allocation_proof,
          allocation_label,
          callsite);
  const bool safe_candidate = base_safe_candidate || formal_last_use_proof;
  const bool large_backing =
      stack_subresource_lifetime_dry_run_is_large_backing(
          role, bytes, provenance);
  const char* const missing_proof_reason =
      stack_region_lifetime_missing_proof_reason(
          kind,
          role,
          resource_class,
          provenance,
          allocation_proof,
          safe_candidate,
          large_backing,
          formal_last_use_proof);
  const bool capture_or_public_output =
      provenance.escapes_stack || provenance.requested_intermediate ||
      provenance.final_output ||
      role == VulkanRetiredResourceRole::StackRequestedOutput ||
      role == VulkanRetiredResourceRole::StackFinalOutput;
  std::ostringstream key;
  key << "resource=1 origin=" << submit_origin_name(origin)
      << " phase=" << submit_phase_name(phase)
      << " callsite=" << retire_call_site_name(callsite)
      << " kind=" << retired_resource_kind_name(kind)
      << " role=" << retired_resource_role_name(role)
      << " reason=" << (reason ? reason : "unknown")
      << " safety=" << stack_temp_lifetime_safety_name(safety)
      << " resource_class=" << resource_class
      << " safe_candidate=" << (safe_candidate ? 1 : 0)
      << " formal_last_use_proof=" << (formal_last_use_proof ? 1 : 0)
      << " large_backing=" << (large_backing ? 1 : 0)
      << " missing_proof_reason=" << missing_proof_reason
      << " producer_substep="
      << stack_region_lifetime_producer_substep(
             kind, role, resource_class, provenance, allocation_label)
      << " last_use_candidate="
      << stack_region_lifetime_last_use_candidate(
             kind, role, resource_class, provenance, allocation_label)
      << " capture_or_public_output="
      << (capture_or_public_output ? 1 : 0)
      << " qkv_would_batch=" << (qkv_would_batch ? 1 : 0)
      << " queue_submit=" << (queue_submit ? 1 : 0)
      << " had_pending_work=" << (had_pending_work ? 1 : 0)
      << " stack_phase=" << vision_stack_phase_name(provenance.phase)
      << " block=" << provenance.block_index
      << " lifetime=" << stack_tensor_lifetime_name(provenance.lifetime)
      << " shape=" << format_sizes(provenance.shape)
      << " dtype=" << provenance.dtype
      << " stack_provenance=" << (provenance.defined ? 1 : 0)
      << " last_use_proof=" << (provenance.has_last_use_proof ? 1 : 0)
      << " expected_consumer_phase="
      << vision_stack_phase_name(provenance.expected_consumer_phase)
      << " expected_consumer_block="
      << provenance.expected_consumer_block_index
      << " final_consumer_before_stack_submit="
      << (provenance.final_consumer_before_stack_submit ? 1 : 0)
      << " internal_non_escaping="
      << (provenance.internal_non_escaping ? 1 : 0)
      << " requested_intermediate="
      << (provenance.requested_intermediate ? 1 : 0)
      << " final_output=" << (provenance.final_output ? 1 : 0)
      << " alias_or_view=" << (provenance.alias_or_view ? 1 : 0)
      << " aliases_runtime_input="
      << (provenance.aliases_runtime_input ? 1 : 0)
      << " aliases_runtime_output="
      << (provenance.aliases_runtime_output ? 1 : 0)
      << " provenance_source="
      << stack_retire_provenance_source_name(provenance.source)
      << " provenance_label="
      << stack_retire_provenance_source_name(provenance.source)
      << " allocation_label="
      << (allocation_label.empty() ? "unknown" : allocation_label)
      << " allocation_id=" << allocation_proof.allocation_id
      << " allocation_has_generation="
      << (allocation_proof.has_generation ? 1 : 0)
      << " allocation_generation="
      << allocation_proof.allocation_generation
      << " allocation_has_byte_range="
      << (allocation_proof.has_byte_range ? 1 : 0)
      << " allocation_byte_offset=" << allocation_proof.byte_offset
      << " allocation_byte_range=" << allocation_proof.byte_range
      << " allocation_allocated_bytes=" << allocation_proof.allocated_bytes
      << " diagnostic_stack_scope=" << (inside_vision_stack_phase() ? 1 : 0)
      << " diagnostic_stack_phase="
      << vision_stack_phase_name(current_vision_stack_phase())
      << " diagnostic_stack_block=" << current_vision_stack_block_index()
      << " provenance_loss_reason="
      << stack_provenance_loss_reason(role, provenance);
  std::lock_guard<std::mutex> lock(region_lifetime_submit_attribution_mutex());
  auto& value = region_lifetime_submit_attribution_rows()[key.str()];
  value.count += 1u;
  value.bytes += bytes;
  if (queue_submit) {
    value.queue_submit_count += 1u;
  } else {
    value.poll_only_count += 1u;
  }
}

const char* stack_subresource_lifetime_dry_run_resource_class(
    const VulkanRetiredResourceKind kind,
    const VulkanRetiredResourceRole role,
    const VulkanStackRetireProvenance& provenance,
    const bool qkv_would_batch,
    const VulkanStackRawResourceAllocationProof& allocation_proof) {
  if (
      role == VulkanRetiredResourceRole::StackRequestedOutput ||
      role == VulkanRetiredResourceRole::StackFinalOutput ||
      provenance.requested_intermediate || provenance.escapes_stack ||
      provenance.final_output || provenance.aliases_runtime_input ||
      provenance.aliases_runtime_output) {
    return kDryRunHostVisibleOrRequestedOutput;
  }
  if (is_metadata_or_uniform_resource(kind, role)) {
    return kDryRunMetadataUniform;
  }
  if (
      provenance.defined &&
      provenance.source ==
          VulkanStackRetireProvenanceSource::
              ProgramScratchArenaBackingStorage) {
    return kDryRunAllocatorOrScratchBacking;
  }
  if (is_attention_score_probability_subresource(role, provenance)) {
    return kDryRunAttentionScoreProbabilitySubresource;
  }
  if (is_layernorm_internal_stat_buffer(role, provenance)) {
    return kDryRunLayerNormInternalStatBuffer;
  }
  if (is_layernorm_stat_resource(role, provenance)) {
    return kDryRunLayerNormStatBuffer;
  }
  if (is_attention_subresource_role(role)) {
    return has_proven_internal_stack_temp_lifetime(provenance)
        ? kDryRunProvenStackActivation
        : classify_unproven_attention_subresource(
              kind, role, provenance, allocation_proof);
  }
  if (qkv_would_batch || has_proven_internal_stack_temp_lifetime(provenance)) {
    return kDryRunProvenStackActivation;
  }
  if (!provenance.defined) {
    if (role == VulkanRetiredResourceRole::SetupStaging) {
      return kDryRunNonStackSetupStagingPending;
    }
    if (is_stack_temp_role(role)) {
      if (allocation_proof.has_generation && allocation_proof.has_byte_range) {
        return classify_stack_internal_raw_generation_range(
            role, allocation_proof);
      }
      return kDryRunStackInternalRawMissingGeneration;
    }
    if (kind == VulkanRetiredResourceKind::Buffer) {
      return kDryRunUnscopedRawBufferNoStackProof;
    }
    if (
        role == VulkanRetiredResourceRole::Unknown &&
        kind == VulkanRetiredResourceKind::Unknown) {
      return kDryRunTrulyUnknownRawResource;
    }
    return kDryRunRawNoProvenance;
  }
  if (
      stack_activation_phase_boundary_carry_candidate(role, provenance) &&
      vision_stack_capture_dependency_active() &&
      vision_stack_capture_dependency_reaches_block(
          provenance.expected_consumer_block_index)) {
    return kDryRunCaptureSensitiveStackActivation;
  }
  if (is_stack_temp_role(role)) {
    return kDryRunMissingStackActivationProof;
  }
  return kDryRunRawNoProvenance;
}

bool stack_subresource_lifetime_dry_run_resource_is_safe(
    const char* const resource_class) {
  const std::string key(resource_class);
  return key == kDryRunProvenStackActivation ||
      key == kDryRunAttentionScoreProbabilitySubresource ||
      key == kDryRunLayerNormInternalStatBuffer ||
      key == kDryRunMetadataUniform;
}

bool stack_subresource_lifetime_dry_run_has_formal_norm2_last_use_proof(
    const VulkanRetiredResourceKind kind,
    const VulkanRetiredResourceRole role,
    const char* const resource_class,
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof,
    const std::string& allocation_label) {
  return stack_subresource_lifetime_dry_run_has_formal_stack_owner_last_use_proof(
      kind,
      role,
      resource_class,
      provenance,
      allocation_proof,
      allocation_label,
      VulkanRetireCallSite::StackOwnerNorm2);
}

bool stack_subresource_lifetime_dry_run_has_formal_stack_owner_last_use_proof(
    const VulkanRetiredResourceKind kind,
    const VulkanRetiredResourceRole role,
    const char* const resource_class,
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof,
    const std::string& allocation_label,
    const VulkanRetireCallSite callsite) {
  if (
      current_submit_phase() != VulkanSubmitPhase::StackOwner ||
      current_vision_stack_block_index() < 0 ||
      g_stack_last_use_proofs.empty()) {
    return false;
  }
  const bool norm2_retire_group =
      current_vision_stack_phase() == VulkanVisionStackPhase::Norm2 ||
      callsite == VulkanRetireCallSite::StackOwnerNorm2;
  const bool phase_boundary_explicit_sync =
      callsite == VulkanRetireCallSite::StackOwnerPhaseBoundary ||
      callsite == VulkanRetireCallSite::StackOwnerNorm1 ||
      callsite == VulkanRetireCallSite::StackOwnerNorm2;
  if (!norm2_retire_group && !phase_boundary_explicit_sync) {
    return false;
  }
  if (
      !allocation_proof.has_generation || !allocation_proof.has_byte_range ||
      allocation_proof.byte_range == 0u || allocation_label.empty()) {
    return false;
  }
  if (
      role == VulkanRetiredResourceRole::StackRequestedOutput ||
      role == VulkanRetiredResourceRole::StackFinalOutput ||
      provenance.escapes_stack || provenance.requested_intermediate ||
      provenance.final_output || provenance.alias_or_view ||
      provenance.aliases_runtime_input || provenance.aliases_runtime_output) {
    return false;
  }

  const std::string key(resource_class ? resource_class : "");
  if (
      phase_boundary_explicit_sync && provenance.defined &&
      provenance.has_last_use_proof && provenance.internal_non_escaping &&
      provenance.final_consumer_before_stack_submit &&
      stack_phase_has_reached_consumer(
          current_vision_stack_phase(),
          current_vision_stack_block_index(),
          provenance.expected_consumer_phase,
          provenance.expected_consumer_block_index) &&
      role == provenance.producer_role && is_stack_temp_role(role)) {
    return true;
  }
  if (
      phase_boundary_explicit_sync &&
      stack_activation_phase_boundary_carry_proof(role, provenance)) {
    return true;
  }
  if (
      phase_boundary_explicit_sync &&
      role == VulkanRetiredResourceRole::StackResidual2Output &&
      provenance.defined && provenance.has_last_use_proof &&
      provenance.lifetime ==
          VulkanStackTensorLifetimeClass::BlockOutputForNextBlock &&
      stack_phase_has_reached_consumer(
          current_vision_stack_phase(),
          current_vision_stack_block_index(),
          provenance.expected_consumer_phase,
          provenance.expected_consumer_block_index) &&
      provenance.expected_consumer_phase == VulkanVisionStackPhase::Norm1 &&
      provenance.final_consumer_before_stack_submit &&
      !provenance.escapes_stack && !provenance.requested_intermediate &&
      !provenance.final_output && !provenance.alias_or_view &&
      !provenance.aliases_runtime_input &&
      !provenance.aliases_runtime_output && provenance.direct_buffer &&
      provenance.buffer_storage && !provenance.image_storage) {
    return true;
  }
  if (
      phase_boundary_explicit_sync && !provenance.defined &&
      is_stack_temp_role(role)) {
    const int64_t producer_block =
        stack_raw_block_index_for_label(allocation_label);
    const VulkanVisionStackPhase consumer_phase =
        stack_raw_last_consumer_phase_for_label(role, allocation_label);
    if (
        producer_block >= 0 &&
        consumer_phase != VulkanVisionStackPhase::Unknown &&
        stack_phase_has_reached_consumer(
            current_vision_stack_phase(),
            current_vision_stack_block_index(),
            consumer_phase,
            producer_block)) {
      const std::string key(resource_class ? resource_class : "");
      if (
          key == kDryRunStackInternalRawGenerationRange ||
          key == kDryRunStackInternalTempRawGenerationRangeMissingLastConsumer ||
          is_stack_raw_generation_range_non_escape_last_consumer_class(
              resource_class)) {
        return true;
      }
    }
  }
  if (!norm2_retire_group) {
    return false;
  }
  if (
      key == kDryRunAttentionRawAuxiliaryRangeNonEscapeLastConsumer &&
      role == VulkanRetiredResourceRole::StackAttentionOutput &&
      kind == VulkanRetiredResourceKind::Unknown) {
    return true;
  }
  if (
      key == kDryRunStackResidual1OutputRawGenerationRangeNonEscapeLastConsumer &&
      role == VulkanRetiredResourceRole::StackResidual1Output) {
    return true;
  }
  if (
      key == kDryRunStackProjOutputRawGenerationRangeNonEscapeLastConsumer &&
      role == VulkanRetiredResourceRole::StackProjOutput) {
    return true;
  }
  if (
      key == kDryRunStackQkvOutputRawGenerationRangeNonEscapeLastConsumer &&
      role == VulkanRetiredResourceRole::StackQkvOutput) {
    return true;
  }
  if (
      key == kDryRunStackInternalTempRawGenerationRangeMissingLastConsumer &&
      role == VulkanRetiredResourceRole::StackInternalTemp) {
    const bool qkv_linear_auxiliary = allocation_label.size() >= 4u &&
        allocation_label.compare(allocation_label.size() - 4u, 4u, ".qkv") ==
            0;
    if (
        allocation_label == "transform_bias_rescale_qkv" ||
        allocation_label == "attention_merge_heads" || qkv_linear_auxiliary) {
      return true;
    }
  }
  return false;
}

const char* stack_raw_producer_substep_for_label(
    const std::string& allocation_label) {
  if (allocation_label == "transform_bias_rescale_qkv") {
    return "qkv_transform";
  }
  if (allocation_label == "attention_merge_heads") {
    return "attention_merge_heads";
  }
  if (
      allocation_label.size() >= 4u &&
      allocation_label.compare(allocation_label.size() - 4u, 4u, ".qkv") ==
          0) {
    return "qkv_linear";
  }
  if (
      allocation_label.size() >= 5u &&
      allocation_label.compare(allocation_label.size() - 5u, 5u, ".proj") ==
          0) {
    return "proj_linear";
  }
  if (
      allocation_label.size() >= 4u &&
      allocation_label.compare(allocation_label.size() - 4u, 4u, ".fc1") ==
          0) {
    return "fc1_gelu";
  }
  if (
      allocation_label.size() >= 4u &&
      allocation_label.compare(allocation_label.size() - 4u, 4u, ".fc2") ==
          0) {
    return "fc2";
  }
  return "unknown";
}

const char* stack_raw_last_consumer_for_label(
    const std::string& allocation_label) {
  if (
      allocation_label == "transform_bias_rescale_qkv" ||
      (allocation_label.size() >= 4u &&
       allocation_label.compare(allocation_label.size() - 4u, 4u, ".qkv") ==
           0)) {
    return "qkv_transform";
  }
  if (allocation_label == "attention_merge_heads") {
    return "attention";
  }
  if (
      allocation_label.size() >= 5u &&
      allocation_label.compare(allocation_label.size() - 5u, 5u, ".proj") ==
          0) {
    return "residual1";
  }
  if (
      allocation_label.size() >= 4u &&
      allocation_label.compare(allocation_label.size() - 4u, 4u, ".fc1") ==
          0) {
    return "fc2";
  }
  if (
      allocation_label.size() >= 4u &&
      allocation_label.compare(allocation_label.size() - 4u, 4u, ".fc2") ==
          0) {
    return "residual2";
  }
  return "unknown";
}

int64_t stack_raw_block_index_for_label(const std::string& allocation_label) {
  const std::string marker = ".block";
  const size_t marker_pos = allocation_label.find(marker);
  if (marker_pos == std::string::npos) {
    return -1;
  }
  size_t digit_pos = marker_pos + marker.size();
  if (
      digit_pos >= allocation_label.size() ||
      !std::isdigit(static_cast<unsigned char>(allocation_label[digit_pos]))) {
    return -1;
  }
  int64_t block_index = 0;
  while (
      digit_pos < allocation_label.size() &&
      std::isdigit(static_cast<unsigned char>(allocation_label[digit_pos]))) {
    block_index =
        block_index * 10 + (allocation_label[digit_pos] - static_cast<char>('0'));
    ++digit_pos;
  }
  return block_index;
}

VulkanVisionStackPhase stack_raw_last_consumer_phase_for_label(
    const VulkanRetiredResourceRole role,
    const std::string& allocation_label) {
  if (
      role == VulkanRetiredResourceRole::StackQkvOutput ||
      (role == VulkanRetiredResourceRole::StackInternalTemp &&
       allocation_label.size() >= 4u &&
       allocation_label.compare(allocation_label.size() - 4u, 4u, ".qkv") ==
           0)) {
    return VulkanVisionStackPhase::QkvTransform;
  }
  if (role == VulkanRetiredResourceRole::StackProjOutput) {
    return VulkanVisionStackPhase::Residual1;
  }
  if (role == VulkanRetiredResourceRole::StackFc1GeluOutput) {
    return VulkanVisionStackPhase::Fc2;
  }
  if (role == VulkanRetiredResourceRole::StackFc2Output) {
    return VulkanVisionStackPhase::Residual2;
  }
  return VulkanVisionStackPhase::Unknown;
}

bool stack_subresource_lifetime_dry_run_is_formal_norm2_last_use_label(
    const char* const resource_class,
    const std::string& allocation_label) {
  const std::string key(resource_class ? resource_class : "");
  if (key != kDryRunStackInternalTempRawGenerationRangeMissingLastConsumer) {
    return false;
  }
  return std::string(stack_raw_producer_substep_for_label(allocation_label)) !=
      "unknown";
}

bool stack_subresource_lifetime_dry_run_is_large_backing(
    const VulkanRetiredResourceRole role,
    const uint64_t bytes,
    const VulkanStackRetireProvenance& provenance) {
  if (
      provenance.defined &&
      provenance.source ==
          VulkanStackRetireProvenanceSource::
              ProgramScratchArenaBackingStorage) {
    return true;
  }
  return (
      role == VulkanRetiredResourceRole::StackQkvOutput ||
      role == VulkanRetiredResourceRole::StackProjOutput) &&
      bytes > kStackSubresourceLifetimeDryRunBlockBudgetBytes;
}

void note_stack_subresource_lifetime_dry_run_resource(
    const VulkanRetiredResourceKind kind,
    const VulkanRetiredResourceRole role,
    const VulkanSubmitPhase phase,
    const VulkanRetireCallSite callsite,
    const uint64_t bytes,
    const char* const resource_class,
    const bool safe_candidate,
    const bool large_backing,
    const bool formal_last_use_proof,
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof,
    const std::string& allocation_label) {
  auto& counters = stack_subresource_lifetime_dry_run_counters();
  note_dry_run_resource_class(counters, resource_class, bytes);
  const bool has_attention_non_escape_last_consumer_evidence =
      is_attention_non_escape_last_consumer_class(resource_class);
  const bool has_raw_generation_range_evidence =
      is_stack_raw_generation_range_evidence_class(resource_class);
  const bool has_raw_non_escape_last_consumer_evidence =
      is_stack_raw_generation_range_non_escape_last_consumer_class(
          resource_class);
  const bool has_stack_internal_label_last_consumer_evidence =
      stack_subresource_lifetime_dry_run_is_formal_norm2_last_use_label(
          resource_class, allocation_label);
  const char* const raw_last_consumer =
      has_stack_internal_label_last_consumer_evidence
      ? stack_raw_last_consumer_for_label(allocation_label)
      : stack_raw_last_consumer_for_dry_run(role, resource_class);
  const bool has_phase_boundary_stack_activation_carry_proof =
      formal_last_use_proof &&
      (callsite == VulkanRetireCallSite::StackOwnerPhaseBoundary ||
       callsite == VulkanRetireCallSite::StackOwnerNorm1 ||
       callsite == VulkanRetireCallSite::StackOwnerNorm2) &&
      stack_activation_phase_boundary_carry_proof(role, provenance);
  if (has_phase_boundary_stack_activation_carry_proof) {
    counters.phase_boundary_stack_activation_carry_proof_count.fetch_add(
        1u, std::memory_order_relaxed);
    counters.phase_boundary_stack_activation_carry_proof_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
  }

  std::ostringstream key;
  key << "resource=1 class=" << resource_class
      << " safe_candidate=" << (safe_candidate ? 1 : 0)
      << " large_backing=" << (large_backing ? 1 : 0)
      << " formal_last_use_proof=" << (formal_last_use_proof ? 1 : 0)
      << " kind=" << retired_resource_kind_name(kind)
      << " role=" << retired_resource_role_name(role)
      << " phase=" << submit_phase_name(phase)
      << " callsite=" << retire_call_site_name(callsite)
      << " stack_phase=" << vision_stack_phase_name(provenance.phase)
      << " block=" << provenance.block_index
      << " lifetime=" << stack_tensor_lifetime_name(provenance.lifetime)
      << " shape=" << format_sizes(provenance.shape)
      << " dtype=" << provenance.dtype
      << " stack_provenance=" << (provenance.defined ? 1 : 0)
      << " allocation_id=" << allocation_proof.allocation_id
      << " allocation_label="
      << (allocation_label.empty() ? "unknown" : allocation_label)
      << " allocation_generation="
      << allocation_proof.allocation_generation
      << " allocation_has_generation="
      << (allocation_proof.has_generation ? 1 : 0)
      << " allocation_byte_offset=" << allocation_proof.byte_offset
      << " allocation_byte_range=" << allocation_proof.byte_range
      << " allocation_has_byte_range="
      << (allocation_proof.has_byte_range ? 1 : 0)
      << " allocation_allocated_bytes=" << allocation_proof.allocated_bytes
      << " last_use_proof=" << (provenance.has_last_use_proof ? 1 : 0)
      << " expected_consumer_phase="
      << vision_stack_phase_name(provenance.expected_consumer_phase)
      << " expected_consumer_block="
      << provenance.expected_consumer_block_index
      << " final_consumer_before_stack_submit="
      << (provenance.final_consumer_before_stack_submit ? 1 : 0)
      << " internal_non_escaping="
      << (provenance.internal_non_escaping ? 1 : 0)
      << " phase_boundary_stack_activation_carry_proof="
      << (has_phase_boundary_stack_activation_carry_proof ? 1 : 0)
      << " requested_intermediate="
      << (provenance.requested_intermediate ? 1 : 0)
      << " final_output=" << (provenance.final_output ? 1 : 0)
      << " alias_or_view=" << (provenance.alias_or_view ? 1 : 0)
      << " diagnostic_stack_scope=" << (inside_vision_stack_phase() ? 1 : 0)
      << " diagnostic_stack_phase="
      << vision_stack_phase_name(current_vision_stack_phase())
      << " diagnostic_stack_block=" << current_vision_stack_block_index()
      << " attention_substep="
      << attention_substep_for_dry_run(kind, resource_class)
      << " attention_producer="
      << attention_producer_for_dry_run(resource_class)
      << " attention_last_consumer="
      << attention_last_consumer_for_dry_run(resource_class)
      << " attention_host_visibility_proof="
      << (provenance.defined
              ? "stack_provenance"
              : (has_attention_non_escape_last_consumer_evidence
                     ? "stack_scope_non_output_range"
                     : "missing"))
      << " attention_alias_escape_proof="
      << (provenance.defined
              ? "stack_provenance"
              : (has_attention_non_escape_last_consumer_evidence
                     ? "stack_scope_non_escape_evidence"
                     : "missing"))
      << " attention_last_use_proof="
      << (provenance.has_last_use_proof
              ? "present"
              : (has_attention_non_escape_last_consumer_evidence
                     ? "diagnostic_last_consumer"
                     : "missing"))
      << " attention_non_escape_evidence="
      << (has_attention_non_escape_last_consumer_evidence
              ? "stack_scope_allocation_range"
              : "missing")
      << " attention_last_consumer_evidence="
      << (has_attention_non_escape_last_consumer_evidence
              ? attention_last_consumer_for_dry_run(resource_class)
              : "missing")
      << " attention_retire_policy_eligible=0"
      << " raw_stack_scope_evidence="
      << (has_raw_generation_range_evidence ? "stack_scope_allocation_range"
                                            : "missing")
      << " raw_alias_escape_proof="
      << (has_raw_generation_range_evidence ? "stack_scope_no_runtime_alias"
                                            : "missing")
      << " raw_non_escape_evidence="
      << (has_raw_generation_range_evidence ? "stack_scope_allocation_range"
                                            : "missing")
      << " raw_last_consumer_evidence="
      << raw_last_consumer
      << " raw_producer_substep="
      << stack_raw_producer_substep_for_label(allocation_label)
      << " raw_last_use_proof="
      << (formal_last_use_proof
              ? "formal_last_consumer"
              : (has_raw_non_escape_last_consumer_evidence
              ? "diagnostic_last_consumer"
              : (has_raw_generation_range_evidence ? "missing" : "not_raw")))
      << " raw_retire_policy_eligible=0"
      << " provenance_source="
      << stack_retire_provenance_source_name(provenance.source)
      << " provenance_loss_reason="
      << stack_provenance_loss_reason(role, provenance);
  std::lock_guard<std::mutex> lock(stack_subresource_lifetime_dry_run_mutex());
  auto& value = stack_subresource_lifetime_dry_run_rows()[key.str()];
  value.count += 1u;
  value.bytes += bytes;
}

void note_stack_subresource_lifetime_dry_run_group(
    const VulkanSubmitPhase phase,
    const VulkanRetireCallSite callsite,
    const bool queue_submit,
    const uint64_t old_path_pending_count,
    const uint64_t old_path_pending_bytes,
    const uint64_t safe_candidate_count,
    const uint64_t safe_candidate_bytes,
    const bool all_safe_group_eligible,
    const bool would_remove_submit_drain,
    const bool actual_removed_submit_drain,
    const std::string& budget_reject,
    const std::string& signature,
    const std::string& blockers) {
  auto& counters = stack_subresource_lifetime_dry_run_counters();
  counters.total_groups.fetch_add(1u, std::memory_order_relaxed);
  if (queue_submit) {
    counters.queue_submit_groups.fetch_add(1u, std::memory_order_relaxed);
  }
  if (old_path_pending_count > 0u) {
    counters.groups_with_old_path_pending.fetch_add(
        1u, std::memory_order_relaxed);
  } else {
    counters.skipped_no_old_path_pending.fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (all_safe_group_eligible) {
    counters.all_safe_group_eligible.fetch_add(1u, std::memory_order_relaxed);
  }
  if (would_remove_submit_drain) {
    counters.would_remove_submit_drains.fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (actual_removed_submit_drain) {
    counters.actual_removed_submit_drains.fetch_add(
        1u, std::memory_order_relaxed);
  }
  update_peak_atomic(
      counters.peak_extra_live_bytes_estimate, safe_candidate_bytes);

  if (budget_reject == "unsafe_resource_class") {
    counters.rejected_unsafe_resource_class.fetch_add(
        1u, std::memory_order_relaxed);
  } else if (budget_reject == "over_block_budget") {
    counters.rejected_over_block_budget.fetch_add(
        1u, std::memory_order_relaxed);
  } else if (budget_reject == "over_scope_budget") {
    counters.rejected_over_scope_budget.fetch_add(
        1u, std::memory_order_relaxed);
  } else if (budget_reject == "large_backing_excluded") {
    counters.rejected_large_backing.fetch_add(1u, std::memory_order_relaxed);
  }

  std::ostringstream key;
  key << "group=1 phase=" << submit_phase_name(phase)
      << " callsite=" << retire_call_site_name(callsite)
      << " queue_submit=" << (queue_submit ? 1 : 0)
      << " old_path_pending=" << old_path_pending_count
      << " safe_candidate_count=" << safe_candidate_count
      << " safe_candidate_bytes=" << safe_candidate_bytes
      << " all_safe_group_eligible=" << (all_safe_group_eligible ? 1 : 0)
      << " would_remove_submit_drain="
      << (would_remove_submit_drain ? 1 : 0)
      << " actual_removed_submit_drain="
      << (actual_removed_submit_drain ? 1 : 0)
      << " peak_extra_live_bytes_estimate=" << safe_candidate_bytes
      << " block_budget_bytes="
      << kStackSubresourceLifetimeDryRunBlockBudgetBytes
      << " scope_budget_bytes="
      << kStackSubresourceLifetimeDryRunScopeBudgetBytes
      << " budget_reject=" << budget_reject
      << " blockers=" << blockers
      << " signature=" << signature;
  std::lock_guard<std::mutex> lock(stack_subresource_lifetime_dry_run_mutex());
  auto& value = stack_subresource_lifetime_dry_run_rows()[key.str()];
  value.count += 1u;
  value.bytes += old_path_pending_bytes;
  if (queue_submit) {
    value.queue_submit_count += 1u;
  } else {
    value.poll_only_count += 1u;
  }
}

void note_stack_phase_boundary_lifetime_dry_run_group(
    const VulkanSubmitPhase phase,
    const VulkanRetireCallSite callsite,
    const bool queue_submit,
    const uint64_t old_path_pending_count,
    const uint64_t old_path_pending_bytes,
    const uint64_t safe_candidate_count,
    const uint64_t safe_candidate_bytes,
    const bool all_safe_group_eligible,
    const bool would_remove_explicit_synchronize,
    const bool actual_removed_explicit_synchronize,
    const uint64_t block_budget_bytes,
    const uint64_t scope_budget_bytes,
    const std::string& budget_reject,
    const std::string& signature,
    const std::string& blockers) {
  auto& counters = stack_subresource_lifetime_dry_run_counters();
  counters.phase_boundary_total_groups.fetch_add(
      1u, std::memory_order_relaxed);
  if (all_safe_group_eligible) {
    counters.phase_boundary_all_safe_group_eligible.fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (would_remove_explicit_synchronize) {
    counters.phase_boundary_would_remove_explicit_synchronizes.fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (actual_removed_explicit_synchronize) {
    counters.phase_boundary_actual_removed_explicit_synchronizes.fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (budget_reject == "unsafe_resource_class") {
    counters.phase_boundary_rejected_unsafe_resource_class.fetch_add(
        1u, std::memory_order_relaxed);
  } else if (budget_reject == "over_block_budget") {
    counters.phase_boundary_rejected_over_block_budget.fetch_add(
        1u, std::memory_order_relaxed);
  } else if (budget_reject == "over_scope_budget") {
    counters.phase_boundary_rejected_over_scope_budget.fetch_add(
        1u, std::memory_order_relaxed);
  } else if (budget_reject == "large_backing_excluded") {
    counters.phase_boundary_rejected_large_backing.fetch_add(
        1u, std::memory_order_relaxed);
  }
  update_peak_atomic(
      counters.peak_extra_live_bytes_estimate, safe_candidate_bytes);

  std::ostringstream key;
  key << "phase_boundary_group=1 phase=" << submit_phase_name(phase)
      << " callsite=" << retire_call_site_name(callsite)
      << " boundary_stack_phase="
      << vision_stack_phase_name(current_vision_stack_phase())
      << " boundary_block=" << current_vision_stack_block_index()
      << " queue_submit=" << (queue_submit ? 1 : 0)
      << " old_path_pending=" << old_path_pending_count
      << " safe_candidate_count=" << safe_candidate_count
      << " safe_candidate_bytes=" << safe_candidate_bytes
      << " all_safe_group_eligible=" << (all_safe_group_eligible ? 1 : 0)
      << " would_remove_explicit_synchronize="
      << (would_remove_explicit_synchronize ? 1 : 0)
      << " actual_removed_phase_boundary_sync="
      << (actual_removed_explicit_synchronize ? 1 : 0)
      << " peak_extra_live_bytes_estimate=" << safe_candidate_bytes
      << " block_budget_bytes=" << block_budget_bytes
      << " scope_budget_bytes=" << scope_budget_bytes
      << " budget_reject=" << budget_reject
      << " blockers=" << blockers
      << " signature=" << signature;
  std::lock_guard<std::mutex> lock(stack_subresource_lifetime_dry_run_mutex());
  auto& value = stack_subresource_lifetime_dry_run_rows()[key.str()];
  value.count += 1u;
  value.bytes += old_path_pending_bytes;
  if (queue_submit) {
    value.queue_submit_count += 1u;
  } else {
    value.poll_only_count += 1u;
  }
}

void note_vulkan_retired_resource(
    VulkanRetiredResourceKind kind,
    VulkanRetiredResourceRole role,
    VulkanSubmitPhase phase,
    VulkanRetireCallSite callsite,
    const uint64_t bytes,
    const bool queue_submit,
    const bool blocking_wait,
    const bool poll_only,
    const VulkanStackRetireProvenance& provenance) {
  RetiredResourceAggregateKey key;
  key.kind = kind;
  key.role = role;
  key.phase = phase;
  key.callsite = callsite;
  if (provenance.defined) {
    key.stack_phase = provenance.phase;
    key.block_index = provenance.block_index;
    key.lifetime = provenance.lifetime;
    key.shape = provenance.shape;
    key.strides = provenance.strides;
    key.dtype = provenance.dtype;
    key.direct_buffer = provenance.direct_buffer;
    key.buffer_storage = provenance.buffer_storage;
    key.image_storage = provenance.image_storage;
    key.escapes_stack = provenance.escapes_stack;
    key.requested_intermediate = provenance.requested_intermediate;
    key.final_output = provenance.final_output;
    key.alias_or_view = provenance.alias_or_view;
    key.has_last_use_proof = provenance.has_last_use_proof;
    key.expected_consumer_phase = provenance.expected_consumer_phase;
    key.expected_consumer_block_index =
        provenance.expected_consumer_block_index;
    key.final_consumer_before_stack_submit =
        provenance.final_consumer_before_stack_submit;
    key.internal_non_escaping = provenance.internal_non_escaping;
    key.has_stack_provenance = true;
  }
  std::lock_guard<std::mutex> lock(retired_resource_aggregate_mutex());
  auto& value = retired_resource_aggregate()[key];
  value.count += 1u;
  value.bytes += bytes;
  if (queue_submit) {
    value.queue_submit_count += 1u;
  }
  if (blocking_wait) {
    value.blocking_wait_count += 1u;
  }
  if (poll_only) {
    value.poll_only_count += 1u;
  }
  if (
      provenance.defined &&
      provenance.source ==
          VulkanStackRetireProvenanceSource::
              ProgramScratchArenaBackingStorage) {
    StackScratchArenaLifetimeKey arena_key;
    arena_key.arena_id = provenance.source_identity;
    arena_key.generation = provenance.source_generation;
    arena_key.phase = phase;
    arena_key.callsite = callsite;
    arena_key.first_producer_phase = provenance.phase;
    arena_key.first_producer_block = provenance.block_index;
    arena_key.last_consumer_phase = VulkanVisionStackPhase::Unknown;
    arena_key.last_consumer_block = -1;
    arena_key.submitted_with_stack_timeline =
        queue_submit &&
        callsite == VulkanRetireCallSite::StackPlannedRecordingEnd;
    arena_key.escapes_stack = provenance.escapes_stack ||
        provenance.requested_intermediate || provenance.final_output;
    arena_key.aliases_runtime_input = provenance.aliases_runtime_input;
    arena_key.aliases_runtime_output = provenance.aliases_runtime_output;
    arena_key.safe_to_retire_on_stack_submit = false;
    std::lock_guard<std::mutex> arena_lock(
        stack_scratch_arena_lifetime_mutex());
    auto& arena_value = stack_scratch_arena_lifetime_aggregate()[arena_key];
    arena_value.count += 1u;
    arena_value.bytes += bytes;
    if (queue_submit) {
      arena_value.queue_submit_count += 1u;
    }
    if (blocking_wait) {
      arena_value.blocking_wait_count += 1u;
    }
    if (poll_only) {
      arena_value.poll_only_count += 1u;
    }
  }
  if (is_stack_temp_role(role)) {
    const VulkanStackTempLifetimeSafety safety =
        classify_stack_temp_lifetime_safety(role, provenance);
    StackTempLifetimeSafetyKey safety_key;
    safety_key.role = role;
    safety_key.safety = safety;
    safety_key.phase = phase;
    safety_key.callsite = callsite;
    if (provenance.defined) {
      safety_key.stack_phase = provenance.phase;
      safety_key.block_index = provenance.block_index;
      safety_key.lifetime = provenance.lifetime;
      safety_key.shape = provenance.shape;
      safety_key.dtype = provenance.dtype;
      safety_key.has_last_use_proof = provenance.has_last_use_proof;
      safety_key.expected_consumer_phase = provenance.expected_consumer_phase;
      safety_key.expected_consumer_block_index =
          provenance.expected_consumer_block_index;
      safety_key.final_consumer_before_stack_submit =
          provenance.final_consumer_before_stack_submit;
      safety_key.internal_non_escaping = provenance.internal_non_escaping;
      safety_key.escapes_stack = provenance.escapes_stack;
      safety_key.requested_intermediate = provenance.requested_intermediate;
      safety_key.final_output = provenance.final_output;
      safety_key.alias_or_view = provenance.alias_or_view;
      safety_key.aliases_runtime_input = provenance.aliases_runtime_input;
      safety_key.aliases_runtime_output = provenance.aliases_runtime_output;
      safety_key.has_stack_provenance = true;
    }
    std::lock_guard<std::mutex> safety_lock(
        stack_temp_lifetime_safety_mutex());
    auto& safety_value = stack_temp_lifetime_safety_aggregate()[safety_key];
    safety_value.count += 1u;
    safety_value.bytes += bytes;
    if (queue_submit) {
      safety_value.queue_submit_count += 1u;
    }
    if (blocking_wait) {
      safety_value.blocking_wait_count += 1u;
    }
    if (poll_only) {
      safety_value.poll_only_count += 1u;
    }
  }
}

const char* vision_stack_phase_name(const VulkanVisionStackPhase phase) {
  switch (phase) {
    case VulkanVisionStackPhase::Unknown:
      return "unknown";
    case VulkanVisionStackPhase::StackEntry:
      return "stack_entry";
    case VulkanVisionStackPhase::BlockEntry:
      return "block_entry";
    case VulkanVisionStackPhase::Norm1:
      return "norm1";
    case VulkanVisionStackPhase::QkvLinear:
      return "qkv_linear";
    case VulkanVisionStackPhase::QkvTransform:
      return "qkv_transform";
    case VulkanVisionStackPhase::Attention:
      return "attention";
    case VulkanVisionStackPhase::ProjLinear:
      return "proj_linear";
    case VulkanVisionStackPhase::Residual1:
      return "residual1";
    case VulkanVisionStackPhase::Norm2:
      return "norm2";
    case VulkanVisionStackPhase::Fc1Gelu:
      return "fc1_gelu";
    case VulkanVisionStackPhase::Fc2:
      return "fc2";
    case VulkanVisionStackPhase::Residual2:
      return "residual2";
    case VulkanVisionStackPhase::IntermediateCapture:
      return "intermediate_capture";
    case VulkanVisionStackPhase::StackExit:
      return "stack_exit";
  }
  return "unknown";
}

const char* stack_tensor_lifetime_name(
    const VulkanStackTensorLifetimeClass lifetime) {
  switch (lifetime) {
    case VulkanStackTensorLifetimeClass::Unknown:
      return "unknown";
    case VulkanStackTensorLifetimeClass::InternalTemp:
      return "internal_temp";
    case VulkanStackTensorLifetimeClass::BlockOutputForNextBlock:
      return "block_output_for_next_block";
    case VulkanStackTensorLifetimeClass::RequestedIntermediateOutput:
      return "requested_intermediate_output";
    case VulkanStackTensorLifetimeClass::FinalStackOutput:
      return "final_stack_output";
    case VulkanStackTensorLifetimeClass::AliasOrView:
      return "alias_or_view";
  }
  return "unknown";
}

VulkanVisionStackPhase current_vision_stack_phase() {
  return g_vision_stack_phase;
}

int64_t current_vision_stack_block_index() {
  return g_vision_stack_block_index;
}

bool inside_vision_stack_phase() {
  return g_vision_stack_phase != VulkanVisionStackPhase::Unknown;
}

bool vision_stack_capture_dependency_active() {
  return !g_vision_stack_capture_indices.empty();
}

bool vision_stack_capture_dependency_reaches_block(
    const int64_t block_index) {
  if (block_index < 0) {
    return true;
  }
  for (const int64_t capture_index : g_vision_stack_capture_indices) {
    if (capture_index >= block_index) {
      return true;
    }
  }
  return false;
}

bool vision_stack_capture_dependency_contains_block(const int64_t block_index) {
  if (block_index < 0) {
    return false;
  }
  for (const int64_t capture_index : g_vision_stack_capture_indices) {
    if (capture_index == block_index) {
      return true;
    }
  }
  return false;
}

void begin_stack_dispatch_dependency_recording_scope() {
  g_stack_dispatch_dependency_scope_id =
      g_next_stack_dispatch_dependency_scope_id.fetch_add(
          1u, std::memory_order_relaxed);
  g_stack_dispatch_dependency_position = 0u;
}

void end_stack_dispatch_dependency_recording_scope() {
  maybe_write_stack_region_dependency_graph_dump();
  g_stack_dispatch_dependency_scope_id = 0u;
  g_stack_dispatch_dependency_position = 0u;
}

void note_vulkan_stack_dispatch(const char* shader_name) {
  if (!inside_vision_stack_phase()) {
    return;
  }
  const uint64_t scope_id = g_stack_dispatch_dependency_scope_id;
  const uint64_t position =
      scope_id == 0u ? 0u : ++g_stack_dispatch_dependency_position;
  std::ostringstream key;
  key << "stack_dispatch"
      << " phase=" << vision_stack_phase_name(g_vision_stack_phase)
      << " block=" << g_vision_stack_block_index
      << " shader=" << (shader_name && shader_name[0] ? shader_name : "unknown")
      << " role=" << vision_stack_phase_name(g_vision_stack_phase);
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  stack_dispatch_aggregate()[key.str()] += 1u;
  if (scope_id != 0u) {
    auto& value = stack_dispatch_dependency_dispatch_rows()
        [stack_dispatch_dependency_dispatch_key(
            scope_id,
            g_vision_stack_phase,
            g_vision_stack_block_index,
            shader_name)];
    value.count += 1u;
    if (value.first_position == 0u) {
      value.first_position = position;
    }
    value.last_position = position;
  }
}

void note_stack_owner_dispatch_dependency_dry_run(
    const VulkanRetiredResourceKind kind,
    const VulkanRetiredResourceRole role,
    const VulkanSubmitPhase phase,
    const VulkanRetireCallSite callsite,
    const bool queue_submit,
    const uint64_t bytes,
    const char* const resource_class,
    const bool formal_last_use_proof,
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof,
    const std::string& allocation_label) {
  if (
      phase != VulkanSubmitPhase::StackOwner ||
      !(callsite == VulkanRetireCallSite::StackOwnerPhaseBoundary ||
        callsite == VulkanRetireCallSite::StackOwnerNorm1 ||
        callsite == VulkanRetireCallSite::StackOwnerNorm2)) {
    return;
  }
  const bool residual2_candidate =
      kind == VulkanRetiredResourceKind::Buffer &&
      role == VulkanRetiredResourceRole::StackResidual2Output &&
      provenance.defined &&
      provenance.phase == VulkanVisionStackPhase::Residual2 &&
      provenance.expected_consumer_phase != VulkanVisionStackPhase::Unknown &&
      provenance.expected_consumer_block_index >= 0;
  if (!residual2_candidate) {
    return;
  }
  const uint64_t scope_id = g_stack_dispatch_dependency_scope_id;
  const StackDispatchDependencyDispatchValue* const producer_dispatch =
      find_stack_dispatch_observation(
          scope_id, provenance.phase, provenance.block_index);
  const StackDispatchDependencyDispatchValue* const consumer_dispatch =
      find_stack_dispatch_observation(
          scope_id,
          provenance.expected_consumer_phase,
          provenance.expected_consumer_block_index);
  const bool producer_dispatch_observed = producer_dispatch != nullptr;
  const bool consumer_dispatch_observed = consumer_dispatch != nullptr;
  const bool consumer_dispatch_planned =
      !consumer_dispatch_observed &&
      provenance.expected_consumer_phase == VulkanVisionStackPhase::Norm1 &&
      provenance.expected_consumer_block_index ==
          provenance.block_index + 1 &&
      !vision_stack_capture_dependency_contains_block(provenance.block_index);
  const bool producer_descriptor_known = true;
  const bool consumer_descriptor_known =
      provenance.expected_consumer_phase == VulkanVisionStackPhase::Norm1 ||
      provenance.expected_consumer_phase ==
          VulkanVisionStackPhase::IntermediateCapture;
  const bool fully_proven =
      allocation_proof.has_generation && allocation_proof.has_byte_range &&
      allocation_proof.byte_range > 0u && formal_last_use_proof &&
      producer_dispatch_observed && consumer_dispatch_observed &&
      producer_descriptor_known && consumer_descriptor_known;
  const std::string reject_reason = stack_dispatch_dependency_reject_reason(
      true,
      allocation_proof.has_generation,
      allocation_proof.has_byte_range && allocation_proof.byte_range > 0u,
      formal_last_use_proof,
      producer_dispatch_observed,
      consumer_dispatch_observed,
      provenance.expected_consumer_phase);

  std::ostringstream key;
  key << "stack_dispatch_dependency=1"
      << " contract=StackOwnerDispatchDependencyContract"
      << " phase=" << submit_phase_name(phase)
      << " callsite=" << retire_call_site_name(callsite)
      << " queue_submit=" << (queue_submit ? 1 : 0)
      << " resource_class="
      << (resource_class && resource_class[0] ? resource_class : "unknown")
      << " resource_kind=" << retired_resource_kind_name(kind)
      << " role=" << retired_resource_role_name(role)
      << " dependency_kind="
      << stack_dispatch_dependency_kind(provenance.expected_consumer_phase)
      << " producer_op=" << stack_dispatch_op_label(provenance.phase)
      << " producer_phase=" << vision_stack_phase_name(provenance.phase)
      << " producer_block=" << provenance.block_index
      << " consumer_op="
      << stack_dispatch_op_label(provenance.expected_consumer_phase)
      << " consumer_phase="
      << vision_stack_phase_name(provenance.expected_consumer_phase)
      << " consumer_block=" << provenance.expected_consumer_block_index
      << " scope_id=" << scope_id
      << " producer_dispatch_observed=" << (producer_dispatch_observed ? 1 : 0)
      << " producer_dispatch_first_position="
      << (producer_dispatch ? producer_dispatch->first_position : 0u)
      << " producer_dispatch_last_position="
      << (producer_dispatch ? producer_dispatch->last_position : 0u)
      << " consumer_dispatch_observed=" << (consumer_dispatch_observed ? 1 : 0)
      << " consumer_dispatch_first_position="
      << (consumer_dispatch ? consumer_dispatch->first_position : 0u)
      << " consumer_dispatch_last_position="
      << (consumer_dispatch ? consumer_dispatch->last_position : 0u)
      << " consumer_dispatch_planned=" << (consumer_dispatch_planned ? 1 : 0)
      << " consumer_dispatch_proof="
      << (consumer_dispatch_observed
              ? "recorded_dispatch"
              : (consumer_dispatch_planned
                     ? "planned_non_capture_residual2_to_norm1"
                     : "missing"))
      << " command_buffer_sequence=" << scope_id
      << " producer_descriptor_role=internal_output"
      << " producer_descriptor_set=0"
      << " producer_descriptor_binding=0"
      << " producer_access=shader_write"
      << " consumer_descriptor_role="
      << (provenance.expected_consumer_phase == VulkanVisionStackPhase::Norm1
              ? "activation_input"
              : (provenance.expected_consumer_phase ==
                         VulkanVisionStackPhase::IntermediateCapture
                     ? "requested_intermediate_output"
                     : "unknown"))
      << " consumer_descriptor_set=0"
      << " consumer_descriptor_binding="
      << (provenance.expected_consumer_phase == VulkanVisionStackPhase::Norm1
              ? 1
              : (provenance.expected_consumer_phase ==
                         VulkanVisionStackPhase::IntermediateCapture
                     ? 0
                     : -1))
      << " consumer_access=shader_read"
      << " allocation_label="
      << (allocation_label.empty() ? "unknown" : allocation_label)
      << " allocation_id=" << allocation_proof.allocation_id
      << " allocation_generation=" << allocation_proof.allocation_generation
      << " allocation_has_generation="
      << (allocation_proof.has_generation ? 1 : 0)
      << " byte_offset=" << allocation_proof.byte_offset
      << " byte_range=" << allocation_proof.byte_range
      << " allocation_has_byte_range="
      << (allocation_proof.has_byte_range ? 1 : 0)
      << " producer_live_range_known="
      << (allocation_proof.has_byte_range ? 1 : 0)
      << " consumer_live_range_known="
      << (allocation_proof.has_byte_range ? 1 : 0)
      << " formal_last_use_proof=" << (formal_last_use_proof ? 1 : 0)
      << " descriptor_binding_known="
      << (producer_descriptor_known && consumer_descriptor_known ? 1 : 0)
      << " fully_proven=" << (fully_proven ? 1 : 0)
      << " reject_reason=" << reject_reason;

  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  auto& value = stack_dispatch_dependency_dry_run_rows()[key.str()];
  value.count += 1u;
  value.bytes += bytes;
  if (queue_submit) {
    value.queue_submit_count += 1u;
  }
  if (fully_proven) {
    value.fully_proven_count += 1u;
  }
}

void note_vulkan_stack_allocation(
    const char* role,
    const VulkanStackTensorLifetimeClass lifetime,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    const int64_t dtype,
    const bool direct_buffer,
    const bool buffer_storage,
    const bool image_storage,
    const bool escapes_stack,
    const bool requested_intermediate,
    const uint64_t bytes) {
  if (!inside_vision_stack_phase()) {
    return;
  }
  std::ostringstream key;
  key << "stack_alloc"
      << " phase=" << vision_stack_phase_name(g_vision_stack_phase)
      << " block=" << g_vision_stack_block_index
      << " role=" << (role && role[0] ? role : "unknown")
      << " lifetime=" << stack_tensor_lifetime_name(lifetime)
      << " shape=" << format_sizes(sizes)
      << " strides=" << format_sizes(strides)
      << " dtype=" << dtype
      << " direct_buffer=" << (direct_buffer ? 1 : 0)
      << " buffer_storage=" << (buffer_storage ? 1 : 0)
      << " image_storage=" << (image_storage ? 1 : 0)
      << " escapes_stack=" << (escapes_stack ? 1 : 0)
      << " requested_intermediate=" << (requested_intermediate ? 1 : 0);
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  StackAllocationValue& value = stack_allocation_aggregate()[key.str()];
  value.count += 1u;
  value.bytes += bytes;
  value.peak_live_estimate_bytes = std::max(value.peak_live_estimate_bytes, bytes);
}

std::vector<std::string> stack_dispatch_aggregate_snapshot() {
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  std::vector<std::string> rows;
  rows.reserve(stack_dispatch_aggregate().size());
  for (const auto& item : stack_dispatch_aggregate()) {
    std::ostringstream row;
    row << item.first << " count=" << item.second;
    rows.push_back(row.str());
  }
  return rows;
}

std::vector<std::string> stack_allocation_aggregate_snapshot() {
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  std::vector<std::string> rows;
  rows.reserve(stack_allocation_aggregate().size());
  for (const auto& item : stack_allocation_aggregate()) {
    std::ostringstream row;
    row << item.first << " count=" << item.second.count
        << " bytes=" << item.second.bytes
        << " peak_live_estimate_bytes="
        << item.second.peak_live_estimate_bytes;
    rows.push_back(row.str());
  }
  return rows;
}

std::vector<std::string> stack_dispatch_dependency_dry_run_snapshot() {
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  std::vector<std::string> rows;
  rows.reserve(
      stack_dispatch_dependency_dispatch_rows().size() +
      stack_dispatch_dependency_dry_run_rows().size());
  for (const auto& item : stack_dispatch_dependency_dispatch_rows()) {
    std::ostringstream row;
    row << item.first << " count=" << item.second.count
        << " first_position=" << item.second.first_position
        << " last_position=" << item.second.last_position;
    rows.push_back(row.str());
  }
  for (const auto& item : stack_dispatch_dependency_dry_run_rows()) {
    std::ostringstream row;
    row << item.first << " count=" << item.second.count
        << " bytes=" << item.second.bytes
        << " queue_submit=" << item.second.queue_submit_count
        << " fully_proven_count=" << item.second.fully_proven_count;
    rows.push_back(row.str());
  }
  return rows;
}

void reset_stack_dispatch_aggregate() {
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  stack_dispatch_aggregate().clear();
}

void reset_stack_allocation_aggregate() {
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  stack_allocation_aggregate().clear();
}

void reset_stack_dispatch_dependency_dry_run() {
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  stack_dispatch_dependency_dispatch_rows().clear();
  stack_dispatch_dependency_dry_run_rows().clear();
}

void note_vulkan_queue_wait_idle() {
  vulkan_sync_counters().queue_wait_idle_count.fetch_add(
      1u, std::memory_order_relaxed);
}

void note_vulkan_forced_sync(VulkanForcedSyncReason reason) {
  VulkanSyncCounters& counters = vulkan_sync_counters();
  counters.forced_sync_count.fetch_add(1u, std::memory_order_relaxed);
  switch (reason) {
    case VulkanForcedSyncReason::ExplicitSynchronize:
      counters.forced_sync_explicit_synchronize_count.fetch_add(
          1u, std::memory_order_relaxed);
      return;
    case VulkanForcedSyncReason::TensorCpuReadback:
      counters.forced_sync_tensor_cpu_readback_count.fetch_add(
          1u, std::memory_order_relaxed);
      return;
    case VulkanForcedSyncReason::EventSynchronize:
      counters.forced_sync_event_synchronize_count.fetch_add(
          1u, std::memory_order_relaxed);
      return;
    case VulkanForcedSyncReason::RetireQueueDrain:
      counters.forced_sync_retire_queue_drain_count.fetch_add(
          1u, std::memory_order_relaxed);
      return;
    case VulkanForcedSyncReason::GpuTimestampQueryReset:
      counters.forced_sync_gpu_timestamp_query_reset_count.fetch_add(
          1u, std::memory_order_relaxed);
      return;
    case VulkanForcedSyncReason::FallbackPolicyReadback:
      counters.forced_sync_fallback_policy_readback_count.fetch_add(
          1u, std::memory_order_relaxed);
      return;
    case VulkanForcedSyncReason::Unknown:
      counters.forced_sync_unknown_count.fetch_add(
          1u, std::memory_order_relaxed);
      return;
  }
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
