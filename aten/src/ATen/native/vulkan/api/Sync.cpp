#include <ATen/native/vulkan/api/Sync.h>

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Resource.h>

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
thread_local std::vector<VulkanStackPlannedDispatchPosition>
    g_stack_planned_dispatch_positions;
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

const VulkanStackPlannedDispatchPosition* find_stack_planned_dispatch_position(
    const VulkanVisionStackPhase phase,
    const int64_t block_index) {
  for (const VulkanStackPlannedDispatchPosition& position :
       g_stack_planned_dispatch_positions) {
    if (position.phase == phase && position.block_index == block_index) {
      return &position;
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

struct StackOutputDeviceConsumerRegistrationValue final {
  uint64_t count = 0u;
};

std::map<std::string, StackOutputDeviceConsumerRegistrationValue>&
stack_output_device_consumer_registrations() {
  static std::map<std::string, StackOutputDeviceConsumerRegistrationValue> rows;
  return rows;
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

struct StackRegionBoundarySubmitPlanValue final {
  uint64_t count = 0u;
  uint64_t bytes = 0u;
  uint64_t queue_submit_count = 0u;
  uint64_t submit_removed_count = 0u;
  uint64_t barrier_inserted_count = 0u;
};

struct StackRegionBarrierOnlyCanaryValue final {
  uint64_t count = 0u;
  uint64_t bytes = 0u;
  uint64_t live_buffer_bound_count = 0u;
  uint64_t submit_removed_count = 0u;
  uint64_t barrier_inserted_count = 0u;
};

std::map<std::string, StackDispatchDependencyDispatchValue>&
stack_dispatch_dependency_dispatch_rows() {
  static std::map<std::string, StackDispatchDependencyDispatchValue> rows;
  return rows;
}

std::map<std::string, StackDispatchDependencyDispatchValue>&
stack_dispatch_dependency_insertion_point_rows() {
  static std::map<std::string, StackDispatchDependencyDispatchValue> rows;
  return rows;
}

std::map<std::string, StackDispatchDependencyDispatchValue>&
stack_dispatch_dependency_live_buffer_binding_rows() {
  static std::map<std::string, StackDispatchDependencyDispatchValue> rows;
  return rows;
}

std::map<std::string, StackDispatchDependencyDryRunValue>&
stack_dispatch_dependency_dry_run_rows() {
  static std::map<std::string, StackDispatchDependencyDryRunValue> rows;
  return rows;
}

std::map<std::string, StackRegionBoundarySubmitPlanValue>&
stack_region_boundary_submit_plan_rows() {
  static std::map<std::string, StackRegionBoundarySubmitPlanValue> rows;
  return rows;
}

std::map<std::string, StackRegionBarrierOnlyCanaryValue>&
stack_region_barrier_only_canary_rows() {
  static std::map<std::string, StackRegionBarrierOnlyCanaryValue> rows;
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

std::string stack_dispatch_dependency_insertion_point_token(
    const uint64_t scope_id,
    const VulkanVisionStackPhase phase,
    const int64_t block,
    const uint64_t planned_position,
    const char* const shader_name) {
  std::ostringstream token;
  token << "stack_scope:" << scope_id << ":before_phase:"
        << vision_stack_phase_name(phase) << ":block:" << block
        << ":planned_step:" << planned_position << ":shader:"
        << (shader_name && shader_name[0] ? shader_name : "unknown");
  return token.str();
}

std::string stack_dispatch_dependency_insertion_point_key(
    const uint64_t scope_id,
    const VulkanVisionStackPhase phase,
    const int64_t block,
    const VulkanStackPlannedDispatchPosition& planned_position,
    const char* const shader_name) {
  const std::string token = stack_dispatch_dependency_insertion_point_token(
      scope_id, phase, block, planned_position.planned_position, shader_name);
  std::ostringstream key;
  key << "pre_dispatch_insertion_point=1"
      << " scope_id=" << scope_id
      << " phase=" << vision_stack_phase_name(phase)
      << " block=" << block
      << " shader=" << (shader_name && shader_name[0] ? shader_name : "unknown")
      << " planned_position=" << planned_position.planned_position
      << " planned_position_space=stack_plan_logical_step"
      << " insertion_point_class=before_stack_plan_step_dispatch"
      << " insertion_point_token=" << token;
  return key.str();
}

std::string stack_live_buffer_binding_handle_token(const VkBuffer handle) {
  std::ostringstream stream;
  stream << handle;
  return stream.str();
}

std::string stack_live_buffer_binding_object_token(
    const VulkanBuffer& buffer) {
  std::ostringstream stream;
  stream << static_cast<const void*>(&buffer);
  return stream.str();
}

std::string stack_dispatch_dependency_live_buffer_binding_key(
    const uint64_t scope_id,
    const VulkanVisionStackPhase phase,
    const int64_t block,
    const char* const shader_name,
    const uint32_t binding_idx,
    const uint64_t next_recorded_position,
    const VulkanBuffer& buffer) {
  const uint64_t allocation_id = buffer.allocation_id();
  const uint64_t allocation_generation =
      vulkan_memory_allocation_generation(allocation_id);
  const bool has_memory = buffer.has_memory();
  const bool has_generation =
      allocation_id != 0u && allocation_generation != 0u;
  const bool has_byte_range = has_memory && buffer.mem_range() != 0u;
  std::ostringstream key;
  key << "live_vulkan_buffer_binding=1"
      << " scope_id=" << scope_id
      << " phase=" << vision_stack_phase_name(phase)
      << " block=" << block
      << " shader=" << (shader_name && shader_name[0] ? shader_name : "unknown")
      << " descriptor_binding=" << binding_idx
      << " command_buffer_sequence=" << scope_id
      << " next_recorded_dispatch_position=" << next_recorded_position
      << " allocation_id=" << allocation_id
      << " allocation_generation=" << allocation_generation
      << " allocation_has_generation=" << (has_generation ? 1 : 0)
      << " byte_offset=" << static_cast<uint64_t>(buffer.mem_offset())
      << " byte_range=" << static_cast<uint64_t>(buffer.mem_range())
      << " allocation_has_byte_range=" << (has_byte_range ? 1 : 0)
      << " allocated_bytes=" << static_cast<uint64_t>(buffer.allocated_size())
      << " allocation_label="
      << (buffer.allocation_label().empty() ? "unknown"
                                            : buffer.allocation_label())
      << " live_buffer_has_memory=" << (has_memory ? 1 : 0)
      << " live_buffer_owns_memory=" << (buffer.owns_memory() ? 1 : 0)
      << " live_vulkan_buffer_handle_present="
      << (buffer.handle() != VK_NULL_HANDLE ? 1 : 0)
      << " live_vulkan_buffer_handle_token="
      << stack_live_buffer_binding_handle_token(buffer.handle())
      << " live_vulkan_buffer_object_token="
      << stack_live_buffer_binding_object_token(buffer)
      << " binding_source=submit_compute_job_descriptor_argument";
  return key.str();
}

std::string stack_region_barrier_only_canary_key(
    const uint64_t scope_id,
    const VulkanVisionStackPhase phase,
    const int64_t block,
    const char* const shader_name,
    const uint32_t binding_idx,
    const uint64_t next_recorded_position,
    const VulkanBuffer& buffer,
    const bool live_buffer_bound,
    const char* const status,
    const char* const reject_reason) {
  const uint64_t allocation_id = buffer.allocation_id();
  const uint64_t allocation_generation =
      vulkan_memory_allocation_generation(allocation_id);
  std::ostringstream key;
  key << "stack_region_barrier_only_canary=1"
      << " contract=StackRegionBarrierOnlyCanary"
      << " schema=StackRegionBarrierOnlyCanary.v0"
      << " opt_in_env=PYTORCH_VULKAN_STACK_REGION_BARRIER_CANARY"
      << " opt_in_target=non_capture_residual2_norm1_block1"
      << " behavior_neutral=1"
      << " default_behavior_unchanged=1"
      << " selected_boundary_id=non_capture_boundary:producer_block=0:consumer_block=1"
      << " selected_scope=non_capture_stack_boundary"
      << " producer_phase=residual2"
      << " producer_block=0"
      << " consumer_phase=norm1"
      << " consumer_block=1"
      << " live_phase=" << vision_stack_phase_name(phase)
      << " live_block=" << block
      << " shader=" << (shader_name && shader_name[0] ? shader_name : "unknown")
      << " descriptor_binding=" << binding_idx
      << " command_buffer_sequence=" << scope_id
      << " next_recorded_dispatch_position=" << next_recorded_position
      << " allocation_id=" << allocation_id
      << " allocation_generation=" << allocation_generation
      << " allocation_has_generation="
      << (allocation_id != 0u && allocation_generation != 0u ? 1 : 0)
      << " byte_offset=" << static_cast<uint64_t>(buffer.mem_offset())
      << " byte_range=" << static_cast<uint64_t>(buffer.mem_range())
      << " allocation_has_byte_range="
      << (buffer.has_memory() && buffer.mem_range() != 0u ? 1 : 0)
      << " allocation_label="
      << (buffer.allocation_label().empty() ? "unknown"
                                            : buffer.allocation_label())
      << " live_vulkan_buffer_binding_available="
      << (live_buffer_bound ? 1 : 0)
      << " live_vulkan_buffer_handle_present="
      << (buffer.handle() != VK_NULL_HANDLE ? 1 : 0)
      << " live_vulkan_buffer_handle_token="
      << stack_live_buffer_binding_handle_token(buffer.handle())
      << " live_vulkan_buffer_object_token="
      << stack_live_buffer_binding_object_token(buffer)
      << " src_stage=compute_shader"
      << " src_access=shader_write"
      << " dst_stage=compute_shader"
      << " dst_access=shader_read"
      << " current_run_proof_match=0"
      << " current_run_proof_status="
      << "missing_pre_dispatch_barrier_plan_proof_record"
      << " proof_source=not_available_before_consumer_dispatch_recording"
      << " barrier_only_status=" << status
      << " reject_reason=" << reject_reason
      << " capture_edge=0"
      << " public_output=0"
      << " final_output=0"
      << " host_visible=0"
      << " readback_edge=0"
      << " behavior_change_allowed=0"
      << " submit_skip_behavior_change_allowed=0"
      << " barrier_behavior_allowed=0"
      << " barriers_inserted=0"
      << " submits_removed=0";
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

const char* stack_region_barrier_only_canary_target() {
  const char* env = std::getenv("PYTORCH_VULKAN_STACK_REGION_BARRIER_CANARY");
  return (env && *env) ? env : nullptr;
}

bool stack_region_barrier_only_canary_target_selected(
    const char* const target) {
  if (target == nullptr) {
    return false;
  }
  const std::string value(target);
  return value == "non_capture_residual2_norm1_block1" ||
      value == "producer_block_0_consumer_block_1";
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

bool dependency_is_requested_capture_edge(
    const std::map<std::string, std::string>& fields) {
  return field_or(fields, "consumer_phase", "unknown") ==
          "intermediate_capture" &&
      field_or(fields, "consumer_descriptor_role", "unknown") ==
          "requested_intermediate_output";
}

std::string stack_output_device_consumer_registration_key(
    const std::string& captured_block,
    const std::string& output_role) {
  return "capture_block=" + captured_block + ":output_role=" + output_role;
}

std::string stack_output_device_consumer_registration_key(
    const std::map<std::string, std::string>& fields) {
  return stack_output_device_consumer_registration_key(
      field_or(fields, "captured_block", "unknown"),
      field_or(fields, "output_role", "unknown"));
}

VulkanVisionStackPhase vision_stack_phase_from_graph_name(
    const std::string& name) {
  if (name == "stack_entry") {
    return VulkanVisionStackPhase::StackEntry;
  }
  if (name == "block_entry") {
    return VulkanVisionStackPhase::BlockEntry;
  }
  if (name == "norm1") {
    return VulkanVisionStackPhase::Norm1;
  }
  if (name == "qkv_linear") {
    return VulkanVisionStackPhase::QkvLinear;
  }
  if (name == "qkv_transform") {
    return VulkanVisionStackPhase::QkvTransform;
  }
  if (name == "attention") {
    return VulkanVisionStackPhase::Attention;
  }
  if (name == "proj_linear") {
    return VulkanVisionStackPhase::ProjLinear;
  }
  if (name == "residual1") {
    return VulkanVisionStackPhase::Residual1;
  }
  if (name == "norm2") {
    return VulkanVisionStackPhase::Norm2;
  }
  if (name == "fc1_gelu") {
    return VulkanVisionStackPhase::Fc1Gelu;
  }
  if (name == "fc2") {
    return VulkanVisionStackPhase::Fc2;
  }
  if (name == "residual2") {
    return VulkanVisionStackPhase::Residual2;
  }
  if (name == "intermediate_capture") {
    return VulkanVisionStackPhase::IntermediateCapture;
  }
  if (name == "stack_exit") {
    return VulkanVisionStackPhase::StackExit;
  }
  return VulkanVisionStackPhase::Unknown;
}

struct BarrierPlanDispatchPosition final {
  bool completed_position_known = false;
  uint64_t completed_first_position = 0u;
  uint64_t completed_last_position = 0u;
  std::string completed_position_source = "missing";
  bool planned_position_known = false;
  uint64_t planned_position = 0u;
  std::string planned_position_source = "missing";
  std::string planned_position_space = "missing";
  bool insertion_point_available = false;
  uint64_t insertion_point_first_position = 0u;
  uint64_t insertion_point_last_position = 0u;
  std::string insertion_point_token = "missing";
  std::string insertion_point_class = "missing";
  std::string insertion_point_source = "missing";
};

struct BarrierPlanLiveBufferBinding final {
  bool available = false;
  uint64_t count = 0u;
  uint64_t first_position = 0u;
  uint64_t last_position = 0u;
  std::string status = "missing_live_vulkan_buffer_binding";
  std::string source = "missing";
  std::string handle_token = "missing";
  std::string object_token = "missing";
  std::string allocation_label = "missing";
  std::string descriptor_binding = "missing";
  std::string shader = "missing";
};

struct CaptureAllocationSummary final {
  uint64_t public_capture_count = 0u;
  uint64_t public_capture_bytes = 0u;
  uint64_t private_bridge_capture_count = 0u;
  uint64_t private_bridge_capture_bytes = 0u;
  std::string public_capture_shape = "missing";
  std::string private_bridge_capture_shape = "missing";
};

struct StackOutputDeviceConsumerRegistrationSummary final {
  uint64_t count = 0u;
  bool consumer_in_same_planned_region = false;
  bool python_public_boundary_before_consumption = false;
  bool host_visible_boundary_before_consumption = false;
  bool host_visible_access_before_consumption = false;
  bool host_readback_before_consumption = false;
  std::string stack_context_id = "missing";
  std::string stack_session_id = "missing";
  std::string stack_plan_id = "missing";
  std::string captured_substep = "missing";
  std::string output_role = "missing";
  std::string output_shape = "missing";
  std::string output_layout = "missing";
  std::string strip_or_view_relation = "missing";
  std::string downstream_consumer_id = "missing";
  std::string downstream_consumer_context = "missing";
  std::string expected_consumer_input_index = "missing";
  std::string expected_consumer_shape = "missing";
  std::string expected_consumer_layout = "missing";
};

enum class CaptureOutputBoundaryScope : uint8_t {
  Combined,
  PublicCapture,
  BridgePrivateCapture,
};

const char* capture_output_boundary_scope_name(
    const CaptureOutputBoundaryScope scope) {
  switch (scope) {
    case CaptureOutputBoundaryScope::Combined:
      return "combined";
    case CaptureOutputBoundaryScope::PublicCapture:
      return "public_capture";
    case CaptureOutputBoundaryScope::BridgePrivateCapture:
      return "bridge_private_capture";
  }
  return "unknown";
}

const char* capture_output_boundary_record_prefix(
    const CaptureOutputBoundaryScope scope) {
  switch (scope) {
    case CaptureOutputBoundaryScope::Combined:
      return "capture_output_boundary_edge_";
    case CaptureOutputBoundaryScope::PublicCapture:
      return "capture_output_boundary_public_edge_";
    case CaptureOutputBoundaryScope::BridgePrivateCapture:
      return "capture_output_boundary_bridge_private_edge_";
  }
  return "capture_output_boundary_unknown_edge_";
}

std::string barrier_plan_dispatch_position_key(
    const std::string& scope_id,
    const std::string& phase,
    const std::string& block) {
  return "scope=" + scope_id + ":phase=" + phase + ":block=" + block;
}

std::string barrier_plan_dispatch_position_key(
    const std::map<std::string, std::string>& fields,
    const char* phase_key,
    const char* block_key) {
  return barrier_plan_dispatch_position_key(
      field_or(fields, "scope_id", "unknown"),
      field_or(fields, phase_key, "unknown"),
      field_or(fields, block_key, "unknown"));
}

std::string barrier_plan_live_buffer_binding_key(
    const std::string& scope_id,
    const std::string& phase,
    const std::string& block,
    const std::string& descriptor_binding,
    const std::string& allocation_id,
    const std::string& allocation_generation,
    const std::string& byte_offset,
    const std::string& byte_range) {
  return "scope=" + scope_id + ":phase=" + phase + ":block=" + block +
      ":binding=" + descriptor_binding + ":allocation=" + allocation_id +
      ":generation=" + allocation_generation + ":offset=" + byte_offset +
      ":range=" + byte_range;
}

std::string barrier_plan_live_buffer_binding_key(
    const std::map<std::string, std::string>& fields,
    const char* phase_key,
    const char* block_key,
    const char* descriptor_binding_key) {
  return barrier_plan_live_buffer_binding_key(
      field_or(fields, "scope_id", "unknown"),
      field_or(fields, phase_key, "unknown"),
      field_or(fields, block_key, "unknown"),
      field_or(fields, descriptor_binding_key, "unknown"),
      field_or(fields, "allocation_id", "unknown"),
      field_or(fields, "allocation_generation", "unknown"),
      field_or(fields, "byte_offset", "unknown"),
      field_or(fields, "byte_range", "unknown"));
}

std::string barrier_plan_live_buffer_binding_allocation_key(
    const std::map<std::string, std::string>& fields,
    const char* phase_key,
    const char* block_key,
    const char* descriptor_binding_key) {
  return "scope=" + field_or(fields, "scope_id", "unknown") + ":phase=" +
      field_or(fields, phase_key, "unknown") + ":block=" +
      field_or(fields, block_key, "unknown") + ":binding=" +
      field_or(fields, descriptor_binding_key, "unknown") + ":allocation=" +
      field_or(fields, "allocation_id", "unknown") + ":generation=" +
      field_or(fields, "allocation_generation", "unknown");
}

std::map<std::string, BarrierPlanDispatchPosition>
build_barrier_plan_dispatch_positions(const std::vector<std::string>& rows) {
  std::map<std::string, BarrierPlanDispatchPosition> positions;
  for (const auto& row : rows) {
    const auto fields = parse_space_separated_fields(row);
    const std::string key =
        barrier_plan_dispatch_position_key(fields, "phase", "block");
    auto& position = positions[key];
    position.completed_position_known = true;
    position.completed_position_source = "completed_graph_dispatch_node";
    const uint64_t first_position = parsed_u64(fields, "first_position");
    const uint64_t last_position = parsed_u64(fields, "last_position");
    if (
        position.completed_first_position == 0u ||
        (first_position != 0u &&
         first_position < position.completed_first_position)) {
      position.completed_first_position = first_position;
    }
    if (last_position > position.completed_last_position) {
      position.completed_last_position = last_position;
    }
  }
  return positions;
}

std::map<std::string, BarrierPlanDispatchPosition>
build_barrier_plan_insertion_points(const std::vector<std::string>& rows) {
  std::map<std::string, BarrierPlanDispatchPosition> positions;
  for (const auto& row : rows) {
    const auto fields = parse_space_separated_fields(row);
    const std::string key =
        barrier_plan_dispatch_position_key(fields, "phase", "block");
    auto& position = positions[key];
    position.insertion_point_available = true;
    position.insertion_point_source = "pre_dispatch_command_recording_hook";
    position.insertion_point_token =
        field_or(fields, "insertion_point_token", "missing");
    position.insertion_point_class =
        field_or(fields, "insertion_point_class", "missing");
    position.planned_position_known = true;
    position.planned_position = parsed_u64(fields, "planned_position");
    position.planned_position_source = "pre_dispatch_insertion_point";
    position.planned_position_space =
        field_or(fields, "planned_position_space", "missing");
    const uint64_t first_position =
        parsed_u64(fields, "next_recorded_dispatch_first_position");
    const uint64_t last_position =
        parsed_u64(fields, "next_recorded_dispatch_last_position");
    if (
        position.insertion_point_first_position == 0u ||
        (first_position != 0u &&
         first_position < position.insertion_point_first_position)) {
      position.insertion_point_first_position = first_position;
    }
    if (last_position > position.insertion_point_last_position) {
      position.insertion_point_last_position = last_position;
    }
  }
  return positions;
}

std::map<std::string, BarrierPlanLiveBufferBinding>
build_barrier_plan_live_buffer_bindings(
    const std::vector<std::string>& rows,
    std::map<std::string, uint64_t>& allocation_binding_counts) {
  std::map<std::string, BarrierPlanLiveBufferBinding> bindings;
  for (const auto& row : rows) {
    const auto fields = parse_space_separated_fields(row);
    const std::string exact_key = barrier_plan_live_buffer_binding_key(
        fields, "phase", "block", "descriptor_binding");
    const std::string allocation_key =
        barrier_plan_live_buffer_binding_allocation_key(
            fields, "phase", "block", "descriptor_binding");
    const uint64_t count = parsed_u64(fields, "count");
    allocation_binding_counts[allocation_key] += count;
    auto& binding = bindings[exact_key];
    binding.available =
        field_or(fields, "allocation_has_generation", "0") == "1" &&
        field_or(fields, "allocation_has_byte_range", "0") == "1" &&
        field_or(fields, "live_buffer_has_memory", "0") == "1" &&
        field_or(fields, "live_vulkan_buffer_handle_present", "0") == "1";
    binding.status =
        binding.available ? "live_buffer_bound" : "missing_live_vulkan_buffer";
    binding.source = field_or(fields, "binding_source", "missing");
    binding.handle_token =
        field_or(fields, "live_vulkan_buffer_handle_token", "missing");
    binding.object_token =
        field_or(fields, "live_vulkan_buffer_object_token", "missing");
    binding.allocation_label = field_or(fields, "allocation_label", "missing");
    binding.descriptor_binding =
        field_or(fields, "descriptor_binding", "missing");
    binding.shader = field_or(fields, "shader", "missing");
    binding.count += count;
    const uint64_t first_position =
        parsed_u64(fields, "next_recorded_dispatch_first_position");
    const uint64_t last_position =
        parsed_u64(fields, "next_recorded_dispatch_last_position");
    if (
        binding.first_position == 0u ||
        (first_position != 0u && first_position < binding.first_position)) {
      binding.first_position = first_position;
    }
    if (last_position > binding.last_position) {
      binding.last_position = last_position;
    }
  }
  return bindings;
}

BarrierPlanDispatchPosition barrier_plan_consumer_dispatch_position(
    const std::map<std::string, std::string>& fields,
    const std::map<std::string, BarrierPlanDispatchPosition>& positions,
    const std::map<std::string, BarrierPlanDispatchPosition>& insertion_points) {
  if (field_or(fields, "consumer_dispatch_observed", "0") == "1") {
    BarrierPlanDispatchPosition position;
    position.completed_position_known = true;
    position.completed_first_position =
        parsed_u64(fields, "consumer_dispatch_first_position");
    position.completed_last_position =
        parsed_u64(fields, "consumer_dispatch_last_position");
    position.completed_position_source = "recorded_dependency_edge";
    position.planned_position_known = true;
    position.planned_position = position.completed_first_position;
    position.planned_position_source = "recorded_dependency_edge";
    position.planned_position_space = "command_recording_dispatch_sequence";
    return position;
  }
  if (!boundary_has_planned_non_capture_norm1_consumer(fields)) {
    return {};
  }
  BarrierPlanDispatchPosition position;
  const std::string key = barrier_plan_dispatch_position_key(
      fields, "consumer_phase", "consumer_block");
  const auto it = positions.find(key);
  if (it != positions.end()) {
    position.completed_position_known = it->second.completed_position_known;
    position.completed_first_position = it->second.completed_first_position;
    position.completed_last_position = it->second.completed_last_position;
    position.completed_position_source = it->second.completed_position_source;
  }
  const auto insertion_it = insertion_points.find(key);
  if (insertion_it != insertion_points.end()) {
    position.insertion_point_available =
        insertion_it->second.insertion_point_available;
    position.insertion_point_first_position =
        insertion_it->second.insertion_point_first_position;
    position.insertion_point_last_position =
        insertion_it->second.insertion_point_last_position;
    position.insertion_point_token = insertion_it->second.insertion_point_token;
    position.insertion_point_class = insertion_it->second.insertion_point_class;
    position.insertion_point_source = insertion_it->second.insertion_point_source;
    position.planned_position_known =
        insertion_it->second.planned_position_known;
    position.planned_position = insertion_it->second.planned_position;
    position.planned_position_source =
        insertion_it->second.planned_position_source;
    position.planned_position_space =
        insertion_it->second.planned_position_space;
  }
  const VulkanVisionStackPhase consumer_phase =
      vision_stack_phase_from_graph_name(
          field_or(fields, "consumer_phase", "unknown"));
  const int64_t consumer_block =
      static_cast<int64_t>(parsed_u64(fields, "consumer_block"));
  if (const VulkanStackPlannedDispatchPosition* const planned_position =
          find_stack_planned_dispatch_position(consumer_phase, consumer_block)) {
    if (!position.planned_position_known) {
      position.planned_position_known = true;
      position.planned_position = planned_position->planned_position;
      position.planned_position_source = "pre_recording_stack_shape_plan";
      position.planned_position_space = "stack_plan_logical_step";
    }
  }
  return position;
}

BarrierPlanDispatchPosition barrier_plan_consumer_dispatch_position(
    const std::map<std::string, std::string>& fields,
    const std::map<std::string, BarrierPlanDispatchPosition>& positions) {
  static const std::map<std::string, BarrierPlanDispatchPosition>
      kNoInsertionPoints;
  return barrier_plan_consumer_dispatch_position(
      fields, positions, kNoInsertionPoints);
}

BarrierPlanLiveBufferBinding barrier_plan_live_buffer_binding(
    const std::map<std::string, std::string>& fields,
    const std::map<std::string, BarrierPlanLiveBufferBinding>& bindings,
    const std::map<std::string, uint64_t>& allocation_binding_counts) {
  const std::string exact_key = barrier_plan_live_buffer_binding_key(
      fields,
      "consumer_phase",
      "consumer_block",
      "consumer_descriptor_binding");
  const auto it = bindings.find(exact_key);
  if (it != bindings.end()) {
    return it->second;
  }
  const std::string allocation_key =
      barrier_plan_live_buffer_binding_allocation_key(
          fields,
          "consumer_phase",
          "consumer_block",
          "consumer_descriptor_binding");
  const auto allocation_it = allocation_binding_counts.find(allocation_key);
  BarrierPlanLiveBufferBinding missing;
  if (allocation_it != allocation_binding_counts.end() &&
      allocation_it->second > 0u) {
    missing.status = "binding_range_mismatch";
    missing.count = allocation_it->second;
  }
  return missing;
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

std::vector<std::string> barrier_plan_missing_dependency_metadata_fields(
    const std::map<std::string, std::string>& fields,
    const BarrierPlanDispatchPosition& consumer_position) {
  std::vector<std::string> missing = missing_dependency_metadata_fields(fields);
  if (boundary_has_planned_non_capture_norm1_consumer(fields)) {
    missing.erase(
        std::remove(
            missing.begin(), missing.end(), std::string("consumer_dispatch")),
        missing.end());
    if (
        !consumer_position.completed_position_known &&
        !consumer_position.planned_position_known) {
      missing.emplace_back("consumer_dispatch_position");
    } else if (!consumer_position.planned_position_known) {
      missing.emplace_back("pre_recording_consumer_dispatch_position_api");
    } else if (
        !consumer_position.insertion_point_available &&
        consumer_position.planned_position_space !=
        "command_recording_dispatch_sequence") {
      missing.emplace_back(
          "pre_recording_command_buffer_dispatch_position_api");
    }
  }
  return missing;
}

std::vector<std::string> barrier_plan_missing_dependency_metadata_fields(
    const std::map<std::string, std::string>& fields) {
  return barrier_plan_missing_dependency_metadata_fields(
      fields, BarrierPlanDispatchPosition{});
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
    const std::map<std::string, std::string>& fields,
    const BarrierPlanDispatchPosition& consumer_position) {
  const std::vector<std::string> missing =
      barrier_plan_missing_dependency_metadata_fields(fields, consumer_position);
  if (!missing.empty()) {
    if (
        dependency_is_requested_capture_edge(fields) &&
        missing.front() == "formal_last_use_proof") {
      return "capture_output_boundary_contract_incomplete";
    }
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

std::string barrier_plan_rejection_reason(
    const std::map<std::string, std::string>& fields) {
  return barrier_plan_rejection_reason(fields, BarrierPlanDispatchPosition{});
}

bool barrier_plan_record_is_plannable(
    const std::map<std::string, std::string>& fields,
    const BarrierPlanDispatchPosition& consumer_position) {
  return barrier_plan_rejection_reason(fields, consumer_position) == "none";
}

bool barrier_plan_record_is_plannable(
    const std::map<std::string, std::string>& fields) {
  return barrier_plan_record_is_plannable(
      fields, BarrierPlanDispatchPosition{});
}

bool barrier_plan_stage_access_available(
    const std::string& producer_access,
    const std::string& consumer_access) {
  return producer_access != "unknown" && consumer_access != "unknown" &&
      stage_for_access(producer_access) != "unknown_stage" &&
      stage_for_access(consumer_access) != "unknown_stage";
}

std::string barrier_plan_visibility_dependency_status(
    const bool plannable,
    const bool stage_access_available,
    const BarrierPlanDispatchPosition& consumer_position,
    const BarrierPlanLiveBufferBinding& live_binding) {
  if (!stage_access_available) {
    return "missing_stage_access";
  }
  if (!plannable && !consumer_position.insertion_point_available) {
    return "missing_consumer_dispatch_live_position";
  }
  if (!live_binding.available) {
    return live_binding.status;
  }
  return "live_buffer_bound_behavior_change_vetoed";
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
    const std::map<std::string, BarrierPlanDispatchPosition>& positions,
    const std::map<std::string, BarrierPlanDispatchPosition>& insertion_points,
    const std::map<std::string, BarrierPlanLiveBufferBinding>& live_bindings,
    const std::map<std::string, uint64_t>& live_allocation_binding_counts,
    const size_t index) {
  const auto fields = parse_space_separated_fields(row);
  const BarrierPlanDispatchPosition consumer_position =
      barrier_plan_consumer_dispatch_position(fields, positions, insertion_points);
  const BarrierPlanLiveBufferBinding live_binding =
      barrier_plan_live_buffer_binding(
          fields, live_bindings, live_allocation_binding_counts);
  const bool plannable =
      barrier_plan_record_is_plannable(fields, consumer_position);
  const std::string producer_access = field_or(fields, "producer_access", "unknown");
  const std::string consumer_access = field_or(fields, "consumer_access", "unknown");
  const bool consumer_dispatch_observed =
      field_or(fields, "consumer_dispatch_observed", "0") == "1";
  const bool consumer_dispatch_planned =
      boundary_has_planned_non_capture_norm1_consumer(fields);
  const std::string consumer_position_string =
      consumer_position.completed_position_known
      ? std::to_string(consumer_position.completed_first_position)
      : (consumer_position.planned_position_known
             ? std::to_string(consumer_position.planned_position)
             : field_or(fields, "consumer_dispatch_first_position", "unknown"));
  const bool planned_completed_agree =
      consumer_position.planned_position_known &&
      consumer_position.completed_position_known &&
      consumer_position.insertion_point_available &&
      consumer_position.insertion_point_first_position ==
          consumer_position.completed_first_position;
  const std::string pre_recording_position_status =
      consumer_position.planned_position_source == "recorded_dependency_edge"
      ? "recorded_dependency_edge"
      : (consumer_position.insertion_point_available
             ? "pre_dispatch_insertion_point_available"
             : (consumer_position.planned_position_known
                    ? (consumer_position.planned_position_space ==
                               "command_recording_dispatch_sequence"
                           ? "pre_recording_command_position_available"
                           : "pre_recording_logical_position_available_missing_command_buffer_position_api")
             : (consumer_dispatch_planned
                    ? (consumer_position.completed_position_known
                           ? "completed_graph_position_available_missing_pre_recording_position_api"
                           : "missing_completed_graph_and_pre_recording_position")
                    : "not_planned")));
  const bool stage_access_available =
      barrier_plan_stage_access_available(producer_access, consumer_access);
  const std::string visibility_dependency_status =
      barrier_plan_visibility_dependency_status(
          plannable, stage_access_available, consumer_position, live_binding);
  const bool barrier_canary_candidate_if_behavior_allowed =
      plannable && stage_access_available && live_binding.available &&
      field_or(fields, "queue_submit", "0") == "1";
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
      consumer_position_string, first);
  append_json_string(
      out,
      "consumer_dispatch_last_position",
      consumer_position.completed_position_known
          ? std::to_string(consumer_position.completed_last_position)
          : field_or(fields, "consumer_dispatch_last_position", "unknown"),
      first);
  append_json_bool(
      out, "consumer_dispatch_observed", consumer_dispatch_observed, first);
  append_json_bool(
      out, "consumer_dispatch_planned", consumer_dispatch_planned, first);
  append_json_string(
      out,
      "consumer_dispatch_proof",
      field_or(fields, "consumer_dispatch_proof", "missing"),
      first);
  append_json_string(
      out,
      "consumer_dispatch_position_status",
      consumer_dispatch_observed
          ? "recorded"
          : (consumer_dispatch_planned &&
             consumer_position.completed_position_known
                 ? "completed_graph_observed"
                 : (consumer_dispatch_planned
                        ? "planned_missing_completed_graph_position"
                        : "missing")),
      first);
  append_json_string(
      out,
      "consumer_dispatch_position_source",
      consumer_position.completed_position_source,
      first);
  append_json_bool(
      out,
      "completed_consumer_dispatch_position_available",
      consumer_position.completed_position_known,
      first);
  append_json_string(
      out,
      "completed_consumer_dispatch_position",
      consumer_position.completed_position_known
          ? std::to_string(consumer_position.completed_first_position)
          : "missing",
      first);
  append_json_string(
      out,
      "completed_consumer_dispatch_position_space",
      consumer_position.completed_position_known
          ? "command_recording_dispatch_sequence"
          : "missing",
      first);
  append_json_string(
      out,
      "completed_consumer_dispatch_position_source",
      consumer_position.completed_position_source,
      first);
  append_json_string(
      out,
      "planned_consumer_dispatch_position",
      consumer_position.planned_position_known
          ? std::to_string(consumer_position.planned_position)
          : (consumer_dispatch_planned ? "missing_pre_recording_position"
                                       : "not_planned"),
      first);
  append_json_bool(
      out,
      "pre_recording_position_available",
      consumer_position.planned_position_known,
      first);
  append_json_bool(
      out,
      "planned_consumer_dispatch_position_available",
      consumer_position.planned_position_known,
      first);
  append_json_string(
      out,
      "planned_consumer_dispatch_position_space",
      consumer_position.planned_position_space,
      first);
  append_json_string(
      out,
      "planned_consumer_dispatch_position_source",
      consumer_position.planned_position_source,
      first);
  append_json_bool(
      out,
      "planned_completed_position_agree",
      planned_completed_agree,
      first);
  append_json_bool(
      out,
      "pre_recording_barrier_insertion_point_available",
      consumer_position.insertion_point_available,
      first);
  append_json_string(
      out,
      "pre_recording_barrier_insertion_point_token",
      consumer_position.insertion_point_token,
      first);
  append_json_string(
      out,
      "pre_recording_barrier_insertion_point_class",
      consumer_position.insertion_point_class,
      first);
  append_json_string(
      out,
      "pre_recording_barrier_insertion_point_source",
      consumer_position.insertion_point_source,
      first);
  append_json_string(
      out,
      "pre_recording_barrier_insertion_point_next_dispatch_position",
      consumer_position.insertion_point_available
          ? std::to_string(consumer_position.insertion_point_first_position)
          : "missing",
      first);
  append_json_string(
      out,
      "planned_completed_position_agreement_status",
      !consumer_position.planned_position_known ||
              !consumer_position.completed_position_known
          ? "missing_position"
          : (planned_completed_agree
                 ? "agree"
                 : (!consumer_position.insertion_point_available
                        ? "different_position_spaces"
                        : "mismatch")),
      first);
  append_json_string(
      out,
      "pre_recording_position_status",
      pre_recording_position_status,
      first);
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
  append_json_bool(
      out, "stage_access_available", stage_access_available, first);
  append_json_string(
      out,
      "stage_access_status",
      stage_access_available ? "available" : "missing_stage_access",
      first);
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
  append_json_string(
      out,
      "barrier_insertion_location_class",
      plannable
          ? consumer_position.insertion_point_class
          : (consumer_position.insertion_point_available
                 ? consumer_position.insertion_point_class
                 : (consumer_position.planned_position_known
                        ? "before_planned_consumer_logical_step_dry_run_only"
                        : "missing_pre_recording_position")),
      first);
  append_json_bool(out, "plannable", plannable, first);
  append_json_bool(
      out,
      "could_theoretically_replace_phase_boundary_submit",
      plannable && field_or(fields, "queue_submit", "0") == "1",
      first);
  append_json_bool(out, "behavior_change_allowed", false, first);
  append_json_bool(
      out,
      "live_vulkan_buffer_binding_available",
      live_binding.available,
      first);
  append_json_string(
      out,
      "live_vulkan_buffer_binding_status",
      live_binding.status,
      first);
  append_json_string(
      out,
      "proof_to_live_buffer_binding_status",
      live_binding.status,
      first);
  append_json_string(
      out, "live_vulkan_buffer_binding_source", live_binding.source, first);
  append_json_string(
      out,
      "live_vulkan_buffer_descriptor_binding",
      live_binding.descriptor_binding,
      first);
  append_json_string(
      out,
      "live_vulkan_buffer_binding_dispatch_position",
      live_binding.first_position == 0u
          ? "missing"
          : std::to_string(live_binding.first_position),
      first);
  append_json_string(
      out,
      "live_vulkan_buffer_handle_token",
      live_binding.handle_token,
      first);
  append_json_string(
      out,
      "live_vulkan_buffer_object_token",
      live_binding.object_token,
      first);
  append_json_string(
      out,
      "live_vulkan_buffer_allocation_label",
      live_binding.allocation_label,
      first);
  append_json_bool(out, "visibility_dependency_validated", false, first);
  append_json_string(
      out, "visibility_dependency_status", visibility_dependency_status, first);
  append_json_bool(out, "no_visibility_dependency_proof", false, first);
  append_json_string(
      out,
      "no_visibility_dependency_proof_status",
      "missing_explicit_no_visibility_dependency_proof",
      first);
  append_json_bool(out, "barrier_canary_ready", false, first);
  append_json_bool(
      out,
      "barrier_canary_candidate_if_behavior_allowed",
      barrier_canary_candidate_if_behavior_allowed,
      first);
  append_json_string(
      out,
      "barrier_canary_reject_reason",
      barrier_canary_candidate_if_behavior_allowed
          ? "rejected_behavior_change_not_allowed"
          : visibility_dependency_status,
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
      plannable ? "none" : barrier_plan_rejection_reason(fields, consumer_position),
      first);
  append_json_string_array(
      out,
      "missing_metadata_fields",
      barrier_plan_missing_dependency_metadata_fields(fields, consumer_position),
      first);
  append_json_comma(out, first);
  out << "\"source_edge_fields\":";
  append_json_fields_object(out, fields);
  out << '}';
}

void append_barrier_plan_json(
    std::ostream& out,
    const std::vector<std::string>& dependency_edges,
    const std::map<std::string, BarrierPlanDispatchPosition>& positions,
    const std::map<std::string, BarrierPlanDispatchPosition>& insertion_points,
    const std::map<std::string, BarrierPlanLiveBufferBinding>& live_bindings,
    const std::map<std::string, uint64_t>& live_allocation_binding_counts,
    bool& first) {
  uint64_t candidate_records = 0u;
  uint64_t plannable_records = 0u;
  uint64_t rejected_records = 0u;
  uint64_t phase_boundary_replace_candidate_records = 0u;
  uint64_t consumer_dispatch_planned_records = 0u;
  uint64_t consumer_dispatch_missing_reduced_records = 0u;
  uint64_t consumer_dispatch_position_known_records = 0u;
  uint64_t consumer_dispatch_position_missing_records = 0u;
  uint64_t planned_consumer_dispatch_position_known_records = 0u;
  uint64_t completed_consumer_dispatch_position_known_records = 0u;
  uint64_t planned_completed_position_agree_records = 0u;
  uint64_t planned_completed_position_different_space_records = 0u;
  uint64_t pre_recording_barrier_insertion_point_available_records = 0u;
  uint64_t pre_recording_barrier_insertion_point_missing_records = 0u;
  uint64_t pre_recording_consumer_dispatch_position_api_missing_records = 0u;
  uint64_t pre_recording_command_buffer_dispatch_position_api_missing_records =
      0u;
  uint64_t stage_access_available_records = 0u;
  uint64_t stage_access_missing_records = 0u;
  uint64_t live_vulkan_buffer_binding_available_records = 0u;
  uint64_t live_vulkan_buffer_binding_missing_records = 0u;
  uint64_t live_vulkan_buffer_binding_range_mismatch_records = 0u;
  uint64_t visibility_dependency_validated_records = 0u;
  uint64_t barrier_canary_ready_records = 0u;
  uint64_t barrier_canary_candidate_if_behavior_allowed_records = 0u;
  std::map<std::string, uint64_t> rejection_reasons;
  std::map<std::string, uint64_t> live_binding_status_counts;
  std::map<std::string, uint64_t> visibility_dependency_status_counts;
  for (const auto& row : dependency_edges) {
    const auto fields = parse_space_separated_fields(row);
    const uint64_t count = parsed_u64(fields, "count");
    const BarrierPlanDispatchPosition consumer_position =
        barrier_plan_consumer_dispatch_position(
            fields, positions, insertion_points);
    const BarrierPlanLiveBufferBinding live_binding =
        barrier_plan_live_buffer_binding(
            fields, live_bindings, live_allocation_binding_counts);
    const bool plannable =
        barrier_plan_record_is_plannable(fields, consumer_position);
    const std::string rejection =
        plannable ? "none" : barrier_plan_rejection_reason(fields, consumer_position);
    const std::string producer_access = field_or(fields, "producer_access", "unknown");
    const std::string consumer_access = field_or(fields, "consumer_access", "unknown");
    const bool stage_access_available =
        barrier_plan_stage_access_available(producer_access, consumer_access);
    const std::string visibility_dependency_status =
        barrier_plan_visibility_dependency_status(
            plannable, stage_access_available, consumer_position, live_binding);
    const auto strict_missing = missing_dependency_metadata_fields(fields);
    const auto plan_missing =
        barrier_plan_missing_dependency_metadata_fields(fields, consumer_position);
    if (stage_access_available) {
      stage_access_available_records += count;
    } else {
      stage_access_missing_records += count;
    }
    if (plannable) {
      if (live_binding.available) {
        live_vulkan_buffer_binding_available_records += count;
        if (stage_access_available && field_or(fields, "queue_submit", "0") == "1") {
          barrier_canary_candidate_if_behavior_allowed_records += count;
        }
      } else if (live_binding.status == "binding_range_mismatch") {
        live_vulkan_buffer_binding_range_mismatch_records += count;
      } else {
        live_vulkan_buffer_binding_missing_records += count;
      }
    }
    live_binding_status_counts[live_binding.status] += count;
    visibility_dependency_status_counts[visibility_dependency_status] += count;
    if (boundary_has_planned_non_capture_norm1_consumer(fields)) {
      consumer_dispatch_planned_records += count;
      const bool strict_missing_consumer =
          std::find(
              strict_missing.begin(),
              strict_missing.end(),
              std::string("consumer_dispatch")) != strict_missing.end();
      const bool plan_missing_consumer =
          std::find(
              plan_missing.begin(),
              plan_missing.end(),
              std::string("consumer_dispatch")) != plan_missing.end();
      const bool plan_missing_position =
          std::find(
              plan_missing.begin(),
              plan_missing.end(),
              std::string("consumer_dispatch_position")) != plan_missing.end();
      const bool plan_missing_pre_recording_api =
          std::find(
              plan_missing.begin(),
              plan_missing.end(),
              std::string("pre_recording_consumer_dispatch_position_api")) !=
          plan_missing.end();
      const bool plan_missing_pre_recording_command_buffer_api =
          std::find(
              plan_missing.begin(),
              plan_missing.end(),
              std::string(
                  "pre_recording_command_buffer_dispatch_position_api")) !=
          plan_missing.end();
      if (strict_missing_consumer && !plan_missing_consumer) {
        consumer_dispatch_missing_reduced_records += count;
      }
      if (
          consumer_position.completed_position_known ||
          consumer_position.planned_position_known) {
        consumer_dispatch_position_known_records += count;
      }
      if (consumer_position.planned_position_known) {
        planned_consumer_dispatch_position_known_records += count;
      }
      if (consumer_position.completed_position_known) {
        completed_consumer_dispatch_position_known_records += count;
      }
      if (
          consumer_position.planned_position_known &&
          consumer_position.completed_position_known &&
          consumer_position.insertion_point_available &&
          consumer_position.insertion_point_first_position ==
              consumer_position.completed_first_position) {
        planned_completed_position_agree_records += count;
      } else if (
          consumer_position.planned_position_known &&
          consumer_position.completed_position_known &&
          !consumer_position.insertion_point_available) {
        planned_completed_position_different_space_records += count;
      }
      if (consumer_position.insertion_point_available) {
        pre_recording_barrier_insertion_point_available_records += count;
      } else {
        pre_recording_barrier_insertion_point_missing_records += count;
      }
      if (plan_missing_position) {
        consumer_dispatch_position_missing_records += count;
      }
      if (plan_missing_pre_recording_api) {
        pre_recording_consumer_dispatch_position_api_missing_records += count;
      }
      if (plan_missing_pre_recording_command_buffer_api) {
        pre_recording_command_buffer_dispatch_position_api_missing_records +=
            count;
      }
    }
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
  append_json_u64(
      out,
      "consumer_dispatch_planned_records",
      consumer_dispatch_planned_records,
      plan_first);
  append_json_u64(
      out,
      "consumer_dispatch_missing_reduced_records",
      consumer_dispatch_missing_reduced_records,
      plan_first);
  append_json_u64(
      out,
      "consumer_dispatch_position_known_records",
      consumer_dispatch_position_known_records,
      plan_first);
  append_json_u64(
      out,
      "consumer_dispatch_position_missing_records",
      consumer_dispatch_position_missing_records,
      plan_first);
  append_json_u64(
      out,
      "planned_consumer_dispatch_position_known_records",
      planned_consumer_dispatch_position_known_records,
      plan_first);
  append_json_u64(
      out,
      "completed_consumer_dispatch_position_known_records",
      completed_consumer_dispatch_position_known_records,
      plan_first);
  append_json_u64(
      out,
      "planned_completed_position_agree_records",
      planned_completed_position_agree_records,
      plan_first);
  append_json_u64(
      out,
      "planned_completed_position_different_space_records",
      planned_completed_position_different_space_records,
      plan_first);
  append_json_u64(
      out,
      "pre_recording_barrier_insertion_point_available_records",
      pre_recording_barrier_insertion_point_available_records,
      plan_first);
  append_json_u64(
      out,
      "pre_recording_barrier_insertion_point_missing_records",
      pre_recording_barrier_insertion_point_missing_records,
      plan_first);
  append_json_u64(
      out,
      "pre_recording_consumer_dispatch_position_api_missing_records",
      pre_recording_consumer_dispatch_position_api_missing_records,
      plan_first);
  append_json_u64(
      out,
      "pre_recording_command_buffer_dispatch_position_api_missing_records",
      pre_recording_command_buffer_dispatch_position_api_missing_records,
      plan_first);
  append_json_u64(
      out,
      "stage_access_available_records",
      stage_access_available_records,
      plan_first);
  append_json_u64(
      out, "stage_access_missing_records", stage_access_missing_records, plan_first);
  append_json_u64(
      out,
      "live_vulkan_buffer_binding_available_records",
      live_vulkan_buffer_binding_available_records,
      plan_first);
  append_json_u64(
      out,
      "live_vulkan_buffer_binding_missing_records",
      live_vulkan_buffer_binding_missing_records,
      plan_first);
  append_json_u64(
      out,
      "live_vulkan_buffer_binding_range_mismatch_records",
      live_vulkan_buffer_binding_range_mismatch_records,
      plan_first);
  append_json_u64(
      out,
      "visibility_dependency_validated_records",
      visibility_dependency_validated_records,
      plan_first);
  append_json_u64(
      out,
      "barrier_canary_ready_records",
      barrier_canary_ready_records,
      plan_first);
  append_json_u64(
      out,
      "barrier_canary_candidate_if_behavior_allowed_records",
      barrier_canary_candidate_if_behavior_allowed_records,
      plan_first);
  append_json_bool(out, "behavior_change_allowed", false, plan_first);
  append_json_string(
      out,
      "behavior_change_veto_reason",
      "rejected_behavior_change_not_allowed",
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
  append_json_comma(out, plan_first);
  out << "\"visibility_dependency_status_counts\":{";
  bool visibility_first = true;
  for (const auto& item : visibility_dependency_status_counts) {
    append_json_u64(out, item.first.c_str(), item.second, visibility_first);
  }
  out << "}";
  append_json_comma(out, plan_first);
  out << "\"live_vulkan_buffer_binding_status_counts\":{";
  bool live_binding_first = true;
  for (const auto& item : live_binding_status_counts) {
    append_json_u64(
        out, item.first.c_str(), item.second, live_binding_first);
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
    append_barrier_plan_record(
        out,
        dependency_edges[i],
        positions,
        insertion_points,
        live_bindings,
        live_allocation_binding_counts,
        i);
  }
  out << "]}";
}

void append_u64_map_object(
    std::ostream& out,
    const std::map<std::string, uint64_t>& values);

std::map<std::string, CaptureAllocationSummary>
build_capture_allocation_summaries(const std::vector<std::string>& rows) {
  std::map<std::string, CaptureAllocationSummary> summaries;
  for (const auto& row : rows) {
    const auto fields = parse_space_separated_fields(row);
    if (field_or(fields, "phase", "unknown") != "intermediate_capture") {
      continue;
    }
    const std::string block = field_or(fields, "block", "unknown");
    auto& summary = summaries[block];
    const uint64_t count = parsed_u64(fields, "count");
    const uint64_t bytes = parsed_u64(fields, "bytes");
    const std::string role = field_or(fields, "role", "unknown");
    if (role == "vision_stack_capture") {
      summary.public_capture_count += count;
      summary.public_capture_bytes += bytes;
      summary.public_capture_shape = field_or(fields, "shape", "missing");
    } else if (role == "vision_stack_private_device_capture") {
      summary.private_bridge_capture_count += count;
      summary.private_bridge_capture_bytes += bytes;
      summary.private_bridge_capture_shape = field_or(fields, "shape", "missing");
    }
  }
  return summaries;
}

std::map<std::string, StackOutputDeviceConsumerRegistrationSummary>
build_stack_output_device_consumer_registration_summaries(
    const std::vector<std::string>& rows) {
  std::map<std::string, StackOutputDeviceConsumerRegistrationSummary> summaries;
  for (const auto& row : rows) {
    const auto fields = parse_space_separated_fields(row);
    if (
        field_or(fields, "stack_output_device_consumer_registration", "0") !=
        "1") {
      continue;
    }
    auto& summary = summaries[stack_output_device_consumer_registration_key(
        fields)];
    summary.count += parsed_u64(fields, "count");
    summary.consumer_in_same_planned_region =
        summary.consumer_in_same_planned_region ||
        field_or(fields, "consumer_in_same_planned_region", "0") == "1";
    summary.python_public_boundary_before_consumption =
        summary.python_public_boundary_before_consumption ||
        field_or(fields, "python_public_boundary_before_consumption", "1") ==
            "1";
    summary.host_visible_boundary_before_consumption =
        summary.host_visible_boundary_before_consumption ||
        field_or(fields, "host_visible_boundary_before_consumption", "1") ==
            "1";
    summary.host_visible_access_before_consumption =
        summary.host_visible_access_before_consumption ||
        field_or(fields, "host_visible_access_before_consumption", "1") ==
            "1";
    summary.host_readback_before_consumption =
        summary.host_readback_before_consumption ||
        field_or(fields, "host_readback_before_consumption", "1") == "1";
    summary.stack_context_id = field_or(fields, "stack_context_id", "missing");
    summary.stack_session_id = field_or(fields, "stack_session_id", "missing");
    summary.stack_plan_id = field_or(fields, "stack_plan_id", "missing");
    summary.captured_substep =
        field_or(fields, "captured_substep", "missing");
    summary.output_role = field_or(fields, "output_role", "missing");
    summary.output_shape = field_or(fields, "output_shape", "missing");
    summary.output_layout = field_or(fields, "output_layout", "missing");
    summary.strip_or_view_relation =
        field_or(fields, "strip_or_view_relation", "missing");
    summary.downstream_consumer_id =
        field_or(fields, "downstream_consumer_id", "missing");
    summary.downstream_consumer_context =
        field_or(fields, "downstream_consumer_context", "missing");
    summary.expected_consumer_input_index =
        field_or(fields, "expected_consumer_input_index", "missing");
    summary.expected_consumer_shape =
        field_or(fields, "expected_consumer_shape", "missing");
    summary.expected_consumer_layout =
        field_or(fields, "expected_consumer_layout", "missing");
  }
  return summaries;
}

CaptureAllocationSummary capture_scope_summary(
    const CaptureAllocationSummary& summary,
    const CaptureOutputBoundaryScope scope) {
  if (scope == CaptureOutputBoundaryScope::Combined) {
    return summary;
  }
  CaptureAllocationSummary scoped;
  if (scope == CaptureOutputBoundaryScope::PublicCapture) {
    scoped.public_capture_count = summary.public_capture_count;
    scoped.public_capture_bytes = summary.public_capture_bytes;
    scoped.public_capture_shape = summary.public_capture_shape;
  } else if (scope == CaptureOutputBoundaryScope::BridgePrivateCapture) {
    scoped.private_bridge_capture_count = summary.private_bridge_capture_count;
    scoped.private_bridge_capture_bytes = summary.private_bridge_capture_bytes;
    scoped.private_bridge_capture_shape = summary.private_bridge_capture_shape;
  }
  return scoped;
}

const char* capture_storage_class_name(
    const CaptureAllocationSummary& summary) {
  if (
      summary.public_capture_count > 0u &&
      summary.private_bridge_capture_count > 0u) {
    return "mixed_public_and_private_capture_observed";
  }
  if (summary.private_bridge_capture_count > 0u) {
    return "bridge_private_internal_capture";
  }
  if (summary.public_capture_count > 0u) {
    return "public_tensor_array_capture";
  }
  return "unknown_capture_storage";
}

bool capture_consumer_registration_accepts_same_region(
    const StackOutputDeviceConsumerRegistrationSummary* const registration) {
  return registration && registration->count > 0u &&
      registration->consumer_in_same_planned_region &&
      !registration->python_public_boundary_before_consumption &&
      !registration->host_visible_boundary_before_consumption &&
      !registration->host_visible_access_before_consumption &&
      !registration->host_readback_before_consumption;
}

const char* capture_consumer_registration_reason(
    const StackOutputDeviceConsumerRegistrationSummary* const registration) {
  if (!registration || registration->count == 0u) {
    return "downstream_device_consumer_registration_missing";
  }
  if (!registration->consumer_in_same_planned_region) {
    return "downstream_device_consumer_not_in_same_planned_region";
  }
  if (registration->python_public_boundary_before_consumption) {
    return "python_public_boundary_before_downstream_consumer";
  }
  if (registration->host_visible_boundary_before_consumption) {
    return "host_visible_boundary_before_downstream_consumer";
  }
  if (registration->host_visible_access_before_consumption) {
    return "host_visible_access_before_downstream_consumer";
  }
  if (registration->host_readback_before_consumption) {
    return "host_readback_before_downstream_consumer";
  }
  return "same_region_device_consumer_registered";
}

bool capture_scope_fields_complete(
    const CaptureAllocationSummary& summary,
    const StackOutputDeviceConsumerRegistrationSummary* const registration,
    const bool allocation_generation_proven,
    const bool allocation_range_proven) {
  return capture_consumer_registration_accepts_same_region(registration) &&
      summary.private_bridge_capture_count > 0u &&
      summary.public_capture_count == 0u && allocation_generation_proven &&
      allocation_range_proven;
}

const char* capture_boundary_sync_required_reason(
    const CaptureAllocationSummary& summary,
    const StackOutputDeviceConsumerRegistrationSummary* const registration,
    const bool capture_scope_fields_complete) {
  if (summary.public_capture_count > 0u) {
    return "public_tensor_array_capture_requires_boundary_submit";
  }
  if (summary.private_bridge_capture_count > 0u) {
    if (!capture_consumer_registration_accepts_same_region(registration)) {
      return capture_consumer_registration_reason(registration);
    }
    if (capture_scope_fields_complete) {
      return "capture_scope_complete_boundary_dependency_set_required";
    }
    return "bridge_private_capture_needs_value_preservation_and_complete_boundary_proof";
  }
  return "capture_storage_mode_unknown";
}

void capture_output_missing_proof_fields(
    const CaptureAllocationSummary& summary,
    const StackOutputDeviceConsumerRegistrationSummary* const registration,
    const bool capture_scope_fields_complete,
    std::vector<std::string>& missing) {
  if (
      summary.public_capture_count > 0u &&
      summary.private_bridge_capture_count > 0u) {
    missing.emplace_back("scope_split_for_public_vs_private_capture_observations");
  }
  if (summary.public_capture_count > 0u) {
    missing.emplace_back("public_tensor_array_boundary_elision_contract");
  }
  if (summary.private_bridge_capture_count > 0u) {
    if (!capture_consumer_registration_accepts_same_region(registration)) {
      missing.emplace_back("downstream_consumer_registration_in_stack_graph");
    }
    if (!capture_scope_fields_complete) {
      missing.emplace_back("capture_value_preservation_proof");
    }
  }
  if (
      summary.public_capture_count == 0u &&
      summary.private_bridge_capture_count == 0u) {
    missing.emplace_back("capture_storage_mode");
  }
  missing.emplace_back("complete_boundary_dependency_set");
}

void append_capture_output_boundary_record(
    std::ostream& out,
    const std::string& row,
    const std::map<std::string, CaptureAllocationSummary>& summaries,
    const std::map<std::string, StackOutputDeviceConsumerRegistrationSummary>&
        registrations,
    const CaptureOutputBoundaryScope scope,
    const size_t index) {
  const auto fields = parse_space_separated_fields(row);
  const std::string capture_block = field_or(fields, "consumer_block", "unknown");
  const auto summary_it = summaries.find(capture_block);
  const CaptureAllocationSummary empty_summary;
  const CaptureAllocationSummary& raw_summary =
      summary_it == summaries.end() ? empty_summary : summary_it->second;
  const CaptureAllocationSummary summary = capture_scope_summary(raw_summary, scope);
  const std::string registration_key =
      stack_output_device_consumer_registration_key(
          capture_block, field_or(fields, "role", "unknown"));
  const auto registration_it = registrations.find(registration_key);
  const StackOutputDeviceConsumerRegistrationSummary* const registration =
      registration_it == registrations.end() ? nullptr : &registration_it->second;
  const bool same_region_consumer_registered =
      capture_consumer_registration_accepts_same_region(registration);
  const bool allocation_generation_proven =
      field_or(fields, "allocation_has_generation", "0") == "1";
  const bool allocation_range_proven =
      field_or(fields, "allocation_has_byte_range", "0") == "1";
  const bool capture_specific_proof_complete = capture_scope_fields_complete(
      summary, registration, allocation_generation_proven, allocation_range_proven);
  std::vector<std::string> missing_proof_fields;
  capture_output_missing_proof_fields(
      summary,
      registration,
      capture_specific_proof_complete,
      missing_proof_fields);
  bool first = true;
  out << '{';
  append_json_string(
      out,
      "record_id",
      std::string(capture_output_boundary_record_prefix(scope)) +
          std::to_string(index),
      first);
  append_json_string(out, "contract", "CaptureOutputBoundaryContract", first);
  append_json_string(
      out, "capture_scope", capture_output_boundary_scope_name(scope), first);
  append_json_string(
      out, "producer_block", field_or(fields, "producer_block", "unknown"), first);
  append_json_string(
      out,
      "producer_substep",
      field_or(fields, "producer_phase", "unknown"),
      first);
  append_json_string(
      out, "producer_role", field_or(fields, "role", "unknown"), first);
  append_json_string(out, "capture_block", capture_block, first);
  append_json_string(out, "capture_index", capture_block, first);
  append_json_string(
      out,
      "capture_substep",
      field_or(fields, "consumer_phase", "unknown"),
      first);
  append_json_string(
      out,
      "capture_output_role",
      field_or(fields, "consumer_descriptor_role", "unknown"),
      first);
  append_json_string(
      out,
      "allocation_id",
      field_or(fields, "allocation_id", "unknown"),
      first);
  append_json_string(
      out,
      "allocation_generation",
      field_or(fields, "allocation_generation", "unknown"),
      first);
  append_json_string(
      out, "byte_offset", field_or(fields, "byte_offset", "unknown"), first);
  append_json_string(
      out, "byte_range", field_or(fields, "byte_range", "unknown"), first);
  append_json_string(out, "bytes", field_or(fields, "bytes", "unknown"), first);
  append_json_bool(
      out,
      "allocation_generation_proven",
      allocation_generation_proven,
      first);
  append_json_bool(
      out,
      "allocation_range_proven",
      allocation_range_proven,
      first);
  append_json_bool(out, "requested_intermediate", true, first);
  append_json_bool(
      out,
      "public_tensor_array_capture_observed",
      summary.public_capture_count > 0u,
      first);
  append_json_bool(
      out,
      "private_bridge_internal_capture_observed",
      summary.private_bridge_capture_count > 0u,
      first);
  append_json_string(
      out, "capture_storage_class", capture_storage_class_name(summary), first);
  append_json_u64(
      out, "public_capture_observation_count", summary.public_capture_count, first);
  append_json_u64(
      out,
      "private_bridge_capture_observation_count",
      summary.private_bridge_capture_count,
      first);
  append_json_string(
      out, "public_capture_shape", summary.public_capture_shape, first);
  append_json_string(
      out,
      "private_bridge_capture_shape",
      summary.private_bridge_capture_shape,
      first);
  append_json_bool(out, "final_output", false, first);
  append_json_bool(
      out,
      "host_visible_or_requested_output",
      field_or(fields, "resource_class", "unknown") ==
          "host_visible_or_requested_output",
      first);
  append_json_bool(
      out,
      "same_region_consumer_registered",
      same_region_consumer_registered,
      first);
  append_json_string(
      out,
      "consumer_registration_accept_reject_reason",
      capture_consumer_registration_reason(registration),
      first);
  append_json_u64(
      out,
      "downstream_consumer_registration_count",
      registration ? registration->count : 0u,
      first);
  append_json_string(
      out,
      "consumer_context_id",
      registration ? registration->downstream_consumer_context : "missing",
      first);
  append_json_string(
      out,
      "consumer_id",
      registration ? registration->downstream_consumer_id : "missing",
      first);
  append_json_string(
      out,
      "consumer_expected_input_index",
      registration ? registration->expected_consumer_input_index : "missing",
      first);
  append_json_string(
      out,
      "consumer_expected_shape",
      registration ? registration->expected_consumer_shape : "missing",
      first);
  append_json_string(
      out,
      "consumer_expected_layout",
      registration ? registration->expected_consumer_layout : "missing",
      first);
  append_json_string(
      out,
      "stack_context_id",
      registration ? registration->stack_context_id : "missing",
      first);
  append_json_string(
      out,
      "stack_session_id",
      registration ? registration->stack_session_id : "missing",
      first);
  append_json_string(
      out,
      "stack_plan_id",
      registration ? registration->stack_plan_id : "missing",
      first);
  append_json_string(
      out,
      "capture_output_layout",
      registration ? registration->output_layout : "missing",
      first);
  append_json_string(
      out,
      "strip_or_view_relation",
      registration ? registration->strip_or_view_relation : "missing",
      first);
  append_json_bool(
      out,
      "consumer_in_same_planned_region",
      registration && registration->consumer_in_same_planned_region,
      first);
  append_json_bool(
      out,
      "python_public_boundary_before_consumption",
      !registration || registration->python_public_boundary_before_consumption,
      first);
  append_json_bool(
      out,
      "host_visible_boundary_before_consumption",
      !registration || registration->host_visible_boundary_before_consumption,
      first);
  append_json_bool(
      out,
      "host_visible_access_before_consumption",
      !registration || registration->host_visible_access_before_consumption,
      first);
  append_json_bool(
      out,
      "host_readback_before_consumption",
      !registration || registration->host_readback_before_consumption,
      first);
  append_json_bool(
      out, "capture_specific_proof_complete", capture_specific_proof_complete, first);
  append_json_string(
      out,
      "boundary_sync_required_reason",
      capture_boundary_sync_required_reason(
          summary, registration, capture_specific_proof_complete),
      first);
  append_json_string_array(
      out, "missing_capture_boundary_proof_fields", missing_proof_fields, first);
  append_json_comma(out, first);
  out << "\"source_edge_fields\":";
  append_json_fields_object(out, fields);
  out << '}';
}

void append_capture_output_boundary_contract_json(
    std::ostream& out,
    const std::vector<std::string>& capture_edges,
    const std::map<std::string, CaptureAllocationSummary>& summaries,
    const std::map<std::string, StackOutputDeviceConsumerRegistrationSummary>&
        registrations,
    bool& first) {
  uint64_t candidate_records = 0u;
  uint64_t proof_complete_records = 0u;
  uint64_t public_tensor_array_records = 0u;
  uint64_t bridge_private_records = 0u;
  uint64_t mixed_capture_records = 0u;
  uint64_t public_capture_records = 0u;
  uint64_t bridge_private_capture_records = 0u;
  uint64_t mixed_scope_rejected_records = 0u;
  uint64_t bridge_private_proof_complete_records = 0u;
  uint64_t unknown_capture_storage_records = 0u;
  uint64_t requested_intermediate_records = 0u;
  uint64_t consumer_registration_records = 0u;
  uint64_t same_region_consumer_registered_records = 0u;
  uint64_t consumer_registration_missing_records = 0u;
  uint64_t public_boundary_rejected_records = 0u;
  uint64_t host_visible_rejected_records = 0u;
  std::map<std::string, uint64_t> boundary_sync_required_reasons;
  for (const auto& row : capture_edges) {
    const auto fields = parse_space_separated_fields(row);
    const uint64_t count = parsed_u64(fields, "count");
    const std::string capture_block = field_or(fields, "consumer_block", "unknown");
    const auto summary_it = summaries.find(capture_block);
    const CaptureAllocationSummary empty_summary;
    const CaptureAllocationSummary& summary =
        summary_it == summaries.end() ? empty_summary : summary_it->second;
    const auto registration_it = registrations.find(
        stack_output_device_consumer_registration_key(
            capture_block, field_or(fields, "role", "unknown")));
    const StackOutputDeviceConsumerRegistrationSummary* const registration =
        registration_it == registrations.end() ? nullptr : &registration_it->second;
    const bool same_region_consumer_registered =
        capture_consumer_registration_accepts_same_region(registration);
    const bool allocation_generation_proven =
        field_or(fields, "allocation_has_generation", "0") == "1";
    const bool allocation_range_proven =
        field_or(fields, "allocation_has_byte_range", "0") == "1";
    const bool capture_specific_proof_complete = capture_scope_fields_complete(
        summary, registration, allocation_generation_proven, allocation_range_proven);
    const CaptureAllocationSummary bridge_private_summary =
        capture_scope_summary(summary, CaptureOutputBoundaryScope::BridgePrivateCapture);
    const bool bridge_private_proof_complete = capture_scope_fields_complete(
        bridge_private_summary,
        registration,
        allocation_generation_proven,
        allocation_range_proven);
    candidate_records += count;
    requested_intermediate_records += count;
    if (registration && registration->count > 0u) {
      consumer_registration_records += count;
    } else {
      consumer_registration_missing_records += count;
    }
    if (same_region_consumer_registered) {
      same_region_consumer_registered_records += count;
    }
    if (
        registration &&
        (registration->host_visible_boundary_before_consumption ||
         registration->host_visible_access_before_consumption ||
         registration->host_readback_before_consumption)) {
      host_visible_rejected_records += count;
    }
    if (summary.public_capture_count > 0u) {
      public_tensor_array_records += count;
      public_capture_records += count;
      public_boundary_rejected_records += count;
    }
    if (summary.private_bridge_capture_count > 0u) {
      bridge_private_records += count;
      bridge_private_capture_records += count;
    }
    if (
        summary.public_capture_count > 0u &&
        summary.private_bridge_capture_count > 0u) {
      mixed_capture_records += count;
      mixed_scope_rejected_records += count;
    }
    if (
        summary.public_capture_count == 0u &&
        summary.private_bridge_capture_count == 0u) {
      unknown_capture_storage_records += count;
    }
    boundary_sync_required_reasons[capture_boundary_sync_required_reason(
        summary, registration, capture_specific_proof_complete)] += count;
    if (capture_specific_proof_complete) {
      proof_complete_records += count;
    }
    if (bridge_private_proof_complete) {
      bridge_private_proof_complete_records += count;
    }
  }

  append_json_comma(out, first);
  out << "\"capture_output_boundary_contract\":{";
  bool contract_first = true;
  append_json_string(
      out, "schema", "CaptureOutputBoundaryContract.v0", contract_first);
  append_json_bool(out, "behavior_neutral", true, contract_first);
  append_json_bool(out, "dry_run_only", true, contract_first);
  append_json_u64(out, "candidate_records", candidate_records, contract_first);
  append_json_u64(
      out, "requested_intermediate_records", requested_intermediate_records, contract_first);
  append_json_u64(
      out,
      "consumer_registration_records",
      consumer_registration_records,
      contract_first);
  append_json_u64(
      out,
      "same_region_consumer_registered_records",
      same_region_consumer_registered_records,
      contract_first);
  append_json_u64(
      out,
      "consumer_registration_missing_records",
      consumer_registration_missing_records,
      contract_first);
  append_json_u64(
      out,
      "public_boundary_rejected_records",
      public_boundary_rejected_records,
      contract_first);
  append_json_u64(
      out,
      "host_visible_rejected_records",
      host_visible_rejected_records,
      contract_first);
  append_json_u64(
      out, "public_tensor_array_records", public_tensor_array_records, contract_first);
  append_json_u64(
      out, "bridge_private_records", bridge_private_records, contract_first);
  append_json_u64(out, "mixed_capture_records", mixed_capture_records, contract_first);
  append_json_u64(
      out, "public_capture_records", public_capture_records, contract_first);
  append_json_u64(
      out,
      "bridge_private_capture_records",
      bridge_private_capture_records,
      contract_first);
  append_json_u64(
      out,
      "mixed_scope_rejected_records",
      mixed_scope_rejected_records,
      contract_first);
  append_json_u64(
      out,
      "unknown_capture_storage_records",
      unknown_capture_storage_records,
      contract_first);
  append_json_u64(
      out, "proof_complete_records", proof_complete_records, contract_first);
  append_json_u64(
      out,
      "bridge_private_proof_complete_records",
      bridge_private_proof_complete_records,
      contract_first);
  append_json_u64(out, "barriers_inserted", 0u, contract_first);
  append_json_u64(out, "submits_removed", 0u, contract_first);
  append_json_comma(out, contract_first);
  out << "\"boundary_sync_required_reasons\":";
  append_u64_map_object(out, boundary_sync_required_reasons);
  append_json_string_array(
      out,
      "proof_complete_blockers",
      {"public_tensor_array_boundary_elision_contract",
       "downstream_consumer_registration_in_stack_graph",
       "capture_value_preservation_proof",
       "complete_boundary_dependency_set"},
      contract_first);
  append_json_comma(out, contract_first);
  out << "\"records\":[";
  for (size_t i = 0; i < capture_edges.size(); ++i) {
    if (i > 0) {
      out << ',';
    }
    append_capture_output_boundary_record(
        out,
        capture_edges[i],
        summaries,
        registrations,
        CaptureOutputBoundaryScope::Combined,
        i);
  }
  out << "]";
  append_json_comma(out, contract_first);
  out << "\"public_capture_scope_records\":[";
  bool public_first = true;
  for (size_t i = 0; i < capture_edges.size(); ++i) {
    const auto fields = parse_space_separated_fields(capture_edges[i]);
    const auto summary_it =
        summaries.find(field_or(fields, "consumer_block", "unknown"));
    if (
        summary_it == summaries.end() ||
        summary_it->second.public_capture_count == 0u) {
      continue;
    }
    if (!public_first) {
      out << ',';
    }
    public_first = false;
    append_capture_output_boundary_record(
        out,
        capture_edges[i],
        summaries,
        registrations,
        CaptureOutputBoundaryScope::PublicCapture,
        i);
  }
  out << "]";
  append_json_comma(out, contract_first);
  out << "\"bridge_private_capture_scope_records\":[";
  bool private_first = true;
  for (size_t i = 0; i < capture_edges.size(); ++i) {
    const auto fields = parse_space_separated_fields(capture_edges[i]);
    const auto summary_it =
        summaries.find(field_or(fields, "consumer_block", "unknown"));
    if (
        summary_it == summaries.end() ||
        summary_it->second.private_bridge_capture_count == 0u) {
      continue;
    }
    if (!private_first) {
      out << ',';
    }
    private_first = false;
    append_capture_output_boundary_record(
        out,
        capture_edges[i],
        summaries,
        registrations,
        CaptureOutputBoundaryScope::BridgePrivateCapture,
        i);
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
  uint64_t formal_last_use_planned_records = 0u;
  uint64_t formal_last_use_missing_reduced_records = 0u;
  std::map<std::string, uint64_t> edge_rejection_reasons;
  std::map<std::string, uint64_t> missing_fields;
  std::map<std::string, uint64_t> consumer_dispatch_proofs;
  std::map<std::string, uint64_t> formal_last_use_proofs;
  std::map<std::string, BoundaryResourceClassSummary> retire_only_resources;
  std::map<std::string, BoundaryResourceClassSummary> ordering_required_resources;
  std::map<std::string, BoundaryResourceClassSummary> public_blockers;
  std::map<std::string, uint64_t> boundary_reject_reasons;
  std::vector<std::string> boundary_rows;
};

struct CaptureBoundaryDependencySetProof final {
  std::string boundary_id;
  std::string capture_block;
  uint64_t required_capture_edge_records = 0u;
  uint64_t required_capture_edge_bytes = 0u;
  uint64_t combined_capture_proof_complete_records = 0u;
  uint64_t public_capture_records = 0u;
  uint64_t public_capture_proof_complete_records = 0u;
  uint64_t bridge_private_capture_records = 0u;
  uint64_t bridge_private_capture_proof_complete_records = 0u;
  uint64_t mixed_scope_rejected_records = 0u;
  uint64_t queue_submit_records = 0u;
  uint64_t stack_activation_capture_candidate_records = 0u;
  uint64_t stack_activation_capture_proof_complete_records = 0u;
  uint64_t stack_activation_capture_public_rejected_records = 0u;
  uint64_t pending_bytes_before_proof_classification = 0u;
  uint64_t pending_bytes_after_proof_classification = 0u;
  uint64_t ordering_required_bytes_after_proof = 0u;
  uint64_t retire_only_bytes_after_proof = 0u;
  uint64_t proof_classified_capture_activation_bytes = 0u;
  uint64_t peak_extra_live_bytes_estimate = 0u;
  uint64_t block_budget_bytes = 0u;
  uint64_t scope_budget_bytes = 0u;
  bool recomputed_block_budget_ok = false;
  bool recomputed_scope_budget_ok = false;
  bool recomputed_bridge_private_boundary_complete = false;
  std::string recomputed_incomplete_reason = "not_recomputed";
  std::map<std::string, uint64_t> boundary_reject_reasons;
  std::map<std::string, uint64_t> stack_activation_capture_reject_reasons;
  std::map<std::string, BoundaryResourceClassSummary> boundary_resources;
  std::map<std::string, BoundaryResourceClassSummary>
      stack_activation_capture_before_blockers;
  std::map<std::string, BoundaryResourceClassSummary>
      stack_activation_capture_after_blockers;
  std::map<std::string, BoundaryResourceClassSummary>
      recomputed_retire_only_resources;
  std::map<std::string, BoundaryResourceClassSummary>
      recomputed_ordering_required_resources;
  std::map<std::string, BoundaryResourceClassSummary>
      recomputed_public_host_final_requested_blockers;
  std::map<std::string, BoundaryResourceClassSummary>
      recomputed_proof_classified_resources;
  std::map<std::string, BoundaryResourceClassSummary> remaining_full_boundary_blockers;
  std::vector<std::string> stack_activation_capture_edge_rows;
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

std::string capture_boundary_dependency_set_key(
    const std::map<std::string, std::string>& fields) {
  std::ostringstream stream;
  stream << "capture_boundary:producer_block="
         << field_or(fields, "producer_block", "unknown")
         << ":capture_block=" << field_or(fields, "consumer_block", "unknown");
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

bool capture_boundary_dependency_set_bridge_private_complete(
    const CaptureBoundaryDependencySetProof& proof) {
  return proof.required_capture_edge_records > 0u &&
      proof.bridge_private_capture_proof_complete_records ==
      proof.required_capture_edge_records && !proof.boundary_rows.empty();
}

bool capture_boundary_dependency_set_combined_complete(
    const CaptureBoundaryDependencySetProof& proof) {
  return proof.required_capture_edge_records > 0u &&
      proof.combined_capture_proof_complete_records ==
      proof.required_capture_edge_records && !proof.boundary_rows.empty() &&
      proof.mixed_scope_rejected_records == 0u && proof.public_capture_records == 0u;
}

bool stack_activation_capture_proof_complete(
    const CaptureBoundaryDependencySetProof& proof) {
  return proof.required_capture_edge_records > 0u &&
      proof.stack_activation_capture_proof_complete_records ==
      proof.required_capture_edge_records &&
      capture_boundary_dependency_set_bridge_private_complete(proof);
}

bool field_is_true(
    const std::map<std::string, std::string>& fields,
    const char* key) {
  return field_or(fields, key, "0") == "1";
}

std::vector<std::string> missing_stack_activation_capture_proof_fields(
    const std::map<std::string, std::string>& fields,
    const CaptureAllocationSummary& bridge_private_summary,
    const StackOutputDeviceConsumerRegistrationSummary* const registration,
    const bool capture_dependency_set_member,
    const bool bridge_private_dependency_set_complete) {
  std::vector<std::string> missing;
  const auto missing_if_not_true = [&fields, &missing](const char* key) {
    if (!field_is_true(fields, key)) {
      missing.emplace_back(key);
    }
  };
  if (bridge_private_summary.private_bridge_capture_count == 0u) {
    missing.emplace_back("bridge_private_capture_scope");
  }
  if (!capture_consumer_registration_accepts_same_region(registration)) {
    missing.emplace_back("same_region_device_consumer_registration");
  }
  if (!capture_dependency_set_member) {
    missing.emplace_back("capture_boundary_dependency_set_membership");
  }
  if (!bridge_private_dependency_set_complete) {
    missing.emplace_back("bridge_private_capture_dependency_set_complete");
  }
  if (field_or(fields, "producer_phase", "unknown") != "residual2") {
    missing.emplace_back("producer_residual2_substep");
  }
  if (field_or(fields, "role", "unknown") != "stack_residual2_output") {
    missing.emplace_back("stack_residual2_output_role");
  }
  if (field_or(fields, "consumer_phase", "unknown") != "intermediate_capture") {
    missing.emplace_back("intermediate_capture_consumer");
  }
  if (field_or(fields, "producer_block", "unknown") !=
      field_or(fields, "consumer_block", "unknown")) {
    missing.emplace_back("same_block_capture_relation");
  }
  if (field_or(fields, "scope_id", "0") == "0") {
    missing.emplace_back("stack_owner_scope_id");
  }
  if (!field_is_true(fields, "stack_provenance_defined")) {
    missing.emplace_back("stack_provenance_defined");
  }
  missing_if_not_true("allocation_has_generation");
  missing_if_not_true("allocation_has_byte_range");
  if (parsed_u64(fields, "byte_range") == 0u) {
    missing.emplace_back("nonzero_byte_range");
  }
  missing_if_not_true("producer_dispatch_observed");
  missing_if_not_true("producer_live_range_known");
  missing_if_not_true("consumer_live_range_known");
  missing_if_not_true("descriptor_binding_known");
  missing_if_not_true("direct_buffer");
  missing_if_not_true("buffer_storage");
  if (field_is_true(fields, "image_storage")) {
    missing.emplace_back("non_image_storage");
  }
  if (field_is_true(fields, "final_output")) {
    missing.emplace_back("not_final_output");
  }
  if (field_is_true(fields, "alias_or_view")) {
    missing.emplace_back("no_alias_or_view");
  }
  if (field_is_true(fields, "aliases_runtime_input")) {
    missing.emplace_back("no_runtime_input_alias");
  }
  if (field_is_true(fields, "aliases_runtime_output")) {
    missing.emplace_back("no_runtime_output_alias");
  }
  if (
      registration &&
      (registration->python_public_boundary_before_consumption ||
       registration->host_visible_boundary_before_consumption ||
       registration->host_visible_access_before_consumption ||
       registration->host_readback_before_consumption)) {
    missing.emplace_back("no_public_or_host_visible_boundary");
  }
  return missing;
}

std::string stack_activation_capture_reject_reason(
    const std::vector<std::string>& missing) {
  if (missing.empty()) {
    return "complete";
  }
  return "missing_" + missing.front();
}

void append_capture_boundary_dependency_set_record(
    std::ostream& out,
    const CaptureBoundaryDependencySetProof& proof) {
  const bool bridge_private_complete =
      capture_boundary_dependency_set_bridge_private_complete(proof);
  const bool combined_complete =
      capture_boundary_dependency_set_combined_complete(proof);
  const bool activation_capture_complete =
      stack_activation_capture_proof_complete(proof);
  bool first = true;
  out << '{';
  append_json_string(out, "boundary_id", proof.boundary_id, first);
  append_json_string(out, "capture_block", proof.capture_block, first);
  append_json_string(out, "boundary_phase", "intermediate_capture", first);
  append_json_u64(
      out,
      "required_capture_edge_records",
      proof.required_capture_edge_records,
      first);
  append_json_u64(
      out,
      "required_capture_edge_bytes",
      proof.required_capture_edge_bytes,
      first);
  append_json_u64(
      out,
      "combined_capture_proof_complete_records",
      proof.combined_capture_proof_complete_records,
      first);
  append_json_u64(
      out, "public_capture_records", proof.public_capture_records, first);
  append_json_u64(
      out,
      "public_capture_proof_complete_records",
      proof.public_capture_proof_complete_records,
      first);
  append_json_u64(
      out,
      "bridge_private_capture_records",
      proof.bridge_private_capture_records,
      first);
  append_json_u64(
      out,
      "bridge_private_capture_proof_complete_records",
      proof.bridge_private_capture_proof_complete_records,
      first);
  append_json_u64(
      out,
      "mixed_scope_rejected_records",
      proof.mixed_scope_rejected_records,
      first);
  append_json_u64(out, "queue_submit_records", proof.queue_submit_records, first);
  append_json_bool(
      out, "combined_capture_dependency_set_complete", combined_complete, first);
  append_json_bool(out, "public_capture_dependency_set_complete", false, first);
  append_json_bool(
      out,
      "bridge_private_capture_dependency_set_complete",
      bridge_private_complete,
      first);
  append_json_bool(
      out,
      "stack_activation_capture_proof_complete",
      activation_capture_complete,
      first);
  append_json_u64(
      out,
      "stack_activation_capture_candidate_records",
      proof.stack_activation_capture_candidate_records,
      first);
  append_json_u64(
      out,
      "stack_activation_capture_proof_complete_records",
      proof.stack_activation_capture_proof_complete_records,
      first);
  append_json_u64(
      out,
      "stack_activation_capture_public_rejected_records",
      proof.stack_activation_capture_public_rejected_records,
      first);
  append_json_u64(
      out,
      "pending_bytes_before_proof_classification",
      proof.pending_bytes_before_proof_classification,
      first);
  append_json_u64(
      out,
      "pending_bytes_after_proof_classification",
      proof.pending_bytes_after_proof_classification,
      first);
  append_json_u64(
      out,
      "ordering_required_bytes_after_proof",
      proof.ordering_required_bytes_after_proof,
      first);
  append_json_u64(
      out,
      "retire_only_bytes_after_proof",
      proof.retire_only_bytes_after_proof,
      first);
  append_json_u64(
      out,
      "proof_classified_capture_activation_bytes",
      proof.proof_classified_capture_activation_bytes,
      first);
  append_json_u64(
      out,
      "peak_extra_live_bytes_estimate",
      proof.peak_extra_live_bytes_estimate,
      first);
  append_json_u64(out, "block_budget_bytes", proof.block_budget_bytes, first);
  append_json_u64(out, "scope_budget_bytes", proof.scope_budget_bytes, first);
  append_json_bool(
      out, "recomputed_block_budget_ok", proof.recomputed_block_budget_ok, first);
  append_json_bool(
      out, "recomputed_scope_budget_ok", proof.recomputed_scope_budget_ok, first);
  append_json_bool(
      out,
      "recomputed_bridge_private_boundary_complete",
      proof.recomputed_bridge_private_boundary_complete,
      first);
  append_json_string(
      out,
      "recomputed_incomplete_reason",
      proof.recomputed_incomplete_reason,
      first);
  append_json_bool(
      out,
      "full_boundary_complete",
      proof.recomputed_bridge_private_boundary_complete,
      first);
  append_json_bool(out, "behavior_change_allowed", false, first);
  append_json_comma(out, first);
  out << "\"boundary_resources\":";
  append_resource_class_summary_object(out, proof.boundary_resources);
  append_json_comma(out, first);
  out << "\"stack_activation_capture_before_blockers\":";
  append_resource_class_summary_object(
      out, proof.stack_activation_capture_before_blockers);
  append_json_comma(out, first);
  out << "\"stack_activation_capture_after_blockers\":";
  append_resource_class_summary_object(
      out, proof.stack_activation_capture_after_blockers);
  append_json_comma(out, first);
  out << "\"recomputed_retire_only_resources\":";
  append_resource_class_summary_object(out, proof.recomputed_retire_only_resources);
  append_json_comma(out, first);
  out << "\"recomputed_ordering_required_resources\":";
  append_resource_class_summary_object(
      out, proof.recomputed_ordering_required_resources);
  append_json_comma(out, first);
  out << "\"recomputed_public_host_final_requested_blockers\":";
  append_resource_class_summary_object(
      out, proof.recomputed_public_host_final_requested_blockers);
  append_json_comma(out, first);
  out << "\"recomputed_proof_classified_resources\":";
  append_resource_class_summary_object(
      out, proof.recomputed_proof_classified_resources);
  append_json_comma(out, first);
  out << "\"remaining_full_boundary_blockers\":";
  append_resource_class_summary_object(out, proof.remaining_full_boundary_blockers);
  append_json_comma(out, first);
  out << "\"stack_activation_capture_reject_reasons\":";
  append_u64_map_object(out, proof.stack_activation_capture_reject_reasons);
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

void append_stack_activation_capture_edge_record(
    std::ostream& out,
    const std::string& row,
    const CaptureBoundaryDependencySetProof& proof,
    const std::map<std::string, CaptureAllocationSummary>& summaries,
    const std::map<std::string, StackOutputDeviceConsumerRegistrationSummary>&
        registrations,
    const size_t index) {
  const auto fields = parse_space_separated_fields(row);
  const std::string capture_block = field_or(fields, "consumer_block", "unknown");
  const auto summary_it = summaries.find(capture_block);
  const CaptureAllocationSummary empty_summary;
  const CaptureAllocationSummary& raw_summary =
      summary_it == summaries.end() ? empty_summary : summary_it->second;
  const CaptureAllocationSummary bridge_private_summary =
      capture_scope_summary(raw_summary, CaptureOutputBoundaryScope::BridgePrivateCapture);
  const auto registration_it = registrations.find(
      stack_output_device_consumer_registration_key(
          capture_block, field_or(fields, "role", "unknown")));
  const StackOutputDeviceConsumerRegistrationSummary* const registration =
      registration_it == registrations.end() ? nullptr : &registration_it->second;
  const std::vector<std::string> missing =
      missing_stack_activation_capture_proof_fields(
          fields,
          bridge_private_summary,
          registration,
          !proof.boundary_rows.empty(),
          capture_boundary_dependency_set_bridge_private_complete(proof));
  const bool proof_complete = missing.empty();
  bool first = true;
  out << '{';
  append_json_string(
      out,
      "record_id",
      "stack_activation_capture_edge_" + std::to_string(index),
      first);
  append_json_string(out, "contract", "StackActivationCaptureProof", first);
  append_json_string(
      out, "capture_scope", "bridge_private_capture", first);
  append_json_string(
      out, "boundary_id", proof.boundary_id, first);
  append_json_string(
      out, "producer_block", field_or(fields, "producer_block", "unknown"), first);
  append_json_string(
      out,
      "producer_substep",
      field_or(fields, "producer_phase", "unknown"),
      first);
  append_json_string(
      out, "producer_role", field_or(fields, "role", "unknown"), first);
  append_json_string(out, "capture_block", capture_block, first);
  append_json_string(
      out,
      "capture_substep",
      field_or(fields, "consumer_phase", "unknown"),
      first);
  append_json_string(
      out,
      "capture_output_role",
      field_or(fields, "consumer_descriptor_role", "unknown"),
      first);
  append_json_string(out, "stack_owner_scope_id", field_or(fields, "scope_id", "0"), first);
  append_json_string(
      out,
      "stack_context_id",
      registration ? registration->stack_context_id : "missing",
      first);
  append_json_string(
      out,
      "stack_session_id",
      registration ? registration->stack_session_id : "missing",
      first);
  append_json_string(
      out,
      "stack_plan_id",
      registration ? registration->stack_plan_id : "missing",
      first);
  append_json_string(
      out,
      "allocation_id",
      field_or(fields, "allocation_id", "unknown"),
      first);
  append_json_string(
      out,
      "allocation_generation",
      field_or(fields, "allocation_generation", "unknown"),
      first);
  append_json_string(
      out, "byte_offset", field_or(fields, "byte_offset", "unknown"), first);
  append_json_string(
      out, "byte_range", field_or(fields, "byte_range", "unknown"), first);
  append_json_string(out, "bytes", field_or(fields, "bytes", "unknown"), first);
  append_json_bool(
      out,
      "allocation_generation_proven",
      field_is_true(fields, "allocation_has_generation"),
      first);
  append_json_bool(
      out,
      "allocation_range_proven",
      field_is_true(fields, "allocation_has_byte_range"),
      first);
  append_json_bool(
      out,
      "same_region_consumer_registered",
      capture_consumer_registration_accepts_same_region(registration),
      first);
  append_json_string(
      out,
      "consumer_registration_accept_reject_reason",
      capture_consumer_registration_reason(registration),
      first);
  append_json_bool(
      out,
      "capture_boundary_dependency_set_member",
      !proof.boundary_rows.empty(),
      first);
  append_json_bool(
      out,
      "bridge_private_capture_dependency_set_complete",
      capture_boundary_dependency_set_bridge_private_complete(proof),
      first);
  append_json_bool(
      out,
      "public_tensor_array_capture_observed",
      bridge_private_summary.public_capture_count > 0u,
      first);
  append_json_bool(
      out,
      "private_bridge_internal_capture_observed",
      bridge_private_summary.private_bridge_capture_count > 0u,
      first);
  append_json_bool(
      out,
      "python_public_boundary_before_consumption",
      !registration || registration->python_public_boundary_before_consumption,
      first);
  append_json_bool(
      out,
      "host_visible_boundary_before_consumption",
      !registration || registration->host_visible_boundary_before_consumption,
      first);
  append_json_bool(
      out,
      "host_visible_access_before_consumption",
      !registration || registration->host_visible_access_before_consumption,
      first);
  append_json_bool(
      out,
      "host_readback_before_consumption",
      !registration || registration->host_readback_before_consumption,
      first);
  append_json_bool(
      out,
      "stack_provenance_defined",
      field_is_true(fields, "stack_provenance_defined"),
      first);
  append_json_string(
      out, "stack_lifetime", field_or(fields, "stack_lifetime", "missing"), first);
  append_json_bool(out, "direct_buffer", field_is_true(fields, "direct_buffer"), first);
  append_json_bool(out, "buffer_storage", field_is_true(fields, "buffer_storage"), first);
  append_json_bool(out, "image_storage", field_is_true(fields, "image_storage"), first);
  append_json_bool(out, "source_escapes_stack", field_is_true(fields, "escapes_stack"), first);
  append_json_bool(out, "requested_intermediate", field_is_true(fields, "requested_intermediate"), first);
  append_json_bool(out, "final_output", field_is_true(fields, "final_output"), first);
  append_json_bool(out, "alias_or_view", field_is_true(fields, "alias_or_view"), first);
  append_json_bool(
      out,
      "aliases_runtime_input",
      field_is_true(fields, "aliases_runtime_input"),
      first);
  append_json_bool(
      out,
      "aliases_runtime_output",
      field_is_true(fields, "aliases_runtime_output"),
      first);
  append_json_bool(out, "proof_complete", proof_complete, first);
  append_json_string(
      out,
      "accept_reject_reason",
      stack_activation_capture_reject_reason(missing),
      first);
  append_json_string_array(out, "missing_proof_fields", missing, first);
  append_json_comma(out, first);
  out << "\"source_edge_fields\":";
  append_json_fields_object(out, fields);
  out << '}';
}

void append_stack_activation_capture_proof_json(
    std::ostream& out,
    const std::map<std::string, CaptureBoundaryDependencySetProof>& proofs,
    const std::map<std::string, CaptureAllocationSummary>& summaries,
    const std::map<std::string, StackOutputDeviceConsumerRegistrationSummary>&
        registrations,
    const uint64_t missing_stack_activation_before_records,
    const uint64_t missing_stack_activation_after_records,
    const uint64_t capture_sensitive_before_records,
    const uint64_t capture_sensitive_after_records,
    const uint64_t unsafe_resource_class_before_records,
    const uint64_t unsafe_resource_class_after_records,
    bool& first) {
  uint64_t candidate_records = 0u;
  uint64_t bridge_private_records = 0u;
  uint64_t proof_complete_records = 0u;
  uint64_t public_scope_rejected_records = 0u;
  uint64_t capture_dependency_member_records = 0u;
  std::map<std::string, uint64_t> reject_reasons;
  for (const auto& item : proofs) {
    const auto& proof = item.second;
    candidate_records += proof.stack_activation_capture_candidate_records;
    bridge_private_records += proof.bridge_private_capture_records;
    proof_complete_records +=
        proof.stack_activation_capture_proof_complete_records;
    public_scope_rejected_records +=
        proof.stack_activation_capture_public_rejected_records;
    if (!proof.boundary_rows.empty()) {
      capture_dependency_member_records +=
          proof.stack_activation_capture_candidate_records;
    }
    for (const auto& reason : proof.stack_activation_capture_reject_reasons) {
      reject_reasons[reason.first] += reason.second;
    }
  }

  append_json_comma(out, first);
  out << "\"stack_activation_capture_proof\":{";
  bool proof_first = true;
  append_json_string(
      out, "schema", "StackActivationCaptureProof.v0", proof_first);
  append_json_bool(out, "behavior_neutral", true, proof_first);
  append_json_bool(out, "dry_run_only", true, proof_first);
  append_json_string(
      out,
      "target_boundary_class",
      "bridge_private_capture_residual2_activation",
      proof_first);
  append_json_u64(out, "candidate_records", candidate_records, proof_first);
  append_json_u64(
      out, "bridge_private_capture_records", bridge_private_records, proof_first);
  append_json_u64(
      out,
      "capture_dependency_set_member_records",
      capture_dependency_member_records,
      proof_first);
  append_json_u64(
      out, "proof_complete_records", proof_complete_records, proof_first);
  append_json_u64(
      out,
      "public_scope_rejected_records",
      public_scope_rejected_records,
      proof_first);
  append_json_u64(
      out,
      "missing_stack_activation_proof_before_records",
      missing_stack_activation_before_records,
      proof_first);
  append_json_u64(
      out,
      "missing_stack_activation_proof_after_records",
      missing_stack_activation_after_records,
      proof_first);
  append_json_u64(
      out,
      "capture_sensitive_stack_activation_before_records",
      capture_sensitive_before_records,
      proof_first);
  append_json_u64(
      out,
      "capture_sensitive_stack_activation_after_records",
      capture_sensitive_after_records,
      proof_first);
  append_json_u64(
      out,
      "unsafe_resource_class_before_records",
      unsafe_resource_class_before_records,
      proof_first);
  append_json_u64(
      out,
      "unsafe_resource_class_after_records",
      unsafe_resource_class_after_records,
      proof_first);
  append_json_u64(out, "barriers_inserted", 0u, proof_first);
  append_json_u64(out, "submits_removed", 0u, proof_first);
  append_json_comma(out, proof_first);
  out << "\"reject_reasons\":";
  append_u64_map_object(out, reject_reasons);
  append_json_comma(out, proof_first);
  out << "\"records\":[";
  bool first_record = true;
  size_t index = 0u;
  for (const auto& item : proofs) {
    for (const auto& row : item.second.stack_activation_capture_edge_rows) {
      if (!first_record) {
        out << ',';
      }
      first_record = false;
      append_stack_activation_capture_edge_record(
          out, row, item.second, summaries, registrations, index++);
    }
  }
  out << "]}";
}

void append_phase_boundary_budget_recompute_record(
    std::ostream& out,
    const CaptureBoundaryDependencySetProof& proof) {
  bool first = true;
  out << '{';
  append_json_string(out, "boundary_id", proof.boundary_id, first);
  append_json_string(out, "capture_block", proof.capture_block, first);
  append_json_string(
      out, "capture_scope", "bridge_private_capture", first);
  append_json_u64(
      out,
      "pending_bytes_before_proof_classification",
      proof.pending_bytes_before_proof_classification,
      first);
  append_json_u64(
      out,
      "pending_bytes_after_proof_classification",
      proof.pending_bytes_after_proof_classification,
      first);
  append_json_u64(
      out,
      "ordering_required_bytes_after_proof",
      proof.ordering_required_bytes_after_proof,
      first);
  append_json_u64(
      out,
      "retire_only_bytes_after_proof",
      proof.retire_only_bytes_after_proof,
      first);
  append_json_u64(
      out,
      "proof_classified_capture_activation_bytes",
      proof.proof_classified_capture_activation_bytes,
      first);
  append_json_u64(
      out,
      "peak_extra_live_bytes_estimate",
      proof.peak_extra_live_bytes_estimate,
      first);
  append_json_u64(out, "block_budget_bytes", proof.block_budget_bytes, first);
  append_json_bool(
      out, "block_budget_ok", proof.recomputed_block_budget_ok, first);
  append_json_u64(out, "scope_budget_bytes", proof.scope_budget_bytes, first);
  append_json_bool(
      out, "scope_budget_ok", proof.recomputed_scope_budget_ok, first);
  append_json_bool(
      out,
      "bridge_private_capture_dependency_set_complete",
      capture_boundary_dependency_set_bridge_private_complete(proof),
      first);
  append_json_bool(
      out,
      "stack_activation_capture_proof_complete",
      stack_activation_capture_proof_complete(proof),
      first);
  append_json_bool(
      out,
      "recomputed_bridge_private_boundary_complete",
      proof.recomputed_bridge_private_boundary_complete,
      first);
  append_json_string(
      out,
      "complete_or_incomplete_reason",
      proof.recomputed_incomplete_reason,
      first);
  append_json_bool(out, "behavior_change_allowed", false, first);
  append_json_bool(out, "canary_ready", false, first);
  append_json_string(
      out,
      "submit_skip_hard_veto_reason",
      "rejected_behavior_change_not_allowed",
      first);
  append_json_bool(
      out,
      "requires_barrier_or_no_visibility_dependency_proof",
      true,
      first);
  append_json_bool(out, "real_barrier_records_inserted", false, first);
  append_json_bool(out, "no_visibility_dependency_proof", false, first);
  append_json_string(
      out,
      "visibility_dependency_proof_status",
      "missing_live_visibility_or_no_dependency_proof",
      first);
  append_json_comma(out, first);
  out << "\"retire_only_resources_after_proof\":";
  append_resource_class_summary_object(out, proof.recomputed_retire_only_resources);
  append_json_comma(out, first);
  out << "\"ordering_required_resources_after_proof\":";
  append_resource_class_summary_object(
      out, proof.recomputed_ordering_required_resources);
  append_json_comma(out, first);
  out << "\"public_host_final_requested_blockers_after_proof\":";
  append_resource_class_summary_object(
      out, proof.recomputed_public_host_final_requested_blockers);
  append_json_comma(out, first);
  out << "\"proof_classified_resources\":";
  append_resource_class_summary_object(
      out, proof.recomputed_proof_classified_resources);
  out << '}';
}

void append_phase_boundary_budget_recompute_json(
    std::ostream& out,
    const std::map<std::string, CaptureBoundaryDependencySetProof>& proofs,
    bool& first) {
  uint64_t candidate_boundaries = 0u;
  uint64_t recomputed_complete_boundaries = 0u;
  uint64_t public_combined_rejected_records = 0u;
  uint64_t pending_bytes_before = 0u;
  uint64_t pending_bytes_after = 0u;
  uint64_t ordering_required_bytes_after = 0u;
  uint64_t retire_only_bytes_after = 0u;
  uint64_t proof_classified_bytes = 0u;
  uint64_t block_budget_ok_boundaries = 0u;
  uint64_t scope_budget_ok_boundaries = 0u;
  std::map<std::string, uint64_t> incomplete_reasons;
  for (const auto& item : proofs) {
    const auto& proof = item.second;
    ++candidate_boundaries;
    pending_bytes_before += proof.pending_bytes_before_proof_classification;
    pending_bytes_after += proof.pending_bytes_after_proof_classification;
    ordering_required_bytes_after += proof.ordering_required_bytes_after_proof;
    retire_only_bytes_after += proof.retire_only_bytes_after_proof;
    proof_classified_bytes += proof.proof_classified_capture_activation_bytes;
    public_combined_rejected_records += proof.public_capture_records;
    if (proof.recomputed_block_budget_ok) {
      ++block_budget_ok_boundaries;
    }
    if (proof.recomputed_scope_budget_ok) {
      ++scope_budget_ok_boundaries;
    }
    if (proof.recomputed_bridge_private_boundary_complete) {
      ++recomputed_complete_boundaries;
    } else {
      incomplete_reasons[proof.recomputed_incomplete_reason] += 1u;
    }
  }

  append_json_comma(out, first);
  out << "\"phase_boundary_budget_recompute\":{";
  bool recompute_first = true;
  append_json_string(
      out, "schema", "PhaseBoundaryBudgetRecompute.v0", recompute_first);
  append_json_bool(out, "behavior_neutral", true, recompute_first);
  append_json_bool(out, "dry_run_only", true, recompute_first);
  append_json_string(
      out,
      "target_boundary_class",
      "bridge_private_intermediate_capture",
      recompute_first);
  append_json_u64(
      out, "candidate_boundaries", candidate_boundaries, recompute_first);
  append_json_u64(
      out,
      "recomputed_bridge_private_complete_boundaries",
      recomputed_complete_boundaries,
      recompute_first);
  append_json_u64(
      out,
      "public_combined_scope_rejected_records",
      public_combined_rejected_records,
      recompute_first);
  append_json_u64(
      out,
      "pending_bytes_before_proof_classification",
      pending_bytes_before,
      recompute_first);
  append_json_u64(
      out,
      "pending_bytes_after_proof_classification",
      pending_bytes_after,
      recompute_first);
  append_json_u64(
      out,
      "ordering_required_bytes_after_proof",
      ordering_required_bytes_after,
      recompute_first);
  append_json_u64(
      out,
      "retire_only_bytes_after_proof",
      retire_only_bytes_after,
      recompute_first);
  append_json_u64(
      out,
      "proof_classified_capture_activation_bytes",
      proof_classified_bytes,
      recompute_first);
  append_json_u64(
      out,
      "block_budget_ok_boundaries",
      block_budget_ok_boundaries,
      recompute_first);
  append_json_u64(
      out,
      "scope_budget_ok_boundaries",
      scope_budget_ok_boundaries,
      recompute_first);
  append_json_u64(out, "barriers_inserted", 0u, recompute_first);
  append_json_u64(out, "submits_removed", 0u, recompute_first);
  append_json_bool(out, "behavior_change_allowed", false, recompute_first);
  append_json_bool(
      out,
      "canary_ready",
      false,
      recompute_first);
  append_json_string(
      out,
      "canary_blocked_reason",
      "rejected_behavior_change_not_allowed",
      recompute_first);
  append_json_bool(
      out,
      "requires_barrier_or_no_visibility_dependency_proof",
      true,
      recompute_first);
  append_json_bool(
      out, "no_visibility_dependency_proof", false, recompute_first);
  append_json_u64(
      out,
      "dry_run_complete_boundaries",
      recomputed_complete_boundaries,
      recompute_first);
  append_json_comma(out, recompute_first);
  out << "\"incomplete_reasons\":";
  append_u64_map_object(out, incomplete_reasons);
  append_json_comma(out, recompute_first);
  out << "\"records\":[";
  bool first_record = true;
  for (const auto& item : proofs) {
    if (!first_record) {
      out << ',';
    }
    first_record = false;
    append_phase_boundary_budget_recompute_record(out, item.second);
  }
  out << "]}";
}

void append_capture_boundary_dependency_set_json(
    std::ostream& out,
    const std::vector<std::string>& capture_edges,
    const std::vector<std::string>& boundary_nodes,
    const std::map<std::string, CaptureAllocationSummary>& summaries,
    const std::map<std::string, StackOutputDeviceConsumerRegistrationSummary>&
        registrations,
    bool& first) {
  std::map<std::string, CaptureBoundaryDependencySetProof> proofs;
  for (const auto& row : capture_edges) {
    const auto fields = parse_space_separated_fields(row);
    const std::string key = capture_boundary_dependency_set_key(fields);
    auto& proof = proofs[key];
    proof.boundary_id = key;
    proof.capture_block = field_or(fields, "consumer_block", "unknown");
    const uint64_t count = parsed_u64(fields, "count");
    proof.required_capture_edge_records += count;
    proof.required_capture_edge_bytes += parsed_u64(fields, "bytes");
    proof.queue_submit_records += parsed_u64(fields, "queue_submit");

    const auto summary_it = summaries.find(proof.capture_block);
    const CaptureAllocationSummary empty_summary;
    const CaptureAllocationSummary& summary =
        summary_it == summaries.end() ? empty_summary : summary_it->second;
    const auto registration_it = registrations.find(
        stack_output_device_consumer_registration_key(
            proof.capture_block, field_or(fields, "role", "unknown")));
    const StackOutputDeviceConsumerRegistrationSummary* const registration =
        registration_it == registrations.end() ? nullptr : &registration_it->second;
    const bool allocation_generation_proven =
        field_or(fields, "allocation_has_generation", "0") == "1";
    const bool allocation_range_proven =
        field_or(fields, "allocation_has_byte_range", "0") == "1";
    if (summary.public_capture_count > 0u) {
      proof.public_capture_records += count;
      proof.stack_activation_capture_public_rejected_records += count;
    }
    if (summary.private_bridge_capture_count > 0u) {
      proof.bridge_private_capture_records += count;
      proof.stack_activation_capture_candidate_records += count;
      proof.stack_activation_capture_edge_rows.emplace_back(row);
    }
    if (
        summary.public_capture_count > 0u &&
        summary.private_bridge_capture_count > 0u) {
      proof.mixed_scope_rejected_records += count;
    }
    if (capture_scope_fields_complete(
            summary,
            registration,
            allocation_generation_proven,
            allocation_range_proven)) {
      proof.combined_capture_proof_complete_records += count;
    }
    const CaptureAllocationSummary public_summary =
        capture_scope_summary(summary, CaptureOutputBoundaryScope::PublicCapture);
    if (capture_scope_fields_complete(
            public_summary,
            registration,
            allocation_generation_proven,
            allocation_range_proven)) {
      proof.public_capture_proof_complete_records += count;
    }
    const CaptureAllocationSummary bridge_private_summary =
        capture_scope_summary(summary, CaptureOutputBoundaryScope::BridgePrivateCapture);
    if (capture_scope_fields_complete(
            bridge_private_summary,
            registration,
            allocation_generation_proven,
            allocation_range_proven)) {
      proof.bridge_private_capture_proof_complete_records += count;
    }
  }

  for (const auto& row : boundary_nodes) {
    const auto fields = parse_space_separated_fields(row);
    if (field_or(fields, "boundary_stack_phase", "unknown") != "block_entry") {
      continue;
    }
    const std::string boundary_block =
        field_or(fields, "boundary_block", "unknown");
    for (auto& item : proofs) {
      auto& proof = item.second;
      if (proof.capture_block != boundary_block) {
        continue;
      }
      proof.boundary_rows.emplace_back(row);
      const uint64_t boundary_count =
          std::max<uint64_t>(parsed_u64(fields, "count"), 1u);
      const auto signature = fields.find("signature");
      if (signature != fields.end()) {
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
          if (first_hash == std::string::npos ||
              second_hash == std::string::npos) {
            proof.boundary_reject_reasons["malformed_boundary_signature"] +=
                boundary_count;
            continue;
          }
          const std::string resource_class = token.substr(0, first_hash);
          uint64_t resource_count = 0u;
          uint64_t resource_bytes = 0u;
          try {
            resource_count = static_cast<uint64_t>(std::stoull(token.substr(
                first_hash + 1u, second_hash - first_hash - 1u)));
            resource_bytes = static_cast<uint64_t>(
                std::stoull(token.substr(second_hash + 1u)));
          } catch (...) {
            proof.boundary_reject_reasons["malformed_boundary_signature"] +=
                boundary_count;
            continue;
          }
          resource_count *= boundary_count;
          resource_bytes *= boundary_count;
          add_boundary_resource_class(
              proof.boundary_resources,
              resource_class,
              resource_count,
              resource_bytes);
          const bool activation_capture_complete =
              stack_activation_capture_proof_complete(proof);
          if (
              resource_class == kDryRunMissingStackActivationProof ||
              resource_class == kDryRunCaptureSensitiveStackActivation) {
            add_boundary_resource_class(
                proof.stack_activation_capture_before_blockers,
                resource_class,
                resource_count,
                resource_bytes);
            if (!activation_capture_complete) {
              add_boundary_resource_class(
                  proof.stack_activation_capture_after_blockers,
                  resource_class,
                  resource_count,
                  resource_bytes);
            }
          }
          if (
              !signature_resource_class_is_retire_only(resource_class) &&
              (resource_class != kDryRunMissingStackActivationProof ||
               !activation_capture_complete) &&
              (resource_class != kDryRunCaptureSensitiveStackActivation ||
               !activation_capture_complete)) {
            add_boundary_resource_class(
                proof.remaining_full_boundary_blockers,
                resource_class,
                resource_count,
                resource_bytes);
          }
        }
      } else {
        proof.boundary_reject_reasons["missing_boundary_signature"] +=
            boundary_count;
      }
      const std::string budget_reject =
          field_or(fields, "budget_reject", "missing_budget_reject");
      if (budget_reject != "none") {
        proof.boundary_reject_reasons["budget_reject:" + budget_reject] +=
            parsed_u64(fields, "count");
        if (
            budget_reject == "unsafe_resource_class" &&
            stack_activation_capture_proof_complete(proof)) {
          const uint64_t reject_count = parsed_u64(fields, "count");
          add_boundary_resource_class(
              proof.stack_activation_capture_before_blockers,
              "unsafe_resource_class",
              reject_count,
              parsed_u64(fields, "bytes"));
          add_boundary_resource_class(
              proof.stack_activation_capture_after_blockers,
              "phase_boundary_budget_recompute_required",
              reject_count,
              parsed_u64(fields, "bytes"));
        }
      }
      const std::string blockers = field_or(fields, "blockers", "none");
      if (blockers != "none") {
        proof.boundary_reject_reasons["blockers:" + blockers] +=
            parsed_u64(fields, "count");
      }
    }
  }

  for (auto& item : proofs) {
    auto& proof = item.second;
    proof.stack_activation_capture_proof_complete_records = 0u;
    proof.stack_activation_capture_reject_reasons.clear();
    for (const auto& row : proof.stack_activation_capture_edge_rows) {
      const auto fields = parse_space_separated_fields(row);
      const uint64_t count = parsed_u64(fields, "count");
      const std::string capture_block =
          field_or(fields, "consumer_block", "unknown");
      const auto summary_it = summaries.find(capture_block);
      const CaptureAllocationSummary empty_summary;
      const CaptureAllocationSummary& raw_summary =
          summary_it == summaries.end() ? empty_summary : summary_it->second;
      const CaptureAllocationSummary bridge_private_summary =
          capture_scope_summary(
              raw_summary, CaptureOutputBoundaryScope::BridgePrivateCapture);
      const auto registration_it = registrations.find(
          stack_output_device_consumer_registration_key(
              capture_block, field_or(fields, "role", "unknown")));
      const StackOutputDeviceConsumerRegistrationSummary* const registration =
          registration_it == registrations.end() ? nullptr : &registration_it->second;
      const std::vector<std::string> activation_missing_fields =
          missing_stack_activation_capture_proof_fields(
              fields,
              bridge_private_summary,
              registration,
              !proof.boundary_rows.empty(),
              capture_boundary_dependency_set_bridge_private_complete(proof));
      const std::string activation_reason =
          stack_activation_capture_reject_reason(activation_missing_fields);
      if (activation_reason == "complete") {
        proof.stack_activation_capture_proof_complete_records += count;
      } else {
        proof.stack_activation_capture_reject_reasons[activation_reason] += count;
      }
    }
  }

  for (auto& item : proofs) {
    auto& proof = item.second;
    proof.boundary_reject_reasons.clear();
    proof.boundary_resources.clear();
    proof.stack_activation_capture_before_blockers.clear();
    proof.stack_activation_capture_after_blockers.clear();
    proof.recomputed_retire_only_resources.clear();
    proof.recomputed_ordering_required_resources.clear();
    proof.recomputed_public_host_final_requested_blockers.clear();
    proof.recomputed_proof_classified_resources.clear();
    proof.remaining_full_boundary_blockers.clear();
    proof.pending_bytes_before_proof_classification = 0u;
    proof.pending_bytes_after_proof_classification = 0u;
    proof.ordering_required_bytes_after_proof = 0u;
    proof.retire_only_bytes_after_proof = 0u;
    proof.proof_classified_capture_activation_bytes = 0u;
    proof.peak_extra_live_bytes_estimate = 0u;
    proof.block_budget_bytes = 0u;
    proof.scope_budget_bytes = 0u;
    proof.recomputed_block_budget_ok = false;
    proof.recomputed_scope_budget_ok = false;
    proof.recomputed_bridge_private_boundary_complete = false;
    proof.recomputed_incomplete_reason = "not_recomputed";
    for (const auto& row : proof.boundary_rows) {
      const auto fields = parse_space_separated_fields(row);
      const uint64_t boundary_count =
          std::max<uint64_t>(parsed_u64(fields, "count"), 1u);
      proof.pending_bytes_before_proof_classification +=
          parsed_u64(fields, "bytes");
      proof.pending_bytes_after_proof_classification +=
          parsed_u64(fields, "bytes");
      proof.peak_extra_live_bytes_estimate = std::max<uint64_t>(
          proof.peak_extra_live_bytes_estimate,
          parsed_u64(fields, "peak_extra_live_bytes_estimate"));
      proof.block_budget_bytes = std::max<uint64_t>(
          proof.block_budget_bytes, parsed_u64(fields, "block_budget_bytes"));
      proof.scope_budget_bytes = std::max<uint64_t>(
          proof.scope_budget_bytes, parsed_u64(fields, "scope_budget_bytes"));
      const auto signature = fields.find("signature");
      if (signature != fields.end()) {
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
          if (first_hash == std::string::npos ||
              second_hash == std::string::npos) {
            proof.boundary_reject_reasons["malformed_boundary_signature"] +=
                boundary_count;
            continue;
          }
          const std::string resource_class = token.substr(0, first_hash);
          uint64_t resource_count = 0u;
          uint64_t resource_bytes = 0u;
          try {
            resource_count = static_cast<uint64_t>(std::stoull(token.substr(
                first_hash + 1u, second_hash - first_hash - 1u)));
            resource_bytes = static_cast<uint64_t>(
                std::stoull(token.substr(second_hash + 1u)));
          } catch (...) {
            proof.boundary_reject_reasons["malformed_boundary_signature"] +=
                boundary_count;
            continue;
          }
          resource_count *= boundary_count;
          resource_bytes *= boundary_count;
          add_boundary_resource_class(
              proof.boundary_resources,
              resource_class,
              resource_count,
              resource_bytes);
          const bool activation_capture_complete =
              stack_activation_capture_proof_complete(proof);
          if (signature_resource_class_is_retire_only(resource_class)) {
            add_boundary_resource_class(
                proof.recomputed_retire_only_resources,
                resource_class,
                resource_count,
                resource_bytes);
          } else if (
              activation_capture_complete &&
              (resource_class == kDryRunMissingStackActivationProof ||
               resource_class == kDryRunCaptureSensitiveStackActivation)) {
            add_boundary_resource_class(
                proof.recomputed_proof_classified_resources,
                resource_class,
                resource_count,
                resource_bytes);
          } else {
            add_boundary_resource_class(
                proof.recomputed_ordering_required_resources,
                resource_class,
                resource_count,
                resource_bytes);
            if (signature_resource_class_is_public_blocker(resource_class)) {
              add_boundary_resource_class(
                  proof.recomputed_public_host_final_requested_blockers,
                  resource_class,
                  resource_count,
                  resource_bytes);
            }
          }
          if (
              resource_class == kDryRunMissingStackActivationProof ||
              resource_class == kDryRunCaptureSensitiveStackActivation) {
            add_boundary_resource_class(
                proof.stack_activation_capture_before_blockers,
                resource_class,
                resource_count,
                resource_bytes);
            if (!activation_capture_complete) {
              add_boundary_resource_class(
                  proof.stack_activation_capture_after_blockers,
                  resource_class,
                  resource_count,
                  resource_bytes);
            }
          }
          if (
              !signature_resource_class_is_retire_only(resource_class) &&
              (resource_class != kDryRunMissingStackActivationProof ||
               !activation_capture_complete) &&
              (resource_class != kDryRunCaptureSensitiveStackActivation ||
               !activation_capture_complete)) {
            add_boundary_resource_class(
                proof.remaining_full_boundary_blockers,
                resource_class,
                resource_count,
                resource_bytes);
          }
        }
      } else {
        proof.boundary_reject_reasons["missing_boundary_signature"] +=
            boundary_count;
      }
      const std::string budget_reject =
          field_or(fields, "budget_reject", "missing_budget_reject");
      if (budget_reject != "none") {
        proof.boundary_reject_reasons["budget_reject:" + budget_reject] +=
            parsed_u64(fields, "count");
        if (
            budget_reject == "unsafe_resource_class" &&
            stack_activation_capture_proof_complete(proof)) {
          const uint64_t reject_count = parsed_u64(fields, "count");
          add_boundary_resource_class(
              proof.stack_activation_capture_before_blockers,
              "unsafe_resource_class",
              reject_count,
              parsed_u64(fields, "bytes"));
          add_boundary_resource_class(
              proof.stack_activation_capture_after_blockers,
              "phase_boundary_budget_recompute_required",
              reject_count,
              parsed_u64(fields, "bytes"));
        }
      }
      const std::string blockers = field_or(fields, "blockers", "none");
      if (blockers != "none") {
        proof.boundary_reject_reasons["blockers:" + blockers] +=
            parsed_u64(fields, "count");
      }
    }
    for (const auto& item : proof.recomputed_retire_only_resources) {
      proof.retire_only_bytes_after_proof += item.second.bytes;
    }
    for (const auto& item : proof.recomputed_ordering_required_resources) {
      proof.ordering_required_bytes_after_proof += item.second.bytes;
    }
    for (const auto& item : proof.recomputed_proof_classified_resources) {
      proof.proof_classified_capture_activation_bytes += item.second.bytes;
    }
    proof.recomputed_block_budget_ok =
        proof.block_budget_bytes == 0u ||
        proof.peak_extra_live_bytes_estimate <= proof.block_budget_bytes;
    proof.recomputed_scope_budget_ok =
        proof.scope_budget_bytes == 0u ||
        proof.peak_extra_live_bytes_estimate <= proof.scope_budget_bytes;
    if (proof.boundary_rows.empty()) {
      proof.recomputed_incomplete_reason = "missing_phase_boundary_rows";
    } else if (!capture_boundary_dependency_set_bridge_private_complete(proof)) {
      proof.recomputed_incomplete_reason =
          "bridge_private_capture_dependency_set_incomplete";
    } else if (!stack_activation_capture_proof_complete(proof)) {
      proof.recomputed_incomplete_reason =
          "stack_activation_capture_proof_incomplete";
    } else if (!proof.recomputed_public_host_final_requested_blockers.empty()) {
      proof.recomputed_incomplete_reason =
          "public_host_final_requested_blocker_after_recompute";
    } else if (!proof.recomputed_ordering_required_resources.empty()) {
      proof.recomputed_incomplete_reason =
          "ordering_required_resource_after_recompute";
    } else if (!proof.recomputed_block_budget_ok) {
      proof.recomputed_incomplete_reason = "block_budget_exceeded";
    } else if (!proof.recomputed_scope_budget_ok) {
      proof.recomputed_incomplete_reason = "scope_budget_exceeded";
    } else {
      proof.recomputed_bridge_private_boundary_complete = true;
      proof.recomputed_incomplete_reason = "none";
    }
  }

  uint64_t candidate_boundaries = 0u;
  uint64_t combined_complete_boundaries = 0u;
  uint64_t public_complete_boundaries = 0u;
  uint64_t bridge_private_complete_boundaries = 0u;
  uint64_t full_boundary_complete_boundaries = 0u;
  uint64_t required_capture_edge_records = 0u;
  uint64_t bridge_private_complete_records = 0u;
  std::map<std::string, uint64_t> remaining_blockers;
  uint64_t missing_stack_activation_before_records = 0u;
  uint64_t missing_stack_activation_after_records = 0u;
  uint64_t capture_sensitive_before_records = 0u;
  uint64_t capture_sensitive_after_records = 0u;
  uint64_t unsafe_resource_class_before_records = 0u;
  uint64_t unsafe_resource_class_after_records = 0u;
  for (const auto& item : proofs) {
    const auto& proof = item.second;
    ++candidate_boundaries;
    required_capture_edge_records += proof.required_capture_edge_records;
    bridge_private_complete_records +=
        proof.bridge_private_capture_proof_complete_records;
    if (capture_boundary_dependency_set_combined_complete(proof)) {
      ++combined_complete_boundaries;
    }
    if (
        proof.public_capture_proof_complete_records ==
            proof.required_capture_edge_records &&
        proof.required_capture_edge_records > 0u) {
      ++public_complete_boundaries;
    }
    if (capture_boundary_dependency_set_bridge_private_complete(proof)) {
      ++bridge_private_complete_boundaries;
    }
    if (proof.recomputed_bridge_private_boundary_complete) {
      ++full_boundary_complete_boundaries;
    }
    for (const auto& resource : proof.remaining_full_boundary_blockers) {
      remaining_blockers[resource.first] += resource.second.count;
    }
    for (const auto& reason : proof.boundary_reject_reasons) {
      if (
          reason.first == "blockers:capture_sensitive_stack_activation" &&
          stack_activation_capture_proof_complete(proof)) {
        continue;
      }
      if (
          reason.first == "budget_reject:unsafe_resource_class" &&
          stack_activation_capture_proof_complete(proof)) {
        if (!proof.recomputed_bridge_private_boundary_complete) {
          remaining_blockers
              ["boundary:budget_reject:" + proof.recomputed_incomplete_reason] +=
              reason.second;
        }
        continue;
      }
      remaining_blockers["boundary:" + reason.first] += reason.second;
    }
    const auto before_missing =
        proof.stack_activation_capture_before_blockers.find(
            kDryRunMissingStackActivationProof);
    if (before_missing != proof.stack_activation_capture_before_blockers.end()) {
      missing_stack_activation_before_records += before_missing->second.count;
    }
    const auto after_missing =
        proof.stack_activation_capture_after_blockers.find(
            kDryRunMissingStackActivationProof);
    if (after_missing != proof.stack_activation_capture_after_blockers.end()) {
      missing_stack_activation_after_records += after_missing->second.count;
    }
    const auto before_capture =
        proof.stack_activation_capture_before_blockers.find(
            kDryRunCaptureSensitiveStackActivation);
    if (before_capture != proof.stack_activation_capture_before_blockers.end()) {
      capture_sensitive_before_records += before_capture->second.count;
    }
    const auto after_capture =
        proof.stack_activation_capture_after_blockers.find(
            kDryRunCaptureSensitiveStackActivation);
    if (after_capture != proof.stack_activation_capture_after_blockers.end()) {
      capture_sensitive_after_records += after_capture->second.count;
    }
    const auto before_unsafe =
        proof.stack_activation_capture_before_blockers.find(
            "unsafe_resource_class");
    if (before_unsafe != proof.stack_activation_capture_before_blockers.end()) {
      unsafe_resource_class_before_records += before_unsafe->second.count;
    }
    const auto after_unsafe =
        proof.stack_activation_capture_after_blockers.find("unsafe_resource_class");
    if (after_unsafe != proof.stack_activation_capture_after_blockers.end()) {
      unsafe_resource_class_after_records += after_unsafe->second.count;
    }
  }

  append_json_comma(out, first);
  out << "\"capture_boundary_dependency_set\":{";
  bool proof_first = true;
  append_json_string(
      out, "schema", "CaptureBoundaryDependencySet.v0", proof_first);
  append_json_bool(out, "behavior_neutral", true, proof_first);
  append_json_bool(out, "dry_run_only", true, proof_first);
  append_json_string(
      out,
      "target_boundary_class",
      "bridge_private_intermediate_capture",
      proof_first);
  append_json_u64(
      out, "candidate_boundaries", candidate_boundaries, proof_first);
  append_json_u64(
      out,
      "combined_complete_boundaries",
      combined_complete_boundaries,
      proof_first);
  append_json_u64(
      out,
      "public_complete_boundaries",
      public_complete_boundaries,
      proof_first);
  append_json_u64(
      out,
      "bridge_private_complete_boundaries",
      bridge_private_complete_boundaries,
      proof_first);
  append_json_u64(
      out,
      "full_boundary_complete_boundaries",
      full_boundary_complete_boundaries,
      proof_first);
  append_json_u64(
      out,
      "required_capture_edge_records",
      required_capture_edge_records,
      proof_first);
  append_json_u64(
      out,
      "bridge_private_capture_proof_complete_records",
      bridge_private_complete_records,
      proof_first);
  append_json_u64(out, "barriers_inserted", 0u, proof_first);
  append_json_u64(out, "submits_removed", 0u, proof_first);
  append_json_comma(out, proof_first);
  out << "\"remaining_full_boundary_blockers\":";
  append_u64_map_object(out, remaining_blockers);
  append_json_comma(out, proof_first);
  out << "\"records\":[";
  bool first_record = true;
  for (const auto& item : proofs) {
    if (!first_record) {
      out << ',';
    }
    first_record = false;
    append_capture_boundary_dependency_set_record(out, item.second);
  }
  out << "]}";
  append_stack_activation_capture_proof_json(
      out,
      proofs,
      summaries,
      registrations,
      missing_stack_activation_before_records,
      missing_stack_activation_after_records,
      capture_sensitive_before_records,
      capture_sensitive_after_records,
      unsafe_resource_class_before_records,
      unsafe_resource_class_after_records,
      first);
  append_phase_boundary_budget_recompute_json(out, proofs, first);
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
  append_json_u64(
      out,
      "formal_last_use_planned_records",
      proof.formal_last_use_planned_records,
      first);
  append_json_u64(
      out,
      "formal_last_use_missing_reduced_records",
      proof.formal_last_use_missing_reduced_records,
      first);
  append_json_bool(out, "complete", complete, first);
  append_json_bool(out, "behavior_change_allowed", false, first);
  append_json_comma(out, first);
  out << "\"consumer_dispatch_proofs\":";
  append_u64_map_object(out, proof.consumer_dispatch_proofs);
  append_json_comma(out, first);
  out << "\"formal_last_use_proofs\":";
  append_u64_map_object(out, proof.formal_last_use_proofs);
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
    const std::map<std::string, BarrierPlanDispatchPosition>& positions,
    const std::map<std::string, BarrierPlanDispatchPosition>& insertion_points,
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
    if (field_or(fields, "formal_last_use_planned", "0") == "1") {
      proof.formal_last_use_planned_records += count;
      proof.formal_last_use_missing_reduced_records += count;
    }
    proof.formal_last_use_proofs
        [field_or(fields, "formal_last_use_proof_source", "missing")] += count;
    const BarrierPlanDispatchPosition consumer_position =
        barrier_plan_consumer_dispatch_position(
            fields, positions, insertion_points);
    const bool plannable =
        barrier_plan_record_is_plannable(fields, consumer_position);
    if (plannable) {
      proof.covered_edge_records += count;
    } else {
      proof.rejected_edge_records += count;
      proof.edge_rejection_reasons
          [barrier_plan_rejection_reason(fields, consumer_position)] += count;
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
  uint64_t formal_last_use_planned_records = 0u;
  uint64_t formal_last_use_missing_reduced_records = 0u;
  std::map<std::string, uint64_t> blocker_reasons;
  for (const auto& item : proofs) {
    const auto& proof = item.second;
    ++candidate_boundaries;
    required_edge_records += proof.required_edge_records;
    covered_edge_records += proof.covered_edge_records;
    consumer_dispatch_planned_records += proof.consumer_dispatch_planned_records;
    consumer_dispatch_missing_reduced_records +=
        proof.consumer_dispatch_missing_reduced_records;
    formal_last_use_planned_records += proof.formal_last_use_planned_records;
    formal_last_use_missing_reduced_records +=
        proof.formal_last_use_missing_reduced_records;
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
  append_json_u64(
      out,
      "formal_last_use_planned_records",
      formal_last_use_planned_records,
      proof_first);
  append_json_u64(
      out,
      "formal_last_use_missing_reduced_records",
      formal_last_use_missing_reduced_records,
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

void append_stack_region_boundary_submit_plan_record(
    std::ostream& out,
    const std::string& row,
    const size_t index) {
  const auto fields = parse_space_separated_fields(row);
  bool first = true;
  out << '{';
  append_json_string(
      out,
      "plan_record_id",
      "boundary_submit_plan_" + std::to_string(index),
      first);
  append_json_string(
      out,
      "schema",
      field_or(fields, "schema", "StackRegionBoundarySubmitPlan.v0"),
      first);
  append_json_string(
      out,
      "live_boundary_id",
      field_or(fields, "live_boundary_id", "none"),
      first);
  append_json_string(
      out,
      "selected_boundary_id",
      field_or(fields, "selected_boundary_id", "none"),
      first);
  append_json_string(
      out,
      "selected_scope",
      field_or(fields, "selected_scope", "none"),
      first);
  append_json_string(
      out,
      "selected_proof_id",
      field_or(fields, "selected_proof_id", "none"),
      first);
  append_json_string(
      out,
      "selected_proof_version",
      field_or(fields, "selected_proof_version", "none"),
      first);
  append_json_string(
      out,
      "online_plan_status",
      field_or(fields, "online_plan_status", "missing"),
      first);
  append_json_bool(
      out,
      "live_boundary_matches_selected",
      field_or(fields, "live_boundary_matches_selected", "0") == "1",
      first);
  append_json_bool(
      out,
      "same_region_consumer_registration_present",
      field_or(fields, "same_region_consumer_registration_present", "0") ==
          "1",
      first);
  append_json_bool(
      out,
      "public_scope_rejected",
      field_or(fields, "public_scope_rejected", "0") == "1",
      first);
  append_json_string(
      out,
      "stale_proof_check",
      field_or(fields, "stale_proof_check", "missing"),
      first);
  append_json_bool(
      out, "queue_submit", field_or(fields, "queue_submit", "0") == "1", first);
  append_json_u64(
      out,
      "old_path_pending",
      parsed_u64(fields, "old_path_pending"),
      first);
  append_json_u64(
      out,
      "safe_candidate_bytes",
      parsed_u64(fields, "safe_candidate_bytes"),
      first);
  append_json_string(
      out, "budget_reject", field_or(fields, "budget_reject", "missing"), first);
  append_json_string(
      out, "blockers", field_or(fields, "blockers", "missing"), first);
  const bool live_boundary_match =
      field_or(fields, "live_boundary_matches_selected", "0") == "1";
  append_json_bool(out, "current_run_proof_matched", live_boundary_match, first);
  append_json_bool(out, "behavior_change_allowed", false, first);
  append_json_string(
      out,
      "submit_skip_planning_status",
      live_boundary_match ? "rejected_behavior_change_not_allowed"
                          : field_or(fields, "online_plan_status", "missing"),
      first);
  append_json_string(
      out,
      "submit_skip_hard_veto_reason",
      "rejected_behavior_change_not_allowed",
      first);
  append_json_bool(
      out,
      "requires_barrier_or_no_visibility_dependency_proof",
      true,
      first);
  append_json_bool(out, "real_barrier_records_inserted", false, first);
  append_json_bool(out, "no_visibility_dependency_proof", false, first);
  append_json_string(
      out,
      "visibility_dependency_proof_status",
      "missing_live_visibility_or_no_dependency_proof",
      first);
  append_json_u64(out, "barriers_inserted", 0u, first);
  append_json_u64(out, "submits_removed", 0u, first);
  append_json_comma(out, first);
  out << "\"source_fields\":";
  append_json_fields_object(out, fields);
  out << '}';
}

void append_stack_region_boundary_submit_plan_json(
    std::ostream& out,
    const std::vector<std::string>& rows,
    bool& first) {
  uint64_t candidate_records = 0u;
  uint64_t selected_live_match_records = 0u;
  uint64_t same_region_registration_records = 0u;
  uint64_t queue_submit_records = 0u;
  uint64_t behavior_change_veto_records = 0u;
  std::map<std::string, uint64_t> status_counts;
  std::map<std::string, uint64_t> submit_skip_status_counts;
  std::map<std::string, uint64_t> selected_boundary_counts;
  for (const auto& row : rows) {
    const auto fields = parse_space_separated_fields(row);
    const uint64_t count = std::max<uint64_t>(parsed_u64(fields, "count"), 1u);
    candidate_records += count;
    status_counts[field_or(fields, "online_plan_status", "missing")] += count;
    const bool live_boundary_match =
        field_or(fields, "live_boundary_matches_selected", "0") == "1";
    const std::string submit_skip_status =
        live_boundary_match ? "rejected_behavior_change_not_allowed"
                            : field_or(fields, "online_plan_status", "missing");
    submit_skip_status_counts[submit_skip_status] += count;
    selected_boundary_counts[field_or(fields, "selected_boundary_id", "none")] +=
        count;
    if (live_boundary_match) {
      selected_live_match_records += count;
      behavior_change_veto_records += count;
    }
    if (
        field_or(fields, "same_region_consumer_registration_present", "0") ==
        "1") {
      same_region_registration_records += count;
    }
    if (field_or(fields, "queue_submit", "0") == "1") {
      queue_submit_records += count;
    }
  }

  append_json_comma(out, first);
  out << "\"stack_region_boundary_submit_plan\":{";
  bool plan_first = true;
  append_json_string(
      out, "schema", "StackRegionBoundarySubmitPlan.v0", plan_first);
  append_json_bool(out, "behavior_neutral", true, plan_first);
  append_json_bool(out, "dry_run_only", true, plan_first);
  append_json_string(
      out,
      "planning_stage",
      "online_phase_boundary_submit_hook",
      plan_first);
  append_json_u64(out, "candidate_records", candidate_records, plan_first);
  append_json_u64(
      out,
      "selected_live_boundary_match_records",
      selected_live_match_records,
      plan_first);
  append_json_u64(
      out,
      "same_region_consumer_registration_records",
      same_region_registration_records,
      plan_first);
  append_json_u64(
      out, "queue_submit_records", queue_submit_records, plan_first);
  append_json_u64(
      out,
      "behavior_change_veto_records",
      behavior_change_veto_records,
      plan_first);
  append_json_u64(out, "barriers_inserted", 0u, plan_first);
  append_json_u64(out, "submits_removed", 0u, plan_first);
  append_json_bool(out, "behavior_change_allowed", false, plan_first);
  append_json_string(
      out,
      "submit_skip_hard_veto_reason",
      "rejected_behavior_change_not_allowed",
      plan_first);
  append_json_bool(
      out,
      "requires_barrier_or_no_visibility_dependency_proof",
      true,
      plan_first);
  append_json_bool(out, "no_visibility_dependency_proof", false, plan_first);
  append_json_string(
      out,
      "behavior_change_status",
      "disabled_default_submit_preserved",
      plan_first);
  append_json_comma(out, plan_first);
  out << "\"online_plan_status_counts\":";
  append_u64_map_object(out, status_counts);
  append_json_comma(out, plan_first);
  out << "\"submit_skip_planning_status_counts\":";
  append_u64_map_object(out, submit_skip_status_counts);
  append_json_comma(out, plan_first);
  out << "\"selected_boundary_counts\":";
  append_u64_map_object(out, selected_boundary_counts);
  append_json_comma(out, plan_first);
  out << "\"records\":[";
  for (size_t i = 0; i < rows.size(); ++i) {
    if (i > 0) {
      out << ',';
    }
    append_stack_region_boundary_submit_plan_record(out, rows[i], i);
  }
  out << "]}";
}

void append_stack_region_barrier_only_canary_record(
    std::ostream& out,
    const std::string& row,
    const size_t index) {
  const auto fields = parse_space_separated_fields(row);
  bool first = true;
  out << '{';
  append_json_string(
      out,
      "canary_record_id",
      "barrier_only_canary_" + std::to_string(index),
      first);
  append_json_string(
      out,
      "schema",
      field_or(fields, "schema", "StackRegionBarrierOnlyCanary.v0"),
      first);
  append_json_string(
      out,
      "selected_boundary_id",
      field_or(fields, "selected_boundary_id", "none"),
      first);
  append_json_string(
      out, "selected_scope", field_or(fields, "selected_scope", "none"), first);
  append_json_string(
      out, "producer_phase", field_or(fields, "producer_phase", "none"), first);
  append_json_u64(out, "producer_block", parsed_u64(fields, "producer_block"), first);
  append_json_string(
      out, "consumer_phase", field_or(fields, "consumer_phase", "none"), first);
  append_json_u64(out, "consumer_block", parsed_u64(fields, "consumer_block"), first);
  append_json_string(out, "live_phase", field_or(fields, "live_phase", "none"), first);
  append_json_u64(out, "live_block", parsed_u64(fields, "live_block"), first);
  append_json_string(out, "shader", field_or(fields, "shader", "unknown"), first);
  append_json_u64(
      out, "descriptor_binding", parsed_u64(fields, "descriptor_binding"), first);
  append_json_u64(
      out,
      "next_recorded_dispatch_position",
      parsed_u64(fields, "next_recorded_dispatch_position"),
      first);
  append_json_u64(out, "allocation_id", parsed_u64(fields, "allocation_id"), first);
  append_json_u64(
      out,
      "allocation_generation",
      parsed_u64(fields, "allocation_generation"),
      first);
  append_json_u64(out, "byte_offset", parsed_u64(fields, "byte_offset"), first);
  append_json_u64(out, "byte_range", parsed_u64(fields, "byte_range"), first);
  append_json_bool(
      out,
      "live_vulkan_buffer_binding_available",
      field_or(fields, "live_vulkan_buffer_binding_available", "0") == "1",
      first);
  append_json_string(
      out,
      "live_vulkan_buffer_handle_token",
      field_or(fields, "live_vulkan_buffer_handle_token", "missing"),
      first);
  append_json_string(
      out,
      "live_vulkan_buffer_object_token",
      field_or(fields, "live_vulkan_buffer_object_token", "missing"),
      first);
  append_json_string(out, "src_stage", field_or(fields, "src_stage", "missing"), first);
  append_json_string(out, "src_access", field_or(fields, "src_access", "missing"), first);
  append_json_string(out, "dst_stage", field_or(fields, "dst_stage", "missing"), first);
  append_json_string(out, "dst_access", field_or(fields, "dst_access", "missing"), first);
  append_json_bool(
      out,
      "current_run_proof_match",
      field_or(fields, "current_run_proof_match", "0") == "1",
      first);
  append_json_string(
      out,
      "current_run_proof_status",
      field_or(fields, "current_run_proof_status", "missing"),
      first);
  append_json_string(
      out,
      "barrier_only_status",
      field_or(fields, "barrier_only_status", "missing"),
      first);
  append_json_string(
      out, "reject_reason", field_or(fields, "reject_reason", "missing"), first);
  append_json_bool(out, "capture_edge", field_or(fields, "capture_edge", "0") == "1", first);
  append_json_bool(out, "public_output", field_or(fields, "public_output", "0") == "1", first);
  append_json_bool(out, "final_output", field_or(fields, "final_output", "0") == "1", first);
  append_json_bool(out, "host_visible", field_or(fields, "host_visible", "0") == "1", first);
  append_json_bool(out, "readback_edge", field_or(fields, "readback_edge", "0") == "1", first);
  append_json_bool(out, "behavior_change_allowed", false, first);
  append_json_u64(
      out, "barriers_inserted", parsed_u64(fields, "barriers_inserted"), first);
  append_json_u64(
      out, "submits_removed", parsed_u64(fields, "submits_removed"), first);
  append_json_u64(out, "count", parsed_u64(fields, "count"), first);
  append_json_u64(out, "bytes", parsed_u64(fields, "bytes"), first);
  append_json_comma(out, first);
  out << "\"source_fields\":";
  append_json_fields_object(out, fields);
  out << '}';
}

void append_stack_region_barrier_only_canary_json(
    std::ostream& out,
    const std::vector<std::string>& rows,
    bool& first) {
  uint64_t candidate_records = 0u;
  uint64_t live_buffer_bound_records = 0u;
  uint64_t barriers_inserted = 0u;
  uint64_t submits_removed = 0u;
  std::map<std::string, uint64_t> status_counts;
  std::map<std::string, uint64_t> reject_reason_counts;
  for (const auto& row : rows) {
    const auto fields = parse_space_separated_fields(row);
    const uint64_t count = std::max<uint64_t>(parsed_u64(fields, "count"), 1u);
    candidate_records += count;
    live_buffer_bound_records +=
        parsed_u64(fields, "live_buffer_bound_count");
    barriers_inserted += parsed_u64(fields, "barriers_inserted");
    submits_removed += parsed_u64(fields, "submit_removed");
    status_counts[field_or(fields, "barrier_only_status", "missing")] += count;
    reject_reason_counts[field_or(fields, "reject_reason", "missing")] += count;
  }

  append_json_comma(out, first);
  out << "\"stack_region_barrier_only_canary\":{";
  bool canary_first = true;
  append_json_string(
      out, "schema", "StackRegionBarrierOnlyCanary.v0", canary_first);
  append_json_bool(out, "opt_in_only", true, canary_first);
  append_json_bool(out, "default_behavior_unchanged", true, canary_first);
  append_json_string(
      out,
      "opt_in_env",
      "PYTORCH_VULKAN_STACK_REGION_BARRIER_CANARY",
      canary_first);
  append_json_string(
      out,
      "selected_boundary_id",
      "non_capture_boundary:producer_block=0:consumer_block=1",
      canary_first);
  append_json_string(
      out, "target_boundary_class", "non_capture_residual2_to_norm1", canary_first);
  append_json_u64(out, "candidate_records", candidate_records, canary_first);
  append_json_u64(
      out, "live_buffer_bound_records", live_buffer_bound_records, canary_first);
  append_json_u64(out, "barriers_inserted", barriers_inserted, canary_first);
  append_json_u64(out, "submits_removed", submits_removed, canary_first);
  append_json_bool(out, "submit_skip_behavior_change_allowed", false, canary_first);
  append_json_bool(out, "barrier_behavior_allowed", false, canary_first);
  append_json_string(
      out,
      "fail_closed_reason",
      candidate_records == 0u
          ? "selected_boundary_not_observed"
          : "missing_current_run_proof_match_at_consumer_recording",
      canary_first);
  append_json_comma(out, canary_first);
  out << "\"status_counts\":";
  append_u64_map_object(out, status_counts);
  append_json_comma(out, canary_first);
  out << "\"reject_reason_counts\":";
  append_u64_map_object(out, reject_reason_counts);
  append_json_comma(out, canary_first);
  out << "\"records\":[";
  for (size_t i = 0; i < rows.size(); ++i) {
    if (i > 0) {
      out << ',';
    }
    append_stack_region_barrier_only_canary_record(out, rows[i], i);
  }
  out << "]}";
}

void split_stack_graph_rows(
    const std::vector<std::string>& rows,
    std::vector<std::string>& dispatch_nodes,
    std::vector<std::string>& insertion_point_nodes,
    std::vector<std::string>& live_buffer_binding_nodes,
    std::vector<std::string>& dependency_edges,
    std::vector<std::string>& capture_edges,
    std::vector<std::string>& boundary_submit_plan_rows,
    std::vector<std::string>& barrier_only_canary_rows) {
  for (const auto& row : rows) {
    if (row.find("stack_region_barrier_only_canary=1") != std::string::npos) {
      barrier_only_canary_rows.emplace_back(row);
      continue;
    }
    if (row.find("stack_region_boundary_submit_plan=1") != std::string::npos) {
      boundary_submit_plan_rows.emplace_back(row);
      continue;
    }
    if (row.find("pre_dispatch_insertion_point=1") != std::string::npos) {
      insertion_point_nodes.emplace_back(row);
      continue;
    }
    if (row.find("live_vulkan_buffer_binding=1") != std::string::npos) {
      live_buffer_binding_nodes.emplace_back(row);
      continue;
    }
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
  const std::vector<std::string> consumer_registration_rows =
      stack_output_device_consumer_registration_snapshot();
  const std::vector<std::string> lifetime_rows =
      stack_subresource_lifetime_dry_run_snapshot();
  const std::vector<std::string> region_rows =
      region_lifetime_submit_attribution_snapshot();

  std::vector<std::string> dispatch_nodes;
  std::vector<std::string> insertion_point_nodes;
  std::vector<std::string> live_buffer_binding_nodes;
  std::vector<std::string> dependency_edges;
  std::vector<std::string> capture_edges;
  std::vector<std::string> boundary_submit_plan_rows;
  std::vector<std::string> barrier_only_canary_rows;
  split_stack_graph_rows(
      dispatch_dependency_rows,
      dispatch_nodes,
      insertion_point_nodes,
      live_buffer_binding_nodes,
      dependency_edges,
      capture_edges,
      boundary_submit_plan_rows,
      barrier_only_canary_rows);

  std::vector<std::string> resource_nodes;
  std::vector<std::string> boundary_nodes;
  split_lifetime_graph_rows(lifetime_rows, resource_nodes, boundary_nodes);
  const auto barrier_plan_dispatch_positions =
      build_barrier_plan_dispatch_positions(dispatch_nodes);
  const auto barrier_plan_insertion_points =
      build_barrier_plan_insertion_points(insertion_point_nodes);
  std::map<std::string, uint64_t> barrier_plan_live_allocation_bindings;
  const auto barrier_plan_live_buffer_bindings =
      build_barrier_plan_live_buffer_bindings(
          live_buffer_binding_nodes, barrier_plan_live_allocation_bindings);
  const auto capture_allocation_summaries =
      build_capture_allocation_summaries(allocation_rows);
  const auto consumer_registration_summaries =
      build_stack_output_device_consumer_registration_summaries(
          consumer_registration_rows);

  uint64_t fully_proven_edge_records = 0u;
  uint64_t total_dependency_records = 0u;
  uint64_t queue_submit_dependency_records = 0u;
  uint64_t capture_output_boundary_records = 0u;
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
  for (const auto& row : capture_edges) {
    capture_output_boundary_records +=
        parsed_u64(parse_space_separated_fields(row), "count");
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
      out,
      "pre_dispatch_insertion_point_nodes",
      insertion_point_nodes.size(),
      summary_first);
  append_json_u64(
      out,
      "live_vulkan_buffer_binding_nodes",
      live_buffer_binding_nodes.size(),
      summary_first);
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
  append_json_u64(
      out,
      "capture_output_boundary_records",
      capture_output_boundary_records,
      summary_first);
  append_json_u64(
      out,
      "stack_output_device_consumer_registration_rows",
      consumer_registration_rows.size(),
      summary_first);
  append_json_u64(
      out,
      "stack_region_boundary_submit_plan_rows",
      boundary_submit_plan_rows.size(),
      summary_first);
  append_json_u64(
      out,
      "stack_region_barrier_only_canary_rows",
      barrier_only_canary_rows.size(),
      summary_first);
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
      out,
      "pre_dispatch_insertion_point_nodes",
      insertion_point_nodes,
      "pre_dispatch_insertion_point",
      first);
  append_graph_array(
      out,
      "live_vulkan_buffer_binding_nodes",
      live_buffer_binding_nodes,
      "live_vulkan_buffer_binding",
      first);
  append_graph_array(
      out, "dependency_edges", dependency_edges, "dependency_edge", first);
  append_graph_array(out, "capture_edges", capture_edges, "capture_edge", first);
  append_graph_array(out, "resource_nodes", resource_nodes, "resource", first);
  append_graph_array(
      out, "allocation_nodes", allocation_rows, "allocation", first);
  append_graph_array(
      out, "phase_boundary_nodes", boundary_nodes, "phase_boundary", first);
  append_graph_array(
      out,
      "stack_region_boundary_submit_plan_live_rows",
      boundary_submit_plan_rows,
      "stack_region_boundary_submit_plan",
      first);
  append_graph_array(
      out,
      "stack_region_barrier_only_canary_live_rows",
      barrier_only_canary_rows,
      "stack_region_barrier_only_canary",
      first);
  append_graph_array(
      out,
      "stack_output_device_consumer_registrations",
      consumer_registration_rows,
      "stack_output_device_consumer_registration",
      first);
  append_capture_output_boundary_contract_json(
      out,
      capture_edges,
      capture_allocation_summaries,
      consumer_registration_summaries,
      first);
  append_stack_region_boundary_submit_plan_json(
      out, boundary_submit_plan_rows, first);
  append_stack_region_barrier_only_canary_json(
      out, barrier_only_canary_rows, first);
  append_barrier_plan_json(
      out,
      dependency_edges,
      barrier_plan_dispatch_positions,
      barrier_plan_insertion_points,
      barrier_plan_live_buffer_bindings,
      barrier_plan_live_allocation_bindings,
      first);
  append_boundary_complete_dependency_proof_json(
      out,
      dependency_edges,
      boundary_nodes,
      barrier_plan_dispatch_positions,
      barrier_plan_insertion_points,
      first);
  append_capture_boundary_dependency_set_json(
      out,
      capture_edges,
      boundary_nodes,
      capture_allocation_summaries,
      consumer_registration_summaries,
      first);
  append_graph_array(
      out, "region_lifetime_rows", region_rows, "region_lifetime", first);
  append_json_string_array(
      out,
      "unproven_or_missing_metadata_fields",
      {"region_id",
       "stack_context_id",
       "bridge_session_id",
       "complete_boundary_dependency_set",
       "capture_output_boundary_value_preservation",
       "capture_output_downstream_consumer_registration_in_graph",
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

VulkanStackPlannedDispatchPositionScope::
    VulkanStackPlannedDispatchPositionScope(
        std::vector<VulkanStackPlannedDispatchPosition> positions)
    : previous_(std::move(g_stack_planned_dispatch_positions)) {
  g_stack_planned_dispatch_positions = std::move(positions);
}

VulkanStackPlannedDispatchPositionScope::
    ~VulkanStackPlannedDispatchPositionScope() {
  g_stack_planned_dispatch_positions = std::move(previous_);
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

int64_t parsed_i64_or(
    const std::map<std::string, std::string>& fields,
    const char* key,
    const int64_t fallback) {
  const auto it = fields.find(key);
  if (it == fields.end()) {
    return fallback;
  }
  try {
    return static_cast<int64_t>(std::stoll(it->second));
  } catch (...) {
    return fallback;
  }
}

std::string capture_boundary_id_for_block(const int64_t block) {
  std::ostringstream stream;
  stream << "capture_boundary:producer_block=" << block
         << ":capture_block=" << block;
  return stream.str();
}

struct StackRegionBoundarySubmitPlanSelection final {
  bool has_same_region_registration = false;
  int64_t selected_capture_block = -1;
  std::string selected_boundary_id = "none";
  std::string selected_proof_id = "none";
  std::string selected_registration_key = "none";
};

StackRegionBoundarySubmitPlanSelection
select_stack_region_boundary_submit_plan_locked() {
  StackRegionBoundarySubmitPlanSelection selection;
  for (const auto& item : stack_output_device_consumer_registrations()) {
    const auto fields = parse_space_separated_fields(item.first);
    if (
        field_or(fields, "consumer_in_same_planned_region", "0") != "1" ||
        field_or(fields, "python_public_boundary_before_consumption", "1") !=
            "0" ||
        field_or(fields, "host_visible_boundary_before_consumption", "1") !=
            "0" ||
        field_or(fields, "host_visible_access_before_consumption", "1") !=
            "0" ||
        field_or(fields, "host_readback_before_consumption", "1") != "0" ||
        field_or(fields, "output_role", "unknown") !=
            "stack_residual2_output") {
      continue;
    }
    const int64_t block = parsed_i64_or(fields, "captured_block", -1);
    if (block <= 0) {
      continue;
    }
    if (
        !selection.has_same_region_registration ||
        block < selection.selected_capture_block) {
      selection.has_same_region_registration = true;
      selection.selected_capture_block = block;
      selection.selected_boundary_id = capture_boundary_id_for_block(block);
      selection.selected_proof_id =
          "PhaseBoundaryBudgetRecompute.v0:" + selection.selected_boundary_id;
      selection.selected_registration_key =
          stack_output_device_consumer_registration_key(
              std::to_string(block), "stack_residual2_output");
    }
  }
  return selection;
}

void note_stack_region_boundary_submit_plan(
    const VulkanSubmitPhase phase,
    const VulkanRetireCallSite callsite,
    const bool queue_submit,
    const uint64_t old_path_pending_count,
    const uint64_t old_path_pending_bytes,
    const uint64_t safe_candidate_count,
    const uint64_t safe_candidate_bytes,
    const std::string& budget_reject,
    const std::string& blockers) {
  if (
      phase != VulkanSubmitPhase::StackOwner ||
      !(callsite == VulkanRetireCallSite::StackOwnerPhaseBoundary ||
        callsite == VulkanRetireCallSite::StackOwnerNorm1 ||
        callsite == VulkanRetireCallSite::StackOwnerNorm2)) {
    return;
  }
  const int64_t live_block = current_vision_stack_block_index();
  bool live_capture_boundary = false;
  if (live_block >= 0) {
    for (const int64_t capture_index : g_vision_stack_capture_indices) {
      if (capture_index == live_block) {
        live_capture_boundary = true;
        break;
      }
    }
  }
  const std::string live_boundary_id =
      live_capture_boundary ? capture_boundary_id_for_block(live_block) : "none";

  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  const StackRegionBoundarySubmitPlanSelection selection =
      select_stack_region_boundary_submit_plan_locked();
  std::string online_status = "not_planned";
  if (!selection.has_same_region_registration) {
    online_status = !g_vision_stack_capture_indices.empty()
        ? "rejected_public_scope_or_no_same_region_consumer"
        : "not_planned";
  } else if (live_boundary_id == selection.selected_boundary_id) {
    online_status = "planned_live_boundary_match_proof_pending";
  } else if (live_capture_boundary) {
    online_status = "rejected_boundary_mismatch";
  } else {
    online_status = "not_selected_boundary";
  }

  std::ostringstream key;
  key << "stack_region_boundary_submit_plan=1"
      << " contract=StackRegionBoundarySubmitPlan"
      << " schema=StackRegionBoundarySubmitPlan.v0"
      << " behavior_neutral=1"
      << " dry_run_only=1"
      << " phase=" << submit_phase_name(phase)
      << " callsite=" << retire_call_site_name(callsite)
      << " live_boundary_stack_phase="
      << vision_stack_phase_name(current_vision_stack_phase())
      << " live_boundary_block=" << live_block
      << " live_boundary_id=" << live_boundary_id
      << " selected_boundary_id=" << selection.selected_boundary_id
      << " selected_scope="
      << (selection.has_same_region_registration ? "bridge_private_capture"
                                                 : "none")
      << " selected_proof_id=" << selection.selected_proof_id
      << " selected_proof_version=PhaseBoundaryBudgetRecompute.v0"
      << " selected_registration_key="
      << selection.selected_registration_key
      << " online_plan_status=" << online_status
      << " live_boundary_matches_selected="
      << (live_boundary_id == selection.selected_boundary_id ? 1 : 0)
      << " same_region_consumer_registration_present="
      << (selection.has_same_region_registration ? 1 : 0)
      << " public_scope_rejected="
      << (!selection.has_same_region_registration &&
              !g_vision_stack_capture_indices.empty()
          ? 1
          : 0)
      << " behavior_change_allowed=0"
      << " submit_skip_hard_veto_reason=rejected_behavior_change_not_allowed"
      << " requires_barrier_or_no_visibility_dependency_proof=1"
      << " real_barrier_records_inserted=0"
      << " no_visibility_dependency_proof=0"
      << " visibility_dependency_proof_status=missing_live_visibility_or_no_dependency_proof"
      << " proof_source=current_graph_run_required"
      << " stale_proof_check=fail_closed_without_current_graph_match"
      << " queue_submit=" << (queue_submit ? 1 : 0)
      << " old_path_pending=" << old_path_pending_count
      << " safe_candidate_count=" << safe_candidate_count
      << " safe_candidate_bytes=" << safe_candidate_bytes
      << " budget_reject=" << budget_reject
      << " blockers=" << blockers
      << " barriers_inserted=0"
      << " submits_removed=0";
  auto& value = stack_region_boundary_submit_plan_rows()[key.str()];
  value.count += 1u;
  value.bytes += old_path_pending_bytes;
  if (queue_submit) {
    value.queue_submit_count += 1u;
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

bool vision_stack_capture_dependency_between_blocks(
    const int64_t producer_block,
    const int64_t consumer_block) {
  if (producer_block < 0 || consumer_block < 0) {
    return true;
  }
  for (const int64_t capture_index : g_vision_stack_capture_indices) {
    if (capture_index > producer_block && capture_index < consumer_block) {
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

void note_vulkan_stack_pre_dispatch_insertion_point(const char* shader_name) {
  if (!inside_vision_stack_phase()) {
    return;
  }
  const uint64_t scope_id = g_stack_dispatch_dependency_scope_id;
  if (scope_id == 0u) {
    return;
  }
  const VulkanStackPlannedDispatchPosition* const planned_position =
      find_stack_planned_dispatch_position(
          g_vision_stack_phase, g_vision_stack_block_index);
  if (!planned_position) {
    return;
  }
  const uint64_t next_recorded_position =
      g_stack_dispatch_dependency_position + 1u;
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  auto& value = stack_dispatch_dependency_insertion_point_rows()
      [stack_dispatch_dependency_insertion_point_key(
          scope_id,
          g_vision_stack_phase,
          g_vision_stack_block_index,
          *planned_position,
          shader_name)];
  value.count += 1u;
  if (value.first_position == 0u) {
    value.first_position = next_recorded_position;
  }
  value.last_position = next_recorded_position;
}

void note_vulkan_stack_live_descriptor_binding(
    const uint32_t binding_idx,
    const char* shader_name,
    const VulkanBuffer& buffer) {
  if (!inside_vision_stack_phase()) {
    return;
  }
  const uint64_t scope_id = g_stack_dispatch_dependency_scope_id;
  if (scope_id == 0u) {
    return;
  }
  const uint64_t next_recorded_position =
      g_stack_dispatch_dependency_position + 1u;
  auto& rows = stack_dispatch_dependency_live_buffer_binding_rows();
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  auto& value = rows[stack_dispatch_dependency_live_buffer_binding_key(
      scope_id,
      g_vision_stack_phase,
      g_vision_stack_block_index,
      shader_name,
      binding_idx,
      next_recorded_position,
      buffer)];
  value.count += 1u;
  if (value.first_position == 0u) {
    value.first_position = next_recorded_position;
  }
  value.last_position = next_recorded_position;
}

void note_vulkan_stack_barrier_only_canary_descriptor(
    const uint32_t binding_idx,
    const char* shader_name,
    const VulkanBuffer& buffer) {
  const char* const target = stack_region_barrier_only_canary_target();
  if (!stack_region_barrier_only_canary_target_selected(target)) {
    return;
  }
  if (!inside_vision_stack_phase()) {
    return;
  }
  const uint64_t scope_id = g_stack_dispatch_dependency_scope_id;
  if (scope_id == 0u) {
    return;
  }
  if (
      g_vision_stack_phase != VulkanVisionStackPhase::Norm1 ||
      g_vision_stack_block_index != 1 || binding_idx != 6u) {
    return;
  }
  const uint64_t allocation_id = buffer.allocation_id();
  const uint64_t allocation_generation =
      vulkan_memory_allocation_generation(allocation_id);
  const bool live_buffer_bound =
      allocation_id != 0u && allocation_generation != 0u &&
      buffer.has_memory() && buffer.mem_range() != 0u &&
      buffer.handle() != VK_NULL_HANDLE;
  const char* const status = live_buffer_bound
      ? "fail_closed_missing_pre_dispatch_barrier_plan_proof"
      : "fail_closed_missing_live_vulkan_buffer_binding";
  const char* const reject_reason = live_buffer_bound
      ? "missing_current_run_proof_match_at_consumer_recording"
      : "missing_live_vulkan_buffer_binding";
  const uint64_t next_recorded_position =
      g_stack_dispatch_dependency_position + 1u;
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  auto& value = stack_region_barrier_only_canary_rows()
      [stack_region_barrier_only_canary_key(
          scope_id,
          g_vision_stack_phase,
          g_vision_stack_block_index,
          shader_name,
          binding_idx,
          next_recorded_position,
          buffer,
          live_buffer_bound,
          status,
          reject_reason)];
  value.count += 1u;
  value.bytes += static_cast<uint64_t>(buffer.mem_range());
  if (live_buffer_bound) {
    value.live_buffer_bound_count += 1u;
  }
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
  const bool capture_between_producer_and_consumer =
      vision_stack_capture_dependency_between_blocks(
          provenance.block_index, provenance.expected_consumer_block_index);
  const bool planned_formal_last_use_proof =
      consumer_dispatch_planned && provenance.has_last_use_proof &&
      provenance.lifetime ==
          VulkanStackTensorLifetimeClass::BlockOutputForNextBlock &&
      provenance.phase == VulkanVisionStackPhase::Residual2 &&
      provenance.producer_role == role &&
      provenance.expected_consumer_phase == VulkanVisionStackPhase::Norm1 &&
      provenance.expected_consumer_block_index == provenance.block_index + 1 &&
      provenance.final_consumer_before_stack_submit &&
      !capture_between_producer_and_consumer && !provenance.escapes_stack &&
      !provenance.requested_intermediate && !provenance.final_output &&
      !provenance.alias_or_view && !provenance.aliases_runtime_input &&
      !provenance.aliases_runtime_output && provenance.direct_buffer &&
      provenance.buffer_storage && !provenance.image_storage;
  const bool dependency_formal_last_use_proof =
      formal_last_use_proof || planned_formal_last_use_proof;
  const bool producer_descriptor_known = true;
  const bool consumer_descriptor_known =
      provenance.expected_consumer_phase == VulkanVisionStackPhase::Norm1 ||
      provenance.expected_consumer_phase ==
          VulkanVisionStackPhase::IntermediateCapture;
  const bool fully_proven =
      allocation_proof.has_generation && allocation_proof.has_byte_range &&
      allocation_proof.byte_range > 0u && dependency_formal_last_use_proof &&
      producer_dispatch_observed && consumer_dispatch_observed &&
      producer_descriptor_known && consumer_descriptor_known;
  const std::string reject_reason = stack_dispatch_dependency_reject_reason(
      true,
      allocation_proof.has_generation,
      allocation_proof.has_byte_range && allocation_proof.byte_range > 0u,
      dependency_formal_last_use_proof,
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
      << " stack_provenance_defined=" << (provenance.defined ? 1 : 0)
      << " stack_lifetime=" << stack_tensor_lifetime_name(provenance.lifetime)
      << " direct_buffer=" << (provenance.direct_buffer ? 1 : 0)
      << " buffer_storage=" << (provenance.buffer_storage ? 1 : 0)
      << " image_storage=" << (provenance.image_storage ? 1 : 0)
      << " escapes_stack=" << (provenance.escapes_stack ? 1 : 0)
      << " requested_intermediate="
      << (provenance.requested_intermediate ? 1 : 0)
      << " final_output=" << (provenance.final_output ? 1 : 0)
      << " alias_or_view=" << (provenance.alias_or_view ? 1 : 0)
      << " aliases_runtime_input="
      << (provenance.aliases_runtime_input ? 1 : 0)
      << " aliases_runtime_output="
      << (provenance.aliases_runtime_output ? 1 : 0)
      << " final_consumer_before_stack_submit="
      << (provenance.final_consumer_before_stack_submit ? 1 : 0)
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
              ? 6
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
      << " formal_last_use_proof="
      << (dependency_formal_last_use_proof ? 1 : 0)
      << " formal_last_use_runtime_proof="
      << (formal_last_use_proof ? 1 : 0)
      << " formal_last_use_planned="
      << (planned_formal_last_use_proof ? 1 : 0)
      << " formal_last_use_proof_source="
      << (formal_last_use_proof
              ? "runtime_stack_lifetime"
              : (planned_formal_last_use_proof
                     ? "planned_non_capture_residual2_to_norm1"
                     : "missing"))
      << " capture_between_producer_and_consumer="
      << (capture_between_producer_and_consumer ? 1 : 0)
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

void note_stack_output_device_consumer_registration(
    const VulkanStackOutputDeviceConsumerRegistration& registration) {
  std::ostringstream key;
  key << "stack_output_device_consumer_registration=1"
      << " captured_block=" << registration.captured_block_index
      << " captured_substep=" << registration.captured_substep
      << " output_role=" << registration.output_role
      << " output_shape=" << format_sizes(registration.output_shape)
      << " stack_context_id=" << registration.stack_context_id
      << " stack_session_id=" << registration.stack_session_id
      << " stack_plan_id=" << registration.stack_plan_id
      << " output_layout=" << registration.output_layout
      << " strip_or_view_relation=" << registration.strip_or_view_relation
      << " downstream_consumer_id=" << registration.downstream_consumer_id
      << " downstream_consumer_context="
      << registration.downstream_consumer_context
      << " expected_consumer_input_index="
      << registration.expected_consumer_input_index
      << " expected_consumer_shape="
      << format_sizes(registration.expected_consumer_shape)
      << " expected_consumer_layout="
      << registration.expected_consumer_layout
      << " consumer_in_same_planned_region="
      << (registration.consumer_in_same_planned_region ? 1 : 0)
      << " python_public_boundary_before_consumption="
      << (registration.python_public_boundary_before_consumption ? 1 : 0)
      << " host_visible_boundary_before_consumption="
      << (registration.host_visible_boundary_before_consumption ? 1 : 0)
      << " host_visible_access_before_consumption="
      << (registration.host_visible_access_before_consumption ? 1 : 0)
      << " host_readback_before_consumption="
      << (registration.host_readback_before_consumption ? 1 : 0);
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  auto& value = stack_output_device_consumer_registrations()[key.str()];
  value.count += 1u;
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

std::vector<std::string>
stack_output_device_consumer_registration_snapshot() {
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  std::vector<std::string> rows;
  rows.reserve(stack_output_device_consumer_registrations().size());
  for (const auto& item : stack_output_device_consumer_registrations()) {
    std::ostringstream row;
    row << item.first << " count=" << item.second.count;
    rows.push_back(row.str());
  }
  return rows;
}

std::vector<std::string> stack_dispatch_dependency_dry_run_snapshot() {
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  std::vector<std::string> rows;
  rows.reserve(
      stack_dispatch_dependency_dispatch_rows().size() +
      stack_dispatch_dependency_insertion_point_rows().size() +
      stack_dispatch_dependency_live_buffer_binding_rows().size() +
      stack_dispatch_dependency_dry_run_rows().size() +
      stack_region_boundary_submit_plan_rows().size() +
      stack_region_barrier_only_canary_rows().size());
  for (const auto& item : stack_dispatch_dependency_dispatch_rows()) {
    std::ostringstream row;
    row << item.first << " count=" << item.second.count
        << " first_position=" << item.second.first_position
        << " last_position=" << item.second.last_position;
    rows.push_back(row.str());
  }
  for (const auto& item : stack_dispatch_dependency_insertion_point_rows()) {
    std::ostringstream row;
    row << item.first << " count=" << item.second.count
        << " next_recorded_dispatch_first_position="
        << item.second.first_position
        << " next_recorded_dispatch_last_position=" << item.second.last_position;
    rows.push_back(row.str());
  }
  for (const auto& item : stack_dispatch_dependency_live_buffer_binding_rows()) {
    std::ostringstream row;
    row << item.first << " count=" << item.second.count
        << " next_recorded_dispatch_first_position="
        << item.second.first_position
        << " next_recorded_dispatch_last_position=" << item.second.last_position;
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
  for (const auto& item : stack_region_boundary_submit_plan_rows()) {
    std::ostringstream row;
    row << item.first << " count=" << item.second.count
        << " bytes=" << item.second.bytes
        << " queue_submit=" << item.second.queue_submit_count
        << " submit_removed=" << item.second.submit_removed_count
        << " barriers_inserted=" << item.second.barrier_inserted_count;
    rows.push_back(row.str());
  }
  for (const auto& item : stack_region_barrier_only_canary_rows()) {
    std::ostringstream row;
    row << item.first << " count=" << item.second.count
        << " bytes=" << item.second.bytes
        << " live_buffer_bound_count=" << item.second.live_buffer_bound_count
        << " submit_removed=" << item.second.submit_removed_count
        << " barriers_inserted=" << item.second.barrier_inserted_count;
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
  stack_dispatch_dependency_insertion_point_rows().clear();
  stack_dispatch_dependency_live_buffer_binding_rows().clear();
  stack_dispatch_dependency_dry_run_rows().clear();
  stack_region_boundary_submit_plan_rows().clear();
  stack_region_barrier_only_canary_rows().clear();
  stack_output_device_consumer_registrations().clear();
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
