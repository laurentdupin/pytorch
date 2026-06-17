#include <ATen/native/vulkan/api/Sync.h>

#ifdef USE_VULKAN_API

#include <algorithm>
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
thread_local VulkanSubmitPhase g_submit_phase = VulkanSubmitPhase::Unknown;
thread_local VulkanRetiredResourceKind g_retired_resource_kind =
    VulkanRetiredResourceKind::Unknown;
thread_local VulkanRetiredResourceRole g_retired_resource_role =
    VulkanRetiredResourceRole::Unknown;
thread_local std::vector<VulkanStackLastUseProof> g_stack_last_use_proofs;

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
constexpr const char* kDryRunStackInternalRawMissingGeneration =
    "stack_internal_raw_missing_generation";
constexpr const char* kDryRunStackInternalRawGenerationRange =
    "stack_internal_raw_generation_range";
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
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof) {
  if (provenance.defined) {
    return kDryRunAttentionProvenanceMissingLastUse;
  }
  if (allocation_proof.has_generation && allocation_proof.has_byte_range) {
    return kind == VulkanRetiredResourceKind::Buffer
        ? kDryRunAttentionBufferGenerationRangeMissingStackProof
        : kDryRunAttentionRawGenerationRangeMissingStackProof;
  }
  return kDryRunAttentionUnknownSubresource;
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
  } else if (key == kDryRunRawNoProvenance) {
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
    const VulkanStackRetireProvenance& provenance) {
  const char* reason =
      stack_drain_blocker_reason(kind, role, provenance, qkv_would_batch);
  const VulkanStackTempLifetimeSafety safety =
      classify_stack_temp_lifetime_safety(role, provenance);
  std::ostringstream key;
  key << "role=" << retired_resource_role_name(role)
      << " reason=" << reason
      << " safety=" << stack_temp_lifetime_safety_name(safety)
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
              kind, provenance, allocation_proof);
  }
  if (qkv_would_batch || has_proven_internal_stack_temp_lifetime(provenance)) {
    return kDryRunProvenStackActivation;
  }
  if (!provenance.defined) {
    if (
        is_stack_temp_role(role) && kind == VulkanRetiredResourceKind::Unknown) {
      if (allocation_proof.has_generation && allocation_proof.has_byte_range) {
        return kDryRunStackInternalRawGenerationRange;
      }
      return kDryRunStackInternalRawMissingGeneration;
    }
    if (
        role == VulkanRetiredResourceRole::Unknown &&
        kind == VulkanRetiredResourceKind::Unknown) {
      return kDryRunTrulyUnknownRawResource;
    }
    return kDryRunRawNoProvenance;
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
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof) {
  auto& counters = stack_subresource_lifetime_dry_run_counters();
  note_dry_run_resource_class(counters, resource_class, bytes);

  std::ostringstream key;
  key << "resource=1 class=" << resource_class
      << " safe_candidate=" << (safe_candidate ? 1 : 0)
      << " large_backing=" << (large_backing ? 1 : 0)
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
      << " requested_intermediate="
      << (provenance.requested_intermediate ? 1 : 0)
      << " final_output=" << (provenance.final_output ? 1 : 0)
      << " alias_or_view=" << (provenance.alias_or_view ? 1 : 0)
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

void note_vulkan_stack_dispatch(const char* shader_name) {
  if (!inside_vision_stack_phase()) {
    return;
  }
  std::ostringstream key;
  key << "stack_dispatch"
      << " phase=" << vision_stack_phase_name(g_vision_stack_phase)
      << " block=" << g_vision_stack_block_index
      << " shader=" << (shader_name && shader_name[0] ? shader_name : "unknown")
      << " role=" << vision_stack_phase_name(g_vision_stack_phase);
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  stack_dispatch_aggregate()[key.str()] += 1u;
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

void reset_stack_dispatch_aggregate() {
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  stack_dispatch_aggregate().clear();
}

void reset_stack_allocation_aggregate() {
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  stack_allocation_aggregate().clear();
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
