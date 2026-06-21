#pragma once

#ifdef USE_VULKAN_API

#include <c10/macros/Export.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace api {

class VulkanBuffer;
class VulkanImage;
struct PipelineBarrier;

enum class VulkanForcedSyncReason : uint8_t {
  ExplicitSynchronize = 0,
  TensorCpuReadback,
  EventSynchronize,
  RetireQueueDrain,
  GpuTimestampQueryReset,
  FallbackPolicyReadback,
  Unknown,
};

enum class VulkanVisionStackPhase : uint8_t {
  Unknown = 0,
  StackEntry,
  BlockEntry,
  Norm1,
  QkvLinear,
  QkvTransform,
  Attention,
  ProjLinear,
  Residual1,
  Norm2,
  Fc1Gelu,
  Fc2,
  Residual2,
  IntermediateCapture,
  StackExit,
};

enum class VulkanStackTensorLifetimeClass : uint8_t {
  Unknown = 0,
  InternalTemp,
  BlockOutputForNextBlock,
  RequestedIntermediateOutput,
  FinalStackOutput,
  AliasOrView,
};

class TORCH_API VulkanVisionStackPhaseScope final {
 public:
  explicit VulkanVisionStackPhaseScope(VulkanVisionStackPhase phase);
  ~VulkanVisionStackPhaseScope();

  VulkanVisionStackPhaseScope(const VulkanVisionStackPhaseScope&) = delete;
  VulkanVisionStackPhaseScope& operator=(const VulkanVisionStackPhaseScope&) =
      delete;

 private:
  VulkanVisionStackPhase previous_;
};

class TORCH_API VulkanVisionStackBlockScope final {
 public:
  explicit VulkanVisionStackBlockScope(int64_t block_index);
  ~VulkanVisionStackBlockScope();

  VulkanVisionStackBlockScope(const VulkanVisionStackBlockScope&) = delete;
  VulkanVisionStackBlockScope& operator=(const VulkanVisionStackBlockScope&) =
      delete;

 private:
  int64_t previous_;
};

class TORCH_API VulkanVisionStackCaptureScope final {
 public:
  explicit VulkanVisionStackCaptureScope(std::vector<int64_t> capture_indices);
  ~VulkanVisionStackCaptureScope();

  VulkanVisionStackCaptureScope(const VulkanVisionStackCaptureScope&) = delete;
  VulkanVisionStackCaptureScope& operator=(
      const VulkanVisionStackCaptureScope&) = delete;

 private:
  std::vector<int64_t> previous_;
};

struct VulkanSyncCounters final {
  std::atomic<uint64_t> compute_dispatch_count{0u};
  std::atomic<uint64_t> submit_compute_job_count{0u};
  std::atomic<uint64_t> stream_submit_count{0u};
  std::atomic<uint64_t> event_record_count{0u};
  std::atomic<uint64_t> event_block_count{0u};
  std::atomic<uint64_t> event_wait_count{0u};
  std::atomic<uint64_t> retire_poll_count{0u};
  std::atomic<uint64_t> retired_resource_count{0u};
  std::atomic<uint64_t> queue_wait_idle_count{0u};
  std::atomic<uint64_t> forced_sync_count{0u};
  std::atomic<uint64_t> fallback_sync_readback_count{0u};
  std::atomic<uint64_t> allocation_record_stream_count{0u};
  std::atomic<uint64_t> allocation_reuse_deferred_count{0u};
  std::atomic<uint64_t> allocation_reuse_after_timeline_count{0u};
  std::atomic<uint64_t> forced_sync_explicit_synchronize_count{0u};
  std::atomic<uint64_t> forced_sync_tensor_cpu_readback_count{0u};
  std::atomic<uint64_t> forced_sync_event_synchronize_count{0u};
  std::atomic<uint64_t> forced_sync_retire_queue_drain_count{0u};
  std::atomic<uint64_t> forced_sync_gpu_timestamp_query_reset_count{0u};
  std::atomic<uint64_t> forced_sync_fallback_policy_readback_count{0u};
  std::atomic<uint64_t> forced_sync_unknown_count{0u};
};

enum class VulkanSubmitOrigin : uint8_t {
  Unknown = 0,
  NormalCmdSubmitFrequency,
  StackPlannedRecordingSubmit,
  PreStackFlush,
  PostStackFlush,
  ExplicitSynchronize,
  TensorCpuReadback,
  FallbackReadback,
  RetireQueueDrain,
  ProfilingTimestampReset,
  ProfilingTimestampReadback,
  ContextShutdown,
  DebugValidation,
  ConvPrepackUpload,
};

constexpr size_t kNumSubmitOrigins = 14u;

enum class VulkanSubmitPhase : uint8_t {
  Unknown = 0,
  ModelSetup,
  PatchEmbed,
  PositionalEmbeddingSetup,
  StackOwner,
  StackOwnerNorm,
  StackOwnerAttention,
  StackOwnerLinear,
  StackOwnerResidual,
  Decoder,
  DecoderConv,
  DecoderUpsample,
  DecoderPointwise,
  Readback,
  ExplicitSynchronize,
  Retire,
  Profiling,
  Shutdown,
  TestHarness,
};

constexpr size_t kNumSubmitPhases = 19u;

enum class VulkanRetireDrainReason : uint8_t {
  Unknown = 0,
  ExplicitDrain,
  Shutdown,
  ResourcePressure,
  DescriptorPoolPressure,
  CommandBufferRecycle,
  ReadbackPreparation,
  Synchronize,
  StackScopeEnd,
  DecoderPhase,
  SetupPhase,
  DebugValidation,
};

enum class VulkanRetireCallSite : uint8_t {
  Unknown = 0,
  ContextFlushPending,
  ContextSubmitFrequency,
  ContextExplicitSynchronize,
  ContextReadback,
  ContextShutdown,
  StackPlannedRecordingEnd,
  StackOwnerPhaseBoundary,
  StackOwnerNorm1,
  StackOwnerNorm2,
  StackOwnerAttention,
  StackOwnerLinear,
  StackOwnerResidual,
  NativeLayerNormMetadata,
  NativeLayerNormUniform,
  AttentionMetadata,
  LinearMetadata,
  ConvMetadata,
  AddResidualMetadata,
  DescriptorRecycle,
  CommandBufferRecycle,
  StagingBufferRecycle,
  UniformBufferRecycle,
  MetadataBufferRecycle,
  BenchmarkReadback,
  BenchmarkSetup,
  DebugValidation,
};

enum class VulkanRetiredResourceKind : uint8_t {
  Unknown = 0,
  Buffer,
  Image,
  UniformBuffer,
  MetadataBuffer,
  DescriptorSet,
  DescriptorPool,
  CommandBuffer,
  StagingBuffer,
  QueryBuffer,
  Other,
};

enum class VulkanRetiredResourceRole : uint8_t {
  Unknown = 0,
  NativeLayerNormUniform,
  NativeLayerNormMetadata,
  AttentionMetadata,
  LinearMetadata,
  ConvMetadata,
  ResidualAddMetadata,
  StackInternalTemp,
  StackNorm1Output,
  StackQkvOutput,
  StackQView,
  StackKView,
  StackVView,
  StackAttentionOutput,
  StackProjOutput,
  StackResidual1Output,
  StackNorm2Output,
  StackFc1GeluOutput,
  StackFc2Output,
  StackResidual2Output,
  StackRequestedOutput,
  StackFinalOutput,
  DescriptorRecycle,
  CommandBufferRecycle,
  ReadbackStaging,
  SetupStaging,
};

enum class VulkanStackTempLifetimeSafety : uint8_t {
  Unknown = 0,
  SafeToDeferUntilStackSubmit,
  SafeToDeferUntilStackScopeEnd,
  MustRetireAtPhaseBoundary,
  EscapesAsRequestedIntermediate,
  EscapesAsFinalOutput,
  AliasesRuntimeInput,
  AliasesRuntimeOutput,
  UnsafeUnknownConsumer,
};

enum class VulkanStackRetireProvenanceSource : uint8_t {
  Unknown = 0,
  TensorAllocation,
  StorageReallocation,
  ProgramScratchArenaBackingStorage,
};

struct VulkanStackRetireProvenance final {
  bool defined = false;
  VulkanVisionStackPhase phase = VulkanVisionStackPhase::Unknown;
  int64_t block_index = -1;
  VulkanRetiredResourceRole producer_role = VulkanRetiredResourceRole::Unknown;
  VulkanStackRetireProvenanceSource source =
      VulkanStackRetireProvenanceSource::Unknown;
  uint64_t source_identity = 0u;
  uint64_t source_generation = 0u;
  bool has_last_use_proof = false;
  VulkanVisionStackPhase expected_consumer_phase =
      VulkanVisionStackPhase::Unknown;
  int64_t expected_consumer_block_index = -1;
  bool final_consumer_before_stack_submit = false;
  bool internal_non_escaping = false;
  bool aliases_runtime_input = false;
  bool aliases_runtime_output = false;
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
};

struct VulkanStackRawResourceAllocationProof final {
  bool has_generation = false;
  bool has_byte_range = false;
  uint64_t allocation_id = 0u;
  uint64_t allocation_generation = 0u;
  uint64_t byte_offset = 0u;
  uint64_t byte_range = 0u;
  uint64_t allocated_bytes = 0u;
};

struct VulkanStackLastUseProof final {
  VulkanVisionStackPhase producer_phase = VulkanVisionStackPhase::Unknown;
  int64_t producer_block_index = -1;
  VulkanRetiredResourceRole producer_role = VulkanRetiredResourceRole::Unknown;
  std::vector<int64_t> shape;
  int64_t dtype = -1;
  VulkanVisionStackPhase expected_consumer_phase =
      VulkanVisionStackPhase::Unknown;
  int64_t expected_consumer_block_index = -1;
  bool final_consumer_before_stack_submit = false;
  bool internal_non_escaping = false;
  bool escapes_stack = false;
  bool requested_intermediate = false;
  bool final_output = false;
  bool aliases_runtime_input = false;
  bool aliases_runtime_output = false;
};

struct VulkanStackPlannedDispatchPosition final {
  VulkanVisionStackPhase phase = VulkanVisionStackPhase::Unknown;
  int64_t block_index = -1;
  uint64_t planned_position = 0u;
};

struct VulkanStackOutputDeviceConsumerRegistration final {
  int64_t captured_block_index = -1;
  std::string captured_substep = "unknown";
  std::string output_role = "unknown";
  std::vector<int64_t> output_shape;
  std::string stack_context_id = "unknown";
  std::string stack_session_id = "unknown";
  std::string stack_plan_id = "unknown";
  std::string output_layout = "unknown";
  std::string strip_or_view_relation = "unknown";
  std::string downstream_consumer_id = "unknown";
  std::string downstream_consumer_context = "unknown";
  int64_t expected_consumer_input_index = -1;
  std::vector<int64_t> expected_consumer_shape;
  std::string expected_consumer_layout = "unknown";
  bool consumer_in_same_planned_region = false;
  bool python_public_boundary_before_consumption = true;
  bool host_visible_boundary_before_consumption = true;
  bool host_visible_access_before_consumption = true;
  bool host_readback_before_consumption = true;
};

class TORCH_API VulkanStackPlannedDispatchPositionScope final {
 public:
  explicit VulkanStackPlannedDispatchPositionScope(
      std::vector<VulkanStackPlannedDispatchPosition>);
  ~VulkanStackPlannedDispatchPositionScope();

  VulkanStackPlannedDispatchPositionScope(
      const VulkanStackPlannedDispatchPositionScope&) = delete;
  VulkanStackPlannedDispatchPositionScope& operator=(
      const VulkanStackPlannedDispatchPositionScope&) = delete;

 private:
  std::vector<VulkanStackPlannedDispatchPosition> previous_;
};

class TORCH_API VulkanStackLastUseProofScope final {
 public:
  explicit VulkanStackLastUseProofScope(std::vector<VulkanStackLastUseProof>);
  ~VulkanStackLastUseProofScope();

  VulkanStackLastUseProofScope(const VulkanStackLastUseProofScope&) = delete;
  VulkanStackLastUseProofScope& operator=(
      const VulkanStackLastUseProofScope&) = delete;

 private:
  std::vector<VulkanStackLastUseProof> previous_;
};

struct VulkanSubmitOriginCounters final {
  std::atomic<uint64_t> total_queue_submits{0u};
  std::atomic<uint64_t> normal_cmd_submit_frequency{0u};
  std::atomic<uint64_t> stack_planned_recording_submit{0u};
  std::atomic<uint64_t> pre_stack_flush{0u};
  std::atomic<uint64_t> post_stack_flush{0u};
  std::atomic<uint64_t> explicit_synchronize{0u};
  std::atomic<uint64_t> tensor_cpu_readback{0u};
  std::atomic<uint64_t> fallback_readback{0u};
  std::atomic<uint64_t> retire_queue_drain{0u};
  std::atomic<uint64_t> profiling_timestamp_reset{0u};
  std::atomic<uint64_t> profiling_timestamp_readback{0u};
  std::atomic<uint64_t> shutdown{0u};
  std::atomic<uint64_t> debug_validation{0u};
  std::atomic<uint64_t> conv_prepack_upload{0u};
  std::atomic<uint64_t> unknown{0u};
};

struct VulkanSubmitOriginPhaseCounters final {
  std::array<
      std::array<std::atomic<uint64_t>, kNumSubmitPhases>,
      kNumSubmitOrigins>
      counts{};
};

struct VulkanRetireDrainCounters final {
  std::atomic<uint64_t> total{0u};
  std::atomic<uint64_t> queue_submit_count{0u};
  std::atomic<uint64_t> blocking_wait_count{0u};
  std::atomic<uint64_t> poll_only_count{0u};
  std::atomic<uint64_t> pending_resource_count_total{0u};
  std::atomic<uint64_t> pending_bytes_total{0u};
  std::atomic<uint64_t> explicit_drain{0u};
  std::atomic<uint64_t> shutdown{0u};
  std::atomic<uint64_t> resource_pressure{0u};
  std::atomic<uint64_t> descriptor_pool_pressure{0u};
  std::atomic<uint64_t> command_buffer_recycle{0u};
  std::atomic<uint64_t> readback_preparation{0u};
  std::atomic<uint64_t> synchronize{0u};
  std::atomic<uint64_t> stack_scope_end{0u};
  std::atomic<uint64_t> decoder_phase{0u};
  std::atomic<uint64_t> setup_phase{0u};
  std::atomic<uint64_t> debug_validation{0u};
  std::atomic<uint64_t> unknown{0u};
};

struct VulkanRetireCallSiteCounter final {
  std::atomic<uint64_t> total{0u};
  std::atomic<uint64_t> queue_submit_count{0u};
  std::atomic<uint64_t> blocking_wait_count{0u};
  std::atomic<uint64_t> poll_only_count{0u};
  std::atomic<uint64_t> pending_resource_count_total{0u};
  std::atomic<uint64_t> pending_bytes_total{0u};
};

struct VulkanStackInternalTempRetireBatchCounters final {
  std::atomic<uint64_t> total_attempts{0u};
  std::atomic<uint64_t> batch_candidate_count{0u};
  std::atomic<uint64_t> batch_candidate_bytes{0u};
  std::atomic<uint64_t> batch_accepted_count{0u};
  std::atomic<uint64_t> batch_accepted_bytes{0u};
  std::atomic<uint64_t> batch_rejected_count{0u};
  std::atomic<uint64_t> batch_rejected_bytes{0u};
  std::atomic<uint64_t> submitted_batch_count{0u};
  std::atomic<uint64_t> submitted_batch_bytes{0u};
  std::atomic<uint64_t> rejected_not_target_role{0u};
  std::atomic<uint64_t> rejected_missing_proof{0u};
  std::atomic<uint64_t> rejected_not_internal_non_escaping{0u};
  std::atomic<uint64_t> rejected_consumer_after_submit{0u};
  std::atomic<uint64_t> rejected_requested_intermediate{0u};
  std::atomic<uint64_t> rejected_final_output{0u};
  std::atomic<uint64_t> rejected_alias{0u};
  std::atomic<uint64_t> rejected_runtime_alias{0u};
  std::atomic<uint64_t> rejected_lifetime{0u};
  std::atomic<uint64_t> rejected_not_stack_recording{0u};
};

struct VulkanStackRetireDrainBlockerCounters final {
  std::atomic<uint64_t> total_drains{0u};
  std::atomic<uint64_t> queue_submit_drains{0u};
  std::atomic<uint64_t> drains_with_old_path_pending{0u};
  std::atomic<uint64_t> drains_with_only_already_batched{0u};
  std::atomic<uint64_t> drains_qkv_would_remove{0u};
  std::atomic<uint64_t> drains_blocked_requested_intermediate{0u};
  std::atomic<uint64_t> drains_blocked_missing_proof{0u};
  std::atomic<uint64_t> drains_blocked_generic_stack_internal_temp{0u};
  std::atomic<uint64_t> drains_blocked_metadata_or_uniform{0u};
  std::atomic<uint64_t> drains_blocked_other_roles{0u};
  std::atomic<uint64_t> old_path_pending_count{0u};
  std::atomic<uint64_t> old_path_pending_bytes{0u};
  std::atomic<uint64_t> qkv_hypothetical_count{0u};
  std::atomic<uint64_t> qkv_hypothetical_bytes{0u};
  std::atomic<uint64_t> skipped_no_old_path_pending{0u};
  std::atomic<uint64_t> skipped_no_pending_command_work{0u};
};

constexpr uint64_t kStackSubresourceLifetimeDryRunBlockBudgetBytes =
    4u * 1024u * 1024u;
constexpr uint64_t kStackSubresourceLifetimeDryRunScopeBudgetBytes =
    32u * 1024u * 1024u;

struct VulkanStackSubresourceLifetimeDryRunCounters final {
  std::atomic<uint64_t> total_groups{0u};
  std::atomic<uint64_t> queue_submit_groups{0u};
  std::atomic<uint64_t> groups_with_old_path_pending{0u};
  std::atomic<uint64_t> all_safe_group_eligible{0u};
  std::atomic<uint64_t> would_remove_submit_drains{0u};
  std::atomic<uint64_t> actual_removed_submit_drains{0u};
  std::atomic<uint64_t> peak_extra_live_bytes_estimate{0u};
  std::atomic<uint64_t> skipped_no_old_path_pending{0u};
  std::atomic<uint64_t> proven_stack_activation_count{0u};
  std::atomic<uint64_t> missing_stack_activation_proof_count{0u};
  std::atomic<uint64_t> attention_subresource_count{0u};
  std::atomic<uint64_t> attention_score_probability_subresource_count{0u};
  std::atomic<uint64_t> layernorm_stat_buffer_count{0u};
  std::atomic<uint64_t> layernorm_internal_stat_buffer_count{0u};
  std::atomic<uint64_t> metadata_uniform_count{0u};
  std::atomic<uint64_t> raw_no_provenance_count{0u};
  std::atomic<uint64_t> stack_internal_raw_missing_generation_count{0u};
  std::atomic<uint64_t> stack_internal_raw_generation_range_count{0u};
  std::atomic<uint64_t> truly_unknown_raw_resource_count{0u};
  std::atomic<uint64_t> host_visible_or_requested_output_count{0u};
  std::atomic<uint64_t> allocator_or_scratch_backing_count{0u};
  std::atomic<uint64_t> proven_stack_activation_bytes{0u};
  std::atomic<uint64_t> missing_stack_activation_proof_bytes{0u};
  std::atomic<uint64_t> attention_subresource_bytes{0u};
  std::atomic<uint64_t> attention_score_probability_subresource_bytes{0u};
  std::atomic<uint64_t> layernorm_stat_buffer_bytes{0u};
  std::atomic<uint64_t> layernorm_internal_stat_buffer_bytes{0u};
  std::atomic<uint64_t> metadata_uniform_bytes{0u};
  std::atomic<uint64_t> raw_no_provenance_bytes{0u};
  std::atomic<uint64_t> stack_internal_raw_missing_generation_bytes{0u};
  std::atomic<uint64_t> stack_internal_raw_generation_range_bytes{0u};
  std::atomic<uint64_t> truly_unknown_raw_resource_bytes{0u};
  std::atomic<uint64_t> host_visible_or_requested_output_bytes{0u};
  std::atomic<uint64_t> allocator_or_scratch_backing_bytes{0u};
  std::atomic<uint64_t> rejected_unsafe_resource_class{0u};
  std::atomic<uint64_t> rejected_over_block_budget{0u};
  std::atomic<uint64_t> rejected_over_scope_budget{0u};
  std::atomic<uint64_t> rejected_large_backing{0u};
  std::atomic<uint64_t> attention_buffer_generation_range_missing_stack_proof_count{
      0u};
  std::atomic<uint64_t> attention_raw_generation_range_missing_stack_proof_count{
      0u};
  std::atomic<uint64_t> attention_provenance_missing_last_use_count{0u};
  std::atomic<uint64_t> attention_unknown_subresource_count{0u};
  std::atomic<uint64_t> attention_buffer_generation_range_missing_stack_proof_bytes{
      0u};
  std::atomic<uint64_t> attention_raw_generation_range_missing_stack_proof_bytes{
      0u};
  std::atomic<uint64_t> attention_provenance_missing_last_use_bytes{0u};
  std::atomic<uint64_t> attention_unknown_subresource_bytes{0u};
  std::atomic<uint64_t>
      attention_score_probability_range_missing_alias_escape_proof_count{0u};
  std::atomic<uint64_t>
      attention_raw_auxiliary_range_missing_alias_escape_proof_count{0u};
  std::atomic<uint64_t>
      attention_score_probability_range_missing_alias_escape_proof_bytes{0u};
  std::atomic<uint64_t>
      attention_raw_auxiliary_range_missing_alias_escape_proof_bytes{0u};
  std::atomic<uint64_t>
      attention_score_probability_range_non_escape_last_consumer_count{0u};
  std::atomic<uint64_t>
      attention_raw_auxiliary_range_non_escape_last_consumer_count{0u};
  std::atomic<uint64_t>
      attention_score_probability_range_non_escape_last_consumer_bytes{0u};
  std::atomic<uint64_t>
      attention_raw_auxiliary_range_non_escape_last_consumer_bytes{0u};
  std::atomic<uint64_t>
      stack_internal_temp_raw_generation_range_missing_last_consumer_count{0u};
  std::atomic<uint64_t>
      stack_qkv_output_raw_generation_range_non_escape_last_consumer_count{0u};
  std::atomic<uint64_t>
      stack_proj_output_raw_generation_range_non_escape_last_consumer_count{0u};
  std::atomic<uint64_t>
      stack_residual1_output_raw_generation_range_non_escape_last_consumer_count{
          0u};
  std::atomic<uint64_t>
      stack_internal_temp_raw_generation_range_missing_last_consumer_bytes{0u};
  std::atomic<uint64_t>
      stack_qkv_output_raw_generation_range_non_escape_last_consumer_bytes{0u};
  std::atomic<uint64_t>
      stack_proj_output_raw_generation_range_non_escape_last_consumer_bytes{0u};
  std::atomic<uint64_t>
      stack_residual1_output_raw_generation_range_non_escape_last_consumer_bytes{
          0u};
  std::atomic<uint64_t> phase_boundary_total_groups{0u};
  std::atomic<uint64_t> phase_boundary_all_safe_group_eligible{0u};
  std::atomic<uint64_t> phase_boundary_would_remove_explicit_synchronizes{0u};
  std::atomic<uint64_t> phase_boundary_actual_removed_explicit_synchronizes{0u};
  std::atomic<uint64_t> phase_boundary_rejected_unsafe_resource_class{0u};
  std::atomic<uint64_t> phase_boundary_rejected_over_block_budget{0u};
  std::atomic<uint64_t> phase_boundary_rejected_over_scope_budget{0u};
  std::atomic<uint64_t> phase_boundary_rejected_large_backing{0u};
  std::atomic<uint64_t> phase_boundary_stack_activation_carry_proof_count{0u};
  std::atomic<uint64_t> phase_boundary_stack_activation_carry_proof_bytes{0u};
};

class VulkanSubmitPhaseScope final {
 public:
  explicit VulkanSubmitPhaseScope(VulkanSubmitPhase phase);
  ~VulkanSubmitPhaseScope();
  VulkanSubmitPhaseScope(const VulkanSubmitPhaseScope&) = delete;
  VulkanSubmitPhaseScope& operator=(const VulkanSubmitPhaseScope&) = delete;

 private:
  VulkanSubmitPhase previous_;
};

class VulkanRetiredResourceScope final {
 public:
  VulkanRetiredResourceScope(
      VulkanRetiredResourceKind kind,
      VulkanRetiredResourceRole role);
  ~VulkanRetiredResourceScope();
  VulkanRetiredResourceScope(const VulkanRetiredResourceScope&) = delete;
  VulkanRetiredResourceScope& operator=(const VulkanRetiredResourceScope&) =
      delete;

 private:
  VulkanRetiredResourceKind previous_kind_;
  VulkanRetiredResourceRole previous_role_;
};

TORCH_API VulkanSyncCounters& vulkan_sync_counters();
TORCH_API void reset_vulkan_sync_counters();
TORCH_API VulkanSubmitOriginCounters& vulkan_submit_origin_counters();
TORCH_API void reset_vulkan_submit_origin_counters();
TORCH_API void note_vulkan_queue_submit(VulkanSubmitOrigin origin);
TORCH_API VulkanSubmitOriginPhaseCounters&
vulkan_submit_origin_phase_counters();
TORCH_API void reset_vulkan_submit_origin_phase_counters();
TORCH_API std::vector<std::string> submit_origin_phase_snapshot();
TORCH_API VulkanRetireDrainCounters& vulkan_retire_drain_counters();
TORCH_API void reset_vulkan_retire_drain_counters();
TORCH_API std::vector<int64_t> retire_drain_counters_snapshot();
TORCH_API std::vector<std::string> retire_call_site_counters_snapshot();
TORCH_API void reset_retire_call_site_counters();
TORCH_API std::vector<std::string> retired_resource_aggregate_snapshot();
TORCH_API void reset_retired_resource_aggregate();
TORCH_API std::vector<std::string> stack_temp_lifetime_safety_snapshot();
TORCH_API void reset_stack_temp_lifetime_safety_snapshot();
TORCH_API VulkanStackInternalTempRetireBatchCounters&
stack_internal_temp_retire_batch_counters();
TORCH_API std::vector<int64_t> stack_internal_temp_retire_batch_counters_snapshot();
TORCH_API std::vector<std::string> stack_internal_temp_retire_batch_snapshot();
TORCH_API void reset_stack_internal_temp_retire_batch_counters();
TORCH_API VulkanStackRetireDrainBlockerCounters&
stack_retire_drain_blocker_counters();
TORCH_API std::vector<int64_t> stack_retire_drain_blocker_counters_snapshot();
TORCH_API std::vector<std::string> stack_retire_drain_blocker_snapshot();
TORCH_API void reset_stack_retire_drain_blocker_counters();
TORCH_API std::vector<std::string> region_lifetime_submit_attribution_snapshot();
TORCH_API void reset_region_lifetime_submit_attribution();
TORCH_API VulkanStackSubresourceLifetimeDryRunCounters&
stack_subresource_lifetime_dry_run_counters();
TORCH_API std::vector<int64_t>
stack_subresource_lifetime_dry_run_counters_snapshot();
TORCH_API std::vector<std::string>
stack_subresource_lifetime_dry_run_snapshot();
TORCH_API void reset_stack_subresource_lifetime_dry_run_counters();
TORCH_API std::vector<std::string> stack_scratch_arena_lifetime_snapshot();
TORCH_API void reset_stack_scratch_arena_lifetime_snapshot();
TORCH_API const char* submit_origin_name(VulkanSubmitOrigin origin);
TORCH_API const char* submit_phase_name(VulkanSubmitPhase phase);
TORCH_API const char* retire_call_site_name(VulkanRetireCallSite callsite);
TORCH_API const char* retired_resource_kind_name(VulkanRetiredResourceKind kind);
TORCH_API const char* retired_resource_role_name(VulkanRetiredResourceRole role);
TORCH_API const char* stack_temp_lifetime_safety_name(
    VulkanStackTempLifetimeSafety safety);
TORCH_API bool is_stack_temp_retired_resource_role(
    VulkanRetiredResourceRole role);
TORCH_API VulkanStackTempLifetimeSafety stack_retire_lifetime_safety_for_resource(
    VulkanRetiredResourceRole role,
    const VulkanStackRetireProvenance& provenance);
TORCH_API const char* stack_retire_drain_blocker_reason(
    VulkanRetiredResourceKind kind,
    VulkanRetiredResourceRole role,
    const VulkanStackRetireProvenance& provenance,
    bool qkv_would_batch);
TORCH_API VulkanRetiredResourceRole stack_retired_resource_role_for_phase(
    VulkanVisionStackPhase phase);
TORCH_API VulkanStackRetireProvenance current_stack_retire_provenance(
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides,
    int64_t dtype,
    bool direct_buffer,
    bool buffer_storage,
    bool image_storage,
    bool alias_or_view,
    VulkanStackRetireProvenanceSource source =
        VulkanStackRetireProvenanceSource::TensorAllocation);
TORCH_API bool is_safe_stack_temp_retire_batch_candidate(
    const VulkanStackRetireProvenance& provenance);
TORCH_API bool is_qkv_stack_temp_retire_batch_candidate(
    const VulkanStackRetireProvenance& provenance);
TORCH_API void note_stack_internal_temp_retire_batch_decision(
    const VulkanStackRetireProvenance& provenance,
    uint64_t bytes,
    bool stack_recording_active,
    bool accepted);
TORCH_API void note_stack_internal_temp_retire_batch_submitted(uint64_t bytes);
TORCH_API void note_stack_retire_drain_blocker_resource(
    VulkanRetiredResourceKind kind,
    VulkanRetiredResourceRole role,
    VulkanSubmitPhase phase,
    VulkanRetireCallSite callsite,
    uint64_t bytes,
    bool qkv_would_batch,
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof,
    const std::string& allocation_label);
TORCH_API void note_stack_retire_drain_blocker_summary(
    VulkanSubmitPhase phase,
    VulkanRetireCallSite callsite,
    bool queue_submit,
    uint64_t old_path_pending_count,
    uint64_t old_path_pending_bytes,
    uint64_t qkv_hypothetical_count,
    uint64_t qkv_hypothetical_bytes,
    bool qkv_would_remove_drain,
    bool only_already_batched,
    bool blocked_requested_intermediate,
    bool blocked_missing_proof,
    bool blocked_generic_stack_internal_temp,
    bool blocked_metadata_or_uniform,
    bool blocked_other_roles,
    bool skipped_no_old_path_pending,
    bool skipped_no_pending_command_work);
TORCH_API void note_stack_retire_drain_copresent_group(
    VulkanSubmitPhase phase,
    VulkanRetireCallSite callsite,
    bool queue_submit,
    uint64_t old_path_pending_count,
    uint64_t old_path_pending_bytes,
    uint64_t qkv_hypothetical_count,
    bool qkv_would_remove_drain,
    bool skipped_no_old_path_pending,
    const std::string& signature,
    const std::string& blockers);
TORCH_API void note_region_lifetime_submit_attribution_group(
    VulkanSubmitOrigin origin,
    VulkanSubmitPhase phase,
    VulkanRetireCallSite callsite,
    bool queue_submit,
    bool had_pending_work,
    uint64_t pending_resource_count,
    uint64_t pending_bytes,
    const std::string& signature,
    const std::string& blockers);
TORCH_API void note_region_lifetime_submit_attribution_resource(
    VulkanSubmitOrigin origin,
    VulkanSubmitPhase phase,
    VulkanRetireCallSite callsite,
    VulkanRetiredResourceKind kind,
    VulkanRetiredResourceRole role,
    uint64_t bytes,
    const char* reason,
    VulkanStackTempLifetimeSafety safety,
    bool queue_submit,
    bool had_pending_work,
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof,
    const std::string& allocation_label);
TORCH_API const char* stack_subresource_lifetime_dry_run_resource_class(
    VulkanRetiredResourceKind kind,
    VulkanRetiredResourceRole role,
    const VulkanStackRetireProvenance& provenance,
    bool qkv_would_batch,
    const VulkanStackRawResourceAllocationProof& allocation_proof);
TORCH_API bool stack_subresource_lifetime_dry_run_resource_is_safe(
    const char* resource_class);
TORCH_API bool stack_subresource_lifetime_dry_run_has_formal_norm2_last_use_proof(
    VulkanRetiredResourceKind kind,
    VulkanRetiredResourceRole role,
    const char* resource_class,
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof,
    const std::string& allocation_label);
TORCH_API bool
stack_subresource_lifetime_dry_run_has_formal_stack_owner_last_use_proof(
    VulkanRetiredResourceKind kind,
    VulkanRetiredResourceRole role,
    const char* resource_class,
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof,
    const std::string& allocation_label,
    VulkanRetireCallSite callsite);
TORCH_API bool stack_subresource_lifetime_dry_run_is_large_backing(
    VulkanRetiredResourceRole role,
    uint64_t bytes,
    const VulkanStackRetireProvenance& provenance);
TORCH_API void note_stack_subresource_lifetime_dry_run_resource(
    VulkanRetiredResourceKind kind,
    VulkanRetiredResourceRole role,
    VulkanSubmitPhase phase,
    VulkanRetireCallSite callsite,
    uint64_t bytes,
    const char* resource_class,
    bool safe_candidate,
    bool large_backing,
    bool formal_last_use_proof,
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof,
    const std::string& allocation_label);
TORCH_API void note_stack_subresource_lifetime_dry_run_group(
    VulkanSubmitPhase phase,
    VulkanRetireCallSite callsite,
    bool queue_submit,
    uint64_t old_path_pending_count,
    uint64_t old_path_pending_bytes,
    uint64_t safe_candidate_count,
    uint64_t safe_candidate_bytes,
    bool all_safe_group_eligible,
    bool would_remove_submit_drain,
    bool actual_removed_submit_drain,
    const std::string& budget_reject,
    const std::string& signature,
    const std::string& blockers);
TORCH_API void note_stack_phase_boundary_lifetime_dry_run_group(
    VulkanSubmitPhase phase,
    VulkanRetireCallSite callsite,
    bool queue_submit,
    uint64_t old_path_pending_count,
    uint64_t old_path_pending_bytes,
    uint64_t safe_candidate_count,
    uint64_t safe_candidate_bytes,
    bool all_safe_group_eligible,
    bool would_remove_explicit_synchronize,
    bool actual_removed_explicit_synchronize,
    uint64_t block_budget_bytes,
    uint64_t scope_budget_bytes,
    const std::string& budget_reject,
    const std::string& signature,
    const std::string& blockers);
TORCH_API void note_stack_region_boundary_submit_plan(
    VulkanSubmitPhase phase,
    VulkanRetireCallSite callsite,
    bool queue_submit,
    uint64_t old_path_pending_count,
    uint64_t old_path_pending_bytes,
    uint64_t safe_candidate_count,
    uint64_t safe_candidate_bytes,
    uint64_t command_buffer_recording_id,
    uint64_t submit_epoch_before,
    uint64_t submit_epoch_after,
    uint64_t pending_dispatch_count,
    const std::string& budget_reject,
    const std::string& resource_signature,
    const std::string& allocation_signature,
    const std::string& raw_provenance_signature,
    const std::string& blockers);
TORCH_API bool maybe_elide_stack_region_boundary_submit_canary(
    VulkanSubmitPhase phase,
    VulkanRetireCallSite callsite);
TORCH_API VulkanSubmitPhase current_submit_phase();
TORCH_API void set_submit_phase(VulkanSubmitPhase phase);
TORCH_API void reset_submit_phase();
TORCH_API VulkanRetiredResourceKind current_retired_resource_kind();
TORCH_API VulkanRetiredResourceRole current_retired_resource_role();
TORCH_API void note_vulkan_retire_drain(
    VulkanRetireDrainReason reason,
    VulkanRetireCallSite callsite,
    bool queue_submit,
    bool blocking_wait,
    uint64_t pending_resource_count,
    uint64_t pending_bytes);
TORCH_API void note_vulkan_retired_resource(
    VulkanRetiredResourceKind kind,
    VulkanRetiredResourceRole role,
    VulkanSubmitPhase phase,
    VulkanRetireCallSite callsite,
    uint64_t bytes,
    bool queue_submit,
    bool blocking_wait,
    bool poll_only,
    const VulkanStackRetireProvenance& provenance = {});

TORCH_API void note_vulkan_queue_wait_idle();
TORCH_API void note_vulkan_forced_sync(
    VulkanForcedSyncReason reason = VulkanForcedSyncReason::Unknown);

TORCH_API const char* vision_stack_phase_name(VulkanVisionStackPhase phase);
TORCH_API const char* stack_tensor_lifetime_name(
    VulkanStackTensorLifetimeClass lifetime);
TORCH_API VulkanVisionStackPhase current_vision_stack_phase();
TORCH_API int64_t current_vision_stack_block_index();
TORCH_API bool inside_vision_stack_phase();
TORCH_API bool vision_stack_capture_dependency_active();
TORCH_API bool vision_stack_capture_dependency_reaches_block(
    int64_t block_index);

TORCH_API void begin_stack_dispatch_dependency_recording_scope();
TORCH_API void end_stack_dispatch_dependency_recording_scope();
TORCH_API void set_stack_region_command_buffer_diagnostic_context(
    uint64_t command_buffer_recording_id,
    uint64_t submit_epoch_before);
TORCH_API void note_vulkan_stack_pre_dispatch_insertion_point(
    const char* shader_name);
TORCH_API void note_vulkan_stack_live_descriptor_binding(
    uint32_t binding_idx,
    const char* shader_name,
    const VulkanBuffer& buffer);
TORCH_API void note_vulkan_stack_live_image_descriptor_binding(
    uint32_t binding_idx,
    const char* shader_name,
    const VulkanImage& image);
TORCH_API void note_vulkan_stack_descriptor_set_update_generation(
    const char* shader_name,
    uint64_t descriptor_set_handle_token,
    uint64_t update_generation,
    uint64_t write_count);
TORCH_API void note_vulkan_stack_pre_dispatch_proof_table_descriptor(
    uint32_t binding_idx,
    const char* shader_name,
    const VulkanBuffer& buffer);
TORCH_API bool maybe_insert_vulkan_stack_barrier_only_canary_descriptor(
    uint32_t binding_idx,
    const char* shader_name,
    const VulkanBuffer& buffer,
    PipelineBarrier& pipeline_barrier);
TORCH_API void note_vulkan_stack_dispatch(const char* shader_name);
TORCH_API void note_stack_owner_dispatch_dependency_dry_run(
    VulkanRetiredResourceKind kind,
    VulkanRetiredResourceRole role,
    VulkanSubmitPhase phase,
    VulkanRetireCallSite callsite,
    bool queue_submit,
    uint64_t bytes,
    const char* resource_class,
    bool formal_last_use_proof,
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof,
    const std::string& allocation_label);
TORCH_API void note_vulkan_stack_allocation(
    const char* role,
    VulkanStackTensorLifetimeClass lifetime,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    int64_t dtype,
    bool direct_buffer,
    bool buffer_storage,
    bool image_storage,
    bool escapes_stack,
    bool requested_intermediate,
    uint64_t bytes);
TORCH_API void note_stack_raw_resource_producer_registration(
    uint64_t allocation_id,
    uint64_t allocation_generation,
    uint64_t byte_offset,
    uint64_t byte_range,
    uint64_t allocated_bytes,
    const char* kind,
    const std::string& allocation_label,
    const std::string& allocation_role,
    bool owns_memory);
TORCH_API void note_stack_output_device_consumer_registration(
    const VulkanStackOutputDeviceConsumerRegistration& registration);
TORCH_API std::vector<std::string> stack_dispatch_aggregate_snapshot();
TORCH_API std::vector<std::string> stack_allocation_aggregate_snapshot();
TORCH_API std::vector<std::string>
stack_output_device_consumer_registration_snapshot();
TORCH_API std::vector<std::string>
stack_dispatch_dependency_dry_run_snapshot();
TORCH_API void reset_stack_dispatch_aggregate();
TORCH_API void reset_stack_allocation_aggregate();
TORCH_API void reset_stack_dispatch_dependency_dry_run();

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
