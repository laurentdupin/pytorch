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
};

constexpr size_t kNumSubmitOrigins = 13u;

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

struct VulkanStackRetireProvenance final {
  bool defined = false;
  VulkanVisionStackPhase phase = VulkanVisionStackPhase::Unknown;
  int64_t block_index = -1;
  VulkanRetiredResourceRole producer_role = VulkanRetiredResourceRole::Unknown;
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
TORCH_API const char* submit_origin_name(VulkanSubmitOrigin origin);
TORCH_API const char* submit_phase_name(VulkanSubmitPhase phase);
TORCH_API const char* retire_call_site_name(VulkanRetireCallSite callsite);
TORCH_API const char* retired_resource_kind_name(VulkanRetiredResourceKind kind);
TORCH_API const char* retired_resource_role_name(VulkanRetiredResourceRole role);
TORCH_API const char* stack_temp_lifetime_safety_name(
    VulkanStackTempLifetimeSafety safety);
TORCH_API VulkanRetiredResourceRole stack_retired_resource_role_for_phase(
    VulkanVisionStackPhase phase);
TORCH_API VulkanStackRetireProvenance current_stack_retire_provenance(
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides,
    int64_t dtype,
    bool direct_buffer,
    bool buffer_storage,
    bool image_storage,
    bool alias_or_view);
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

TORCH_API void note_vulkan_stack_dispatch(const char* shader_name);
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
TORCH_API std::vector<std::string> stack_dispatch_aggregate_snapshot();
TORCH_API std::vector<std::string> stack_allocation_aggregate_snapshot();
TORCH_API void reset_stack_dispatch_aggregate();
TORCH_API void reset_stack_allocation_aggregate();

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
