#pragma once

#ifdef USE_VULKAN_API

#include <c10/macros/Export.h>

#include <array>
#include <atomic>
#include <cstddef>
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

struct VulkanSyncCounters final {
  std::atomic<uint64_t> compute_dispatch_count{0u};
  std::atomic<uint64_t> submit_compute_job_count{0u};
  std::atomic<uint64_t> retire_cleanup_callback_count{0u};
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

struct VulkanGraphProgramInvocationCounters final {
  std::atomic<uint64_t> scope_begun_count{0u};
  std::atomic<uint64_t> normal_submit_token_capture_count{0u};
  std::atomic<uint64_t> aborted_submit_count{0u};
  std::atomic<uint64_t> rejected_incompatible_state_count{0u};
  std::atomic<uint64_t> bounded_region_host_sync_rejected_count{0u};
  std::atomic<uint64_t> scratch_captured_count{0u};
  std::atomic<uint64_t> scratch_reused_count{0u};
  std::atomic<uint64_t> scratch_transient_overflow_count{0u};
  std::atomic<uint64_t> scratch_retire_enqueued_count{0u};
  std::atomic<uint64_t> scratch_immediate_release_count{0u};
};

enum class VulkanSubmitOrigin : uint8_t {
  Unknown = 0,
  NormalCmdSubmitFrequency,
  StackPlannedRecordingSubmit,
  PreStackFlush,
  PostStackFlush,
  ExplicitSynchronize,
  TensorCpuReadback,
  HostUpload,
  FallbackReadback,
  RetireQueueDrain,
  ProfilingTimestampReset,
  ProfilingTimestampReadback,
  ContextShutdown,
  DebugValidation,
  ConvPrepackUpload,
  PendingCommandFlush,
};

constexpr size_t kNumSubmitOrigins = 16u;

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

struct VulkanSubmitOriginCounters final {
  std::atomic<uint64_t> total_queue_submits{0u};
  std::atomic<uint64_t> normal_cmd_submit_frequency{0u};
  std::atomic<uint64_t> stack_planned_recording_submit{0u};
  std::atomic<uint64_t> pre_stack_flush{0u};
  std::atomic<uint64_t> post_stack_flush{0u};
  std::atomic<uint64_t> explicit_synchronize{0u};
  std::atomic<uint64_t> tensor_cpu_readback{0u};
  std::atomic<uint64_t> host_upload{0u};
  std::atomic<uint64_t> fallback_readback{0u};
  std::atomic<uint64_t> retire_queue_drain{0u};
  std::atomic<uint64_t> profiling_timestamp_reset{0u};
  std::atomic<uint64_t> profiling_timestamp_readback{0u};
  std::atomic<uint64_t> shutdown{0u};
  std::atomic<uint64_t> debug_validation{0u};
  std::atomic<uint64_t> conv_prepack_upload{0u};
  std::atomic<uint64_t> pending_command_flush{0u};
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

TORCH_API VulkanSyncCounters& vulkan_sync_counters();
TORCH_API void reset_vulkan_sync_counters();
TORCH_API VulkanGraphProgramInvocationCounters&
vulkan_graph_program_invocation_counters();
TORCH_API void reset_vulkan_graph_program_invocation_counters();
TORCH_API std::vector<int64_t> graph_program_invocation_counters_snapshot();
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
TORCH_API const char* submit_origin_name(VulkanSubmitOrigin origin);
TORCH_API const char* submit_phase_name(VulkanSubmitPhase phase);
TORCH_API const char* retire_call_site_name(VulkanRetireCallSite callsite);
TORCH_API VulkanSubmitPhase current_submit_phase();
TORCH_API void set_submit_phase(VulkanSubmitPhase phase);
TORCH_API void reset_submit_phase();
TORCH_API void note_vulkan_retire_drain(
    VulkanRetireDrainReason reason,
    VulkanRetireCallSite callsite,
    bool queue_submit,
    bool blocking_wait,
    uint64_t pending_resource_count,
    uint64_t pending_bytes);
TORCH_API void note_vulkan_queue_wait_idle();
TORCH_API void note_vulkan_forced_sync(
    VulkanForcedSyncReason reason = VulkanForcedSyncReason::Unknown);

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
