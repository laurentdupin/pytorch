#pragma once

#ifdef USE_VULKAN_API

#include <c10/macros/Export.h>

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

TORCH_API VulkanSyncCounters& vulkan_sync_counters();
TORCH_API void reset_vulkan_sync_counters();
TORCH_API VulkanSubmitOriginCounters& vulkan_submit_origin_counters();
TORCH_API void reset_vulkan_submit_origin_counters();
TORCH_API void note_vulkan_queue_submit(VulkanSubmitOrigin origin);

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
