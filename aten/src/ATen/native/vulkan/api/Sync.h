#pragma once

#ifdef USE_VULKAN_API

#include <atomic>
#include <cstdint>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct VulkanSyncCounters final {
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
};

VulkanSyncCounters& vulkan_sync_counters();
void reset_vulkan_sync_counters();

void note_vulkan_queue_wait_idle();
void note_vulkan_forced_sync();

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
