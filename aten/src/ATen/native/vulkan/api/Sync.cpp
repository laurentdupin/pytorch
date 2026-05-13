#include <ATen/native/vulkan/api/Sync.h>

#ifdef USE_VULKAN_API

namespace at {
namespace native {
namespace vulkan {
namespace api {

VulkanSyncCounters& vulkan_sync_counters() {
  static VulkanSyncCounters counters;
  return counters;
}

void reset_vulkan_sync_counters() {
  VulkanSyncCounters& counters = vulkan_sync_counters();
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
}

void note_vulkan_queue_wait_idle() {
  vulkan_sync_counters().queue_wait_idle_count.fetch_add(
      1u, std::memory_order_relaxed);
}

void note_vulkan_forced_sync() {
  vulkan_sync_counters().forced_sync_count.fetch_add(
      1u, std::memory_order_relaxed);
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
