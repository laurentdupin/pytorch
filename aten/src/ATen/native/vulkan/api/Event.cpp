#include <ATen/native/vulkan/api/Event.h>

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/SyncCounters.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

void record_vulkan_event(VulkanEventState& event, const c10::Stream& stream) {
  if (Context* const stream_context = context(stream.device_index())) {
    stream_context->flush_if_current_stream(stream);
  }
  VulkanStreamState& vk_stream = vulkan_stream_pool().unwrap(stream);
  event.device_index = stream.device_index();
  event.recorded = true;
  event.stream_id = vk_stream.id;
  event.timeline = vk_stream.timeline;
  event.value =
      vk_stream.last_submitted_value.load(std::memory_order_acquire);
  vulkan_sync_counters().event_record_count.fetch_add(
      1u, std::memory_order_relaxed);
}

void block_vulkan_event(VulkanEventState& event, const c10::Stream& stream) {
  if (!event.recorded || event.timeline == VK_NULL_HANDLE || event.value == 0u) {
    return;
  }
  if (Context* const stream_context = context(stream.device_index())) {
    stream_context->flush_if_current_stream(stream);
  }
  VulkanStreamState& vk_stream = vulkan_stream_pool().unwrap(stream);
  {
    std::lock_guard<std::mutex> lock(vk_stream.mutex);
    vk_stream.pending_waits.push_back(VulkanStreamState::PendingWait{
        event.timeline,
        event.value,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    });
  }
  vulkan_sync_counters().event_block_count.fetch_add(
      1u, std::memory_order_relaxed);
}

bool query_vulkan_event(const VulkanEventState& event) {
  if (!event.recorded || event.timeline == VK_NULL_HANDLE || event.value == 0u) {
    return true;
  }
  VulkanStreamState& stream =
      vulkan_stream_pool().get_stream(event.device_index, event.stream_id);
  return vulkan_stream_pool().query_complete(stream, event.value);
}

void synchronize_vulkan_event(const VulkanEventState& event) {
  if (!event.recorded || event.timeline == VK_NULL_HANDLE || event.value == 0u) {
    return;
  }
  VulkanStreamState& stream =
      vulkan_stream_pool().get_stream(event.device_index, event.stream_id);
  vulkan_sync_counters().event_wait_count.fetch_add(
      1u, std::memory_order_relaxed);
  vulkan_stream_pool().wait_complete(
      stream, event.value, VulkanForcedSyncReason::EventSynchronize);
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
