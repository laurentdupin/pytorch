#include <ATen/native/vulkan/api/RetireQueue.h>

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Exception.h>
#include <ATen/native/vulkan/api/Sync.h>

#include <limits>

namespace at {
namespace native {
namespace vulkan {
namespace api {

void RetireQueue::retire(RetiredResource resource) {
  if (!resource.cleanup) {
    return;
  }
  VK_CHECK_COND(
      resource.timeline != VK_NULL_HANDLE,
      "RetiredResource has null timeline");
  VK_CHECK_COND(
      resource.value > 0u,
      "RetiredResource has invalid timeline value");
  std::lock_guard<std::mutex> lock(mutex_);
  retired_.push_back(std::move(resource));
}

void RetireQueue::poll(VkDevice device) {
  vulkan_sync_counters().retire_poll_count.fetch_add(
      1u, std::memory_order_relaxed);
  std::vector<std::function<void()>> ready;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = retired_.begin();
    while (it != retired_.end()) {
      VK_CHECK_COND(
          it->timeline != VK_NULL_HANDLE,
          "RetiredResource has null timeline");
      VK_CHECK_COND(
          it->value > 0u,
          "RetiredResource has invalid timeline value");
      uint64_t completed = 0u;
      VK_CHECK(vkGetSemaphoreCounterValue(device, it->timeline, &completed));
      if (completed >= it->value) {
        ready.emplace_back(std::move(it->cleanup));
        it = retired_.erase(it);
      } else {
        ++it;
      }
    }
  }
  for (auto& cleanup : ready) {
    cleanup();
    vulkan_sync_counters().retired_resource_count.fetch_add(
        1u, std::memory_order_relaxed);
  }
}

void RetireQueue::drain(VkDevice device) {
  std::vector<RetiredResource> items;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    items.swap(retired_);
  }
  for (auto& item : items) {
    VK_CHECK_COND(
        item.timeline != VK_NULL_HANDLE,
        "RetiredResource has null timeline");
    VK_CHECK_COND(
        item.value > 0u,
        "RetiredResource has invalid timeline value");
    VkSemaphoreWaitInfo wait_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        nullptr,
        0u,
        1u,
        &item.timeline,
        &item.value,
    };
    uint64_t completed = 0u;
    VK_CHECK(vkGetSemaphoreCounterValue(device, item.timeline, &completed));
    if (completed < item.value) {
      note_vulkan_forced_sync(VulkanForcedSyncReason::RetireQueueDrain);
      VK_CHECK(vkWaitSemaphores(
          device, &wait_info, std::numeric_limits<uint64_t>::max()));
    }
    item.cleanup();
    vulkan_sync_counters().retired_resource_count.fetch_add(
        1u, std::memory_order_relaxed);
  }
}

bool RetireQueue::empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return retired_.empty();
}

size_t RetireQueue::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return retired_.size();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
