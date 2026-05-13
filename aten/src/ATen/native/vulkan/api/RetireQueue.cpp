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
  std::lock_guard<std::mutex> lock(mutex_);
  retired_.push_back(std::move(resource));
}

void RetireQueue::poll(VkDevice device) {
  vulkan_sync_counters().retire_poll_count.fetch_add(
      1u, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = retired_.begin();
  while (it != retired_.end()) {
    uint64_t completed = 0u;
    VK_CHECK(vkGetSemaphoreCounterValue(device, it->timeline, &completed));
    if (completed >= it->value) {
      it->cleanup();
      vulkan_sync_counters().retired_resource_count.fetch_add(
          1u, std::memory_order_relaxed);
      it = retired_.erase(it);
    } else {
      ++it;
    }
  }
}

void RetireQueue::drain(VkDevice device) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& item : retired_) {
    VkSemaphoreWaitInfo wait_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        nullptr,
        0u,
        1u,
        &item.timeline,
        &item.value,
    };
    note_vulkan_forced_sync();
    VK_CHECK(vkWaitSemaphores(
        device, &wait_info, std::numeric_limits<uint64_t>::max()));
    item.cleanup();
    vulkan_sync_counters().retired_resource_count.fetch_add(
        1u, std::memory_order_relaxed);
  }
  retired_.clear();
}

bool RetireQueue::empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return retired_.empty();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
