#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Stream.h>

#include <c10/macros/Export.h>

#include <cstddef>
#include <functional>
#include <mutex>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct RetiredResource final {
  VulkanStreamId stream_id{0};
  VkSemaphore timeline{VK_NULL_HANDLE};
  uint64_t value{0u};
  std::function<void()> cleanup;
};

class TORCH_API RetireQueue final {
 public:
  void retire(RetiredResource resource);
  void poll(VkDevice device);
  void drain(VkDevice device);
  bool empty() const;
  size_t size() const;

 private:
  mutable std::mutex mutex_;
  std::vector<RetiredResource> retired_;
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
