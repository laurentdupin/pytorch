#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Stream.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct VulkanEventState final {
  c10::DeviceIndex device_index{-1};
  bool recorded{false};
  VulkanStreamId stream_id{0};
  VkSemaphore timeline{VK_NULL_HANDLE};
  uint64_t value{0u};
};

TORCH_API void record_vulkan_event(
    VulkanEventState& event,
    const c10::Stream& stream);
TORCH_API void block_vulkan_event(
    VulkanEventState& event,
    const c10::Stream& stream);
TORCH_API bool query_vulkan_event(const VulkanEventState& event);
TORCH_API void synchronize_vulkan_event(const VulkanEventState& event);

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
