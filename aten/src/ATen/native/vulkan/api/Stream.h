#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/vk_api.h>

#include <c10/core/Stream.h>
#include <c10/macros/Export.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace api {

using VulkanStreamId = c10::StreamId;

struct VulkanSubmission final {
  VulkanStreamId stream_id{0};
  VkSemaphore timeline{VK_NULL_HANDLE};
  uint64_t timeline_value{0u};
};

struct VulkanStreamState final {
  c10::DeviceIndex device_index{-1};
  VulkanStreamId id{0};
  Adapter::Queue queue{};
  VkDevice device{VK_NULL_HANDLE};
  VkSemaphore timeline{VK_NULL_HANDLE};
  std::atomic<uint64_t> last_submitted_value{0u};
  mutable std::atomic<uint64_t> last_known_completed_value{0u};

  struct PendingWait final {
    VkSemaphore semaphore{VK_NULL_HANDLE};
    uint64_t value{0u};
    VkPipelineStageFlags wait_stage{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
  };

  std::mutex mutex;
  std::vector<PendingWait> pending_waits;

  VulkanStreamState(
      c10::DeviceIndex device_index,
      VulkanStreamId id,
      Adapter::Queue queue,
      VkDevice device,
      VkSemaphore timeline);

  VulkanStreamState(const VulkanStreamState&) = delete;
  VulkanStreamState& operator=(const VulkanStreamState&) = delete;

  ~VulkanStreamState();

  uint64_t reserve_signal_value();
};

class TORCH_API VulkanStreamPool final {
 public:
  VulkanStreamState& get_default_stream(c10::DeviceIndex device_index);
  VulkanStreamState& get_stream(
      c10::DeviceIndex device_index,
      VulkanStreamId id);
  VulkanStreamState& get_new_stream(c10::DeviceIndex device_index);

  c10::Stream make_c10_stream(
      c10::DeviceIndex device_index,
      VulkanStreamId id) const;
  VulkanStreamState& unwrap(const c10::Stream& stream);

  c10::Stream get_current_c10_stream(c10::DeviceIndex device_index);
  VulkanStreamState& get_current_stream(c10::DeviceIndex device_index);
  void set_current_stream(const c10::Stream& stream);

  bool query_complete(const VulkanStreamState& stream, uint64_t value);
  void wait_complete(const VulkanStreamState& stream, uint64_t value);
  void wait_all(c10::DeviceIndex device_index);

 private:
  uint64_t key(c10::DeviceIndex device_index, VulkanStreamId id) const;

  std::mutex mutex_;
  std::unordered_map<uint64_t, std::unique_ptr<VulkanStreamState>> streams_;
  std::atomic<VulkanStreamId> next_stream_id_{1};
};

TORCH_API VulkanStreamPool& vulkan_stream_pool();

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
