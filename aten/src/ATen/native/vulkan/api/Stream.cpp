#include <ATen/native/vulkan/api/Stream.h>

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Runtime.h>
#include <ATen/native/vulkan/api/Sync.h>
#include <ATen/native/vulkan/api/Exception.h>

#include <limits>

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace {

VkSemaphore create_timeline_semaphore(VkDevice device) {
  VkSemaphoreTypeCreateInfo type_info{
      VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      nullptr,
      VK_SEMAPHORE_TYPE_TIMELINE,
      0u,
  };
  VkSemaphoreCreateInfo create_info{
      VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      &type_info,
      0u,
  };
  VkSemaphore semaphore = VK_NULL_HANDLE;
  VK_CHECK(vkCreateSemaphore(device, &create_info, nullptr, &semaphore));
  return semaphore;
}

std::unordered_map<c10::DeviceIndex, VulkanStreamId>& current_streams() {
  static thread_local std::unordered_map<c10::DeviceIndex, VulkanStreamId>
      streams;
  return streams;
}

} // namespace

VulkanStreamState::VulkanStreamState(
    c10::DeviceIndex device_index,
    VulkanStreamId id,
    Adapter::Queue queue,
    VkDevice device,
    VkSemaphore timeline)
    : device_index(device_index),
      id(id),
      queue(queue),
      device(device),
      timeline(timeline),
      last_submitted_value(0u),
      last_known_completed_value(0u),
      mutex(),
      pending_waits() {}

VulkanStreamState::~VulkanStreamState() {
  if (timeline != VK_NULL_HANDLE) {
    vkDestroySemaphore(device, timeline, nullptr);
  }
}

uint64_t VulkanStreamState::reserve_signal_value() {
  return last_submitted_value.fetch_add(1u, std::memory_order_relaxed) + 1u;
}

VulkanStreamState& VulkanStreamPool::get_default_stream(
    c10::DeviceIndex device_index) {
  return get_stream(device_index, 0);
}

VulkanStreamState& VulkanStreamPool::get_stream(
    c10::DeviceIndex device_index,
    VulkanStreamId id) {
  std::lock_guard<std::mutex> lock(mutex_);
  const uint64_t stream_key = key(device_index, id);
  auto it = streams_.find(stream_key);
  if (it != streams_.end()) {
    return *it->second;
  }

  Adapter* adapter = runtime()->get_adapter_p_for_device(device_index);
  Adapter::Queue queue = adapter->request_queue();
  VkDevice device = adapter->device_handle();
  auto stream = std::make_unique<VulkanStreamState>(
      device_index,
      id,
      queue,
      device,
      create_timeline_semaphore(device));
  VulkanStreamState& stream_ref = *stream;
  streams_.emplace(stream_key, std::move(stream));
  return stream_ref;
}

VulkanStreamState& VulkanStreamPool::get_new_stream(
    c10::DeviceIndex device_index) {
  const VulkanStreamId id =
      next_stream_id_.fetch_add(1, std::memory_order_relaxed);
  return get_stream(device_index, id);
}

c10::Stream VulkanStreamPool::make_c10_stream(
    c10::DeviceIndex device_index,
    VulkanStreamId id) const {
  return c10::Stream(
      c10::Stream::UNSAFE, c10::Device(c10::DeviceType::Vulkan, device_index), id);
}

VulkanStreamState& VulkanStreamPool::unwrap(const c10::Stream& stream) {
  VK_CHECK_COND(
      stream.device_type() == c10::DeviceType::Vulkan,
      "Expected a Vulkan stream, got ",
      stream.device());
  return get_stream(stream.device_index(), stream.id());
}

c10::Stream VulkanStreamPool::get_current_c10_stream(
    c10::DeviceIndex device_index) {
  auto& streams = current_streams();
  auto it = streams.find(device_index);
  if (it == streams.end()) {
    streams.emplace(device_index, 0);
    return make_c10_stream(device_index, 0);
  }
  return make_c10_stream(device_index, it->second);
}

VulkanStreamState& VulkanStreamPool::get_current_stream(
    c10::DeviceIndex device_index) {
  return unwrap(get_current_c10_stream(device_index));
}

void VulkanStreamPool::set_current_stream(const c10::Stream& stream) {
  VK_CHECK_COND(
      stream.device_type() == c10::DeviceType::Vulkan,
      "Expected a Vulkan stream, got ",
      stream.device());
  unwrap(stream);
  current_streams()[stream.device_index()] = stream.id();
}

bool VulkanStreamPool::query_complete(
    const VulkanStreamState& stream,
    uint64_t value) {
  if (value == 0u) {
    return true;
  }
  uint64_t completed = 0u;
  VK_CHECK(vkGetSemaphoreCounterValue(stream.device, stream.timeline, &completed));
  stream.last_known_completed_value.store(completed, std::memory_order_relaxed);
  return completed >= value;
}

void VulkanStreamPool::wait_complete(
    const VulkanStreamState& stream,
    uint64_t value) {
  if (value == 0u) {
    return;
  }
  VkSemaphoreWaitInfo wait_info{
      VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      nullptr,
      0u,
      1u,
      &stream.timeline,
      &value,
  };
  note_vulkan_forced_sync();
  VK_CHECK(vkWaitSemaphores(
      stream.device, &wait_info, std::numeric_limits<uint64_t>::max()));
  stream.last_known_completed_value.store(value, std::memory_order_relaxed);
}

void VulkanStreamPool::wait_all(c10::DeviceIndex device_index) {
  std::vector<std::pair<VulkanStreamState*, uint64_t>> streams_to_wait;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& item : streams_) {
      VulkanStreamState* stream = item.second.get();
      if (stream->device_index != device_index) {
        continue;
      }
      streams_to_wait.emplace_back(
          stream,
          stream->last_submitted_value.load(std::memory_order_acquire));
    }
  }
  for (const auto& item : streams_to_wait) {
    wait_complete(*item.first, item.second);
  }
}

uint64_t VulkanStreamPool::key(
    c10::DeviceIndex device_index,
    VulkanStreamId id) const {
  return (static_cast<uint64_t>(static_cast<uint16_t>(device_index)) << 48) |
      (static_cast<uint64_t>(id) & ((1ull << 48) - 1ull));
}

VulkanStreamPool& vulkan_stream_pool() {
  static VulkanStreamPool* pool = new VulkanStreamPool();
  return *pool;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
