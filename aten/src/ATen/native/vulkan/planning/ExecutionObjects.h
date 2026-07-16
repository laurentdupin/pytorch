#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/planning/Request.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

std::string make_vulkan_runtime_object_label(
    const VulkanPlanningRequest& request,
    const char* label_suffix);

struct VulkanScratchArenaSpec final {
  ScalarType dtype{kByte};
  size_t num_bytes{0u};
  uint32_t alignment{256u};
  api::ExecutionLayout execution_layout{api::ExecutionLayout::BUFFER_DIRECT};
  api::GPUMemoryLayout memory_layout{
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  api::StorageType storage_type{api::StorageType::BUFFER};
  bool persistent{true};
};

struct VulkanScratchSlice final {
  size_t offset_bytes{0u};
  size_t size_bytes{0u};
};

class ScratchArena final {
 private:
  friend ScratchArena create_vulkan_scratch_arena(
      const VulkanScratchArenaSpec&);

  struct State final {
    Tensor storage_;
    size_t size_bytes_{0u};
    uint32_t default_alignment_{256u};
    size_t next_offset_bytes_{0u};
    api::ExecutionLayout execution_layout_{api::ExecutionLayout::BUFFER_DIRECT};
    bool persistent_{true};
    mutable std::mutex mutex_;

    State(
        Tensor storage,
        size_t size_bytes,
        uint32_t default_alignment,
        api::ExecutionLayout execution_layout,
        bool persistent)
        : storage_(std::move(storage)),
          size_bytes_(size_bytes),
          default_alignment_(default_alignment),
          execution_layout_(execution_layout),
          persistent_(persistent) {}
  };

  std::shared_ptr<State> state_;

 public:
  ScratchArena() = default;
  explicit ScratchArena(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  const Tensor& storage() const;
  size_t size_bytes() const;
  uint32_t alignment() const;
  void reset();
  VulkanScratchSlice reserve(
      size_t size_bytes,
      uint32_t alignment = 0u);
  api::ExecutionLayout execution_layout() const;
  bool persistent() const;
  const void* identity() const;
};

struct VulkanReadbackBufferSpec final {
  size_t num_bytes{0u};
  bool persistent{true};
};

class ReadbackBufferObject final {
 private:
  friend ReadbackBufferObject create_vulkan_readback_buffer_object(
      const VulkanReadbackBufferSpec&);

  struct State final {
    api::VulkanBuffer buffer_;
    size_t size_bytes_{0u};
    bool persistent_{true};
    mutable std::mutex mutex_;

    State(api::VulkanBuffer buffer, size_t size_bytes, bool persistent)
        : buffer_(std::move(buffer)),
          size_bytes_(size_bytes),
          persistent_(persistent) {}
  };

  std::shared_ptr<State> state_;

 public:
  ReadbackBufferObject() = default;
  explicit ReadbackBufferObject(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  api::VulkanBuffer& buffer() const;
  size_t size_bytes() const;
  bool persistent() const;
  std::mutex& mutex() const;
  const void* identity() const;
};

ScratchArena create_vulkan_scratch_arena(const VulkanScratchArenaSpec&);

ReadbackBufferObject create_vulkan_readback_buffer_object(
    const VulkanReadbackBufferSpec&);

ScratchArena lookup_or_create_labeled_scratch_arena(
    const std::string& allocation_label,
    const VulkanScratchArenaSpec&);

ReadbackBufferObject lookup_or_create_labeled_readback_buffer_object(
    const std::string& allocation_label,
    const VulkanReadbackBufferSpec&);

std::optional<ScratchArena> prime_labeled_scratch_arena_for_request(
    const Tensor& reference,
    const VulkanPlanningRequest& request,
    size_t requested_bytes,
    const char* label_suffix = "scratch");

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
