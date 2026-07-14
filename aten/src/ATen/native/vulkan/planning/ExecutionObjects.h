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

struct VulkanKVCacheSpec final {
  ScalarType dtype{kFloat};
  std::vector<int64_t> sizes;
  int64_t sequence_dim{2};
  api::ExecutionLayout execution_layout{api::ExecutionLayout::BUFFER_DIRECT};
  api::GPUMemoryLayout memory_layout{
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  api::StorageType storage_type{api::StorageType::BUFFER};
  bool persistent{true};
};

class KVCacheObject final {
 private:
  friend KVCacheObject create_vulkan_kv_cache_object(const VulkanKVCacheSpec&);

  struct State final {
    Tensor storage_;
    std::vector<int64_t> sizes_;
    int64_t sequence_dim_{2};
    int64_t sequence_length_{0};
    api::ExecutionLayout execution_layout_{api::ExecutionLayout::BUFFER_DIRECT};
    api::GPUMemoryLayout memory_layout_{
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
    api::StorageType storage_type_{api::StorageType::BUFFER};
    bool persistent_{true};
    mutable std::mutex mutex_;

    State(
        Tensor storage,
        std::vector<int64_t> sizes,
        int64_t sequence_dim,
        api::ExecutionLayout execution_layout,
        api::GPUMemoryLayout memory_layout,
        api::StorageType storage_type,
        bool persistent)
        : storage_(std::move(storage)),
          sizes_(std::move(sizes)),
          sequence_dim_(sequence_dim),
          execution_layout_(execution_layout),
          memory_layout_(memory_layout),
          storage_type_(storage_type),
          persistent_(persistent) {}
  };

  std::shared_ptr<State> state_;

 public:
  KVCacheObject() = default;
  explicit KVCacheObject(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  const Tensor& storage() const;
  const std::vector<int64_t>& sizes() const;
  int64_t sequence_dim() const;
  int64_t max_sequence_length() const;
  int64_t sequence_length() const;
  void reset();
  void set_sequence_length(int64_t sequence_length);
  Tensor read_view(int64_t start, int64_t length) const;
  Tensor append_view(int64_t length);
  api::ExecutionLayout execution_layout() const;
  api::GPUMemoryLayout memory_layout() const;
  api::StorageType storage_type() const;
  bool persistent() const;
  const void* identity() const;
};

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
    api::GPUMemoryLayout memory_layout_{
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
    bool persistent_{true};
    mutable std::mutex mutex_;

    State(
        Tensor storage,
        size_t size_bytes,
        uint32_t default_alignment,
        api::ExecutionLayout execution_layout,
        api::GPUMemoryLayout memory_layout,
        bool persistent)
        : storage_(std::move(storage)),
          size_bytes_(size_bytes),
          default_alignment_(default_alignment),
          execution_layout_(execution_layout),
          memory_layout_(memory_layout),
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
  size_t used_bytes() const;
  size_t available_bytes() const;
  uint32_t alignment() const;
  void reset();
  VulkanScratchSlice reserve(
      size_t size_bytes,
      uint32_t alignment = 0u);
  api::ExecutionLayout execution_layout() const;
  api::GPUMemoryLayout memory_layout() const;
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

KVCacheObject create_vulkan_kv_cache_object(const VulkanKVCacheSpec&);

ScratchArena create_vulkan_scratch_arena(const VulkanScratchArenaSpec&);

ReadbackBufferObject create_vulkan_readback_buffer_object(
    const VulkanReadbackBufferSpec&);

KVCacheObject lookup_or_create_labeled_kv_cache_object(
    const std::string& allocation_label,
    const VulkanKVCacheSpec&);

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
