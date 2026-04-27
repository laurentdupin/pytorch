#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <string>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

enum class VulkanTensorRepr {
  CpuTensor,
  TexturePacked,
  BufferDirect,
  BufferWidthPacked,
  HostVisibleUploadBuffer,
  BufferMetadataView,
  ReplayOwnedOutput,
  MaterializedContiguous,
  CpuReadbackStaging,
  PackedWeight,
  Invalid,
};

enum class VulkanTensorUse {
  Read,
  Write,
  ReadWrite,
  ShapeOnly,
  RawCopySource,
  RawCopyDestination,
  KernelInput,
  KernelOutput,
  ReplayExport,
};

struct VulkanTensorStateDesc final {
  VulkanTensorRepr repr{VulkanTensorRepr::Invalid};
  api::StorageType storage_type{api::StorageType::UNKNOWN};
  api::GPUMemoryLayout memory_layout{
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED};
  api::ExecutionLayout execution_layout{api::ExecutionLayout::TEXTURE};

  ScalarType dtype{ScalarType::Undefined};
  std::vector<int64_t> logical_sizes;
  std::vector<int64_t> logical_strides;
  std::vector<int64_t> physical_sizes;
  std::vector<int64_t> physical_strides;

  int64_t storage_offset{0};
  int64_t buffer_length{0};
  bool is_view{false};
  bool is_direct_buffer{false};
  bool host_visible{false};
  bool persistent{false};
  bool replay_owned{false};
  bool last_write_was_compute{false};

  uint64_t storage_id{0};
  uint64_t view_id{0};
  uint64_t generation{0};
  uint64_t logical_desc_hash{0};
  std::string producer;
  std::string route;
};

struct VulkanTensorStateValidation final {
  bool ok{true};
  std::string reason;
  std::string message;
};

const char* vulkan_tensor_repr_name(VulkanTensorRepr);
const char* vulkan_tensor_use_name(VulkanTensorUse);
const char* vulkan_storage_type_name(api::StorageType);
const char* vulkan_memory_layout_name(api::GPUMemoryLayout);

bool vulkan_tensor_state_validation_enabled();
bool vulkan_tensor_state_logging_enabled();

uint64_t tensor_logical_desc_hash(const Tensor&);
uint64_t tensor_storage_identity(const Tensor&);

VulkanTensorStateDesc inspect_tensor_state(const Tensor&);

VulkanTensorStateValidation validate_tensor_state(
    const Tensor&,
    VulkanTensorUse use,
    const char* op_name,
    const char* route_name = nullptr);

void assert_valid_tensor_state_debug(
    const Tensor&,
    VulkanTensorUse use,
    const char* op_name,
    const char* route_name = nullptr);

std::string describe_tensor_state(const Tensor&);
std::string describe_tensor_state(const VulkanTensorStateDesc&);

void log_tensor_state(
    const Tensor&,
    VulkanTensorUse use,
    const char* op_name,
    const char* route_name = nullptr);

bool is_raw_buffer_copy_legal(const Tensor& src, const Tensor& dst);
bool requires_logical_pack_shader(const Tensor& src, const Tensor& dst);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
