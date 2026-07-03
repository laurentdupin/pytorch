#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/TensorState.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

struct VulkanLayoutTarget final {
  api::StorageType storage_type{api::StorageType::BUFFER};
  api::GPUMemoryLayout memory_layout{
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  api::ExecutionLayout execution_layout{api::ExecutionLayout::BUFFER_DIRECT};
  bool require_direct{true};
  bool allow_metadata_view{false};
  bool host_visible{false};
  bool persistent{false};
};

enum class VulkanMaterializeReason {
  KernelRequiresBuffer,
  KernelRequiresDirectBuffer,
  KernelRequiresTexture,
  RawCopyRequiresContiguous,
  ReplayOutputEscaped,
  MetadataViewUnsupported,
  Readback,
  Upload,
  DTypeCast,
  MetadataViewCreated,
  TypedMetadataViewCreated,
  Unknown,
};

const char* vulkan_materialize_reason_name(VulkanMaterializeReason);

Tensor ensure_vulkan_layout(
    const Tensor& input,
    const VulkanLayoutTarget& target,
    VulkanMaterializeReason reason,
    const char* op_name);

Tensor make_buffer_metadata_view_checked(
    const Tensor& base,
    IntArrayRef sizes,
    IntArrayRef logical_strides,
    IntArrayRef physical_strides,
    int64_t storage_offset,
    const char* producer_op);

Tensor make_typed_buffer_metadata_view_checked(
    const Tensor& base,
    ScalarType dtype,
    IntArrayRef sizes,
    IntArrayRef logical_strides,
    IntArrayRef physical_strides,
    int64_t storage_offset,
    int64_t buffer_length_override,
    api::ExecutionLayout execution_layout,
    const char* producer_op);

Tensor materialize_vulkan_tensor(
    const Tensor& input,
    const VulkanLayoutTarget& target,
    VulkanMaterializeReason reason,
    const char* producer_op);

void log_layout_transition(
    const char* op_name,
    VulkanMaterializeReason reason,
    const Tensor& src,
    const Tensor& dst);

bool is_raw_buffer_readback_legal(const vTensor& src);

bool is_raw_buffer_readback_legal(
    const vTensor& src,
    size_t staging_size_bytes);

bool is_buffer_snapshot_readback_legal(
    const vTensor& src,
    size_t staging_size_bytes);

bool requires_logical_pack_shader_for_readback(const vTensor& src);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
