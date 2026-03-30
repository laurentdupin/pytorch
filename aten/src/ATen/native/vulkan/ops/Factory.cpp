#include <ATen/native/vulkan/ops/Factory.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

api::StorageType choose_storage_type(
    const IntArrayRef sizes,
    const std::optional<MemoryFormat> memory_format) {
  api::StorageType storage_type = api::StorageType::TEXTURE_3D;

  // Generic Vulkan tensors default to TEXTURE_3D storage. For raw 4D tensors,
  // the effective image depth becomes N * ceil(C / 4). Large module weights
  // can exceed the device's 3D image limits in that layout before they are
  // ever repacked into the backend-specific shader formats.
  if (sizes.size() == 4) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(
        api::context()->adapter_ptr()->physical_handle(), &properties);

    const auto memory_layout = memory_format
        ? get_gpu_memory_layout(storage_type, *memory_format)
        : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;

    if (memory_layout == api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED) {
      const uint32_t width =
          api::utils::safe_downcast<uint32_t>(sizes[3]);
      const uint32_t height =
          api::utils::safe_downcast<uint32_t>(sizes[2]);
      const uint32_t batch = api::utils::safe_downcast<uint32_t>(sizes[0]);
      const uint32_t channels =
          api::utils::safe_downcast<uint32_t>(api::utils::align_up(sizes[1], INT64_C(4)) / 4);
      const uint32_t packed_depth = batch * channels;
      if (
          width > properties.limits.maxImageDimension3D ||
          height > properties.limits.maxImageDimension3D ||
          packed_depth > properties.limits.maxImageDimension3D) {
        storage_type = api::StorageType::BUFFER;
      }
    }
  }

  return storage_type;
}

} // namespace

Tensor _empty_affine_quantized(
    const IntArrayRef sizes,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory,
    const double scale,
    const int64_t zero_point,
    const std::optional<MemoryFormat> memory_format) {
  api::StorageType storage_type = choose_storage_type(sizes, memory_format);
  return convert_quantized(vTensor{
      api::context(),
      sizes.vec(),
      scale,
      zero_point,
      convert_dtype(dtype ? *dtype : c10::kFloat),
      storage_type,
      memory_format ? get_gpu_memory_layout(storage_type, *memory_format)
                    : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  });
}

static Tensor empty_memory_format(
    const IntArrayRef sizes,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory,
    const std::optional<MemoryFormat> memory_format) {
  api::StorageType storage_type = choose_storage_type(sizes, memory_format);
  return convert(vTensor{
      api::context(),
      sizes.vec(),
      convert_dtype(dtype ? *dtype : c10::kFloat),
      storage_type,
      memory_format ? get_gpu_memory_layout(storage_type, *memory_format)
                    : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  });
}

static Tensor empty_strided(
    const IntArrayRef sizes,
    const IntArrayRef /* strides */,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory) {
  return empty_memory_format(
      sizes, dtype, layout, device, pin_memory, c10::MemoryFormat::Contiguous);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::empty.memory_format"),
      at::native::vulkan::ops::empty_memory_format);
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_empty_affine_quantized"),
      at::native::vulkan::ops::_empty_affine_quantized);
  m.impl(
      TORCH_SELECTIVE_NAME("aten::empty_strided"),
      TORCH_FN(at::native::vulkan::ops::empty_strided));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
