#include <ATen/native/vulkan/ops/Factory.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

constexpr int64_t kLargeFloatMatrixNumelThreshold = 1 << 20;

api::GPUMemoryLayout default_memory_layout_for_storage_type(
    const api::StorageType storage_type) {
  return storage_type == api::StorageType::BUFFER
      ? api::GPUMemoryLayout::TENSOR_WIDTH_PACKED
      : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;
}

api::StorageType choose_storage_type(
    const IntArrayRef sizes,
    const std::optional<MemoryFormat> memory_format,
    const std::optional<ScalarType> dtype) {
  api::StorageType storage_type = api::StorageType::TEXTURE_3D;

  if (dtype && (*dtype == c10::kLong || *dtype == c10::kBFloat16)) {
    return api::StorageType::BUFFER;
  }

  if (
      dtype && *dtype == c10::kFloat && sizes.size() == 2 &&
      c10::multiply_integers(sizes) >= kLargeFloatMatrixNumelThreshold) {
    return api::StorageType::BUFFER;
  }

  if (sizes.size() > 4) {
    return api::StorageType::BUFFER;
  }

  // Generic Vulkan tensors default to TEXTURE_3D storage, but raw tensors of
  // any rank up to 4 can exceed the device image limits in that layout. Large
  // embedding tables are a common case: a 2D [V, D] matrix maps to a 3D image
  // with height V under channels-packed storage. When that exceeds the
  // adapter's image limits, force BUFFER storage instead.
  if (sizes.size() <= 4) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(
        api::context()->adapter_ptr()->physical_handle(), &properties);
    const auto size_vec = sizes.vec();

    const auto memory_layout = memory_format
        ? get_gpu_memory_layout(storage_type, *memory_format)
        : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;

    if (memory_layout == api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED) {
      const uint32_t width = api::utils::safe_downcast<uint32_t>(
          api::utils::val_at(-1, size_vec));
      const uint32_t height = api::utils::safe_downcast<uint32_t>(
          api::utils::val_at(-2, size_vec));
      const uint32_t batch = api::utils::safe_downcast<uint32_t>(
          api::utils::val_at(-4, size_vec));
      const uint32_t channels = api::utils::safe_downcast<uint32_t>(
          api::utils::align_up(api::utils::val_at(-3, size_vec), INT64_C(4)) / 4);
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
  api::StorageType storage_type = choose_storage_type(sizes, memory_format, dtype);
  return convert_quantized(vTensor{
      api::context(),
      sizes.vec(),
      scale,
      zero_point,
      convert_dtype(dtype ? *dtype : c10::kFloat),
      storage_type,
      memory_format ? get_gpu_memory_layout(storage_type, *memory_format)
                    : default_memory_layout_for_storage_type(storage_type),
  });
}

static Tensor empty_memory_format(
    const IntArrayRef sizes,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory,
    const std::optional<MemoryFormat> memory_format) {
  api::StorageType storage_type = choose_storage_type(sizes, memory_format, dtype);
  return convert(vTensor{
      api::context(),
      sizes.vec(),
      convert_dtype(dtype ? *dtype : c10::kFloat),
      storage_type,
      memory_format ? get_gpu_memory_layout(storage_type, *memory_format)
                    : default_memory_layout_for_storage_type(storage_type),
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
