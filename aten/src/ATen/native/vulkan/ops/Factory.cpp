#include <ATen/native/vulkan/ops/Factory.h>
#include <c10/core/DefaultDtype.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

api::Context* resolve_vulkan_context(const std::optional<Device> device) {
  if (device.has_value()) {
    TORCH_CHECK(
        device->type() == kVulkan,
        "Vulkan factory expected a Vulkan device but got ",
        *device);
  }

  const c10::DeviceIndex device_index =
      device.has_value() && device->has_index()
      ? device->index()
      : api::current_device();
  api::set_current_device(device_index);
  return api::context(device_index);
}

api::GPUMemoryLayout default_memory_layout_for_storage_type(
    const api::StorageType storage_type) {
  return storage_type == api::StorageType::BUFFER
      ? api::GPUMemoryLayout::TENSOR_WIDTH_PACKED
      : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;
}

bool should_force_low_rank_float_buffer_storage(
    const IntArrayRef sizes,
    const std::optional<MemoryFormat> memory_format,
    const std::optional<ScalarType> dtype) {
  if (
      !dtype || !c10::isFloatingType(*dtype) || sizes.size() < 1 ||
      c10::multiply_integers(sizes) <= 0) {
    return false;
  }
  if (sizes.size() <= 3) {
    return true;
  }
  // Tensor.to("vulkan") commonly allocates through empty_strided with an
  // explicit contiguous memory format. Route 4D contiguous activations to the
  // generic buffer path while leaving bare torch.empty(..., device="vulkan")
  // texture-backed so tests and callers can still request the legacy texture
  // path without a separate API.
  return sizes.size() == 4 && memory_format &&
      *memory_format == c10::MemoryFormat::Contiguous;
}

api::StorageType choose_storage_type(
    const IntArrayRef sizes,
    const std::optional<MemoryFormat> memory_format,
    const std::optional<ScalarType> dtype,
    api::Context* const context) {
  api::StorageType storage_type = api::StorageType::TEXTURE_3D;

  if (dtype && api::requires_buffer_storage(convert_dtype(*dtype), sizes.size())) {
    return api::StorageType::BUFFER;
  }

  if (should_force_low_rank_float_buffer_storage(sizes, memory_format, dtype)) {
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
        context->adapter_ptr()->physical_handle(), &properties);
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
  api::Context* const context = resolve_vulkan_context(device);
  api::StorageType storage_type =
      choose_storage_type(sizes, memory_format, dtype, context);
  return convert_quantized(vTensor{
      context,
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
  api::Context* const context = resolve_vulkan_context(device);
  api::StorageType storage_type =
      choose_storage_type(sizes, memory_format, dtype, context);
  return convert(vTensor{
      context,
      sizes.vec(),
      convert_dtype(dtype.value_or(c10::get_default_dtype_as_scalartype())),
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
