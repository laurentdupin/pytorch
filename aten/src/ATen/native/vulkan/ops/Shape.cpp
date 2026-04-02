#include <ATen/InferSize.h>
#include <ATen/Functions.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <optional>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

Tensor view_internal(
    const Tensor& self_arg,
    const IntArrayRef output_size,
    const IntArrayRef output_stride,
    const std::optional<int64_t> storage_offset = std::nullopt) {
  // Vulkan views are not true metadata aliases yet. Use the proven CPU
  // reshape/as_strided path and rematerialize a fresh Vulkan tensor.
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);
  Tensor cpu = self_arg.cpu();
  Tensor cpu_view = storage_offset.has_value()
      ? cpu.as_strided(output_size.vec(), output_stride.vec(), *storage_offset)
      : cpu.as_strided(output_size.vec(), output_stride.vec());
  Tensor out = at::empty(
      output_size.vec(),
      self_arg.options().device(at::kVulkan));
  ops::copy_(out, cpu_view);
  return out;
}

} // namespace

inline Tensor view(const Tensor& self_arg, IntArrayRef shape) {
  at::DimVector inferred_size = at::infer_size_dv(shape, self_arg.numel());
  IntArrayRef base_sizes = self_arg.sizes();
  IntArrayRef base_strides = self_arg.strides();
  c10::DimVector base_logical_strides;
  if (self_arg.is_vulkan()) {
    const vTensor& v_self = convert(self_arg);
    base_logical_strides = logical_strides(v_self);
    base_sizes = v_self.sizes();
    base_strides = base_logical_strides;
  }
  auto inferred_stride = at::detail::computeStride(
      base_sizes,
      base_strides,
      inferred_size);
  TORCH_CHECK(
      inferred_stride.has_value(),
      "view size is not compatible with input tensor's size and stride");
  return view_internal(self_arg, inferred_size, *inferred_stride);
}

static Tensor _reshape_alias(
    const Tensor& self_arg,
    const IntArrayRef shape,
    const IntArrayRef strides) {
  return view_internal(self_arg, shape, strides);
}

static Tensor as_strided(
    const Tensor& self_arg,
    const IntArrayRef shape,
    const IntArrayRef strides,
    const std::optional<int64_t> storage_offset) {
  return view_internal(self_arg, shape, strides, storage_offset);
}

static Tensor im2col(
    const Tensor& self_arg,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  Tensor cpu = self_arg.cpu();
  Tensor cpu_result =
      at::im2col(cpu, kernel_size.vec(), dilation.vec(), padding.vec(), stride.vec());
  Tensor out = at::empty(
      cpu_result.sizes(),
      self_arg.options().device(at::kVulkan));
  ops::copy_(out, cpu_result);
  return out;
}

static Tensor& im2col_out(
    const Tensor& self_arg,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    Tensor& out) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  Tensor cpu = self_arg.cpu();
  Tensor cpu_result =
      at::im2col(cpu, kernel_size.vec(), dilation.vec(), padding.vec(), stride.vec());
  TORCH_CHECK(
      out.sizes() == cpu_result.sizes(),
      "Vulkan im2col.out requires a pre-sized output tensor; resizing Vulkan outputs is not supported");
  ops::copy_(out, cpu_result);
  return out;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::as_strided"), TORCH_FN(as_strided));
  m.impl(TORCH_SELECTIVE_NAME("aten::im2col"), TORCH_FN(im2col));
  m.impl(TORCH_SELECTIVE_NAME("aten::im2col.out"), TORCH_FN(im2col_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::view"), TORCH_FN(view));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_reshape_alias"), TORCH_FN(_reshape_alias));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
