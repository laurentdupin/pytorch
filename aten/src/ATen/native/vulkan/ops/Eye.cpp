#include <ATen/native/vulkan/ops/Copy.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/eye.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor eye_impl(
    int64_t n,
    int64_t m,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
  TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);

  const auto options =
      at::TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  Tensor cpu_eye;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    cpu_eye = at::eye(n, m, options.device(at::kCPU));
  }

  Tensor out = at::empty({n, m}, options);
  ops::copy_(out, cpu_eye);
  return out;
}

Tensor eye(
    int64_t n,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory) {
  return eye_impl(n, n, dtype, layout, device, pin_memory);
}

Tensor eye_m(
    int64_t n,
    int64_t m,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory) {
  return eye_impl(n, m, dtype, layout, device, pin_memory);
}

Tensor& eye_out(int64_t n, Tensor& out) {
  TORCH_CHECK(out.is_vulkan(), "Vulkan eye.out expects a Vulkan output tensor");
  return rebind_vulkan_output(
      out,
      eye_impl(
          n,
          n,
          std::optional<ScalarType>(out.scalar_type()),
          std::optional<c10::Layout>(out.layout()),
          std::optional<Device>(out.device()),
          std::nullopt));
}

Tensor& eye_m_out(int64_t n, int64_t m, Tensor& out) {
  TORCH_CHECK(out.is_vulkan(), "Vulkan eye.m_out expects a Vulkan output tensor");
  return rebind_vulkan_output(
      out,
      eye_impl(
          n,
          m,
          std::optional<ScalarType>(out.scalar_type()),
          std::optional<c10::Layout>(out.layout()),
          std::optional<Device>(out.device()),
          std::nullopt));
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::eye"), TORCH_FN(eye));
  m.impl(TORCH_SELECTIVE_NAME("aten::eye.m"), TORCH_FN(eye_m));
  m.impl(TORCH_SELECTIVE_NAME("aten::eye.out"), TORCH_FN(eye_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::eye.m_out"), TORCH_FN(eye_m_out));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
