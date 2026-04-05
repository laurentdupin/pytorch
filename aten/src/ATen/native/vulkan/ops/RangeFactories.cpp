#include <ATen/Functions.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>

#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor arange_impl(
    const std::optional<Scalar>& start,
    const Scalar& end,
    const Scalar& step,
    TensorOptions options) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const bool set_to_integral_dtype =
      !options.has_dtype() &&
      ((!start.has_value() || start->isIntegral(true)) && end.isIntegral(true) &&
       step.isIntegral(true));

  Tensor cpu_result = set_to_integral_dtype
      ? at::empty({0}, options.device(at::kCPU).dtype(at::kLong))
      : at::empty({0}, options.device(at::kCPU));
  if (start.has_value()) {
    at::arange_out(cpu_result, *start, end, step);
  } else {
    TORCH_CHECK(
        step.equal(1),
        "Vulkan arange only supports implicit step=1 for the single-end overload");
    at::arange_out(cpu_result, end);
  }
  return cpu_result.vulkan();
}

Tensor& arange_out_impl(
    const std::optional<Scalar>& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& result) {
  // Vulkan does not have a native range factory yet. Match the current
  // correctness-first approach used by other shape/factory fallbacks:
  // materialize on CPU, then copy the final tensor into Vulkan storage.
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  Tensor cpu_result = at::empty({0}, result.options().device(at::kCPU));
  if (start.has_value()) {
    at::arange_out(cpu_result, *start, end, step);
  } else {
    TORCH_CHECK(
        step.equal(1),
        "Vulkan arange.out only supports implicit step=1 for the single-end overload");
    at::arange_out(cpu_result, end);
  }

  Tensor vulkan_result = at::empty(cpu_result.sizes(), result.options());
  ops::copy_(vulkan_result, cpu_result);
  return rebind_vulkan_output(result, vulkan_result);
}

Tensor linspace_impl(
    const Scalar& start,
    const Scalar& end,
    const int64_t steps,
    TensorOptions options) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  Tensor cpu_result = at::empty({0}, options.device(at::kCPU));
  at::linspace_out(cpu_result, start, end, steps);
  return cpu_result.vulkan();
}

Tensor& linspace_out_impl(
    const Scalar& start,
    const Scalar& end,
    const int64_t steps,
    Tensor& result) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  Tensor cpu_result = at::empty({0}, result.options().device(at::kCPU));
  at::linspace_out(cpu_result, start, end, steps);

  Tensor vulkan_result = at::empty(cpu_result.sizes(), result.options());
  ops::copy_(vulkan_result, cpu_result);
  return rebind_vulkan_output(result, vulkan_result);
}

Tensor arange(
    const Scalar& end,
    std::optional<ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return arange_impl(
      std::nullopt,
      end,
      Scalar(1),
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory));
}

Tensor arange_start(
    const Scalar& start,
    const Scalar& end,
    std::optional<ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return arange_impl(
      std::optional<Scalar>(start),
      end,
      Scalar(1),
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory));
}

Tensor arange_start_step(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    std::optional<ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return arange_impl(
      std::optional<Scalar>(start),
      end,
      step,
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory));
}

Tensor& arange_out(const Scalar& end, Tensor& result) {
  return arange_out_impl(std::nullopt, end, Scalar(1), result);
}

Tensor& arange_start_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& result) {
  return arange_out_impl(start, end, step, result);
}

Tensor linspace(
    const Scalar& start,
    const Scalar& end,
    const int64_t steps,
    std::optional<ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return linspace_impl(
      start,
      end,
      steps,
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory));
}

Tensor& linspace_out(
    const Scalar& start,
    const Scalar& end,
    const int64_t steps,
    Tensor& result) {
  return linspace_out_impl(start, end, steps, result);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::arange"), TORCH_FN(arange));
  m.impl(TORCH_SELECTIVE_NAME("aten::arange.start"), TORCH_FN(arange_start));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::arange.start_step"),
      TORCH_FN(arange_start_step));
  m.impl(TORCH_SELECTIVE_NAME("aten::arange.out"), TORCH_FN(arange_out));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::arange.start_out"),
      TORCH_FN(arange_start_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::linspace"), TORCH_FN(linspace));
  m.impl(TORCH_SELECTIVE_NAME("aten::linspace.out"), TORCH_FN(linspace_out));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
