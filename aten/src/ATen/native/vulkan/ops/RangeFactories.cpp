#include <ATen/Functions.h>
#include <ATen/native/RangeUtils.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>

#include <c10/core/DefaultDtype.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/library.h>

#include <algorithm>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Device vulkan_device_from_options(const TensorOptions& options) {
  c10::DeviceIndex device_index = api::current_device();
  if (options.has_device()) {
    TORCH_CHECK(
        options.device().type() == at::kVulkan,
        "Vulkan factory expected a Vulkan device but got ",
        options.device());
    if (options.device().has_index()) {
      device_index = options.device().index();
    }
  }
  api::set_current_device(device_index);
  return Device(at::kVulkan, device_index);
}

bool resolves_to_float_dtype(const TensorOptions& options) {
  return options.has_dtype()
      ? c10::typeMetaToScalarType(*options.dtype_opt()) == kFloat
      : c10::get_default_dtype_as_scalartype() == kFloat;
}

Tensor range_buffer_float(
    const int64_t size,
    const float start,
    const float step,
    const char* op_name) {
  Tensor out = utils::create_buffer_tensor({size}, at::kFloat);
  vTensor& v_out = convert(out);

  api::Context* const context = api::context();
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_out);
  const struct Block final {
    float start;
    float step;
  } block{start, step};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size{
      safe_downcast<uint32_t>(std::max<int64_t>(v_out.numel(), 1)),
      1u,
      1u,
  };

  utils::log_vulkan_op_hit(std::string(op_name) + ".buffer_float");
  context->submit_compute_job(
      VK_KERNEL(range_buffer_float),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_out.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      params.buffer());

  return record_tensor_write_and_return(
      out, op_name, "buffer_float", {});
}

Tensor arange_impl(
    const std::optional<Scalar>& start,
    const Scalar& end,
    const Scalar& step,
    TensorOptions options) {
  vulkan_device_from_options(options);
  const bool inferred_integral_dtype =
      !options.has_dtype() &&
      ((!start.has_value() || start->isIntegral(true)) && end.isIntegral(true) &&
       step.isIntegral(true));
  if (!inferred_integral_dtype && resolves_to_float_dtype(options)) {
    const Scalar effective_start = start.value_or(Scalar(0));
    const int64_t size =
        at::native::compute_arange_size<float>(effective_start, end, step);
    return range_buffer_float(
        size,
        effective_start.to<float>(),
        step.to<float>(),
        "aten::arange");
  }

  report_vulkan_cpu_fallback("aten::arange", "factory_cpu_materialization");
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
  return cpu_result.to(vulkan_device_from_options(options));
}

Tensor& arange_out_impl(
    const std::optional<Scalar>& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& result) {
  if (result.scalar_type() == kFloat) {
    const Scalar effective_start = start.value_or(Scalar(0));
    const int64_t size =
        at::native::compute_arange_size<float>(effective_start, end, step);
    Tensor out = range_buffer_float(
        size,
        effective_start.to<float>(),
        step.to<float>(),
        "aten::arange.out");
    return rebind_vulkan_output(result, out);
  }

  // Vulkan does not have a native range factory yet. Match the current
  // correctness-first approach used by other shape/factory fallbacks:
  // materialize on CPU, then copy the final tensor into Vulkan storage.
  report_vulkan_cpu_fallback(
      "aten::arange.out", "factory_cpu_materialization", {result});
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
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");
  vulkan_device_from_options(options);
  if (resolves_to_float_dtype(options)) {
    const float step =
        steps > 1 ? (end.to<float>() - start.to<float>()) /
                static_cast<float>(steps - 1)
                  : 0.0f;
    return range_buffer_float(
        steps, start.to<float>(), step, "aten::linspace");
  }

  report_vulkan_cpu_fallback("aten::linspace", "factory_cpu_materialization");
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  Tensor cpu_result = at::empty({0}, options.device(at::kCPU));
  at::linspace_out(cpu_result, start, end, steps);
  return cpu_result.to(vulkan_device_from_options(options));
}

Tensor& linspace_out_impl(
    const Scalar& start,
    const Scalar& end,
    const int64_t steps,
    Tensor& result) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");
  if (result.scalar_type() == kFloat) {
    const float step =
        steps > 1 ? (end.to<float>() - start.to<float>()) /
                static_cast<float>(steps - 1)
                  : 0.0f;
    Tensor out =
        range_buffer_float(steps, start.to<float>(), step, "aten::linspace.out");
    return rebind_vulkan_output(result, out);
  }

  report_vulkan_cpu_fallback(
      "aten::linspace.out", "factory_cpu_materialization", {result});
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
