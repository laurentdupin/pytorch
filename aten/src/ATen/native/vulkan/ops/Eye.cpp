#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/core/DefaultDtype.h>
#include <torch/library.h>

#include <algorithm>

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

using namespace api::utils;

Tensor eye_buffer_float(const int64_t n, const int64_t m) {
  Tensor out = utils::create_buffer_tensor({n, m}, at::kFloat);
  vTensor& v_out = convert(out);

  api::Context* const context = api::context();
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_out);
  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size{
      safe_downcast<uint32_t>(std::max<int64_t>(v_out.numel(), 1)),
      1u,
      1u,
  };

  utils::log_vulkan_op_hit("aten::eye.buffer_float");
  context->submit_compute_job(
      VK_KERNEL(eye_buffer_float),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_out.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer());

  return record_tensor_write_and_return(
      out, "aten::eye", "buffer_float", {});
}

Device resolve_vulkan_factory_device(const std::optional<Device>& device) {
  if (device.has_value()) {
    TORCH_CHECK(
        device->type() == at::kVulkan,
        "Vulkan factory expected a Vulkan device but got ",
        *device);
  }
  const c10::DeviceIndex device_index =
      device.has_value() && device->has_index() ? device->index()
                                                : api::current_device();
  api::set_current_device(device_index);
  return Device(at::kVulkan, device_index);
}

Tensor eye_impl(
    int64_t n,
    int64_t m,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
  TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);

  const Device resolved_device = resolve_vulkan_factory_device(device);
  const auto options =
      at::TensorOptions()
          .dtype(dtype)
          .layout(layout)
          .device(resolved_device)
          .pinned_memory(pin_memory);
  const ScalarType target_dtype =
      dtype.value_or(c10::get_default_dtype_as_scalartype());

  if (target_dtype == kFloat) {
    return eye_buffer_float(n, m);
  }

  Tensor cpu_eye;
  {
    report_vulkan_cpu_fallback("aten::eye", "factory_cpu_materialization");
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
