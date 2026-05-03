#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/core/DefaultDtype.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor& zero_cpu_fallback(Tensor& self) {
  report_vulkan_cpu_fallback(
      "aten::zero_", "unsupported_shape_storage_or_dtype", {self});
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);
  Tensor cpu_zeros = at::zeros(self.sizes(), self.options().device(at::kCPU));
  ops::copy_(self, cpu_zeros);
  return self;
}

Tensor& zero_(at::Tensor& self) {
  vTensor& v_self = convert(self);
  if (self.dim() > 4) {
    return zero_cpu_fallback(self);
  }
  if (v_self.storage_type() == api::StorageType::BUFFER) {
    if (self.scalar_type() == at::kFloat) {
      return utils::fill_buffer_float_(self, 0.0f, "aten::zero_");
    }
    return zero_cpu_fallback(self);
  }
  if (!api::supports_texture_storage(v_self.dtype())) {
    return zero_cpu_fallback(self);
  }

  // Get the global Vulkan context
  api::Context* const context = api::context();

  // Required to determine how to insert memory barriers in the command buffer
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(zero),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE));

  return self;
}

Tensor zeros(
    const IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  const ScalarType target_dtype =
      dtype.value_or(c10::get_default_dtype_as_scalartype());
  const Device resolved_device =
      device.value_or(Device(at::kVulkan, api::current_device()));
  Tensor out = at::empty(
      size,
      at::TensorOptions().device(resolved_device).dtype(target_dtype));
  zero_(out);
  return out;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::zero_"), TORCH_FN(zero_));
  m.impl(TORCH_SELECTIVE_NAME("aten::zeros"), TORCH_FN(zeros));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
