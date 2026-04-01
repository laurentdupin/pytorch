#include <ATen/ops/zeros.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor& zero_(at::Tensor& self) {
  TORCH_CHECK(self.dim() <= 4, "Vulkan zero_ supports up to 4d tensors");

  vTensor& v_self = convert(self);

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
  TORCH_CHECK(size.size() <= 4, "Vulkan zeros supports up to 4d tensors");

  const ScalarType target_dtype = dtype ? *dtype : c10::kFloat;
  if (target_dtype == c10::kLong || target_dtype == c10::kBFloat16) {
    Tensor cpu_zeros = at::zeros(
        size,
        at::TensorOptions().device(at::kCPU).dtype(target_dtype));
    return convert(ops::to_vulkan(cpu_zeros, api::StorageType::BUFFER));
  }

  // Get the global Vulkan context
  api::Context* const context = api::context();

  // Create the output texture
  vTensor v_output{
      context,
      size.vec(),
      api::ScalarType::Float,
  };

  // Required to determine how to insert memory barriers in the command buffer
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(zero),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE));

  return convert(v_output);
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
