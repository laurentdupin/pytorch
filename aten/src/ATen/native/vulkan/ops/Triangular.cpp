#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/tril.h>
#include <ATen/ops/triu.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

bool supports_native_triangular_texture_path(const Tensor& self_arg) {
  return self_arg.dim() >= 2 && self_arg.dim() <= 4 &&
      (self_arg.scalar_type() == at::kFloat ||
       self_arg.scalar_type() == at::kHalf);
}

Tensor triangular_cpu_fallback(
    const Tensor& self_arg,
    int64_t diagonal,
    bool upper) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);
  Tensor cpu = self_arg.cpu();
  Tensor cpu_out = upper ? at::triu(cpu, diagonal) : at::tril(cpu, diagonal);
  Tensor out = at::empty(cpu_out.sizes(), self_arg.options().device(at::kVulkan));
  ops::copy_(out, cpu_out);
  return out;
}

Tensor triangular_texture(
    const Tensor& self_arg,
    int64_t diagonal,
    bool upper) {
  TORCH_CHECK(
      self_arg.dim() >= 2 && self_arg.dim() <= 4,
      "Vulkan triangular texture kernels support tensors with 2 to 4 dimensions");

  api::Context* const context = api::context();

  Tensor self = utils::prepare_vulkan_execution_tensor(
      self_arg, utils::VulkanExecutionPlanKind::TextureComputeInput);
  const vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const struct Block final {
    ivec4 extents;
    int32_t diagonal;
  } block{
      {safe_downcast<int32_t>(v_output.extents().data[0]),
       safe_downcast<int32_t>(v_output.extents().data[1]),
       safe_downcast<int32_t>(v_output.extents().data[2]),
       0},
      safe_downcast<int32_t>(diagonal),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      upper ? VK_KERNEL(triu) : VK_KERNEL(tril),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor vulkan_tril(const Tensor& self_arg, int64_t diagonal) {
  if (!supports_native_triangular_texture_path(self_arg)) {
    return triangular_cpu_fallback(self_arg, diagonal, false);
  }
  return triangular_texture(self_arg, diagonal, false);
}

Tensor& tril_out(const Tensor& self, int64_t diagonal, Tensor& out) {
  TORCH_CHECK(out.is_vulkan(), "Vulkan tril.out expects a Vulkan output tensor");
  return rebind_vulkan_output(out, vulkan_tril(self, diagonal));
}

Tensor& tril_(Tensor& self, int64_t diagonal) {
  TORCH_CHECK(self.is_vulkan(), "Vulkan tril_ expects a Vulkan tensor");
  self.copy_(vulkan_tril(self, diagonal));
  return self;
}

Tensor vulkan_triu(const Tensor& self_arg, int64_t diagonal) {
  if (!supports_native_triangular_texture_path(self_arg)) {
    return triangular_cpu_fallback(self_arg, diagonal, true);
  }
  return triangular_texture(self_arg, diagonal, true);
}

Tensor& triu_out(const Tensor& self, int64_t diagonal, Tensor& out) {
  TORCH_CHECK(out.is_vulkan(), "Vulkan triu.out expects a Vulkan output tensor");
  return rebind_vulkan_output(out, vulkan_triu(self, diagonal));
}

Tensor& triu_(Tensor& self, int64_t diagonal) {
  TORCH_CHECK(self.is_vulkan(), "Vulkan triu_ expects a Vulkan tensor");
  self.copy_(vulkan_triu(self, diagonal));
  return self;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::tril"), TORCH_FN(vulkan_tril));
  m.impl(TORCH_SELECTIVE_NAME("aten::tril.out"), TORCH_FN(tril_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::tril_"), TORCH_FN(tril_));
  m.impl(TORCH_SELECTIVE_NAME("aten::triu"), TORCH_FN(vulkan_triu));
  m.impl(TORCH_SELECTIVE_NAME("aten::triu.out"), TORCH_FN(triu_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::triu_"), TORCH_FN(triu_));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
