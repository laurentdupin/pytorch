#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor tril(const Tensor& self_arg, int64_t diagonal) {
  TORCH_CHECK(
      self_arg.dim() >= 2 && self_arg.dim() <= 4,
      "Vulkan tril supports tensors with 2 to 4 dimensions");

  api::Context* const context = api::context();

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  if (convert(self).storage_type() == api::StorageType::BUFFER) {
    self = utils::ensure_texture_storage(self);
  }
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
      VK_KERNEL(tril),
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

Tensor& tril_out(const Tensor& self, int64_t diagonal, Tensor& out) {
  TORCH_CHECK(out.is_vulkan(), "Vulkan tril.out expects a Vulkan output tensor");
  out.copy_(tril(self, diagonal));
  return out;
}

Tensor& tril_(Tensor& self, int64_t diagonal) {
  TORCH_CHECK(self.is_vulkan(), "Vulkan tril_ expects a Vulkan tensor");
  self.copy_(tril(self, diagonal));
  return self;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::tril"), TORCH_FN(tril));
  m.impl(TORCH_SELECTIVE_NAME("aten::tril.out"), TORCH_FN(tril_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::tril_"), TORCH_FN(tril_));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
