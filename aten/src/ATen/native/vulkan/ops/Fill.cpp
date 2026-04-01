#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor& fill_scalar_(Tensor& self_arg, const Scalar& value) {
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: fill_.Scalar is only supported on Vulkan tensors.");
  TORCH_CHECK(self_arg.dim() <= 4, "Vulkan fill_.Scalar supports up to 4d tensors");

  vTensor& v_self = convert(self_arg);
  TORCH_CHECK(
      v_self.storage_type() != api::StorageType::BUFFER,
      "In-place Vulkan fill_.Scalar is not yet supported on buffer-backed logical views");

  api::Context* const context = api::context();

  const struct Block final {
    ivec4 extents;
    float value;
  } block{
      {safe_downcast<int32_t>(v_self.extents().data[0]),
       safe_downcast<int32_t>(v_self.extents().data[1]),
       safe_downcast<int32_t>(v_self.extents().data[2]),
       0},
      value.to<float>(),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(fill),
      pipeline_barrier,
      v_self.extents(),
      adaptive_work_group_size(v_self.extents()),
      VK_NULL_HANDLE,
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      params.buffer());

  return self_arg;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::fill_.Scalar"), TORCH_FN(fill_scalar_));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
