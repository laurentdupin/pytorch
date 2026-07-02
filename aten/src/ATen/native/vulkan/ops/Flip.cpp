#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor flip_buffer_float(
    const Tensor& input,
    const vTensor& v_input,
    const std::vector<int32_t>& dim_args) {
  api::Context* const context = api::context();
  vTensor v_output{
      context,
      v_input.sizes(),
      v_input.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct Block final {
    uvec4 dims;
  } block{{static_cast<uint32_t>(dim_args[3]),
           static_cast<uint32_t>(dim_args[2]),
           static_cast<uint32_t>(dim_args[1]),
           static_cast<uint32_t>(dim_args[0])}};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size{
      safe_downcast<uint32_t>(v_output.numel()),
      1u,
      1u,
  };
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);

  context->submit_compute_job(
      VK_KERNEL(flip_buffer_float),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer(),
      params.buffer());

  return record_tensor_write_and_return(
      convert(v_output), "aten::flip", "buffer_float", {input});
}

Tensor flip(const at::Tensor& self, const IntArrayRef dim_list) {
  TORCH_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan flip supports up to 4d tensors as input!");

  // Get the global Vulkan context
  api::Context* const context = api::context();

  // Cast the input Tensor to a vTensor
  const Tensor input = self.is_vulkan() ? self : self.vulkan();
  const vTensor& v_input = convert(input);

  // Create the output texture
  vTensor v_output{
      context,
      v_input.sizes(),
      convert_dtype(self.scalar_type()),
  };

  // Required to determine how to insert memory barriers in the command buffer
  api::PipelineBarrier pipeline_barrier{};

  // Create dim args
  std::vector<int32_t> dim_args = {0, 0, 0, 0};
  for (const auto dim : dim_list) {
    TORCH_CHECK(
        dim >= -self.dim() - 1 && dim <= self.dim(),
        "Vulkan flip dimension out of range expected to be in range of [",
        -self.dim() - 1,
        ",",
        self.dim(),
        "], but got ",
        dim);
    // Normalize
    int normalized_dim = utils::normalize(dim, self.dim());

    // Shift into 4d range
    if (self.dim() < 4) {
      normalized_dim += (4 - self.dim());
    }
    dim_args[normalized_dim] = 1;
  }

  if (
      v_input.dtype() == api::kFloat &&
      v_input.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_metadata_view_fast_path(v_input)) {
    return flip_buffer_float(input, v_input, dim_args);
  }

  // Create the params buffer
  const struct Block final {
    uvec4 extents;
    ivec4 dims;
  } block{
      {get_dim<Dim4D::Width>(v_output),
       get_dim<Dim4D::Height>(v_output),
       get_dim<Dim4D::Channel>(v_output),
       get_dim<Dim4D::Batch>(v_output)},
      {dim_args[3], dim_args[2], dim_args[1], dim_args[0]},
  };

  api::UniformParamsBuffer params(context, block);

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(flip),
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
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());
  return convert(v_output);
};

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::flip"), TORCH_FN(flip));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
