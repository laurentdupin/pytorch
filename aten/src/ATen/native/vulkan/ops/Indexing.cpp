#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor gather_rows_2d(const Tensor& weight_arg, const Tensor& indices_arg) {
  TORCH_CHECK(weight_arg.is_vulkan(), "Vulkan gather expects weight on Vulkan");
  TORCH_CHECK(
      weight_arg.dim() == 2,
      "Vulkan gather_rows_2d expects a 2D weight tensor");
  TORCH_CHECK(
      indices_arg.device().type() == kCPU,
      "Vulkan gather_rows_2d expects CPU indices");
  TORCH_CHECK(
      indices_arg.dim() == 1 || indices_arg.dim() == 2,
      "Vulkan gather_rows_2d expects 1D or 2D indices");
  TORCH_CHECK(
      indices_arg.scalar_type() == kLong || indices_arg.scalar_type() == kInt,
      "Vulkan gather_rows_2d expects int32 or int64 indices");

  Tensor weight = weight_arg;
  vTensor v_weight = convert(weight);

  const int64_t row_count = weight_arg.size(0);
  const int64_t row_width = weight_arg.size(1);
  const Tensor indices = indices_arg.contiguous();
  const int64_t num_indices = indices.numel();

  api::Context* const context = api::context();
  api::StorageBuffer index_buffer(context, api::kInt, num_indices);
  {
    api::MemoryMap mapping(index_buffer.buffer(), api::MemoryAccessType::WRITE);
    int32_t* const dst = mapping.template data<int32_t>();

    if (indices.scalar_type() == kLong) {
      const int64_t* const src = indices.const_data_ptr<int64_t>();
      for (const auto idx : c10::irange(num_indices)) {
        const int64_t value = src[idx];
        TORCH_CHECK_INDEX(
            value >= 0 && value < row_count,
            "Vulkan gather_rows_2d: index ",
            value,
            " is out of bounds for dimension 0 with size ",
            row_count);
        dst[idx] = safe_downcast<int32_t>(value);
      }
    } else {
      const int32_t* const src = indices.const_data_ptr<int32_t>();
      for (const auto idx : c10::irange(num_indices)) {
        const int64_t value = src[idx];
        TORCH_CHECK_INDEX(
            value >= 0 && value < row_count,
            "Vulkan gather_rows_2d: index ",
            value,
            " is out of bounds for dimension 0 with size ",
            row_count);
        dst[idx] = src[idx];
      }
    }
  }

  std::vector<int64_t> output_sizes = indices.sizes().vec();
  output_sizes.push_back(row_width);
  if (weight_arg.scalar_type() == kFloat && num_indices > 65535) {
    // Large 2D gathers such as BEiT's relative-position-bias lookup still
    // exceed the reliable Vulkan gather envelope on this backend. Materialize
    // the rows on CPU, then move the gathered result back to Vulkan.
    Tensor cpu_result =
        weight_arg.cpu().index_select(0, indices.reshape({num_indices}));
    return cpu_result.reshape(output_sizes).vulkan();
  }

  if (weight_arg.scalar_type() == kFloat) {
    if (
        v_weight.storage_type() != api::StorageType::BUFFER ||
        !v_weight.has_direct_buffer_layout() ||
        v_weight.gpu_memory_layout() !=
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED) {
      v_weight = utils::materialize_to_contiguous_buffer(
          v_weight, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
    }

    constexpr int64_t kRowChunk = 4096;
    const int64_t dispatch_rows = std::min(num_indices, kRowChunk);
    const int64_t dispatch_depth = div_up(num_indices, kRowChunk);

    vTensor v_output{
        context,
        output_sizes,
        convert_dtype(weight_arg.scalar_type()),
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
    };

    const struct Block final {
      ivec4 info;
    } block{
        safe_downcast<int32_t>(row_width),
        safe_downcast<int32_t>(num_indices),
        safe_downcast<int32_t>(kRowChunk),
        0,
    };

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};

    context->submit_compute_job(
        VK_KERNEL(gather_rows_2d_buffer),
        pipeline_barrier,
        {safe_downcast<uint32_t>(row_width),
         safe_downcast<uint32_t>(dispatch_rows),
         safe_downcast<uint32_t>(dispatch_depth)},
        adaptive_work_group_size(
            {safe_downcast<uint32_t>(row_width),
             safe_downcast<uint32_t>(dispatch_rows),
             safe_downcast<uint32_t>(dispatch_depth)}),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        index_buffer.buffer(),
        params.buffer());

    return convert(v_output);
  }

  TORCH_CHECK(
      v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
      "Vulkan gather_rows_2d expects channel-packed 2D Vulkan weights");

  vTensor v_output{
      context,
      output_sizes,
      convert_dtype(weight_arg.scalar_type()),
  };

  const int64_t index_rows = indices.dim() == 2 ? indices.size(0) : 1;
  const int64_t index_cols = indices.dim() == 2 ? indices.size(1) : num_indices;
  const struct Block final {
    ivec4 out_extents;
    ivec4 index_info;
  } block{
      {safe_downcast<int32_t>(v_output.extents().data[0u]),
       safe_downcast<int32_t>(v_output.extents().data[1u]),
       safe_downcast<int32_t>(v_output.extents().data[2u]),
       0},
      {safe_downcast<int32_t>(index_rows),
       safe_downcast<int32_t>(index_cols),
       safe_downcast<int32_t>(indices.dim()),
       0},
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(gather_rows_2d),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      index_buffer.buffer(),
      params.buffer());

  return convert(v_output);
}

Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index) {
  api::AllocationScope allocation_scope("index_select");

  TORCH_CHECK(self.dim() == 2, "Vulkan index_select currently supports 2D tensors only");
  const int64_t normalized_dim = maybe_wrap_dim(dim, self.dim());
  TORCH_CHECK(
      normalized_dim == 0,
      "Vulkan index_select currently only supports dim=0 for 2D tensors");
  TORCH_CHECK(
      index.dim() <= 1,
      "index_select(): Index is supposed to be a vector");

  const Tensor flat_index = index.dim() == 0 ? index.reshape({1}) : index;
  return gather_rows_2d(self, flat_index);
}

Tensor embedding(
    const Tensor& weight,
    const Tensor& indices,
    c10::SymInt /*padding_idx*/,
    bool /*scale_grad_by_freq*/,
    bool /*sparse*/) {
  api::AllocationScope allocation_scope("embedding");

  TORCH_CHECK(weight.dim() == 2, "'weight' must be 2-D");
  TORCH_CHECK(
      indices.scalar_type() == kLong || indices.scalar_type() == kInt,
      "embedding(): Expected dtype int32 or int64 for indices");
  TORCH_CHECK(
      indices.dim() == 1 || indices.dim() == 2,
      "Vulkan embedding currently supports 1D or 2D indices");

  return gather_rows_2d(weight, indices);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::index_select"), TORCH_FN(index_select));
  m.impl(TORCH_SELECTIVE_NAME("aten::embedding"), TORCH_FN(embedding));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
