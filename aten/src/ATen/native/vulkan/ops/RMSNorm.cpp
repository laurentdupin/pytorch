#include <ATen/native/vulkan/ops/RMSNorm.h>

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Norm.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/core/InferenceMode.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor rms_norm_buffer_width(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    double eps) {
  api::AllocationScope allocation_scope("rms_norm.buffer_width");
  api::Context* const context = api::context();
  utils::log_vulkan_op_hit("aten::rms_norm.fused_width");
  utils::log_vulkan_op_hit("aten::rms_norm.buffer_width");

  Tensor input = utils::ensure_buffer_storage(
      input_arg, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  Tensor weight = utils::ensure_buffer_storage(
      *weight_opt, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  Tensor output = utils::create_buffer_tensor(
      input.sizes(), input.scalar_type(), /*persistent=*/false);

  const vTensor& v_input = convert(input);
  const vTensor& v_weight = convert(weight);
  vTensor& v_output = convert(output);

  TORCH_CHECK(
      utils::supports_buffer_reduction_compute(v_input) &&
          utils::supports_buffer_elementwise_compute(v_weight),
      "Vulkan rms_norm buffer path requires buffer-compatible input and weight");

  const struct Block final {
    float eps;
    float fill0;
    float fill1;
    float fill2;
  } block{
      api::utils::safe_downcast<float>(eps),
      0.0f,
      0.0f,
      0.0f,
  };

  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer input_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);
  api::UniformParamsBuffer weight_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_weight);
  api::UniformParamsBuffer params(context, block);

  const uint32_t normalized_size = api::utils::safe_downcast<uint32_t>(
      std::max<int64_t>(normalized_shape.front(), 1));
  const uint32_t row_count = api::utils::safe_downcast<uint32_t>(
      v_output.numel() / normalized_size);
  const api::utils::uvec3 global_size{row_count, 1u, 1u};

  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      VK_KERNEL(rms_norm_width_buffer_float),
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
      input_meta.buffer(),
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      weight_meta.buffer(),
      params.buffer());

  if (c10::InferenceMode::is_enabled()) {
    maybe_synchronize_after_norm();
  }
  return record_tensor_write_and_return(
      output,
      "aten::rms_norm",
      "buffer_width",
      {input, weight});
}

Tensor rms_norm_fused_width(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    double eps) {
  static constexpr FusedNormWidthSpec kSpec{
      "rms_norm.fused_width",
      "rms_norm_width",
      "aten::rms_norm.fused_width",
      false,
  };
  Tensor output = fused_norm_width_impl(
      input_arg, normalized_shape, weight_opt, std::nullopt, eps, kSpec);
  if (c10::InferenceMode::is_enabled()) {
    maybe_synchronize_after_norm();
  }
  return output;
}

} // namespace

bool supports_fused_rms_norm_last_dim(
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight) {
  return supports_fused_norm_last_dim(
      input, normalized_shape, weight, std::nullopt, false);
}

Tensor rms_norm_impl(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight,
    double eps) {
  utils::log_vulkan_op_hit("aten::rms_norm");
  TORCH_CHECK(
      supports_fused_rms_norm_last_dim(input, normalized_shape, weight),
      "Vulkan rms_norm expects 2d-4d float input, last-dim normalization, and float weight");
  return rms_norm_buffer_width(input, normalized_shape, weight, eps);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
