#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/NativeLayerNorm.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/core/InferenceMode.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

void maybe_synchronize_vulkan_context() {
  api::Context* const context = api::context();
  if (context->should_sync_and_reclaim()) {
    // The no-spill path only needs a lightweight device-side reclaim barrier
    // when deferred Vulkan cleanup pressure or submission age gets too high.
    context->sync_and_reclaim();
  }
}

} // namespace

void check_layer_norm_inputs(
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight /* optional */,
    const std::optional<Tensor>& bias /* optional */) {
  const auto normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight->defined() || weight->sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight->sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias->defined() || bias->sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias->sizes(),
      " and normalized_shape = ",
      normalized_shape);

  const auto input_shape = input.sizes();
  const auto input_ndim = input.sizes().size();

  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    TORCH_CHECK(false, ss.str());
  }
}

bool supports_fused_layer_norm_last_dim(
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias) {
  return normalized_shape.size() == 1u && input.dim() >= 2 && input.dim() <= 4 &&
      normalized_shape.front() == input.size(-1) &&
      input.scalar_type() == kFloat &&
      weight && bias && weight->defined() && bias->defined() &&
      weight->scalar_type() == kFloat && bias->scalar_type() == kFloat;
}

namespace {

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_fused_width(
    const at::Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    double eps) {
  api::AllocationScope allocation_scope("layer_norm.fused_width");
  api::Context* const context = api::context();

  Tensor input = utils::prepare_vulkan_execution_tensor(
      input_arg, utils::VulkanExecutionPlanKind::NormInput);
  Tensor weight = utils::prepare_vulkan_execution_tensor(
      *weight_opt, utils::VulkanExecutionPlanKind::NormInput);
  Tensor bias = utils::prepare_vulkan_execution_tensor(
      *bias_opt, utils::VulkanExecutionPlanKind::NormInput);

  const vTensor& v_input = convert(input);
  const vTensor& v_weight = convert(weight);
  const vTensor& v_bias = convert(bias);

  std::vector<int64_t> stats_sizes = input_arg.sizes().vec();
  stats_sizes.back() = 1;

  vTensor v_output{
      context,
      v_input.sizes(),
      v_input.dtype(),
  };
  vTensor v_mean{
      context,
      stats_sizes,
      v_input.dtype(),
  };
  vTensor v_std_inv{
      context,
      stats_sizes,
      v_input.dtype(),
  };

  const struct Block final {
    ivec4 input_extents;
    ivec4 stats_extents;
    int32_t normalized_size;
    float eps;
    ivec2 fill0;
  } block{
      ivec4{
          safe_downcast<int32_t>(v_input.extents().data[0u]),
          safe_downcast<int32_t>(v_input.extents().data[1u]),
          safe_downcast<int32_t>(v_input.extents().data[2u]),
          0,
      },
      ivec4{
          safe_downcast<int32_t>(v_mean.extents().data[0u]),
          safe_downcast<int32_t>(v_mean.extents().data[1u]),
          safe_downcast<int32_t>(v_mean.extents().data[2u]),
          0,
      },
      safe_downcast<int32_t>(normalized_shape.front()),
      safe_downcast<float>(eps),
      ivec2{0, 0},
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(native_layer_norm_width),
      pipeline_barrier,
      v_mean.extents(),
      adaptive_work_group_size(v_mean.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_mean.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_std_inv.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  utils::log_vulkan_op_hit("aten::native_layer_norm.fused_width");
  if (c10::InferenceMode::is_enabled()) {
    maybe_synchronize_vulkan_context();
  }
  return std::make_tuple(convert(v_output), convert(v_mean), convert(v_std_inv));
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_fallback(
    const at::Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    double eps) {
  api::AllocationScope allocation_scope("layer_norm.fallback");

  Tensor input = utils::prepare_vulkan_execution_tensor(
      input_arg, utils::VulkanExecutionPlanKind::NormInput);

  const Tensor weight = utils::prepare_vulkan_execution_tensor(
      *weight_opt, utils::VulkanExecutionPlanKind::NormInput);
  const Tensor bias = utils::prepare_vulkan_execution_tensor(
      *bias_opt, utils::VulkanExecutionPlanKind::NormInput);

  std::vector<int64_t> dims_to_reduce;
  dims_to_reduce.reserve(normalized_shape.size());
  for (const auto i : c10::irange(normalized_shape.size())) {
    dims_to_reduce.push_back(input_arg.dim() - i - 1);
  }
  const IntArrayRef dims_to_reduce_ref = IntArrayRef(dims_to_reduce);

  auto mean = input.mean(dims_to_reduce_ref, true);
  auto input_minus_mean = input.sub(mean);
  auto var =
      input_minus_mean.mul(input_minus_mean).mean(dims_to_reduce_ref, true);
  auto std_inv = var.add(eps).pow(-0.5f);
  auto layernorm = input_minus_mean.mul(std_inv).mul(weight).add(bias);
  if (c10::InferenceMode::is_enabled()) {
    maybe_synchronize_vulkan_context();
  }
  return std::make_tuple(layernorm, mean, std_inv);
}

} // namespace

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_impl(
    const at::Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps) {
  api::AllocationScope allocation_scope("layer_norm");
  utils::log_vulkan_op_hit("aten::native_layer_norm");
  check_layer_norm_inputs(input_arg, normalized_shape, weight_opt, bias_opt);

  TORCH_CHECK(
      input_arg.dim() >= 2 && input_arg.dim() <= 4,
      "Vulkan layernorm expects input of 2d, 3d or 4d!");

  TORCH_CHECK(
      weight_opt->defined() && bias_opt->defined(),
      "Vulkan layernorm expects weight and bias arguments");

  if (supports_fused_layer_norm_last_dim(
          input_arg, normalized_shape, weight_opt, bias_opt)) {
    return native_layer_norm_fused_width(
        input_arg, normalized_shape, weight_opt, bias_opt, eps);
  }

  return native_layer_norm_fallback(
      input_arg, normalized_shape, weight_opt, bias_opt, eps);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
