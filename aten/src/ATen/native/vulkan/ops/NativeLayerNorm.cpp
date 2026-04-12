#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/NativeLayerNorm.h>
#include <ATen/native/vulkan/ops/Norm.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/planning/ExecutionObjects.h>
#include <ATen/ops/rsqrt.h>
#include <c10/core/InferenceMode.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

size_t native_layer_norm_runtime_scratch_bytes(const Tensor& input) {
  return std::max<size_t>(
      64u * 1024u,
      static_cast<size_t>(std::max<int64_t>(1, input.numel())) *
          sizeof(float) * 2u);
}

void check_layer_norm_inputs_impl(
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

bool supports_fused_layer_norm_last_dim_impl(
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias) {
  return supports_fused_norm_last_dim(
      input, normalized_shape, weight, bias, true);
}

bool prefer_buffer_layer_norm_impl(
    const Tensor& input_arg,
    IntArrayRef normalized_shape) {
  if (!input_arg.is_vulkan()) {
    return false;
  }
  const auto request = utils::make_vulkan_tensor_norm_request(
      input_arg, utils::VulkanTensorRole::Input);
  const auto runtime_policy = utils::build_vulkan_runtime_policy(request);
  return runtime_policy.norm_kernel_family ==
          utils::VulkanNormKernelFamily::UnifiedBufferView &&
      normalized_shape.size() == 1u &&
      normalized_shape.front() == input_arg.size(-1);
}

bool can_run_buffer_layer_norm_width(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt) {
  if (
      !input_arg.is_vulkan() ||
      input_arg.scalar_type() != kFloat ||
      input_arg.dim() < 2 ||
      input_arg.dim() > 4 ||
      normalized_shape.size() != 1u ||
      normalized_shape.front() != input_arg.size(-1) ||
      !weight_opt->defined() ||
      !bias_opt->defined() ||
      weight_opt->scalar_type() != kFloat ||
      bias_opt->scalar_type() != kFloat) {
    return false;
  }

  const vTensor& v_input = convert(input_arg);
  return v_input.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_reduction_compute(v_input);
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_fused_width(
    const at::Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    double eps) {
  const auto input_request =
      utils::make_vulkan_tensor_norm_request(
          input_arg, utils::VulkanTensorRole::Input);
  const auto runtime_policy = utils::build_vulkan_runtime_policy(input_request);
  api::AllocationScope allocation_scope("layer_norm.fused_width");
  api::Context* const context = api::context();

  auto weight_request = input_request;
  weight_request.tensor_role = utils::VulkanTensorRole::Weight;
  auto bias_request = input_request;
  bias_request.tensor_role = utils::VulkanTensorRole::Bias;
  log_norm_kernel_family_choice(runtime_policy);
  utils::prime_labeled_scratch_arena_for_request(
      input_arg,
      input_request,
      native_layer_norm_runtime_scratch_bytes(input_arg),
      "native_layer_norm_decode");
  Tensor input = utils::prepare_vulkan_execution_tensor(
      input_arg, utils::VulkanExecutionPlanKind::NormInput, input_request);
  Tensor weight = utils::prepare_vulkan_execution_tensor(
      *weight_opt, utils::VulkanExecutionPlanKind::NormInput, weight_request);
  Tensor bias = utils::prepare_vulkan_execution_tensor(
      *bias_opt, utils::VulkanExecutionPlanKind::NormInput, bias_request);

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
    maybe_synchronize_after_norm();
  }
  return std::make_tuple(convert(v_output), convert(v_mean), convert(v_std_inv));
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_buffer_width(
    const at::Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    double eps) {
  const auto input_request = utils::make_vulkan_tensor_norm_request(
      input_arg, utils::VulkanTensorRole::Input);
  const auto runtime_policy = utils::build_vulkan_runtime_policy(input_request);
  api::AllocationScope allocation_scope("layer_norm.buffer_width");
  api::Context* const context = api::context();

  log_norm_kernel_family_choice(runtime_policy);
  utils::log_vulkan_op_hit("aten::native_layer_norm.buffer_width");

  Tensor weight = utils::prepare_vulkan_execution_tensor(
      *weight_opt, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  Tensor bias = utils::prepare_vulkan_execution_tensor(
      *bias_opt, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);

  const vTensor& v_input = convert(input_arg);
  const vTensor& v_weight = convert(weight);
  const vTensor& v_bias = convert(bias);

  std::vector<int64_t> stats_sizes = input_arg.sizes().vec();
  stats_sizes.back() = 1;

  vTensor v_output{
      context,
      v_input.sizes(),
      v_input.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };
  vTensor v_mean{
      context,
      stats_sizes,
      v_input.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };
  vTensor v_std_inv{
      context,
      stats_sizes,
      v_input.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const uint32_t normalized_size = safe_downcast<uint32_t>(
      std::max<int64_t>(normalized_shape.front(), 1));
  const uint32_t row_count =
      safe_downcast<uint32_t>(v_input.numel() / normalized_size);

  const struct Block final {
    float eps;
    float fill0;
    float fill1;
    float fill2;
  } block{
      safe_downcast<float>(eps),
      0.0f,
      0.0f,
      0.0f,
  };

  api::UniformParamsBuffer params(context, block);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer mean_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_mean);
  api::UniformParamsBuffer std_inv_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_std_inv);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);
  api::UniformParamsBuffer weight_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_weight);
  api::UniformParamsBuffer bias_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_bias);

  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{row_count, 1u, 1u};
  context->submit_compute_job(
      VK_KERNEL(native_layer_norm_width_buffer_float),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_mean.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      mean_meta.buffer(),
      v_std_inv.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      std_inv_meta.buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer(),
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      weight_meta.buffer(),
      v_bias.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      bias_meta.buffer(),
      params.buffer());

  if (c10::InferenceMode::is_enabled()) {
    maybe_synchronize_after_norm();
  }

  return std::make_tuple(
      utils::mark_tensor_execution(
          convert(v_output), api::ExecutionLayout::BUFFER_DIRECT),
      utils::mark_tensor_execution(
          convert(v_mean), api::ExecutionLayout::BUFFER_DIRECT),
      utils::mark_tensor_execution(
          convert(v_std_inv), api::ExecutionLayout::BUFFER_DIRECT));
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_fallback(
    const at::Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    double eps,
    bool prefer_buffer_path) {
  const auto input_request =
      utils::make_vulkan_tensor_norm_request(
          input_arg, utils::VulkanTensorRole::Input);
  const auto runtime_policy = utils::build_vulkan_runtime_policy(input_request);
  api::AllocationScope allocation_scope("layer_norm.fallback");

  auto weight_request = input_request;
  weight_request.tensor_role = utils::VulkanTensorRole::Weight;
  auto bias_request = input_request;
  bias_request.tensor_role = utils::VulkanTensorRole::Bias;
  log_norm_kernel_family_choice(runtime_policy);
  Tensor input;
  Tensor weight;
  Tensor bias;
  if (prefer_buffer_path) {
    utils::log_vulkan_op_hit("aten::native_layer_norm.buffer_fallback");
    input = utils::prepare_vulkan_execution_tensor(
        input_arg, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
    weight = utils::prepare_vulkan_execution_tensor(
        *weight_opt, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
    bias = utils::prepare_vulkan_execution_tensor(
        *bias_opt, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  } else {
    input = utils::prepare_vulkan_execution_tensor(
        input_arg, utils::VulkanExecutionPlanKind::NormInput, input_request);
    weight = utils::prepare_vulkan_execution_tensor(
        *weight_opt, utils::VulkanExecutionPlanKind::NormInput, weight_request);
    bias = utils::prepare_vulkan_execution_tensor(
        *bias_opt, utils::VulkanExecutionPlanKind::NormInput, bias_request);
  }

  std::vector<int64_t> dims_to_reduce;
  dims_to_reduce.reserve(normalized_shape.size());
  for (const auto i : c10::irange(normalized_shape.size())) {
    dims_to_reduce.push_back(input_arg.dim() - i - 1);
  }
  const IntArrayRef dims_to_reduce_ref = IntArrayRef(dims_to_reduce);

  Tensor mean = input.mean(dims_to_reduce_ref, true);
  Tensor input_minus_mean = input.sub(mean);
  Tensor squared_centered = input_minus_mean.mul(input_minus_mean);
  Tensor var = squared_centered.mean(dims_to_reduce_ref, true);
  Tensor var_plus_eps = var.add(eps);
  Tensor std_inv = at::rsqrt(var_plus_eps);
  Tensor normalized = input_minus_mean.mul(std_inv);
  Tensor scaled = normalized.mul(weight);
  Tensor layernorm = scaled.add(bias);
  if (c10::InferenceMode::is_enabled()) {
    maybe_synchronize_after_norm();
  }
  return std::make_tuple(layernorm, mean, std_inv);
}

} // namespace

void check_layer_norm_inputs(
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias) {
  check_layer_norm_inputs_impl(input, normalized_shape, weight, bias);
}

bool supports_fused_layer_norm_last_dim(
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias) {
  return supports_fused_layer_norm_last_dim_impl(
      input, normalized_shape, weight, bias);
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_impl(
    const at::Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps) {
  api::AllocationScope allocation_scope("layer_norm");
  utils::log_vulkan_op_hit("aten::native_layer_norm");
  check_layer_norm_inputs_impl(
      input_arg, normalized_shape, weight_opt, bias_opt);

  TORCH_CHECK(
      input_arg.dim() >= 2 && input_arg.dim() <= 4,
      "Vulkan layernorm expects input of 2d, 3d or 4d!");

  TORCH_CHECK(
      weight_opt->defined() && bias_opt->defined(),
      "Vulkan layernorm expects weight and bias arguments");

  const bool prefer_buffer_path =
      prefer_buffer_layer_norm_impl(input_arg, normalized_shape);

  if (
      prefer_buffer_path &&
      can_run_buffer_layer_norm_width(
          input_arg, normalized_shape, weight_opt, bias_opt)) {
    return native_layer_norm_buffer_width(
        input_arg, normalized_shape, weight_opt, bias_opt, eps);
  }

  if (
      !prefer_buffer_path &&
      supports_fused_layer_norm_last_dim_impl(
          input_arg, normalized_shape, weight_opt, bias_opt)) {
    return native_layer_norm_fused_width(
        input_arg, normalized_shape, weight_opt, bias_opt, eps);
  }

  return native_layer_norm_fallback(
      input_arg,
      normalized_shape,
      weight_opt,
      bias_opt,
      eps,
      prefer_buffer_path);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
