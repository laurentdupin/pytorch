#include <ATen/native/vulkan/ops/Norm.h>

#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/planning/ExecutionObjects.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

using namespace api::utils;

namespace {

size_t norm_runtime_scratch_bytes(const Tensor& input) {
  return std::max<size_t>(
      64u * 1024u,
      static_cast<size_t>(std::max<int64_t>(1, input.numel())) *
          sizeof(float));
}

Tensor& ensure_texture_output_tensor(
    Tensor& output,
    const std::vector<int64_t>& sizes,
    const c10::ScalarType dtype) {
  bool needs_allocation = !output.defined() || !output.is_vulkan() ||
      output.scalar_type() != dtype || !output.sizes().equals(sizes);
  if (!needs_allocation) {
    const vTensor& v_output = convert(output);
    needs_allocation =
        v_output.storage_type() != api::StorageType::TEXTURE_3D;
  }
  if (needs_allocation) {
    output =
        convert(vTensor{api::context(), sizes, convert_dtype(dtype)});
  }
  return output;
}

Tensor fused_norm_width_impl_internal(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps,
    const FusedNormWidthSpec& spec,
    Tensor* output_opt) {
  const auto input_request =
      utils::make_vulkan_tensor_norm_request(
          input_arg, utils::VulkanTensorRole::Input);
  const auto runtime_policy = utils::build_vulkan_runtime_policy(input_request);
  api::AllocationScope allocation_scope(spec.allocation_scope);
  api::Context* const context = api::context();

  auto weight_request = input_request;
  weight_request.tensor_role = utils::VulkanTensorRole::Weight;
  log_norm_kernel_family_choice(runtime_policy);
  utils::prime_labeled_scratch_arena_for_request(
      input_arg,
      input_request,
      norm_runtime_scratch_bytes(input_arg),
      "norm_decode");
  Tensor input = utils::prepare_vulkan_execution_tensor(
      input_arg, utils::VulkanExecutionPlanKind::NormInput, input_request);
  Tensor weight = utils::prepare_vulkan_execution_tensor(
      *weight_opt, utils::VulkanExecutionPlanKind::NormInput, weight_request);

  const vTensor& v_input = convert(input);
  const vTensor& v_weight = convert(weight);
  Tensor output_tensor = output_opt
      ? ensure_texture_output_tensor(*output_opt, v_input.sizes(), input.scalar_type())
      : convert(vTensor{
            context,
            v_input.sizes(),
            v_input.dtype(),
        });
  vTensor& v_output = convert(output_tensor);

  const struct Block final {
    ivec4 output_extents;
    int32_t normalized_size;
    float eps;
    ivec2 fill0;
  } block{
      ivec4{
          safe_downcast<int32_t>(v_output.extents().data[0u]),
          safe_downcast<int32_t>(v_output.extents().data[1u]),
          safe_downcast<int32_t>(v_output.extents().data[2u]),
          0,
      },
      safe_downcast<int32_t>(normalized_shape.front()),
      safe_downcast<float>(eps),
      ivec2{0, 0},
  };
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  if (spec.has_bias) {
    auto bias_request = input_request;
    bias_request.tensor_role = utils::VulkanTensorRole::Bias;
    Tensor bias = utils::prepare_vulkan_execution_tensor(
        *bias_opt,
        utils::VulkanExecutionPlanKind::NormInput,
        bias_request);
    const vTensor& v_bias = convert(bias);
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(spec.shader_name),
        pipeline_barrier,
        v_output.extents(),
        adaptive_work_group_size(v_output.extents()),
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  } else {
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(spec.shader_name),
        pipeline_barrier,
        v_output.extents(),
        adaptive_work_group_size(v_output.extents()),
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  }

  utils::log_vulkan_op_hit(spec.op_hit_name);
  std::vector<Tensor> provenance_inputs{input_arg, *weight_opt};
  if (bias_opt && bias_opt->defined()) {
    provenance_inputs.emplace_back(*bias_opt);
  }
  return record_tensor_write_and_return(
      output_tensor, "aten::norm", spec.op_hit_name, provenance_inputs);
}

} // namespace

bool supports_fused_norm_last_dim(
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    bool require_bias) {
  return normalized_shape.size() == 1u && input.dim() >= 2 && input.dim() <= 4 &&
      normalized_shape.front() == input.size(-1) &&
      input.scalar_type() == kFloat &&
      weight && weight->defined() && weight->scalar_type() == kFloat &&
      weight->sizes().equals(normalized_shape) &&
      (!require_bias ||
       (bias && bias->defined() && bias->scalar_type() == kFloat &&
        bias->sizes().equals(normalized_shape)));
}

void maybe_synchronize_after_norm() {
  api::Context* const context = api::context();
  if (context->owns_graph_program_invocation()) {
    return;
  }
  context->submit_pending_work_and_poll_retire();
}

Tensor fused_norm_width_impl(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps,
    const FusedNormWidthSpec& spec) {
  return fused_norm_width_impl_internal(
      input_arg,
      normalized_shape,
      weight_opt,
      bias_opt,
      eps,
      spec,
      nullptr);
}

Tensor fused_norm_width_impl(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps,
    const FusedNormWidthSpec& spec,
    Tensor& output) {
  return fused_norm_width_impl_internal(
      input_arg,
      normalized_shape,
      weight_opt,
      bias_opt,
      eps,
      spec,
      &output);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
