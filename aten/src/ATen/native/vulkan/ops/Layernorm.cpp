#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/NativeLayerNorm.h>

#include <ATen/native/vulkan/ops/Common.h>
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
    context->sync_and_reclaim();
  }
}

Tensor layer_norm_fused_width(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps) {
  api::AllocationScope allocation_scope("layer_norm.output_only");
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

  vTensor v_output{
      context,
      v_input.sizes(),
      v_input.dtype(),
  };

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

  context->submit_compute_job(
      VK_KERNEL(layer_norm_width),
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

  utils::log_vulkan_op_hit("aten::layer_norm.fused_width");
  if (c10::InferenceMode::is_enabled()) {
    maybe_synchronize_vulkan_context();
  }
  return convert(v_output);
}

} // namespace

Tensor layer_norm_impl(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps) {
  utils::log_vulkan_op_hit("aten::layer_norm");
  check_layer_norm_inputs(input_arg, normalized_shape, weight_opt, bias_opt);

  TORCH_CHECK(
      weight_opt->defined() && bias_opt->defined(),
      "Vulkan layer_norm expects weight and bias arguments");

  if (supports_fused_layer_norm_last_dim(
          input_arg, normalized_shape, weight_opt, bias_opt)) {
    return layer_norm_fused_width(
        input_arg, normalized_shape, weight_opt, bias_opt, eps);
  }

  return std::get<0>(native_layer_norm_impl(
      input_arg, normalized_shape, weight_opt, bias_opt, eps));
}

LayernormPackedContext::LayernormPackedContext(
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    double eps) {
  TORCH_CHECK(weight, "Weight must be provided!");
  weight_ = weight->vulkan();
  TORCH_CHECK(bias, "Bias must be provided!");
  bias_ = bias->vulkan();
  eps_ = eps;
}

LayernormPackedContext LayernormPackedContext::pack(
    c10::impl::GenericList unpacked) {
  return LayernormPackedContext(
      get_optional_tensor(unpacked, ListArgs::kWeight),
      get_optional_tensor(unpacked, ListArgs::kBias),
      unpacked.get(ListArgs::kEps).toDouble());
}

const c10::impl::GenericList LayernormPackedContext::unpack() const {
  c10::impl::GenericList unpacked{c10::AnyType::get()};
  unpacked.reserve(ListArgs::kNumArgs);
  unpacked.emplace_back(weight_.cpu());
  unpacked.emplace_back(bias_.cpu());
  unpacked.emplace_back(eps_);
  return unpacked;
}

c10::intrusive_ptr<LayernormPackedContext> create_layernorm_context(
    std::optional<Tensor>&& weight,
    std::optional<Tensor>&& bias,
    double eps) {
  return c10::make_intrusive<LayernormPackedContext>(
      LayernormPackedContext(weight, bias, eps));
}

Tensor run_layernorm_context(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& layernorm_context) {
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const std::optional<Tensor> weight_opt = layernorm_context->weight();
  const std::optional<Tensor> bias_opt = layernorm_context->bias();
  const float eps = api::utils::safe_downcast<float>(layernorm_context->eps());

  return layer_norm_impl(input, normalized_shape, weight_opt, bias_opt, eps);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
