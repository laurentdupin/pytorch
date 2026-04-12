#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/ops/NativeLayerNorm.h>
#include <ATen/native/vulkan/ops/Norm.h>
#include <ATen/native/vulkan/ops/Utils.h>

#include <ATen/ops/rsqrt.h>
#include <c10/core/InferenceMode.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

bool prefer_buffer_layer_norm(
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

Tensor layer_norm_fused_width(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps) {
  static constexpr FusedNormWidthSpec kSpec{
      "layer_norm.output_only",
      "layer_norm_width",
      "aten::layer_norm.fused_width",
      true,
  };
  Tensor output = fused_norm_width_impl(
      input_arg, normalized_shape, weight_opt, bias_opt, eps, kSpec);
  if (c10::InferenceMode::is_enabled()) {
    maybe_synchronize_after_norm();
  }
  return output;
}

Tensor layer_norm_fused_width_out(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps,
    Tensor& output) {
  static constexpr FusedNormWidthSpec kSpec{
      "layer_norm.output_only",
      "layer_norm_width",
      "aten::layer_norm.fused_width",
      true,
  };
  Tensor result = fused_norm_width_impl(
      input_arg, normalized_shape, weight_opt, bias_opt, eps, kSpec, output);
  if (c10::InferenceMode::is_enabled()) {
    maybe_synchronize_after_norm();
  }
  return result;
}

Tensor layer_norm_context_parameter_to_buffer(const Tensor& tensor) {
  Tensor vulkan_tensor = tensor.is_vulkan() ? tensor : tensor.vulkan();
  return utils::mark_tensor_execution(
      utils::ensure_buffer_storage(
          vulkan_tensor, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
      api::ExecutionLayout::BUFFER_DIRECT,
      true);
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

  if (
      !prefer_buffer_layer_norm(input_arg, normalized_shape) &&
      supports_fused_layer_norm_last_dim(
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
    double eps,
    std::string allocation_label) {
  TORCH_CHECK(weight, "Weight must be provided!");
  weight_ = layer_norm_context_parameter_to_buffer(*weight);
  TORCH_CHECK(bias, "Bias must be provided!");
  bias_ = layer_norm_context_parameter_to_buffer(*bias);
  eps_ = eps;
  allocation_label_ = std::move(allocation_label);
}

LayernormPackedContext LayernormPackedContext::pack(
    c10::impl::GenericList unpacked) {
  return LayernormPackedContext(
      get_optional_tensor(unpacked, ListArgs::kWeight),
      get_optional_tensor(unpacked, ListArgs::kBias),
      unpacked.get(ListArgs::kEps).toDouble(),
      unpacked.get(ListArgs::kLabel).toStringRef());
}

const c10::impl::GenericList LayernormPackedContext::unpack() const {
  c10::impl::GenericList unpacked{c10::AnyType::get()};
  unpacked.reserve(ListArgs::kNumArgs);
  unpacked.emplace_back(weight_.cpu());
  unpacked.emplace_back(bias_.cpu());
  unpacked.emplace_back(eps_);
  unpacked.emplace_back(allocation_label_);
  return unpacked;
}

c10::intrusive_ptr<LayernormPackedContext> create_layernorm_context(
    std::optional<Tensor>&& weight,
    std::optional<Tensor>&& bias,
    double eps) {
  return c10::make_intrusive<LayernormPackedContext>(
      LayernormPackedContext(weight, bias, eps));
}

c10::intrusive_ptr<LayernormPackedContext> create_layernorm_context_labeled(
    std::optional<Tensor>&& weight,
    std::optional<Tensor>&& bias,
    double eps,
    std::string label) {
  return c10::make_intrusive<LayernormPackedContext>(
      LayernormPackedContext(weight, bias, eps, std::move(label)));
}

Tensor run_layernorm_context(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& layernorm_context) {
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const std::optional<Tensor> weight_opt = layernorm_context->weight();
  const std::optional<Tensor> bias_opt = layernorm_context->bias();
  const float eps = api::utils::safe_downcast<float>(layernorm_context->eps());
  std::optional<api::RuntimeLabelScope> runtime_scope;
  if (!layernorm_context->allocation_label().empty()) {
    runtime_scope.emplace(layernorm_context->allocation_label());
  }

  return layer_norm_impl(input, normalized_shape, weight_opt, bias_opt, eps);
}

Tensor run_layernorm_context_out(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& layernorm_context,
    Tensor& output) {
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const std::optional<Tensor> weight_opt = layernorm_context->weight();
  const std::optional<Tensor> bias_opt = layernorm_context->bias();
  const float eps = api::utils::safe_downcast<float>(layernorm_context->eps());
  std::optional<api::RuntimeLabelScope> runtime_scope;
  if (!layernorm_context->allocation_label().empty()) {
    runtime_scope.emplace(layernorm_context->allocation_label());
  }
  const bool prefer_buffer_path =
      prefer_buffer_layer_norm(input, normalized_shape);

  if (
      !prefer_buffer_path &&
      supports_fused_layer_norm_last_dim(
          input, normalized_shape, weight_opt, bias_opt)) {
    return layer_norm_fused_width_out(
        input, normalized_shape, weight_opt, bias_opt, eps, output);
  }

  Tensor result =
      layer_norm_impl(input, normalized_shape, weight_opt, bias_opt, eps);
  if (output.defined() && output.is_vulkan()) {
    if (prefer_buffer_path) {
      output = result;
      return output;
    }
    return rebind_vulkan_output(output, result);
  }
  output = result;
  return output;
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
