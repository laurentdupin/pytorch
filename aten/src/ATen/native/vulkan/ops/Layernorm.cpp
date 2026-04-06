#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/NativeLayerNorm.h>
#include <ATen/native/vulkan/ops/Norm.h>
#include <ATen/native/vulkan/ops/Utils.h>

#include <c10/core/InferenceMode.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

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
