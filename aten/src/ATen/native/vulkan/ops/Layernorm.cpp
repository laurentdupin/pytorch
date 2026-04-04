#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/Utils.h>

#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/native_layer_norm.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {

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

  // We invoke native_layer_norm which returns a tuple of tensors: <layer_norm,
  // mean, 1/sqrt(var+eps)>, but we only need the first tensor (layer_norm).
  std::tuple<Tensor, Tensor, Tensor> native_layer_norm_output =
      at::native_layer_norm(input, normalized_shape, weight_opt, bias_opt, eps);
  return std::get<0>(native_layer_norm_output);
}

static Tensor layer_norm(
    const at::Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    double eps,
    bool /* cudnn_enable, deprecated */) {
  return run_layernorm_context(
      input_arg,
      normalized_shape,
      c10::make_intrusive<LayernormPackedContext>(
          LayernormPackedContext(weight_opt, bias_opt, eps)));
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::layer_norm"), TORCH_FN(layer_norm));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
