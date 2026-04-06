#include <ATen/native/vulkan/ops/RMSNorm.h>

#include <ATen/native/vulkan/ops/Norm.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/core/InferenceMode.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

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
  return rms_norm_fused_width(input, normalized_shape, weight, eps);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
