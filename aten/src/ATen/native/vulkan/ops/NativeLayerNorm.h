#pragma once

#ifdef USE_VULKAN_API

#include <ATen/core/Tensor.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

void check_layer_norm_inputs(
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias);

bool supports_fused_layer_norm_last_dim(
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias);

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_impl(
    const at::Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
