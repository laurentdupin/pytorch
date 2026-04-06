#pragma once

#ifdef USE_VULKAN_API

#include <ATen/core/Tensor.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

bool supports_fused_rms_norm_last_dim(
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight);

Tensor rms_norm_impl(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight,
    double eps);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
