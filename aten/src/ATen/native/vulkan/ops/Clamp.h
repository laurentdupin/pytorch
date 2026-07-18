#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

#include <optional>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Tensor relu_buffer_out_vulkan(
    const Tensor& input,
    Tensor& output);

std::optional<Tensor> try_relu_buffer_out_vulkan(
    const Tensor& input,
    Tensor& output);

Tensor& gelu_buffer_inplace_vulkan(
    Tensor& input,
    std::string_view approximate);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
