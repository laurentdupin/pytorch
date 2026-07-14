#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

#include <optional>
#include <utility>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Tensor add_buffer_out_vulkan(
    const Tensor& self,
    const Tensor& other,
    Tensor& output,
    const std::optional<Scalar>& alpha = std::nullopt,
    const char* callsite = nullptr);

std::optional<Tensor> try_add_scaled_buffer_out_vulkan(
    const Tensor& self,
    const Tensor& other,
    const Tensor& scale,
    Tensor& output);

std::optional<std::pair<Tensor, Tensor>> try_add_relu_buffer_out_vulkan(
    const Tensor& self,
    const Tensor& other,
    Tensor& add_output,
    Tensor& relu_output);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
