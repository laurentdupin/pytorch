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
    const std::optional<Scalar>& alpha = std::nullopt);

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

Tensor materialize_deferred_image_normalize_candidate_if_needed(
    const Tensor& tensor);

void move_deferred_image_normalize_candidate_to_alias(
    const Tensor& source,
    const Tensor& alias);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
