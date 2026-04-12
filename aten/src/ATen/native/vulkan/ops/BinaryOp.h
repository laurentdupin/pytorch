#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Tensor add_buffer_out_vulkan(
    const Tensor& self,
    const Tensor& other,
    Tensor& output,
    const std::optional<Scalar>& alpha = std::nullopt);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
