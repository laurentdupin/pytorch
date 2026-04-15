#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Tensor relu_buffer_out_vulkan(
    const Tensor& input,
    Tensor& output);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
