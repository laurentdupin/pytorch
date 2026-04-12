#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Tensor upsample_bilinear2d_buffer_out_vulkan(
    const Tensor& input,
    const IntArrayRef output_sizes,
    bool align_corners,
    const std::optional<double> scales_h,
    const std::optional<double> scales_w,
    Tensor& output);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
