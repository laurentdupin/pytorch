#pragma once

#ifdef USE_VULKAN_API

#include <ATen/core/ArrayRef.h>
#include <ATen/core/ScalarType.h>

#include <cstdint>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

enum class SmallSpatialPointwiseConvFamily : uint8_t {
  None = 0u,
  DepthVisionProjection,
  OCRProjection,
  DiffusionProjection,
};

struct SmallSpatialPointwiseConvMatch final {
  bool matched{false};
  SmallSpatialPointwiseConvFamily family{
      SmallSpatialPointwiseConvFamily::None};
  const char* tuple_id{nullptr};
};

const char* small_spatial_pointwise_conv_family_name(
    SmallSpatialPointwiseConvFamily family);

const char* small_spatial_pointwise_conv_route_label(
    SmallSpatialPointwiseConvFamily family);

const char* small_spatial_pointwise_conv_op_hit_label(
    SmallSpatialPointwiseConvFamily family);

SmallSpatialPointwiseConvMatch match_small_spatial_pointwise_conv_contract(
    IntArrayRef input_sizes,
    IntArrayRef weight_sizes,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    ScalarType dtype);

bool matches_small_spatial_pointwise_conv_contract(
    IntArrayRef input_sizes,
    IntArrayRef weight_sizes,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    ScalarType dtype);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
