#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/DynamicProgramRuntime.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsSmallSpatialPointwiseConvSpec.h>

#include <string_view>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

constexpr ExecutionContractMetadata kDynamicPointwiseConv1x1DirectBufferMetadata{
    "SmallSpatialPointwiseConvContract",
    "GenericDynamicHW",
    "pointwise_1x1_direct_buffer_generic_dynamic_hw",
    "dynamic_pointwise_conv1x1_unseen_hw_tests",
    "dynamic_pointwise_conv1x1_semantic_guards",
    "unsupported_semantics_hard_fail",
    "native_buffer_kernel"};

SmallSpatialPointwiseConvFamily small_spatial_pointwise_conv_family_from_name(
    const char* const family_name) {
  const std::string_view family{family_name};
  if (family == "DepthVisionProjection") {
    return SmallSpatialPointwiseConvFamily::DepthVisionProjection;
  }
  if (family == "OCRProjection") {
    return SmallSpatialPointwiseConvFamily::OCRProjection;
  }
  if (family == "DiffusionProjection") {
    return SmallSpatialPointwiseConvFamily::DiffusionProjection;
  }
  return SmallSpatialPointwiseConvFamily::None;
}

} // namespace

const char* small_spatial_pointwise_conv_family_name(
    const SmallSpatialPointwiseConvFamily family) {
  switch (family) {
    case SmallSpatialPointwiseConvFamily::DepthVisionProjection:
      return "DepthVisionProjection";
    case SmallSpatialPointwiseConvFamily::OCRProjection:
      return "OCRProjection";
    case SmallSpatialPointwiseConvFamily::DiffusionProjection:
      return "DiffusionProjection";
    case SmallSpatialPointwiseConvFamily::None:
      return "None";
  }
  return "None";
}

const char* small_spatial_pointwise_conv_route_label(
    const SmallSpatialPointwiseConvFamily family) {
  switch (family) {
    case SmallSpatialPointwiseConvFamily::DepthVisionProjection:
      return "SelectedSmallSpatialPointwiseConvDepthVisionProjection";
    case SmallSpatialPointwiseConvFamily::OCRProjection:
      return "SelectedSmallSpatialPointwiseConvOCRProjection";
    case SmallSpatialPointwiseConvFamily::DiffusionProjection:
      return "SelectedSmallSpatialPointwiseConvDiffusionProjection";
    case SmallSpatialPointwiseConvFamily::None:
      return "SelectedSmallSpatialPointwiseConvNone";
  }
  return "SelectedSmallSpatialPointwiseConvNone";
}

const char* small_spatial_pointwise_conv_op_hit_label(
    const SmallSpatialPointwiseConvFamily family) {
  switch (family) {
    case SmallSpatialPointwiseConvFamily::DepthVisionProjection:
      return "aten::convolution.buffer_float_1x1_skip.small_spatial_pointwise.depth_vision_projection";
    case SmallSpatialPointwiseConvFamily::OCRProjection:
      return "aten::convolution.buffer_float_1x1_skip.small_spatial_pointwise.ocr_projection";
    case SmallSpatialPointwiseConvFamily::DiffusionProjection:
      return "aten::convolution.buffer_float_1x1_skip.small_spatial_pointwise.diffusion_projection";
    case SmallSpatialPointwiseConvFamily::None:
      return "aten::convolution.buffer_float_1x1_skip.small_spatial_pointwise";
  }
  return "aten::convolution.buffer_float_1x1_skip.small_spatial_pointwise";
}

const char* dynamic_pointwise_conv1x1_direct_buffer_family_name(
    const DynamicPointwiseConv1x1DirectBufferFamily family) {
  switch (family) {
    case DynamicPointwiseConv1x1DirectBufferFamily::GenericDynamicHW:
      return "GenericDynamicHW";
    case DynamicPointwiseConv1x1DirectBufferFamily::None:
      return "None";
  }
  return "None";
}

const char* dynamic_pointwise_conv1x1_direct_buffer_route_label(
    const DynamicPointwiseConv1x1DirectBufferFamily family) {
  switch (family) {
    case DynamicPointwiseConv1x1DirectBufferFamily::GenericDynamicHW:
      return "SelectedDynamicPointwiseConv1x1DirectBuffer";
    case DynamicPointwiseConv1x1DirectBufferFamily::None:
      return "SelectedDynamicPointwiseConv1x1DirectBufferNone";
  }
  return "SelectedDynamicPointwiseConv1x1DirectBufferNone";
}

const char* dynamic_pointwise_conv1x1_direct_buffer_op_hit_label(
    const DynamicPointwiseConv1x1DirectBufferFamily family) {
  switch (family) {
    case DynamicPointwiseConv1x1DirectBufferFamily::GenericDynamicHW:
      return "aten::convolution.buffer_float_1x1.dynamic_pointwise_direct";
    case DynamicPointwiseConv1x1DirectBufferFamily::None:
      return "aten::convolution.buffer_float_1x1.dynamic_pointwise_direct.none";
  }
  return "aten::convolution.buffer_float_1x1.dynamic_pointwise_direct.none";
}

DynamicPointwiseConv1x1DirectBufferMatch
match_dynamic_pointwise_conv1x1_direct_buffer_contract(
    const IntArrayRef input_sizes,
    const IntArrayRef weight_sizes,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups,
    const ScalarType dtype) {
  DynamicPointwiseConv1x1DirectBufferMatch result;
  if (
      dtype != kFloat || input_sizes.size() != 4 || weight_sizes.size() != 4 ||
      stride.size() != 2 || padding.size() != 2 || dilation.size() != 2 ||
      input_sizes[0] < 1 || groups != 1 ||
      input_sizes[1] != weight_sizes[1] || weight_sizes[2] != 1 ||
      weight_sizes[3] != 1 || stride[0] != 1 || stride[1] != 1 ||
      padding[0] != 0 || padding[1] != 0 || dilation[0] != 1 ||
      dilation[1] != 1) {
    return result;
  }

  const DynamicProgramDecision decision = build_dynamic_program_runtime_plan(
      make_pointwise_conv1x1_direct_buffer_dynamic_program(
          input_sizes,
          weight_sizes,
          stride,
          padding,
          dilation,
          groups,
          dtype,
          &kDynamicPointwiseConv1x1DirectBufferMetadata,
          /*behavior_enabled=*/true));
  if (!decision.runtime_selection_authorized) {
    return result;
  }

  result.matched = true;
  result.family =
      DynamicPointwiseConv1x1DirectBufferFamily::GenericDynamicHW;
  result.tuple_id =
      kDynamicPointwiseConv1x1DirectBufferMetadata.tuple_id;
  result.metadata = &kDynamicPointwiseConv1x1DirectBufferMetadata;
  return result;
}

SmallSpatialPointwiseConvMatch match_small_spatial_pointwise_conv_contract(
    const IntArrayRef input_sizes,
    const IntArrayRef weight_sizes,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups,
    const ScalarType dtype) {
  SmallSpatialPointwiseConvMatch result;
  if (
      dtype != kFloat || input_sizes.size() != 4 || weight_sizes.size() != 4 ||
      stride.size() != 2 || padding.size() != 2 || dilation.size() != 2 ||
      groups != 1 || input_sizes[0] < 1 || input_sizes[0] > 8 ||
      !generated::small_spatial_pointwise_conv_sparse_projection_rows_input_weight_channels_equal(
          input_sizes[1], weight_sizes[1]) ||
      weight_sizes[2] != 1 || weight_sizes[3] != 1 || stride[0] != 1 ||
      stride[1] != 1 || padding[0] != 0 || padding[1] != 0 ||
      dilation[0] != 1 || dilation[1] != 1) {
    return result;
  }

  if (input_sizes[0] == 1 &&
      generated::small_spatial_pointwise_conv_depth_vision_factorized_projection_matches(
          input_sizes[1], input_sizes[2], input_sizes[3], weight_sizes[0])) {
    result.matched = true;
    const char* const family_name =
        generated::
            small_spatial_pointwise_conv_depth_vision_factorized_projection_family_name();
    result.family = small_spatial_pointwise_conv_family_from_name(
        family_name);
    result.tuple_id = generated::
        small_spatial_pointwise_conv_depth_vision_factorized_projection_tuple_id();
    result.metadata = generated::
        small_spatial_pointwise_conv_depth_vision_factorized_projection_metadata();
    return result;
  }

  const auto* const row =
      generated::small_spatial_pointwise_conv_projection_rows_find(
          input_sizes[1], input_sizes[2], input_sizes[3], weight_sizes[0]);
  if (row != nullptr) {
    const SmallSpatialPointwiseConvFamily family =
        small_spatial_pointwise_conv_family_from_name(row->family);
    if (
        input_sizes[0] != 1 &&
        family != SmallSpatialPointwiseConvFamily::OCRProjection) {
      return result;
    }
    result.matched = true;
    result.family = family;
    result.tuple_id = row->tuple_id;
    result.metadata = &row->metadata;
    return result;
  }
  return result;
}

bool matches_small_spatial_pointwise_conv_contract(
    const IntArrayRef input_sizes,
    const IntArrayRef weight_sizes,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups,
    const ScalarType dtype) {
  return match_small_spatial_pointwise_conv_contract(
             input_sizes, weight_sizes, stride, padding, dilation, groups, dtype)
      .matched;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
