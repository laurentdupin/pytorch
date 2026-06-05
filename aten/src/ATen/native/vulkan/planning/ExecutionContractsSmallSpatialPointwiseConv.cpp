#include <ATen/native/vulkan/planning/ExecutionContracts.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

struct SmallSpatialPointwiseConvTuple final {
  SmallSpatialPointwiseConvFamily family;
  int64_t input_c;
  int64_t input_h;
  int64_t input_w;
  int64_t output_c;
  const char* tuple_id;
  ExecutionContractMetadata metadata;
};

constexpr ExecutionContractMetadata make_execution_contract_metadata(
    const char* contract_name,
    const char* family_name,
    const char* tuple_id,
    const char* evidence_id,
    const char* guard_id,
    const char* fallback_policy,
    const char* materialization_policy) {
  return ExecutionContractMetadata{
      contract_name,
      family_name,
      tuple_id,
      evidence_id,
      guard_id,
      fallback_policy,
      materialization_policy};
}

constexpr const char* kFallbackUnsupportedShapesDoNotMatch =
    "unsupported_shapes_do_not_match";
constexpr const char* kMaterializationNativeBufferKernel =
    "native_buffer_kernel";

#define SMALL_SPATIAL_POINTWISE_CONV_TUPLE(                            \
    FAMILY, INPUT_C, INPUT_H, INPUT_W, OUTPUT_C, TUPLE_ID)             \
  {                                                                    \
      SmallSpatialPointwiseConvFamily::FAMILY,                         \
      INPUT_C,                                                         \
      INPUT_H,                                                         \
      INPUT_W,                                                         \
      OUTPUT_C,                                                        \
      TUPLE_ID,                                                        \
      make_execution_contract_metadata(                                \
          "SmallSpatialPointwiseConvContract",                         \
          #FAMILY,                                                     \
          TUPLE_ID,                                                    \
          "small_spatial_pointwise_conv_focused_tests",                \
          "small_spatial_pointwise_conv_adjacent_guards",              \
          kFallbackUnsupportedShapesDoNotMatch,                        \
          kMaterializationNativeBufferKernel)}

constexpr SmallSpatialPointwiseConvTuple kSmallSpatialPointwiseConvTuples[] = {
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DepthVisionProjection, 384, 15, 10, 192, "depth_projection_384_15x10_192"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DepthVisionProjection, 384, 15, 10, 384, "depth_projection_384_15x10_384"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DepthVisionProjection, 384, 20, 13, 192, "depth_projection_384_20x13_192"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DepthVisionProjection, 384, 20, 13, 384, "depth_projection_384_20x13_384"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DepthVisionProjection, 384, 30, 20, 192, "depth_projection_384_30x20_192"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DepthVisionProjection, 384, 30, 20, 384, "depth_projection_384_30x20_384"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DepthVisionProjection, 384, 37, 57, 192, "depth_projection_384_37x57_192"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DepthVisionProjection, 384, 37, 57, 384, "depth_projection_384_37x57_384"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DepthVisionProjection, 384, 45, 30, 192, "depth_projection_384_45x30_192"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DepthVisionProjection, 384, 45, 30, 384, "depth_projection_384_45x30_384"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        OCRProjection, 384, 7, 7, 384, "ocr_projection_384_7x7_384"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        OCRProjection, 512, 7, 7, 512, "ocr_projection_512_7x7_512"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        OCRProjection, 512, 14, 14, 192, "ocr_projection_512_14x14_192"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        OCRProjection, 512, 14, 14, 1024, "ocr_projection_512_14x14_1024"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        OCRProjection, 512, 1, 1, 1280, "ocr_projection_512_1x1_1280"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        OCRProjection, 1024, 7, 7, 384, "ocr_projection_1024_7x7_384"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        OCRProjection, 1024, 7, 7, 2048, "ocr_projection_1024_7x7_2048"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        OCRProjection, 1024, 14, 14, 192, "ocr_projection_1024_14x14_192"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        OCRProjection, 1024, 14, 14, 256, "ocr_projection_1024_14x14_256"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        OCRProjection, 1664, 14, 14, 512, "ocr_projection_1664_14x14_512"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        OCRProjection, 2048, 7, 7, 256, "ocr_projection_2048_7x7_256"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        OCRProjection, 2176, 14, 14, 512, "ocr_projection_2176_14x14_512"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        OCRProjection, 3328, 7, 7, 1024, "ocr_projection_3328_7x7_1024"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DiffusionProjection, 128, 72, 112, 256, "diffusion_projection_128_72x112_256"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DiffusionProjection, 256, 36, 56, 512, "diffusion_projection_256_36x56_512"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DiffusionProjection, 8, 18, 28, 8, "diffusion_projection_8_18x28_8"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DiffusionProjection, 320, 9, 14, 640, "diffusion_projection_320_9x14_640"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DiffusionProjection, 640, 5, 7, 1280, "diffusion_projection_640_5x7_1280"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DiffusionProjection, 2560, 3, 4, 1280, "diffusion_projection_2560_3x4_1280"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DiffusionProjection, 2560, 5, 7, 1280, "diffusion_projection_2560_5x7_1280"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DiffusionProjection, 1920, 5, 7, 1280, "diffusion_projection_1920_5x7_1280"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DiffusionProjection, 1920, 9, 14, 640, "diffusion_projection_1920_9x14_640"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DiffusionProjection, 1280, 9, 14, 640, "diffusion_projection_1280_9x14_640"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DiffusionProjection, 960, 9, 14, 640, "diffusion_projection_960_9x14_640"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DiffusionProjection, 960, 18, 28, 320, "diffusion_projection_960_18x28_320"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DiffusionProjection, 640, 18, 28, 320, "diffusion_projection_640_18x28_320"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DiffusionProjection, 4, 18, 28, 4, "diffusion_projection_4_18x28_4"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DiffusionProjection, 512, 72, 112, 256, "diffusion_projection_512_72x112_256"),
    SMALL_SPATIAL_POINTWISE_CONV_TUPLE(
        DiffusionProjection, 256, 144, 224, 128, "diffusion_projection_256_144x224_128"),
};

#undef SMALL_SPATIAL_POINTWISE_CONV_TUPLE

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
      groups != 1 || input_sizes[0] != 1 || input_sizes[1] != weight_sizes[1] ||
      weight_sizes[2] != 1 || weight_sizes[3] != 1 || stride[0] != 1 ||
      stride[1] != 1 || padding[0] != 0 || padding[1] != 0 ||
      dilation[0] != 1 || dilation[1] != 1) {
    return result;
  }

  for (const SmallSpatialPointwiseConvTuple& tuple :
       kSmallSpatialPointwiseConvTuples) {
    if (
        input_sizes[1] == tuple.input_c &&
        input_sizes[2] == tuple.input_h &&
        input_sizes[3] == tuple.input_w &&
        weight_sizes[0] == tuple.output_c) {
      result.matched = true;
      result.family = tuple.family;
      result.tuple_id = tuple.tuple_id;
      result.metadata = &tuple.metadata;
      return result;
    }
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
