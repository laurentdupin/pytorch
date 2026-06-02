#include <ATen/native/vulkan/planning/ExecutionContracts.h>

#include <cmath>

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
};

struct DiffusionSDPATuple final {
  DiffusionSDPAFamily family;
  int64_t heads;
  int64_t query_sequence;
  int64_t key_value_sequence;
  int64_t head_dim;
  const char* tuple_id;
};

constexpr double kTransformerHeadDim128Scale = 0.08838834764831845;
constexpr double kHeadDim64Scale = 0.125;
constexpr double kHeadDim512Scale = 0.04419417382415922;

constexpr SmallSpatialPointwiseConvTuple kSmallSpatialPointwiseConvTuples[] = {
    {SmallSpatialPointwiseConvFamily::DepthVisionProjection, 384, 15, 10, 192, "depth_projection_384_15x10_192"},
    {SmallSpatialPointwiseConvFamily::DepthVisionProjection, 384, 15, 10, 384, "depth_projection_384_15x10_384"},
    {SmallSpatialPointwiseConvFamily::DepthVisionProjection, 384, 20, 13, 192, "depth_projection_384_20x13_192"},
    {SmallSpatialPointwiseConvFamily::DepthVisionProjection, 384, 20, 13, 384, "depth_projection_384_20x13_384"},
    {SmallSpatialPointwiseConvFamily::DepthVisionProjection, 384, 30, 20, 192, "depth_projection_384_30x20_192"},
    {SmallSpatialPointwiseConvFamily::DepthVisionProjection, 384, 30, 20, 384, "depth_projection_384_30x20_384"},
    {SmallSpatialPointwiseConvFamily::DepthVisionProjection, 384, 37, 57, 192, "depth_projection_384_37x57_192"},
    {SmallSpatialPointwiseConvFamily::DepthVisionProjection, 384, 37, 57, 384, "depth_projection_384_37x57_384"},
    {SmallSpatialPointwiseConvFamily::DepthVisionProjection, 384, 45, 30, 192, "depth_projection_384_45x30_192"},
    {SmallSpatialPointwiseConvFamily::DepthVisionProjection, 384, 45, 30, 384, "depth_projection_384_45x30_384"},
    {SmallSpatialPointwiseConvFamily::OCRProjection, 384, 7, 7, 384, "ocr_projection_384_7x7_384"},
    {SmallSpatialPointwiseConvFamily::OCRProjection, 512, 7, 7, 512, "ocr_projection_512_7x7_512"},
    {SmallSpatialPointwiseConvFamily::OCRProjection, 512, 14, 14, 192, "ocr_projection_512_14x14_192"},
    {SmallSpatialPointwiseConvFamily::OCRProjection, 512, 14, 14, 1024, "ocr_projection_512_14x14_1024"},
    {SmallSpatialPointwiseConvFamily::OCRProjection, 512, 1, 1, 1280, "ocr_projection_512_1x1_1280"},
    {SmallSpatialPointwiseConvFamily::OCRProjection, 1024, 7, 7, 384, "ocr_projection_1024_7x7_384"},
    {SmallSpatialPointwiseConvFamily::OCRProjection, 1024, 7, 7, 2048, "ocr_projection_1024_7x7_2048"},
    {SmallSpatialPointwiseConvFamily::OCRProjection, 1024, 14, 14, 192, "ocr_projection_1024_14x14_192"},
    {SmallSpatialPointwiseConvFamily::OCRProjection, 1024, 14, 14, 256, "ocr_projection_1024_14x14_256"},
    {SmallSpatialPointwiseConvFamily::OCRProjection, 1664, 14, 14, 512, "ocr_projection_1664_14x14_512"},
    {SmallSpatialPointwiseConvFamily::OCRProjection, 2048, 7, 7, 256, "ocr_projection_2048_7x7_256"},
    {SmallSpatialPointwiseConvFamily::OCRProjection, 2176, 14, 14, 512, "ocr_projection_2176_14x14_512"},
    {SmallSpatialPointwiseConvFamily::OCRProjection, 3328, 7, 7, 1024, "ocr_projection_3328_7x7_1024"},
    {SmallSpatialPointwiseConvFamily::DiffusionProjection, 128, 72, 112, 256, "diffusion_projection_128_72x112_256"},
    {SmallSpatialPointwiseConvFamily::DiffusionProjection, 256, 36, 56, 512, "diffusion_projection_256_36x56_512"},
    {SmallSpatialPointwiseConvFamily::DiffusionProjection, 8, 18, 28, 8, "diffusion_projection_8_18x28_8"},
    {SmallSpatialPointwiseConvFamily::DiffusionProjection, 320, 9, 14, 640, "diffusion_projection_320_9x14_640"},
    {SmallSpatialPointwiseConvFamily::DiffusionProjection, 640, 5, 7, 1280, "diffusion_projection_640_5x7_1280"},
    {SmallSpatialPointwiseConvFamily::DiffusionProjection, 2560, 3, 4, 1280, "diffusion_projection_2560_3x4_1280"},
    {SmallSpatialPointwiseConvFamily::DiffusionProjection, 2560, 5, 7, 1280, "diffusion_projection_2560_5x7_1280"},
    {SmallSpatialPointwiseConvFamily::DiffusionProjection, 1920, 5, 7, 1280, "diffusion_projection_1920_5x7_1280"},
    {SmallSpatialPointwiseConvFamily::DiffusionProjection, 1920, 9, 14, 640, "diffusion_projection_1920_9x14_640"},
    {SmallSpatialPointwiseConvFamily::DiffusionProjection, 1280, 9, 14, 640, "diffusion_projection_1280_9x14_640"},
    {SmallSpatialPointwiseConvFamily::DiffusionProjection, 960, 9, 14, 640, "diffusion_projection_960_9x14_640"},
    {SmallSpatialPointwiseConvFamily::DiffusionProjection, 960, 18, 28, 320, "diffusion_projection_960_18x28_320"},
    {SmallSpatialPointwiseConvFamily::DiffusionProjection, 640, 18, 28, 320, "diffusion_projection_640_18x28_320"},
    {SmallSpatialPointwiseConvFamily::DiffusionProjection, 4, 18, 28, 4, "diffusion_projection_4_18x28_4"},
    {SmallSpatialPointwiseConvFamily::DiffusionProjection, 512, 72, 112, 256, "diffusion_projection_512_72x112_256"},
    {SmallSpatialPointwiseConvFamily::DiffusionProjection, 256, 144, 224, 128, "diffusion_projection_256_144x224_128"},
};

constexpr DiffusionSDPATuple kDiffusionSDPATuples[] = {
    {DiffusionSDPAFamily::SquareSelfAttention, 1, 640, 640, 512, "square_heads1_sequence640_dim512"},
    {DiffusionSDPAFamily::SquareSelfAttention, 5, 640, 640, 64, "square_heads5_sequence640_dim64"},
    {DiffusionSDPAFamily::SquareSelfAttention, 1, 504, 504, 512, "square_heads1_sequence504_dim512"},
    {DiffusionSDPAFamily::SquareSelfAttention, 5, 504, 504, 64, "square_heads5_sequence504_dim64"},
    {DiffusionSDPAFamily::SquareSelfAttention, 10, 126, 126, 64, "square_heads10_sequence126_dim64"},
    {DiffusionSDPAFamily::SquareSelfAttention, 20, 35, 35, 64, "square_heads20_sequence35_dim64"},
    {DiffusionSDPAFamily::SquareSelfAttention, 20, 12, 12, 64, "square_heads20_sequence12_dim64"},
    {DiffusionSDPAFamily::CrossAttention, 5, 504, 2, 64, "cross_heads5_query504_kv2_dim64"},
    {DiffusionSDPAFamily::CrossAttention, 10, 126, 2, 64, "cross_heads10_query126_kv2_dim64"},
    {DiffusionSDPAFamily::CrossAttention, 20, 35, 2, 64, "cross_heads20_query35_kv2_dim64"},
    {DiffusionSDPAFamily::CrossAttention, 20, 12, 2, 64, "cross_heads20_query12_kv2_dim64"},
};

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

const char* transformer_gqa_sdpa_family_name(
    const TransformerGQASDPAFamily family) {
  switch (family) {
    case TransformerGQASDPAFamily::CausalPrefill:
      return "TransformerGQASDPACausalPrefill";
    case TransformerGQASDPAFamily::SmallNonCausalGQA:
      return "TransformerGQASDPASmallNonCausalGQA";
    case TransformerGQASDPAFamily::DecodeGQA:
      return "TransformerGQASDPADecodeGQA";
    case TransformerGQASDPAFamily::None:
      return "None";
  }
  return "None";
}

const char* transformer_gqa_sdpa_route_label(
    const TransformerGQASDPAFamily family) {
  switch (family) {
    case TransformerGQASDPAFamily::CausalPrefill:
      return "SelectedTransformerGQASDPACausalPrefill";
    case TransformerGQASDPAFamily::SmallNonCausalGQA:
      return "SelectedTransformerGQASDPASmallNonCausalGQA";
    case TransformerGQASDPAFamily::DecodeGQA:
      return "SelectedTransformerGQASDPADecodeGQA";
    case TransformerGQASDPAFamily::None:
      return "SelectedTransformerGQASDPANone";
  }
  return "SelectedTransformerGQASDPANone";
}

TransformerGQASDPAMatch match_transformer_gqa_sdpa_contract(
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const IntArrayRef value_sizes,
    const ScalarType query_dtype,
    const ScalarType key_dtype,
    const ScalarType value_dtype,
    const bool has_attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa) {
  TransformerGQASDPAMatch result;
  if (
      has_attn_mask || dropout_p != 0.0 || (!is_causal && !enable_gqa) ||
      query_dtype != kFloat || key_dtype != kFloat || value_dtype != kFloat ||
      query_sizes.size() != 4 || key_sizes.size() != 4 ||
      value_sizes.size() != 4) {
    return result;
  }
  if (
      scale.has_value() &&
      std::abs(*scale - kTransformerHeadDim128Scale) > 1.0e-6) {
    return result;
  }
  if (
      query_sizes[0] != 1 || key_sizes[0] != 1 || value_sizes[0] != 1 ||
      query_sizes[1] != 16 || query_sizes[2] < 1 ||
      query_sizes[2] > 128 || query_sizes[3] != 128 ||
      key_sizes[2] < query_sizes[2] || key_sizes[3] != 128 ||
      value_sizes[2] != key_sizes[2] || value_sizes[3] != 128 ||
      key_sizes[1] != value_sizes[1]) {
    return result;
  }
  if (enable_gqa) {
    if (key_sizes[1] != 4) {
      return result;
    }
  } else if (key_sizes[1] != 16) {
    return result;
  }

  if (is_causal) {
    if (query_sizes[2] != key_sizes[2] || key_sizes[2] > 128) {
      return result;
    }
    result.matched = true;
    result.family = TransformerGQASDPAFamily::CausalPrefill;
    result.tuple_id = enable_gqa ? "causal_gqa_head128_len_le_128"
                                 : "causal_mha_head128_len_le_128";
    return result;
  }

  if (
      enable_gqa && query_sizes[2] == 1 && key_sizes[2] >= 100 &&
      key_sizes[2] <= 116) {
    result.matched = true;
    result.family = TransformerGQASDPAFamily::DecodeGQA;
    result.tuple_id = "decode_gqa_head128_source_100_116";
    return result;
  }

  if (query_sizes[2] <= 14 && key_sizes[2] <= 64) {
    result.matched = true;
    result.family = TransformerGQASDPAFamily::SmallNonCausalGQA;
    result.tuple_id = "small_non_causal_gqa_head128";
    return result;
  }
  return result;
}

bool matches_transformer_gqa_sdpa_contract(
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const IntArrayRef value_sizes,
    const ScalarType query_dtype,
    const ScalarType key_dtype,
    const ScalarType value_dtype,
    const bool has_attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa) {
  return match_transformer_gqa_sdpa_contract(
             query_sizes,
             key_sizes,
             value_sizes,
             query_dtype,
             key_dtype,
             value_dtype,
             has_attn_mask,
             dropout_p,
             is_causal,
             scale,
             enable_gqa)
      .matched;
}

const char* masked_tiny_sdpa_route_label(
    const MaskedTinySDPAFamily family) {
  switch (family) {
    case MaskedTinySDPAFamily::AdditiveFloatMask:
      return "SelectedMaskedTinySDPAAdditiveFloatMask";
    case MaskedTinySDPAFamily::None:
      return "SelectedMaskedTinySDPANone";
  }
  return "SelectedMaskedTinySDPANone";
}

MaskedTinySDPAMatch match_masked_tiny_sdpa_contract(
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const IntArrayRef value_sizes,
    const IntArrayRef attn_mask_sizes,
    const ScalarType query_dtype,
    const ScalarType key_dtype,
    const ScalarType value_dtype,
    const ScalarType attn_mask_dtype,
    const bool has_attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa) {
  MaskedTinySDPAMatch result;
  if (
      !has_attn_mask || dropout_p != 0.0 || is_causal || enable_gqa ||
      query_dtype != kFloat || key_dtype != kFloat || value_dtype != kFloat ||
      attn_mask_dtype != kFloat || query_sizes.size() != 4 ||
      key_sizes.size() != 4 || value_sizes.size() != 4 ||
      attn_mask_sizes.size() != 4) {
    return result;
  }
  if (scale.has_value() && std::abs(*scale - kHeadDim64Scale) > 1.0e-6) {
    return result;
  }
  if (
      query_sizes[0] == 1 && key_sizes[0] == 1 && value_sizes[0] == 1 &&
      query_sizes[1] == 16 && key_sizes[1] == 16 && value_sizes[1] == 16 &&
      query_sizes[2] == 2 && key_sizes[2] == 2 && value_sizes[2] == 2 &&
      query_sizes[3] == 64 && key_sizes[3] == 64 && value_sizes[3] == 64 &&
      attn_mask_sizes[0] == 1 && attn_mask_sizes[1] == 1 &&
      attn_mask_sizes[2] == 2 && attn_mask_sizes[3] == 2) {
    result.matched = true;
    result.family = MaskedTinySDPAFamily::AdditiveFloatMask;
    result.tuple_id = "qkv_1x16x2x64_mask_1x1x2x2";
  }
  return result;
}

bool matches_masked_tiny_sdpa_contract(
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const IntArrayRef value_sizes,
    const IntArrayRef attn_mask_sizes,
    const ScalarType query_dtype,
    const ScalarType key_dtype,
    const ScalarType value_dtype,
    const ScalarType attn_mask_dtype,
    const bool has_attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa) {
  return match_masked_tiny_sdpa_contract(
             query_sizes,
             key_sizes,
             value_sizes,
             attn_mask_sizes,
             query_dtype,
             key_dtype,
             value_dtype,
             attn_mask_dtype,
             has_attn_mask,
             dropout_p,
             is_causal,
             scale,
             enable_gqa)
      .matched;
}

const char* diffusion_sdpa_route_label(const DiffusionSDPAFamily family) {
  switch (family) {
    case DiffusionSDPAFamily::SquareSelfAttention:
      return "SelectedDiffusionSDPASquareSelfAttention";
    case DiffusionSDPAFamily::CrossAttention:
      return "SelectedDiffusionSDPACrossAttention";
    case DiffusionSDPAFamily::None:
      return "SelectedDiffusionSDPANone";
  }
  return "SelectedDiffusionSDPANone";
}

DiffusionSDPAMatch match_diffusion_sdpa_contract(
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const IntArrayRef value_sizes,
    const ScalarType query_dtype,
    const ScalarType key_dtype,
    const ScalarType value_dtype,
    const bool has_attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa) {
  DiffusionSDPAMatch result;
  if (
      has_attn_mask || dropout_p != 0.0 || is_causal || enable_gqa ||
      query_dtype != kFloat || key_dtype != kFloat || value_dtype != kFloat ||
      query_sizes.size() != 4 || key_sizes.size() != 4 ||
      value_sizes.size() != 4) {
    return result;
  }
  if (
      query_sizes[0] != 1 || key_sizes[0] != 1 || value_sizes[0] != 1 ||
      query_sizes[1] != key_sizes[1] || query_sizes[1] != value_sizes[1] ||
      key_sizes[2] != value_sizes[2] || query_sizes[3] != key_sizes[3] ||
      query_sizes[3] != value_sizes[3]) {
    return result;
  }

  const int64_t heads = query_sizes[1];
  const int64_t query_sequence = query_sizes[2];
  const int64_t key_value_sequence = key_sizes[2];
  const int64_t head_dim = query_sizes[3];
  for (const DiffusionSDPATuple& tuple : kDiffusionSDPATuples) {
    if (
        heads != tuple.heads ||
        query_sequence != tuple.query_sequence ||
        key_value_sequence != tuple.key_value_sequence ||
        head_dim != tuple.head_dim) {
      continue;
    }
    if (scale.has_value()) {
      const double expected_scale =
          head_dim == 512 ? kHeadDim512Scale : kHeadDim64Scale;
      if (std::abs(*scale - expected_scale) > 1.0e-6) {
        return result;
      }
    }
    result.matched = true;
    result.family = tuple.family;
    result.tuple_id = tuple.tuple_id;
    return result;
  }
  return result;
}

bool matches_diffusion_sdpa_contract(
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const IntArrayRef value_sizes,
    const ScalarType query_dtype,
    const ScalarType key_dtype,
    const ScalarType value_dtype,
    const bool has_attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa) {
  return match_diffusion_sdpa_contract(
             query_sizes,
             key_sizes,
             value_sizes,
             query_dtype,
             key_dtype,
             value_dtype,
             has_attn_mask,
             dropout_p,
             is_causal,
             scale,
             enable_gqa)
      .matched;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
