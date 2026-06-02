#pragma once

#ifdef USE_VULKAN_API

#include <ATen/ArrayRef.h>
#include <ATen/core/ScalarType.h>

#include <cstdint>
#include <optional>

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

enum class TransformerGQASDPAFamily : uint8_t {
  None = 0u,
  CausalPrefill,
  SmallNonCausalGQA,
  DecodeGQA,
};

struct TransformerGQASDPAMatch final {
  bool matched{false};
  TransformerGQASDPAFamily family{TransformerGQASDPAFamily::None};
  const char* tuple_id{nullptr};
};

enum class MaskedTinySDPAFamily : uint8_t {
  None = 0u,
  AdditiveFloatMask,
};

struct MaskedTinySDPAMatch final {
  bool matched{false};
  MaskedTinySDPAFamily family{MaskedTinySDPAFamily::None};
  const char* tuple_id{nullptr};
};

enum class DiffusionSDPAFamily : uint8_t {
  None = 0u,
  SquareSelfAttention,
  CrossAttention,
};

struct DiffusionSDPAMatch final {
  bool matched{false};
  DiffusionSDPAFamily family{DiffusionSDPAFamily::None};
  const char* tuple_id{nullptr};
};

enum class KVCacheAppendFamily : uint8_t {
  None = 0u,
  InitialCache,
  SequenceAppend,
};

struct KVCacheAppendMatch final {
  bool matched{false};
  KVCacheAppendFamily family{KVCacheAppendFamily::None};
  const char* tuple_id{nullptr};
  int64_t sequence_length{0};
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

const char* transformer_gqa_sdpa_family_name(
    TransformerGQASDPAFamily family);

const char* transformer_gqa_sdpa_route_label(
    TransformerGQASDPAFamily family);

TransformerGQASDPAMatch match_transformer_gqa_sdpa_contract(
    IntArrayRef query_sizes,
    IntArrayRef key_sizes,
    IntArrayRef value_sizes,
    ScalarType query_dtype,
    ScalarType key_dtype,
    ScalarType value_dtype,
    bool has_attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa);

bool matches_transformer_gqa_sdpa_contract(
    IntArrayRef query_sizes,
    IntArrayRef key_sizes,
    IntArrayRef value_sizes,
    ScalarType query_dtype,
    ScalarType key_dtype,
    ScalarType value_dtype,
    bool has_attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa);

const char* masked_tiny_sdpa_route_label(MaskedTinySDPAFamily family);

MaskedTinySDPAMatch match_masked_tiny_sdpa_contract(
    IntArrayRef query_sizes,
    IntArrayRef key_sizes,
    IntArrayRef value_sizes,
    IntArrayRef attn_mask_sizes,
    ScalarType query_dtype,
    ScalarType key_dtype,
    ScalarType value_dtype,
    ScalarType attn_mask_dtype,
    bool has_attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa);

bool matches_masked_tiny_sdpa_contract(
    IntArrayRef query_sizes,
    IntArrayRef key_sizes,
    IntArrayRef value_sizes,
    IntArrayRef attn_mask_sizes,
    ScalarType query_dtype,
    ScalarType key_dtype,
    ScalarType value_dtype,
    ScalarType attn_mask_dtype,
    bool has_attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa);

const char* diffusion_sdpa_route_label(DiffusionSDPAFamily family);

DiffusionSDPAMatch match_diffusion_sdpa_contract(
    IntArrayRef query_sizes,
    IntArrayRef key_sizes,
    IntArrayRef value_sizes,
    ScalarType query_dtype,
    ScalarType key_dtype,
    ScalarType value_dtype,
    bool has_attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa);

bool matches_diffusion_sdpa_contract(
    IntArrayRef query_sizes,
    IntArrayRef key_sizes,
    IntArrayRef value_sizes,
    ScalarType query_dtype,
    ScalarType key_dtype,
    ScalarType value_dtype,
    bool has_attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa);

const char* kv_cache_append_family_name(KVCacheAppendFamily family);

const char* kv_cache_append_op_hit_label(KVCacheAppendFamily family);

KVCacheAppendMatch match_kv_cache_append_contract(
    IntArrayRef left_sizes,
    IntArrayRef right_sizes,
    ScalarType left_dtype,
    ScalarType right_dtype,
    bool left_is_vulkan,
    bool right_is_vulkan,
    int64_t dim);

bool matches_kv_cache_append_contract(
    IntArrayRef left_sizes,
    IntArrayRef right_sizes,
    ScalarType left_dtype,
    ScalarType right_dtype,
    bool left_is_vulkan,
    bool right_is_vulkan,
    int64_t dim);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
