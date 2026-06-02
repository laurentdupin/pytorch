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

struct ExecutionContractMetadata final {
  const char* contract_name{nullptr};
  const char* family_name{nullptr};
  const char* tuple_id{nullptr};
  const char* evidence_id{nullptr};
  const char* guard_id{nullptr};
  const char* fallback_policy{nullptr};
  const char* materialization_policy{nullptr};
};

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
  const ExecutionContractMetadata* metadata{nullptr};
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
  const ExecutionContractMetadata* metadata{nullptr};
};

enum class MaskedTinySDPAFamily : uint8_t {
  None = 0u,
  AdditiveFloatMask,
};

struct MaskedTinySDPAMatch final {
  bool matched{false};
  MaskedTinySDPAFamily family{MaskedTinySDPAFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
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
  const ExecutionContractMetadata* metadata{nullptr};
};

enum class SDPAExecutionPolicyFamily : uint8_t {
  None = 0u,
  DiffusionMaterializedSquare,
  DiffusionCloneOnlySquare,
  TransformerDecodeGQACloneOnly,
};

struct SDPAExecutionPolicyMatch final {
  bool matched{false};
  SDPAExecutionPolicyFamily family{SDPAExecutionPolicyFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
  bool requires_materialized_math_path{false};
  bool requires_score_pre_materialization{false};
  bool requires_post_softmax_clone{false};
};

struct GQARepeatMatch final {
  bool matched{false};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
  int64_t sequence_length{0};
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
  const ExecutionContractMetadata* metadata{nullptr};
  int64_t sequence_length{0};
};

enum class EmbeddingLookupFamily : uint8_t {
  None = 0u,
  SmallBoundedLookup,
  TokenBatch1,
};

struct EmbeddingLookupMatch final {
  bool matched{false};
  EmbeddingLookupFamily family{EmbeddingLookupFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
  int64_t num_embeddings{0};
  int64_t embedding_dim{0};
  int64_t num_indices{0};
};

struct BatchNormInferenceTensorInfo final {
  bool has_value{false};
  bool defined{false};
  bool is_vulkan{false};
  ScalarType dtype{kFloat};
  int64_t dim{0};
  int64_t channels{0};
  int64_t numel{0};
  bool is_contiguous{false};
  bool has_buffer_storage{false};
};

enum class BatchNormInferenceFamily : uint8_t {
  None = 0u,
  BufferFloat4D,
};

struct BatchNormInferenceMatch final {
  bool matched{false};
  BatchNormInferenceFamily family{BatchNormInferenceFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
};

enum class SafeViewReshapeFamily : uint8_t {
  None = 0u,
  ViewMaterializedDirectBuffer,
  ReshapeAliasDenseBufferDirect,
};

struct SafeViewReshapeMatch final {
  bool matched{false};
  SafeViewReshapeFamily family{SafeViewReshapeFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
};

bool has_complete_execution_contract_metadata(
    const ExecutionContractMetadata* metadata);

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

const char* sdpa_execution_policy_family_name(
    SDPAExecutionPolicyFamily family);

SDPAExecutionPolicyMatch match_sdpa_execution_policy_contract(
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

bool matches_sdpa_buffer_softmax_score_contract(
    IntArrayRef input_sizes,
    ScalarType input_dtype,
    int64_t dim);

GQARepeatMatch match_gqa_repeat_contract(
    IntArrayRef tensor_sizes,
    ScalarType tensor_dtype,
    bool tensor_is_vulkan,
    bool tensor_has_buffer_storage,
    int64_t repeat_factor);

bool matches_gqa_repeat_contract(
    IntArrayRef tensor_sizes,
    ScalarType tensor_dtype,
    bool tensor_is_vulkan,
    bool tensor_has_buffer_storage,
    int64_t repeat_factor);

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

const char* embedding_lookup_family_name(EmbeddingLookupFamily family);

const char* embedding_lookup_write_label(EmbeddingLookupFamily family);

EmbeddingLookupMatch match_embedding_lookup_contract(
    IntArrayRef weight_sizes,
    IntArrayRef indices_sizes,
    ScalarType weight_dtype,
    ScalarType indices_dtype,
    bool weight_is_vulkan,
    bool indices_is_vulkan,
    bool padding_idx_has_hint,
    bool scale_grad_by_freq,
    bool sparse);

bool matches_embedding_lookup_contract(
    IntArrayRef weight_sizes,
    IntArrayRef indices_sizes,
    ScalarType weight_dtype,
    ScalarType indices_dtype,
    bool weight_is_vulkan,
    bool indices_is_vulkan,
    bool padding_idx_has_hint,
    bool scale_grad_by_freq,
    bool sparse);

const char* batch_norm_inference_family_name(
    BatchNormInferenceFamily family);

BatchNormInferenceMatch match_batch_norm_inference_contract(
    const BatchNormInferenceTensorInfo& input,
    const BatchNormInferenceTensorInfo& weight,
    const BatchNormInferenceTensorInfo& bias,
    const BatchNormInferenceTensorInfo& running_mean,
    const BatchNormInferenceTensorInfo& running_var,
    bool training);

bool matches_batch_norm_inference_contract(
    const BatchNormInferenceTensorInfo& input,
    const BatchNormInferenceTensorInfo& weight,
    const BatchNormInferenceTensorInfo& bias,
    const BatchNormInferenceTensorInfo& running_mean,
    const BatchNormInferenceTensorInfo& running_var,
    bool training);

const char* safe_view_reshape_family_name(SafeViewReshapeFamily family);

SafeViewReshapeMatch match_safe_view_reshape_contract(
    IntArrayRef input_sizes,
    IntArrayRef input_logical_strides,
    IntArrayRef output_sizes,
    IntArrayRef output_strides,
    bool input_is_float,
    bool input_has_buffer_storage,
    int64_t storage_offset);

bool matches_safe_view_reshape_contract(
    IntArrayRef input_sizes,
    IntArrayRef input_logical_strides,
    IntArrayRef output_sizes,
    IntArrayRef output_strides,
    bool input_is_float,
    bool input_has_buffer_storage,
    int64_t storage_offset);

SafeViewReshapeMatch match_safe_view_reshape_materialized_direct_buffer_contract(
    IntArrayRef input_sizes,
    IntArrayRef output_sizes,
    IntArrayRef output_strides,
    int64_t storage_offset);

bool matches_safe_view_reshape_materialized_direct_buffer_contract(
    IntArrayRef input_sizes,
    IntArrayRef output_sizes,
    IntArrayRef output_strides,
    int64_t storage_offset);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
