#pragma once

#ifdef USE_VULKAN_API

#include <ATen/ArrayRef.h>
#include <ATen/core/ScalarType.h>

#include <cstdint>
#include <optional>
#include <string_view>

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

enum class DynamicPointwiseConv1x1DirectBufferFamily : uint8_t {
  None = 0u,
  GenericDynamicHW,
};

struct DynamicPointwiseConv1x1DirectBufferMatch final {
  bool matched{false};
  DynamicPointwiseConv1x1DirectBufferFamily family{
      DynamicPointwiseConv1x1DirectBufferFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
};

struct SmallMetadataPaddedConv2DTensorInfo final {
  bool is_vulkan{false};
  ScalarType dtype{kFloat};
  int64_t rank{0};
  int64_t batch{0};
  int64_t channels{0};
  int64_t height{0};
  int64_t width{0};
  bool has_buffer_storage{false};
  bool is_width_packed{false};
  bool has_direct_buffer_layout{false};
  bool supports_buffer_compute{false};
};

struct SmallMetadataPaddedConv2DWeightInfo final {
  bool defined{false};
  ScalarType dtype{kFloat};
  int64_t rank{0};
  int64_t output_channels{0};
  int64_t input_channels{0};
  int64_t kernel_h{0};
  int64_t kernel_w{0};
};

struct SmallMetadataPaddedConv2DOptions final {
  bool transposed{false};
  bool quantized{false};
  int64_t groups{0};
  int64_t stride_h{0};
  int64_t stride_w{0};
  int64_t padding_h{0};
  int64_t padding_w{0};
  int64_t dilation_h{0};
  int64_t dilation_w{0};
  bool output_padding_is_zero{false};
};

enum class SmallMetadataPaddedConv2DFamily : uint8_t {
  None = 0u,
  MaterializedBufferInput2x2,
  RuntimeMaterializedBufferInput2x2,
};

struct SmallMetadataPaddedConv2DMatch final {
  bool matched{false};
  SmallMetadataPaddedConv2DFamily family{
      SmallMetadataPaddedConv2DFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
  bool requires_input_materialization{false};
};

struct NoOverlapConvTranspose2DTensorInfo final {
  bool is_vulkan{false};
  ScalarType dtype{kFloat};
  int64_t rank{0};
  int64_t batch{0};
  int64_t channels{0};
  bool has_buffer_storage{false};
  bool supports_buffer_compute{false};
};

struct NoOverlapConvTranspose2DPackedInfo final {
  bool defined{false};
  bool execution_is_buffer_direct{false};
  bool quantized{false};
  ScalarType weight_dtype{kFloat};
  int64_t weight_rank{0};
  int64_t input_channels{0};
  int64_t output_channels{0};
  int64_t kernel_h{0};
  int64_t kernel_w{0};
  bool weight_has_buffer_storage{false};
  bool bias_has_buffer_storage{false};
  bool bias_is_float{false};
};

struct NoOverlapConvTranspose2DOptions final {
  bool transposed{false};
  bool quantized{false};
  int64_t groups{0};
  int64_t stride_h{0};
  int64_t stride_w{0};
  int64_t padding_h{0};
  int64_t padding_w{0};
  int64_t dilation_h{0};
  int64_t dilation_w{0};
  bool output_padding_is_zero{false};
};

enum class NoOverlapConvTranspose2DFamily : uint8_t {
  None = 0u,
  Kernel2Stride2FloatBuffer,
  DynamicKernelStrideFloatBuffer,
};

struct NoOverlapConvTranspose2DMatch final {
  bool matched{false};
  NoOverlapConvTranspose2DFamily family{
      NoOverlapConvTranspose2DFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
};

enum class TransformerGQASDPAFamily : uint8_t {
  None = 0u,
  CausalPrefill,
  SmallNonCausalGQA,
  DecodeGQA,
  DynamicDirectDecodeGQA,
  DirectNonCausalMHA,
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
  AdditiveFloatMaskRuntimeShape,
  BooleanKeepMaskRuntimeShape,
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
  SquareSelfAttentionRuntimeShape,
  CrossAttention,
  CrossAttentionRuntimeShape,
};

struct DiffusionSDPAMatch final {
  bool matched{false};
  DiffusionSDPAFamily family{DiffusionSDPAFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
};

enum class VisionSelfAttentionSDPAFamily : uint8_t {
  None = 0u,
  Rank3Head64Scale1,
  Rank3Head64Scale1RuntimeShape,
};

struct VisionSelfAttentionSDPAMatch final {
  bool matched{false};
  VisionSelfAttentionSDPAFamily family{
      VisionSelfAttentionSDPAFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
};

enum class SDPAExecutionPolicyFamily : uint8_t {
  None = 0u,
  DiffusionMaterializedSquare,
  DiffusionMaterializedSquareRuntimeShape,
  DiffusionCloneOnlySquare,
  TransformerDecodeGQACloneOnly,
  TransformerDecodeGQACloneOnlyRuntimeShape,
  VisionSelfAttentionCloneOnly,
  RecognizerNonCausalMHACloneOnly,
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

enum class SDPAScoreSoftmaxFamily : uint8_t {
  None = 0u,
  DiffusionSquareScores,
  DiffusionSquareScoresRuntimeShape,
  RectangularScoresRuntimeShape,
  VisionSelfAttentionScores,
  VisionSelfAttentionScoresRuntimeShape,
};

struct SDPAScoreSoftmaxMatch final {
  bool matched{false};
  SDPAScoreSoftmaxFamily family{SDPAScoreSoftmaxFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
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
  InitialCacheDirectBuffer,
  SequenceAppend,
};

struct KVCacheAppendMatch final {
  bool matched{false};
  KVCacheAppendFamily family{KVCacheAppendFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
  int64_t sequence_length{0};
};

struct ChannelCatTensorInfo final {
  bool is_vulkan{false};
  ScalarType dtype{kFloat};
  int64_t rank{0};
  int64_t batch{0};
  int64_t channels{0};
  int64_t height{0};
  int64_t width{0};
  bool is_contiguous{false};
  bool has_buffer_storage{false};
  bool supports_buffer_compute{false};
};

enum class ChannelCatFamily : uint8_t {
  None = 0u,
  Rank4Dim1BufferView,
  GenericRank4Dim1DirectBuffer,
};

struct ChannelCatMatch final {
  bool matched{false};
  ChannelCatFamily family{ChannelCatFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
  int64_t input_count{0};
  int64_t total_channels{0};
};

enum class TokenPrefixCatAddFamily : uint8_t {
  None = 0u,
  Prefix1TokenCountSetFeatureSetAdd,
  GenericPrefix1Dim1BufferAdd,
};

struct TokenPrefixCatAddMatch final {
  bool matched{false};
  TokenPrefixCatAddFamily family{TokenPrefixCatAddFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
  int64_t token_count{0};
  int64_t feature_dim{0};
  int64_t total_tokens{0};
};

enum class PatchEmbedFeatureMapToTokensFamily : uint8_t {
  None = 0u,
  Kernel14Stride14ObservedFeatureMap,
  GenericDirectBuffer,
};

struct PatchEmbedFeatureMapToTokensMatch final {
  bool matched{false};
  PatchEmbedFeatureMapToTokensFamily family{
      PatchEmbedFeatureMapToTokensFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
  int64_t channels{0};
  int64_t feature_h{0};
  int64_t feature_w{0};
  int64_t token_count{0};
};

struct LinearGeluBridgeTensorInfo final {
  int64_t input_rank{0};
  int64_t input_batch{0};
  int64_t input_rows{0};
  int64_t input_features{0};
  int64_t flattened_rank{0};
  int64_t flattened_rows{0};
  int64_t flattened_features{0};
};

struct LinearGeluBridgePackedInfo final {
  int64_t weight_height{0};
  int64_t weight_width{0};
  bool bias_defined{false};
  bool can_run_float_buffer_linear{false};
};

struct LinearGeluBridgeOptions final {
  bool inference_mode_enabled{false};
  bool has_output{false};
  bool post_op_is_none{false};
  bool alpha_is_one{false};
  bool beta_is_one{false};
};

enum class LinearGeluBridgeFamily : uint8_t {
  None = 0u,
  BackboneMlpHidden384To1536,
  GenericRuntimeShape,
};

struct LinearGeluBridgeMatch final {
  bool matched{false};
  LinearGeluBridgeFamily family{LinearGeluBridgeFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
  bool may_defer{false};
  bool may_consume_gelu_none{false};
  bool may_consume_gelu_tanh{false};
};

struct BatchNormInferenceTensorInfo final {
  bool has_value{false};
  bool defined{false};
  bool is_vulkan{false};
  ScalarType dtype{kFloat};
  int64_t dim{0};
  int64_t batch{0};
  int64_t channels{0};
  int64_t height{0};
  int64_t width{0};
  int64_t numel{0};
  bool is_contiguous{false};
  bool has_buffer_storage{false};
  bool supports_buffer_compute{false};
};

enum class BatchNormInferenceFamily : uint8_t {
  None = 0u,
  BufferFloat4D,
  MaterializedBufferFloat4D,
};

struct BatchNormInferenceMatch final {
  bool matched{false};
  BatchNormInferenceFamily family{BatchNormInferenceFamily::None};
  const char* tuple_id{nullptr};
  const ExecutionContractMetadata* metadata{nullptr};
  bool requires_materialization{false};
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

enum class ElementwiseBroadcastOp : uint8_t {
  Unsupported = 0u,
  Add,
  Mul,
  Sub,
};

enum class ElementwiseBroadcastFamily : uint8_t {
  None = 0u,
  FloatTensorTensorBufferBroadcast,
};

struct ElementwiseBroadcastMatch final {
  bool matched{false};
  ElementwiseBroadcastFamily family{ElementwiseBroadcastFamily::None};
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

const char* dynamic_pointwise_conv1x1_direct_buffer_family_name(
    DynamicPointwiseConv1x1DirectBufferFamily family);

const char* dynamic_pointwise_conv1x1_direct_buffer_route_label(
    DynamicPointwiseConv1x1DirectBufferFamily family);

const char* dynamic_pointwise_conv1x1_direct_buffer_op_hit_label(
    DynamicPointwiseConv1x1DirectBufferFamily family);

DynamicPointwiseConv1x1DirectBufferMatch
match_dynamic_pointwise_conv1x1_direct_buffer_contract(
    IntArrayRef input_sizes,
    IntArrayRef weight_sizes,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    ScalarType dtype);

const char* small_metadata_padded_conv2d_family_name(
    SmallMetadataPaddedConv2DFamily family);

SmallMetadataPaddedConv2DMatch match_small_metadata_padded_conv2d_contract(
    const SmallMetadataPaddedConv2DTensorInfo& input,
    const SmallMetadataPaddedConv2DWeightInfo& weight,
    const SmallMetadataPaddedConv2DOptions& options);

bool matches_small_metadata_padded_conv2d_contract(
    const SmallMetadataPaddedConv2DTensorInfo& input,
    const SmallMetadataPaddedConv2DWeightInfo& weight,
    const SmallMetadataPaddedConv2DOptions& options);

const char* no_overlap_conv_transpose2d_family_name(
    NoOverlapConvTranspose2DFamily family);

NoOverlapConvTranspose2DMatch match_no_overlap_conv_transpose2d_contract(
    const NoOverlapConvTranspose2DTensorInfo& input,
    const NoOverlapConvTranspose2DPackedInfo& packed,
    const NoOverlapConvTranspose2DOptions& options);

bool matches_no_overlap_conv_transpose2d_contract(
    const NoOverlapConvTranspose2DTensorInfo& input,
    const NoOverlapConvTranspose2DPackedInfo& packed,
    const NoOverlapConvTranspose2DOptions& options);

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

const char* vision_self_attention_sdpa_route_label(
    VisionSelfAttentionSDPAFamily family);

VisionSelfAttentionSDPAMatch match_vision_self_attention_sdpa_contract(
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

bool matches_vision_self_attention_sdpa_contract(
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

SDPAScoreSoftmaxMatch match_sdpa_buffer_softmax_score_contract(
    IntArrayRef input_sizes,
    ScalarType input_dtype,
    int64_t dim);

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

const ExecutionContractMetadata* kv_cache_initial_dynamic_metadata();

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

const char* channel_cat_family_name(ChannelCatFamily family);

const char* channel_cat_op_hit_label(ChannelCatFamily family);

ChannelCatMatch match_channel_cat_contract(
    ArrayRef<ChannelCatTensorInfo> tensors,
    int64_t dim);

bool matches_channel_cat_contract(
    ArrayRef<ChannelCatTensorInfo> tensors,
    int64_t dim);

const char* token_prefix_cat_add_family_name(TokenPrefixCatAddFamily family);

TokenPrefixCatAddMatch match_token_prefix_cat_add_contract(
    IntArrayRef prefix_sizes,
    IntArrayRef token_sizes,
    IntArrayRef pos_sizes,
    ScalarType prefix_dtype,
    ScalarType token_dtype,
    ScalarType pos_dtype,
    bool prefix_is_vulkan,
    bool tokens_is_vulkan,
    bool pos_is_vulkan,
    int64_t dim,
    bool inplace,
    bool alias_output);

bool matches_token_prefix_cat_add_contract(
    IntArrayRef prefix_sizes,
    IntArrayRef token_sizes,
    IntArrayRef pos_sizes,
    ScalarType prefix_dtype,
    ScalarType token_dtype,
    ScalarType pos_dtype,
    bool prefix_is_vulkan,
    bool tokens_is_vulkan,
    bool pos_is_vulkan,
    int64_t dim,
    bool inplace,
    bool alias_output);

const char* patch_embed_feature_map_to_tokens_family_name(
    PatchEmbedFeatureMapToTokensFamily family);

PatchEmbedFeatureMapToTokensMatch match_patch_embed_feature_map_to_tokens_contract(
    IntArrayRef feature_map_sizes,
    ScalarType dtype,
    bool is_vulkan,
    bool has_buffer_storage,
    bool is_width_packed,
    bool has_zero_storage_offset,
    bool supports_buffer_compute);

bool matches_patch_embed_feature_map_to_tokens_contract(
    IntArrayRef feature_map_sizes,
    ScalarType dtype,
    bool is_vulkan,
    bool has_buffer_storage,
    bool is_width_packed,
    bool has_zero_storage_offset,
    bool supports_buffer_compute);

const char* linear_gelu_bridge_family_name(LinearGeluBridgeFamily family);

LinearGeluBridgeMatch match_linear_gelu_bridge_contract(
    const LinearGeluBridgeTensorInfo& tensor,
    const LinearGeluBridgePackedInfo& packed,
    const LinearGeluBridgeOptions& options);

bool matches_linear_gelu_bridge_contract(
    const LinearGeluBridgeTensorInfo& tensor,
    const LinearGeluBridgePackedInfo& packed,
    const LinearGeluBridgeOptions& options);

bool matches_linear_gelu_bridge_gelu_approximation_contract(
    std::string_view approximate);

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

const char* elementwise_broadcast_family_name(
    ElementwiseBroadcastFamily family);

ElementwiseBroadcastMatch match_elementwise_broadcast_contract(
    IntArrayRef self_sizes,
    IntArrayRef other_sizes,
    ScalarType self_dtype,
    ScalarType other_dtype,
    ScalarType output_dtype,
    bool self_is_vulkan,
    bool other_is_vulkan,
    bool self_supports_buffer_compute,
    bool other_supports_buffer_compute,
    bool buffer_route_selected,
    ElementwiseBroadcastOp op,
    bool alpha_is_one,
    bool has_output,
    bool inplace);

bool matches_elementwise_broadcast_contract(
    IntArrayRef self_sizes,
    IntArrayRef other_sizes,
    ScalarType self_dtype,
    ScalarType other_dtype,
    ScalarType output_dtype,
    bool self_is_vulkan,
    bool other_is_vulkan,
    bool self_supports_buffer_compute,
    bool other_supports_buffer_compute,
    bool buffer_route_selected,
    ElementwiseBroadcastOp op,
    bool alpha_is_one,
    bool has_output,
    bool inplace);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
