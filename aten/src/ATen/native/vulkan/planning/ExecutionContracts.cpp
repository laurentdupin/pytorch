#include <ATen/native/vulkan/planning/ExecutionContracts.h>

#include <c10/util/Exception.h>
#include <c10/util/strides.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

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

struct DiffusionSDPATuple final {
  DiffusionSDPAFamily family;
  int64_t heads;
  int64_t query_sequence;
  int64_t key_value_sequence;
  int64_t head_dim;
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

bool has_text(const char* value) {
  return value != nullptr && value[0] != '\0';
}

constexpr const char* kFallbackUnsupportedShapesDoNotMatch =
    "unsupported_shapes_do_not_match";
constexpr const char* kMaterializationNone = "none";
constexpr const char* kMaterializationNativeBufferKernel =
    "native_buffer_kernel";
constexpr const char* kMaterializationSmallMetadataPaddedConv2DInput =
    "materialize_small_metadata_input_then_conv2d_buffer_float";
constexpr const char* kMaterializationConvTransposeNoOverlapBuffer =
    "conv_transpose2d_no_overlap_buffer_kernel";
constexpr const char* kMaterializationDelegatedToSDPAExecutionPolicy =
    "delegated_to_sdpa_execution_policy";
constexpr const char* kMaterializationScorePreMaterializeAndPostSoftmaxClone =
    "score_pre_materialization_and_post_softmax_clone";
constexpr const char* kMaterializationMaterializedMathAndPostSoftmaxClone =
    "materialized_math_path_and_post_softmax_clone";
constexpr const char* kMaterializationPostSoftmaxClone =
    "post_softmax_clone";
constexpr const char* kMaterializationGQARepeatBuffer =
    "gqa_repeat_buffer_materialization";
constexpr const char* kMaterializationKVCacheAppendBuffer =
    "kv_cache_append_buffer_kernel";
constexpr const char* kMaterializationChannelCatBufferView =
    "channel_cat_buffer_view_copy_kernel";
constexpr const char* kMaterializationEmbeddingLookupBuffer =
    "embedding_lookup_buffer_kernel";
constexpr const char* kMaterializationBatchNormInferenceBuffer =
    "batch_norm_inference_buffer_kernel";
constexpr const char* kMaterializationBatchNormInferenceMaterializedBuffer =
    "materialize_to_buffer_then_batch_norm_inference_buffer_kernel";
constexpr const char* kMaterializationReshapeAliasDirectBuffer =
    "reshape_alias_materialized_direct_buffer";
constexpr const char* kMaterializationViewDirectBuffer =
    "view_materialized_direct_buffer";

constexpr int64_t kNoOverlapConvTranspose2DBatch = 1;
constexpr int64_t kNoOverlapConvTranspose2DMinInputChannels = 64;
constexpr int64_t kNoOverlapConvTranspose2DKernel = 2;
constexpr int64_t kNoOverlapConvTranspose2DStride = 2;
constexpr const char* kNoOverlapConvTranspose2DTupleId =
    "batch1_cin_ge64_kernel2_stride2_float_buffer";
constexpr ExecutionContractMetadata
    kNoOverlapConvTranspose2DKernel2Stride2FloatBufferMetadata =
        make_execution_contract_metadata(
            "NoOverlapConvTranspose2DContract",
            "Kernel2Stride2FloatBuffer",
            kNoOverlapConvTranspose2DTupleId,
            "conv_transpose2d_no_overlap_2x2_stride2_buffer_float",
            "conv_transpose2d_no_overlap_adjacent_guards",
            kFallbackUnsupportedShapesDoNotMatch,
            kMaterializationConvTransposeNoOverlapBuffer);

constexpr int64_t kSmallMetadataPaddedConv2DBatch = 1;
constexpr int64_t kSmallMetadataPaddedConv2DInputChannels = 16;
constexpr int64_t kSmallMetadataPaddedConv2DInputHeight = 721;
constexpr int64_t kSmallMetadataPaddedConv2DInputWidth = 1281;
constexpr int64_t kSmallMetadataPaddedConv2DOutputChannels = 32;
constexpr int64_t kSmallMetadataPaddedConv2DKernel = 2;
constexpr const char* kSmallMetadataPaddedConv2DTupleId =
    "input_1x16x721x1281_weight_32x16x2x2_stride1";
constexpr ExecutionContractMetadata
    kSmallMetadataPaddedConv2DMaterializedBufferInput2x2Metadata =
        make_execution_contract_metadata(
            "SmallMetadataPaddedConv2DContract",
            "MaterializedBufferInput2x2",
            kSmallMetadataPaddedConv2DTupleId,
            "task028_paddleocr_conv2d_pressure_classification",
            "small_metadata_padded_conv2d_adjacent_guards",
            kFallbackUnsupportedShapesDoNotMatch,
            kMaterializationSmallMetadataPaddedConv2DInput);

constexpr double kTransformerHeadDim128Scale = 0.08838834764831845;
constexpr double kHeadDim64Scale = 0.125;
constexpr double kHeadDim512Scale = 0.04419417382415922;

constexpr const char* kTransformerGQASDPACausalGQATupleId =
    "causal_gqa_head128_len_le_128";
constexpr const char* kTransformerGQASDPACausalMHATupleId =
    "causal_mha_head128_len_le_128";
constexpr const char* kTransformerGQASDPADecodeGQATupleId =
    "decode_gqa_head128_source_100_116";
constexpr const char* kTransformerGQASDPASmallNonCausalGQATupleId =
    "small_non_causal_gqa_head128";
constexpr ExecutionContractMetadata kTransformerGQASDPACausalGQAMetadata =
    make_execution_contract_metadata(
        "TransformerGQASDPAContract",
        "CausalPrefill",
        kTransformerGQASDPACausalGQATupleId,
        "transformer_gqa_sdpa_focused_tests",
        "transformer_gqa_sdpa_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationDelegatedToSDPAExecutionPolicy);
constexpr ExecutionContractMetadata kTransformerGQASDPACausalMHAMetadata =
    make_execution_contract_metadata(
        "TransformerGQASDPAContract",
        "CausalPrefill",
        kTransformerGQASDPACausalMHATupleId,
        "transformer_gqa_sdpa_focused_tests",
        "transformer_gqa_sdpa_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationDelegatedToSDPAExecutionPolicy);
constexpr ExecutionContractMetadata kTransformerGQASDPADecodeGQAMetadata =
    make_execution_contract_metadata(
        "TransformerGQASDPAContract",
        "DecodeGQA",
        kTransformerGQASDPADecodeGQATupleId,
        "transformer_gqa_sdpa_focused_tests",
        "transformer_gqa_sdpa_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationDelegatedToSDPAExecutionPolicy);
constexpr ExecutionContractMetadata
    kTransformerGQASDPASmallNonCausalGQAMetadata =
        make_execution_contract_metadata(
            "TransformerGQASDPAContract",
            "SmallNonCausalGQA",
            kTransformerGQASDPASmallNonCausalGQATupleId,
            "transformer_gqa_sdpa_focused_tests",
            "transformer_gqa_sdpa_adjacent_guards",
            kFallbackUnsupportedShapesDoNotMatch,
            kMaterializationDelegatedToSDPAExecutionPolicy);

constexpr const char* kMaskedTinySDPAAdditiveFloatMaskTupleId =
    "qkv_1x16x2x64_mask_1x1x2x2";
constexpr ExecutionContractMetadata kMaskedTinySDPAAdditiveFloatMaskMetadata =
    make_execution_contract_metadata(
        "MaskedTinySDPAContract",
        "AdditiveFloatMask",
        kMaskedTinySDPAAdditiveFloatMaskTupleId,
        "masked_tiny_sdpa_focused_tests",
        "masked_tiny_sdpa_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationNone);

constexpr int64_t kKVCacheAppendBatch = 1;
constexpr int64_t kKVCacheAppendHeads = 4;
constexpr int64_t kKVCacheAppendMinSequence = 99;
constexpr int64_t kKVCacheAppendMaxSequence = 116;
constexpr int64_t kKVCacheAppendMaxSourceSequence = 115;
constexpr int64_t kKVCacheAppendTokenSequence = 1;
constexpr int64_t kKVCacheAppendHeadDim = 128;
constexpr const char* kKVCacheAppendInitialTupleId =
    "initial_empty_s99_to_s116_heads4_dim128";
constexpr const char* kKVCacheAppendSequenceTupleId =
    "sequence_append_s99_to_s115_token1_heads4_dim128";
constexpr ExecutionContractMetadata kKVCacheAppendInitialMetadata =
    make_execution_contract_metadata(
        "KVCacheAppendContract",
        "InitialCache",
        kKVCacheAppendInitialTupleId,
        "kv_cache_append_focused_tests",
        "kv_cache_append_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationKVCacheAppendBuffer);
constexpr ExecutionContractMetadata kKVCacheAppendSequenceMetadata =
    make_execution_contract_metadata(
        "KVCacheAppendContract",
        "SequenceAppend",
        kKVCacheAppendSequenceTupleId,
        "kv_cache_append_focused_tests",
        "kv_cache_append_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationKVCacheAppendBuffer);

constexpr int64_t kChannelCatRank4Dim1MinInputs = 3;
constexpr int64_t kChannelCatRank4Dim1MaxInputs = 8;
constexpr int64_t kChannelCatRank4Dim1Batch = 1;
constexpr int64_t kChannelCatRank4Dim1MaxInputChannels = 256;
constexpr int64_t kChannelCatRank4Dim1MaxTotalChannels = 1024;
constexpr int64_t kChannelCatRank4Dim1MaxHeight = 128;
constexpr int64_t kChannelCatRank4Dim1MaxWidth = 128;
constexpr const char* kChannelCatRank4Dim1BufferViewTupleId =
    "rank4_dim1_inputs3_to_8_c_mult4_spatial_le128_total_c_le1024";
constexpr ExecutionContractMetadata kChannelCatRank4Dim1BufferViewMetadata =
    make_execution_contract_metadata(
        "ChannelCatContract",
        "Rank4Dim1BufferView",
        kChannelCatRank4Dim1BufferViewTupleId,
        "channel_cat_buffer_view_focused_tests",
        "channel_cat_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationChannelCatBufferView);

constexpr int64_t kEmbeddingLookupTokenNumEmbeddings = 120818;
constexpr int64_t kEmbeddingLookupTokenEmbeddingDim = 2048;
constexpr int64_t kEmbeddingLookupTokenBatch = 1;
constexpr int64_t kEmbeddingLookupTokenMinIndices = 1;
constexpr int64_t kEmbeddingLookupTokenMaxIndices = 116;
constexpr int64_t kEmbeddingLookupSmallMaxNumEmbeddings = 4096;
constexpr int64_t kEmbeddingLookupSmallMaxEmbeddingDim = 256;
constexpr int64_t kEmbeddingLookupSmallMaxNumIndices = 128;
constexpr const char* kEmbeddingLookupTokenBatch1TupleId =
    "token_batch1_vocab120818_dim2048_indices1_to_116";
constexpr const char* kEmbeddingLookupSmallBoundedTupleId =
    "small_bounded_vocab4096_dim256_indices128";
constexpr ExecutionContractMetadata kEmbeddingLookupTokenBatch1Metadata =
    make_execution_contract_metadata(
        "EmbeddingLookupContract",
        "TokenBatch1",
        kEmbeddingLookupTokenBatch1TupleId,
        "embedding_lookup_focused_tests",
        "embedding_lookup_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationEmbeddingLookupBuffer);
constexpr ExecutionContractMetadata kEmbeddingLookupSmallBoundedMetadata =
    make_execution_contract_metadata(
        "EmbeddingLookupContract",
        "SmallBoundedLookup",
        kEmbeddingLookupSmallBoundedTupleId,
        "embedding_lookup_focused_tests",
        "embedding_lookup_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationEmbeddingLookupBuffer);

constexpr const char* kBatchNormInferenceBufferFloat4DTupleId =
    "buffer_inference_4d_float";
constexpr ExecutionContractMetadata kBatchNormInferenceBufferFloat4DMetadata =
    make_execution_contract_metadata(
        "BatchNormInferenceContract",
        "BufferFloat4D",
        kBatchNormInferenceBufferFloat4DTupleId,
        "batch_norm_inference_focused_tests",
        "batch_norm_inference_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationBatchNormInferenceBuffer);
constexpr const char* kBatchNormInferenceMaterializedBufferFloat4DTupleId =
    "materialized_buffer_inference_4d_float";
constexpr ExecutionContractMetadata
    kBatchNormInferenceMaterializedBufferFloat4DMetadata =
        make_execution_contract_metadata(
            "BatchNormInferenceContract",
            "MaterializedBufferFloat4D",
            kBatchNormInferenceMaterializedBufferFloat4DTupleId,
            "batch_norm_inference_materialized_buffer_focused_tests",
            "batch_norm_inference_adjacent_guards",
            kFallbackUnsupportedShapesDoNotMatch,
            kMaterializationBatchNormInferenceMaterializedBuffer);

constexpr const char* kSafeViewReshapeAliasDenseBufferDirectTupleId =
    "materialized_direct_buffer_reshape";
constexpr const char* kSafeViewReshapeViewMaterializedDirectBufferTupleId =
    "materialized_direct_buffer_reshape";
constexpr ExecutionContractMetadata
    kSafeViewReshapeViewMaterializedDirectBufferMetadata =
        make_execution_contract_metadata(
            "SafeViewReshapeContract",
            "ViewMaterializedDirectBuffer",
            kSafeViewReshapeViewMaterializedDirectBufferTupleId,
            "view_direct_buffer_focused_tests",
            "view_direct_buffer_adjacent_guards",
            kFallbackUnsupportedShapesDoNotMatch,
            kMaterializationViewDirectBuffer);
constexpr ExecutionContractMetadata
    kSafeViewReshapeAliasDenseBufferDirectMetadata =
        make_execution_contract_metadata(
            "SafeViewReshapeContract",
            "ReshapeAliasDenseBufferDirect",
            kSafeViewReshapeAliasDenseBufferDirectTupleId,
            "reshape_alias_direct_buffer_focused_tests",
            "reshape_alias_direct_buffer_adjacent_guards",
            kFallbackUnsupportedShapesDoNotMatch,
            kMaterializationReshapeAliasDirectBuffer);

constexpr int64_t kGQARepeatBatch = 1;
constexpr int64_t kGQARepeatSourceHeads = 4;
constexpr int64_t kGQARepeatFactor = 4;
constexpr int64_t kGQARepeatMinSequence = 100;
constexpr int64_t kGQARepeatMaxSequence = 116;
constexpr int64_t kGQARepeatHeadDim = 128;
constexpr const char* kGQARepeatTupleId =
    "gqa_repeat_batch1_heads4_factor4_sequence100_to_116_dim128";
constexpr ExecutionContractMetadata kGQARepeatMetadata =
    make_execution_contract_metadata(
        "GQARepeatContract",
        "Batch1Heads4Factor4Sequence100To116Dim128",
        kGQARepeatTupleId,
        "gqa_repeat_focused_tests",
        "gqa_repeat_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationGQARepeatBuffer);

constexpr const char* kSDPAExecutionTransformerDecodeGQACloneOnlyTupleId =
    "transformer_decode_gqa_clone_only_head128_source100_to_116";
constexpr const char* kSDPAExecutionSquareHeads1Sequence640Dim512TupleId =
    "square_heads1_sequence640_dim512";
constexpr const char* kSDPAExecutionSquareHeads5Sequence640Dim64TupleId =
    "square_heads5_sequence640_dim64";
constexpr const char* kSDPAExecutionSquareHeads1Sequence504Dim512TupleId =
    "square_heads1_sequence504_dim512";
constexpr const char* kSDPAExecutionSquareHeads5Sequence504Dim64TupleId =
    "square_heads5_sequence504_dim64";
constexpr const char* kSDPAExecutionSquareHeads10Sequence126Dim64TupleId =
    "square_heads10_sequence126_dim64";
constexpr ExecutionContractMetadata
    kSDPAExecutionSquareHeads1Sequence640Dim512Metadata =
        make_execution_contract_metadata(
            "SDPAExecutionPolicyContract",
            "DiffusionMaterializedSquare",
            kSDPAExecutionSquareHeads1Sequence640Dim512TupleId,
            "sdpa_execution_policy_focused_tests",
            "sdpa_execution_policy_adjacent_guards",
            kFallbackUnsupportedShapesDoNotMatch,
            kMaterializationScorePreMaterializeAndPostSoftmaxClone);
constexpr ExecutionContractMetadata
    kSDPAExecutionSquareHeads5Sequence640Dim64Metadata =
        make_execution_contract_metadata(
            "SDPAExecutionPolicyContract",
            "DiffusionMaterializedSquare",
            kSDPAExecutionSquareHeads5Sequence640Dim64TupleId,
            "sdpa_execution_policy_focused_tests",
            "sdpa_execution_policy_adjacent_guards",
            kFallbackUnsupportedShapesDoNotMatch,
            kMaterializationScorePreMaterializeAndPostSoftmaxClone);
constexpr ExecutionContractMetadata
    kSDPAExecutionSquareHeads1Sequence504Dim512Metadata =
        make_execution_contract_metadata(
            "SDPAExecutionPolicyContract",
            "DiffusionMaterializedSquare",
            kSDPAExecutionSquareHeads1Sequence504Dim512TupleId,
            "sdpa_execution_policy_focused_tests",
            "sdpa_execution_policy_adjacent_guards",
            kFallbackUnsupportedShapesDoNotMatch,
            kMaterializationScorePreMaterializeAndPostSoftmaxClone);
constexpr ExecutionContractMetadata
    kSDPAExecutionSquareHeads5Sequence504Dim64Metadata =
        make_execution_contract_metadata(
            "SDPAExecutionPolicyContract",
            "DiffusionMaterializedSquare",
            kSDPAExecutionSquareHeads5Sequence504Dim64TupleId,
            "sdpa_execution_policy_focused_tests",
            "sdpa_execution_policy_adjacent_guards",
            kFallbackUnsupportedShapesDoNotMatch,
            kMaterializationScorePreMaterializeAndPostSoftmaxClone);
constexpr ExecutionContractMetadata
    kSDPAExecutionSquareHeads10Sequence126Dim64Metadata =
        make_execution_contract_metadata(
            "SDPAExecutionPolicyContract",
            "DiffusionCloneOnlySquare",
            kSDPAExecutionSquareHeads10Sequence126Dim64TupleId,
            "sdpa_execution_policy_focused_tests",
            "sdpa_execution_policy_adjacent_guards",
            kFallbackUnsupportedShapesDoNotMatch,
            kMaterializationMaterializedMathAndPostSoftmaxClone);
constexpr ExecutionContractMetadata
    kSDPAExecutionTransformerDecodeGQACloneOnlyMetadata =
        make_execution_contract_metadata(
            "SDPAExecutionPolicyContract",
            "TransformerDecodeGQACloneOnly",
            kSDPAExecutionTransformerDecodeGQACloneOnlyTupleId,
            "sdpa_execution_policy_focused_tests",
            "sdpa_execution_policy_adjacent_guards",
            kFallbackUnsupportedShapesDoNotMatch,
            kMaterializationPostSoftmaxClone);

const ExecutionContractMetadata* sdpa_execution_policy_metadata(
    const SDPAExecutionPolicyFamily family,
    const char* tuple_id) {
  if (tuple_id == nullptr) {
    return nullptr;
  }
  if (family == SDPAExecutionPolicyFamily::DiffusionMaterializedSquare) {
    if (std::strcmp(
            tuple_id,
            kSDPAExecutionSquareHeads1Sequence640Dim512TupleId) == 0) {
      return &kSDPAExecutionSquareHeads1Sequence640Dim512Metadata;
    }
    if (std::strcmp(
            tuple_id, kSDPAExecutionSquareHeads5Sequence640Dim64TupleId) ==
        0) {
      return &kSDPAExecutionSquareHeads5Sequence640Dim64Metadata;
    }
    if (std::strcmp(
            tuple_id,
            kSDPAExecutionSquareHeads1Sequence504Dim512TupleId) == 0) {
      return &kSDPAExecutionSquareHeads1Sequence504Dim512Metadata;
    }
    if (std::strcmp(
            tuple_id, kSDPAExecutionSquareHeads5Sequence504Dim64TupleId) ==
        0) {
      return &kSDPAExecutionSquareHeads5Sequence504Dim64Metadata;
    }
  }
  if (
      family == SDPAExecutionPolicyFamily::DiffusionCloneOnlySquare &&
      std::strcmp(
          tuple_id, kSDPAExecutionSquareHeads10Sequence126Dim64TupleId) == 0) {
    return &kSDPAExecutionSquareHeads10Sequence126Dim64Metadata;
  }
  if (
      family == SDPAExecutionPolicyFamily::TransformerDecodeGQACloneOnly &&
      std::strcmp(
          tuple_id, kSDPAExecutionTransformerDecodeGQACloneOnlyTupleId) == 0) {
    return &kSDPAExecutionTransformerDecodeGQACloneOnlyMetadata;
  }
  return nullptr;
}

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

#define DIFFUSION_SDPA_TUPLE(                                      \
    FAMILY, HEADS, QUERY_SEQUENCE, KEY_VALUE_SEQUENCE, DIM, TUPLE_ID) \
  {                                                                \
      DiffusionSDPAFamily::FAMILY,                                 \
      HEADS,                                                       \
      QUERY_SEQUENCE,                                              \
      KEY_VALUE_SEQUENCE,                                          \
      DIM,                                                         \
      TUPLE_ID,                                                    \
      make_execution_contract_metadata(                            \
          "DiffusionSDPAContract",                                 \
          #FAMILY,                                                 \
          TUPLE_ID,                                                \
          "diffusion_sdpa_focused_tests",                          \
          "diffusion_sdpa_adjacent_guards",                        \
          kFallbackUnsupportedShapesDoNotMatch,                    \
          kMaterializationDelegatedToSDPAExecutionPolicy)}

constexpr DiffusionSDPATuple kDiffusionSDPATuples[] = {
    DIFFUSION_SDPA_TUPLE(
        SquareSelfAttention, 1, 640, 640, 512, "square_heads1_sequence640_dim512"),
    DIFFUSION_SDPA_TUPLE(
        SquareSelfAttention, 5, 640, 640, 64, "square_heads5_sequence640_dim64"),
    DIFFUSION_SDPA_TUPLE(
        SquareSelfAttention, 1, 504, 504, 512, "square_heads1_sequence504_dim512"),
    DIFFUSION_SDPA_TUPLE(
        SquareSelfAttention, 5, 504, 504, 64, "square_heads5_sequence504_dim64"),
    DIFFUSION_SDPA_TUPLE(
        SquareSelfAttention, 10, 126, 126, 64, "square_heads10_sequence126_dim64"),
    DIFFUSION_SDPA_TUPLE(
        SquareSelfAttention, 20, 35, 35, 64, "square_heads20_sequence35_dim64"),
    DIFFUSION_SDPA_TUPLE(
        SquareSelfAttention, 20, 12, 12, 64, "square_heads20_sequence12_dim64"),
    DIFFUSION_SDPA_TUPLE(
        CrossAttention, 5, 504, 2, 64, "cross_heads5_query504_kv2_dim64"),
    DIFFUSION_SDPA_TUPLE(
        CrossAttention, 10, 126, 2, 64, "cross_heads10_query126_kv2_dim64"),
    DIFFUSION_SDPA_TUPLE(
        CrossAttention, 20, 35, 2, 64, "cross_heads20_query35_kv2_dim64"),
    DIFFUSION_SDPA_TUPLE(
        CrossAttention, 20, 12, 2, 64, "cross_heads20_query12_kv2_dim64"),
};

#undef DIFFUSION_SDPA_TUPLE

bool matches_kv_cache_state_shape(const IntArrayRef sizes) {
  return sizes.size() == 4 && sizes[0] == kKVCacheAppendBatch &&
      sizes[1] == kKVCacheAppendHeads &&
      sizes[2] >= kKVCacheAppendMinSequence &&
      sizes[2] <= kKVCacheAppendMaxSequence &&
      sizes[3] == kKVCacheAppendHeadDim;
}

bool matches_kv_cache_token_shape(const IntArrayRef sizes) {
  return sizes.size() == 4 && sizes[0] == kKVCacheAppendBatch &&
      sizes[1] == kKVCacheAppendHeads &&
      sizes[2] == kKVCacheAppendTokenSequence &&
      sizes[3] == kKVCacheAppendHeadDim;
}

bool matches_empty_initial_cache_shape(const IntArrayRef sizes) {
  return sizes.size() == 1 && sizes[0] == 0;
}

bool is_contiguous_stride(
    const IntArrayRef sizes,
    const IntArrayRef strides) {
  return strides.equals(c10::contiguous_strides(sizes));
}

bool batch_norm_float_1d_matches(
    const BatchNormInferenceTensorInfo& tensor,
    const int64_t num_features) {
  return tensor.has_value && tensor.defined && tensor.is_vulkan &&
      tensor.dtype == kFloat && tensor.dim == 1 &&
      tensor.numel == num_features && tensor.is_contiguous;
}

bool batch_norm_float_1d_buffer_matches(
    const BatchNormInferenceTensorInfo& tensor,
    const int64_t num_features) {
  return batch_norm_float_1d_matches(tensor, num_features) &&
      tensor.has_buffer_storage;
}

bool batch_norm_float_1d_materializable_matches(
    const BatchNormInferenceTensorInfo& tensor,
    const int64_t num_features) {
  return batch_norm_float_1d_matches(tensor, num_features) &&
      tensor.supports_buffer_compute;
}

bool batch_norm_optional_float_1d_matches(
    const BatchNormInferenceTensorInfo& tensor,
    const int64_t num_features) {
  return !tensor.has_value ||
      batch_norm_float_1d_matches(tensor, num_features);
}

bool batch_norm_optional_float_1d_materializable_matches(
    const BatchNormInferenceTensorInfo& tensor,
    const int64_t num_features) {
  return !tensor.has_value ||
      batch_norm_float_1d_materializable_matches(tensor, num_features);
}

bool batch_norm_effective_affine_has_buffer_storage(
    const BatchNormInferenceTensorInfo& tensor,
    const BatchNormInferenceTensorInfo& running_mean) {
  return tensor.has_value ? tensor.has_buffer_storage
                          : running_mean.has_buffer_storage;
}

bool batch_norm_effective_affine_supports_buffer_compute(
    const BatchNormInferenceTensorInfo& tensor,
    const BatchNormInferenceTensorInfo& running_mean) {
  return tensor.has_value ? tensor.supports_buffer_compute
                          : running_mean.supports_buffer_compute;
}

bool is_non_overlapping_dense_stride(
    const IntArrayRef sizes,
    const IntArrayRef strides) {
  TORCH_INTERNAL_ASSERT(sizes.size() == strides.size());
  std::vector<size_t> dims;
  dims.reserve(sizes.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] > 1) {
      dims.push_back(i);
    }
  }
  std::sort(dims.begin(), dims.end(), [&](const size_t lhs, const size_t rhs) {
    return strides[lhs] < strides[rhs];
  });
  int64_t expected_stride = 1;
  for (const size_t dim : dims) {
    if (strides[dim] != expected_stride) {
      return false;
    }
    expected_stride *= sizes[dim];
  }
  return true;
}

int64_t product_of_sizes(const IntArrayRef sizes) {
  int64_t product = 1;
  for (const int64_t size : sizes) {
    product *= size;
  }
  return product;
}

} // namespace

bool has_complete_execution_contract_metadata(
    const ExecutionContractMetadata* metadata) {
  return metadata != nullptr && has_text(metadata->contract_name) &&
      has_text(metadata->family_name) && has_text(metadata->tuple_id) &&
      has_text(metadata->evidence_id) && has_text(metadata->guard_id) &&
      has_text(metadata->fallback_policy) &&
      has_text(metadata->materialization_policy);
}

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

const char* small_metadata_padded_conv2d_family_name(
    const SmallMetadataPaddedConv2DFamily family) {
  switch (family) {
    case SmallMetadataPaddedConv2DFamily::MaterializedBufferInput2x2:
      return "SmallMetadataPaddedConv2DMaterializedBufferInput2x2";
    case SmallMetadataPaddedConv2DFamily::None:
      return "SmallMetadataPaddedConv2DNone";
  }
  return "SmallMetadataPaddedConv2DNone";
}

SmallMetadataPaddedConv2DMatch match_small_metadata_padded_conv2d_contract(
    const SmallMetadataPaddedConv2DTensorInfo& input,
    const SmallMetadataPaddedConv2DWeightInfo& weight,
    const SmallMetadataPaddedConv2DOptions& options) {
  SmallMetadataPaddedConv2DMatch result;
  if (
      options.transposed || options.quantized || options.groups != 1 ||
      options.stride_h != 1 || options.stride_w != 1 ||
      options.padding_h != 0 || options.padding_w != 0 ||
      options.dilation_h != 1 || options.dilation_w != 1 ||
      !options.output_padding_is_zero || !input.is_vulkan ||
      input.dtype != kFloat || input.rank != 4 ||
      input.batch != kSmallMetadataPaddedConv2DBatch ||
      input.channels != kSmallMetadataPaddedConv2DInputChannels ||
      input.height != kSmallMetadataPaddedConv2DInputHeight ||
      input.width != kSmallMetadataPaddedConv2DInputWidth ||
      !input.has_buffer_storage || !input.is_width_packed ||
      input.has_direct_buffer_layout || !input.supports_buffer_compute ||
      !weight.defined || weight.dtype != kFloat || weight.rank != 4 ||
      weight.output_channels != kSmallMetadataPaddedConv2DOutputChannels ||
      weight.input_channels != kSmallMetadataPaddedConv2DInputChannels ||
      weight.kernel_h != kSmallMetadataPaddedConv2DKernel ||
      weight.kernel_w != kSmallMetadataPaddedConv2DKernel) {
    return result;
  }

  result.matched = true;
  result.family =
      SmallMetadataPaddedConv2DFamily::MaterializedBufferInput2x2;
  result.tuple_id = kSmallMetadataPaddedConv2DTupleId;
  result.metadata =
      &kSmallMetadataPaddedConv2DMaterializedBufferInput2x2Metadata;
  result.requires_input_materialization = true;
  return result;
}

bool matches_small_metadata_padded_conv2d_contract(
    const SmallMetadataPaddedConv2DTensorInfo& input,
    const SmallMetadataPaddedConv2DWeightInfo& weight,
    const SmallMetadataPaddedConv2DOptions& options) {
  return match_small_metadata_padded_conv2d_contract(input, weight, options)
      .matched;
}

const char* no_overlap_conv_transpose2d_family_name(
    const NoOverlapConvTranspose2DFamily family) {
  switch (family) {
    case NoOverlapConvTranspose2DFamily::Kernel2Stride2FloatBuffer:
      return "NoOverlapConvTranspose2DKernel2Stride2FloatBuffer";
    case NoOverlapConvTranspose2DFamily::None:
      return "NoOverlapConvTranspose2DNone";
  }
  return "NoOverlapConvTranspose2DNone";
}

NoOverlapConvTranspose2DMatch match_no_overlap_conv_transpose2d_contract(
    const NoOverlapConvTranspose2DTensorInfo& input,
    const NoOverlapConvTranspose2DPackedInfo& packed,
    const NoOverlapConvTranspose2DOptions& options) {
  NoOverlapConvTranspose2DMatch result;
  if (
      !options.transposed || options.quantized || options.groups != 1 ||
      options.stride_h != kNoOverlapConvTranspose2DStride ||
      options.stride_w != kNoOverlapConvTranspose2DStride ||
      options.padding_h != 0 || options.padding_w != 0 ||
      options.dilation_h != 1 || options.dilation_w != 1 ||
      !options.output_padding_is_zero || !input.is_vulkan ||
      input.dtype != kFloat || input.rank != 4 ||
      input.batch != kNoOverlapConvTranspose2DBatch ||
      input.channels < kNoOverlapConvTranspose2DMinInputChannels ||
      !input.has_buffer_storage || !input.supports_buffer_compute ||
      !packed.defined || !packed.execution_is_buffer_direct ||
      packed.quantized || packed.weight_dtype != kFloat ||
      packed.weight_rank != 4 ||
      packed.input_channels != input.channels ||
      packed.kernel_h != kNoOverlapConvTranspose2DKernel ||
      packed.kernel_w != kNoOverlapConvTranspose2DKernel ||
      !packed.weight_has_buffer_storage || !packed.bias_has_buffer_storage ||
      !packed.bias_is_float) {
    return result;
  }

  result.matched = true;
  result.family = NoOverlapConvTranspose2DFamily::Kernel2Stride2FloatBuffer;
  result.tuple_id = kNoOverlapConvTranspose2DTupleId;
  result.metadata = &kNoOverlapConvTranspose2DKernel2Stride2FloatBufferMetadata;
  return result;
}

bool matches_no_overlap_conv_transpose2d_contract(
    const NoOverlapConvTranspose2DTensorInfo& input,
    const NoOverlapConvTranspose2DPackedInfo& packed,
    const NoOverlapConvTranspose2DOptions& options) {
  return match_no_overlap_conv_transpose2d_contract(input, packed, options)
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
    result.tuple_id = enable_gqa ? kTransformerGQASDPACausalGQATupleId
                                 : kTransformerGQASDPACausalMHATupleId;
    result.metadata = enable_gqa ? &kTransformerGQASDPACausalGQAMetadata
                                 : &kTransformerGQASDPACausalMHAMetadata;
    return result;
  }

  if (
      enable_gqa && query_sizes[2] == 1 && key_sizes[2] >= 100 &&
      key_sizes[2] <= 116) {
    result.matched = true;
    result.family = TransformerGQASDPAFamily::DecodeGQA;
    result.tuple_id = kTransformerGQASDPADecodeGQATupleId;
    result.metadata = &kTransformerGQASDPADecodeGQAMetadata;
    return result;
  }

  if (query_sizes[2] <= 14 && key_sizes[2] <= 64) {
    result.matched = true;
    result.family = TransformerGQASDPAFamily::SmallNonCausalGQA;
    result.tuple_id = kTransformerGQASDPASmallNonCausalGQATupleId;
    result.metadata = &kTransformerGQASDPASmallNonCausalGQAMetadata;
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
    result.tuple_id = kMaskedTinySDPAAdditiveFloatMaskTupleId;
    result.metadata = &kMaskedTinySDPAAdditiveFloatMaskMetadata;
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
    result.metadata = &tuple.metadata;
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

const char* sdpa_execution_policy_family_name(
    const SDPAExecutionPolicyFamily family) {
  switch (family) {
    case SDPAExecutionPolicyFamily::DiffusionMaterializedSquare:
      return "SDPAExecutionDiffusionMaterializedSquare";
    case SDPAExecutionPolicyFamily::DiffusionCloneOnlySquare:
      return "SDPAExecutionDiffusionCloneOnlySquare";
    case SDPAExecutionPolicyFamily::TransformerDecodeGQACloneOnly:
      return "SDPAExecutionTransformerDecodeGQACloneOnly";
    case SDPAExecutionPolicyFamily::None:
      return "SDPAExecutionNone";
  }
  return "SDPAExecutionNone";
}

SDPAExecutionPolicyMatch match_sdpa_execution_policy_contract(
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
  SDPAExecutionPolicyMatch result;
  if (
      has_attn_mask || dropout_p != 0.0 ||
      query_dtype != kFloat || key_dtype != kFloat || value_dtype != kFloat ||
      query_sizes.size() != 4 || key_sizes.size() != 4 ||
      value_sizes.size() != 4 || query_sizes[0] != 1 ||
      key_sizes[0] != 1 || value_sizes[0] != 1 ||
      key_sizes[2] != value_sizes[2] || query_sizes[3] != key_sizes[3] ||
      query_sizes[3] != value_sizes[3]) {
    return result;
  }

  if (!is_causal && !enable_gqa) {
    const DiffusionSDPAMatch diffusion_match = match_diffusion_sdpa_contract(
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
        enable_gqa);
    if (
        diffusion_match.matched &&
        diffusion_match.family == DiffusionSDPAFamily::SquareSelfAttention) {
      if (
          query_sizes[1] == 1 &&
          (query_sizes[2] == 504 || query_sizes[2] == 640) &&
          query_sizes[3] == 512) {
        result.matched = true;
        result.family = SDPAExecutionPolicyFamily::DiffusionMaterializedSquare;
        result.tuple_id = diffusion_match.tuple_id;
        result.metadata =
            sdpa_execution_policy_metadata(result.family, result.tuple_id);
        result.requires_score_pre_materialization = true;
        result.requires_post_softmax_clone = true;
        return result;
      }
      if (
          query_sizes[1] == 5 &&
          (query_sizes[2] == 504 || query_sizes[2] == 640) &&
          query_sizes[3] == 64) {
        result.matched = true;
        result.family = SDPAExecutionPolicyFamily::DiffusionMaterializedSquare;
        result.tuple_id = diffusion_match.tuple_id;
        result.metadata =
            sdpa_execution_policy_metadata(result.family, result.tuple_id);
        result.requires_score_pre_materialization = true;
        result.requires_post_softmax_clone = true;
        return result;
      }
      if (
          query_sizes[1] == 10 && query_sizes[2] == 126 &&
          query_sizes[3] == 64) {
        result.matched = true;
        result.family = SDPAExecutionPolicyFamily::DiffusionCloneOnlySquare;
        result.tuple_id = diffusion_match.tuple_id;
        result.metadata =
            sdpa_execution_policy_metadata(result.family, result.tuple_id);
        result.requires_materialized_math_path = true;
        result.requires_post_softmax_clone = true;
        return result;
      }
    }
  }

  if (
      !is_causal && enable_gqa && query_sizes[1] == 16 &&
      key_sizes[1] == 16 && value_sizes[1] == 16 &&
      query_sizes[2] == 1 && key_sizes[2] >= 100 &&
      key_sizes[2] <= 116 && query_sizes[3] == 128) {
    result.matched = true;
    result.family = SDPAExecutionPolicyFamily::TransformerDecodeGQACloneOnly;
    result.tuple_id = kSDPAExecutionTransformerDecodeGQACloneOnlyTupleId;
    result.metadata =
        sdpa_execution_policy_metadata(result.family, result.tuple_id);
    result.requires_post_softmax_clone = true;
    return result;
  }

  return result;
}

bool matches_sdpa_buffer_softmax_score_contract(
    const IntArrayRef input_sizes,
    const ScalarType input_dtype,
    const int64_t dim) {
  if (
      input_dtype != kFloat || input_sizes.size() != 3 ||
      dim != static_cast<int64_t>(input_sizes.size()) - 1 ||
      input_sizes[1] != input_sizes[2]) {
    return false;
  }
  const int64_t heads = input_sizes[0];
  const int64_t sequence = input_sizes[1];
  return (heads == 1 && (sequence == 504 || sequence == 640)) ||
      (heads == 5 && (sequence == 504 || sequence == 640));
}

GQARepeatMatch match_gqa_repeat_contract(
    const IntArrayRef tensor_sizes,
    const ScalarType tensor_dtype,
    const bool tensor_is_vulkan,
    const bool tensor_has_buffer_storage,
    const int64_t repeat_factor) {
  GQARepeatMatch result;
  if (
      !tensor_is_vulkan || !tensor_has_buffer_storage ||
      tensor_dtype != kFloat || tensor_sizes.size() != 4 ||
      repeat_factor != kGQARepeatFactor ||
      tensor_sizes[0] != kGQARepeatBatch ||
      tensor_sizes[1] != kGQARepeatSourceHeads ||
      tensor_sizes[2] < kGQARepeatMinSequence ||
      tensor_sizes[2] > kGQARepeatMaxSequence ||
      tensor_sizes[3] != kGQARepeatHeadDim) {
    return result;
  }
  result.matched = true;
  result.tuple_id = kGQARepeatTupleId;
  result.metadata = &kGQARepeatMetadata;
  result.sequence_length = tensor_sizes[2];
  return result;
}

bool matches_gqa_repeat_contract(
    const IntArrayRef tensor_sizes,
    const ScalarType tensor_dtype,
    const bool tensor_is_vulkan,
    const bool tensor_has_buffer_storage,
    const int64_t repeat_factor) {
  return match_gqa_repeat_contract(
             tensor_sizes,
             tensor_dtype,
             tensor_is_vulkan,
             tensor_has_buffer_storage,
             repeat_factor)
      .matched;
}

const char* kv_cache_append_family_name(const KVCacheAppendFamily family) {
  switch (family) {
    case KVCacheAppendFamily::InitialCache:
      return "KVCacheAppendInitialCache";
    case KVCacheAppendFamily::SequenceAppend:
      return "KVCacheAppendSequenceAppend";
    case KVCacheAppendFamily::None:
      return "KVCacheAppendNone";
  }
  return "KVCacheAppendNone";
}

const char* kv_cache_append_op_hit_label(const KVCacheAppendFamily family) {
  switch (family) {
    case KVCacheAppendFamily::InitialCache:
      return "aten::cat.kv_cache_initial_dim2_buffer";
    case KVCacheAppendFamily::SequenceAppend:
      return "aten::cat.kv_cache_append_dim2_buffer";
    case KVCacheAppendFamily::None:
      return "aten::cat.kv_cache_append.none";
  }
  return "aten::cat.kv_cache_append.none";
}

KVCacheAppendMatch match_kv_cache_append_contract(
    const IntArrayRef left_sizes,
    const IntArrayRef right_sizes,
    const ScalarType left_dtype,
    const ScalarType right_dtype,
    const bool left_is_vulkan,
    const bool right_is_vulkan,
    const int64_t dim) {
  KVCacheAppendMatch result;
  if (!left_is_vulkan || !right_is_vulkan || dim != 2) {
    return result;
  }
  if (
      matches_empty_initial_cache_shape(left_sizes) &&
      right_dtype == kFloat && matches_kv_cache_state_shape(right_sizes)) {
    result.matched = true;
    result.family = KVCacheAppendFamily::InitialCache;
    result.tuple_id = kKVCacheAppendInitialTupleId;
    result.metadata = &kKVCacheAppendInitialMetadata;
    result.sequence_length = right_sizes[2];
    return result;
  }
  if (
      left_dtype == kFloat && right_dtype == kFloat &&
      matches_kv_cache_state_shape(left_sizes) &&
      matches_kv_cache_token_shape(right_sizes) &&
      left_sizes[2] <= kKVCacheAppendMaxSourceSequence) {
    result.matched = true;
    result.family = KVCacheAppendFamily::SequenceAppend;
    result.tuple_id = kKVCacheAppendSequenceTupleId;
    result.metadata = &kKVCacheAppendSequenceMetadata;
    result.sequence_length = left_sizes[2];
    return result;
  }
  return result;
}

bool matches_kv_cache_append_contract(
    const IntArrayRef left_sizes,
    const IntArrayRef right_sizes,
    const ScalarType left_dtype,
    const ScalarType right_dtype,
    const bool left_is_vulkan,
    const bool right_is_vulkan,
    const int64_t dim) {
  return match_kv_cache_append_contract(
             left_sizes,
             right_sizes,
             left_dtype,
             right_dtype,
             left_is_vulkan,
             right_is_vulkan,
             dim)
      .matched;
}

const char* channel_cat_family_name(const ChannelCatFamily family) {
  switch (family) {
    case ChannelCatFamily::Rank4Dim1BufferView:
      return "ChannelCatRank4Dim1BufferView";
    case ChannelCatFamily::None:
      return "ChannelCatNone";
  }
  return "ChannelCatNone";
}

const char* channel_cat_op_hit_label(const ChannelCatFamily family) {
  switch (family) {
    case ChannelCatFamily::Rank4Dim1BufferView:
      return "aten::cat.buffer_channel_view";
    case ChannelCatFamily::None:
      return "aten::cat.channel_cat.none";
  }
  return "aten::cat.channel_cat.none";
}

ChannelCatMatch match_channel_cat_contract(
    ArrayRef<ChannelCatTensorInfo> tensors,
    const int64_t dim) {
  ChannelCatMatch result;
  if (
      tensors.size() < kChannelCatRank4Dim1MinInputs ||
      tensors.size() > kChannelCatRank4Dim1MaxInputs || dim != 1) {
    return result;
  }

  const ChannelCatTensorInfo& reference = tensors[0];
  if (
      !reference.is_vulkan || reference.dtype != kFloat ||
      reference.rank != 4 || reference.batch != kChannelCatRank4Dim1Batch ||
      !reference.is_contiguous || reference.height <= 0 ||
      reference.height > kChannelCatRank4Dim1MaxHeight ||
      reference.width <= 0 || reference.width > kChannelCatRank4Dim1MaxWidth) {
    return result;
  }

  int64_t total_channels = 0;
  for (const ChannelCatTensorInfo& tensor : tensors) {
    if (
        !tensor.is_vulkan || tensor.dtype != reference.dtype ||
        tensor.rank != reference.rank || tensor.batch != reference.batch ||
        tensor.height != reference.height || tensor.width != reference.width ||
        !tensor.is_contiguous || !tensor.has_buffer_storage ||
        !tensor.supports_buffer_compute || tensor.channels <= 0 ||
        tensor.channels > kChannelCatRank4Dim1MaxInputChannels ||
        tensor.channels % 4 != 0) {
      return result;
    }
    total_channels += tensor.channels;
  }

  if (
      total_channels <= 0 ||
      total_channels > kChannelCatRank4Dim1MaxTotalChannels ||
      total_channels % 4 != 0) {
    return result;
  }

  result.matched = true;
  result.family = ChannelCatFamily::Rank4Dim1BufferView;
  result.tuple_id = kChannelCatRank4Dim1BufferViewTupleId;
  result.metadata = &kChannelCatRank4Dim1BufferViewMetadata;
  result.input_count = static_cast<int64_t>(tensors.size());
  result.total_channels = total_channels;
  return result;
}

bool matches_channel_cat_contract(
    ArrayRef<ChannelCatTensorInfo> tensors,
    const int64_t dim) {
  return match_channel_cat_contract(tensors, dim).matched;
}

const char* embedding_lookup_family_name(const EmbeddingLookupFamily family) {
  switch (family) {
    case EmbeddingLookupFamily::SmallBoundedLookup:
      return "EmbeddingLookupSmallBoundedLookup";
    case EmbeddingLookupFamily::TokenBatch1:
      return "EmbeddingLookupTokenBatch1";
    case EmbeddingLookupFamily::None:
      return "EmbeddingLookupNone";
  }
  return "EmbeddingLookupNone";
}

const char* embedding_lookup_write_label(const EmbeddingLookupFamily family) {
  switch (family) {
    case EmbeddingLookupFamily::SmallBoundedLookup:
      return "buffer_float_long.small_bounded_lookup";
    case EmbeddingLookupFamily::TokenBatch1:
      return "buffer_float_long.token_batch1";
    case EmbeddingLookupFamily::None:
      return "buffer_float_long.none";
  }
  return "buffer_float_long.none";
}

EmbeddingLookupMatch match_embedding_lookup_contract(
    const IntArrayRef weight_sizes,
    const IntArrayRef indices_sizes,
    const ScalarType weight_dtype,
    const ScalarType indices_dtype,
    const bool weight_is_vulkan,
    const bool indices_is_vulkan,
    const bool padding_idx_has_hint,
    const bool scale_grad_by_freq,
    const bool sparse) {
  EmbeddingLookupMatch result;
  if (
      !weight_is_vulkan || !indices_is_vulkan ||
      weight_dtype != kFloat || indices_dtype != kLong ||
      weight_sizes.size() != 2 ||
      (indices_sizes.size() != 1 && indices_sizes.size() != 2) ||
      !padding_idx_has_hint || scale_grad_by_freq || sparse) {
    return result;
  }

  const int64_t num_embeddings = weight_sizes[0];
  const int64_t embedding_dim = weight_sizes[1];
  const int64_t num_indices = product_of_sizes(indices_sizes);
  result.num_embeddings = num_embeddings;
  result.embedding_dim = embedding_dim;
  result.num_indices = num_indices;

  if (
      num_embeddings == kEmbeddingLookupTokenNumEmbeddings &&
      embedding_dim == kEmbeddingLookupTokenEmbeddingDim &&
      indices_sizes.size() == 2 &&
      indices_sizes[0] == kEmbeddingLookupTokenBatch &&
      indices_sizes[1] >= kEmbeddingLookupTokenMinIndices &&
      indices_sizes[1] <= kEmbeddingLookupTokenMaxIndices) {
    result.matched = true;
    result.family = EmbeddingLookupFamily::TokenBatch1;
    result.tuple_id = kEmbeddingLookupTokenBatch1TupleId;
    result.metadata = &kEmbeddingLookupTokenBatch1Metadata;
    return result;
  }

  if (
      embedding_dim <= kEmbeddingLookupSmallMaxEmbeddingDim &&
      num_indices <= kEmbeddingLookupSmallMaxNumIndices &&
      num_embeddings <= kEmbeddingLookupSmallMaxNumEmbeddings) {
    result.matched = true;
    result.family = EmbeddingLookupFamily::SmallBoundedLookup;
    result.tuple_id = kEmbeddingLookupSmallBoundedTupleId;
    result.metadata = &kEmbeddingLookupSmallBoundedMetadata;
    return result;
  }

  return result;
}

bool matches_embedding_lookup_contract(
    const IntArrayRef weight_sizes,
    const IntArrayRef indices_sizes,
    const ScalarType weight_dtype,
    const ScalarType indices_dtype,
    const bool weight_is_vulkan,
    const bool indices_is_vulkan,
    const bool padding_idx_has_hint,
    const bool scale_grad_by_freq,
    const bool sparse) {
  return match_embedding_lookup_contract(
             weight_sizes,
             indices_sizes,
             weight_dtype,
             indices_dtype,
             weight_is_vulkan,
             indices_is_vulkan,
             padding_idx_has_hint,
             scale_grad_by_freq,
             sparse)
      .matched;
}

const char* batch_norm_inference_family_name(
    const BatchNormInferenceFamily family) {
  switch (family) {
    case BatchNormInferenceFamily::BufferFloat4D:
      return "BatchNormInferenceBufferFloat4D";
    case BatchNormInferenceFamily::MaterializedBufferFloat4D:
      return "BatchNormInferenceMaterializedBufferFloat4D";
    case BatchNormInferenceFamily::None:
      return "BatchNormInferenceNone";
  }
  return "BatchNormInferenceNone";
}

BatchNormInferenceMatch match_batch_norm_inference_contract(
    const BatchNormInferenceTensorInfo& input,
    const BatchNormInferenceTensorInfo& weight,
    const BatchNormInferenceTensorInfo& bias,
    const BatchNormInferenceTensorInfo& running_mean,
    const BatchNormInferenceTensorInfo& running_var,
    const bool training) {
  BatchNormInferenceMatch result;
  if (
      training || !input.has_value || !input.defined || !input.is_vulkan ||
      input.dtype != kFloat || input.dim != 4 || !input.is_contiguous ||
      !input.supports_buffer_compute) {
    return result;
  }

  const int64_t num_features = input.channels;
  const bool buffer_match =
      input.has_buffer_storage &&
      batch_norm_float_1d_buffer_matches(running_mean, num_features) &&
      batch_norm_float_1d_buffer_matches(running_var, num_features) &&
      batch_norm_optional_float_1d_matches(weight, num_features) &&
      batch_norm_optional_float_1d_matches(bias, num_features) &&
      batch_norm_effective_affine_has_buffer_storage(weight, running_mean) &&
      batch_norm_effective_affine_has_buffer_storage(bias, running_mean);
  if (buffer_match) {
    result.matched = true;
    result.family = BatchNormInferenceFamily::BufferFloat4D;
    result.tuple_id = kBatchNormInferenceBufferFloat4DTupleId;
    result.metadata = &kBatchNormInferenceBufferFloat4DMetadata;
    return result;
  }

  if (
      !batch_norm_float_1d_materializable_matches(
          running_mean, num_features) ||
      !batch_norm_float_1d_materializable_matches(running_var, num_features) ||
      !batch_norm_optional_float_1d_materializable_matches(
          weight, num_features) ||
      !batch_norm_optional_float_1d_materializable_matches(
          bias, num_features) ||
      !batch_norm_effective_affine_supports_buffer_compute(
          weight, running_mean) ||
      !batch_norm_effective_affine_supports_buffer_compute(
          bias, running_mean)) {
    return result;
  }

  result.matched = true;
  result.family = BatchNormInferenceFamily::MaterializedBufferFloat4D;
  result.tuple_id = kBatchNormInferenceMaterializedBufferFloat4DTupleId;
  result.metadata = &kBatchNormInferenceMaterializedBufferFloat4DMetadata;
  result.requires_materialization = true;
  return result;
}

bool matches_batch_norm_inference_contract(
    const BatchNormInferenceTensorInfo& input,
    const BatchNormInferenceTensorInfo& weight,
    const BatchNormInferenceTensorInfo& bias,
    const BatchNormInferenceTensorInfo& running_mean,
    const BatchNormInferenceTensorInfo& running_var,
    const bool training) {
  return match_batch_norm_inference_contract(
             input, weight, bias, running_mean, running_var, training)
      .matched;
}

const char* safe_view_reshape_family_name(
    const SafeViewReshapeFamily family) {
  switch (family) {
    case SafeViewReshapeFamily::ViewMaterializedDirectBuffer:
      return "SafeViewReshapeViewMaterializedDirectBuffer";
    case SafeViewReshapeFamily::ReshapeAliasDenseBufferDirect:
      return "SafeViewReshapeReshapeAliasDenseBufferDirect";
    case SafeViewReshapeFamily::None:
      return "SafeViewReshapeNone";
  }
  return "SafeViewReshapeNone";
}

SafeViewReshapeMatch
match_safe_view_reshape_materialized_direct_buffer_contract(
    const IntArrayRef input_sizes,
    const IntArrayRef output_sizes,
    const IntArrayRef output_strides,
    const int64_t storage_offset) {
  SafeViewReshapeMatch result;
  if (
      input_sizes.size() > 4 || output_sizes.size() > 5 ||
      storage_offset != 0 ||
      !is_contiguous_stride(output_sizes, output_strides)) {
    return result;
  }

  if (product_of_sizes(input_sizes) != product_of_sizes(output_sizes)) {
    return result;
  }

  if (!output_sizes.empty() && output_sizes.back() % 4 != 0) {
    return result;
  }

  result.matched = true;
  result.family = SafeViewReshapeFamily::ViewMaterializedDirectBuffer;
  result.tuple_id = kSafeViewReshapeViewMaterializedDirectBufferTupleId;
  result.metadata = &kSafeViewReshapeViewMaterializedDirectBufferMetadata;
  return result;
}

bool matches_safe_view_reshape_materialized_direct_buffer_contract(
    const IntArrayRef input_sizes,
    const IntArrayRef output_sizes,
    const IntArrayRef output_strides,
    const int64_t storage_offset) {
  return match_safe_view_reshape_materialized_direct_buffer_contract(
             input_sizes, output_sizes, output_strides, storage_offset)
      .matched;
}

SafeViewReshapeMatch match_safe_view_reshape_contract(
    const IntArrayRef input_sizes,
    const IntArrayRef input_logical_strides,
    const IntArrayRef output_sizes,
    const IntArrayRef output_strides,
    const bool input_is_float,
    const bool input_has_buffer_storage,
    const int64_t storage_offset) {
  SafeViewReshapeMatch result;
  if (
      !input_is_float || !input_has_buffer_storage ||
      input_sizes.size() > 4 || output_sizes.size() > 5 ||
      storage_offset != 0 ||
      !is_non_overlapping_dense_stride(input_sizes, input_logical_strides) ||
      !is_non_overlapping_dense_stride(output_sizes, output_strides)) {
    return result;
  }

  if (product_of_sizes(input_sizes) != product_of_sizes(output_sizes)) {
    return result;
  }

  if (!output_sizes.empty() && output_sizes.back() % 4 != 0) {
    return result;
  }

  result.matched = true;
  result.family = SafeViewReshapeFamily::ReshapeAliasDenseBufferDirect;
  result.tuple_id = kSafeViewReshapeAliasDenseBufferDirectTupleId;
  result.metadata = &kSafeViewReshapeAliasDenseBufferDirectMetadata;
  return result;
}

bool matches_safe_view_reshape_contract(
    const IntArrayRef input_sizes,
    const IntArrayRef input_logical_strides,
    const IntArrayRef output_sizes,
    const IntArrayRef output_strides,
    const bool input_is_float,
    const bool input_has_buffer_storage,
    const int64_t storage_offset) {
  return match_safe_view_reshape_contract(
             input_sizes,
             input_logical_strides,
             output_sizes,
             output_strides,
             input_is_float,
             input_has_buffer_storage,
             storage_offset)
      .matched;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
