#include <ATen/native/vulkan/planning/ExecutionContracts.h>

#include <c10/util/Exception.h>
#include <cmath>
#include <cstring>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

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
constexpr const char* kMaterializationDelegatedToSDPAExecutionPolicy =
    "delegated_to_sdpa_execution_policy";
constexpr const char* kMaterializationScorePreMaterializeAndPostSoftmaxClone =
    "score_pre_materialization_and_post_softmax_clone";
constexpr const char* kMaterializationMaterializedMathAndPostSoftmaxClone =
    "materialized_math_path_and_post_softmax_clone";
constexpr const char* kMaterializationPostSoftmaxClone =
    "post_softmax_clone";
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

} // namespace

bool has_complete_execution_contract_metadata(
    const ExecutionContractMetadata* metadata) {
  return metadata != nullptr && has_text(metadata->contract_name) &&
      has_text(metadata->family_name) && has_text(metadata->tuple_id) &&
      has_text(metadata->evidence_id) && has_text(metadata->guard_id) &&
      has_text(metadata->fallback_policy) &&
      has_text(metadata->materialization_policy);
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

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
