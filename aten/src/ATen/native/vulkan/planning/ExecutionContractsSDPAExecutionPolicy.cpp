#include <ATen/native/vulkan/planning/ExecutionContracts.h>

#include <cstring>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

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
constexpr const char* kMaterializationScorePreMaterializeAndPostSoftmaxClone =
    "score_pre_materialization_and_post_softmax_clone";
constexpr const char* kMaterializationMaterializedMathAndPostSoftmaxClone =
    "materialized_math_path_and_post_softmax_clone";
constexpr const char* kMaterializationPostSoftmaxClone =
    "post_softmax_clone";

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

} // namespace

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

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
