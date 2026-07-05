#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsSDPAExecutionPolicySpec.h>

#include <cmath>
#include <string_view>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

constexpr int64_t kRuntimeRecognizerMaxSequence = 512;
constexpr int64_t kRuntimeRecognizerMaxHeadDim = 32;
constexpr int64_t kRuntimeRecognizerMaxValueDim = 32;
constexpr int64_t kRuntimeTransformerDecodeMaxScoreElements = 2097152;
constexpr int64_t kRuntimeTransformerDecodeMaxHeadDim = 128;

constexpr ExecutionContractMetadata kRecognizerRuntimeMetadata{
    "SDPAExecutionPolicyContract",
    "RecognizerNonCausalMHARuntimeShape",
    "recognizer_mha_runtime_shape",
    "sdpa_execution_policy_recognizer_dynamic_random_shape_tests",
    "sdpa_execution_policy_recognizer_dynamic_semantic_guards",
    "fallback_on_unsupported_layout_or_semantics",
    "runtime_fused_direct_buffer"};

constexpr ExecutionContractMetadata kDiffusionMaterializedSquareRuntimeMetadata{
    "SDPAExecutionPolicyContract",
    "DiffusionMaterializedSquareRuntimeShape",
    "diffusion_materialized_square_runtime_shape",
    "diffusion_square_sdpa_dynamic_random_shape_tests",
    "diffusion_square_sdpa_dynamic_semantic_guards",
    "fallback_on_unsupported_layout_or_semantics",
    "score_pre_materialization_and_post_softmax_clone"};

constexpr ExecutionContractMetadata kTransformerDecodeGQARuntimeMetadata{
    "SDPAExecutionPolicyContract",
    "TransformerDecodeGQACloneOnlyRuntimeShape",
    "transformer_decode_gqa_clone_only_runtime_shape",
    "sdpa_execution_policy_transformer_decode_gqa_dynamic_random_shape_tests",
    "sdpa_execution_policy_transformer_decode_gqa_dynamic_semantic_guards",
    "fallback_on_unsupported_layout_or_semantics",
    "post_softmax_clone"};

const char* sdpa_execution_policy_contract_family_name(
    const SDPAExecutionPolicyFamily family,
    const char* const fallback_name) {
  switch (family) {
    case SDPAExecutionPolicyFamily::DiffusionMaterializedSquare:
      return "DiffusionMaterializedSquare";
    case SDPAExecutionPolicyFamily::DiffusionMaterializedSquareRuntimeShape:
      return "DiffusionMaterializedSquareRuntimeShape";
    case SDPAExecutionPolicyFamily::DiffusionCloneOnlySquare:
      return "DiffusionCloneOnlySquare";
    case SDPAExecutionPolicyFamily::TransformerDecodeGQACloneOnly:
      return "TransformerDecodeGQACloneOnly";
    case SDPAExecutionPolicyFamily::TransformerDecodeGQACloneOnlyRuntimeShape:
      return "TransformerDecodeGQACloneOnlyRuntimeShape";
    case SDPAExecutionPolicyFamily::VisionSelfAttentionCloneOnly:
      return "VisionSelfAttentionCloneOnly";
    case SDPAExecutionPolicyFamily::RecognizerNonCausalMHACloneOnly:
      return "RecognizerNonCausalMHACloneOnly";
    case SDPAExecutionPolicyFamily::None:
      return fallback_name;
  }
  return fallback_name;
}

const generated::SDPAExecutionPolicyPolicyRowsRow*
find_sdpa_execution_policy_row(
    const SDPAExecutionPolicyFamily family,
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const bool enable_gqa) {
  const char* const family_name =
      sdpa_execution_policy_contract_family_name(family, "");
  for (const auto& row : generated::kSDPAExecutionPolicyPolicyRowsRows) {
    if (generated::sdpa_execution_policy_policy_rows_row_matches(
            row,
            family_name,
            query_sizes[0],
            query_sizes[1],
            key_sizes[1],
            query_sizes[2],
            key_sizes[2],
            query_sizes[3],
            enable_gqa)) {
      return &row;
    }
  }
  return nullptr;
}

void apply_sdpa_execution_policy_row(
    SDPAExecutionPolicyMatch& result,
    const SDPAExecutionPolicyFamily family,
    const generated::SDPAExecutionPolicyPolicyRowsRow& row) {
  result.matched = true;
  result.family = family;
  result.tuple_id = row.tuple_id;
  result.metadata = &row.metadata;
  result.requires_materialized_math_path =
      row.requires_materialized_math_path;
  result.requires_score_pre_materialization =
      row.requires_score_pre_materialization;
  result.requires_post_softmax_clone = row.requires_post_softmax_clone;
}

bool is_recognizer_non_causal_mha_runtime_shape(
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const IntArrayRef value_sizes,
    const bool enable_gqa) {
  if (
      enable_gqa || query_sizes.size() != 4 || key_sizes.size() != 4 ||
      value_sizes.size() != 4) {
    return false;
  }
  const int64_t batch = query_sizes[0];
  const int64_t query_heads = query_sizes[1];
  const int64_t key_value_heads = key_sizes[1];
  const int64_t query_sequence = query_sizes[2];
  const int64_t key_value_sequence = key_sizes[2];
  const int64_t head_dim = query_sizes[3];
  const int64_t value_dim = value_sizes[3];
  return batch > 0 && query_heads > 0 && key_value_heads > 0 &&
      query_heads == key_value_heads && value_sizes[1] == key_value_heads &&
      query_sequence > 0 && key_value_sequence > 0 &&
      query_sequence <= kRuntimeRecognizerMaxSequence &&
      key_value_sequence <= kRuntimeRecognizerMaxSequence && head_dim > 0 &&
      head_dim <= kRuntimeRecognizerMaxHeadDim && value_dim > 0 &&
      value_dim <= kRuntimeRecognizerMaxValueDim &&
      key_sizes[3] == head_dim && value_dim == head_dim;
}

bool scale_matches_head_dim(
    const std::optional<double> scale,
    const int64_t head_dim) {
  if (!scale.has_value()) {
    return true;
  }
  const double expected_scale =
      1.0 / std::sqrt(static_cast<double>(head_dim));
  return std::abs(*scale - expected_scale) <= 1.0e-6;
}

bool is_transformer_decode_gqa_clone_only_runtime_shape(
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const IntArrayRef value_sizes,
    const std::optional<double> scale,
    const bool enable_gqa) {
  if (
      !enable_gqa || query_sizes.size() != 4 || key_sizes.size() != 4 ||
      value_sizes.size() != 4) {
    return false;
  }
  const int64_t batch = query_sizes[0];
  const int64_t query_heads = query_sizes[1];
  const int64_t key_value_heads = key_sizes[1];
  const int64_t query_sequence = query_sizes[2];
  const int64_t key_value_sequence = key_sizes[2];
  const int64_t head_dim = query_sizes[3];
  return batch == 1 && key_sizes[0] == batch && value_sizes[0] == batch &&
      query_heads > 0 && key_value_heads > 0 &&
      key_sizes[1] == value_sizes[1] &&
      query_heads % key_value_heads == 0 && query_sequence == 1 &&
      key_value_sequence > 0 && value_sizes[2] == key_value_sequence &&
      head_dim > 0 && head_dim <= kRuntimeTransformerDecodeMaxHeadDim &&
      key_sizes[3] == head_dim && value_sizes[3] == head_dim &&
      query_heads * key_value_sequence <=
          kRuntimeTransformerDecodeMaxScoreElements &&
      scale_matches_head_dim(scale, head_dim);
}

void apply_recognizer_runtime_policy(SDPAExecutionPolicyMatch& result) {
  result.matched = true;
  result.family = SDPAExecutionPolicyFamily::RecognizerNonCausalMHACloneOnly;
  result.tuple_id = kRecognizerRuntimeMetadata.tuple_id;
  result.metadata = &kRecognizerRuntimeMetadata;
  result.requires_materialized_math_path = false;
  result.requires_score_pre_materialization = false;
  result.requires_post_softmax_clone = false;
}

void apply_diffusion_materialized_square_runtime_policy(
    SDPAExecutionPolicyMatch& result) {
  result.matched = true;
  result.family = SDPAExecutionPolicyFamily::DiffusionMaterializedSquareRuntimeShape;
  result.tuple_id = kDiffusionMaterializedSquareRuntimeMetadata.tuple_id;
  result.metadata = &kDiffusionMaterializedSquareRuntimeMetadata;
  result.requires_materialized_math_path = false;
  result.requires_score_pre_materialization = true;
  result.requires_post_softmax_clone = true;
}

void apply_transformer_decode_gqa_runtime_policy(
    SDPAExecutionPolicyMatch& result) {
  result.matched = true;
  result.family =
      SDPAExecutionPolicyFamily::TransformerDecodeGQACloneOnlyRuntimeShape;
  result.tuple_id = kTransformerDecodeGQARuntimeMetadata.tuple_id;
  result.metadata = &kTransformerDecodeGQARuntimeMetadata;
  result.requires_materialized_math_path = false;
  result.requires_score_pre_materialization = false;
  result.requires_post_softmax_clone = true;
}

} // namespace

const char* sdpa_execution_policy_family_name(
    const SDPAExecutionPolicyFamily family) {
  switch (family) {
    case SDPAExecutionPolicyFamily::DiffusionMaterializedSquare:
      return "SDPAExecutionDiffusionMaterializedSquare";
    case SDPAExecutionPolicyFamily::DiffusionMaterializedSquareRuntimeShape:
      return "SDPAExecutionDiffusionMaterializedSquareRuntimeShape";
    case SDPAExecutionPolicyFamily::DiffusionCloneOnlySquare:
      return "SDPAExecutionDiffusionCloneOnlySquare";
    case SDPAExecutionPolicyFamily::TransformerDecodeGQACloneOnly:
      return "SDPAExecutionTransformerDecodeGQACloneOnly";
    case SDPAExecutionPolicyFamily::TransformerDecodeGQACloneOnlyRuntimeShape:
      return "SDPAExecutionTransformerDecodeGQACloneOnlyRuntimeShape";
    case SDPAExecutionPolicyFamily::VisionSelfAttentionCloneOnly:
      return "SDPAExecutionVisionSelfAttentionCloneOnly";
    case SDPAExecutionPolicyFamily::RecognizerNonCausalMHACloneOnly:
      return "SDPAExecutionRecognizerNonCausalMHACloneOnly";
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
      query_sizes.size() != key_sizes.size() ||
      query_sizes.size() != value_sizes.size()) {
    return result;
  }

  if (query_sizes.size() == 3) {
    const VisionSelfAttentionSDPAMatch vision_match =
        match_vision_self_attention_sdpa_contract(
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
    if (!vision_match.matched) {
      return result;
    }
    result.matched = true;
    result.family = SDPAExecutionPolicyFamily::VisionSelfAttentionCloneOnly;
    result.tuple_id = vision_match.tuple_id;
    result.metadata = vision_match.metadata;
    result.requires_materialized_math_path = true;
    result.requires_score_pre_materialization = false;
    result.requires_post_softmax_clone = true;
    return result;
  }

  if (
      query_sizes.size() != 4 || query_sizes[0] < 1 ||
      key_sizes[0] != query_sizes[0] || value_sizes[0] != query_sizes[0] ||
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
      {
        constexpr auto family =
            SDPAExecutionPolicyFamily::DiffusionMaterializedSquare;
        const auto* const row = find_sdpa_execution_policy_row(
            family, query_sizes, key_sizes, enable_gqa);
        if (row != nullptr) {
          if (std::string_view(row->tuple_id) != diffusion_match.tuple_id) {
            return result;
          }
          apply_sdpa_execution_policy_row(result, family, *row);
          return result;
        }
      }
      {
        constexpr auto family =
            SDPAExecutionPolicyFamily::DiffusionCloneOnlySquare;
        const auto* const row = find_sdpa_execution_policy_row(
            family, query_sizes, key_sizes, enable_gqa);
        if (row != nullptr) {
          if (std::string_view(row->tuple_id) != diffusion_match.tuple_id) {
            return result;
          }
          apply_sdpa_execution_policy_row(result, family, *row);
          return result;
        }
      }
    }

    if (
        diffusion_match.matched &&
        diffusion_match.family ==
            DiffusionSDPAFamily::SquareSelfAttentionRuntimeShape) {
      apply_diffusion_materialized_square_runtime_policy(result);
      return result;
    }

    if (query_sizes[1] == key_sizes[1] && key_sizes[1] == value_sizes[1]) {
      constexpr auto family =
          SDPAExecutionPolicyFamily::RecognizerNonCausalMHACloneOnly;
      const auto* const row = find_sdpa_execution_policy_row(
          family, query_sizes, key_sizes, enable_gqa);
      if (row != nullptr) {
        apply_sdpa_execution_policy_row(result, family, *row);
        return result;
      }
      if (is_recognizer_non_causal_mha_runtime_shape(
              query_sizes, key_sizes, value_sizes, enable_gqa)) {
        apply_recognizer_runtime_policy(result);
        return result;
      }
    }
  }

  if (
      !is_causal && enable_gqa) {
    const SDPAExecutionPolicyFamily family =
        SDPAExecutionPolicyFamily::TransformerDecodeGQACloneOnly;
    const auto* const row = find_sdpa_execution_policy_row(
        family, query_sizes, key_sizes, enable_gqa);
    if (row != nullptr) {
      apply_sdpa_execution_policy_row(result, family, *row);
      return result;
    }
    if (is_transformer_decode_gqa_clone_only_runtime_shape(
            query_sizes, key_sizes, value_sizes, scale, enable_gqa)) {
      apply_transformer_decode_gqa_runtime_policy(result);
      return result;
    }
  }

  return result;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
