#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsSDPAExecutionPolicySpec.h>

#include <string_view>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

const char* sdpa_execution_policy_contract_family_name(
    const SDPAExecutionPolicyFamily family,
    const char* const fallback_name) {
  switch (family) {
    case SDPAExecutionPolicyFamily::DiffusionMaterializedSquare:
      return "DiffusionMaterializedSquare";
    case SDPAExecutionPolicyFamily::DiffusionCloneOnlySquare:
      return "DiffusionCloneOnlySquare";
    case SDPAExecutionPolicyFamily::TransformerDecodeGQACloneOnly:
      return "TransformerDecodeGQACloneOnly";
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

    if (query_sizes[1] == key_sizes[1] && key_sizes[1] == value_sizes[1]) {
      constexpr auto family =
          SDPAExecutionPolicyFamily::RecognizerNonCausalMHACloneOnly;
      const auto* const row = find_sdpa_execution_policy_row(
          family, query_sizes, key_sizes, enable_gqa);
      if (row != nullptr) {
        apply_sdpa_execution_policy_row(result, family, *row);
        return result;
      }
    }
  }

  if (
      !is_causal && enable_gqa && value_sizes[1] == 16) {
    const SDPAExecutionPolicyFamily family =
        SDPAExecutionPolicyFamily::TransformerDecodeGQACloneOnly;
    const auto* const row = find_sdpa_execution_policy_row(
        family, query_sizes, key_sizes, enable_gqa);
    if (row == nullptr) {
      return result;
    }
    apply_sdpa_execution_policy_row(result, family, *row);
    return result;
  }

  return result;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
