#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsTransformerGQASDPASpec.h>

#include <cmath>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

constexpr double kTransformerHeadDim128Scale = 0.08838834764831845;

const char* transformer_gqa_sdpa_contract_family_name(
    const TransformerGQASDPAFamily family) {
  switch (family) {
    case TransformerGQASDPAFamily::CausalPrefill:
      return "CausalPrefill";
    case TransformerGQASDPAFamily::SmallNonCausalGQA:
      return "SmallNonCausalGQA";
    case TransformerGQASDPAFamily::DecodeGQA:
      return "DecodeGQA";
    case TransformerGQASDPAFamily::None:
      return "";
  }
  return "";
}

bool transformer_gqa_sdpa_row_matches(
    const generated::TransformerGQASDPAAttentionRowsRow& row,
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes) {
  return query_sizes[1] == row.query_heads &&
      key_sizes[1] == row.key_value_heads &&
      query_sizes[2] >= row.query_sequence_min &&
      query_sizes[2] <= row.query_sequence_max &&
      key_sizes[2] >= row.key_value_sequence_min &&
      key_sizes[2] <= row.key_value_sequence_max &&
      query_sizes[3] == row.head_dim && key_sizes[3] == row.head_dim &&
      (!row.requires_equal_sequence || query_sizes[2] == key_sizes[2]);
}

bool apply_transformer_gqa_sdpa_row(
    TransformerGQASDPAMatch& result,
    const TransformerGQASDPAFamily family,
    const generated::TransformerGQASDPAAttentionRowsRow* const row,
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes) {
  if (row == nullptr ||
      !transformer_gqa_sdpa_row_matches(*row, query_sizes, key_sizes)) {
    return false;
  }
  result.matched = true;
  result.family = family;
  result.tuple_id = row->tuple_id;
  result.metadata = &row->metadata;
  return true;
}

} // namespace

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

  if (is_causal) {
    const auto family = TransformerGQASDPAFamily::CausalPrefill;
    const auto* const row = generated::transformer_gqasdpa_attention_rows_find(
        transformer_gqa_sdpa_contract_family_name(family), is_causal, enable_gqa);
    if (apply_transformer_gqa_sdpa_row(result, family, row, query_sizes, key_sizes)) {
      return result;
    }
    return result;
  }

  {
    const auto family = TransformerGQASDPAFamily::DecodeGQA;
    const auto* const row = generated::transformer_gqasdpa_attention_rows_find(
        transformer_gqa_sdpa_contract_family_name(family), is_causal, enable_gqa);
    if (apply_transformer_gqa_sdpa_row(result, family, row, query_sizes, key_sizes)) {
      return result;
    }
  }

  const auto family = TransformerGQASDPAFamily::SmallNonCausalGQA;
  const auto* const row = generated::transformer_gqasdpa_attention_rows_find(
      transformer_gqa_sdpa_contract_family_name(family), is_causal, enable_gqa);
  if (apply_transformer_gqa_sdpa_row(result, family, row, query_sizes, key_sizes)) {
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

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
