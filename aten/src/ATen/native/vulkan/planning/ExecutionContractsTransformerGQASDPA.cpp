#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/DynamicProgramRuntime.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsTransformerGQASDPASpec.h>

#include <cmath>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

constexpr ExecutionContractMetadata kDynamicDirectDecodeGQAMetadata{
    "TransformerGQASDPAContract",
    "DynamicDirectDecodeGQA",
    "dynamic_direct_decode_gqa_runtime_shape",
    "dynamic_program_runtime_random_gqa_sdpa",
    "dynamic_direct_decode_gqa_adjacent_guards",
    "unsupported_shapes_do_not_match",
    "direct_gqa_buffer_no_repeat_materialization"};

constexpr ExecutionContractMetadata kDynamicDirectCausalPrefillMetadata{
    "TransformerGQASDPAContract",
    "CausalPrefill",
    "dynamic_direct_causal_prefill_runtime_shape",
    "dynamic_program_runtime_random_causal_prefill_sdpa",
    "dynamic_direct_causal_prefill_mha_gqa_adjacent_guards",
    "unsupported_shapes_do_not_match",
    "direct_gqa_buffer_causal_mask_in_shader"};

constexpr ExecutionContractMetadata kDynamicSmallNonCausalGQAMetadata{
    "TransformerGQASDPAContract",
    "SmallNonCausalGQA",
    "dynamic_small_non_causal_gqa_runtime_shape",
    "dynamic_program_runtime_random_small_non_causal_gqa_sdpa",
    "dynamic_small_non_causal_gqa_adjacent_guards",
    "unsupported_shapes_do_not_match",
    "direct_gqa_buffer_no_repeat_materialization"};

constexpr ExecutionContractMetadata kDynamicDirectNonCausalMHAMetadata{
    "TransformerGQASDPAContract",
    "DirectNonCausalMHA",
    "dynamic_direct_non_causal_mha_runtime_shape",
    "dynamic_program_runtime_random_non_causal_mha_sdpa",
    "dynamic_direct_non_causal_mha_adjacent_guards",
    "unsupported_shapes_do_not_match",
    "direct_gqa_buffer_repeat_factor_one"};

const char* transformer_gqa_sdpa_contract_family_name(
    const TransformerGQASDPAFamily family) {
  switch (family) {
    case TransformerGQASDPAFamily::CausalPrefill:
      return "CausalPrefill";
    case TransformerGQASDPAFamily::SmallNonCausalGQA:
      return "SmallNonCausalGQA";
    case TransformerGQASDPAFamily::DecodeGQA:
      return "DecodeGQA";
    case TransformerGQASDPAFamily::DynamicDirectDecodeGQA:
      return "DynamicDirectDecodeGQA";
    case TransformerGQASDPAFamily::DirectNonCausalMHA:
      return "DirectNonCausalMHA";
    case TransformerGQASDPAFamily::None:
      return "";
  }
  return "";
}

bool apply_transformer_gqa_sdpa_row(
    TransformerGQASDPAMatch& result,
    const TransformerGQASDPAFamily family,
    const generated::TransformerGQASDPAAttentionRowsRow* const row,
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const bool is_causal,
    const bool enable_gqa) {
  if (row == nullptr ||
      !generated::transformer_gqasdpa_attention_rows_row_matches(
          *row,
          query_sizes[1],
          key_sizes[1],
          query_sizes[2],
          key_sizes[2],
          query_sizes[3],
          is_causal,
          enable_gqa)) {
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
    case TransformerGQASDPAFamily::DynamicDirectDecodeGQA:
      return "TransformerGQASDPADynamicDirectDecodeGQA";
    case TransformerGQASDPAFamily::DirectNonCausalMHA:
      return "TransformerGQASDPADirectNonCausalMHA";
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
    case TransformerGQASDPAFamily::DynamicDirectDecodeGQA:
      return "SelectedDynamicDirectDecodeGQASDPA";
    case TransformerGQASDPAFamily::DirectNonCausalMHA:
      return "SelectedTransformerGQASDPADirectNonCausalMHA";
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
      has_attn_mask || dropout_p != 0.0 ||
      query_dtype != kFloat || key_dtype != kFloat || value_dtype != kFloat ||
      query_sizes.size() != 4 || key_sizes.size() != 4 ||
      value_sizes.size() != 4) {
    return result;
  }
  if (
      query_sizes[0] != 1 || key_sizes[0] != 1 || value_sizes[0] != 1 ||
      query_sizes[2] < 1 || key_sizes[2] < 1 ||
      (is_causal && key_sizes[2] < query_sizes[2]) ||
      query_sizes[3] <= 0 || key_sizes[3] != query_sizes[3] ||
      value_sizes[2] != key_sizes[2] || value_sizes[3] <= 0 ||
      key_sizes[1] != value_sizes[1]) {
    return result;
  }
  const double expected_scale =
      1.0 / std::sqrt(static_cast<double>(query_sizes[3]));
  if (scale.has_value() && std::abs(*scale - expected_scale) > 1.0e-6) {
    return result;
  }

  const bool finite_transformer_envelope =
      query_sizes[1] == 16 && query_sizes[2] <= 128 &&
      query_sizes[3] == 128 && value_sizes[3] == 128;

  if (is_causal && finite_transformer_envelope) {
    const auto family = TransformerGQASDPAFamily::CausalPrefill;
    const auto* const row = generated::transformer_gqasdpa_attention_rows_find(
        transformer_gqa_sdpa_contract_family_name(family), is_causal, enable_gqa);
    if (apply_transformer_gqa_sdpa_row(
            result, family, row, query_sizes, key_sizes, is_causal, enable_gqa)) {
      return result;
    }
    return result;
  }

  if (is_causal) {
    const DynamicProgramDecision dynamic_decision =
        build_dynamic_program_runtime_plan(
            make_direct_causal_prefill_gqa_sdpa_direct_buffer_dynamic_program(
                query_sizes,
                key_sizes,
                value_sizes,
                query_dtype,
                key_dtype,
                value_dtype,
                true,
                true,
                true,
                has_attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                &kDynamicDirectCausalPrefillMetadata,
                true));
    if (dynamic_decision.runtime_selection_authorized) {
      result.matched = true;
      result.family = TransformerGQASDPAFamily::CausalPrefill;
      result.tuple_id = kDynamicDirectCausalPrefillMetadata.tuple_id;
      result.metadata = &kDynamicDirectCausalPrefillMetadata;
    }
    return result;
  }

  if (finite_transformer_envelope) {
    const auto family = TransformerGQASDPAFamily::DecodeGQA;
    const auto* const row = generated::transformer_gqasdpa_attention_rows_find(
        transformer_gqa_sdpa_contract_family_name(family), is_causal, enable_gqa);
    if (apply_transformer_gqa_sdpa_row(
            result, family, row, query_sizes, key_sizes, is_causal, enable_gqa)) {
      return result;
    }

    const auto small_family = TransformerGQASDPAFamily::SmallNonCausalGQA;
    const auto* const small_row =
        generated::transformer_gqasdpa_attention_rows_find(
            transformer_gqa_sdpa_contract_family_name(small_family),
            is_causal,
            enable_gqa);
    if (apply_transformer_gqa_sdpa_row(
            result,
            small_family,
            small_row,
            query_sizes,
            key_sizes,
            is_causal,
            enable_gqa)) {
      return result;
    }
  }

  const DynamicProgramDecision non_causal_mha_decision =
      build_dynamic_program_runtime_plan(
          make_direct_non_causal_mha_sdpa_direct_buffer_dynamic_program(
              query_sizes,
              key_sizes,
              value_sizes,
              query_dtype,
              key_dtype,
              value_dtype,
              true,
              true,
              true,
              has_attn_mask,
              dropout_p,
              is_causal,
              scale,
              enable_gqa,
              &kDynamicDirectNonCausalMHAMetadata,
              true));
  if (non_causal_mha_decision.runtime_selection_authorized) {
    result.matched = true;
    result.family = TransformerGQASDPAFamily::DirectNonCausalMHA;
    result.tuple_id = kDynamicDirectNonCausalMHAMetadata.tuple_id;
    result.metadata = &kDynamicDirectNonCausalMHAMetadata;
    return result;
  }

  const DynamicProgramDecision small_non_causal_decision =
      build_dynamic_program_runtime_plan(
          make_small_non_causal_gqa_sdpa_direct_buffer_dynamic_program(
              query_sizes,
              key_sizes,
              value_sizes,
              query_dtype,
              key_dtype,
              value_dtype,
              true,
              true,
              true,
              has_attn_mask,
              dropout_p,
              is_causal,
              scale,
              enable_gqa,
              &kDynamicSmallNonCausalGQAMetadata,
              true));
  if (small_non_causal_decision.runtime_selection_authorized) {
    result.matched = true;
    result.family = TransformerGQASDPAFamily::SmallNonCausalGQA;
    result.tuple_id = kDynamicSmallNonCausalGQAMetadata.tuple_id;
    result.metadata = &kDynamicSmallNonCausalGQAMetadata;
    return result;
  }

  const DynamicProgramDecision dynamic_decision =
      build_dynamic_program_runtime_plan(
          make_direct_decode_gqa_sdpa_direct_buffer_dynamic_program(
              query_sizes,
              key_sizes,
              value_sizes,
              query_dtype,
              key_dtype,
              value_dtype,
              true,
              true,
              true,
              has_attn_mask,
              dropout_p,
              is_causal,
              scale,
              enable_gqa,
              &kDynamicDirectDecodeGQAMetadata,
              true));
  if (dynamic_decision.runtime_selection_authorized) {
    result.matched = true;
    result.family = TransformerGQASDPAFamily::DynamicDirectDecodeGQA;
    result.tuple_id = kDynamicDirectDecodeGQAMetadata.tuple_id;
    result.metadata = &kDynamicDirectDecodeGQAMetadata;
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
