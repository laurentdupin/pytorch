#include <ATen/native/vulkan/planning/ExecutionContracts.h>

#include <cmath>

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
constexpr const char* kMaterializationDelegatedToSDPAExecutionPolicy =
    "delegated_to_sdpa_execution_policy";
constexpr double kTransformerHeadDim128Scale = 0.08838834764831845;

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

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
