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
constexpr const char* kMaterializationNone = "none";
constexpr double kHeadDim64Scale = 0.125;

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

} // namespace

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

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
