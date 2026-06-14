#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsMaskedTinySDPASpec.h>

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

constexpr double kHeadDim64Scale = 0.125;

constexpr ExecutionContractMetadata kMaskedTinySDPAAdditiveFloatMaskMetadata =
    make_execution_contract_metadata(
        generated::kMaskedTinySDPAAdditiveFloatMaskSpec.contract_name,
        generated::kMaskedTinySDPAAdditiveFloatMaskSpec.family_name,
        generated::kMaskedTinySDPAAdditiveFloatMaskSpec.tuple_id,
        generated::kMaskedTinySDPAAdditiveFloatMaskSpec.evidence_id,
        generated::kMaskedTinySDPAAdditiveFloatMaskSpec.guard_id,
        generated::kMaskedTinySDPAAdditiveFloatMaskSpec.fallback_policy,
        generated::kMaskedTinySDPAAdditiveFloatMaskSpec.materialization_policy);

int64_t dim_or_sentinel(const IntArrayRef sizes, const size_t dim) {
  return sizes.size() > dim ? sizes[dim] : -1;
}

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
  const auto& spec = generated::kMaskedTinySDPAAdditiveFloatMaskSpec;
  const bool scale_equivalent_head_dim64 =
      !scale.has_value() ||
      !(std::abs(*scale - kHeadDim64Scale) > 1.0e-6);
  const int64_t qkv_batch =
      dim_or_sentinel(query_sizes, 0) == spec.batch &&
          dim_or_sentinel(key_sizes, 0) == spec.batch &&
          dim_or_sentinel(value_sizes, 0) == spec.batch
      ? spec.batch
      : -1;
  const int64_t qkv_head_dim =
      dim_or_sentinel(query_sizes, 3) == spec.head_dim &&
          dim_or_sentinel(key_sizes, 3) == spec.head_dim &&
          dim_or_sentinel(value_sizes, 3) == spec.head_dim
      ? spec.head_dim
      : -1;
  if (
      !generated::masked_tiny_sdpa_additive_float_mask_options_match(
          spec,
          query_dtype,
          key_dtype,
          value_dtype,
          attn_mask_dtype,
          static_cast<int64_t>(query_sizes.size()),
          static_cast<int64_t>(key_sizes.size()),
          static_cast<int64_t>(value_sizes.size()),
          static_cast<int64_t>(attn_mask_sizes.size()),
          qkv_batch,
          dim_or_sentinel(query_sizes, 1),
          dim_or_sentinel(key_sizes, 1),
          dim_or_sentinel(value_sizes, 1),
          dim_or_sentinel(query_sizes, 2),
          dim_or_sentinel(key_sizes, 2),
          dim_or_sentinel(value_sizes, 2),
          qkv_head_dim,
          dim_or_sentinel(attn_mask_sizes, 0),
          dim_or_sentinel(attn_mask_sizes, 1),
          dim_or_sentinel(attn_mask_sizes, 2),
          dim_or_sentinel(attn_mask_sizes, 3),
          has_attn_mask,
          dropout_p == 0.0,
          is_causal,
          enable_gqa,
          scale_equivalent_head_dim64,
          true,
          true)) {
    return result;
  }
  result.matched = true;
  result.family = MaskedTinySDPAFamily::AdditiveFloatMask;
  result.tuple_id = spec.tuple_id;
  result.metadata = &kMaskedTinySDPAAdditiveFloatMaskMetadata;
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
