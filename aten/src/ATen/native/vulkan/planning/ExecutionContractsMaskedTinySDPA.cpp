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

constexpr ExecutionContractMetadata kMaskedTinySDPARuntimeMetadata =
    make_execution_contract_metadata(
        "MaskedTinySDPAContract",
        "AdditiveFloatMaskRuntimeShape",
        "masked_tiny_additive_float_mask_runtime_shape",
        "masked_tiny_sdpa_dynamic_random_shape_tests",
        "masked_tiny_sdpa_dynamic_semantic_guards",
        "fallback_on_unsupported_layout_or_semantics",
        "runtime_math_path");

constexpr int64_t kRuntimeMaskedTinyMaxBatchHeads = 64;
constexpr int64_t kRuntimeMaskedTinyMaxSequence = 64;
constexpr int64_t kRuntimeMaskedTinyMaxHeadDim = 128;
constexpr int64_t kRuntimeMaskedTinyMaxValueDim = 128;
constexpr int64_t kRuntimeMaskedTinyMaxScoreElements = 65536;

int64_t dim_or_sentinel(const IntArrayRef sizes, const size_t dim) {
  return sizes.size() > dim ? sizes[dim] : -1;
}

bool mask_broadcasts_to_attention_scores(
    const IntArrayRef mask_sizes,
    const int64_t batch,
    const int64_t heads,
    const int64_t target_len,
    const int64_t source_len) {
  if (mask_sizes.size() == 2) {
    return mask_sizes[0] == target_len && mask_sizes[1] == source_len;
  }
  if (mask_sizes.size() == 3) {
    return (mask_sizes[0] == 1 || mask_sizes[0] == batch ||
            mask_sizes[0] == batch * heads) &&
        mask_sizes[1] == target_len && mask_sizes[2] == source_len;
  }
  if (mask_sizes.size() == 4) {
    return (mask_sizes[0] == 1 || mask_sizes[0] == batch) &&
        (mask_sizes[1] == 1 || mask_sizes[1] == heads) &&
        mask_sizes[2] == target_len && mask_sizes[3] == source_len;
  }
  return false;
}

bool is_runtime_additive_float_mask_shape(
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const IntArrayRef value_sizes,
    const IntArrayRef attn_mask_sizes,
    const std::optional<double> scale) {
  if (
      query_sizes.size() != key_sizes.size() ||
      query_sizes.size() != value_sizes.size() ||
      (query_sizes.size() != 3 && query_sizes.size() != 4)) {
    return false;
  }
  if (scale.has_value() && !std::isfinite(*scale)) {
    return false;
  }

  const int64_t batch = query_sizes[0];
  const int64_t heads = query_sizes.size() == 4 ? query_sizes[1] : 1;
  const int64_t query_sequence =
      query_sizes[query_sizes.size() == 4 ? 2 : 1];
  const int64_t key_sequence = key_sizes[key_sizes.size() == 4 ? 2 : 1];
  const int64_t value_sequence = value_sizes[value_sizes.size() == 4 ? 2 : 1];
  const int64_t head_dim = query_sizes[query_sizes.size() == 4 ? 3 : 2];
  const int64_t key_dim = key_sizes[key_sizes.size() == 4 ? 3 : 2];
  const int64_t value_dim = value_sizes[value_sizes.size() == 4 ? 3 : 2];
  if (
      batch <= 0 || heads <= 0 || query_sequence <= 0 || key_sequence <= 0 ||
      value_sequence <= 0 || head_dim <= 0 || key_dim <= 0 ||
      value_dim <= 0) {
    return false;
  }
  if (
      key_sizes[0] != batch || value_sizes[0] != batch ||
      (query_sizes.size() == 4 &&
       (key_sizes[1] != heads || value_sizes[1] != heads)) ||
      key_sequence != value_sequence || head_dim != key_dim) {
    return false;
  }
  if (
      batch * heads > kRuntimeMaskedTinyMaxBatchHeads ||
      query_sequence > kRuntimeMaskedTinyMaxSequence ||
      key_sequence > kRuntimeMaskedTinyMaxSequence ||
      head_dim > kRuntimeMaskedTinyMaxHeadDim ||
      value_dim > kRuntimeMaskedTinyMaxValueDim ||
      batch * heads * query_sequence * key_sequence >
          kRuntimeMaskedTinyMaxScoreElements) {
    return false;
  }
  return mask_broadcasts_to_attention_scores(
      attn_mask_sizes, batch, heads, query_sequence, key_sequence);
}

} // namespace

const char* masked_tiny_sdpa_route_label(
    const MaskedTinySDPAFamily family) {
  switch (family) {
    case MaskedTinySDPAFamily::AdditiveFloatMask:
      return "SelectedMaskedTinySDPAAdditiveFloatMask";
    case MaskedTinySDPAFamily::AdditiveFloatMaskRuntimeShape:
      return "SelectedMaskedTinySDPAAdditiveFloatMaskRuntimeShape";
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
    if (
        has_attn_mask && dropout_p == 0.0 && !is_causal && !enable_gqa &&
        query_dtype == kFloat && key_dtype == kFloat &&
        value_dtype == kFloat && attn_mask_dtype == kFloat &&
        is_runtime_additive_float_mask_shape(
            query_sizes, key_sizes, value_sizes, attn_mask_sizes, scale)) {
      result.matched = true;
      result.family = MaskedTinySDPAFamily::AdditiveFloatMaskRuntimeShape;
      result.tuple_id = kMaskedTinySDPARuntimeMetadata.tuple_id;
      result.metadata = &kMaskedTinySDPARuntimeMetadata;
      return result;
    }
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
