#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsDiffusionSDPASpec.h>

#include <cmath>
#include <string_view>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

constexpr double kHeadDim64Scale = 0.125;
constexpr double kHeadDim512Scale = 0.04419417382415922;
constexpr int64_t kRuntimeDiffusionSquareMaxHeads = 32;
constexpr int64_t kRuntimeDiffusionSquareMaxSequence = 640;
constexpr int64_t kRuntimeDiffusionSquareMaxScoreElements = 2097152;
constexpr int64_t kRuntimeDiffusionCrossMaxHeads = 32;
constexpr int64_t kRuntimeDiffusionCrossMaxQuerySequence = 512;
constexpr int64_t kRuntimeDiffusionCrossMaxKeyValueSequence = 8;
constexpr int64_t kRuntimeDiffusionCrossMaxScoreElements = 65536;

constexpr ExecutionContractMetadata kDiffusionSquareRuntimeMetadata{
    "DiffusionSDPAContract",
    "SquareSelfAttentionRuntimeShape",
    "diffusion_square_self_attention_runtime_shape",
    "diffusion_square_sdpa_dynamic_random_shape_tests",
    "diffusion_square_sdpa_dynamic_semantic_guards",
    "fallback_on_unsupported_layout_or_semantics",
    "delegated_to_sdpa_execution_policy"};

constexpr ExecutionContractMetadata kDiffusionCrossRuntimeMetadata{
    "DiffusionSDPAContract",
    "CrossAttentionRuntimeShape",
    "diffusion_cross_attention_runtime_shape",
    "diffusion_cross_attention_dynamic_random_shape_tests",
    "diffusion_cross_attention_dynamic_semantic_guards",
    "fallback_on_unsupported_layout_or_semantics",
    "delegated_to_sdpa_execution_policy"};

DiffusionSDPAFamily diffusion_sdpa_family_from_name(
    const char* const family_name) {
  const std::string_view family{family_name};
  if (family == "SquareSelfAttention") {
    return DiffusionSDPAFamily::SquareSelfAttention;
  }
  if (family == "SquareSelfAttentionRuntimeShape") {
    return DiffusionSDPAFamily::SquareSelfAttentionRuntimeShape;
  }
  if (family == "CrossAttention") {
    return DiffusionSDPAFamily::CrossAttention;
  }
  return DiffusionSDPAFamily::None;
}

const generated::DiffusionSDPAAttentionRowsRow* find_diffusion_sdpa_row(
    const int64_t heads,
    const int64_t query_sequence,
    const int64_t key_value_sequence,
    const int64_t head_dim) {
  for (const auto& row : generated::kDiffusionSDPAAttentionRowsRows) {
    if (generated::diffusion_sdpa_attention_rows_row_matches(
            row, heads, query_sequence, key_value_sequence, head_dim)) {
      return &row;
    }
  }
  return nullptr;
}

bool scale_matches_head_dim(
    const std::optional<double> scale,
    const int64_t head_dim) {
  if (!scale.has_value()) {
    return true;
  }
  const double expected_scale =
      head_dim == 512 ? kHeadDim512Scale : kHeadDim64Scale;
  return std::abs(*scale - expected_scale) <= 1.0e-6;
}

bool is_runtime_cross_attention_shape(
    const int64_t heads,
    const int64_t query_sequence,
    const int64_t key_value_sequence,
    const int64_t head_dim,
    const std::optional<double> scale) {
  return heads > 0 && heads <= kRuntimeDiffusionCrossMaxHeads &&
      query_sequence > 0 &&
      query_sequence <= kRuntimeDiffusionCrossMaxQuerySequence &&
      key_value_sequence > 0 &&
      key_value_sequence <= kRuntimeDiffusionCrossMaxKeyValueSequence &&
      query_sequence != key_value_sequence && head_dim == 64 &&
      heads * query_sequence * key_value_sequence <=
          kRuntimeDiffusionCrossMaxScoreElements &&
      scale_matches_head_dim(scale, head_dim);
}

bool is_runtime_square_attention_shape(
    const int64_t heads,
    const int64_t query_sequence,
    const int64_t key_value_sequence,
    const int64_t head_dim,
    const std::optional<double> scale) {
  const bool supported_head_dim =
      head_dim == 64 ||
      (heads == 1 && head_dim == 512 && query_sequence % 4 == 0);
  return heads > 0 && heads <= kRuntimeDiffusionSquareMaxHeads &&
      query_sequence > 0 &&
      query_sequence <= kRuntimeDiffusionSquareMaxSequence &&
      query_sequence == key_value_sequence && supported_head_dim &&
      heads * query_sequence * key_value_sequence <=
          kRuntimeDiffusionSquareMaxScoreElements &&
      scale_matches_head_dim(scale, head_dim);
}

} // namespace

const char* diffusion_sdpa_route_label(const DiffusionSDPAFamily family) {
  switch (family) {
    case DiffusionSDPAFamily::SquareSelfAttention:
      return "SelectedDiffusionSDPASquareSelfAttention";
    case DiffusionSDPAFamily::SquareSelfAttentionRuntimeShape:
      return "SelectedDiffusionSDPASquareSelfAttentionRuntimeShape";
    case DiffusionSDPAFamily::CrossAttention:
      return "SelectedDiffusionSDPACrossAttention";
    case DiffusionSDPAFamily::CrossAttentionRuntimeShape:
      return "SelectedDiffusionSDPACrossAttentionRuntimeShape";
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
  const auto* const row = find_diffusion_sdpa_row(
      heads, query_sequence, key_value_sequence, head_dim);
  if (row != nullptr) {
    if (!scale_matches_head_dim(scale, head_dim)) {
      return result;
    }
    result.matched = true;
    result.family = diffusion_sdpa_family_from_name(row->family);
    result.tuple_id = row->tuple_id;
    result.metadata = &row->metadata;
    return result;
  }
  if (is_runtime_square_attention_shape(
          heads, query_sequence, key_value_sequence, head_dim, scale)) {
    result.matched = true;
    result.family = DiffusionSDPAFamily::SquareSelfAttentionRuntimeShape;
    result.tuple_id = kDiffusionSquareRuntimeMetadata.tuple_id;
    result.metadata = &kDiffusionSquareRuntimeMetadata;
    return result;
  }
  if (is_runtime_cross_attention_shape(
          heads, query_sequence, key_value_sequence, head_dim, scale)) {
    result.matched = true;
    result.family = DiffusionSDPAFamily::CrossAttentionRuntimeShape;
    result.tuple_id = kDiffusionCrossRuntimeMetadata.tuple_id;
    result.metadata = &kDiffusionCrossRuntimeMetadata;
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

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
