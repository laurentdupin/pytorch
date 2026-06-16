#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsVisionSelfAttentionSDPASpec.h>

#include <cmath>
#include <string_view>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

VisionSelfAttentionSDPAFamily vision_self_attention_sdpa_family_from_name(
    const char* const family_name) {
  const std::string_view family{family_name};
  if (family == "Rank3Head64Scale1") {
    return VisionSelfAttentionSDPAFamily::Rank3Head64Scale1;
  }
  return VisionSelfAttentionSDPAFamily::None;
}

const generated::VisionSelfAttentionSDPAAttentionRowsRow*
find_vision_self_attention_sdpa_row(
    const int64_t batch_heads,
    const int64_t query_sequence,
    const int64_t key_value_sequence,
    const int64_t head_dim) {
  for (const auto& row : generated::kVisionSelfAttentionSDPAAttentionRowsRows) {
    if (generated::vision_self_attention_sdpa_attention_rows_row_matches(
            row,
            "Rank3Head64Scale1",
            batch_heads,
            query_sequence,
            key_value_sequence,
            head_dim)) {
      return &row;
    }
  }
  return nullptr;
}

} // namespace

const char* vision_self_attention_sdpa_route_label(
    const VisionSelfAttentionSDPAFamily family) {
  switch (family) {
    case VisionSelfAttentionSDPAFamily::Rank3Head64Scale1:
      return "SelectedVisionSelfAttentionSDPA";
    case VisionSelfAttentionSDPAFamily::None:
      return "SelectedVisionSelfAttentionSDPANone";
  }
  return "SelectedVisionSelfAttentionSDPANone";
}

VisionSelfAttentionSDPAMatch match_vision_self_attention_sdpa_contract(
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
  VisionSelfAttentionSDPAMatch result;
  if (
      has_attn_mask || dropout_p != 0.0 || is_causal || enable_gqa ||
      query_dtype != kFloat || key_dtype != kFloat || value_dtype != kFloat ||
      query_sizes.size() != 3 || key_sizes.size() != 3 ||
      value_sizes.size() != 3) {
    return result;
  }
  if (!scale.has_value() || std::abs(*scale - 1.0) > 1.0e-9) {
    return result;
  }
  if (
      query_sizes[0] != key_sizes[0] || query_sizes[0] != value_sizes[0] ||
      query_sizes[1] != key_sizes[1] || query_sizes[1] != value_sizes[1] ||
      query_sizes[2] != key_sizes[2] || query_sizes[2] != value_sizes[2]) {
    return result;
  }

  const auto* const row = find_vision_self_attention_sdpa_row(
      query_sizes[0], query_sizes[1], key_sizes[1], query_sizes[2]);
  if (row == nullptr) {
    return result;
  }
  result.matched = true;
  result.family = vision_self_attention_sdpa_family_from_name(row->family);
  result.tuple_id = row->tuple_id;
  result.metadata = &row->metadata;
  return result;
}

bool matches_vision_self_attention_sdpa_contract(
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
  return match_vision_self_attention_sdpa_contract(
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
