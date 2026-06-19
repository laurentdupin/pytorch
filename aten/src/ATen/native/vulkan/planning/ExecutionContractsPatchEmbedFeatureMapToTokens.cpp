#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsPatchEmbedFeatureMapToTokensSpec.h>

#include <string_view>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

constexpr const char* kPatchEmbedFeatureMapObservedFamily =
    "Kernel14Stride14ObservedFeatureMap";

PatchEmbedFeatureMapToTokensFamily
patch_embed_feature_map_to_tokens_family_from_name(
    const char* const family_name) {
  const std::string_view family{family_name};
  if (family == kPatchEmbedFeatureMapObservedFamily) {
    return PatchEmbedFeatureMapToTokensFamily::
        Kernel14Stride14ObservedFeatureMap;
  }
  return PatchEmbedFeatureMapToTokensFamily::None;
}

} // namespace

const char* patch_embed_feature_map_to_tokens_family_name(
    const PatchEmbedFeatureMapToTokensFamily family) {
  switch (family) {
    case PatchEmbedFeatureMapToTokensFamily::
        Kernel14Stride14ObservedFeatureMap:
      return kPatchEmbedFeatureMapObservedFamily;
    case PatchEmbedFeatureMapToTokensFamily::None:
      return "None";
  }
  return "None";
}

PatchEmbedFeatureMapToTokensMatch match_patch_embed_feature_map_to_tokens_contract(
    const IntArrayRef feature_map_sizes,
    const ScalarType dtype,
    const bool is_vulkan,
    const bool has_buffer_storage,
    const bool is_width_packed,
    const bool has_zero_storage_offset,
    const bool supports_buffer_compute) {
  PatchEmbedFeatureMapToTokensMatch result;
  if (
      !is_vulkan || dtype != kFloat || feature_map_sizes.size() != 4 ||
      !has_buffer_storage || !is_width_packed || !has_zero_storage_offset ||
      !supports_buffer_compute || feature_map_sizes[0] != 1) {
    return result;
  }

  const int64_t token_count = feature_map_sizes[2] * feature_map_sizes[3];
  const auto* const row =
      generated::patch_embed_feature_map_to_tokens_feature_map_rows_find(
          kPatchEmbedFeatureMapObservedFamily,
          feature_map_sizes[1],
          feature_map_sizes[2],
          feature_map_sizes[3],
          token_count);
  if (row == nullptr) {
    return result;
  }

  result.matched = true;
  result.family =
      patch_embed_feature_map_to_tokens_family_from_name(row->family);
  result.tuple_id = row->tuple_id;
  result.metadata = &row->metadata;
  result.channels = row->channels;
  result.feature_h = row->feature_h;
  result.feature_w = row->feature_w;
  result.token_count = row->tokens;
  return result;
}

bool matches_patch_embed_feature_map_to_tokens_contract(
    const IntArrayRef feature_map_sizes,
    const ScalarType dtype,
    const bool is_vulkan,
    const bool has_buffer_storage,
    const bool is_width_packed,
    const bool has_zero_storage_offset,
    const bool supports_buffer_compute) {
  return match_patch_embed_feature_map_to_tokens_contract(
             feature_map_sizes,
             dtype,
             is_vulkan,
             has_buffer_storage,
             is_width_packed,
             has_zero_storage_offset,
             supports_buffer_compute)
      .matched;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
