#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/DynamicProgramRuntime.h>
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

constexpr ExecutionContractMetadata kFeatureMapToTokensDirectBufferMetadata{
    "PatchEmbedFeatureMapToTokensContract",
    "GenericDirectBuffer",
    "feature_map_to_tokens_direct_buffer_generic_runtime_shape",
    "dynamic_feature_map_to_tokens_random_shape_tests",
    "feature_map_to_tokens_semantic_guards",
    "unsupported_semantics_hard_fail",
    "device_layout_transition_feature_map_to_tokens"};

PatchEmbedFeatureMapToTokensFamily
patch_embed_feature_map_to_tokens_family_from_name(
    const char* const family_name) {
  const std::string_view family{family_name};
  if (family == kPatchEmbedFeatureMapObservedFamily) {
    return PatchEmbedFeatureMapToTokensFamily::
        Kernel14Stride14ObservedFeatureMap;
  }
  if (family == "GenericDirectBuffer") {
    return PatchEmbedFeatureMapToTokensFamily::GenericDirectBuffer;
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
    case PatchEmbedFeatureMapToTokensFamily::GenericDirectBuffer:
      return "GenericDirectBuffer";
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
      !has_buffer_storage || !supports_buffer_compute) {
    return result;
  }

  const int64_t token_count = feature_map_sizes[2] * feature_map_sizes[3];
  if (
      is_width_packed && has_zero_storage_offset &&
      feature_map_sizes[0] == 1) {
    const auto* const row =
        generated::patch_embed_feature_map_to_tokens_feature_map_rows_find(
            kPatchEmbedFeatureMapObservedFamily,
            feature_map_sizes[1],
            feature_map_sizes[2],
            feature_map_sizes[3],
            token_count);
    if (row != nullptr) {
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
  }

  const DynamicProgramDecision decision = build_dynamic_program_runtime_plan(
      make_feature_map_to_tokens_direct_buffer_dynamic_program(
          feature_map_sizes,
          dtype,
          /*input_direct_buffer=*/
          has_buffer_storage && is_width_packed && has_zero_storage_offset &&
              supports_buffer_compute,
          /*output_direct_buffer=*/true,
          &kFeatureMapToTokensDirectBufferMetadata,
          /*behavior_enabled=*/true));
  if (!decision.runtime_selection_authorized) {
    return result;
  }

  result.matched = true;
  result.family = PatchEmbedFeatureMapToTokensFamily::GenericDirectBuffer;
  result.tuple_id = kFeatureMapToTokensDirectBufferMetadata.tuple_id;
  result.metadata = &kFeatureMapToTokensDirectBufferMetadata;
  result.channels = feature_map_sizes[1];
  result.feature_h = feature_map_sizes[2];
  result.feature_w = feature_map_sizes[3];
  result.token_count = token_count;
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
