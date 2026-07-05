#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/DynamicProgramRuntime.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsTokenPrefixCatAddSpec.h>

#include <string_view>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

constexpr const char* kTokenPrefixCatAddObservedFamily =
    "Prefix1TokenCountSetFeatureSetAdd";

constexpr ExecutionContractMetadata kTokenPrefixCatAddDynamicMetadata{
    "TokenPrefixCatAddContract",
    "GenericPrefix1Dim1BufferAdd",
    "token_prefix_cat_add_generic_runtime_shape",
    "dynamic_token_prefix_cat_add_random_shape_tests",
    "token_prefix_cat_add_semantic_guards",
    "unsupported_semantics_do_not_match",
    "fused_cat_add_real_contiguous_output"};

TokenPrefixCatAddFamily token_prefix_cat_add_family_from_name(
    const char* const family_name) {
  const std::string_view family{family_name};
  if (family == kTokenPrefixCatAddObservedFamily) {
    return TokenPrefixCatAddFamily::Prefix1TokenCountSetFeatureSetAdd;
  }
  if (family == "GenericPrefix1Dim1BufferAdd") {
    return TokenPrefixCatAddFamily::GenericPrefix1Dim1BufferAdd;
  }
  return TokenPrefixCatAddFamily::None;
}

} // namespace

const char* token_prefix_cat_add_family_name(
    const TokenPrefixCatAddFamily family) {
  switch (family) {
    case TokenPrefixCatAddFamily::Prefix1TokenCountSetFeatureSetAdd:
      return kTokenPrefixCatAddObservedFamily;
    case TokenPrefixCatAddFamily::GenericPrefix1Dim1BufferAdd:
      return "GenericPrefix1Dim1BufferAdd";
    case TokenPrefixCatAddFamily::None:
      return "None";
  }
  return "None";
}

TokenPrefixCatAddMatch match_token_prefix_cat_add_contract(
    const IntArrayRef prefix_sizes,
    const IntArrayRef token_sizes,
    const IntArrayRef pos_sizes,
    const ScalarType prefix_dtype,
    const ScalarType token_dtype,
    const ScalarType pos_dtype,
    const bool prefix_is_vulkan,
    const bool tokens_is_vulkan,
    const bool pos_is_vulkan,
    const int64_t dim,
    const bool inplace,
    const bool alias_output) {
  TokenPrefixCatAddMatch result;
  if (
      !prefix_is_vulkan || !tokens_is_vulkan || !pos_is_vulkan ||
      prefix_dtype != kFloat || token_dtype != kFloat || pos_dtype != kFloat ||
      prefix_sizes.size() != 3 || token_sizes.size() != 3 ||
      pos_sizes.size() != 3 || dim != 1 || inplace || alias_output) {
    return result;
  }
  if (
      prefix_sizes[0] <= 0 || prefix_sizes[1] != 1 ||
      token_sizes[0] != prefix_sizes[0] || pos_sizes[0] != prefix_sizes[0] ||
      token_sizes[2] != prefix_sizes[2] || pos_sizes[2] != prefix_sizes[2] ||
      pos_sizes[1] != token_sizes[1] + prefix_sizes[1]) {
    return result;
  }

  const auto* const row = generated::token_prefix_cat_add_token_rows_find(
      kTokenPrefixCatAddObservedFamily,
      token_sizes[1],
      prefix_sizes[2],
      pos_sizes[1]);
  if (row == nullptr) {
    const DynamicProgramDecision decision = build_dynamic_program_runtime_plan(
        make_token_prefix_cat_add_direct_buffer_dynamic_program(
            prefix_sizes,
            token_sizes,
            pos_sizes,
            prefix_dtype,
            token_dtype,
            pos_dtype,
            /*prefix_buffer_storage=*/true,
            /*token_buffer_storage=*/true,
            /*pos_buffer_storage=*/true,
            dim,
            inplace,
            alias_output,
            &kTokenPrefixCatAddDynamicMetadata,
            /*behavior_enabled=*/true));
    if (!decision.runtime_selection_authorized) {
      return result;
    }

    result.matched = true;
    result.family = TokenPrefixCatAddFamily::GenericPrefix1Dim1BufferAdd;
    result.tuple_id = kTokenPrefixCatAddDynamicMetadata.tuple_id;
    result.metadata = &kTokenPrefixCatAddDynamicMetadata;
    result.token_count = token_sizes[1];
    result.feature_dim = prefix_sizes[2];
    result.total_tokens = pos_sizes[1];
    return result;
  }

  result.matched = true;
  result.family = token_prefix_cat_add_family_from_name(row->family);
  result.tuple_id = row->tuple_id;
  result.metadata = &row->metadata;
  result.token_count = row->tokens;
  result.feature_dim = row->feature_dim;
  result.total_tokens = row->total_tokens;
  return result;
}

bool matches_token_prefix_cat_add_contract(
    const IntArrayRef prefix_sizes,
    const IntArrayRef token_sizes,
    const IntArrayRef pos_sizes,
    const ScalarType prefix_dtype,
    const ScalarType token_dtype,
    const ScalarType pos_dtype,
    const bool prefix_is_vulkan,
    const bool tokens_is_vulkan,
    const bool pos_is_vulkan,
    const int64_t dim,
    const bool inplace,
    const bool alias_output) {
  return match_token_prefix_cat_add_contract(
             prefix_sizes,
             token_sizes,
             pos_sizes,
             prefix_dtype,
             token_dtype,
             pos_dtype,
             prefix_is_vulkan,
             tokens_is_vulkan,
             pos_is_vulkan,
             dim,
             inplace,
             alias_output)
      .matched;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
