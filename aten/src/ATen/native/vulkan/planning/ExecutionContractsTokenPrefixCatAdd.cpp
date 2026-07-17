#include <ATen/native/vulkan/planning/DynamicProgramRuntime.h>
#include <ATen/native/vulkan/planning/ExecutionContracts.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

constexpr ExecutionContractMetadata kTokenPrefixCatAddDynamicMetadata{
    "TokenPrefixCatAddDirectBuffer",
    "GenericRuntimeShape",
    "token_prefix_cat_add_generic_runtime_shape",
    "dynamic_token_prefix_cat_add_random_shape_tests",
    "token_prefix_cat_add_semantic_guards",
    "unsupported_semantics_do_not_match",
    "fused_cat_add_real_contiguous_output"};

} // namespace

const ExecutionContractMetadata* match_token_prefix_cat_add_direct_buffer(
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
  if (!prefix_is_vulkan || !tokens_is_vulkan || !pos_is_vulkan) {
    return nullptr;
  }

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
  return decision.runtime_selection_authorized
      ? &kTokenPrefixCatAddDynamicMetadata
      : nullptr;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
