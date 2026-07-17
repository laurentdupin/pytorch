#include <ATen/native/vulkan/planning/DynamicProgramRuntime.h>
#include <ATen/native/vulkan/planning/ExecutionContracts.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

constexpr ExecutionContractMetadata kFeatureMapToTokensDirectBufferMetadata{
    "FeatureMapToTokensDirectBuffer",
    "GenericRuntimeShape",
    "feature_map_to_tokens_direct_buffer_generic_runtime_shape",
    "dynamic_feature_map_to_tokens_random_shape_tests",
    "feature_map_to_tokens_semantic_guards",
    "unsupported_semantics_hard_fail",
    "device_layout_transition_feature_map_to_tokens"};

} // namespace

const ExecutionContractMetadata* match_feature_map_to_tokens_direct_buffer(
    const IntArrayRef feature_map_sizes,
    const ScalarType dtype,
    const bool is_vulkan,
    const bool has_buffer_storage,
    const bool is_width_packed,
    const bool has_zero_storage_offset,
    const bool supports_buffer_compute) {
  if (!is_vulkan) {
    return nullptr;
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
  return decision.runtime_selection_authorized
      ? &kFeatureMapToTokensDirectBufferMetadata
      : nullptr;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
