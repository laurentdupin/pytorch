#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/DynamicProgramRuntime.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsGQARepeatSpec.h>

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

constexpr ExecutionContractMetadata kGQARepeatMetadata =
    make_execution_contract_metadata(
        generated::kGQARepeatBatch1Heads4Factor4Sequence100To116Dim128Spec
            .contract_name,
        generated::kGQARepeatBatch1Heads4Factor4Sequence100To116Dim128Spec
            .family_name,
        generated::kGQARepeatBatch1Heads4Factor4Sequence100To116Dim128Spec
            .tuple_id,
        generated::kGQARepeatBatch1Heads4Factor4Sequence100To116Dim128Spec
            .evidence_id,
        generated::kGQARepeatBatch1Heads4Factor4Sequence100To116Dim128Spec
            .guard_id,
        generated::kGQARepeatBatch1Heads4Factor4Sequence100To116Dim128Spec
            .fallback_policy,
        generated::kGQARepeatBatch1Heads4Factor4Sequence100To116Dim128Spec
            .materialization_policy);

constexpr ExecutionContractMetadata kGQARepeatDirectBufferMetadata =
    make_execution_contract_metadata(
        "GQARepeatContract",
        "GenericRuntimeShape",
        "gqa_repeat_direct_buffer_runtime_shape",
        "dynamic_gqa_repeat_random_shape_tests",
        "gqa_repeat_semantic_guards",
        "unsupported_semantics_hard_fail",
        "gqa_repeat_buffer_materialization");

} // namespace

GQARepeatMatch match_gqa_repeat_contract(
    const IntArrayRef tensor_sizes,
    const ScalarType tensor_dtype,
    const bool tensor_is_vulkan,
    const bool tensor_has_buffer_storage,
    const int64_t repeat_factor) {
  GQARepeatMatch result;
  const auto& spec =
      generated::kGQARepeatBatch1Heads4Factor4Sequence100To116Dim128Spec;
  if (
      generated::gqa_repeat_batch_1_heads_4_factor_4_sequence_100_to_116_dim_128_options_match(
          spec,
          tensor_dtype,
          static_cast<int64_t>(tensor_sizes.size()),
          tensor_sizes.size() > 0 ? tensor_sizes[0] : -1,
          tensor_sizes.size() > 1 ? tensor_sizes[1] : -1,
          spec.target_heads,
          repeat_factor,
          spec.target_sequence,
          tensor_sizes.size() > 3 ? tensor_sizes[3] : -1,
          tensor_is_vulkan,
          tensor_has_buffer_storage,
          spec.enable_gqa) &&
      tensor_sizes[2] >= spec.min_source_sequence &&
      generated::gqa_repeat_batch_1_heads_4_factor_4_sequence_100_to_116_dim_128_in_bounds(
          spec,
          tensor_sizes[2])) {
    result.matched = true;
    result.tuple_id = spec.tuple_id;
    result.metadata = &kGQARepeatMetadata;
    result.sequence_length = tensor_sizes[2];
    return result;
  }

  const DynamicProgramDecision decision = build_dynamic_program_runtime_plan(
      make_gqa_repeat_direct_buffer_dynamic_program(
          tensor_sizes,
          tensor_dtype,
          tensor_is_vulkan && tensor_has_buffer_storage,
          repeat_factor,
          &kGQARepeatDirectBufferMetadata,
          /*behavior_enabled=*/true));
  if (!decision.runtime_selection_authorized) {
    return result;
  }
  result.matched = true;
  result.tuple_id = kGQARepeatDirectBufferMetadata.tuple_id;
  result.metadata = &kGQARepeatDirectBufferMetadata;
  result.sequence_length = tensor_sizes.size() == 4 ? tensor_sizes[2] : 0;
  return result;
}

bool matches_gqa_repeat_contract(
    const IntArrayRef tensor_sizes,
    const ScalarType tensor_dtype,
    const bool tensor_is_vulkan,
    const bool tensor_has_buffer_storage,
    const int64_t repeat_factor) {
  return match_gqa_repeat_contract(
             tensor_sizes,
             tensor_dtype,
             tensor_is_vulkan,
             tensor_has_buffer_storage,
             repeat_factor)
      .matched;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
