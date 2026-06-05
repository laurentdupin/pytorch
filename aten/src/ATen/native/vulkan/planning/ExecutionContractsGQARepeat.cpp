#include <ATen/native/vulkan/planning/ExecutionContracts.h>

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

constexpr const char* kFallbackUnsupportedShapesDoNotMatch =
    "unsupported_shapes_do_not_match";
constexpr const char* kMaterializationGQARepeatBuffer =
    "gqa_repeat_buffer_materialization";

constexpr int64_t kGQARepeatBatch = 1;
constexpr int64_t kGQARepeatSourceHeads = 4;
constexpr int64_t kGQARepeatFactor = 4;
constexpr int64_t kGQARepeatMinSequence = 100;
constexpr int64_t kGQARepeatMaxSequence = 116;
constexpr int64_t kGQARepeatHeadDim = 128;
constexpr const char* kGQARepeatTupleId =
    "gqa_repeat_batch1_heads4_factor4_sequence100_to_116_dim128";
constexpr ExecutionContractMetadata kGQARepeatMetadata =
    make_execution_contract_metadata(
        "GQARepeatContract",
        "Batch1Heads4Factor4Sequence100To116Dim128",
        kGQARepeatTupleId,
        "gqa_repeat_focused_tests",
        "gqa_repeat_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationGQARepeatBuffer);

} // namespace

GQARepeatMatch match_gqa_repeat_contract(
    const IntArrayRef tensor_sizes,
    const ScalarType tensor_dtype,
    const bool tensor_is_vulkan,
    const bool tensor_has_buffer_storage,
    const int64_t repeat_factor) {
  GQARepeatMatch result;
  if (
      !tensor_is_vulkan || !tensor_has_buffer_storage ||
      tensor_dtype != kFloat || tensor_sizes.size() != 4 ||
      repeat_factor != kGQARepeatFactor ||
      tensor_sizes[0] != kGQARepeatBatch ||
      tensor_sizes[1] != kGQARepeatSourceHeads ||
      tensor_sizes[2] < kGQARepeatMinSequence ||
      tensor_sizes[2] > kGQARepeatMaxSequence ||
      tensor_sizes[3] != kGQARepeatHeadDim) {
    return result;
  }
  result.matched = true;
  result.tuple_id = kGQARepeatTupleId;
  result.metadata = &kGQARepeatMetadata;
  result.sequence_length = tensor_sizes[2];
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
