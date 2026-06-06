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

constexpr const char* kFallbackUnsupportedShapesHardFailOrDoNotMatch =
    "unsupported_shapes_hard_fail_or_do_not_match";
constexpr const char* kMaterializationNone = "none";
constexpr const char* kSDPAScoreSoftmaxDiffusionSquareScoresTupleId =
    "heads1_or5_sequence504_or640_float_rank3_square";
constexpr ExecutionContractMetadata
    kSDPAScoreSoftmaxDiffusionSquareScoresMetadata =
        make_execution_contract_metadata(
            "SDPAScoreSoftmaxContract",
            "DiffusionSquareScores",
            kSDPAScoreSoftmaxDiffusionSquareScoresTupleId,
            "sdpa_score_softmax_focused_tests",
            "sdpa_score_softmax_adjacent_guards",
            kFallbackUnsupportedShapesHardFailOrDoNotMatch,
            kMaterializationNone);

} // namespace

SDPAScoreSoftmaxMatch match_sdpa_buffer_softmax_score_contract(
    const IntArrayRef input_sizes,
    const ScalarType input_dtype,
    const int64_t dim) {
  SDPAScoreSoftmaxMatch result;
  if (
      input_dtype != kFloat || input_sizes.size() != 3 ||
      dim != static_cast<int64_t>(input_sizes.size()) - 1 ||
      input_sizes[1] != input_sizes[2]) {
    return result;
  }
  const int64_t heads = input_sizes[0];
  const int64_t sequence = input_sizes[1];
  if ((heads == 1 || heads == 5) && (sequence == 504 || sequence == 640)) {
    result.matched = true;
    result.family = SDPAScoreSoftmaxFamily::DiffusionSquareScores;
    result.tuple_id = kSDPAScoreSoftmaxDiffusionSquareScoresTupleId;
    result.metadata = &kSDPAScoreSoftmaxDiffusionSquareScoresMetadata;
  }
  return result;
}

bool matches_sdpa_buffer_softmax_score_contract(
    const IntArrayRef input_sizes,
    const ScalarType input_dtype,
    const int64_t dim) {
  return match_sdpa_buffer_softmax_score_contract(input_sizes, input_dtype, dim)
      .matched;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
