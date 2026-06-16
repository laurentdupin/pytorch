#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsSDPAScoreSoftmaxSpec.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsVisionSelfAttentionSDPASpec.h>

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

constexpr ExecutionContractMetadata
    kSDPAScoreSoftmaxDiffusionSquareScoresMetadata =
        make_execution_contract_metadata(
            generated::kSDPAScoreSoftmaxDiffusionSquareScoresSpec
                .contract_name,
            generated::kSDPAScoreSoftmaxDiffusionSquareScoresSpec.family_name,
            generated::kSDPAScoreSoftmaxDiffusionSquareScoresSpec.tuple_id,
            generated::kSDPAScoreSoftmaxDiffusionSquareScoresSpec.evidence_id,
            generated::kSDPAScoreSoftmaxDiffusionSquareScoresSpec.guard_id,
            generated::kSDPAScoreSoftmaxDiffusionSquareScoresSpec
                .fallback_policy,
            generated::kSDPAScoreSoftmaxDiffusionSquareScoresSpec
                .materialization_policy);

constexpr ExecutionContractMetadata
    kSDPAScoreSoftmaxVisionSelfAttentionScoresMetadata =
        make_execution_contract_metadata(
            "SDPAScoreSoftmaxContract",
            "VisionSelfAttentionScores",
            "vision_self_attention_rank3_score_rows",
            "dav2_vision_sdpa_prob_materialization_task296",
            "vision_self_attention_sdpa_adjacent_guards",
            "unsupported_shapes_hard_fail_or_do_not_match",
            "fresh_buffer_probability_materialization_before_value_bmm");

} // namespace

SDPAScoreSoftmaxMatch match_sdpa_buffer_softmax_score_contract(
    const IntArrayRef input_sizes,
    const ScalarType input_dtype,
    const int64_t dim) {
  SDPAScoreSoftmaxMatch result;
  const auto& spec = generated::kSDPAScoreSoftmaxDiffusionSquareScoresSpec;
  const int64_t rank = static_cast<int64_t>(input_sizes.size());
  const int64_t heads = input_sizes.size() > 0 ? input_sizes[0] : -1;
  const int64_t sequence = input_sizes.size() > 1 ? input_sizes[1] : -1;
  const bool square_scores =
      input_sizes.size() > 2 &&
      generated::sdpa_score_softmax_diffusion_square_scores_square_scores_equal(
          input_sizes[1], input_sizes[2]);
  if (generated::sdpa_score_softmax_diffusion_square_scores_options_match(
          spec,
          input_dtype,
          rank,
          spec.dim,
          dim,
          heads,
          sequence,
          square_scores,
          spec.requires_vulkan,
          spec.requires_buffer_storage)) {
    result.matched = true;
    result.family = SDPAScoreSoftmaxFamily::DiffusionSquareScores;
    result.tuple_id = spec.tuple_id;
    result.metadata = &kSDPAScoreSoftmaxDiffusionSquareScoresMetadata;
    return result;
  }

  if (input_sizes.size() == 3 && input_dtype == kFloat && dim == 2 &&
      input_sizes[1] == input_sizes[2]) {
    const auto* const row =
        generated::vision_self_attention_sdpa_attention_rows_find(
            "Rank3Head64Scale1",
            input_sizes[0],
            input_sizes[1],
            input_sizes[2],
            64);
    if (row != nullptr) {
      result.matched = true;
      result.family = SDPAScoreSoftmaxFamily::VisionSelfAttentionScores;
      result.tuple_id = row->tuple_id;
      result.metadata = &kSDPAScoreSoftmaxVisionSelfAttentionScoresMetadata;
      return result;
    }
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
