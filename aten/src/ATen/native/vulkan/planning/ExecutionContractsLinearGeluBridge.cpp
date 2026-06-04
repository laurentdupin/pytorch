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
constexpr const char* kMaterializationLinearGeluBridgeDeferred =
    "defer_linear_until_gelu_or_materialize_plain_linear";

constexpr int64_t kLinearGeluBridgeMinRows = 512;
constexpr int64_t kLinearGeluBridgeHiddenFeatures = 384;
constexpr int64_t kLinearGeluBridgeOutputFeatures = 1536;
constexpr int64_t kLinearGeluBridgeRank3Batch = 1;
constexpr const char* kLinearGeluBridgeBackboneMlpTupleId =
    "backbone_mlp_hidden384_to1536_rows_ge512";
constexpr ExecutionContractMetadata kLinearGeluBridgeBackboneMlpMetadata =
    make_execution_contract_metadata(
        "LinearGeluBridgeContract",
        "BackboneMlpHidden384To1536",
        kLinearGeluBridgeBackboneMlpTupleId,
        "linear_gelu_bridge_focused_tests",
        "linear_gelu_bridge_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationLinearGeluBridgeDeferred);

} // namespace

const char* linear_gelu_bridge_family_name(
    const LinearGeluBridgeFamily family) {
  switch (family) {
    case LinearGeluBridgeFamily::BackboneMlpHidden384To1536:
      return "LinearGeluBridgeBackboneMlpHidden384To1536";
    case LinearGeluBridgeFamily::None:
      return "LinearGeluBridgeNone";
  }
  return "LinearGeluBridgeNone";
}

LinearGeluBridgeMatch match_linear_gelu_bridge_contract(
    const LinearGeluBridgeTensorInfo& tensor,
    const LinearGeluBridgePackedInfo& packed,
    const LinearGeluBridgeOptions& options) {
  LinearGeluBridgeMatch result;
  if (
      options.inference_mode_enabled || options.has_output ||
      !options.post_op_is_none || !options.alpha_is_one ||
      !options.beta_is_one || !packed.bias_defined ||
      !packed.can_run_float_buffer_linear ||
      (tensor.input_rank != 2 && tensor.input_rank != 3) ||
      tensor.flattened_rank != 2 ||
      tensor.flattened_rows < kLinearGeluBridgeMinRows ||
      tensor.flattened_features != kLinearGeluBridgeHiddenFeatures ||
      packed.weight_height != kLinearGeluBridgeHiddenFeatures ||
      packed.weight_width != kLinearGeluBridgeOutputFeatures) {
    return result;
  }

  if (
      tensor.input_rank == 3 &&
      (tensor.input_batch != kLinearGeluBridgeRank3Batch ||
       tensor.input_rows != tensor.flattened_rows ||
       tensor.input_features != tensor.flattened_features)) {
    return result;
  }

  result.matched = true;
  result.family = LinearGeluBridgeFamily::BackboneMlpHidden384To1536;
  result.tuple_id = kLinearGeluBridgeBackboneMlpTupleId;
  result.metadata = &kLinearGeluBridgeBackboneMlpMetadata;
  result.may_defer = true;
  result.may_consume_gelu_none = true;
  result.may_consume_gelu_tanh = true;
  return result;
}

bool matches_linear_gelu_bridge_contract(
    const LinearGeluBridgeTensorInfo& tensor,
    const LinearGeluBridgePackedInfo& packed,
    const LinearGeluBridgeOptions& options) {
  return match_linear_gelu_bridge_contract(tensor, packed, options).matched;
}

bool matches_linear_gelu_bridge_gelu_approximation_contract(
    const std::string_view approximate) {
  return approximate == "none" || approximate == "tanh";
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
