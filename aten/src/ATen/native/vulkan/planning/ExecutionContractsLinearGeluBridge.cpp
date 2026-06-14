#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsLinearGeluBridgeSpec.h>

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

constexpr ExecutionContractMetadata kLinearGeluBridgeBackboneMlpMetadata =
    make_execution_contract_metadata(
        generated::kLinearGeluBridgeBackboneMlpHidden384To1536Spec
            .contract_name,
        generated::kLinearGeluBridgeBackboneMlpHidden384To1536Spec.family_name,
        generated::kLinearGeluBridgeBackboneMlpHidden384To1536Spec.tuple_id,
        generated::kLinearGeluBridgeBackboneMlpHidden384To1536Spec.evidence_id,
        generated::kLinearGeluBridgeBackboneMlpHidden384To1536Spec.guard_id,
        generated::kLinearGeluBridgeBackboneMlpHidden384To1536Spec
            .fallback_policy,
        generated::kLinearGeluBridgeBackboneMlpHidden384To1536Spec
            .materialization_policy);

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
  const auto& spec =
      generated::kLinearGeluBridgeBackboneMlpHidden384To1536Spec;
  const int64_t rank3_batch =
      tensor.input_rank == 3 ? tensor.input_batch : spec.rank3_batch;
  if (
      !generated::linear_gelu_bridge_backbone_mlp_hidden_384_to_1536_options_match(
          spec,
          tensor.flattened_rank,
          tensor.flattened_features,
          packed.weight_height,
          packed.weight_width,
          rank3_batch,
          tensor.input_rank,
          packed.bias_defined,
          packed.can_run_float_buffer_linear,
          options.inference_mode_enabled,
          options.has_output,
          options.post_op_is_none,
          options.alpha_is_one,
          options.beta_is_one,
          spec.may_defer,
          spec.may_consume_gelu_none,
          spec.may_consume_gelu_tanh) ||
      !generated::linear_gelu_bridge_backbone_mlp_hidden_384_to_1536_in_bounds(
          spec,
          tensor.flattened_rows)) {
    return result;
  }

  if (
      tensor.input_rank == 3 &&
      (tensor.input_batch != spec.rank3_batch ||
       tensor.input_rows != tensor.flattened_rows ||
       tensor.input_features != tensor.flattened_features)) {
    return result;
  }

  result.matched = true;
  result.family = LinearGeluBridgeFamily::BackboneMlpHidden384To1536;
  result.tuple_id = spec.tuple_id;
  result.metadata = &kLinearGeluBridgeBackboneMlpMetadata;
  result.may_defer = spec.may_defer;
  result.may_consume_gelu_none = spec.may_consume_gelu_none;
  result.may_consume_gelu_tanh = spec.may_consume_gelu_tanh;
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
