#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsLinearGeluBridgeSpec.h>

#include <limits>

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

constexpr ExecutionContractMetadata kLinearGeluBridgeGenericRuntimeMetadata =
    make_execution_contract_metadata(
        "LinearGeluBridgeContract",
        "GenericRuntimeShape",
        "linear_gelu_bridge_generic_runtime_shape",
        "linear_gelu_bridge_dynamic_random_shapes",
        "linear_gelu_bridge_semantic_guards",
        "unsupported_semantics_do_not_match",
        "defer_linear_until_gelu_or_materialize_plain_linear");

} // namespace

const char* linear_gelu_bridge_family_name(
    const LinearGeluBridgeFamily family) {
  switch (family) {
    case LinearGeluBridgeFamily::BackboneMlpHidden384To1536:
      return "LinearGeluBridgeBackboneMlpHidden384To1536";
    case LinearGeluBridgeFamily::GenericRuntimeShape:
      return "LinearGeluBridgeGenericRuntimeShape";
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
      tensor.flattened_rank != 2 || tensor.flattened_rows <= 0 ||
      tensor.flattened_features <= 0 || packed.weight_height <= 0 ||
      packed.weight_width <= 0 || tensor.flattened_features !=
          packed.weight_height || !packed.bias_defined ||
      !packed.can_run_float_buffer_linear || options.has_output ||
      !options.post_op_is_none || !options.alpha_is_one ||
      !options.beta_is_one) {
    return result;
  }

  if (tensor.input_rank == 2) {
    if (
        tensor.input_rows != tensor.flattened_rows ||
        tensor.input_features != tensor.flattened_features) {
      return result;
    }
  } else if (tensor.input_rank == 3) {
    if (
        tensor.input_batch <= 0 || tensor.input_rows <= 0 ||
        tensor.input_features != tensor.flattened_features ||
        tensor.input_batch > std::numeric_limits<int64_t>::max() /
                tensor.input_rows ||
        tensor.input_batch * tensor.input_rows != tensor.flattened_rows) {
      return result;
    }
  } else {
    return result;
  }

  result.matched = true;
  result.family = LinearGeluBridgeFamily::GenericRuntimeShape;
  result.tuple_id = kLinearGeluBridgeGenericRuntimeMetadata.tuple_id;
  result.metadata = &kLinearGeluBridgeGenericRuntimeMetadata;
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
