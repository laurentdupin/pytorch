#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsElementwiseBroadcastSpec.h>

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

constexpr ExecutionContractMetadata kElementwiseBroadcastMetadata =
    make_execution_contract_metadata(
        generated::kElementwiseBroadcastFloatTensorTensorBufferBroadcastSpec
            .contract_name,
        generated::kElementwiseBroadcastFloatTensorTensorBufferBroadcastSpec
            .family_name,
        generated::kElementwiseBroadcastFloatTensorTensorBufferBroadcastSpec
            .tuple_id,
        generated::kElementwiseBroadcastFloatTensorTensorBufferBroadcastSpec
            .evidence_id,
        generated::kElementwiseBroadcastFloatTensorTensorBufferBroadcastSpec
            .guard_id,
        generated::kElementwiseBroadcastFloatTensorTensorBufferBroadcastSpec
            .fallback_policy,
        generated::kElementwiseBroadcastFloatTensorTensorBufferBroadcastSpec
            .materialization_policy);

} // namespace

const char* elementwise_broadcast_family_name(
    const ElementwiseBroadcastFamily family) {
  switch (family) {
    case ElementwiseBroadcastFamily::FloatTensorTensorBufferBroadcast:
      return generated::kElementwiseBroadcastFloatTensorTensorBufferBroadcastSpec
          .family_name;
    case ElementwiseBroadcastFamily::None:
      return "ElementwiseBroadcastNone";
  }
  return "ElementwiseBroadcastNone";
}

ElementwiseBroadcastMatch match_elementwise_broadcast_contract(
    IntArrayRef self_sizes,
    IntArrayRef other_sizes,
    const ScalarType self_dtype,
    const ScalarType other_dtype,
    const ScalarType output_dtype,
    const bool self_is_vulkan,
    const bool other_is_vulkan,
    const bool self_supports_buffer_compute,
    const bool other_supports_buffer_compute,
    const bool buffer_route_selected,
    const ElementwiseBroadcastOp op,
    const bool alpha_is_one,
    const bool has_output,
    const bool inplace) {
  ElementwiseBroadcastMatch result;
  const auto& spec =
      generated::kElementwiseBroadcastFloatTensorTensorBufferBroadcastSpec;
  if (
      !generated::elementwise_float_tensor_tensor_buffer_broadcast_layout_matches(
          spec,
          self_is_vulkan,
          other_is_vulkan,
          self_supports_buffer_compute,
          other_supports_buffer_compute,
          buffer_route_selected) ||
      !generated::elementwise_float_tensor_tensor_buffer_broadcast_dtype_matches(
          spec, self_dtype, other_dtype, output_dtype) ||
      !generated::elementwise_float_tensor_tensor_buffer_broadcast_rank_in_bounds(
          spec, static_cast<int64_t>(self_sizes.size())) ||
      !generated::elementwise_float_tensor_tensor_buffer_broadcast_rank_in_bounds(
          spec, static_cast<int64_t>(other_sizes.size())) ||
      !generated::elementwise_float_tensor_tensor_buffer_broadcast_attributes_match(
          spec,
          op == ElementwiseBroadcastOp::Add,
          op == ElementwiseBroadcastOp::Mul,
          alpha_is_one,
          has_output,
          inplace)) {
    return result;
  }
  const bool broadcast_compatible =
      generated::
          elementwise_float_tensor_tensor_buffer_broadcast_broadcast_compatible(
              spec, self_sizes, other_sizes);
  if (!broadcast_compatible) {
    return result;
  }

  result.matched = true;
  result.family = ElementwiseBroadcastFamily::FloatTensorTensorBufferBroadcast;
  result.tuple_id = spec.tuple_id;
  result.metadata = &kElementwiseBroadcastMetadata;
  return result;
}

bool matches_elementwise_broadcast_contract(
    IntArrayRef self_sizes,
    IntArrayRef other_sizes,
    const ScalarType self_dtype,
    const ScalarType other_dtype,
    const ScalarType output_dtype,
    const bool self_is_vulkan,
    const bool other_is_vulkan,
    const bool self_supports_buffer_compute,
    const bool other_supports_buffer_compute,
    const bool buffer_route_selected,
    const ElementwiseBroadcastOp op,
    const bool alpha_is_one,
    const bool has_output,
    const bool inplace) {
  return match_elementwise_broadcast_contract(
             self_sizes,
             other_sizes,
             self_dtype,
             other_dtype,
             output_dtype,
             self_is_vulkan,
             other_is_vulkan,
             self_supports_buffer_compute,
             other_supports_buffer_compute,
             buffer_route_selected,
             op,
             alpha_is_one,
             has_output,
             inplace)
      .matched;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
