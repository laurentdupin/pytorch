#include <ATen/native/vulkan/planning/ExecutionContracts.h>

#include <algorithm>

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

constexpr const char* kElementwiseBroadcastTupleId =
    "float32_rank1_to_4_tensor_tensor_buffer_broadcast";
constexpr ExecutionContractMetadata kElementwiseBroadcastMetadata =
    make_execution_contract_metadata(
        "ElementwiseBroadcastContract",
        "FloatTensorTensorBufferBroadcast",
        kElementwiseBroadcastTupleId,
        "float_buffer_binary_broadcast_focused_tests",
        "elementwise_broadcast_adjacent_guards",
        "unsupported_shapes_do_not_match",
        "elementwise_buffer_kernel");

bool broadcast_compatible(IntArrayRef left, IntArrayRef right) {
  const int64_t max_rank =
      std::max<int64_t>(
          static_cast<int64_t>(left.size()),
          static_cast<int64_t>(right.size()));
  for (int64_t axis = 0; axis < max_rank; ++axis) {
    const int64_t left_axis = static_cast<int64_t>(left.size()) - 1 - axis;
    const int64_t right_axis = static_cast<int64_t>(right.size()) - 1 - axis;
    const int64_t left_dim = left_axis >= 0 ? left[left_axis] : 1;
    const int64_t right_dim = right_axis >= 0 ? right[right_axis] : 1;
    if (left_dim != right_dim && left_dim != 1 && right_dim != 1) {
      return false;
    }
  }
  return true;
}

} // namespace

const char* elementwise_broadcast_family_name(
    const ElementwiseBroadcastFamily family) {
  switch (family) {
    case ElementwiseBroadcastFamily::FloatTensorTensorBufferBroadcast:
      return "FloatTensorTensorBufferBroadcast";
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
  if (
      !buffer_route_selected || !self_is_vulkan || !other_is_vulkan ||
      !self_supports_buffer_compute || !other_supports_buffer_compute ||
      self_dtype != kFloat || other_dtype != kFloat || output_dtype != kFloat ||
      self_sizes.empty() || other_sizes.empty() || self_sizes.size() > 4 ||
      other_sizes.size() > 4 || !alpha_is_one || has_output || inplace) {
    return result;
  }
  if (
      op != ElementwiseBroadcastOp::Add &&
      op != ElementwiseBroadcastOp::Mul) {
    return result;
  }
  if (!broadcast_compatible(self_sizes, other_sizes)) {
    return result;
  }

  result.matched = true;
  result.family = ElementwiseBroadcastFamily::FloatTensorTensorBufferBroadcast;
  result.tuple_id = kElementwiseBroadcastTupleId;
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
