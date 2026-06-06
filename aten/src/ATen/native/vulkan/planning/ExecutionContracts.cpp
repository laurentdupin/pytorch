#include <ATen/native/vulkan/planning/ExecutionContracts.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

bool has_text(const char* value) {
  return value != nullptr && value[0] != '\0';
}

} // namespace

bool has_complete_execution_contract_metadata(
    const ExecutionContractMetadata* metadata) {
  return metadata != nullptr && has_text(metadata->contract_name) &&
      has_text(metadata->family_name) && has_text(metadata->tuple_id) &&
      has_text(metadata->evidence_id) && has_text(metadata->guard_id) &&
      has_text(metadata->fallback_policy) &&
      has_text(metadata->materialization_policy);
}

bool matches_sdpa_buffer_softmax_score_contract(
    const IntArrayRef input_sizes,
    const ScalarType input_dtype,
    const int64_t dim) {
  if (
      input_dtype != kFloat || input_sizes.size() != 3 ||
      dim != static_cast<int64_t>(input_sizes.size()) - 1 ||
      input_sizes[1] != input_sizes[2]) {
    return false;
  }
  const int64_t heads = input_sizes[0];
  const int64_t sequence = input_sizes[1];
  return (heads == 1 && (sequence == 504 || sequence == 640)) ||
      (heads == 5 && (sequence == 504 || sequence == 640));
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
