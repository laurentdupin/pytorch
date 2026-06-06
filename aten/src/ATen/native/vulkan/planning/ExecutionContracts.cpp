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

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
