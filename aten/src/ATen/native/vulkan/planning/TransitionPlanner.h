#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/planning/TransitionContracts.h>
#include <cstdint>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

struct VulkanLogicalTensorDesc final {
  const char* dtype = nullptr;
  const char* sizes = nullptr;
  const char* strides = nullptr;
};

struct VulkanPhysicalTensorDesc final {
  const char* layout = nullptr;
  const char* storage = nullptr;
  const char* memory_layout = nullptr;
  const char* execution_layout = nullptr;
};

struct VulkanTransitionRequest final {
  const char* phase = nullptr;
  TransitionReason reason = TransitionReason::UnknownTransitionReason;
  TransitionKind kind = TransitionKind::Unknown;
  int64_t bytes = -1;
  bool host_transfer = false;
  bool physical_copy = false;
  bool sync_required = false;
  bool queue_submit_required = false;
  const char* producer_schema = nullptr;
  const char* consumer_schema = nullptr;
  const char* producer_contract = nullptr;
  const char* consumer_contract = nullptr;
  VulkanLogicalTensorDesc source_logical;
  VulkanPhysicalTensorDesc source_physical;
  VulkanLogicalTensorDesc destination_logical;
  VulkanPhysicalTensorDesc destination_physical;
};

struct VulkanTransitionAdmission final {
  TransitionReason reason = TransitionReason::UnknownTransitionReason;
  TransitionKind kind = TransitionKind::Unknown;
  TransitionOutcome outcome = TransitionOutcome::Unknown;
  int64_t bytes = -1;
  bool host_transfer = false;
  bool physical_copy = false;
  bool sync_required = false;
  bool queue_submit_required = false;
};

bool transition_logging_enabled();
VulkanTransitionAdmission classify_vulkan_transition(
    const VulkanTransitionRequest& request);
void log_vulkan_transition(const VulkanTransitionRequest& request);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
