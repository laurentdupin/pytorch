#include <ATen/native/vulkan/planning/Persistence.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

VulkanPersistenceHints build_vulkan_persistence_hints(
    const VulkanPlanningRequest& request) {
  VulkanPersistenceHints hints;

  switch (request.tensor_role) {
    case VulkanTensorRole::Weight:
      hints.prefer_persistent_weights = true;
      break;
    case VulkanTensorRole::Cache:
      hints.prefer_persistent_contexts = true;
      break;
    case VulkanTensorRole::Input:
    case VulkanTensorRole::Bias:
    case VulkanTensorRole::Scratch:
    case VulkanTensorRole::Mask:
      break;
  }

  switch (request.workload_class) {
    case VulkanWorkloadClass::LinearMatmul:
    case VulkanWorkloadClass::Convolution:
      hints.prefer_persistent_weights = true;
      hints.prefer_persistent_contexts = true;
      break;
    case VulkanWorkloadClass::AttentionCache:
    case VulkanWorkloadClass::VisionBackbone:
      hints.prefer_persistent_contexts = true;
      break;
    case VulkanWorkloadClass::VisionDecoder:
    case VulkanWorkloadClass::LLMDecode:
      hints.prefer_persistent_weights = true;
      hints.prefer_persistent_contexts = true;
      break;
    case VulkanWorkloadClass::Generic:
    case VulkanWorkloadClass::Attention:
    case VulkanWorkloadClass::Norm:
    case VulkanWorkloadClass::ShapeView:
    case VulkanWorkloadClass::Elementwise:
    case VulkanWorkloadClass::Reduction:
      break;
  }

  if (request.model_domain == VulkanModelDomain::LLM &&
      request.execution_phase == VulkanExecutionPhase::Decode) {
    hints.prefer_persistent_weights = true;
    hints.prefer_persistent_contexts = true;
  }

  return hints;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
