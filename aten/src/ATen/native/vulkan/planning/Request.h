#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

enum class VulkanWorkloadClass : uint8_t {
  Generic = 0u,
  LinearMatmul,
  Attention,
  AttentionCache,
  Norm,
  Convolution,
  ShapeView,
  Elementwise,
  Reduction,
  VisionBackbone,
  VisionDecoder,
  LLMDecode,
};

enum class VulkanModelDomain : uint8_t {
  Generic = 0u,
  Vision,
  LLM,
};

enum class VulkanExecutionPhase : uint8_t {
  None = 0u,
  Prefill,
  Decode,
  Backbone,
  Decoder,
};

enum class VulkanTensorRole : uint8_t {
  Input = 0u,
  Weight,
  Bias,
  Cache,
  Scratch,
  Mask,
};

struct VulkanPlanningRequest final {
  VulkanWorkloadClass workload_class{VulkanWorkloadClass::Generic};
  VulkanModelDomain model_domain{VulkanModelDomain::Generic};
  VulkanExecutionPhase execution_phase{VulkanExecutionPhase::None};
  VulkanTensorRole tensor_role{VulkanTensorRole::Input};
  bool inferred_from_label{false};
};

const char* workload_class_name(VulkanWorkloadClass);

const char* model_domain_name(VulkanModelDomain);

const char* execution_phase_name(VulkanExecutionPhase);

const char* tensor_role_name(VulkanTensorRole);

VulkanPlanningRequest make_vulkan_planning_request(
    VulkanWorkloadClass workload_class,
    VulkanTensorRole tensor_role = VulkanTensorRole::Input,
    VulkanModelDomain model_domain = VulkanModelDomain::Generic,
    VulkanExecutionPhase execution_phase = VulkanExecutionPhase::None);

VulkanPlanningRequest make_vulkan_tensor_planning_request(
    const Tensor& tensor,
    VulkanWorkloadClass workload_class,
    VulkanTensorRole tensor_role = VulkanTensorRole::Input,
    VulkanModelDomain model_domain = VulkanModelDomain::Generic,
    VulkanExecutionPhase execution_phase = VulkanExecutionPhase::None);

VulkanPlanningRequest infer_vulkan_planning_request(
    const VulkanPlanningRequest&);

VulkanPlanningRequest infer_vulkan_planning_request(VulkanWorkloadClass);

VulkanPlanningRequest specialize_vulkan_planning_request_for_tensor(
    const Tensor&,
    const VulkanPlanningRequest&);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
