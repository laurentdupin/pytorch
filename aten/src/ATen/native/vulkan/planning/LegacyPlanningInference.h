#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/planning/Request.h>

#include <optional>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {
namespace legacy {

VulkanModelDomain infer_model_domain_from_planning_label();

bool planning_label_allows_llm_tensor_inference();

std::optional<VulkanExecutionPhase> infer_llm_phase_from_tensor_shape(
    const Tensor& tensor);

} // namespace legacy
} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
