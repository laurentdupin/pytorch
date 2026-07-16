#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/planning/Request.h>

#include <cstdint>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

enum class VulkanModelLane : uint8_t {
  Generic = 0u,
  DepthVisionTransformer,
  DepthDiffusion,
  AdjacentDepthVision,
  LLM,
};

const char* model_lane_name(VulkanModelLane lane);

VulkanModelLane infer_model_lane(const VulkanPlanningRequest& request);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
