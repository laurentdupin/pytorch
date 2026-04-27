#include <ATen/native/vulkan/planning/ModelLanePolicy.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

const char* model_lane_name(const VulkanModelLane lane) {
  switch (lane) {
    case VulkanModelLane::Generic:
      return "Generic";
    case VulkanModelLane::DepthVisionTransformer:
      return "DepthVisionTransformer";
    case VulkanModelLane::DepthDiffusion:
      return "DepthDiffusion";
    case VulkanModelLane::AdjacentDepthVision:
      return "AdjacentDepthVision";
    case VulkanModelLane::LLM:
      return "LLM";
  }
  return "Generic";
}

VulkanModelLane infer_model_lane(const VulkanPlanningRequest& request) {
  if (request.model_domain == VulkanModelDomain::LLM) {
    return VulkanModelLane::LLM;
  }
  if (
      request.model_domain == VulkanModelDomain::Vision &&
      (request.execution_phase == VulkanExecutionPhase::Backbone ||
       request.workload_class == VulkanWorkloadClass::VisionBackbone)) {
    return VulkanModelLane::DepthVisionTransformer;
  }
  if (
      request.model_domain == VulkanModelDomain::Vision &&
      (request.execution_phase == VulkanExecutionPhase::Decoder ||
       request.workload_class == VulkanWorkloadClass::VisionDecoder)) {
    return VulkanModelLane::AdjacentDepthVision;
  }
  if (
      request.workload_class == VulkanWorkloadClass::Attention &&
      request.attention_shape.has_value() &&
      request.attention_shape->target_length > 0 &&
      request.attention_shape->source_length > 0 &&
      request.attention_shape->batch_heads > 1) {
    return VulkanModelLane::DepthDiffusion;
  }
  return VulkanModelLane::Generic;
}

bool prefers_buffer_resident_tokens(const VulkanModelLane lane) {
  return lane == VulkanModelLane::DepthVisionTransformer ||
      lane == VulkanModelLane::AdjacentDepthVision ||
      lane == VulkanModelLane::LLM;
}

bool permits_compiled_replay(
    const VulkanModelLane lane,
    const VulkanWorkloadClass workload_class) {
  if (lane == VulkanModelLane::DepthDiffusion) {
    return false;
  }
  return workload_class == VulkanWorkloadClass::VisionBackbone ||
      workload_class == VulkanWorkloadClass::VisionDecoder ||
      workload_class == VulkanWorkloadClass::Attention ||
      workload_class == VulkanWorkloadClass::Generic;
}

bool permits_generic_sdpa(
    const VulkanModelLane lane,
    const VulkanAttentionShapeDesc* attention_shape) {
  if (!attention_shape) {
    return true;
  }
  const bool diffusion_style_4d =
      lane == VulkanModelLane::DepthDiffusion &&
      attention_shape->batch_heads > 1 &&
      attention_shape->head_dim >= 64 &&
      attention_shape->target_length >= 64;
  return !diffusion_style_4d;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
