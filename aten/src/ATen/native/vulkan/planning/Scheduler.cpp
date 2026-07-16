#include <ATen/native/vulkan/planning/Runtime.h>
#include <ATen/native/vulkan/planning/Scheduler.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

const char* backend_route_name(const VulkanBackendRoute backend_route) {
  switch (backend_route) {
    case VulkanBackendRoute::Vulkan:
      return "Vulkan";
    case VulkanBackendRoute::CPU:
      return "CPU";
    case VulkanBackendRoute::Split:
      return "Split";
  }
  return "Vulkan";
}

const char* boundary_kind_name(const VulkanBoundaryKind boundary_kind) {
  switch (boundary_kind) {
    case VulkanBoundaryKind::None:
      return "None";
    case VulkanBoundaryKind::LLMLinearAttentionSplit:
      return "LLMLinearAttentionSplit";
  }
  return "None";
}

const char* boundary_transfer_layout_name(
    const VulkanBoundaryTransferLayout transfer_layout) {
  switch (transfer_layout) {
    case VulkanBoundaryTransferLayout::None:
      return "None";
    case VulkanBoundaryTransferLayout::BufferStaging:
      return "BufferStaging";
    case VulkanBoundaryTransferLayout::CacheAwareBuffer:
      return "CacheAwareBuffer";
  }
  return "None";
}

VulkanSchedulerDecision build_vulkan_scheduler_decision(
    const VulkanPlanningRequest& request,
    const VulkanRuntimeCapabilityProfile& capabilities) {
  VulkanSchedulerDecision decision;

  const bool decode_like =
      request.model_domain == VulkanModelDomain::LLM &&
      request.execution_phase == VulkanExecutionPhase::Decode;
  const bool prefill_like =
      request.model_domain == VulkanModelDomain::LLM &&
      request.execution_phase == VulkanExecutionPhase::Prefill;
  const bool cache_like =
      request.tensor_role == VulkanTensorRole::Cache ||
      request.workload_class == VulkanWorkloadClass::AttentionCache;
  const bool llm_runtime_like =
      request.model_domain == VulkanModelDomain::LLM &&
      (decode_like || prefill_like ||
       request.workload_class == VulkanWorkloadClass::LLMDecode);
  if (llm_runtime_like) {
    decision.scratch_arena_plan = VulkanScratchArenaPlanningDesc{
        true,
        true,
        0u,
        capabilities.has_unified_memory ? 256u : 512u,
    };
  }
  if (
      request.workload_class == VulkanWorkloadClass::VisionBackbone ||
      request.workload_class == VulkanWorkloadClass::VisionDecoder ||
      (request.model_domain == VulkanModelDomain::Vision &&
       (request.execution_phase == VulkanExecutionPhase::Backbone ||
        request.execution_phase == VulkanExecutionPhase::Decoder))) {
    decision.scratch_arena_plan = VulkanScratchArenaPlanningDesc{
        true,
        true,
        0u,
        capabilities.has_unified_memory ? 256u : 512u,
    };
  }

  if (decode_like && cache_like) {
    decision.backend_route = VulkanBackendRoute::Split;
    decision.boundary_plan = VulkanBoundaryPlan{
        VulkanBoundaryKind::LLMLinearAttentionSplit,
        VulkanBoundaryTransferLayout::CacheAwareBuffer,
        VulkanBoundaryTransferLayout::CacheAwareBuffer,
        true,
        true,
        1u,
    };
    return decision;
  }

  if (request.workload_class == VulkanWorkloadClass::LLMDecode) {
    decision.backend_route = VulkanBackendRoute::Split;
    decision.boundary_plan = VulkanBoundaryPlan{
        VulkanBoundaryKind::LLMLinearAttentionSplit,
        VulkanBoundaryTransferLayout::BufferStaging,
        VulkanBoundaryTransferLayout::BufferStaging,
        true,
        true,
        1u,
    };
    return decision;
  }

  if (request.workload_class == VulkanWorkloadClass::AttentionCache &&
      capabilities.num_compute_queues > 1u) {
    decision.backend_route = VulkanBackendRoute::Split;
    decision.boundary_plan = VulkanBoundaryPlan{
        VulkanBoundaryKind::LLMLinearAttentionSplit,
        VulkanBoundaryTransferLayout::CacheAwareBuffer,
        VulkanBoundaryTransferLayout::CacheAwareBuffer,
        true,
        true,
        1u,
    };
    return decision;
  }

  return decision;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
