#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/planning/Capabilities.h>
#include <ATen/native/vulkan/planning/Request.h>

#include <cstddef>
#include <optional>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

enum class VulkanBackendRoute : uint8_t {
  Vulkan = 0u,
  CPU,
  Split,
};

enum class VulkanBoundaryKind : uint8_t {
  None = 0u,
  LLMLinearAttentionSplit,
};

enum class VulkanBoundaryTransferLayout : uint8_t {
  None = 0u,
  BufferStaging,
  CacheAwareBuffer,
};

struct VulkanScratchArenaPlanningDesc final {
  bool prefer_reusable_arena{false};
  bool prefer_buffer_storage{true};
  size_t min_arena_bytes{0u};
  uint32_t alignment{256u};
};

struct VulkanBoundaryPlan final {
  VulkanBoundaryKind kind{VulkanBoundaryKind::None};
  VulkanBoundaryTransferLayout input_transfer_layout{
      VulkanBoundaryTransferLayout::None};
  VulkanBoundaryTransferLayout output_transfer_layout{
      VulkanBoundaryTransferLayout::None};
  bool prefer_backend_owned_execution{false};
  bool requires_scratch_arena{false};
  uint32_t preferred_cpu_threads{0u};
};

struct VulkanSchedulerDecision final {
  VulkanBackendRoute backend_route{VulkanBackendRoute::Vulkan};
  std::optional<VulkanBoundaryPlan> boundary_plan;
  std::optional<VulkanScratchArenaPlanningDesc> scratch_arena_plan;
};

const char* backend_route_name(VulkanBackendRoute);
const char* boundary_kind_name(VulkanBoundaryKind);
const char* boundary_transfer_layout_name(VulkanBoundaryTransferLayout);

VulkanSchedulerDecision build_vulkan_scheduler_decision(
    const VulkanPlanningRequest&,
    const VulkanRuntimeCapabilityProfile&);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
