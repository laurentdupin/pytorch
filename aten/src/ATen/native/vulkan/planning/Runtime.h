#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/planning/Capabilities.h>
#include <ATen/native/vulkan/planning/Request.h>
#include <ATen/native/vulkan/planning/Scheduler.h>

#include <optional>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

enum class VulkanLinearKernelFamily : uint8_t {
  TexturePacked = 0u,
  UnifiedBufferView,
  PersistentPackedTexture,
};

enum class VulkanNormKernelFamily : uint8_t {
  TextureWidth = 0u,
  SharedMemoryWidth,
  UnifiedBufferView,
};

enum class VulkanAttentionKernelFamily : uint8_t {
  TextureMath = 0u,
  CacheAwareTexture,
  SplitCoordinator,
};

enum class VulkanExecutionProgramKind : uint8_t {
  AttentionRuntime = 0u,
  GatedDeltaSplit,
};

struct VulkanExecutionProgramPlanningDesc final {
  VulkanExecutionProgramKind kind{VulkanExecutionProgramKind::AttentionRuntime};
  bool persistent{true};
};

struct VulkanRuntimePolicy final {
  VulkanPlanningRequest request{};
  VulkanBackendRoute backend_route{VulkanBackendRoute::Vulkan};
  VulkanLinearKernelFamily linear_kernel_family{
      VulkanLinearKernelFamily::TexturePacked};
  VulkanNormKernelFamily norm_kernel_family{
      VulkanNormKernelFamily::TextureWidth};
  VulkanAttentionKernelFamily attention_kernel_family{
      VulkanAttentionKernelFamily::TextureMath};
  std::optional<VulkanExecutionProgramPlanningDesc> execution_program_plan;
  std::optional<VulkanBoundaryPlan> boundary_plan;
  std::optional<VulkanKVCachePlanningDesc> kv_cache_plan;
  std::optional<VulkanScratchArenaPlanningDesc> scratch_arena_plan;
};

const char* linear_kernel_family_name(VulkanLinearKernelFamily);
const char* norm_kernel_family_name(VulkanNormKernelFamily);
const char* attention_kernel_family_name(VulkanAttentionKernelFamily);
const char* execution_program_kind_name(VulkanExecutionProgramKind);

VulkanRuntimePolicy build_vulkan_runtime_policy(const VulkanPlanningRequest&);

VulkanRuntimePolicy build_vulkan_runtime_policy(VulkanWorkloadClass);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
