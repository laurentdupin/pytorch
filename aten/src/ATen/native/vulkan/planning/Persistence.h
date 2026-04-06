#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/planning/Request.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

struct VulkanPersistenceHints final {
  bool prefer_persistent_weights{false};
  bool prefer_persistent_contexts{false};
};

VulkanPersistenceHints build_vulkan_persistence_hints(
    const VulkanPlanningRequest&);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
