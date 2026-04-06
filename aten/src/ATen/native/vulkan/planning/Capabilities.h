#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

struct VulkanRuntimeCapabilityProfile final {
  bool has_unified_memory{false};
  bool has_timestamps{false};
  bool has_shader_bfloat16{false};
  bool has_shader_int8{false};
  bool has_storage_buffer_8bit{false};
  bool supports_int8_buffer_arithmetic{false};
  uint32_t num_compute_queues{0u};
  uint32_t max_compute_workgroup_invocations{0u};
  uint32_t max_compute_shared_memory_size{0u};
};

VulkanRuntimeCapabilityProfile query_vulkan_runtime_capability_profile();

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
