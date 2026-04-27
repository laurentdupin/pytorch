#pragma once

#ifdef USE_VULKAN_API

#include <cstdint>
#include <string>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

struct VulkanDevicePolicy final {
  uint32_t vendor_id{0u};
  uint32_t device_id{0u};
  uint32_t driver_version{0u};
  std::string device_name;

  bool supports_int8_buffer_arithmetic{false};
  bool supports_subgroup_ops{false};
  bool prefer_strict_replay_retirement{true};
  bool avoid_large_persistent_weight_cache{false};
  bool disable_generic_tiled_diffusion_linear{true};
  bool disable_generic_4d_sdpa{true};
  bool disable_known_bad_conv_3x3_s1_p1{true};
};

VulkanDevicePolicy current_vulkan_device_policy();

std::string describe_device_policy(const VulkanDevicePolicy& policy);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
