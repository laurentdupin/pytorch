#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/planning/DevicePolicy.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {
namespace legacy {

void apply_device_name_policy(VulkanDevicePolicy& policy);

} // namespace legacy
} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
