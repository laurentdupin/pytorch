#include <ATen/native/vulkan/planning/DevicePolicy.h>

#include <ATen/native/vulkan/api/Context.h>

#include <cstring>
#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

bool contains_name(const std::string& name, const char* needle) {
  return !name.empty() && needle && std::strstr(name.c_str(), needle) != nullptr;
}

} // namespace

VulkanDevicePolicy current_vulkan_device_policy() {
  VulkanDevicePolicy policy;
  api::Context* const context = api::context();
  if (!context || !context->adapter_ptr()) {
    return policy;
  }

  const api::Adapter* const adapter = context->adapter_ptr();
  const auto& properties = adapter->physical_device().properties;
  policy.vendor_id = properties.vendorID;
  policy.device_id = properties.deviceID;
  policy.driver_version = properties.driverVersion;
  policy.device_name = properties.deviceName ? properties.deviceName : "";
  policy.supports_int8_buffer_arithmetic =
      adapter->supports_int8_buffer_arithmetic();
  policy.supports_subgroup_ops =
      adapter->has_subgroup_size_control() && adapter->max_subgroup_size() > 0u;

  const bool gtx_class = contains_name(policy.device_name, "GTX");
  const bool rx_6700_xt = contains_name(policy.device_name, "6700 XT");
  const bool rx_9070 = contains_name(policy.device_name, "RX 9070");
  policy.prefer_strict_replay_retirement = true;
  policy.avoid_large_persistent_weight_cache = gtx_class || rx_6700_xt;
  policy.disable_generic_tiled_diffusion_linear = true;
  policy.disable_generic_4d_sdpa = true;
  policy.disable_large_buffer_conv_3x3 = !rx_9070;
  policy.disable_known_bad_conv_3x3_s1_p1 =
      policy.disable_large_buffer_conv_3x3;
  return policy;
}

std::string describe_device_policy(const VulkanDevicePolicy& policy) {
  std::ostringstream out;
  out << "vendor=0x" << std::hex << policy.vendor_id << " device=0x"
      << policy.device_id << std::dec
      << " driver=" << policy.driver_version
      << " name=" << policy.device_name
      << " int8=" << (policy.supports_int8_buffer_arithmetic ? 1 : 0)
      << " subgroup=" << (policy.supports_subgroup_ops ? 1 : 0)
      << " strict_replay=" << (policy.prefer_strict_replay_retirement ? 1 : 0)
      << " avoid_weight_cache="
      << (policy.avoid_large_persistent_weight_cache ? 1 : 0)
      << " disable_large_conv3x3="
      << (policy.disable_large_buffer_conv_3x3 ? 1 : 0);
  return out.str();
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
