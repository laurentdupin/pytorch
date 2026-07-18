#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/planning/Capabilities.h>

#include <string_view>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

struct VulkanMLFeatureSet final {
  bool has_unified_memory{false};
  bool has_vulkan_1_2{false};
  bool has_vulkan_1_3{false};
  bool has_maintenance4{false};
  bool has_synchronization2{false};
  bool has_buffer_device_address{false};
  bool supports_push_descriptor{false};
  bool supports_descriptor_buffer{false};
  bool has_shader_integer_dot_product{false};
  bool has_shader_bfloat16{false};
  bool has_shader_int8{false};
  bool has_storage_buffer_8bit{false};
  bool has_cooperative_matrix{false};
  bool has_subgroup_size_control{false};
  bool has_compute_full_subgroups{false};
  bool supports_int8_buffer_arithmetic{false};
  bool supports_subgroup_32{false};
  bool supports_subgroup_64{false};
  bool supports_cooperative_matrix_fp16{false};
  bool supports_cooperative_matrix_bf16{false};
  bool supports_cooperative_matrix_fp32{false};
};

struct VulkanCapabilityProfileSpec final {
  const char* id;
  const char* family;
  const char* kind;
  const char* description;
  VulkanRuntimeCapabilityProfile profile;
};

const VulkanCapabilityProfileSpec* find_vulkan_capability_profile(
    std::string_view id);

VulkanRuntimeCapabilityProfile intersect_vulkan_capability_profiles(
    const VulkanRuntimeCapabilityProfile& actual,
    const VulkanRuntimeCapabilityProfile& requested);

VulkanMLFeatureSet normalize_vulkan_ml_feature_set(
    const VulkanRuntimeCapabilityProfile& profile);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
