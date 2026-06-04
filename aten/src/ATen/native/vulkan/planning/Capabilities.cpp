#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/planning/CapabilityProfiles.h>
#include <ATen/native/vulkan/planning/Capabilities.h>

#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdlib>
#include <string_view>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

VulkanRuntimeCapabilityProfile query_vulkan_runtime_capability_profile() {
  VulkanRuntimeCapabilityProfile profile;
  api::Context* const context = api::context();
  if (!context) {
    return profile;
  }

  api::Adapter* const adapter = context->adapter_ptr();
  if (!adapter) {
    return profile;
  }

  profile.has_unified_memory = adapter->has_unified_memory();
  profile.has_timestamps = adapter->timestamp_compute_and_graphics();
  profile.has_vulkan_1_3 = adapter->has_vulkan_1_3();
  profile.has_maintenance4 = adapter->has_maintenance4();
  profile.has_synchronization2 = adapter->has_synchronization2();
  profile.has_shader_zero_initialize_workgroup_memory =
      adapter->has_shader_zero_initialize_workgroup_memory();
  profile.has_shader_integer_dot_product =
      adapter->has_shader_integer_dot_product();
  profile.has_pipeline_creation_cache_control =
      adapter->has_pipeline_creation_cache_control();
  profile.has_shader_bfloat16 = adapter->has_shader_bfloat16();
  profile.has_shader_int8 = adapter->has_shader_int8();
  profile.has_storage_buffer_8bit = adapter->has_storage_buffer_8bit();
  profile.has_cooperative_matrix = adapter->has_cooperative_matrix();
  profile.has_subgroup_size_control = adapter->has_subgroup_size_control();
  profile.has_compute_full_subgroups = adapter->has_compute_full_subgroups();
  profile.supports_int8_buffer_arithmetic =
      adapter->supports_int8_buffer_arithmetic();
  profile.min_subgroup_size = adapter->min_subgroup_size();
  profile.max_subgroup_size = adapter->max_subgroup_size();
  profile.max_compute_workgroup_subgroups =
      adapter->max_compute_workgroup_subgroups();
  profile.required_subgroup_size_stages =
      adapter->required_subgroup_size_stages();
  profile.cooperative_matrix_supported_stages =
      adapter->cooperative_matrix_supported_stages();
  profile.cooperative_matrix_property_count =
      adapter->cooperative_matrix_property_count();
  const auto& cooperative_matrix_properties =
      adapter->cooperative_matrix_properties();
  for (const auto& property : cooperative_matrix_properties) {
    profile.cooperative_matrix_max_m =
        std::max(profile.cooperative_matrix_max_m, property.m_size);
    profile.cooperative_matrix_max_n =
        std::max(profile.cooperative_matrix_max_n, property.n_size);
    profile.cooperative_matrix_max_k =
        std::max(profile.cooperative_matrix_max_k, property.k_size);
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
    if (property.scope != static_cast<uint32_t>(VK_SCOPE_SUBGROUP_KHR)) {
      continue;
    }
    if (
        property.a_type ==
            static_cast<uint32_t>(VK_COMPONENT_TYPE_FLOAT16_KHR) &&
        property.b_type ==
            static_cast<uint32_t>(VK_COMPONENT_TYPE_FLOAT16_KHR)) {
      profile.has_subgroup_float16_cooperative_matrix_inputs = true;
    }
    if (
        property.a_type ==
            static_cast<uint32_t>(VK_COMPONENT_TYPE_BFLOAT16_KHR) &&
        property.b_type ==
            static_cast<uint32_t>(VK_COMPONENT_TYPE_BFLOAT16_KHR)) {
      profile.has_subgroup_bfloat16_cooperative_matrix_inputs = true;
    }
    if (
        property.a_type ==
            static_cast<uint32_t>(VK_COMPONENT_TYPE_FLOAT32_KHR) &&
        property.b_type ==
            static_cast<uint32_t>(VK_COMPONENT_TYPE_FLOAT32_KHR)) {
      profile.has_subgroup_float32_cooperative_matrix_inputs = true;
    }
#endif
  }
  profile.num_compute_queues = adapter->num_compute_queues();
  profile.api_version = adapter->api_version();
  profile.max_compute_workgroup_invocations =
      adapter->physical_handle() != VK_NULL_HANDLE
      ? adapter->physical_device().properties.limits.maxComputeWorkGroupInvocations
      : 0u;
  profile.max_compute_shared_memory_size =
      adapter->physical_handle() != VK_NULL_HANDLE
      ? adapter->physical_device().properties.limits.maxComputeSharedMemorySize
      : 0u;
  return effective_vulkan_runtime_capability_profile(profile);
}

VulkanRuntimeCapabilityProfile effective_vulkan_runtime_capability_profile(
    const VulkanRuntimeCapabilityProfile& actual) {
  const char* const requested_id =
      std::getenv("PYTORCH_VULKAN_CAPABILITY_PROFILE");
  if (requested_id == nullptr || requested_id[0] == '\0') {
    return actual;
  }

  const VulkanCapabilityProfileSpec* const requested =
      find_vulkan_capability_profile(std::string_view(requested_id));
  TORCH_CHECK(
      requested != nullptr,
      "Unknown PYTORCH_VULKAN_CAPABILITY_PROFILE: ",
      requested_id);
  return intersect_vulkan_capability_profiles(actual, requested->profile);
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
