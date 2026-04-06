#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/planning/Capabilities.h>

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
  profile.has_shader_bfloat16 = adapter->has_shader_bfloat16();
  profile.has_shader_int8 = adapter->has_shader_int8();
  profile.has_storage_buffer_8bit = adapter->has_storage_buffer_8bit();
  profile.supports_int8_buffer_arithmetic =
      adapter->supports_int8_buffer_arithmetic();
  profile.num_compute_queues = adapter->num_compute_queues();
  profile.max_compute_workgroup_invocations =
      adapter->physical_handle() != VK_NULL_HANDLE
      ? adapter->physical_device().properties.limits.maxComputeWorkGroupInvocations
      : 0u;
  profile.max_compute_shared_memory_size =
      adapter->physical_handle() != VK_NULL_HANDLE
      ? adapter->physical_device().properties.limits.maxComputeSharedMemorySize
      : 0u;
  return profile;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
