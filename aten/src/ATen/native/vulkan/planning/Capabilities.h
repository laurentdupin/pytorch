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
  bool has_vulkan_1_3{false};
  bool has_maintenance4{false};
  bool has_synchronization2{false};
  bool has_shader_zero_initialize_workgroup_memory{false};
  bool has_shader_integer_dot_product{false};
  bool has_pipeline_creation_cache_control{false};
  bool has_shader_bfloat16{false};
  bool has_shader_int8{false};
  bool has_storage_buffer_8bit{false};
  bool has_cooperative_matrix{false};
  bool has_subgroup_size_control{false};
  bool has_compute_full_subgroups{false};
  bool has_subgroup_float16_cooperative_matrix_inputs{false};
  bool has_subgroup_bfloat16_cooperative_matrix_inputs{false};
  bool has_subgroup_float32_cooperative_matrix_inputs{false};
  bool supports_int8_buffer_arithmetic{false};
  uint32_t min_subgroup_size{0u};
  uint32_t max_subgroup_size{0u};
  uint32_t max_compute_workgroup_subgroups{0u};
  uint32_t required_subgroup_size_stages{0u};
  uint32_t cooperative_matrix_supported_stages{0u};
  uint32_t cooperative_matrix_property_count{0u};
  uint32_t cooperative_matrix_max_m{0u};
  uint32_t cooperative_matrix_max_n{0u};
  uint32_t cooperative_matrix_max_k{0u};
  uint32_t num_compute_queues{0u};
  uint32_t api_version{0u};
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
