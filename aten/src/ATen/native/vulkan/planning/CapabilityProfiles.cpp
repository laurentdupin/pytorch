#include <ATen/native/vulkan/planning/CapabilityProfiles.h>

#include <algorithm>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

constexpr uint32_t api_version(const uint32_t major, const uint32_t minor) {
  return VK_MAKE_API_VERSION(0u, major, minor, 0u);
}

constexpr VulkanRuntimeCapabilityProfile profile(
    const uint32_t api,
    const bool unified_memory,
    const bool maintenance4,
    const bool synchronization2,
    const bool shader_integer_dot_product,
    const bool shader_bfloat16,
    const bool shader_int8,
    const bool storage_buffer_8bit,
    const bool cooperative_matrix,
    const bool subgroup_size_control,
    const bool compute_full_subgroups,
    const bool int8_buffer_arithmetic,
    const uint32_t min_subgroup_size,
    const uint32_t max_subgroup_size,
    const uint32_t max_compute_workgroup_invocations,
    const uint32_t max_compute_shared_memory_size,
    const uint32_t cooperative_matrix_property_count = 0u,
    const uint32_t cooperative_matrix_max_m = 0u,
    const uint32_t cooperative_matrix_max_n = 0u,
    const uint32_t cooperative_matrix_max_k = 0u) {
  VulkanRuntimeCapabilityProfile result;
  result.has_unified_memory = unified_memory;
  result.has_timestamps = false;
  result.has_vulkan_1_3 = api >= api_version(1u, 3u);
  result.has_maintenance4 = maintenance4;
  result.has_synchronization2 = synchronization2;
  result.has_shader_zero_initialize_workgroup_memory = false;
  result.has_shader_integer_dot_product = shader_integer_dot_product;
  result.has_pipeline_creation_cache_control = false;
  result.has_shader_bfloat16 = shader_bfloat16;
  result.has_shader_int8 = shader_int8;
  result.has_storage_buffer_8bit = storage_buffer_8bit;
  result.has_cooperative_matrix = cooperative_matrix;
  result.has_subgroup_size_control = subgroup_size_control;
  result.has_compute_full_subgroups = compute_full_subgroups;
  result.has_subgroup_float16_cooperative_matrix_inputs = cooperative_matrix;
  result.has_subgroup_bfloat16_cooperative_matrix_inputs =
      cooperative_matrix && shader_bfloat16;
  result.has_subgroup_float32_cooperative_matrix_inputs = cooperative_matrix;
  result.supports_int8_buffer_arithmetic = int8_buffer_arithmetic;
  result.min_subgroup_size = min_subgroup_size;
  result.max_subgroup_size = max_subgroup_size;
  result.max_compute_workgroup_subgroups =
      max_subgroup_size == 0u ? 0u : std::max<uint32_t>(1u, 256u / max_subgroup_size);
  result.required_subgroup_size_stages = subgroup_size_control ? 1u : 0u;
  result.cooperative_matrix_supported_stages = cooperative_matrix ? 1u : 0u;
  result.cooperative_matrix_property_count = cooperative_matrix_property_count;
  result.cooperative_matrix_max_m = cooperative_matrix_max_m;
  result.cooperative_matrix_max_n = cooperative_matrix_max_n;
  result.cooperative_matrix_max_k = cooperative_matrix_max_k;
  result.num_compute_queues = 1u;
  result.api_version = api;
  result.max_compute_workgroup_invocations = max_compute_workgroup_invocations;
  result.max_compute_shared_memory_size = max_compute_shared_memory_size;
  return result;
}

constexpr VulkanCapabilityProfileSpec kCapabilityProfiles[] = {
    {"amd_polaris",
     "AMD Polaris",
     "vendor_family_bucket",
     "Conservative AMD Polaris-era discrete Vulkan compute feature mask.",
     profile(
         api_version(1u, 1u),
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         64u,
         64u,
         256u,
         32768u)},
    {"amd_vega",
     "AMD Vega",
     "vendor_family_bucket",
     "Conservative AMD Vega-era discrete Vulkan compute feature mask.",
     profile(
         api_version(1u, 1u),
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         64u,
         64u,
         256u,
         32768u)},
    {"amd_rdna1",
     "AMD RDNA1",
     "vendor_family_bucket",
     "Conservative AMD RDNA1-era discrete Vulkan compute feature mask.",
     profile(
         api_version(1u, 2u),
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         true,
         false,
         false,
         32u,
         64u,
         256u,
         32768u)},
    {"amd_rdna2",
     "AMD RDNA2",
     "vendor_family_bucket",
     "Conservative AMD RDNA2-era discrete Vulkan compute feature mask.",
     profile(
         api_version(1u, 2u),
         false,
         false,
         false,
         true,
         false,
         true,
         true,
         false,
         true,
         true,
         true,
         32u,
         64u,
         256u,
         32768u)},
    {"amd_rdna3",
     "AMD RDNA3",
     "vendor_family_bucket",
     "Conservative AMD RDNA3-era discrete Vulkan compute feature mask.",
     profile(
         api_version(1u, 3u),
         false,
         true,
         true,
         true,
         false,
         true,
         true,
         false,
         true,
         true,
         true,
         32u,
         64u,
         256u,
         32768u)},
    {"amd_rdna4",
     "AMD RDNA4",
     "vendor_family_bucket",
     "Conservative AMD RDNA4-era discrete Vulkan compute feature mask.",
     profile(
         api_version(1u, 3u),
         false,
         true,
         true,
         true,
         true,
         true,
         true,
         true,
         true,
         true,
         true,
         32u,
         64u,
         256u,
         32768u,
         1u,
         16u,
         16u,
         16u)},
    {"nvidia_pascal",
     "NVIDIA Pascal",
     "vendor_family_bucket",
     "Conservative NVIDIA Pascal-era discrete Vulkan compute feature mask.",
     profile(
         api_version(1u, 1u),
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         32u,
         32u,
         1024u,
         49152u)},
    {"nvidia_volta",
     "NVIDIA Volta",
     "vendor_family_bucket",
     "Conservative NVIDIA Volta-era discrete Vulkan compute feature mask.",
     profile(
         api_version(1u, 1u),
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         32u,
         32u,
         1024u,
         49152u)},
    {"nvidia_turing",
     "NVIDIA Turing",
     "vendor_family_bucket",
     "Conservative NVIDIA Turing-era discrete Vulkan compute feature mask.",
     profile(
         api_version(1u, 2u),
         false,
         false,
         false,
         true,
         false,
         true,
         true,
         false,
         true,
         true,
         true,
         32u,
         32u,
         1024u,
         49152u)},
    {"nvidia_ampere",
     "NVIDIA Ampere",
     "vendor_family_bucket",
     "Conservative NVIDIA Ampere-era discrete Vulkan compute feature mask.",
     profile(
         api_version(1u, 3u),
         false,
         true,
         true,
         true,
         false,
         true,
         true,
         true,
         true,
         true,
         true,
         32u,
         32u,
         1024u,
         49152u,
         1u,
         16u,
         16u,
         16u)},
    {"nvidia_ada",
     "NVIDIA Ada",
     "vendor_family_bucket",
     "Conservative NVIDIA Ada-era discrete Vulkan compute feature mask.",
     profile(
         api_version(1u, 3u),
         false,
         true,
         true,
         true,
         false,
         true,
         true,
         true,
         true,
         true,
         true,
         32u,
         32u,
         1024u,
         49152u,
         1u,
         16u,
         16u,
         16u)},
    {"nvidia_blackwell",
     "NVIDIA Blackwell",
     "vendor_family_bucket",
     "Conservative NVIDIA Blackwell-era discrete Vulkan compute feature mask.",
     profile(
         api_version(1u, 3u),
         false,
         true,
         true,
         true,
         true,
         true,
         true,
         true,
         true,
         true,
         true,
         32u,
         32u,
         1024u,
         49152u,
         1u,
         16u,
         16u,
         16u)},
    {"vk_min_1_1_compute",
     "Vulkan minimum compute",
     "standard_floor",
     "Portable Vulkan 1.1 compute floor with optional ML features disabled.",
     profile(
         api_version(1u, 1u),
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         0u,
         0u,
         128u,
         16384u)},
    {"vk_min_1_2_compute",
     "Vulkan minimum compute",
     "standard_floor",
     "Portable Vulkan 1.2 compute floor with optional ML features disabled.",
     profile(
         api_version(1u, 2u),
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         0u,
         0u,
         256u,
         32768u)},
    {"roadmap_2022",
     "Vulkan Roadmap 2022",
     "standard_floor",
     "Conservative Vulkan Roadmap 2022-style compute feature mask.",
     profile(
         api_version(1u, 3u),
         false,
         true,
         true,
         true,
         false,
         true,
         true,
         false,
         true,
         true,
         true,
         32u,
         64u,
         256u,
         32768u)},
    {"roadmap_2024",
     "Vulkan Roadmap 2024",
     "standard_floor",
     "Conservative Vulkan Roadmap 2024-style compute feature mask.",
     profile(
         api_version(1u, 3u),
         false,
         true,
         true,
         true,
         true,
         true,
         true,
         true,
         true,
         true,
         true,
         32u,
         64u,
         256u,
         32768u,
         1u,
         16u,
         16u,
         16u)},
};

} // namespace

const VulkanCapabilityProfileSpec* find_vulkan_capability_profile(
    const std::string_view id) {
  for (const auto& spec : kCapabilityProfiles) {
    if (id == spec.id) {
      return &spec;
    }
  }
  return nullptr;
}

VulkanRuntimeCapabilityProfile intersect_vulkan_capability_profiles(
    const VulkanRuntimeCapabilityProfile& actual,
    const VulkanRuntimeCapabilityProfile& requested) {
  VulkanRuntimeCapabilityProfile result;
  result.has_unified_memory =
      actual.has_unified_memory && requested.has_unified_memory;
  result.has_timestamps = actual.has_timestamps && requested.has_timestamps;
  result.api_version = std::min(actual.api_version, requested.api_version);
  result.has_vulkan_1_3 = actual.has_vulkan_1_3 && requested.has_vulkan_1_3 &&
      result.api_version >= api_version(1u, 3u);
  result.has_maintenance4 = actual.has_maintenance4 && requested.has_maintenance4;
  result.has_synchronization2 =
      actual.has_synchronization2 && requested.has_synchronization2;
  result.has_shader_zero_initialize_workgroup_memory =
      actual.has_shader_zero_initialize_workgroup_memory &&
      requested.has_shader_zero_initialize_workgroup_memory;
  result.has_shader_integer_dot_product =
      actual.has_shader_integer_dot_product &&
      requested.has_shader_integer_dot_product;
  result.has_pipeline_creation_cache_control =
      actual.has_pipeline_creation_cache_control &&
      requested.has_pipeline_creation_cache_control;
  result.has_shader_bfloat16 =
      actual.has_shader_bfloat16 && requested.has_shader_bfloat16;
  result.has_shader_int8 = actual.has_shader_int8 && requested.has_shader_int8;
  result.has_storage_buffer_8bit =
      actual.has_storage_buffer_8bit && requested.has_storage_buffer_8bit;
  result.has_cooperative_matrix =
      actual.has_cooperative_matrix && requested.has_cooperative_matrix;
  result.has_subgroup_size_control =
      actual.has_subgroup_size_control && requested.has_subgroup_size_control;
  result.has_compute_full_subgroups =
      actual.has_compute_full_subgroups && requested.has_compute_full_subgroups;
  result.supports_int8_buffer_arithmetic =
      actual.supports_int8_buffer_arithmetic &&
      requested.supports_int8_buffer_arithmetic;
  result.min_subgroup_size =
      std::max(actual.min_subgroup_size, requested.min_subgroup_size);
  result.max_subgroup_size =
      std::min(actual.max_subgroup_size, requested.max_subgroup_size);
  result.max_compute_workgroup_subgroups = std::min(
      actual.max_compute_workgroup_subgroups,
      requested.max_compute_workgroup_subgroups);
  result.required_subgroup_size_stages =
      actual.required_subgroup_size_stages & requested.required_subgroup_size_stages;
  result.num_compute_queues =
      std::min(actual.num_compute_queues, requested.num_compute_queues);
  result.max_compute_workgroup_invocations = std::min(
      actual.max_compute_workgroup_invocations,
      requested.max_compute_workgroup_invocations);
  result.max_compute_shared_memory_size = std::min(
      actual.max_compute_shared_memory_size,
      requested.max_compute_shared_memory_size);

  if (
      result.max_subgroup_size == 0u ||
      result.min_subgroup_size > result.max_subgroup_size) {
    result.min_subgroup_size = 0u;
    result.max_subgroup_size = 0u;
    result.max_compute_workgroup_subgroups = 0u;
    result.required_subgroup_size_stages = 0u;
    result.has_subgroup_size_control = false;
    result.has_compute_full_subgroups = false;
  }

  if (result.has_cooperative_matrix) {
    result.has_subgroup_float16_cooperative_matrix_inputs =
        actual.has_subgroup_float16_cooperative_matrix_inputs &&
        requested.has_subgroup_float16_cooperative_matrix_inputs;
    result.has_subgroup_bfloat16_cooperative_matrix_inputs =
        actual.has_subgroup_bfloat16_cooperative_matrix_inputs &&
        requested.has_subgroup_bfloat16_cooperative_matrix_inputs;
    result.has_subgroup_float32_cooperative_matrix_inputs =
        actual.has_subgroup_float32_cooperative_matrix_inputs &&
        requested.has_subgroup_float32_cooperative_matrix_inputs;
    result.cooperative_matrix_supported_stages =
        actual.cooperative_matrix_supported_stages &
        requested.cooperative_matrix_supported_stages;
    result.cooperative_matrix_property_count = std::min(
        actual.cooperative_matrix_property_count,
        requested.cooperative_matrix_property_count);
    result.cooperative_matrix_max_m = std::min(
        actual.cooperative_matrix_max_m, requested.cooperative_matrix_max_m);
    result.cooperative_matrix_max_n = std::min(
        actual.cooperative_matrix_max_n, requested.cooperative_matrix_max_n);
    result.cooperative_matrix_max_k = std::min(
        actual.cooperative_matrix_max_k, requested.cooperative_matrix_max_k);
  }

  return result;
}

VulkanMLFeatureSet normalize_vulkan_ml_feature_set(
    const VulkanRuntimeCapabilityProfile& profile) {
  VulkanMLFeatureSet features;
  features.has_unified_memory = profile.has_unified_memory;
  features.has_vulkan_1_2 = profile.api_version >= api_version(1u, 2u);
  features.has_vulkan_1_3 =
      profile.has_vulkan_1_3 && profile.api_version >= api_version(1u, 3u);
  features.has_maintenance4 = profile.has_maintenance4;
  features.has_synchronization2 = profile.has_synchronization2;
  features.has_shader_integer_dot_product =
      profile.has_shader_integer_dot_product;
  features.has_shader_bfloat16 = profile.has_shader_bfloat16;
  features.has_shader_int8 = profile.has_shader_int8;
  features.has_storage_buffer_8bit = profile.has_storage_buffer_8bit;
  features.has_cooperative_matrix = profile.has_cooperative_matrix;
  features.has_subgroup_size_control = profile.has_subgroup_size_control;
  features.has_compute_full_subgroups = profile.has_compute_full_subgroups;
  features.supports_int8_buffer_arithmetic =
      profile.supports_int8_buffer_arithmetic;
  features.supports_subgroup_32 =
      profile.min_subgroup_size <= 32u && profile.max_subgroup_size >= 32u;
  features.supports_subgroup_64 =
      profile.min_subgroup_size <= 64u && profile.max_subgroup_size >= 64u;
  features.supports_cooperative_matrix_fp16 =
      profile.has_cooperative_matrix &&
      profile.has_subgroup_float16_cooperative_matrix_inputs;
  features.supports_cooperative_matrix_bf16 =
      profile.has_cooperative_matrix &&
      profile.has_subgroup_bfloat16_cooperative_matrix_inputs;
  features.supports_cooperative_matrix_fp32 =
      profile.has_cooperative_matrix &&
      profile.has_subgroup_float32_cooperative_matrix_inputs;
  return features;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
