#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Diagnostics.h>

#include <algorithm>
#include <bitset>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <utility>

namespace at {
namespace native {
namespace vulkan {
namespace api {

namespace {

void query_cooperative_matrix_support(
    VkInstance instance,
    VkPhysicalDevice physical_device_handle,
    const bool has_cooperative_matrix,
    uint32_t& cooperative_matrix_supported_stages,
    std::vector<CooperativeMatrixProperty>& cooperative_matrix_properties);

void query_subgroup_size_control_support(
    VkPhysicalDevice physical_device_handle,
    uint32_t& min_subgroup_size,
    uint32_t& max_subgroup_size,
    uint32_t& max_compute_workgroup_subgroups,
    uint32_t& required_subgroup_size_stages);

} // namespace

PhysicalDevice::PhysicalDevice(
    VkInstance instance,
    VkPhysicalDevice physical_device_handle)
    : handle(physical_device_handle),
      properties{},
      memory_properties{},
      queue_families{},
      api_version(0u),
      has_vulkan_1_3(false),
      num_compute_queues(0),
      has_unified_memory(false),
      has_timestamps(false),
      has_maintenance4(false),
      has_synchronization2(false),
      has_shader_zero_initialize_workgroup_memory(false),
      has_shader_integer_dot_product(false),
      has_pipeline_creation_cache_control(false),
      has_timeline_semaphore(false),
      has_shader_bfloat16(false),
      has_shader_int8(false),
      has_storage_buffer_8bit(false),
      has_cooperative_matrix(false),
      has_subgroup_size_control(false),
      has_compute_full_subgroups(false),
      min_subgroup_size(0u),
      max_subgroup_size(0u),
      max_compute_workgroup_subgroups(0u),
      required_subgroup_size_stages(0u),
      cooperative_matrix_supported_stages(0u),
      cooperative_matrix_properties{},
      timestamp_period(0.0f) {
  // Extract physical device properties
  vkGetPhysicalDeviceProperties(handle, &properties);
  vkGetPhysicalDeviceMemoryProperties(handle, &memory_properties);
  api_version = properties.apiVersion;
  has_vulkan_1_3 = api_version >= VK_API_VERSION_1_3;
  has_timestamps = properties.limits.timestampComputeAndGraphics;
  timestamp_period = properties.limits.timestampPeriod;

#ifdef VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME
  VkPhysicalDeviceShaderBfloat16FeaturesKHR shader_bfloat16_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
      nullptr,
      VK_FALSE,
      VK_FALSE,
      VK_FALSE,
  };
#endif
  VkPhysicalDeviceVulkan11Features vulkan11_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      nullptr,
  };
  VkPhysicalDeviceVulkan12Features vulkan12_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      nullptr,
  };
  VkPhysicalDeviceVulkan13Features vulkan13_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      nullptr,
  };
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_matrix_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
      nullptr,
      VK_FALSE,
      VK_FALSE,
  };
#endif
  void* features2_pnext = nullptr;
#ifdef VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME
  shader_bfloat16_features.pNext = features2_pnext;
  features2_pnext = &shader_bfloat16_features;
#endif
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
  cooperative_matrix_features.pNext = features2_pnext;
  features2_pnext = &cooperative_matrix_features;
#endif
  vulkan13_features.pNext = features2_pnext;
  features2_pnext = &vulkan13_features;
  vulkan12_features.pNext = features2_pnext;
  features2_pnext = &vulkan12_features;
  vulkan11_features.pNext = features2_pnext;
  features2_pnext = &vulkan11_features;
  VkPhysicalDeviceFeatures2 features2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      features2_pnext,
      {},
  };
  vkGetPhysicalDeviceFeatures2(handle, &features2);
#ifdef VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME
  has_shader_bfloat16 =
      shader_bfloat16_features.shaderBFloat16Type == VK_TRUE;
#endif
  has_shader_int8 = vulkan12_features.shaderInt8 == VK_TRUE;
  has_storage_buffer_8bit =
      vulkan12_features.storageBuffer8BitAccess == VK_TRUE;
  has_maintenance4 = vulkan13_features.maintenance4 == VK_TRUE;
  has_synchronization2 = vulkan13_features.synchronization2 == VK_TRUE;
  has_shader_zero_initialize_workgroup_memory =
      vulkan13_features.shaderZeroInitializeWorkgroupMemory == VK_TRUE;
  has_shader_integer_dot_product =
      vulkan13_features.shaderIntegerDotProduct == VK_TRUE;
  has_pipeline_creation_cache_control =
      vulkan13_features.pipelineCreationCacheControl == VK_TRUE;
  has_timeline_semaphore = vulkan12_features.timelineSemaphore == VK_TRUE;
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
  has_cooperative_matrix =
      cooperative_matrix_features.cooperativeMatrix == VK_TRUE;
  query_cooperative_matrix_support(
      instance,
      handle,
      has_cooperative_matrix,
      cooperative_matrix_supported_stages,
      cooperative_matrix_properties);
#endif
  has_subgroup_size_control = vulkan13_features.subgroupSizeControl == VK_TRUE;
  has_compute_full_subgroups =
      vulkan13_features.computeFullSubgroups == VK_TRUE;
  query_subgroup_size_control_support(
      handle,
      min_subgroup_size,
      max_subgroup_size,
      max_compute_workgroup_subgroups,
      required_subgroup_size_stages);

  // Check if there are any memory types have both the HOST_VISIBLE and the
  // DEVICE_LOCAL property flags
  const VkMemoryPropertyFlags unified_memory_flags =
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  for (size_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
    if (
        (memory_properties.memoryTypes[i].propertyFlags &
         unified_memory_flags) == unified_memory_flags) {
      has_unified_memory = true;
      break;
    }
  }

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
      handle, &queue_family_count, nullptr);

  queue_families.resize(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      handle, &queue_family_count, queue_families.data());

  // Find the total number of compute queues
  for (const VkQueueFamilyProperties& p : queue_families) {
    // Check if this family has compute capability
    if (p.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      num_compute_queues += p.queueCount;
    }
  }
}

namespace {

std::string get_shader_stage_flags_str(const uint32_t flags) {
  std::stringstream ss("|");
  if (flags & VK_SHADER_STAGE_COMPUTE_BIT) {
    ss << " COMPUTE |";
  }
  return ss.str();
}

std::string get_component_type_str(const uint32_t type) {
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
  switch (static_cast<VkComponentTypeKHR>(type)) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR:
      return "FLOAT16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR:
      return "FLOAT32";
    case VK_COMPONENT_TYPE_FLOAT64_KHR:
      return "FLOAT64";
    case VK_COMPONENT_TYPE_SINT8_KHR:
      return "SINT8";
    case VK_COMPONENT_TYPE_SINT16_KHR:
      return "SINT16";
    case VK_COMPONENT_TYPE_SINT32_KHR:
      return "SINT32";
    case VK_COMPONENT_TYPE_SINT64_KHR:
      return "SINT64";
    case VK_COMPONENT_TYPE_UINT8_KHR:
      return "UINT8";
    case VK_COMPONENT_TYPE_UINT16_KHR:
      return "UINT16";
    case VK_COMPONENT_TYPE_UINT32_KHR:
      return "UINT32";
    case VK_COMPONENT_TYPE_UINT64_KHR:
      return "UINT64";
    case VK_COMPONENT_TYPE_BFLOAT16_KHR:
      return "BFLOAT16";
    default:
      break;
  }
#endif
  return "UNKNOWN(" + std::to_string(type) + ')';
}

std::string get_scope_str(const uint32_t scope) {
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
  switch (static_cast<VkScopeKHR>(scope)) {
    case VK_SCOPE_DEVICE_KHR:
      return "DEVICE";
    case VK_SCOPE_WORKGROUP_KHR:
      return "WORKGROUP";
    case VK_SCOPE_SUBGROUP_KHR:
      return "SUBGROUP";
    case VK_SCOPE_QUEUE_FAMILY_KHR:
      return "QUEUE_FAMILY";
    default:
      break;
  }
#endif
  return "UNKNOWN(" + std::to_string(scope) + ')';
}

void query_cooperative_matrix_support(
    VkInstance instance,
    VkPhysicalDevice physical_device_handle,
    const bool has_cooperative_matrix,
    uint32_t& cooperative_matrix_supported_stages,
    std::vector<CooperativeMatrixProperty>& cooperative_matrix_properties) {
  cooperative_matrix_supported_stages = 0u;
  cooperative_matrix_properties.clear();

#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
  if (
      instance == VK_NULL_HANDLE || physical_device_handle == VK_NULL_HANDLE ||
      !has_cooperative_matrix) {
    return;
  }

  VkPhysicalDeviceCooperativeMatrixPropertiesKHR stage_properties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
      nullptr,
      0u,
  };
  VkPhysicalDeviceProperties2 properties2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      &stage_properties,
      {},
  };
  vkGetPhysicalDeviceProperties2(physical_device_handle, &properties2);
  cooperative_matrix_supported_stages =
      static_cast<uint32_t>(stage_properties.cooperativeMatrixSupportedStages);

  const auto get_properties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
          vkGetInstanceProcAddr(
              instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  if (!get_properties) {
    return;
  }

  uint32_t property_count = 0u;
  VkResult query_result =
      get_properties(physical_device_handle, &property_count, nullptr);
  if (
      (query_result != VK_SUCCESS && query_result != VK_INCOMPLETE) ||
      property_count == 0u) {
    return;
  }

  std::vector<VkCooperativeMatrixPropertiesKHR> properties(property_count);
  for (auto& property : properties) {
    property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    property.pNext = nullptr;
  }

  query_result =
      get_properties(physical_device_handle, &property_count, properties.data());
  if (query_result != VK_SUCCESS && query_result != VK_INCOMPLETE) {
    return;
  }

  properties.resize(property_count);
  cooperative_matrix_properties.reserve(properties.size());
  for (const auto& property : properties) {
    cooperative_matrix_properties.push_back({
        property.MSize,
        property.NSize,
        property.KSize,
        static_cast<uint32_t>(property.AType),
        static_cast<uint32_t>(property.BType),
        static_cast<uint32_t>(property.CType),
        static_cast<uint32_t>(property.ResultType),
        property.saturatingAccumulation == VK_TRUE,
        static_cast<uint32_t>(property.scope),
    });
  }
#else
  (void)instance;
  (void)physical_device_handle;
  (void)has_cooperative_matrix;
#endif
}

void query_subgroup_size_control_support(
    VkPhysicalDevice physical_device_handle,
    uint32_t& min_subgroup_size,
    uint32_t& max_subgroup_size,
    uint32_t& max_compute_workgroup_subgroups,
    uint32_t& required_subgroup_size_stages) {
  min_subgroup_size = 0u;
  max_subgroup_size = 0u;
  max_compute_workgroup_subgroups = 0u;
  required_subgroup_size_stages = 0u;

#if defined(VK_VERSION_1_3) || defined(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)
  if (physical_device_handle == VK_NULL_HANDLE) {
    return;
  }

  VkPhysicalDeviceSubgroupSizeControlProperties subgroup_properties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
      nullptr,
      0u,
      0u,
      0u,
      0u,
  };
  VkPhysicalDeviceProperties2 properties2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      &subgroup_properties,
      {},
  };
  vkGetPhysicalDeviceProperties2(physical_device_handle, &properties2);
  min_subgroup_size = subgroup_properties.minSubgroupSize;
  max_subgroup_size = subgroup_properties.maxSubgroupSize;
  max_compute_workgroup_subgroups =
      subgroup_properties.maxComputeWorkgroupSubgroups;
  required_subgroup_size_stages =
      static_cast<uint32_t>(subgroup_properties.requiredSubgroupSizeStages);
#else
  (void)physical_device_handle;
#endif
}

void find_requested_device_extensions(
    VkPhysicalDevice physical_device,
    std::vector<const char*>& enabled_extensions,
    const std::vector<const char*>& requested_extensions) {
  uint32_t device_extension_properties_count = 0;
  VK_CHECK(vkEnumerateDeviceExtensionProperties(
      physical_device, nullptr, &device_extension_properties_count, nullptr));
  std::vector<VkExtensionProperties> device_extension_properties(
      device_extension_properties_count);
  VK_CHECK(vkEnumerateDeviceExtensionProperties(
      physical_device,
      nullptr,
      &device_extension_properties_count,
      device_extension_properties.data()));

  std::vector<const char*> enabled_device_extensions;

  for (const auto& requested_extension : requested_extensions) {
    for (const auto& extension : device_extension_properties) {
      if (strcmp(requested_extension, extension.extensionName) == 0) {
        enabled_extensions.push_back(requested_extension);
        break;
      }
    }
  }
}

VkDevice create_logical_device(
    const PhysicalDevice& physical_device,
    const uint32_t num_queues_to_create,
    std::vector<Adapter::Queue>& queues,
    std::vector<uint32_t>& queue_usage) {
  // Find compute queues up to the requested number of queues

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  queue_create_infos.reserve(num_queues_to_create);

  std::vector<std::pair<uint32_t, uint32_t>> queues_to_get;
  queues_to_get.reserve(num_queues_to_create);

  uint32_t remaining_queues = num_queues_to_create;
  for (uint32_t family_i = 0; family_i < physical_device.queue_families.size();
       ++family_i) {
    const VkQueueFamilyProperties& queue_properties =
        physical_device.queue_families.at(family_i);
    // Check if this family has compute capability
    if (queue_properties.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      const uint32_t queues_to_init =
          std::min(remaining_queues, queue_properties.queueCount);

      const std::vector<float> queue_priorities(queues_to_init, 1.0f);
      queue_create_infos.push_back({
          VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, // sType
          nullptr, // pNext
          0u, // flags
          family_i, // queueFamilyIndex
          queues_to_init, // queueCount
          queue_priorities.data(), // pQueuePriorities
      });

      for (size_t queue_i = 0; queue_i < queues_to_init; ++queue_i) {
        // Use this to get the queue handle once device is created
        queues_to_get.emplace_back(family_i, queue_i);
      }
      remaining_queues -= queues_to_init;
    }
    if (remaining_queues == 0) {
      break;
    }
  }

  queues.reserve(queues_to_get.size());
  queue_usage.reserve(queues_to_get.size());

  // Create the VkDevice

  std::vector<const char*> requested_device_extensions{
#ifdef VK_KHR_portability_subset
      VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
#endif /* VK_KHR_portability_subset */
#ifdef VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME
      VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME,
#endif /* VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME */
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
      VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
#endif /* VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME */
  };

  std::vector<const char*> enabled_device_extensions;
  find_requested_device_extensions(
      physical_device.handle,
      enabled_device_extensions,
      requested_device_extensions);

#ifdef VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME
  VkPhysicalDeviceShaderBfloat16FeaturesKHR shader_bfloat16_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
      nullptr,
      VK_FALSE,
      VK_FALSE,
      VK_FALSE,
  };
#endif
  VkPhysicalDeviceVulkan11Features vulkan11_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      nullptr,
  };
  VkPhysicalDeviceVulkan12Features vulkan12_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      nullptr,
  };
  VkPhysicalDeviceVulkan13Features vulkan13_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      nullptr,
  };
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_matrix_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
      nullptr,
      VK_FALSE,
      VK_FALSE,
  };
#endif
  VkPhysicalDeviceFeatures2 enabled_features2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      nullptr,
      {},
  };

#ifdef VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME
  const bool enable_shader_bfloat16 =
      physical_device.has_shader_bfloat16 &&
      std::find(
          enabled_device_extensions.begin(),
          enabled_device_extensions.end(),
          VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) !=
          enabled_device_extensions.end();
  if (enable_shader_bfloat16) {
    shader_bfloat16_features.shaderBFloat16Type = VK_TRUE;
    enabled_features2.pNext = &shader_bfloat16_features;
  }
#endif
  if (physical_device.has_shader_int8) {
    vulkan12_features.shaderInt8 = VK_TRUE;
  }
  if (physical_device.has_storage_buffer_8bit) {
    vulkan12_features.storageBuffer8BitAccess = VK_TRUE;
  }
  vulkan12_features.timelineSemaphore =
      physical_device.has_timeline_semaphore ? VK_TRUE : VK_FALSE;
  vulkan13_features.maintenance4 =
      physical_device.has_maintenance4 ? VK_TRUE : VK_FALSE;
  vulkan13_features.synchronization2 =
      physical_device.has_synchronization2 ? VK_TRUE : VK_FALSE;
  vulkan13_features.shaderZeroInitializeWorkgroupMemory =
      physical_device.has_shader_zero_initialize_workgroup_memory ? VK_TRUE
                                                                  : VK_FALSE;
  vulkan13_features.shaderIntegerDotProduct =
      physical_device.has_shader_integer_dot_product ? VK_TRUE : VK_FALSE;
  vulkan13_features.pipelineCreationCacheControl =
      physical_device.has_pipeline_creation_cache_control ? VK_TRUE : VK_FALSE;
  if (physical_device.has_subgroup_size_control) {
    vulkan13_features.subgroupSizeControl = VK_TRUE;
    vulkan13_features.computeFullSubgroups =
        physical_device.has_compute_full_subgroups ? VK_TRUE : VK_FALSE;
  }
  vulkan13_features.pNext = enabled_features2.pNext;
  enabled_features2.pNext = &vulkan13_features;
  vulkan12_features.pNext = enabled_features2.pNext;
  enabled_features2.pNext = &vulkan12_features;
  vulkan11_features.pNext = enabled_features2.pNext;
  enabled_features2.pNext = &vulkan11_features;
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
  const bool enable_cooperative_matrix =
      physical_device.has_cooperative_matrix &&
      std::find(
          enabled_device_extensions.begin(),
          enabled_device_extensions.end(),
          VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) !=
          enabled_device_extensions.end();
  if (enable_cooperative_matrix) {
    cooperative_matrix_features.cooperativeMatrix = VK_TRUE;
    cooperative_matrix_features.cooperativeMatrixRobustBufferAccess = VK_TRUE;
    cooperative_matrix_features.pNext = enabled_features2.pNext;
    enabled_features2.pNext = &cooperative_matrix_features;
  }
#endif

  const VkDeviceCreateInfo device_create_info{
      VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, // sType
      enabled_features2.pNext ? &enabled_features2 : nullptr, // pNext
      0u, // flags
      static_cast<uint32_t>(queue_create_infos.size()), // queueCreateInfoCount
      queue_create_infos.data(), // pQueueCreateInfos
      0u, // enabledLayerCount
      nullptr, // ppEnabledLayerNames
      static_cast<uint32_t>(
          enabled_device_extensions.size()), // enabledExtensionCount
      enabled_device_extensions.data(), // ppEnabledExtensionNames
      nullptr, // pEnabledFeatures
  };

  VkDevice handle = nullptr;
  VK_CHECK(vkCreateDevice(
      physical_device.handle, &device_create_info, nullptr, &handle));

#ifdef USE_VULKAN_VOLK
  volkLoadDevice(handle);
#endif /* USE_VULKAN_VOLK */
  // Obtain handles for the created queues and initialize queue usage heuristic

  for (const std::pair<uint32_t, uint32_t>& queue_idx : queues_to_get) {
    VkQueue queue_handle = VK_NULL_HANDLE;
    VkQueueFlags flags =
        physical_device.queue_families.at(queue_idx.first).queueFlags;
    vkGetDeviceQueue(handle, queue_idx.first, queue_idx.second, &queue_handle);
    queues.push_back({queue_idx.first, queue_idx.second, flags, queue_handle});
    // Initial usage value
    queue_usage.push_back(0);
  }

  return handle;
}

// Print utils

std::string get_device_type_str(const VkPhysicalDeviceType type) {
  switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return "INTEGRATED_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return "DISCRETE_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return "VIRTUAL_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return "CPU";
    default:
      return "UNKNOWN";
  }
}

std::string get_memory_properties_str(const VkMemoryPropertyFlags flags) {
  std::bitset<10> values(flags);
  std::stringstream ss("|");
  if (values[0]) {
    ss << " DEVICE_LOCAL |";
  }
  if (values[1]) {
    ss << " HOST_VISIBLE |";
  }
  if (values[2]) {
    ss << " HOST_COHERENT |";
  }
  if (values[3]) {
    ss << " HOST_CACHED |";
  }
  if (values[4]) {
    ss << " LAZILY_ALLOCATED |";
  }

  return ss.str();
}

std::string get_queue_family_properties_str(const VkQueueFlags flags) {
  std::bitset<10> values(flags);
  std::stringstream ss("|");
  if (values[0]) {
    ss << " GRAPHICS |";
  }
  if (values[1]) {
    ss << " COMPUTE |";
  }
  if (values[2]) {
    ss << " TRANSFER |";
  }

  return ss.str();
}

} // namespace

//
// DeviceHandle
//

DeviceHandle::DeviceHandle(VkDevice device) : handle_(device) {}

DeviceHandle::DeviceHandle(DeviceHandle&& other) noexcept
    : handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

DeviceHandle::~DeviceHandle() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyDevice(handle_, nullptr);
}

//
// Adapter
//

Adapter::Adapter(
    VkInstance instance,
    PhysicalDevice physical_device,
    const uint32_t num_queues)
    : queue_usage_mutex_{},
      physical_device_(std::move(physical_device)),
      queues_{},
      queue_usage_{},
      queue_mutexes_{},
      instance_(instance),
      device_(create_logical_device(
          physical_device_,
          num_queues,
          queues_,
          queue_usage_)),
      shader_layout_cache_(device_.handle_),
      shader_cache_(device_.handle_),
      pipeline_layout_cache_(device_.handle_),
      compute_pipeline_cache_(device_.handle_),
      sampler_cache_(device_.handle_),
      vma_(instance_, physical_device_.handle, device_.handle_) {}

Adapter::Queue Adapter::request_queue() {
  // Lock the mutex as multiple threads can request a queue at the same time
  std::lock_guard<std::mutex> lock(queue_usage_mutex_);

  uint32_t min_usage = UINT32_MAX;
  uint32_t min_used_i = 0;
  for (size_t i = 0; i < queues_.size(); ++i) {
    if (queue_usage_[i] < min_usage) {
      min_used_i = i;
      min_usage = queue_usage_[i];
    }
  }
  queue_usage_[min_used_i] += 1;

  return queues_[min_used_i];
}

void Adapter::return_queue(Adapter::Queue& compute_queue) {
  for (size_t i = 0; i < queues_.size(); ++i) {
    if ((queues_[i].family_index == compute_queue.family_index) &&
        (queues_[i].queue_index == compute_queue.queue_index)) {
      std::lock_guard<std::mutex> lock(queue_usage_mutex_);
      queue_usage_[i] -= 1;
      break;
    }
  }
}

void Adapter::submit_cmd(
    const Adapter::Queue& device_queue,
    VkCommandBuffer cmd,
    VkFence fence) {
  const VkSubmitInfo submit_info{
      VK_STRUCTURE_TYPE_SUBMIT_INFO, // sType
      nullptr, // pNext
      0u, // waitSemaphoreCount
      nullptr, // pWaitSemaphores
      nullptr, // pWaitDstStageMask
      1u, // commandBufferCount
      &cmd, // pCommandBuffers
      0u, // signalSemaphoreCount
      nullptr, // pSignalSemaphores
  };

  std::lock_guard<std::mutex> queue_lock(
      queue_mutexes_[device_queue.queue_index % NUM_QUEUE_MUTEXES]);

  const VkResult submit_result =
      vkQueueSubmit(device_queue.handle, 1u, &submit_info, fence);
  VK_CHECK(submit_result);
}

void Adapter::submit_cmds(
    const Adapter::Queue& device_queue,
    const std::vector<VkCommandBuffer>& cmds,
    VkFence fence) {
  const VkSubmitInfo submit_info{
      VK_STRUCTURE_TYPE_SUBMIT_INFO, // sType
      nullptr, // pNext
      0u, // waitSemaphoreCount
      nullptr, // pWaitSemaphores
      nullptr, // pWaitDstStageMask
      utils::safe_downcast<uint32_t>(cmds.size()), // commandBufferCount
      cmds.data(), // pCommandBuffers
      0u, // signalSemaphoreCount
      nullptr, // pSignalSemaphores
  };

  const VkResult submit_result =
      vkQueueSubmit(device_queue.handle, 1u, &submit_info, fence);
  VK_CHECK(submit_result);
}

void Adapter::submit_cmd_timeline(
    const Adapter::Queue& device_queue,
    VkCommandBuffer cmd,
    const std::vector<VkSemaphore>& wait_semaphores,
    const std::vector<uint64_t>& wait_values,
    const std::vector<VkPipelineStageFlags>& wait_stages,
    VkSemaphore signal_semaphore,
    uint64_t signal_value,
    VkFence fence) {
  VK_CHECK_COND(
      wait_semaphores.size() == wait_values.size() &&
          wait_semaphores.size() == wait_stages.size(),
      "Vulkan timeline submit wait arrays must have matching sizes.");
  VK_CHECK_COND(
      signal_semaphore != VK_NULL_HANDLE,
      "Vulkan timeline submit requires a signal semaphore.");

  const VkTimelineSemaphoreSubmitInfo timeline_info{
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      nullptr,
      utils::safe_downcast<uint32_t>(wait_values.size()),
      wait_values.empty() ? nullptr : wait_values.data(),
      1u,
      &signal_value,
  };

  const VkSubmitInfo submit_info{
      VK_STRUCTURE_TYPE_SUBMIT_INFO,
      &timeline_info,
      utils::safe_downcast<uint32_t>(wait_semaphores.size()),
      wait_semaphores.empty() ? nullptr : wait_semaphores.data(),
      wait_stages.empty() ? nullptr : wait_stages.data(),
      1u,
      &cmd,
      1u,
      &signal_semaphore,
  };

  std::lock_guard<std::mutex> queue_lock(
      queue_mutexes_[device_queue.queue_index % NUM_QUEUE_MUTEXES]);
  VK_CHECK(vkQueueSubmit(device_queue.handle, 1u, &submit_info, fence));
}

std::string Adapter::stringize() const {
  std::stringstream ss;

  VkPhysicalDeviceProperties properties = physical_device_.properties;
  uint32_t v_major = VK_VERSION_MAJOR(properties.apiVersion);
  uint32_t v_minor = VK_VERSION_MINOR(properties.apiVersion);
  std::string device_type = get_device_type_str(properties.deviceType);
  VkPhysicalDeviceLimits limits = properties.limits;

  ss << '{' << std::endl;
  ss << "  Physical Device Info {" << std::endl;
  ss << "    apiVersion:    " << v_major << '.' << v_minor << std::endl;
  ss << "    driverversion: " << properties.driverVersion << std::endl;
  ss << "    deviceType:    " << device_type << std::endl;
  ss << "    deviceName:    " << properties.deviceName << std::endl;
  ss << "    vulkan13:      "
     << (physical_device_.has_vulkan_1_3 ? "true" : "false") << std::endl;
  ss << "    maintenance4:  "
     << (physical_device_.has_maintenance4 ? "true" : "false") << std::endl;
  ss << "    synchronization2: "
     << (physical_device_.has_synchronization2 ? "true" : "false")
     << std::endl;
  ss << "    zeroInitializeWorkgroupMemory: "
     << (physical_device_.has_shader_zero_initialize_workgroup_memory ? "true"
                                                                      : "false")
     << std::endl;
  ss << "    shaderIntegerDotProduct: "
     << (physical_device_.has_shader_integer_dot_product ? "true" : "false")
     << std::endl;
  ss << "    pipelineCreationCacheControl: "
     << (physical_device_.has_pipeline_creation_cache_control ? "true"
                                                              : "false")
     << std::endl;
  ss << "    shaderBFloat16: "
     << (physical_device_.has_shader_bfloat16 ? "true" : "false")
     << std::endl;
  ss << "    shaderInt8:    "
     << (physical_device_.has_shader_int8 ? "true" : "false") << std::endl;
  ss << "    storage8Bit:   "
     << (physical_device_.has_storage_buffer_8bit ? "true" : "false")
     << std::endl;
  ss << "    cooperativeMatrix: "
     << (physical_device_.has_cooperative_matrix ? "true" : "false")
     << std::endl;
  ss << "    subgroupSizeControl: "
     << (physical_device_.has_subgroup_size_control ? "true" : "false")
     << std::endl;
  ss << "    computeFullSubgroups: "
     << (physical_device_.has_compute_full_subgroups ? "true" : "false")
     << std::endl;
  ss << "    minSubgroupSize: " << physical_device_.min_subgroup_size
     << std::endl;
  ss << "    maxSubgroupSize: " << physical_device_.max_subgroup_size
     << std::endl;
  ss << "    maxComputeWorkgroupSubgroups: "
     << physical_device_.max_compute_workgroup_subgroups << std::endl;
  ss << "    requiredSubgroupSizeStages: "
     << get_shader_stage_flags_str(physical_device_.required_subgroup_size_stages)
     << std::endl;
  ss << "    cooperativeMatrixSupportedStages: "
     << get_shader_stage_flags_str(
            physical_device_.cooperative_matrix_supported_stages)
     << std::endl;
  ss << "    cooperativeMatrixPropertyCount: "
     << physical_device_.cooperative_matrix_properties.size() << std::endl;
  if (!physical_device_.cooperative_matrix_properties.empty()) {
    ss << "    Cooperative Matrix Properties [" << std::endl;
    for (const auto& property : physical_device_.cooperative_matrix_properties) {
      ss << "      M=" << property.m_size << " N=" << property.n_size
         << " K=" << property.k_size
         << " A=" << get_component_type_str(property.a_type)
         << " B=" << get_component_type_str(property.b_type)
         << " C=" << get_component_type_str(property.c_type)
         << " Result=" << get_component_type_str(property.result_type)
         << " scope=" << get_scope_str(property.scope)
         << " saturating="
         << (property.saturating_accumulation ? "true" : "false")
         << std::endl;
    }
    ss << "    ]" << std::endl;
  }

#define PRINT_LIMIT_PROP(name)                                         \
  ss << "      " << std::left << std::setw(36) << #name << limits.name \
     << std::endl;

#define PRINT_LIMIT_PROP_VEC3(name)                                       \
  ss << "      " << std::left << std::setw(36) << #name << limits.name[0] \
     << ',' << limits.name[1] << ',' << limits.name[2] << std::endl;

  ss << "    Physical Device Limits {" << std::endl;
  PRINT_LIMIT_PROP(maxImageDimension1D);
  PRINT_LIMIT_PROP(maxImageDimension2D);
  PRINT_LIMIT_PROP(maxImageDimension3D);
  PRINT_LIMIT_PROP(maxTexelBufferElements);
  PRINT_LIMIT_PROP(maxPushConstantsSize);
  PRINT_LIMIT_PROP(maxMemoryAllocationCount);
  PRINT_LIMIT_PROP(maxSamplerAllocationCount);
  PRINT_LIMIT_PROP(maxComputeSharedMemorySize);
  PRINT_LIMIT_PROP_VEC3(maxComputeWorkGroupCount);
  PRINT_LIMIT_PROP(maxComputeWorkGroupInvocations);
  PRINT_LIMIT_PROP_VEC3(maxComputeWorkGroupSize);
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ;

  const VkPhysicalDeviceMemoryProperties& mem_props =
      physical_device_.memory_properties;

  ss << "  Memory Info {" << std::endl;
  ss << "    Memory Types [" << std::endl;
  for (size_t i = 0; i < mem_props.memoryTypeCount; ++i) {
    ss << "      "
       << " [Heap " << mem_props.memoryTypes[i].heapIndex << "] "
       << get_memory_properties_str(mem_props.memoryTypes[i].propertyFlags)
       << std::endl;
  }
  ss << "    ]" << std::endl;
  ss << "    Memory Heaps [" << std::endl;
  for (size_t i = 0; i < mem_props.memoryHeapCount; ++i) {
    ss << "      " << mem_props.memoryHeaps[i].size << std::endl;
  }
  ss << "    ]" << std::endl;
  ss << "  }" << std::endl;

  ss << "  Queue Families {" << std::endl;
  for (const VkQueueFamilyProperties& queue_family_props :
       physical_device_.queue_families) {
    ss << "    (" << queue_family_props.queueCount << " Queues) "
       << get_queue_family_properties_str(queue_family_props.queueFlags)
       << std::endl;
  }
  ss << "  }" << std::endl;
  ss << "  VkDevice: " << device_.handle_ << std::endl;
  ss << "  Compute Queues [" << std::endl;
  for (const Adapter::Queue& compute_queue : queues_) {
    ss << "    Family " << compute_queue.family_index << ", Queue "
       << compute_queue.queue_index << ": " << compute_queue.handle
       << std::endl;
    ;
  }
  ss << "  ]" << std::endl;
  ss << '}';

  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Adapter& adapter) {
  os << adapter.stringize() << std::endl;
  return os;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
