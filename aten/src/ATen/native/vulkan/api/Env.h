#pragma once

#include <c10/macros/Export.h>

#include <cstddef>

namespace at::native::vulkan::api {

enum class VulkanEnvFlagKind {
  Logging,
  Profiling,
  CorrectnessDiagnostic,
  ActiveImplementationGate,
  HardwareCapabilityGate,
  CacheLimit,
};

struct VulkanEnvFlagSpec final {
  const char* name;
  VulkanEnvFlagKind kind;
  const char* reason;
  const char* coverage;
};

TORCH_API const VulkanEnvFlagSpec* registered_vulkan_env_flags(size_t* count);
TORCH_API const VulkanEnvFlagSpec* find_vulkan_env_flag(const char* name);

} // namespace at::native::vulkan::api
