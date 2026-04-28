#pragma once

#ifdef USE_VULKAN_API

#include <string>

namespace at {
namespace native {
namespace vulkan {
namespace api {

enum class VulkanFailureClass {
  TensorStateInvalid,
  MetadataViewInvalid,
  RawCopyIllegal,
  ReplayViewStale,
  RouteHardFail,
  KernelIncorrect,
  DeviceLost,
  Unsupported,
  ReplayHangRisk,
  Unknown,
};

const char* vulkan_failure_class_name(VulkanFailureClass failure_class);

bool vulkan_failure_logging_enabled();

void log_vulkan_failure(
    VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail = std::string());

std::string format_vulkan_failure(
    VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail = std::string());

std::string report_vulkan_failure(
    VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail = std::string());

[[noreturn]] void fail_vulkan(
    VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail = std::string());

void check_vulkan(
    bool condition,
    VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail = std::string());

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
