#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/api/Resource.h>

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace api {

namespace {

std::string failure_log_path() {
  const char* env = std::getenv("PYTORCH_VULKAN_FAILURE_LOG");
  return env ? std::string(env) : std::string();
}

std::mutex& failure_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

} // namespace

const char* vulkan_failure_class_name(
    const VulkanFailureClass failure_class) {
  switch (failure_class) {
    case VulkanFailureClass::TensorStateInvalid:
      return "TensorStateInvalid";
    case VulkanFailureClass::MetadataViewInvalid:
      return "MetadataViewInvalid";
    case VulkanFailureClass::RawCopyIllegal:
      return "RawCopyIllegal";
    case VulkanFailureClass::ReplayViewStale:
      return "ReplayViewStale";
    case VulkanFailureClass::RouteHardFail:
      return "RouteHardFail";
    case VulkanFailureClass::KernelIncorrect:
      return "KernelIncorrect";
    case VulkanFailureClass::DeviceLost:
      return "DeviceLost";
    case VulkanFailureClass::Unsupported:
      return "Unsupported";
    case VulkanFailureClass::ReplayHangRisk:
      return "ReplayHangRisk";
    case VulkanFailureClass::Unknown:
      return "Unknown";
  }
  return "Unknown";
}

bool vulkan_failure_logging_enabled() {
  return !failure_log_path().empty();
}

std::string format_vulkan_failure(
    const VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail) {
  std::ostringstream out;
  out << "Vulkan failure"
      << " failure_class=" << vulkan_failure_class_name(failure_class);
  if (op_name && op_name[0] != '\0') {
    out << " op=" << op_name;
  }
  if (reason && reason[0] != '\0') {
    out << " reason=" << reason;
  }
  if (!current_allocation_label().empty()) {
    out << " caller=" << current_allocation_label();
  }
  if (!current_runtime_label().empty()) {
    out << " runtime=" << current_runtime_label();
  }
  if (!detail.empty()) {
    out << " detail={" << detail << '}';
  }
  return out.str();
}

void log_vulkan_failure(
    const VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail) {
  if (!vulkan_failure_logging_enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(failure_log_mutex());
  std::ofstream out(failure_log_path(), std::ios::app);
  out << format_vulkan_failure(failure_class, op_name, reason, detail) << '\n';
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
