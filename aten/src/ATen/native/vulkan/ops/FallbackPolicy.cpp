#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/api/Sync.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <atomic>
#include <cstdlib>
#include <sstream>
#include <string>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

std::atomic<uint64_t>& cpu_fallback_counter() {
  static std::atomic<uint64_t> counter{0};
  return counter;
}

std::atomic<uint64_t>& sync_readback_counter() {
  static std::atomic<uint64_t> counter{0};
  return counter;
}

bool env_flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  const std::string s(value);
  return !(s.empty() || s == "0" || s == "false" || s == "False" ||
           s == "FALSE" || s == "off" || s == "OFF");
}

std::string env_string(const char* name) {
  const char* value = std::getenv(name);
  return value ? std::string(value) : std::string();
}

bool cpu_fallback_log_enabled(const VulkanCpuFallbackKind kind) {
  return env_flag_enabled("PYTORCH_VULKAN_LOG_CPU_FALLBACK") ||
      env_flag_enabled("PYTORCH_VULKAN_LOG_FALLBACK") ||
      (kind == VulkanCpuFallbackKind::SyncReadback &&
       env_flag_enabled("PYTORCH_VULKAN_LOG_SYNC_READBACK"));
}

bool cpu_fallback_warn_enabled(const VulkanCpuFallbackKind kind) {
  const std::string policy = env_string("PYTORCH_VULKAN_CPU_FALLBACK");
  return policy == "warn" || policy == "WARN" || cpu_fallback_log_enabled(kind);
}

bool cpu_fallback_error_enabled(const VulkanCpuFallbackKind kind) {
  if (env_flag_enabled("PYTORCH_VULKAN_NO_CPU_FALLBACK")) {
    return true;
  }
  if (
      kind == VulkanCpuFallbackKind::SyncReadback &&
      env_flag_enabled("PYTORCH_VULKAN_FAIL_ON_SYNC_READBACK")) {
    return true;
  }
  const std::string policy = env_string("PYTORCH_VULKAN_CPU_FALLBACK");
  return policy == "error" || policy == "ERROR" || policy == "fail" ||
      policy == "FAIL";
}

const char* fallback_kind_name(const VulkanCpuFallbackKind kind) {
  switch (kind) {
    case VulkanCpuFallbackKind::OpFallback:
      return "cpu_fallback";
    case VulkanCpuFallbackKind::SyncReadback:
      return "sync_readback";
  }
  return "cpu_fallback";
}

std::string tensor_detail(const Tensor& tensor) {
  std::ostringstream out;
  out << "device=" << tensor.device() << " dtype=" << tensor.scalar_type()
      << " sizes=" << tensor.sizes();
  if (tensor.is_vulkan()) {
    out << " layout=" << tensor.layout();
  }
  return out.str();
}

std::string fallback_detail(
    const VulkanCpuFallbackKind kind,
    ArrayRef<Tensor> tensors) {
  std::ostringstream out;
  out << "kind=" << fallback_kind_name(kind);
  for (const auto i : c10::irange(tensors.size())) {
    out << " tensor" << i << "={" << tensor_detail(tensors[i]) << '}';
  }
  return out.str();
}

bool has_vulkan_tensor(ArrayRef<Tensor> tensors) {
  for (const Tensor& tensor : tensors) {
    if (tensor.defined() && tensor.is_vulkan()) {
      return true;
    }
  }
  return false;
}

} // namespace

uint64_t vulkan_cpu_fallback_count() {
  return cpu_fallback_counter().load(std::memory_order_relaxed);
}

uint64_t vulkan_sync_readback_count() {
  return sync_readback_counter().load(std::memory_order_relaxed);
}

void reset_vulkan_fallback_counters() {
  cpu_fallback_counter().store(0, std::memory_order_relaxed);
  sync_readback_counter().store(0, std::memory_order_relaxed);
}

void report_vulkan_cpu_fallback(
    const char* op_name,
    const char* reason,
    ArrayRef<Tensor> tensors,
    const VulkanCpuFallbackKind kind) {
  if (!tensors.empty() && !has_vulkan_tensor(tensors)) {
    return;
  }

  if (kind == VulkanCpuFallbackKind::SyncReadback) {
    sync_readback_counter().fetch_add(1, std::memory_order_relaxed);
    api::vulkan_sync_counters().fallback_sync_readback_count.fetch_add(
        1, std::memory_order_relaxed);
  } else {
    cpu_fallback_counter().fetch_add(1, std::memory_order_relaxed);
  }

  const std::string detail = fallback_detail(kind, tensors);
  const std::string message = api::format_vulkan_failure(
      api::VulkanFailureClass::Unsupported, op_name, reason, detail);
  if (cpu_fallback_warn_enabled(kind)) {
    TORCH_WARN(message);
  }
  if (cpu_fallback_log_enabled(kind)) {
    api::log_vulkan_failure(
        api::VulkanFailureClass::Unsupported, op_name, reason, detail);
  }
  if (cpu_fallback_error_enabled(kind)) {
    api::fail_vulkan(
        api::VulkanFailureClass::Unsupported, op_name, reason, detail);
  }
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
