#pragma once

#ifdef USE_VULKAN_API

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>

#include <cstdint>
#include <string>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

enum class VulkanCpuFallbackKind {
  OpFallback,
  SyncReadback,
};

uint64_t vulkan_cpu_fallback_count();
uint64_t vulkan_sync_readback_count();
void reset_vulkan_fallback_counters();

void report_vulkan_cpu_fallback(
    const char* op_name,
    const char* reason,
    ArrayRef<Tensor> tensors = {},
    VulkanCpuFallbackKind kind = VulkanCpuFallbackKind::OpFallback);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
