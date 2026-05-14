#pragma once

#ifdef USE_VULKAN_API

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>

#include <cstdint>
#include <string>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

enum class VulkanCpuFallbackKind {
  OpFallback,
  SyncReadback,
};

enum class VulkanFallbackPhase : uint8_t {
  Unknown = 0,
  ModelSetup,
  OwnerContextCreate,
  OwnerForward,
  DecoderSetup,
  PositionalEmbeddingSetup,
  Readback,
  TestHarness,
};

uint64_t vulkan_cpu_fallback_count();
uint64_t vulkan_sync_readback_count();
void reset_vulkan_fallback_counters();
std::vector<int64_t> vulkan_fallback_phase_counters_snapshot();
void reset_vulkan_fallback_phase_counters();
void set_vulkan_fallback_phase(VulkanFallbackPhase phase);
VulkanFallbackPhase current_vulkan_fallback_phase();
const char* vulkan_fallback_phase_name(VulkanFallbackPhase phase);

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
