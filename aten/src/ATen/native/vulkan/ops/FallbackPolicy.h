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
uint64_t vulkan_deferred_value_creation_count();
int64_t begin_vulkan_graph_execution_scope();
std::vector<int64_t> end_vulkan_graph_execution_scope(int64_t token);
void guard_vulkan_deferred_value_registration(const char* producer);
void reset_vulkan_fallback_counters();
std::vector<int64_t> vulkan_fallback_phase_counters_snapshot();
std::vector<int64_t> vulkan_timed_fallback_phase_counters_snapshot();
void reset_vulkan_fallback_phase_counters();
void set_vulkan_fallback_phase(VulkanFallbackPhase phase);
void set_vulkan_benchmark_timed_region(bool enabled);
VulkanFallbackPhase current_vulkan_fallback_phase();
bool current_vulkan_benchmark_timed_region();
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
