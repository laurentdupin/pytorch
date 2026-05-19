#include <ATen/native/vulkan/api/Sync.h>

#ifdef USE_VULKAN_API

#include <algorithm>
#include <map>
#include <mutex>
#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace api {

namespace {

thread_local VulkanVisionStackPhase g_vision_stack_phase =
    VulkanVisionStackPhase::Unknown;
thread_local int64_t g_vision_stack_block_index = -1;

std::mutex& stack_aggregate_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<std::string, uint64_t>& stack_dispatch_aggregate() {
  static std::map<std::string, uint64_t> aggregate;
  return aggregate;
}

struct StackAllocationValue final {
  uint64_t count = 0u;
  uint64_t bytes = 0u;
  uint64_t peak_live_estimate_bytes = 0u;
};

std::map<std::string, StackAllocationValue>& stack_allocation_aggregate() {
  static std::map<std::string, StackAllocationValue> aggregate;
  return aggregate;
}

std::string format_sizes(const std::vector<int64_t>& values) {
  std::ostringstream stream;
  stream << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << ',';
    }
    stream << values[i];
  }
  stream << ']';
  return stream.str();
}

} // namespace

VulkanSyncCounters& vulkan_sync_counters() {
  static VulkanSyncCounters counters;
  return counters;
}

VulkanVisionStackPhaseScope::VulkanVisionStackPhaseScope(
    VulkanVisionStackPhase phase)
    : previous_(g_vision_stack_phase) {
  g_vision_stack_phase = phase;
}

VulkanVisionStackPhaseScope::~VulkanVisionStackPhaseScope() {
  g_vision_stack_phase = previous_;
}

VulkanVisionStackBlockScope::VulkanVisionStackBlockScope(
    const int64_t block_index)
    : previous_(g_vision_stack_block_index) {
  g_vision_stack_block_index = block_index;
}

VulkanVisionStackBlockScope::~VulkanVisionStackBlockScope() {
  g_vision_stack_block_index = previous_;
}

void reset_vulkan_sync_counters() {
  VulkanSyncCounters& counters = vulkan_sync_counters();
  counters.compute_dispatch_count.store(0u, std::memory_order_relaxed);
  counters.submit_compute_job_count.store(0u, std::memory_order_relaxed);
  counters.stream_submit_count.store(0u, std::memory_order_relaxed);
  counters.event_record_count.store(0u, std::memory_order_relaxed);
  counters.event_block_count.store(0u, std::memory_order_relaxed);
  counters.event_wait_count.store(0u, std::memory_order_relaxed);
  counters.retire_poll_count.store(0u, std::memory_order_relaxed);
  counters.retired_resource_count.store(0u, std::memory_order_relaxed);
  counters.queue_wait_idle_count.store(0u, std::memory_order_relaxed);
  counters.forced_sync_count.store(0u, std::memory_order_relaxed);
  counters.fallback_sync_readback_count.store(0u, std::memory_order_relaxed);
  counters.allocation_record_stream_count.store(0u, std::memory_order_relaxed);
  counters.allocation_reuse_deferred_count.store(0u, std::memory_order_relaxed);
  counters.allocation_reuse_after_timeline_count.store(
      0u, std::memory_order_relaxed);
  counters.forced_sync_explicit_synchronize_count.store(
      0u, std::memory_order_relaxed);
  counters.forced_sync_tensor_cpu_readback_count.store(
      0u, std::memory_order_relaxed);
  counters.forced_sync_event_synchronize_count.store(
      0u, std::memory_order_relaxed);
  counters.forced_sync_retire_queue_drain_count.store(
      0u, std::memory_order_relaxed);
  counters.forced_sync_gpu_timestamp_query_reset_count.store(
      0u, std::memory_order_relaxed);
  counters.forced_sync_fallback_policy_readback_count.store(
      0u, std::memory_order_relaxed);
  counters.forced_sync_unknown_count.store(0u, std::memory_order_relaxed);
}

const char* vision_stack_phase_name(const VulkanVisionStackPhase phase) {
  switch (phase) {
    case VulkanVisionStackPhase::Unknown:
      return "unknown";
    case VulkanVisionStackPhase::StackEntry:
      return "stack_entry";
    case VulkanVisionStackPhase::BlockEntry:
      return "block_entry";
    case VulkanVisionStackPhase::Norm1:
      return "norm1";
    case VulkanVisionStackPhase::QkvLinear:
      return "qkv_linear";
    case VulkanVisionStackPhase::QkvTransform:
      return "qkv_transform";
    case VulkanVisionStackPhase::Attention:
      return "attention";
    case VulkanVisionStackPhase::ProjLinear:
      return "proj_linear";
    case VulkanVisionStackPhase::Residual1:
      return "residual1";
    case VulkanVisionStackPhase::Norm2:
      return "norm2";
    case VulkanVisionStackPhase::Fc1Gelu:
      return "fc1_gelu";
    case VulkanVisionStackPhase::Fc2:
      return "fc2";
    case VulkanVisionStackPhase::Residual2:
      return "residual2";
    case VulkanVisionStackPhase::IntermediateCapture:
      return "intermediate_capture";
    case VulkanVisionStackPhase::StackExit:
      return "stack_exit";
  }
  return "unknown";
}

const char* stack_tensor_lifetime_name(
    const VulkanStackTensorLifetimeClass lifetime) {
  switch (lifetime) {
    case VulkanStackTensorLifetimeClass::Unknown:
      return "unknown";
    case VulkanStackTensorLifetimeClass::InternalTemp:
      return "internal_temp";
    case VulkanStackTensorLifetimeClass::BlockOutputForNextBlock:
      return "block_output_for_next_block";
    case VulkanStackTensorLifetimeClass::RequestedIntermediateOutput:
      return "requested_intermediate_output";
    case VulkanStackTensorLifetimeClass::FinalStackOutput:
      return "final_stack_output";
    case VulkanStackTensorLifetimeClass::AliasOrView:
      return "alias_or_view";
  }
  return "unknown";
}

VulkanVisionStackPhase current_vision_stack_phase() {
  return g_vision_stack_phase;
}

int64_t current_vision_stack_block_index() {
  return g_vision_stack_block_index;
}

bool inside_vision_stack_phase() {
  return g_vision_stack_phase != VulkanVisionStackPhase::Unknown;
}

void note_vulkan_stack_dispatch(const char* shader_name) {
  if (!inside_vision_stack_phase()) {
    return;
  }
  std::ostringstream key;
  key << "stack_dispatch"
      << " phase=" << vision_stack_phase_name(g_vision_stack_phase)
      << " block=" << g_vision_stack_block_index
      << " shader=" << (shader_name && shader_name[0] ? shader_name : "unknown")
      << " role=" << vision_stack_phase_name(g_vision_stack_phase);
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  stack_dispatch_aggregate()[key.str()] += 1u;
}

void note_vulkan_stack_allocation(
    const char* role,
    const VulkanStackTensorLifetimeClass lifetime,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    const int64_t dtype,
    const bool direct_buffer,
    const bool buffer_storage,
    const bool image_storage,
    const bool escapes_stack,
    const bool requested_intermediate,
    const uint64_t bytes) {
  if (!inside_vision_stack_phase()) {
    return;
  }
  std::ostringstream key;
  key << "stack_alloc"
      << " phase=" << vision_stack_phase_name(g_vision_stack_phase)
      << " block=" << g_vision_stack_block_index
      << " role=" << (role && role[0] ? role : "unknown")
      << " lifetime=" << stack_tensor_lifetime_name(lifetime)
      << " shape=" << format_sizes(sizes)
      << " strides=" << format_sizes(strides)
      << " dtype=" << dtype
      << " direct_buffer=" << (direct_buffer ? 1 : 0)
      << " buffer_storage=" << (buffer_storage ? 1 : 0)
      << " image_storage=" << (image_storage ? 1 : 0)
      << " escapes_stack=" << (escapes_stack ? 1 : 0)
      << " requested_intermediate=" << (requested_intermediate ? 1 : 0);
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  StackAllocationValue& value = stack_allocation_aggregate()[key.str()];
  value.count += 1u;
  value.bytes += bytes;
  value.peak_live_estimate_bytes = std::max(value.peak_live_estimate_bytes, bytes);
}

std::vector<std::string> stack_dispatch_aggregate_snapshot() {
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  std::vector<std::string> rows;
  rows.reserve(stack_dispatch_aggregate().size());
  for (const auto& item : stack_dispatch_aggregate()) {
    std::ostringstream row;
    row << item.first << " count=" << item.second;
    rows.push_back(row.str());
  }
  return rows;
}

std::vector<std::string> stack_allocation_aggregate_snapshot() {
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  std::vector<std::string> rows;
  rows.reserve(stack_allocation_aggregate().size());
  for (const auto& item : stack_allocation_aggregate()) {
    std::ostringstream row;
    row << item.first << " count=" << item.second.count
        << " bytes=" << item.second.bytes
        << " peak_live_estimate_bytes="
        << item.second.peak_live_estimate_bytes;
    rows.push_back(row.str());
  }
  return rows;
}

void reset_stack_dispatch_aggregate() {
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  stack_dispatch_aggregate().clear();
}

void reset_stack_allocation_aggregate() {
  std::lock_guard<std::mutex> guard(stack_aggregate_mutex());
  stack_allocation_aggregate().clear();
}

void note_vulkan_queue_wait_idle() {
  vulkan_sync_counters().queue_wait_idle_count.fetch_add(
      1u, std::memory_order_relaxed);
}

void note_vulkan_forced_sync(VulkanForcedSyncReason reason) {
  VulkanSyncCounters& counters = vulkan_sync_counters();
  counters.forced_sync_count.fetch_add(1u, std::memory_order_relaxed);
  switch (reason) {
    case VulkanForcedSyncReason::ExplicitSynchronize:
      counters.forced_sync_explicit_synchronize_count.fetch_add(
          1u, std::memory_order_relaxed);
      return;
    case VulkanForcedSyncReason::TensorCpuReadback:
      counters.forced_sync_tensor_cpu_readback_count.fetch_add(
          1u, std::memory_order_relaxed);
      return;
    case VulkanForcedSyncReason::EventSynchronize:
      counters.forced_sync_event_synchronize_count.fetch_add(
          1u, std::memory_order_relaxed);
      return;
    case VulkanForcedSyncReason::RetireQueueDrain:
      counters.forced_sync_retire_queue_drain_count.fetch_add(
          1u, std::memory_order_relaxed);
      return;
    case VulkanForcedSyncReason::GpuTimestampQueryReset:
      counters.forced_sync_gpu_timestamp_query_reset_count.fetch_add(
          1u, std::memory_order_relaxed);
      return;
    case VulkanForcedSyncReason::FallbackPolicyReadback:
      counters.forced_sync_fallback_policy_readback_count.fetch_add(
          1u, std::memory_order_relaxed);
      return;
    case VulkanForcedSyncReason::Unknown:
      counters.forced_sync_unknown_count.fetch_add(
          1u, std::memory_order_relaxed);
      return;
  }
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
