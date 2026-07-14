#include <ATen/native/vulkan/api/SyncCounters.h>

#ifdef USE_VULKAN_API

#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace {

thread_local VulkanSubmitPhase g_submit_phase = VulkanSubmitPhase::Unknown;

std::array<VulkanRetireCallSiteCounter, 27>& retire_call_site_counters() {
  static std::array<VulkanRetireCallSiteCounter, 27> counters;
  return counters;
}

} // namespace

VulkanSyncCounters& vulkan_sync_counters() {
  static VulkanSyncCounters counters;
  return counters;
}

VulkanGraphProgramInvocationCounters&
vulkan_graph_program_invocation_counters() {
  static VulkanGraphProgramInvocationCounters counters;
  return counters;
}

VulkanSubmitOriginCounters& vulkan_submit_origin_counters() {
  static VulkanSubmitOriginCounters counters;
  return counters;
}

VulkanSubmitOriginPhaseCounters& vulkan_submit_origin_phase_counters() {
  static VulkanSubmitOriginPhaseCounters counters;
  return counters;
}

VulkanRetireDrainCounters& vulkan_retire_drain_counters() {
  static VulkanRetireDrainCounters counters;
  return counters;
}

VulkanSubmitPhaseScope::VulkanSubmitPhaseScope(VulkanSubmitPhase phase)
    : previous_(g_submit_phase) {
  g_submit_phase = phase;
}

VulkanSubmitPhaseScope::~VulkanSubmitPhaseScope() {
  g_submit_phase = previous_;
}

void reset_vulkan_sync_counters() {
  VulkanSyncCounters& counters = vulkan_sync_counters();
  counters.compute_dispatch_count.store(0u, std::memory_order_relaxed);
  counters.submit_compute_job_count.store(0u, std::memory_order_relaxed);
  counters.retire_cleanup_callback_count.store(0u, std::memory_order_relaxed);
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

void reset_vulkan_graph_program_invocation_counters() {
  VulkanGraphProgramInvocationCounters& counters =
      vulkan_graph_program_invocation_counters();
  counters.scope_begun_count.store(0u, std::memory_order_relaxed);
  counters.normal_submit_token_capture_count.store(
      0u, std::memory_order_relaxed);
  counters.aborted_submit_count.store(0u, std::memory_order_relaxed);
  counters.rejected_incompatible_state_count.store(
      0u, std::memory_order_relaxed);
  counters.bounded_region_host_sync_rejected_count.store(
      0u, std::memory_order_relaxed);
  counters.scratch_captured_count.store(0u, std::memory_order_relaxed);
  counters.scratch_reused_count.store(0u, std::memory_order_relaxed);
  counters.scratch_transient_overflow_count.store(
      0u, std::memory_order_relaxed);
  counters.scratch_retire_enqueued_count.store(
      0u, std::memory_order_relaxed);
  counters.scratch_immediate_release_count.store(
      0u, std::memory_order_relaxed);
}

std::vector<int64_t> graph_program_invocation_counters_snapshot() {
  const VulkanGraphProgramInvocationCounters& counters =
      vulkan_graph_program_invocation_counters();
  return {
      static_cast<int64_t>(
          counters.scope_begun_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.normal_submit_token_capture_count.load(
          std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.aborted_submit_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.rejected_incompatible_state_count.load(
          std::memory_order_relaxed)),
      static_cast<int64_t>(counters.bounded_region_host_sync_rejected_count.load(
          std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.scratch_captured_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.scratch_reused_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.scratch_transient_overflow_count.load(
          std::memory_order_relaxed)),
      static_cast<int64_t>(counters.scratch_retire_enqueued_count.load(
          std::memory_order_relaxed)),
      static_cast<int64_t>(counters.scratch_immediate_release_count.load(
          std::memory_order_relaxed)),
  };
}

void reset_vulkan_submit_origin_counters() {
  VulkanSubmitOriginCounters& counters = vulkan_submit_origin_counters();
  counters.total_queue_submits.store(0u, std::memory_order_relaxed);
  counters.normal_cmd_submit_frequency.store(0u, std::memory_order_relaxed);
  counters.stack_planned_recording_submit.store(0u, std::memory_order_relaxed);
  counters.pre_stack_flush.store(0u, std::memory_order_relaxed);
  counters.post_stack_flush.store(0u, std::memory_order_relaxed);
  counters.explicit_synchronize.store(0u, std::memory_order_relaxed);
  counters.tensor_cpu_readback.store(0u, std::memory_order_relaxed);
  counters.host_upload.store(0u, std::memory_order_relaxed);
  counters.fallback_readback.store(0u, std::memory_order_relaxed);
  counters.retire_queue_drain.store(0u, std::memory_order_relaxed);
  counters.profiling_timestamp_reset.store(0u, std::memory_order_relaxed);
  counters.profiling_timestamp_readback.store(0u, std::memory_order_relaxed);
  counters.shutdown.store(0u, std::memory_order_relaxed);
  counters.debug_validation.store(0u, std::memory_order_relaxed);
  counters.conv_prepack_upload.store(0u, std::memory_order_relaxed);
  counters.pending_command_flush.store(0u, std::memory_order_relaxed);
  counters.unknown.store(0u, std::memory_order_relaxed);
}

void reset_vulkan_submit_origin_phase_counters() {
  VulkanSubmitOriginPhaseCounters& counters =
      vulkan_submit_origin_phase_counters();
  for (auto& origin_counts : counters.counts) {
    for (auto& count : origin_counts) {
      count.store(0u, std::memory_order_relaxed);
    }
  }
}

void reset_vulkan_retire_drain_counters() {
  VulkanRetireDrainCounters& counters = vulkan_retire_drain_counters();
  counters.total.store(0u, std::memory_order_relaxed);
  counters.queue_submit_count.store(0u, std::memory_order_relaxed);
  counters.blocking_wait_count.store(0u, std::memory_order_relaxed);
  counters.poll_only_count.store(0u, std::memory_order_relaxed);
  counters.pending_resource_count_total.store(0u, std::memory_order_relaxed);
  counters.pending_bytes_total.store(0u, std::memory_order_relaxed);
  counters.explicit_drain.store(0u, std::memory_order_relaxed);
  counters.shutdown.store(0u, std::memory_order_relaxed);
  counters.resource_pressure.store(0u, std::memory_order_relaxed);
  counters.descriptor_pool_pressure.store(0u, std::memory_order_relaxed);
  counters.command_buffer_recycle.store(0u, std::memory_order_relaxed);
  counters.readback_preparation.store(0u, std::memory_order_relaxed);
  counters.synchronize.store(0u, std::memory_order_relaxed);
  counters.stack_scope_end.store(0u, std::memory_order_relaxed);
  counters.decoder_phase.store(0u, std::memory_order_relaxed);
  counters.setup_phase.store(0u, std::memory_order_relaxed);
  counters.debug_validation.store(0u, std::memory_order_relaxed);
  counters.unknown.store(0u, std::memory_order_relaxed);
}

void reset_retire_call_site_counters() {
  for (auto& counter : retire_call_site_counters()) {
    counter.total.store(0u, std::memory_order_relaxed);
    counter.queue_submit_count.store(0u, std::memory_order_relaxed);
    counter.blocking_wait_count.store(0u, std::memory_order_relaxed);
    counter.poll_only_count.store(0u, std::memory_order_relaxed);
    counter.pending_resource_count_total.store(0u, std::memory_order_relaxed);
    counter.pending_bytes_total.store(0u, std::memory_order_relaxed);
  }
}

void note_vulkan_queue_submit(VulkanSubmitOrigin origin) {
  VulkanSubmitOriginCounters& counters = vulkan_submit_origin_counters();
  counters.total_queue_submits.fetch_add(1u, std::memory_order_relaxed);
  const size_t origin_index = static_cast<size_t>(origin);
  const size_t phase_index = static_cast<size_t>(current_submit_phase());
  if (origin_index < kNumSubmitOrigins && phase_index < kNumSubmitPhases) {
    vulkan_submit_origin_phase_counters()
        .counts[origin_index][phase_index]
        .fetch_add(1u, std::memory_order_relaxed);
  }
  switch (origin) {
    case VulkanSubmitOrigin::NormalCmdSubmitFrequency:
      counters.normal_cmd_submit_frequency.fetch_add(
          1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::StackPlannedRecordingSubmit:
      counters.stack_planned_recording_submit.fetch_add(
          1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::PreStackFlush:
      counters.pre_stack_flush.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::PostStackFlush:
      counters.post_stack_flush.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::ExplicitSynchronize:
      counters.explicit_synchronize.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::TensorCpuReadback:
      counters.tensor_cpu_readback.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::HostUpload:
      counters.host_upload.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::FallbackReadback:
      counters.fallback_readback.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::RetireQueueDrain:
      counters.retire_queue_drain.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::ProfilingTimestampReset:
      counters.profiling_timestamp_reset.fetch_add(
          1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::ProfilingTimestampReadback:
      counters.profiling_timestamp_readback.fetch_add(
          1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::ContextShutdown:
      counters.shutdown.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::DebugValidation:
      counters.debug_validation.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::ConvPrepackUpload:
      counters.conv_prepack_upload.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::PendingCommandFlush:
      counters.pending_command_flush.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanSubmitOrigin::Unknown:
    default:
      counters.unknown.fetch_add(1u, std::memory_order_relaxed);
      break;
  }
}

const char* submit_origin_name(const VulkanSubmitOrigin origin) {
  switch (origin) {
    case VulkanSubmitOrigin::NormalCmdSubmitFrequency:
      return "normal_cmd_submit_frequency";
    case VulkanSubmitOrigin::StackPlannedRecordingSubmit:
      return "stack_planned_recording_submit";
    case VulkanSubmitOrigin::PreStackFlush:
      return "pre_stack_flush";
    case VulkanSubmitOrigin::PostStackFlush:
      return "post_stack_flush";
    case VulkanSubmitOrigin::ExplicitSynchronize:
      return "explicit_synchronize";
    case VulkanSubmitOrigin::TensorCpuReadback:
      return "tensor_cpu_readback";
    case VulkanSubmitOrigin::HostUpload:
      return "host_upload";
    case VulkanSubmitOrigin::FallbackReadback:
      return "fallback_readback";
    case VulkanSubmitOrigin::RetireQueueDrain:
      return "retire_queue_drain";
    case VulkanSubmitOrigin::ProfilingTimestampReset:
      return "profiling_timestamp_reset";
    case VulkanSubmitOrigin::ProfilingTimestampReadback:
      return "profiling_timestamp_readback";
    case VulkanSubmitOrigin::ContextShutdown:
      return "shutdown";
    case VulkanSubmitOrigin::DebugValidation:
      return "debug_validation";
    case VulkanSubmitOrigin::ConvPrepackUpload:
      return "conv_prepack_upload";
    case VulkanSubmitOrigin::PendingCommandFlush:
      return "pending_command_flush";
    case VulkanSubmitOrigin::Unknown:
    default:
      return "unknown";
  }
}

const char* submit_phase_name(const VulkanSubmitPhase phase) {
  switch (phase) {
    case VulkanSubmitPhase::ModelSetup:
      return "model_setup";
    case VulkanSubmitPhase::PatchEmbed:
      return "patch_embed";
    case VulkanSubmitPhase::PositionalEmbeddingSetup:
      return "positional_embedding_setup";
    case VulkanSubmitPhase::StackOwner:
      return "stack_owner";
    case VulkanSubmitPhase::StackOwnerNorm:
      return "stack_owner_norm";
    case VulkanSubmitPhase::StackOwnerAttention:
      return "stack_owner_attention";
    case VulkanSubmitPhase::StackOwnerLinear:
      return "stack_owner_linear";
    case VulkanSubmitPhase::StackOwnerResidual:
      return "stack_owner_residual";
    case VulkanSubmitPhase::Decoder:
      return "decoder";
    case VulkanSubmitPhase::DecoderConv:
      return "decoder_conv";
    case VulkanSubmitPhase::DecoderUpsample:
      return "decoder_upsample";
    case VulkanSubmitPhase::DecoderPointwise:
      return "decoder_pointwise";
    case VulkanSubmitPhase::Readback:
      return "readback";
    case VulkanSubmitPhase::ExplicitSynchronize:
      return "explicit_synchronize";
    case VulkanSubmitPhase::Retire:
      return "retire";
    case VulkanSubmitPhase::Profiling:
      return "profiling";
    case VulkanSubmitPhase::Shutdown:
      return "shutdown";
    case VulkanSubmitPhase::TestHarness:
      return "test_harness";
    case VulkanSubmitPhase::Unknown:
    default:
      return "unknown";
  }
}

const char* retire_call_site_name(const VulkanRetireCallSite callsite) {
  switch (callsite) {
    case VulkanRetireCallSite::ContextFlushPending:
      return "context_flush_pending";
    case VulkanRetireCallSite::ContextSubmitFrequency:
      return "context_submit_frequency";
    case VulkanRetireCallSite::ContextExplicitSynchronize:
      return "context_explicit_synchronize";
    case VulkanRetireCallSite::ContextReadback:
      return "context_readback";
    case VulkanRetireCallSite::ContextShutdown:
      return "context_shutdown";
    case VulkanRetireCallSite::StackPlannedRecordingEnd:
      return "stack_planned_recording_end";
    case VulkanRetireCallSite::StackOwnerPhaseBoundary:
      return "stack_owner_phase_boundary";
    case VulkanRetireCallSite::StackOwnerNorm1:
      return "stack_owner_norm1";
    case VulkanRetireCallSite::StackOwnerNorm2:
      return "stack_owner_norm2";
    case VulkanRetireCallSite::StackOwnerAttention:
      return "stack_owner_attention";
    case VulkanRetireCallSite::StackOwnerLinear:
      return "stack_owner_linear";
    case VulkanRetireCallSite::StackOwnerResidual:
      return "stack_owner_residual";
    case VulkanRetireCallSite::NativeLayerNormMetadata:
      return "native_layer_norm_metadata";
    case VulkanRetireCallSite::NativeLayerNormUniform:
      return "native_layer_norm_uniform";
    case VulkanRetireCallSite::AttentionMetadata:
      return "attention_metadata";
    case VulkanRetireCallSite::LinearMetadata:
      return "linear_metadata";
    case VulkanRetireCallSite::ConvMetadata:
      return "conv_metadata";
    case VulkanRetireCallSite::AddResidualMetadata:
      return "add_residual_metadata";
    case VulkanRetireCallSite::DescriptorRecycle:
      return "descriptor_recycle";
    case VulkanRetireCallSite::CommandBufferRecycle:
      return "command_buffer_recycle";
    case VulkanRetireCallSite::StagingBufferRecycle:
      return "staging_buffer_recycle";
    case VulkanRetireCallSite::UniformBufferRecycle:
      return "uniform_buffer_recycle";
    case VulkanRetireCallSite::MetadataBufferRecycle:
      return "metadata_buffer_recycle";
    case VulkanRetireCallSite::BenchmarkReadback:
      return "benchmark_readback";
    case VulkanRetireCallSite::BenchmarkSetup:
      return "benchmark_setup";
    case VulkanRetireCallSite::DebugValidation:
      return "debug_validation";
    case VulkanRetireCallSite::Unknown:
    default:
      return "unknown";
  }
}

VulkanSubmitPhase current_submit_phase() {
  return g_submit_phase;
}

void set_submit_phase(const VulkanSubmitPhase phase) {
  g_submit_phase = phase;
}

void reset_submit_phase() {
  g_submit_phase = VulkanSubmitPhase::Unknown;
}

std::vector<std::string> submit_origin_phase_snapshot() {
  const auto& counters = vulkan_submit_origin_phase_counters();
  std::vector<std::string> rows;
  for (size_t origin = 0; origin < kNumSubmitOrigins; ++origin) {
    for (size_t phase = 0; phase < kNumSubmitPhases; ++phase) {
      const uint64_t count =
          counters.counts[origin][phase].load(std::memory_order_relaxed);
      if (count == 0u) {
        continue;
      }
      std::ostringstream stream;
      stream << "submit_origin_phase origin="
             << submit_origin_name(static_cast<VulkanSubmitOrigin>(origin))
             << " phase="
             << submit_phase_name(static_cast<VulkanSubmitPhase>(phase))
             << " count=" << count;
      rows.emplace_back(stream.str());
    }
  }
  return rows;
}

std::vector<int64_t> retire_drain_counters_snapshot() {
  const auto& counters = vulkan_retire_drain_counters();
  return {
      static_cast<int64_t>(counters.total.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.queue_submit_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.blocking_wait_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.poll_only_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.pending_resource_count_total.load(
          std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.pending_bytes_total.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.explicit_drain.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.shutdown.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.resource_pressure.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.descriptor_pool_pressure.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.command_buffer_recycle.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.readback_preparation.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.synchronize.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.stack_scope_end.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.decoder_phase.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.setup_phase.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.debug_validation.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.unknown.load(std::memory_order_relaxed)),
  };
}

std::vector<std::string> retire_call_site_counters_snapshot() {
  const auto& counters = retire_call_site_counters();
  std::vector<std::string> rows;
  for (size_t index = 0; index < counters.size(); ++index) {
    const auto& counter = counters[index];
    const uint64_t total = counter.total.load(std::memory_order_relaxed);
    if (total == 0u) {
      continue;
    }
    std::ostringstream stream;
    stream << "retire_call_site callsite="
           << retire_call_site_name(static_cast<VulkanRetireCallSite>(index))
           << " total=" << total << " submit="
           << counter.queue_submit_count.load(std::memory_order_relaxed)
           << " poll="
           << counter.poll_only_count.load(std::memory_order_relaxed)
           << " blocking_wait="
           << counter.blocking_wait_count.load(std::memory_order_relaxed)
           << " pending_resources="
           << counter.pending_resource_count_total.load(
                  std::memory_order_relaxed)
           << " pending_bytes="
           << counter.pending_bytes_total.load(std::memory_order_relaxed);
    rows.emplace_back(stream.str());
  }
  return rows;
}

void note_vulkan_retire_drain(
    VulkanRetireDrainReason reason,
    VulkanRetireCallSite callsite,
    const bool queue_submit,
    const bool blocking_wait,
    const uint64_t pending_resource_count,
    const uint64_t pending_bytes) {
  auto& counters = vulkan_retire_drain_counters();
  counters.total.fetch_add(1u, std::memory_order_relaxed);
  if (queue_submit) {
    counters.queue_submit_count.fetch_add(1u, std::memory_order_relaxed);
  } else {
    counters.poll_only_count.fetch_add(1u, std::memory_order_relaxed);
  }
  if (blocking_wait) {
    counters.blocking_wait_count.fetch_add(1u, std::memory_order_relaxed);
  }
  counters.pending_resource_count_total.fetch_add(
      pending_resource_count, std::memory_order_relaxed);
  counters.pending_bytes_total.fetch_add(
      pending_bytes, std::memory_order_relaxed);
  const size_t callsite_index = static_cast<size_t>(callsite);
  if (callsite_index < retire_call_site_counters().size()) {
    auto& counter = retire_call_site_counters()[callsite_index];
    counter.total.fetch_add(1u, std::memory_order_relaxed);
    if (queue_submit) {
      counter.queue_submit_count.fetch_add(1u, std::memory_order_relaxed);
    } else {
      counter.poll_only_count.fetch_add(1u, std::memory_order_relaxed);
    }
    if (blocking_wait) {
      counter.blocking_wait_count.fetch_add(1u, std::memory_order_relaxed);
    }
    counter.pending_resource_count_total.fetch_add(
        pending_resource_count, std::memory_order_relaxed);
    counter.pending_bytes_total.fetch_add(
        pending_bytes, std::memory_order_relaxed);
  }
  switch (reason) {
    case VulkanRetireDrainReason::ExplicitDrain:
      counters.explicit_drain.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::Shutdown:
      counters.shutdown.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::ResourcePressure:
      counters.resource_pressure.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::DescriptorPoolPressure:
      counters.descriptor_pool_pressure.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::CommandBufferRecycle:
      counters.command_buffer_recycle.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::ReadbackPreparation:
      counters.readback_preparation.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::Synchronize:
      counters.synchronize.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::StackScopeEnd:
      counters.stack_scope_end.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::DecoderPhase:
      counters.decoder_phase.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::SetupPhase:
      counters.setup_phase.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::DebugValidation:
      counters.debug_validation.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanRetireDrainReason::Unknown:
    default:
      counters.unknown.fetch_add(1u, std::memory_order_relaxed);
      break;
  }
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
