#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/vk_api.h>

#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Command.h>
#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/Pipeline.h>
#include <ATen/native/vulkan/api/QueryPool.h>
#include <ATen/native/vulkan/api/RetireQueue.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Runtime.h>
#include <ATen/native/vulkan/api/Shader.h>
#include <ATen/native/vulkan/api/Stream.h>
#include <ATen/native/vulkan/api/Sync.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <c10/macros/Export.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace api {

bool cpu_timeline_logging_enabled();
uint64_t cpu_timeline_now_us();
void append_cpu_timeline_log_line(const std::string& line);
void dump_cpu_timeline_summary_log();

struct ContextConfig final {
  uint32_t cmdSubmitFrequency;
  CommandPoolConfig cmdPoolConfig;
  DescriptorPoolConfig descriptorPoolConfig;
  QueryPoolConfig queryPoolConfig;
};

struct StackPlannedRecordingStats final {
  uint64_t recorded_compute_jobs = 0u;
  uint64_t recorded_descriptor_writes = 0u;
  uint64_t recorded_barriers = 0u;
  uint64_t suppressed_frequency_flushes = 0u;
  uint64_t premature_submits = 0u;
};

struct PendingRetireBuffer final {
  VulkanBuffer buffer;
  VulkanRetiredResourceKind kind = VulkanRetiredResourceKind::Unknown;
  VulkanRetiredResourceRole role = VulkanRetiredResourceRole::Unknown;
  VulkanSubmitPhase phase = VulkanSubmitPhase::Unknown;
  VulkanRetireCallSite callsite = VulkanRetireCallSite::Unknown;
  uint64_t bytes = 0u;
  VulkanStackRetireProvenance stack_provenance;
};

struct PendingRetireImage final {
  VulkanImage image;
  VulkanRetiredResourceKind kind = VulkanRetiredResourceKind::Image;
  VulkanRetiredResourceRole role = VulkanRetiredResourceRole::Unknown;
  VulkanSubmitPhase phase = VulkanSubmitPhase::Unknown;
  VulkanRetireCallSite callsite = VulkanRetireCallSite::Unknown;
  uint64_t bytes = 0u;
  VulkanStackRetireProvenance stack_provenance;
};

enum class PendingWorkRetireDrainPolicy : uint8_t {
  SubmitOldPathPending,
  DeferTinyOldPathPending,
};

enum class PendingCommandFlushReason : uint8_t {
  Unknown,
  AddmmEagerSubmit,
  LinearEagerSubmit,
  LinearRawDirectWeightEagerSubmit,
  LinearGeluEagerSubmit,
  RepeatTemporaryCloneLifetime,
  AttentionReplayInputUpload,
  AttentionReplayWarmup,
  CompiledReplaySubmitGuard,
  VisionReplayInputUpload,
  VisionReplayWarmup,
  VisionReplaySubmitGuard,
  VisionReplayOutputMaterialization,
  VisionCompiledSessionInputUpload,
  VisionCompiledSessionWarmup,
  VisionBundleInputUpload,
  VisionBundleWarmup,
  VisionStackReplayInputUpload,
  VisionStackReplayWarmup,
  VisionStackReplayStepSubmitGuard,
};

const char* pending_command_flush_reason_name(PendingCommandFlushReason reason);

//
// Vulkan Context holds onto all relevant Vulkan state as it pertains to our
// use of Vulkan in PyTorch.  A Context is associated with one, and only one,
// Adapter as a precursor to multi-GPU support.  All Vulkan tensors in PyTorch
// are associated with a Context to make tensor <-> device affinity explicit.
// The context is currently a global object, but technically it does not need
// to be if we were to make it explicit to the user.
//

class TORCH_API Context final {
 public:
  class TORCH_API ScopedExternalCommandRecording final {
   public:
    ScopedExternalCommandRecording(Context&, CommandBuffer&);

    ScopedExternalCommandRecording(const ScopedExternalCommandRecording&) =
        delete;
    ScopedExternalCommandRecording& operator=(
        const ScopedExternalCommandRecording&) = delete;

    ScopedExternalCommandRecording(ScopedExternalCommandRecording&&) = delete;
    ScopedExternalCommandRecording& operator=(
        ScopedExternalCommandRecording&&) = delete;

    ~ScopedExternalCommandRecording();

   private:
    Context* context_{nullptr};
  };

  class TORCH_API GraphProgramInvocationScope final {
   public:
    enum class State : uint8_t {
      Active,
      Submitted,
      Aborted,
    };

    explicit GraphProgramInvocationScope(Context&);

    GraphProgramInvocationScope(const GraphProgramInvocationScope&) = delete;
    GraphProgramInvocationScope& operator=(
        const GraphProgramInvocationScope&) = delete;

    GraphProgramInvocationScope(GraphProgramInvocationScope&&) = delete;
    GraphProgramInvocationScope& operator=(
        GraphProgramInvocationScope&&) = delete;

    ~GraphProgramInvocationScope() noexcept;

    VulkanSubmission submit();
    void abort();
    bool active() const;
    State state() const;
    const VulkanSubmission& submission() const;

   private:
    Context* context_{nullptr};
    std::unique_lock<std::mutex> lock_;
    VulkanSubmission submission_{};
    State state_{State::Active};
  };

  explicit Context(c10::DeviceIndex device_index, const ContextConfig&);

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  ~Context();

 private:
  // Config
  ContextConfig config_;
  // Important handles
  c10::DeviceIndex device_index_;
  Adapter* adapter_p_;
  VkDevice device_;
  Adapter::Queue queue_;
  // Resource Pools
  CommandPool command_pool_;
  DescriptorPool descriptor_pool_;
  CommandPool persistent_command_pool_;
  DescriptorPool persistent_descriptor_pool_;
  struct StackPlannedRecordingDescriptorPoolLease final {
    DescriptorPool descriptor_pool;

    StackPlannedRecordingDescriptorPoolLease(
        VkDevice device,
        const ContextConfig& config)
        : descriptor_pool(device, config.descriptorPoolConfig) {}
  };
  std::shared_ptr<StackPlannedRecordingDescriptorPoolLease>
      stack_planned_recording_descriptor_pool_lease_;
  FencePool fences_;
  // Diagnostics
  bool enable_op_profiling_{false};
  QueryPool querypool_;
  // Command buffers submission
  std::mutex cmd_mutex_;
  std::atomic<bool> graph_program_invocation_active_{false};
  CommandBuffer cmd_;
  CommandBuffer stack_region_owned_cmd_;
  uint32_t submit_count_;
  uint64_t command_buffer_recording_id_;
  uint64_t next_command_buffer_recording_id_;
  std::atomic<bool> stack_planned_recording_active_;
  std::atomic<bool> stack_region_recording_domain_observation_active_;
  std::atomic<bool> stack_region_owned_command_buffer_active_;
  std::thread::id stack_planned_recording_owner_;
  StackPlannedRecordingStats stack_planned_recording_stats_;
  uint64_t stack_region_owned_recording_dispatch_count_;
  uint64_t stack_region_external_command_buffer_acquire_count_;
  uint64_t stack_region_external_descriptor_set_count_;
  uint64_t stack_region_external_command_buffer_acquire_at_begin_;
  uint64_t stack_region_external_descriptor_set_count_at_begin_;
  std::vector<PendingRetireBuffer> stack_region_owned_recording_retained_buffers_;
  std::vector<PendingRetireImage> stack_region_owned_recording_retained_images_;
  std::atomic<uint64_t> stack_region_single_recording_plan_id_;
  std::atomic<uint64_t> next_stack_region_single_recording_plan_id_;
  std::atomic<uint32_t> stack_region_single_recording_plan_state_;
  std::atomic<uint64_t> stack_region_single_recording_owner_id_;
  std::atomic<uint64_t> next_stack_region_single_recording_owner_id_;
  std::atomic<uint32_t> stack_region_single_recording_owner_state_;
  std::atomic<uint64_t> stack_region_command_buffer_batch_lease_id_;
  std::atomic<uint64_t> next_stack_region_command_buffer_batch_lease_id_;
  std::atomic<uint32_t> stack_region_command_buffer_batch_lease_state_;
  std::atomic<uint64_t> stack_region_close_submit_owner_id_;
  std::atomic<uint64_t> next_stack_region_close_submit_owner_id_;
  std::atomic<uint32_t> stack_region_close_submit_owner_state_;
  std::atomic<uint64_t> stack_region_command_ownership_id_;
  std::atomic<uint64_t> next_stack_region_command_ownership_id_;
  std::atomic<uint32_t> stack_region_command_ownership_state_;
  std::atomic<uint64_t> stack_region_command_pool_reset_deferral_owner_id_;
  std::atomic<uint64_t>
      next_stack_region_command_pool_reset_deferral_owner_id_;
  std::atomic<uint32_t> stack_region_command_pool_reset_deferral_owner_state_;
  std::atomic<uint64_t> stack_region_retire_timeline_owner_id_;
  std::atomic<uint64_t> next_stack_region_retire_timeline_owner_id_;
  std::atomic<uint32_t> stack_region_retire_timeline_owner_state_;
  std::atomic<uint64_t> stack_region_pending_retire_transfer_owner_id_;
  std::atomic<uint64_t> next_stack_region_pending_retire_transfer_owner_id_;
  std::atomic<uint32_t> stack_region_pending_retire_transfer_owner_state_;
  std::atomic<uint64_t> stack_region_pending_retire_transfer_source_id_;
  std::atomic<uint64_t> next_stack_region_pending_retire_transfer_source_id_;
  std::atomic<uint32_t> stack_region_pending_retire_transfer_source_state_;
  std::atomic<uint64_t> stack_region_pending_retire_transfer_source_count_;
  std::atomic<uint64_t> stack_region_pending_retire_transfer_source_bytes_;
  struct StackRegionPendingRetireTransferSourceSnapshot final {
    uint32_t state = 0u;
    uint64_t resource_count = 0u;
    uint64_t resource_bytes = 0u;
    std::string allocation_signature;
  };
  std::mutex stack_region_pending_retire_transfer_source_signature_mutex_;
  std::string stack_region_pending_retire_transfer_source_signature_;
  std::map<uint64_t, StackRegionPendingRetireTransferSourceSnapshot>
      stack_region_pending_retire_transfer_sources_;
  std::map<std::string, StackRegionPendingRetireTransferSourceSnapshot>
      stack_region_pending_retire_transfer_sources_by_state_;
  // Memory Management
  std::mutex pending_retire_buffers_mutex_;
  std::vector<PendingRetireBuffer> pending_retire_buffers_;
  std::mutex pending_retire_images_mutex_;
  std::vector<PendingRetireImage> pending_retire_images_;
  std::atomic<uint64_t> pending_retire_bytes_;
  std::mutex stack_internal_temp_retire_batch_mutex_;
  std::vector<PendingRetireBuffer> stack_internal_temp_retire_batch_buffers_;
  std::vector<PendingRetireImage> stack_internal_temp_retire_batch_images_;
  std::mutex stack_region_pending_retire_handoff_batch_mutex_;
  std::vector<PendingRetireBuffer> stack_region_pending_retire_handoff_buffers_;
  std::vector<PendingRetireImage> stack_region_pending_retire_handoff_images_;
  std::mutex bridge_private_capture_pending_retire_handoff_batch_mutex_;
  std::vector<PendingRetireBuffer>
      bridge_private_capture_pending_retire_handoff_buffers_;
  std::vector<PendingRetireImage>
      bridge_private_capture_pending_retire_handoff_images_;
  RetireQueue retire_queue_;
  VulkanSubmission last_submission_;
  uint64_t stack_region_exit_work_batch_executor_depth_ = 0u;
  enum class StackRegionExitWorkAction {
    LogBeforeHandoffBatches,
    SnapshotPendingRetireTransferSource,
    RetireStackInternalTempBatch,
    RetireStackRegionHandoffBatch,
    FinalizeStackRecording,
    LogAfterFinalize,
  };

  struct StackRegionRetainedStatePayload final {
    bool captured = false;
    std::string event;
    uint64_t plan_id = 0u;
    uint64_t owner_id = 0u;
    uint64_t command_recording_id = 0u;
    uint64_t submit_count = 0u;
    uint64_t owned_command_buffer_active = 0u;
    uint64_t external_recording_active = 0u;
    uint64_t external_keepalive_buffers = 0u;
    uint64_t external_keepalive_images = 0u;
    uint64_t external_keepalive_buffer_bytes = 0u;
    uint64_t external_keepalive_image_bytes = 0u;
    uint64_t retained_buffers = 0u;
    uint64_t retained_images = 0u;
    uint64_t retained_buffer_bytes = 0u;
    uint64_t retained_image_bytes = 0u;
    uint64_t pending_retire_buffers = 0u;
    uint64_t pending_retire_images = 0u;
    uint64_t pending_retire_bytes = 0u;
    uint64_t stack_internal_temp_buffers = 0u;
    uint64_t stack_internal_temp_images = 0u;
    uint64_t stack_internal_temp_bytes = 0u;
    uint64_t region_handoff_buffers = 0u;
    uint64_t region_handoff_images = 0u;
    uint64_t region_handoff_bytes = 0u;
    uint64_t bridge_handoff_buffers = 0u;
    uint64_t bridge_handoff_images = 0u;
    uint64_t bridge_handoff_bytes = 0u;
    uint64_t retire_queue_size = 0u;
    uint64_t stack_descriptor_pool_lease_active = 0u;
    uint64_t external_cmd_acquire_count = 0u;
    uint64_t external_desc_set_count = 0u;
    uint64_t external_cmd_acquire_delta = 0u;
    uint64_t external_desc_set_delta = 0u;
    uint64_t recorded_dispatch_count = 0u;
    uint64_t transfer_source_count = 0u;
    uint64_t transfer_source_bytes = 0u;
    uint64_t transfer_source_map_count = 0u;
    uint64_t transfer_source_by_state_count = 0u;
    uint64_t transfer_source_signature_empty = 0u;
    uint64_t stream_last_submitted = 0u;
    uint64_t submission_value = 0u;
  };

  struct StackRegionExitWorkBatch final {
    VulkanSubmission submission;
    std::vector<StackRegionExitWorkAction> actions;
    bool prepared = false;
    bool drained_inline = false;
    bool pending_retire_handoff_at_stack_exit = false;
    bool bind_stack_internal_source_at_stack_exit = false;
    bool preserve_larger_source = false;
    uint64_t source_snapshot_state = 0u;
    uint64_t stack_internal_temp_batch_count = 0u;
    uint64_t stack_internal_temp_batch_bytes = 0u;
    uint64_t stack_region_handoff_batch_count = 0u;
    uint64_t stack_region_handoff_batch_bytes = 0u;
    uint64_t drained_action_count = 0u;
    const char* executor_mode = "not_started";
    uint64_t executor_depth = 0u;
    uint64_t executor_depth_before = 0u;
    uint64_t executor_depth_after = 0u;
    bool executor_reentry_rejected = false;
    const char* executor_reentry_status = "not_entered";
    const char* executor_fail_closed_reason = "none";
    uint64_t retained_state_live_log_reread_count = 0u;
    StackRegionRetainedStatePayload before_handoff_retained_state_payload;
    StackRegionRetainedStatePayload after_finalize_retained_state_payload;
    std::vector<StackRegionRetainedStatePayload>
        retained_state_payloads_to_publish;
  };

  void clear_pending_retire_resources_locked();
  void clear_stack_internal_temp_retire_batch_locked();
  void clear_stack_region_pending_retire_handoff_batch_locked();
  void clear_bridge_private_capture_pending_retire_handoff_batch_locked();
  void restore_stack_internal_temp_retire_batch_to_pending_locked();
  void restore_stack_region_pending_retire_handoff_batch_to_pending_locked();
  void restore_bridge_private_capture_pending_retire_handoff_batch_to_pending_locked();
  void retire_stack_internal_temp_retire_batch_locked(
      const VulkanSubmission& submission);
  void retire_stack_region_pending_retire_handoff_batch_locked(
      const VulkanSubmission& submission);
  void retire_bridge_private_capture_pending_retire_handoff_batch_locked(
      const VulkanSubmission& submission);
  void flush_persistent_external_recording_pools_if_idle();
  void retire_stack_planned_recording_descriptor_pool_lease(
      const VulkanSubmission& submission);
  void release_stack_planned_recording_descriptor_pool_lease_now();
  void log_stack_region_retained_state_locked(
      const char* event,
      const VulkanSubmission* submission = nullptr);
  StackRegionRetainedStatePayload
  capture_stack_region_retained_state_payload_locked(
      const char* event,
      const VulkanSubmission* submission = nullptr);
  void publish_stack_region_retained_state_payload(
      const StackRegionRetainedStatePayload& payload);
  bool transfer_pending_retires_to_stack_region_handoff_locked(
      VulkanRetireCallSite callsite,
      const std::string& target_allocation_signature);
  bool has_stack_region_pending_retire_handoff_batch_locked();
  void snapshot_stack_region_pending_retire_transfer_source_locked(
      uint32_t state,
      bool include_context_pending_retires = false,
      bool preserve_larger_source = false);
  std::unique_ptr<StackRegionExitWorkBatch>
  prepare_stack_region_exit_work_batch_locked(
      const VulkanSubmission& submission);
  void drain_stack_region_exit_work_batch_locked(
      StackRegionExitWorkBatch& batch);
  std::vector<StackRegionRetainedStatePayload>
  execute_stack_region_exit_work_batch_locked(
      std::unique_ptr<StackRegionExitWorkBatch> batch);
  CommandBuffer* external_recording_cmd();
  const CommandBuffer* external_recording_cmd() const;
  bool is_inside_owned_program_recording() const;
  bool graph_program_invocation_active_for_current_thread() const;
  bool graph_program_invocation_active() const;
  bool stack_planned_recording_owned_by_current_thread() const;
  DescriptorPool& active_descriptor_pool();
  CommandBuffer& active_cmd();
  void capture_external_recording_buffer_cleanup(PendingRetireBuffer&&);
  void capture_external_recording_image_cleanup(PendingRetireImage&&);
  void note_external_recording_cleanup_logical_boundary(
      VulkanSubmitPhase phase,
      VulkanRetireCallSite callsite,
      uint64_t command_buffer_recording_id,
      uint64_t submit_epoch_before,
      uint64_t pending_dispatch_count);
  void begin_external_command_recording(CommandBuffer&);
  void end_external_command_recording();
  uint32_t gpu_profile_begin(
      CommandBuffer&,
      const std::string&,
      const VkExtent3D,
      const VkExtent3D);
  void gpu_profile_end(CommandBuffer&, uint32_t);
  void dump_gpu_profile_log(const char* reason);
  void reset_gpu_profile_queries();
  VulkanSubmission submit_cmd_handle_to_gpu(
      VulkanStreamState&,
      VkCommandBuffer,
      VulkanSubmitOrigin origin,
      VkFence fence_handle = VK_NULL_HANDLE,
      const bool final_use = false);
  VulkanSubmission close_submit_stack_planned_region_exit();
  std::string format_submit_failure_diagnostics(
      const VulkanStreamState&,
      VulkanSubmitOrigin origin,
      uint64_t signal_value,
      size_t wait_count,
      VkFence fence_handle,
      bool final_use);
  void retire_deferred_cleanup(VulkanSubmission, VulkanSubmitOrigin);
  void retire_external_recording_cleanup_resources(
      VulkanSubmission,
      std::vector<PendingRetireBuffer>&,
      std::vector<PendingRetireImage>&,
      uint64_t command_buffer_recording_id,
      uint64_t pending_dispatch_count,
      bool segment_metadata_observed,
      uint64_t segment_count,
      uint64_t segment_index,
      uint64_t segment_start_block,
      uint64_t segment_end_block,
      uint64_t segment_planned_dispatch_count);

 public:
  // Adapter access

  inline Adapter* adapter_ptr() {
    return adapter_p_;
  }

  inline c10::DeviceIndex device_index() const {
    return device_index_;
  }

  inline void enable_op_profiling() {
    enable_op_profiling_ = true;
  }

  inline void disable_op_profiling() {
    enable_op_profiling_ = false;
  }

  inline bool op_profiling_enabled() {
    return enable_op_profiling_;
  }

  uint32_t begin_external_gpu_profile(
      const std::string&,
      const VkExtent3D,
      const VkExtent3D);
  void end_external_gpu_profile(uint32_t);

  inline VkDevice device() {
    return device_;
  }

  inline VkQueue queue() {
    return queue_.handle;
  }

  // Device Caches

  inline ShaderLayoutCache& shader_layout_cache() {
    return adapter_ptr()->shader_layout_cache();
  }

  inline ShaderCache& shader_cache() {
    return adapter_ptr()->shader_cache();
  }

  inline PipelineLayoutCache& pipeline_layout_cache() {
    return adapter_ptr()->pipeline_layout_cache();
  }

  inline ComputePipelineCache& pipeline_cache() {
    return adapter_ptr()->compute_pipeline_cache();
  }

  // Resource Pools

  inline DescriptorPool& descriptor_pool() {
    return descriptor_pool_;
  }

  inline DescriptorPool& persistent_descriptor_pool() {
    return persistent_descriptor_pool_;
  }

  inline FencePool& fences() {
    return fences_;
  }

  // Diagnostics

  inline QueryPool& querypool() {
    return querypool_;
  }

  inline void reset_querypool() {
    set_cmd();
    querypool_.reset(cmd_);
  }

  // Memory Management
  void register_buffer_cleanup(
      VulkanBuffer& buffer,
      VulkanRetiredResourceKind kind = VulkanRetiredResourceKind::Buffer,
      VulkanRetiredResourceRole role = VulkanRetiredResourceRole::Unknown,
      VulkanSubmitPhase phase = current_submit_phase(),
      VulkanRetireCallSite callsite = VulkanRetireCallSite::Unknown,
      VulkanStackRetireProvenance stack_provenance = {}) {
    const uint64_t bytes = buffer.owns_memory()
        ? static_cast<uint64_t>(buffer.allocated_size())
        : 0u;
    if (role == VulkanRetiredResourceRole::Unknown) {
      if (phase == VulkanSubmitPhase::StackOwner) {
        role = stack_retired_resource_role_for_phase(
            current_vision_stack_phase());
      } else if (
          phase == VulkanSubmitPhase::ModelSetup ||
          phase == VulkanSubmitPhase::PatchEmbed ||
          phase == VulkanSubmitPhase::PositionalEmbeddingSetup) {
        role = VulkanRetiredResourceRole::SetupStaging;
      } else if (phase == VulkanSubmitPhase::Readback) {
        role = VulkanRetiredResourceRole::ReadbackStaging;
      }
    }
    if (stack_provenance.defined) {
      role = stack_provenance.producer_role;
      phase = VulkanSubmitPhase::StackOwner;
    }
    PendingRetireBuffer pending{
        std::move(buffer),
        kind,
        role,
        phase,
        callsite,
        bytes,
        std::move(stack_provenance)};
    if (external_recording_cmd()) {
      capture_external_recording_buffer_cleanup(std::move(pending));
      return;
    }
    const bool batch_candidate =
        is_safe_stack_temp_retire_batch_candidate(pending.stack_provenance);
    const bool stack_recording_active =
        batch_candidate && is_stack_planned_recording_active() &&
        stack_planned_recording_owned_by_current_thread();
    if (batch_candidate && stack_recording_active) {
      note_stack_internal_temp_retire_batch_decision(
          pending.stack_provenance,
          pending.bytes,
          stack_recording_active,
          /*accepted=*/true);
      mark_vulkan_memory_residency_state(
          pending.buffer.allocation_id(), "stack_batched_retire");
      std::lock_guard<std::mutex> batch_lock(
          stack_internal_temp_retire_batch_mutex_);
      stack_internal_temp_retire_batch_buffers_.push_back(std::move(pending));
      return;
    }
    if (pending.stack_provenance.defined) {
      note_stack_internal_temp_retire_batch_decision(
          pending.stack_provenance,
          pending.bytes,
          stack_recording_active,
          /*accepted=*/false);
    }
    if (pending.buffer.owns_memory()) {
      mark_vulkan_memory_residency_state(
          pending.buffer.allocation_id(), "pending_retire");
      pending_retire_bytes_.fetch_add(
          pending.bytes, std::memory_order_relaxed);
    }
    std::lock_guard<std::mutex> bufferlist_lock(
        pending_retire_buffers_mutex_);
    pending_retire_buffers_.push_back(std::move(pending));
  }

  bool graph_program_submission_complete(const VulkanSubmission&) const;
  void retire_graph_program_resource(
      VulkanSubmission,
      std::function<void()> cleanup);

  void register_image_cleanup(
      VulkanImage& image,
      VulkanRetiredResourceRole role = VulkanRetiredResourceRole::Unknown,
      VulkanSubmitPhase phase = current_submit_phase(),
      VulkanRetireCallSite callsite = VulkanRetireCallSite::Unknown,
      VulkanStackRetireProvenance stack_provenance = {}) {
    const uint64_t bytes = image.owns_memory()
        ? static_cast<uint64_t>(image.allocated_size())
        : 0u;
    if (role == VulkanRetiredResourceRole::Unknown) {
      if (phase == VulkanSubmitPhase::StackOwner) {
        role = stack_retired_resource_role_for_phase(
            current_vision_stack_phase());
      } else if (
          phase == VulkanSubmitPhase::ModelSetup ||
          phase == VulkanSubmitPhase::PatchEmbed ||
          phase == VulkanSubmitPhase::PositionalEmbeddingSetup) {
        role = VulkanRetiredResourceRole::SetupStaging;
      } else if (phase == VulkanSubmitPhase::Readback) {
        role = VulkanRetiredResourceRole::ReadbackStaging;
      }
    }
    if (stack_provenance.defined) {
      role = stack_provenance.producer_role;
      phase = VulkanSubmitPhase::StackOwner;
    }
    PendingRetireImage pending{
        std::move(image),
        VulkanRetiredResourceKind::Image,
        role,
        phase,
        callsite,
        bytes,
        std::move(stack_provenance)};
    if (external_recording_cmd()) {
      capture_external_recording_image_cleanup(std::move(pending));
      return;
    }
    const bool batch_candidate =
        is_safe_stack_temp_retire_batch_candidate(pending.stack_provenance);
    const bool stack_recording_active =
        batch_candidate && is_stack_planned_recording_active() &&
        stack_planned_recording_owned_by_current_thread();
    if (batch_candidate && stack_recording_active) {
      note_stack_internal_temp_retire_batch_decision(
          pending.stack_provenance,
          pending.bytes,
          stack_recording_active,
          /*accepted=*/true);
      mark_vulkan_memory_residency_state(
          pending.image.allocation_id(), "stack_batched_retire");
      std::lock_guard<std::mutex> batch_lock(
          stack_internal_temp_retire_batch_mutex_);
      stack_internal_temp_retire_batch_images_.push_back(std::move(pending));
      return;
    }
    if (pending.stack_provenance.defined) {
      note_stack_internal_temp_retire_batch_decision(
          pending.stack_provenance,
          pending.bytes,
          stack_recording_active,
          /*accepted=*/false);
    }
    if (pending.image.owns_memory()) {
      mark_vulkan_memory_residency_state(
          pending.image.allocation_id(), "pending_retire");
      pending_retire_bytes_.fetch_add(
          pending.bytes, std::memory_order_relaxed);
    }
    std::lock_guard<std::mutex> imagelist_lock(pending_retire_images_mutex_);
    pending_retire_images_.push_back(std::move(pending));
  }

  inline uint64_t pending_retire_bytes() const {
    return pending_retire_bytes_.load(std::memory_order_relaxed);
  }

  void poll_retire_queue();
  void submit_pending_work_and_poll_retire(
      PendingWorkRetireDrainPolicy policy =
          PendingWorkRetireDrainPolicy::SubmitOldPathPending);
  bool has_pending_work_for_current_stream() const;
  void flush_if_current_stream(const c10::Stream&);
  VulkanStreamState& current_stream();
  c10::Stream current_c10_stream();
  c10::Stream exchange_stream(c10::Stream);
  bool query_stream(const c10::Stream&);
  void synchronize_stream(const c10::Stream&);
  void synchronize_device();

  // GPU RPC

  inline std::unique_lock<std::mutex> dispatch_lock() {
    return std::unique_lock<std::mutex>(cmd_mutex_);
  }

  inline void set_cmd(bool reusable = false) {
    if (external_recording_cmd()) {
      return;
    }
    if (!cmd_) {
      cmd_ = command_pool_.get_new_cmd(reusable);
      cmd_.begin();
      command_buffer_recording_id_ = next_command_buffer_recording_id_++;
    }
  }

  DescriptorSet get_descriptor_set(const ShaderInfo&, const utils::uvec3&);

  void register_shader_dispatch(
      const DescriptorSet&,
      PipelineBarrier&,
      const ShaderInfo&,
      const utils::uvec3&);

  template <class S, class D>
  bool submit_copy(
      PipelineBarrier&,
      const S&,
      const D&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      VkFence fence_handle,
      VulkanSubmitOrigin origin = VulkanSubmitOrigin::Unknown);

  template <typename... Arguments>
  bool submit_compute_job(
      const ShaderInfo&,
      PipelineBarrier&,
      const utils::uvec3&,
      const utils::uvec3&,
      VkFence fence_handle,
      Arguments&&...);

  VulkanSubmission submit_cmd_to_gpu(
      VkFence fence_handle = VK_NULL_HANDLE,
      const bool final_use = false,
      VulkanSubmitOrigin origin = VulkanSubmitOrigin::Unknown);
  void flush_pending_cmds(
      PendingCommandFlushReason reason = PendingCommandFlushReason::Unknown,
      VkFence fence_handle = VK_NULL_HANDLE);
  bool is_stack_planned_recording_active() const;
  StackPlannedRecordingStats snapshot_stack_planned_recording_stats() const;
  StackRegionSingleRecordingPlanResult
  snapshot_stack_region_single_recording_plan(
      const StackRegionSingleRecordingPlanRequest& request) const;
  StackRegionSingleRecordingOwnerResult
  snapshot_stack_region_single_recording_owner(
      const StackRegionSingleRecordingOwnerRequest& request) const;
  StackRegionCommandBufferTopologyPlanResult
  snapshot_stack_region_command_buffer_topology_plan(
      const StackRegionCommandBufferTopologyPlanRequest& request) const;
  StackRegionCommandBufferAcquireHookResult
  request_stack_region_command_buffer_acquire(
      const StackRegionCommandBufferAcquireHookRequest& request) const;
  StackRegionCommandPoolResetDeferralOwnerResult
  snapshot_stack_region_command_pool_reset_deferral_owner(
      const StackRegionCommandPoolResetDeferralOwnerRequest& request) const;
  StackRegionRetireTimelineOwnerResult
  snapshot_stack_region_retire_timeline_owner(
      const StackRegionRetireTimelineOwnerRequest& request) const;
  StackRegionPendingRetireTransferResult
  snapshot_stack_region_pending_retire_transfer(
      const StackRegionPendingRetireTransferRequest& request);
  StackRegionPendingRetireTransferOwnerResult
  snapshot_stack_region_pending_retire_transfer_owner(
      const StackRegionPendingRetireTransferOwnerRequest& request) const;
  void begin_stack_planned_recording(
      bool allow_stack_owned_command_buffer_canary = true);
  StackPlannedRecordingStats end_stack_planned_recording_and_submit();
  StackPlannedRecordingStats cancel_stack_planned_recording();
  void set_external_recording_stack_segment_metadata(
      uint64_t segment_count,
      uint64_t segment_index,
      uint64_t segment_start_block,
      uint64_t segment_end_block,
      uint64_t segment_planned_dispatch_count);
  CommandBuffer acquire_persistent_command_buffer();
  void submit_prepared_command_buffer(
      CommandBuffer&,
      VkFence fence_handle = VK_NULL_HANDLE,
      const bool final_use = false,
      const char* profile_label = nullptr);
  void take_external_recording_cleanup_resources(
      std::vector<PendingRetireBuffer>& buffers,
      std::vector<PendingRetireImage>& images);
  void take_external_recording_cleanup_resources(
      std::vector<VulkanBuffer>& buffers,
      std::vector<VulkanImage>& images);

  void flush();
  void retire_after_fence_wait(bool flush_descriptor_pool = true);
  void flush_after_fence_wait();
};

class UniformParamsBuffer final {
 private:
  Context* context_p_;
  size_t nbytes_;
  VulkanRetiredResourceKind retire_kind_;
  VulkanRetiredResourceRole retire_role_;
  VulkanSubmitPhase retire_phase_;
  VulkanRetireCallSite retire_callsite_;
  VulkanBuffer vulkan_buffer_;

 public:
  UniformParamsBuffer()
      : context_p_{nullptr},
        nbytes_(0u),
        retire_kind_(VulkanRetiredResourceKind::Unknown),
        retire_role_(VulkanRetiredResourceRole::Unknown),
        retire_phase_(VulkanSubmitPhase::Unknown),
        retire_callsite_(VulkanRetireCallSite::Unknown),
        vulkan_buffer_{} {}

  template <typename Block>
  UniformParamsBuffer(Context* context_p, const Block& block)
      : context_p_(context_p),
        nbytes_(sizeof(block)),
        retire_kind_(current_retired_resource_kind()),
        retire_role_(current_retired_resource_role()),
        retire_phase_(current_submit_phase()),
        retire_callsite_(VulkanRetireCallSite::Unknown),
        vulkan_buffer_(
            context_p_->adapter_ptr()->vma().create_params_buffer(block)) {
    if (retire_kind_ == VulkanRetiredResourceKind::Unknown) {
      retire_kind_ = VulkanRetiredResourceKind::UniformBuffer;
    }
    if (
        retire_role_ == VulkanRetiredResourceRole::Unknown &&
        retire_phase_ == VulkanSubmitPhase::StackOwner) {
      retire_role_ =
          stack_retired_resource_role_for_phase(current_vision_stack_phase());
    }
  }

  UniformParamsBuffer(const UniformParamsBuffer&);
  UniformParamsBuffer& operator=(const UniformParamsBuffer&);

  UniformParamsBuffer(UniformParamsBuffer&&) = default;
  UniformParamsBuffer& operator=(UniformParamsBuffer&&) = default;

  ~UniformParamsBuffer() {
    if (vulkan_buffer_) {
      context_p_->register_buffer_cleanup(
          vulkan_buffer_,
          retire_kind_,
          retire_role_,
          retire_phase_,
          retire_callsite_);
    }
  }

  VulkanBuffer& buffer() {
    return vulkan_buffer_;
  }

  template <typename Block>
  void update(const Block& block) {
    if (sizeof(block) != nbytes_) {
      VK_THROW(
          "Attempted to update UniformParamsBuffer with data of different size");
    }
    // Fill the uniform buffer with data in block
    {
      MemoryMap mapping(vulkan_buffer_, MemoryAccessType::WRITE);
      Block* data_ptr = mapping.template data<Block>();

      *data_ptr = block;
    }
  }
};

class StorageBuffer final {
 private:
  Context* context_p_;
  ScalarType dtype_;
  size_t numel_;
  size_t nbytes_;
  VulkanRetiredResourceKind retire_kind_;
  VulkanRetiredResourceRole retire_role_;
  VulkanSubmitPhase retire_phase_;
  VulkanRetireCallSite retire_callsite_;
  VulkanBuffer vulkan_buffer_;

 public:
  StorageBuffer(
      Context* context_p,
      const ScalarType dtype,
      const size_t numel,
      const bool gpuonly = false,
      const MemoryAllocator::BufferHostAccess host_access =
          MemoryAllocator::BufferHostAccess::SequentialWrite)
      : context_p_(context_p),
        dtype_(dtype),
        numel_(numel),
        nbytes_(element_size(dtype_) * numel_),
        retire_kind_(VulkanRetiredResourceKind::Buffer),
        retire_role_(current_retired_resource_role()),
        retire_phase_(current_submit_phase()),
        retire_callsite_(VulkanRetireCallSite::Unknown),
        vulkan_buffer_(context_p_->adapter_ptr()->vma().create_storage_buffer(
            nbytes_,
            gpuonly,
            true,
            host_access)) {
    if (
        retire_role_ == VulkanRetiredResourceRole::Unknown &&
        retire_phase_ == VulkanSubmitPhase::StackOwner) {
      retire_role_ =
          stack_retired_resource_role_for_phase(current_vision_stack_phase());
    }
  }

  StorageBuffer(const StorageBuffer&) = delete;
  StorageBuffer& operator=(const StorageBuffer&) = delete;

  StorageBuffer(StorageBuffer&&) = default;
  StorageBuffer& operator=(StorageBuffer&&) = default;

  ~StorageBuffer() {
    context_p_->register_buffer_cleanup(
        vulkan_buffer_,
        retire_kind_,
        retire_role_,
        retire_phase_,
        retire_callsite_);
  }

  inline ScalarType dtype() {
    return dtype_;
  }

  inline VulkanBuffer& buffer() {
    return vulkan_buffer_;
  }

  inline size_t numel() {
    return numel_;
  }

  inline size_t nbytes() {
    return nbytes_;
  }
};

TORCH_API bool available();
TORCH_API c10::DeviceIndex device_count();
TORCH_API c10::DeviceIndex current_device();
TORCH_API void set_current_device(c10::DeviceIndex device_index);
TORCH_API c10::DeviceIndex exchange_device(c10::DeviceIndex device_index);

// The global runtime is retrieved using this function, where it is declared as
// a static local variable.
TORCH_API Context* context();
TORCH_API Context* context(c10::DeviceIndex device_index);

namespace detail {

inline void arg_is_empty(bool& any_is_empty, const VulkanBuffer& buffer) {
  // bool(buffer) will evaluate to false if no memory has been allocated
  any_is_empty = any_is_empty || !buffer;
}

inline void arg_is_empty(bool& any_is_empty, const VulkanImage& image) {
  // bool(image) will evaluate to false if no memory has been allocated
  any_is_empty = any_is_empty || !image;
}

/*
  Reports if any VulkanBuffer or VulkanImage argument in a variadic argument
  list does not have any memory associated with it.
 */
template <typename... Arguments>
inline bool any_arg_is_empty(Arguments&&... arguments) {
  bool any_is_empty = false;
  VK_UNUSED const int _[]{
      0,
      (arg_is_empty(any_is_empty, std::forward<Arguments>(arguments)), 0)...,
  };

  return any_is_empty;
}

template <size_t... Indices, typename... Arguments>
inline void bind(
    DescriptorSet& descriptor_set,
    const std::index_sequence<Indices...>&,
    Arguments&&... arguments) {
  VK_UNUSED const int _[]{
      0,
      (descriptor_set.bind(Indices, std::forward<Arguments>(arguments)), 0)...,
  };
}

inline void note_stack_live_descriptor_binding(
    const uint32_t binding_idx,
    const char* shader_name,
    const VulkanBuffer& buffer) {
  note_vulkan_stack_live_descriptor_binding(binding_idx, shader_name, buffer);
}

inline void note_stack_live_descriptor_binding(
    const uint32_t binding_idx,
    const char* shader_name,
    const VulkanImage& image) {
  note_vulkan_stack_live_image_descriptor_binding(
      binding_idx, shader_name, image);
}

inline void note_stack_pre_dispatch_proof_table_descriptor(
    const uint32_t binding_idx,
    const char* shader_name,
    const VulkanBuffer& buffer) {
  note_vulkan_stack_pre_dispatch_proof_table_descriptor(
      binding_idx, shader_name, buffer);
}

inline void note_stack_pre_dispatch_proof_table_descriptor(
    const uint32_t,
    const char*,
    const VulkanImage&) {}

inline void note_stack_barrier_only_canary_descriptor(
    const uint32_t binding_idx,
    const char* shader_name,
    const VulkanBuffer& buffer,
    PipelineBarrier& pipeline_barrier) {
  maybe_insert_vulkan_stack_barrier_only_canary_descriptor(
      binding_idx, shader_name, buffer, pipeline_barrier);
}

inline void note_stack_barrier_only_canary_descriptor(
    const uint32_t,
    const char*,
    const VulkanImage&,
    PipelineBarrier&) {}

template <size_t... Indices, typename... Arguments>
inline void note_stack_live_descriptor_bindings(
    const char* shader_name,
    const std::index_sequence<Indices...>&,
    Arguments&&... arguments) {
  VK_UNUSED const int _[]{
      0,
      (note_stack_live_descriptor_binding(
           Indices, shader_name, std::forward<Arguments>(arguments)),
       0)...,
  };
}

template <size_t... Indices, typename... Arguments>
inline void note_stack_pre_dispatch_proof_table_descriptors(
    const char* shader_name,
    const std::index_sequence<Indices...>&,
    Arguments&&... arguments) {
  VK_UNUSED const int _[]{
      0,
      (note_stack_pre_dispatch_proof_table_descriptor(
           Indices, shader_name, std::forward<Arguments>(arguments)),
       0)...,
  };
}

template <size_t... Indices, typename... Arguments>
inline void note_stack_barrier_only_canary_descriptors(
    const char* shader_name,
    PipelineBarrier& pipeline_barrier,
    const std::index_sequence<Indices...>&,
    Arguments&&... arguments) {
  VK_UNUSED const int _[]{
      0,
      (note_stack_barrier_only_canary_descriptor(
           Indices,
           shader_name,
           std::forward<Arguments>(arguments),
           pipeline_barrier),
       0)...,
  };
}

} // namespace detail

template <class S, class D>
inline void record_copy(
    CommandBuffer& cmd,
    const S& source,
    const D& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) = delete;

template <>
inline void record_copy<VulkanBuffer, VulkanBuffer>(
    CommandBuffer& cmd,
    const VulkanBuffer& source,
    const VulkanBuffer& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  cmd.copy_buffer_to_buffer(
      source, destination, copy_range, src_offset, dst_offset);
}

template <>
inline void record_copy<VulkanImage, VulkanImage>(
    CommandBuffer& cmd,
    const VulkanImage& source,
    const VulkanImage& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  cmd.copy_texture_to_texture(
      source, destination, copy_range, src_offset, dst_offset);
}

template <>
inline void record_copy<VulkanImage, VulkanBuffer>(
    CommandBuffer& cmd,
    const VulkanImage& source,
    const VulkanBuffer& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  cmd.copy_texture_to_buffer(
      source, destination, copy_range, src_offset, dst_offset);
}

template <>
inline void record_copy<VulkanBuffer, VulkanImage>(
    CommandBuffer& cmd,
    const VulkanBuffer& source,
    const VulkanImage& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  cmd.copy_buffer_to_texture(
      source, destination, copy_range, src_offset, dst_offset);
}

/*
  Records a GPU data copy into the current command buffer. If the number of
  submit_*_job calls exceeds the configured frequency, or if a fence is
  provided, then the command buffer is submitted to the GPU for execution.
  Returns a bool indicating whether or not the function call resulted in a GPU
  queue submission.
 */
template <class S, class D>
inline bool Context::submit_copy(
    PipelineBarrier& pipeline_barrier,
    const S& source,
    const D& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset,
    VkFence fence_handle,
    VulkanSubmitOrigin origin) {
  const bool external_recording = external_recording_cmd() != nullptr;
  const bool graph_program_invocation =
      graph_program_invocation_active_for_current_thread();
  VK_CHECK_COND(
      !graph_program_invocation_active() || graph_program_invocation,
      "Vulkan graph program invocation is active on another thread");
  const bool stack_planned_recording =
      is_stack_planned_recording_active() && !external_recording;
  VK_CHECK_COND(
      !stack_planned_recording || stack_planned_recording_owned_by_current_thread(),
      "Vulkan stack planned recording used from the wrong thread");
  VK_CHECK_COND(
      !graph_program_invocation ||
          (!external_recording && !stack_planned_recording &&
           fence_handle == VK_NULL_HANDLE),
      "Vulkan graph program invocation requires normal unfenced Context "
      "recording");
  const bool cpu_timeline = cpu_timeline_logging_enabled();
  const uint64_t cpu_start_us =
      cpu_timeline ? cpu_timeline_now_us() : 0u;
  const VulkanSubmitOrigin fenced_submit_origin =
      origin == VulkanSubmitOrigin::Unknown
      ? VulkanSubmitOrigin::TensorCpuReadback
      : origin;

  // If any of the provided arguments does not have memory associated with it,
  // then exit early as there is no work to be done. However, if a fence has
  // been passed the command buffer is not empty, then the current command
  // buffer must still be submitted so that the fence can be signaled.
  if (!source || !destination) {
    if (!external_recording && fence_handle != VK_NULL_HANDLE &&
        submit_count_ > 0) {
      submit_cmd_to_gpu(
          fence_handle, false, fenced_submit_origin);
      if (cpu_timeline) {
        std::ostringstream stream;
        stream << "event=submit_copy_empty submitted=1 record_us="
               << (cpu_timeline_now_us() - cpu_start_us)
               << " fence=" << (fence_handle != VK_NULL_HANDLE ? 1 : 0);
        append_cpu_timeline_log_line(stream.str());
      }
      return true;
    }
    if (cpu_timeline) {
      std::ostringstream stream;
      stream << "event=submit_copy_empty submitted=0 record_us="
             << (cpu_timeline_now_us() - cpu_start_us)
             << " fence=" << (fence_handle != VK_NULL_HANDLE ? 1 : 0);
      append_cpu_timeline_log_line(stream.str());
    }
    return false;
  }

  // Serialize recording to the shared command buffer. Do not initialize with a
  // mutex just yet, since in some cases it will be externally managed.
  std::unique_lock<std::mutex> cmd_lock;
  // Refer to comments in submit_compute_job for explanation.
  if (!external_recording && !graph_program_invocation &&
      fence_handle == VK_NULL_HANDLE) {
    cmd_lock = std::unique_lock<std::mutex>(cmd_mutex_);
  }

  set_cmd();
  CommandBuffer& cmd = active_cmd();

  uint32_t log_idx = UINT32_MAX;
  if (enable_op_profiling_ && !external_recording) {
    std::string label = "cmd_copy";
    log_idx = gpu_profile_begin(
        cmd, label, create_extent3d({0, 0, 0}), create_extent3d({0, 0, 0}));
  }

  cmd.insert_barrier(pipeline_barrier);

  record_copy(cmd, source, destination, copy_range, src_offset, dst_offset);

  if (enable_op_profiling_ && !external_recording) {
    gpu_profile_end(cmd, log_idx);
  }

  if (external_recording) {
    if (stack_planned_recording_active_.load(std::memory_order_acquire) &&
        stack_region_owned_command_buffer_active_.load(
            std::memory_order_acquire)) {
      stack_region_owned_recording_dispatch_count_++;
    }
    if (cpu_timeline) {
      std::ostringstream stream;
      stream << "event=submit_copy submitted=0 record_us="
             << (cpu_timeline_now_us() - cpu_start_us)
             << " fence=" << (fence_handle != VK_NULL_HANDLE ? 1 : 0)
             << " external_recording=1"
             << " copy_range=" << copy_range.data[0u] << "x"
             << copy_range.data[1u] << "x" << copy_range.data[2u];
      append_cpu_timeline_log_line(stream.str());
    }
    return false;
  }

  submit_count_++;
  bool submitted = false;
  if (fence_handle != VK_NULL_HANDLE ||
      (!stack_planned_recording &&
       !graph_program_invocation &&
       submit_count_ >= config_.cmdSubmitFrequency)) {
    submit_cmd_to_gpu(
        fence_handle,
        false,
        fence_handle != VK_NULL_HANDLE
            ? fenced_submit_origin
            : VulkanSubmitOrigin::NormalCmdSubmitFrequency);
    submitted = true;
  } else if (
      stack_planned_recording &&
      submit_count_ >= config_.cmdSubmitFrequency) {
    stack_planned_recording_stats_.suppressed_frequency_flushes++;
  }
  if (cpu_timeline) {
    std::ostringstream stream;
    stream << "event=submit_copy submitted=" << (submitted ? 1 : 0)
           << " record_us=" << (cpu_timeline_now_us() - cpu_start_us)
           << " fence=" << (fence_handle != VK_NULL_HANDLE ? 1 : 0)
           << " copy_range=" << copy_range.data[0u] << "x"
           << copy_range.data[1u] << "x" << copy_range.data[2u];
    append_cpu_timeline_log_line(stream.str());
  }
  return submitted;
}

/*
  Records a compute shader dispatch into the current command buffer. If the
  number of submit_*_job calls exceeds the configured frequency, or if a fence
  is provided, then the command buffer is submitted to the GPU for execution.
  Returns a bool indicating whether or not the function call resulted in a GPU
  queue submission.
 */
template <typename... Arguments>
inline bool Context::submit_compute_job(
    const ShaderInfo& shader,
    PipelineBarrier& pipeline_barrier,
    const utils::uvec3& global_work_group,
    const utils::uvec3& local_work_group_size,
    VkFence fence_handle,
    Arguments&&... arguments) {
  const bool external_recording = external_recording_cmd() != nullptr;
  const bool graph_program_invocation =
      graph_program_invocation_active_for_current_thread();
  VK_CHECK_COND(
      !graph_program_invocation_active() || graph_program_invocation,
      "Vulkan graph program invocation is active on another thread");
  const bool stack_planned_recording =
      is_stack_planned_recording_active() && !external_recording;
  VK_CHECK_COND(
      !stack_planned_recording ||
          stack_planned_recording_owned_by_current_thread(),
      "Vulkan stack planned recording used from the wrong thread");
  VK_CHECK_COND(
      !graph_program_invocation ||
          (!external_recording && !stack_planned_recording &&
           fence_handle == VK_NULL_HANDLE),
      "Vulkan graph program invocation requires normal unfenced Context "
      "recording");
  const bool cpu_timeline = cpu_timeline_logging_enabled();
  const uint64_t cpu_start_us =
      cpu_timeline ? cpu_timeline_now_us() : 0u;

  // If any of the provided arguments does not have memory associated with it,
  // then exit early as there is no work to be done. However, if a fence has
  // been passed the command buffer is not empty, then the current command
  // buffer must still be submitted so that the fence can be signaled.
  if (detail::any_arg_is_empty(arguments...)) {
    if (!external_recording && fence_handle != VK_NULL_HANDLE &&
        submit_count_ > 0) {
      submit_cmd_to_gpu(
          fence_handle, false, VulkanSubmitOrigin::TensorCpuReadback);
      if (cpu_timeline) {
        std::ostringstream stream;
        stream << "event=submit_compute_empty kernel=" << shader.kernel_name
               << " submitted=1 record_us="
               << (cpu_timeline_now_us() - cpu_start_us)
               << " fence=" << (fence_handle != VK_NULL_HANDLE ? 1 : 0);
        append_cpu_timeline_log_line(stream.str());
      }
      return true;
    }
    if (cpu_timeline) {
      std::ostringstream stream;
      stream << "event=submit_compute_empty kernel=" << shader.kernel_name
             << " submitted=0 record_us="
             << (cpu_timeline_now_us() - cpu_start_us)
             << " fence=" << (fence_handle != VK_NULL_HANDLE ? 1 : 0);
      append_cpu_timeline_log_line(stream.str());
    }
    return false;
  }
  vulkan_sync_counters().submit_compute_job_count.fetch_add(
      1u,
      std::memory_order_relaxed);

  // Serialize recording to the shared command buffer. Do not initialize with a
  // mutex just yet, since in some cases it will be externally managed.
  std::unique_lock<std::mutex> cmd_lock;
  // If a fence was passed, then assume that the host intends to sync with
  // the GPU, implying there will be imminent calls to fence.wait() and flush().
  // We therefore assume the mutex is externally managed in this case, and the
  // calling thread has already locked the mutex prior to calling the function,
  // and will release the mutex manually after calling flush(). This will
  // prevent more dispatches from being recorded until we have flushed the
  // Context.
  if (!external_recording && !graph_program_invocation &&
      fence_handle == VK_NULL_HANDLE) {
    cmd_lock = std::unique_lock<std::mutex>(cmd_mutex_);
  }

  set_cmd();
  CommandBuffer& cmd = active_cmd();

  uint32_t log_idx = UINT32_MAX;
  if (enable_op_profiling_) {
    log_idx = gpu_profile_begin(
        cmd,
        shader.kernel_name,
        create_extent3d(global_work_group),
        create_extent3d(local_work_group_size));
  }

  // Factor out template parameter independent code to minimize code bloat.
  DescriptorSet descriptor_set =
      get_descriptor_set(shader, local_work_group_size);

  if (!external_recording) {
    const uint64_t submit_epoch_before =
        current_stream().last_submitted_value.load(std::memory_order_relaxed);
    set_stack_region_command_buffer_diagnostic_context(
        command_buffer_recording_id_, submit_epoch_before);
  }

  if (stack_descriptor_dependency_diagnostics_enabled()) {
    detail::note_stack_live_descriptor_bindings(
        shader.kernel_name.c_str(),
        std::index_sequence_for<Arguments...>{},
        std::forward<Arguments>(arguments)...);
    detail::note_stack_pre_dispatch_proof_table_descriptors(
        shader.kernel_name.c_str(),
        std::index_sequence_for<Arguments...>{},
        std::forward<Arguments>(arguments)...);
    detail::note_stack_barrier_only_canary_descriptors(
        shader.kernel_name.c_str(),
        pipeline_barrier,
        std::index_sequence_for<Arguments...>{},
        std::forward<Arguments>(arguments)...);
  }

  detail::bind(
      descriptor_set,
      std::index_sequence_for<Arguments...>{},
      std::forward<Arguments>(arguments)...);

  note_vulkan_stack_pre_dispatch_insertion_point(shader.kernel_name.c_str());

  // Factor out template parameter independent code to minimize code bloat.
  register_shader_dispatch(
      descriptor_set, pipeline_barrier, shader, global_work_group);
  vulkan_sync_counters().compute_dispatch_count.fetch_add(
      1u,
      std::memory_order_relaxed);
  note_vulkan_stack_dispatch(shader.kernel_name.c_str());
  if (external_recording && stack_planned_recording_active_.load(
                                std::memory_order_acquire) &&
      stack_region_owned_command_buffer_active_.load(
          std::memory_order_acquire)) {
    stack_region_owned_recording_dispatch_count_++;
  }
  if (stack_planned_recording) {
    const uint64_t stack_compute_job =
        ++stack_planned_recording_stats_.recorded_compute_jobs;
    stack_planned_recording_stats_.recorded_descriptor_writes +=
        sizeof...(Arguments);
    stack_planned_recording_stats_.recorded_barriers++;
    if (config_.cmdSubmitFrequency > 0u &&
        stack_compute_job % config_.cmdSubmitFrequency == 0u) {
      stack_planned_recording_stats_.suppressed_frequency_flushes++;
    }
  }

  if (enable_op_profiling_) {
    gpu_profile_end(cmd, log_idx);
  }

  if (external_recording) {
    if (cpu_timeline) {
      std::ostringstream stream;
      stream << "event=submit_compute kernel=" << shader.kernel_name
             << " submitted=0 record_us="
             << (cpu_timeline_now_us() - cpu_start_us)
             << " fence=" << (fence_handle != VK_NULL_HANDLE ? 1 : 0)
             << " external_recording=1"
             << " global=" << global_work_group.data[0u] << "x"
             << global_work_group.data[1u] << "x"
             << global_work_group.data[2u]
             << " local=" << local_work_group_size.data[0u] << "x"
             << local_work_group_size.data[1u] << "x"
             << local_work_group_size.data[2u];
      append_cpu_timeline_log_line(stream.str());
    }
    return false;
  }

  submit_count_++;
  bool submitted = false;
  if (fence_handle != VK_NULL_HANDLE ||
      (!stack_planned_recording &&
       !graph_program_invocation &&
       submit_count_ >= config_.cmdSubmitFrequency)) {
    if (stack_planned_recording) {
      stack_planned_recording_stats_.premature_submits++;
    }
    submit_cmd_to_gpu(
        fence_handle,
        false,
        fence_handle != VK_NULL_HANDLE
            ? VulkanSubmitOrigin::TensorCpuReadback
            : VulkanSubmitOrigin::NormalCmdSubmitFrequency);
    submitted = true;
  }

  if (cpu_timeline) {
    std::ostringstream stream;
    stream << "event=submit_compute kernel=" << shader.kernel_name
           << " submitted=" << (submitted ? 1 : 0)
           << " record_us=" << (cpu_timeline_now_us() - cpu_start_us)
           << " fence=" << (fence_handle != VK_NULL_HANDLE ? 1 : 0)
           << " global=" << global_work_group.data[0u] << "x"
           << global_work_group.data[1u] << "x"
           << global_work_group.data[2u]
           << " local=" << local_work_group_size.data[0u] << "x"
           << local_work_group_size.data[1u] << "x"
           << local_work_group_size.data[2u];
    append_cpu_timeline_log_line(stream.str());
  }

  return submitted;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
