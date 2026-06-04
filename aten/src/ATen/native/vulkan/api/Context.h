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
#include <sstream>
#include <string>
#include <thread>

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
  FencePool fences_;
  // Diagnostics
  bool enable_op_profiling_{false};
  QueryPool querypool_;
  // Command buffers submission
  std::mutex cmd_mutex_;
  CommandBuffer cmd_;
  uint32_t submit_count_;
  std::atomic<bool> stack_planned_recording_active_;
  std::thread::id stack_planned_recording_owner_;
  StackPlannedRecordingStats stack_planned_recording_stats_;
  // Memory Management
  std::mutex pending_retire_buffers_mutex_;
  std::vector<PendingRetireBuffer> pending_retire_buffers_;
  std::mutex pending_retire_images_mutex_;
  std::vector<PendingRetireImage> pending_retire_images_;
  std::atomic<uint64_t> pending_retire_bytes_;
  std::mutex stack_internal_temp_retire_batch_mutex_;
  std::vector<PendingRetireBuffer> stack_internal_temp_retire_batch_buffers_;
  std::vector<PendingRetireImage> stack_internal_temp_retire_batch_images_;
  RetireQueue retire_queue_;
  VulkanSubmission last_submission_;

  void clear_pending_retire_resources_locked();
  void clear_stack_internal_temp_retire_batch_locked();
  void restore_stack_internal_temp_retire_batch_to_pending_locked();
  void retire_stack_internal_temp_retire_batch_locked(
      const VulkanSubmission& submission);
  CommandBuffer* external_recording_cmd();
  const CommandBuffer* external_recording_cmd() const;
  bool is_inside_owned_program_recording() const;
  bool stack_planned_recording_owned_by_current_thread() const;
  DescriptorPool& active_descriptor_pool();
  CommandBuffer& active_cmd();
  void capture_external_recording_buffer_cleanup(VulkanBuffer&&);
  void capture_external_recording_image_cleanup(VulkanImage&&);
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
  std::string format_submit_failure_diagnostics(
      const VulkanStreamState&,
      VulkanSubmitOrigin origin,
      uint64_t signal_value,
      size_t wait_count,
      VkFence fence_handle,
      bool final_use);
  void retire_deferred_cleanup(VulkanSubmission, VulkanSubmitOrigin);

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
    if (external_recording_cmd()) {
      capture_external_recording_buffer_cleanup(std::move(buffer));
      return;
    }
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

  void register_image_cleanup(
      VulkanImage& image,
      VulkanRetiredResourceRole role = VulkanRetiredResourceRole::Unknown,
      VulkanSubmitPhase phase = current_submit_phase(),
      VulkanRetireCallSite callsite = VulkanRetireCallSite::Unknown,
      VulkanStackRetireProvenance stack_provenance = {}) {
    if (external_recording_cmd()) {
      capture_external_recording_image_cleanup(std::move(image));
      return;
    }
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
      VkFence fence_handle);

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
  void flush_pending_cmds(VkFence fence_handle = VK_NULL_HANDLE);
  bool is_stack_planned_recording_active() const;
  void begin_stack_planned_recording();
  StackPlannedRecordingStats end_stack_planned_recording_and_submit();
  StackPlannedRecordingStats cancel_stack_planned_recording();
  CommandBuffer acquire_persistent_command_buffer();
  void submit_prepared_command_buffer(
      CommandBuffer&,
      VkFence fence_handle = VK_NULL_HANDLE,
      const bool final_use = false,
      const char* profile_label = nullptr);
  void take_external_recording_cleanup_resources(
      std::vector<VulkanBuffer>& buffers,
      std::vector<VulkanImage>& images);

  void flush();
  void retire_after_fence_wait();
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
            context_p_->adapter_ptr()->vma().create_params_buffer(block)) {}

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
        vulkan_buffer_(context_p_->adapter_ptr()->vma().create_storage_buffer(
            nbytes_,
            gpuonly,
            true,
            host_access)) {}

  StorageBuffer(const StorageBuffer&) = delete;
  StorageBuffer& operator=(const StorageBuffer&) = delete;

  StorageBuffer(StorageBuffer&&) = default;
  StorageBuffer& operator=(StorageBuffer&&) = default;

  ~StorageBuffer() {
    context_p_->register_buffer_cleanup(vulkan_buffer_);
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
    VkFence fence_handle) {
  const bool external_recording = external_recording_cmd() != nullptr;
  const bool stack_planned_recording =
      is_stack_planned_recording_active() && !external_recording;
  VK_CHECK_COND(
      !stack_planned_recording || stack_planned_recording_owned_by_current_thread(),
      "Vulkan stack planned recording used from the wrong thread");
  const bool cpu_timeline =
      cpu_timeline_logging_enabled() && !external_recording;
  const uint64_t cpu_start_us =
      cpu_timeline ? cpu_timeline_now_us() : 0u;

  // If any of the provided arguments does not have memory associated with it,
  // then exit early as there is no work to be done. However, if a fence has
  // been passed the command buffer is not empty, then the current command
  // buffer must still be submitted so that the fence can be signaled.
  if (!source || !destination) {
    if (!external_recording && fence_handle != VK_NULL_HANDLE &&
        submit_count_ > 0) {
      submit_cmd_to_gpu(
          fence_handle, false, VulkanSubmitOrigin::TensorCpuReadback);
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
  if (!external_recording && fence_handle == VK_NULL_HANDLE) {
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
    return false;
  }

  submit_count_++;
  bool submitted = false;
  if (fence_handle != VK_NULL_HANDLE ||
      (!stack_planned_recording &&
       submit_count_ >= config_.cmdSubmitFrequency)) {
    submit_cmd_to_gpu(
        fence_handle,
        false,
        fence_handle != VK_NULL_HANDLE
            ? VulkanSubmitOrigin::TensorCpuReadback
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
  const bool stack_planned_recording =
      is_stack_planned_recording_active() && !external_recording;
  VK_CHECK_COND(
      !stack_planned_recording ||
          stack_planned_recording_owned_by_current_thread(),
      "Vulkan stack planned recording used from the wrong thread");
  const bool cpu_timeline =
      cpu_timeline_logging_enabled() && !external_recording;
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
  if (!external_recording && fence_handle == VK_NULL_HANDLE) {
    cmd_lock = std::unique_lock<std::mutex>(cmd_mutex_);
  }

  set_cmd();
  CommandBuffer& cmd = active_cmd();

  uint32_t log_idx = UINT32_MAX;
  if (enable_op_profiling_ && !external_recording) {
    log_idx = gpu_profile_begin(
        cmd,
        shader.kernel_name,
        create_extent3d(global_work_group),
        create_extent3d(local_work_group_size));
  }

  // Factor out template parameter independent code to minimize code bloat.
  DescriptorSet descriptor_set =
      get_descriptor_set(shader, local_work_group_size);

  detail::bind(
      descriptor_set,
      std::index_sequence_for<Arguments...>{},
      std::forward<Arguments>(arguments)...);

  // Factor out template parameter independent code to minimize code bloat.
  register_shader_dispatch(
      descriptor_set, pipeline_barrier, shader, global_work_group);
  vulkan_sync_counters().compute_dispatch_count.fetch_add(
      1u,
      std::memory_order_relaxed);
  note_vulkan_stack_dispatch(shader.kernel_name.c_str());
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

  if (enable_op_profiling_ && !external_recording) {
    gpu_profile_end(cmd, log_idx);
  }

  if (external_recording) {
    return false;
  }

  submit_count_++;
  bool submitted = false;
  if (fence_handle != VK_NULL_HANDLE ||
      (!stack_planned_recording &&
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
