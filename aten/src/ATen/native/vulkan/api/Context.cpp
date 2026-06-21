#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Sync.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <sstream>
#include <set>
#include <unordered_map>

#ifndef VULKAN_DESCRIPTOR_POOL_SIZE
#define VULKAN_DESCRIPTOR_POOL_SIZE 1024u
#endif

#ifndef VULKAN_QUERY_POOL_SIZE
#define VULKAN_QUERY_POOL_SIZE 4096u
#endif

namespace at {
namespace native {
namespace vulkan {
namespace api {

namespace {

const std::string& sync_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_SYNC_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool sync_logging_enabled() {
  return !sync_log_path().empty();
}

const std::string& gpu_timestamp_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_GPU_TIMESTAMP_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool gpu_timestamp_logging_enabled() {
  return !gpu_timestamp_log_path().empty();
}

const std::string& cpu_timeline_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_CPU_TIMELINE_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

const std::string& cpu_timeline_summary_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_CPU_TIMELINE_SUMMARY_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool cpu_timeline_line_logging_enabled() {
  return !cpu_timeline_log_path().empty();
}

bool cpu_timeline_summary_logging_enabled() {
  return !cpu_timeline_summary_log_path().empty();
}

VulkanStackRawResourceAllocationProof stack_raw_allocation_proof(
    const PendingRetireBuffer& pending) {
  VulkanStackRawResourceAllocationProof proof;
  proof.allocation_id = pending.buffer.allocation_id();
  proof.allocation_generation =
      vulkan_memory_allocation_generation(proof.allocation_id);
  proof.byte_offset = static_cast<uint64_t>(pending.buffer.mem_offset());
  proof.byte_range = static_cast<uint64_t>(pending.buffer.mem_range());
  proof.allocated_bytes = pending.bytes;
  proof.has_generation =
      proof.allocation_id != 0u && proof.allocation_generation != 0u;
  proof.has_byte_range = pending.buffer.has_memory() &&
      pending.buffer.owns_memory() && proof.byte_range != 0u &&
      proof.allocated_bytes != 0u;
  return proof;
}

VulkanStackRawResourceAllocationProof stack_raw_allocation_proof(
    const PendingRetireImage& pending) {
  VulkanStackRawResourceAllocationProof proof;
  proof.allocation_id = pending.image.allocation_id();
  proof.allocation_generation =
      vulkan_memory_allocation_generation(proof.allocation_id);
  proof.allocated_bytes = pending.bytes;
  proof.has_generation =
      proof.allocation_id != 0u && proof.allocation_generation != 0u;
  return proof;
}

const std::string& pending_retire_allocation_label(
    const PendingRetireBuffer& pending) {
  return pending.buffer.allocation_label();
}

const std::string& pending_retire_allocation_label(
    const PendingRetireImage& pending) {
  return pending.image.allocation_label();
}

VulkanRetireDrainReason retire_drain_reason_for_current_phase() {
  switch (current_submit_phase()) {
    case VulkanSubmitPhase::ModelSetup:
    case VulkanSubmitPhase::PatchEmbed:
    case VulkanSubmitPhase::PositionalEmbeddingSetup:
      return VulkanRetireDrainReason::SetupPhase;
    case VulkanSubmitPhase::StackOwner:
    case VulkanSubmitPhase::StackOwnerNorm:
    case VulkanSubmitPhase::StackOwnerAttention:
    case VulkanSubmitPhase::StackOwnerLinear:
    case VulkanSubmitPhase::StackOwnerResidual:
      return VulkanRetireDrainReason::StackScopeEnd;
    case VulkanSubmitPhase::Decoder:
    case VulkanSubmitPhase::DecoderConv:
    case VulkanSubmitPhase::DecoderUpsample:
    case VulkanSubmitPhase::DecoderPointwise:
      return VulkanRetireDrainReason::DecoderPhase;
    case VulkanSubmitPhase::Readback:
      return VulkanRetireDrainReason::ReadbackPreparation;
    case VulkanSubmitPhase::ExplicitSynchronize:
      return VulkanRetireDrainReason::Synchronize;
    case VulkanSubmitPhase::Shutdown:
      return VulkanRetireDrainReason::Shutdown;
    case VulkanSubmitPhase::TestHarness:
      return VulkanRetireDrainReason::DebugValidation;
    case VulkanSubmitPhase::Retire:
      return VulkanRetireDrainReason::ExplicitDrain;
    case VulkanSubmitPhase::Profiling:
      return VulkanRetireDrainReason::ResourcePressure;
    case VulkanSubmitPhase::Unknown:
    default:
      return VulkanRetireDrainReason::ResourcePressure;
  }
}

VulkanRetireCallSite retire_call_site_for_current_phase() {
  if (current_submit_phase() == VulkanSubmitPhase::StackOwner) {
    switch (current_vision_stack_phase()) {
      case VulkanVisionStackPhase::Norm1:
        return VulkanRetireCallSite::StackOwnerNorm1;
      case VulkanVisionStackPhase::Norm2:
        return VulkanRetireCallSite::StackOwnerNorm2;
      case VulkanVisionStackPhase::Attention:
        return VulkanRetireCallSite::StackOwnerAttention;
      case VulkanVisionStackPhase::QkvLinear:
      case VulkanVisionStackPhase::ProjLinear:
      case VulkanVisionStackPhase::Fc1Gelu:
      case VulkanVisionStackPhase::Fc2:
        return VulkanRetireCallSite::StackOwnerLinear;
      case VulkanVisionStackPhase::Residual1:
      case VulkanVisionStackPhase::Residual2:
        return VulkanRetireCallSite::StackOwnerResidual;
      default:
        return VulkanRetireCallSite::StackOwnerPhaseBoundary;
    }
  }
  switch (current_submit_phase()) {
    case VulkanSubmitPhase::Readback:
      return VulkanRetireCallSite::BenchmarkReadback;
    case VulkanSubmitPhase::ModelSetup:
    case VulkanSubmitPhase::PatchEmbed:
    case VulkanSubmitPhase::PositionalEmbeddingSetup:
      return VulkanRetireCallSite::BenchmarkSetup;
    case VulkanSubmitPhase::ExplicitSynchronize:
      return VulkanRetireCallSite::ContextExplicitSynchronize;
    case VulkanSubmitPhase::Shutdown:
      return VulkanRetireCallSite::ContextShutdown;
    case VulkanSubmitPhase::TestHarness:
      return VulkanRetireCallSite::DebugValidation;
    default:
      return VulkanRetireCallSite::ContextFlushPending;
  }
}

template <typename PendingRetire>
void append_region_lifetime_submit_signature(
    const PendingRetire& pending,
    const VulkanRetireCallSite default_callsite,
    std::map<std::string, std::pair<uint64_t, uint64_t>>& resources,
    std::set<std::string>& blockers) {
  const bool qkv_would_batch =
      is_qkv_stack_temp_retire_batch_candidate(pending.stack_provenance);
  const char* const blocker_reason = stack_retire_drain_blocker_reason(
      pending.kind, pending.role, pending.stack_provenance, qkv_would_batch);
  const VulkanStackTempLifetimeSafety safety =
      stack_retire_lifetime_safety_for_resource(
          pending.role, pending.stack_provenance);
  const VulkanRetireCallSite effective_callsite =
      pending.callsite == VulkanRetireCallSite::Unknown ? default_callsite
                                                        : pending.callsite;
  std::ostringstream key;
  key << retired_resource_kind_name(pending.kind) << ":"
      << retired_resource_role_name(pending.role) << ":"
      << retire_call_site_name(effective_callsite) << ":" << blocker_reason
      << ":" << stack_temp_lifetime_safety_name(safety);
  auto& value = resources[key.str()];
  value.first += 1u;
  value.second += pending.bytes;
  blockers.insert(blocker_reason);
}

struct RegionLifetimeSubmitResourceAttribution final {
  VulkanSubmitPhase phase = VulkanSubmitPhase::Unknown;
  VulkanRetireCallSite callsite = VulkanRetireCallSite::Unknown;
  VulkanRetiredResourceKind kind = VulkanRetiredResourceKind::Unknown;
  VulkanRetiredResourceRole role = VulkanRetiredResourceRole::Unknown;
  uint64_t bytes = 0u;
  const char* reason = "unknown";
  VulkanStackTempLifetimeSafety safety = VulkanStackTempLifetimeSafety::Unknown;
  VulkanStackRetireProvenance provenance;
  VulkanStackRawResourceAllocationProof allocation_proof;
  std::string allocation_label;
};

template <typename PendingRetire>
RegionLifetimeSubmitResourceAttribution
make_region_lifetime_submit_resource_attribution(
    const PendingRetire& pending,
    const VulkanSubmitPhase phase,
    const VulkanRetireCallSite default_callsite) {
  const bool qkv_would_batch =
      is_qkv_stack_temp_retire_batch_candidate(pending.stack_provenance);
  const VulkanRetireCallSite effective_callsite =
      pending.callsite == VulkanRetireCallSite::Unknown ? default_callsite
                                                        : pending.callsite;
  return RegionLifetimeSubmitResourceAttribution{
      phase,
      effective_callsite,
      pending.kind,
      pending.role,
      pending.bytes,
      stack_retire_drain_blocker_reason(
          pending.kind, pending.role, pending.stack_provenance, qkv_would_batch),
      stack_retire_lifetime_safety_for_resource(
          pending.role, pending.stack_provenance),
      pending.stack_provenance,
      stack_raw_allocation_proof(pending),
      pending_retire_allocation_label(pending)};
}

std::string format_region_lifetime_submit_signature(
    const std::map<std::string, std::pair<uint64_t, uint64_t>>& resources) {
  std::ostringstream signature;
  for (const auto& entry : resources) {
    if (signature.tellp() > 0) {
      signature << ",";
    }
    signature << entry.first << "#" << entry.second.first << "#"
              << entry.second.second;
  }
  return signature.str();
}

std::string format_region_lifetime_submit_blockers(
    const std::set<std::string>& blockers) {
  std::ostringstream signature;
  for (const auto& blocker : blockers) {
    if (signature.tellp() > 0) {
      signature << ",";
    }
    signature << blocker;
  }
  return signature.str();
}

std::string stack_region_diagnostic_token(const std::string& value) {
  if (value.empty()) {
    return "none";
  }
  std::string token;
  token.reserve(value.size());
  for (const char c : value) {
    if (
        c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '#' ||
        c == ',' || c == '|' || c == '=') {
      token.push_back('_');
    } else {
      token.push_back(c);
    }
  }
  return token.empty() ? "none" : token;
}

bool stack_region_raw_provenance_diagnostic_class(
    const char* const resource_class) {
  const std::string key(resource_class ? resource_class : "");
  return key == "unscoped_raw_buffer_no_stack_proof" ||
      key == "stack_internal_raw_generation_range" ||
      key == "stack_internal_temp_raw_generation_range_missing_last_consumer" ||
      key == "stack_qkv_output_raw_generation_range_non_escape_last_consumer" ||
      key == "stack_proj_output_raw_generation_range_non_escape_last_consumer" ||
      key ==
      "stack_residual1_output_raw_generation_range_non_escape_last_consumer" ||
      key == "raw_no_provenance" ||
      key == "truly_unknown_raw_resource" ||
      key == "non_stack_setup_staging_pending" ||
      key == "capture_sensitive_stack_activation" ||
      key == "missing_stack_activation_proof" ||
      key == "host_visible_or_requested_output";
}

const char* stack_region_raw_provenance_status(
    const bool safe_candidate,
    const bool large_backing,
    const bool formal_last_use_proof,
    const VulkanStackRetireProvenance& provenance,
    const VulkanStackRawResourceAllocationProof& allocation_proof) {
  if (safe_candidate && !large_backing) {
    return formal_last_use_proof ? "formal_last_use_proven"
                                 : "retire_only_or_nonescaping_proven";
  }
  if (large_backing) {
    return "large_backing_blocked";
  }
  if (!provenance.defined) {
    if (allocation_proof.has_generation && allocation_proof.has_byte_range) {
      return "missing_stack_scope_proof_with_allocation_range";
    }
    if (allocation_proof.has_generation) {
      return "missing_stack_scope_proof_with_generation_only";
    }
    return "missing_stack_scope_and_allocation_proof";
  }
  if (
      provenance.requested_intermediate || provenance.escapes_stack ||
      provenance.final_output) {
    return "public_or_host_visible_blocker";
  }
  if (
      provenance.alias_or_view || provenance.aliases_runtime_input ||
      provenance.aliases_runtime_output) {
    return "alias_or_escape_uncertain";
  }
  if (!provenance.has_last_use_proof) {
    return "missing_last_use_proof";
  }
  if (!provenance.internal_non_escaping) {
    return "missing_non_escape_proof";
  }
  return "ordering_required_unproven";
}

struct CpuTimelineSummary final {
  uint64_t count{0u};
  uint64_t submitted{0u};
  uint64_t total_us{0u};
  uint64_t max_us{0u};
};

std::mutex& cpu_timeline_summary_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::string, CpuTimelineSummary>& cpu_timeline_summaries() {
  static std::unordered_map<std::string, CpuTimelineSummary> summaries;
  return summaries;
}

std::string extract_cpu_timeline_token(
    const std::string& line,
    const char* key) {
  const std::string prefix = std::string(key) + "=";
  const size_t begin = line.find(prefix);
  if (begin == std::string::npos) {
    return {};
  }
  const size_t value_begin = begin + prefix.size();
  const size_t value_end = line.find(' ', value_begin);
  return line.substr(
      value_begin,
      value_end == std::string::npos ? std::string::npos
                                     : value_end - value_begin);
}

uint64_t extract_cpu_timeline_u64(
    const std::string& line,
    const char* key) {
  const std::string token = extract_cpu_timeline_token(line, key);
  if (token.empty()) {
    return 0u;
  }
  try {
    return static_cast<uint64_t>(std::stoull(token));
  } catch (...) {
    return 0u;
  }
}

std::string cpu_timeline_summary_key(const std::string& line) {
  const std::string event = extract_cpu_timeline_token(line, "event");
  if (event.empty()) {
    return {};
  }

  std::ostringstream key;
  key << "event=" << event;

  const std::string kernel = extract_cpu_timeline_token(line, "kernel");
  if (!kernel.empty()) {
    key << " kernel=" << kernel;
  }

  const std::string storage = extract_cpu_timeline_token(line, "storage");
  if (!storage.empty()) {
    key << " storage=" << storage;
  }

  const std::string direct = extract_cpu_timeline_token(line, "direct_buffer");
  if (!direct.empty()) {
    key << " direct_buffer=" << direct;
  }

  const std::string sizes = extract_cpu_timeline_token(line, "sizes");
  if (!sizes.empty()) {
    key << " sizes=" << sizes;
  }

  const std::string copy_range = extract_cpu_timeline_token(line, "copy_range");
  if (!copy_range.empty()) {
    key << " copy_range=" << copy_range;
  }

  const std::string active_cmd = extract_cpu_timeline_token(line, "active_cmd");
  if (!active_cmd.empty()) {
    key << " active_cmd=" << active_cmd;
  }

  const std::string full_pool_flush =
      extract_cpu_timeline_token(line, "full_pool_flush");
  if (!full_pool_flush.empty()) {
    key << " full_pool_flush=" << full_pool_flush;
  }

  const std::string fence = extract_cpu_timeline_token(line, "fence");
  if (!fence.empty()) {
    key << " fence=" << fence;
  }

  const std::string final_use = extract_cpu_timeline_token(line, "final_use");
  if (!final_use.empty()) {
    key << " final_use=" << final_use;
  }

  return key.str();
}

void record_cpu_timeline_summary_line(const std::string& line) {
  if (!cpu_timeline_summary_logging_enabled()) {
    return;
  }
  const std::string key = cpu_timeline_summary_key(line);
  if (key.empty()) {
    return;
  }
  uint64_t duration_us = extract_cpu_timeline_u64(line, "duration_us");
  if (duration_us == 0u) {
    duration_us = extract_cpu_timeline_u64(line, "record_us");
  }
  const bool submitted = extract_cpu_timeline_token(line, "submitted") == "1";

  std::lock_guard<std::mutex> lock(cpu_timeline_summary_mutex());
  CpuTimelineSummary& summary = cpu_timeline_summaries()[key];
  summary.count++;
  summary.submitted += submitted ? 1u : 0u;
  summary.total_us += duration_us;
  summary.max_us = std::max(summary.max_us, duration_us);
}

void dump_cpu_timeline_summary_log_impl() {
  if (!cpu_timeline_summary_logging_enabled()) {
    return;
  }

  std::vector<std::pair<std::string, CpuTimelineSummary>> entries;
  {
    std::lock_guard<std::mutex> lock(cpu_timeline_summary_mutex());
    entries.assign(
        cpu_timeline_summaries().begin(), cpu_timeline_summaries().end());
    cpu_timeline_summaries().clear();
  }

  std::sort(
      entries.begin(),
      entries.end(),
      [](const auto& lhs, const auto& rhs) {
        return lhs.second.total_us > rhs.second.total_us;
      });

  std::ofstream out(cpu_timeline_summary_log_path(), std::ios::app);
  out << "cpu_timeline_summary begin entries=" << entries.size() << '\n';
  for (const auto& entry : entries) {
    const CpuTimelineSummary& summary = entry.second;
    const uint64_t avg_us =
        summary.count == 0u ? 0u : summary.total_us / summary.count;
    out << entry.first
        << " count=" << summary.count
        << " submitted=" << summary.submitted
        << " total_us=" << summary.total_us
        << " avg_us=" << avg_us
        << " max_us=" << summary.max_us << '\n';
  }
  out << "cpu_timeline_summary end\n";
}

std::string format_sync_bytes(const uint64_t bytes) {
  std::ostringstream stream;
  const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
  stream.setf(std::ios::fixed);
  stream.precision(2);
  stream << mib << " MiB";
  return stream.str();
}

bool should_defer_tiny_old_path_retire_drain(
    const PendingWorkRetireDrainPolicy policy,
    const uint64_t pending_resource_count,
    const uint64_t pending_bytes) {
  constexpr uint64_t kTinyPendingRetireResourceCountLimit = 32u;
  constexpr uint64_t kTinyPendingRetireBytesLimit = 4u * 1024u;
  return policy == PendingWorkRetireDrainPolicy::DeferTinyOldPathPending &&
      pending_resource_count > 0u &&
      pending_resource_count <= kTinyPendingRetireResourceCountLimit &&
      pending_bytes <= kTinyPendingRetireBytesLimit;
}

void append_sync_log_line(const std::string& line) {
  if (!sync_logging_enabled()) {
    return;
  }

  std::ofstream out(sync_log_path(), std::ios::app);
  out << line << '\n';
}

std::string format_gpu_profile_extent(const VkExtent3D extent) {
  std::ostringstream stream;
  stream << extent.width << "x" << extent.height << "x" << extent.depth;
  return stream.str();
}

void append_gpu_timestamp_log_line(const std::string& line) {
  if (!gpu_timestamp_logging_enabled()) {
    return;
  }

  std::ofstream out(gpu_timestamp_log_path(), std::ios::app);
  out << line << '\n';
}

struct ExternalCommandRecordingState final {
  api::CommandBuffer* cmd{nullptr};
  std::vector<VulkanBuffer> buffers_to_keep_alive;
  std::vector<VulkanImage> images_to_keep_alive;
};

thread_local ExternalCommandRecordingState g_external_command_recording_state{};
thread_local c10::DeviceIndex g_current_device_index = -1;

ContextConfig default_context_config() {
  const uint32_t submit_frequency = 16u;

  const CommandPoolConfig cmd_config{
      32u, // cmdPoolInitialSize
      8u, // cmdPoolBatchSize
  };

  const DescriptorPoolConfig descriptor_pool_config{
      VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorPoolMaxSets
      VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorUniformBufferCount
      VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorStorageBufferCount
      VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorCombinedSamplerCount
      VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorStorageImageCount
      32u, // descriptorPileSizes
  };

  const QueryPoolConfig query_pool_config{
      VULKAN_QUERY_POOL_SIZE, // maxQueryCount
      256u, // initialReserveSize
  };

  return ContextConfig{
      submit_frequency, // cmdSubmitFrequency
      cmd_config, // cmdPoolConfig
      descriptor_pool_config, // descriptorPoolConfig
      query_pool_config, // queryPoolConfig
  };
}

void validate_device_index(c10::DeviceIndex device_index) {
  const uint32_t count = runtime()->device_count();
  VK_CHECK_COND(
      device_index >= 0,
      "Pytorch Vulkan Context: Device index must be non-negative!");
  VK_CHECK_COND(
      static_cast<uint32_t>(device_index) < count,
      "Pytorch Vulkan Context: Device index ",
      device_index,
      " is out of range for ",
      count,
      " Vulkan devices.");
}

} // namespace

void dump_cpu_timeline_summary_log() {
  dump_cpu_timeline_summary_log_impl();
}

bool cpu_timeline_logging_enabled() {
  return cpu_timeline_line_logging_enabled() ||
      cpu_timeline_summary_logging_enabled();
}

uint64_t cpu_timeline_now_us() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void append_cpu_timeline_log_line(const std::string& line) {
  record_cpu_timeline_summary_line(line);

  if (cpu_timeline_line_logging_enabled()) {
    std::ofstream out(cpu_timeline_log_path(), std::ios::app);
    out << line << '\n';
  }
}

Context::Context(c10::DeviceIndex device_index, const ContextConfig& config)
    : config_(config),
      // Important handles
      device_index_(device_index),
      adapter_p_(runtime()->get_adapter_p_for_device(device_index_)),
      device_(adapter_p_->device_handle()),
      queue_(adapter_p_->request_queue()),
      // Resource pools
      command_pool_(device_, queue_.family_index, config_.cmdPoolConfig),
      descriptor_pool_(device_, config_.descriptorPoolConfig),
      persistent_command_pool_(
          device_,
          queue_.family_index,
          config_.cmdPoolConfig),
      persistent_descriptor_pool_(device_, config_.descriptorPoolConfig),
      fences_(device_),
      querypool_(config_.queryPoolConfig, adapter_p_),
      // Command buffer submission
      cmd_mutex_{},
      cmd_(VK_NULL_HANDLE, 0u, nullptr),
      submit_count_{0u},
      command_buffer_recording_id_{0u},
      next_command_buffer_recording_id_{1u},
      stack_planned_recording_active_{false},
      stack_planned_recording_owner_{},
      stack_planned_recording_stats_{},
      // Memory Management
      pending_retire_buffers_mutex_{},
      pending_retire_buffers_{},
      pending_retire_images_mutex_{},
      pending_retire_images_{},
      pending_retire_bytes_{0u},
      stack_internal_temp_retire_batch_mutex_{},
      stack_internal_temp_retire_batch_buffers_{},
      stack_internal_temp_retire_batch_images_{},
      retire_queue_{},
      last_submission_{} {
  enable_op_profiling_ =
      gpu_timestamp_logging_enabled() && querypool_.is_enabled();
  if (gpu_timestamp_logging_enabled()) {
    std::ostringstream stream;
    stream << "gpu_timestamp_status enabled="
           << (enable_op_profiling_ ? "1" : "0")
           << " querypool=" << (querypool_.is_enabled() ? "1" : "0")
           << " timestamp_compute_and_graphics="
           << (adapter_p_->timestamp_compute_and_graphics() ? "1" : "0")
           << " ns_per_tick=" << querypool_.ns_per_tick_;
    append_gpu_timestamp_log_line(stream.str());
  }
}

Context::~Context() {
  try {
    flush();
    dump_cpu_timeline_summary_log();
    // Let the device know the context is done with the queue
    adapter_p_->return_queue(queue_);
  } catch (...) {
  }
}

Context::ScopedExternalCommandRecording::ScopedExternalCommandRecording(
    Context& context,
    CommandBuffer& cmd)
    : context_(&context) {
  context_->begin_external_command_recording(cmd);
}

Context::ScopedExternalCommandRecording::~ScopedExternalCommandRecording() {
  if (context_) {
    context_->end_external_command_recording();
  }
}

CommandBuffer* Context::external_recording_cmd() {
  return g_external_command_recording_state.cmd;
}

const CommandBuffer* Context::external_recording_cmd() const {
  return g_external_command_recording_state.cmd;
}

bool Context::is_inside_owned_program_recording() const {
  return external_recording_cmd() != nullptr;
}

bool Context::is_stack_planned_recording_active() const {
  return stack_planned_recording_active_.load(std::memory_order_acquire);
}

bool Context::stack_planned_recording_owned_by_current_thread() const {
  return is_stack_planned_recording_active() &&
      stack_planned_recording_owner_ == std::this_thread::get_id();
}

DescriptorPool& Context::active_descriptor_pool() {
  return external_recording_cmd() ? persistent_descriptor_pool_ : descriptor_pool_;
}

CommandBuffer& Context::active_cmd() {
  if (CommandBuffer* const external_cmd = external_recording_cmd()) {
    return *external_cmd;
  }
  return cmd_;
}

void Context::begin_external_command_recording(CommandBuffer& cmd) {
  VK_CHECK_COND(
      g_external_command_recording_state.cmd == nullptr,
      "Vulkan external command recording is already active");
  g_external_command_recording_state.cmd = &cmd;
  g_external_command_recording_state.buffers_to_keep_alive.clear();
  g_external_command_recording_state.images_to_keep_alive.clear();
}

void Context::end_external_command_recording() {
  VK_CHECK_COND(
      g_external_command_recording_state.cmd != nullptr,
      "Vulkan external command recording is not active");
  g_external_command_recording_state.cmd = nullptr;
}

void Context::capture_external_recording_buffer_cleanup(VulkanBuffer&& buffer) {
  g_external_command_recording_state.buffers_to_keep_alive.emplace_back(
      std::move(buffer));
}

void Context::capture_external_recording_image_cleanup(VulkanImage&& image) {
  g_external_command_recording_state.images_to_keep_alive.emplace_back(
      std::move(image));
}

uint32_t Context::gpu_profile_begin(
    CommandBuffer& cmd,
    const std::string& label,
    const VkExtent3D global_workgroup_size,
    const VkExtent3D local_workgroup_size) {
  if (!enable_op_profiling_ || !querypool_.is_enabled()) {
    return UINT32_MAX;
  }
  return querypool_.shader_profile_begin(
      cmd, label, global_workgroup_size, local_workgroup_size);
}

void Context::gpu_profile_end(CommandBuffer& cmd, const uint32_t log_idx) {
  if (!enable_op_profiling_ || !querypool_.is_enabled() ||
      log_idx == UINT32_MAX) {
    return;
  }
  querypool_.shader_profile_end(cmd, log_idx);
}

uint32_t Context::begin_external_gpu_profile(
    const std::string& label,
    const VkExtent3D global_workgroup_size,
    const VkExtent3D local_workgroup_size) {
  if (!enable_op_profiling_ || !querypool_.is_enabled()) {
    return UINT32_MAX;
  }
  CommandBuffer* const cmd = external_recording_cmd();
  if (cmd == nullptr) {
    return UINT32_MAX;
  }
  return gpu_profile_begin(
      *cmd, label, global_workgroup_size, local_workgroup_size);
}

void Context::end_external_gpu_profile(const uint32_t log_idx) {
  if (!enable_op_profiling_ || !querypool_.is_enabled() ||
      log_idx == UINT32_MAX) {
    return;
  }
  CommandBuffer* const cmd = external_recording_cmd();
  if (cmd == nullptr) {
    return;
  }
  gpu_profile_end(*cmd, log_idx);
}

void Context::reset_gpu_profile_queries() {
  if (!enable_op_profiling_ || !querypool_.is_enabled() ||
      !querypool_.has_entries()) {
    return;
  }
  CommandBuffer reset_cmd = command_pool_.get_new_cmd(/*reusable=*/false);
  reset_cmd.begin();
  querypool_.clear_after_reset(reset_cmd);
  reset_cmd.end();
  adapter_p_->submit_cmd(
      queue_, reset_cmd.get_submit_handle(/*final_use=*/true));
  note_vulkan_queue_submit(VulkanSubmitOrigin::ProfilingTimestampReset);
  note_vulkan_queue_wait_idle();
  VK_CHECK(vkQueueWaitIdle(queue()));
}

void Context::dump_gpu_profile_log(const char* reason) {
  if (!enable_op_profiling_ || !gpu_timestamp_logging_enabled() ||
      !querypool_.is_enabled() || !querypool_.has_pending_results()) {
    return;
  }

  querypool_.extract_results();
  querypool_.shader_log_for_each([reason](const ShaderDuration& entry) {
    if (entry.end_query_idx == UINT32_MAX) {
      return;
    }
    std::ostringstream stream;
    stream << "gpu_timestamp reason=" << (reason ? reason : "unspecified")
           << " name=" << entry.kernel_name
           << " runtime=" << entry.runtime_label
           << " start_ns=" << entry.start_time_ns
           << " end_ns=" << entry.end_time_ns
           << " duration_ns=" << entry.execution_duration_ns
           << " global=" << format_gpu_profile_extent(entry.global_workgroup_size)
           << " local=" << format_gpu_profile_extent(entry.local_workgroup_size);
    append_gpu_timestamp_log_line(stream.str());
  });
  reset_gpu_profile_queries();
}

DescriptorSet Context::get_descriptor_set(
    const ShaderInfo& shader_descriptor,
    const utils::uvec3& local_workgroup_size) {
  VkDescriptorSetLayout shader_layout =
      shader_layout_cache().retrieve(shader_descriptor.kernel_layout);

  VkPipelineLayout pipeline_layout =
      pipeline_layout_cache().retrieve(shader_layout);

  VkPipeline pipeline = pipeline_cache().retrieve(
      {pipeline_layout,
       shader_cache().retrieve(shader_descriptor),
       local_workgroup_size,
       shader_descriptor.required_subgroup_size,
       shader_descriptor.require_full_subgroups});

  active_cmd().bind_pipeline(pipeline, pipeline_layout, local_workgroup_size);

  return active_descriptor_pool().get_descriptor_set(
      shader_layout, shader_descriptor.kernel_layout);
}

void Context::register_shader_dispatch(
    const DescriptorSet& descriptors,
    PipelineBarrier& pipeline_barrier,
    const ShaderInfo& shader_descriptor,
    const utils::uvec3& global_workgroup_size) {
  // Adjust the global workgroup size based on the output tile size
  const utils::uvec3 effective_global_wg = {
      utils::div_up(
          global_workgroup_size.data[0u],
          shader_descriptor.out_tile_size.data[0u]),
      utils::div_up(
          global_workgroup_size.data[1u],
          shader_descriptor.out_tile_size.data[1u]),
      utils::div_up(
          global_workgroup_size.data[2u],
          shader_descriptor.out_tile_size.data[2u]),
  };

  CommandBuffer& cmd = active_cmd();
  cmd.bind_descriptors(descriptors.get_bind_handle());
  cmd.insert_barrier(pipeline_barrier);

  cmd.dispatch(effective_global_wg);
}

VulkanStreamState& Context::current_stream() {
  return vulkan_stream_pool().get_current_stream(device_index_);
}

c10::Stream Context::current_c10_stream() {
  return vulkan_stream_pool().get_current_c10_stream(device_index_);
}

bool Context::has_pending_work_for_current_stream() const {
  return submit_count_ > 0u;
}

void Context::flush_if_current_stream(const c10::Stream& stream) {
  VK_CHECK_COND(
      stream.device_type() == c10::DeviceType::Vulkan,
      "Expected a Vulkan stream, got ",
      stream.device());
  VK_CHECK_COND(
      stream.device_index() == device_index_,
      "Cannot flush a Vulkan stream for device ",
      stream.device_index(),
      " on context for device ",
      device_index_);
  std::unique_lock<std::mutex> context_lock(dispatch_lock());
  // Version-one invariant: a Context owns one active command buffer, and
  // exchange_stream() flushes before switching streams. Therefore unsubmitted
  // work can only belong to current_c10_stream().
  if (stream != current_c10_stream() || !has_pending_work_for_current_stream()) {
    return;
  }
  submit_cmd_to_gpu(
      VK_NULL_HANDLE, false, VulkanSubmitOrigin::ExplicitSynchronize);
}

c10::Stream Context::exchange_stream(c10::Stream stream) {
  VK_CHECK_COND(
      stream.device_type() == c10::DeviceType::Vulkan,
      "Expected a Vulkan stream, got ",
      stream.device());
  VK_CHECK_COND(
      stream.device_index() == device_index_,
      "Cannot set a Vulkan stream for device ",
      stream.device_index(),
      " on context for device ",
      device_index_);
  std::unique_lock<std::mutex> context_lock(dispatch_lock());
  submit_cmd_to_gpu(
      VK_NULL_HANDLE, false, VulkanSubmitOrigin::ExplicitSynchronize);
  const c10::Stream previous = current_c10_stream();
  vulkan_stream_pool().set_current_stream(stream);
  return previous;
}

bool Context::query_stream(const c10::Stream& stream) {
  if (stream == current_c10_stream() && submit_count_ > 0u) {
    return false;
  }
  VulkanStreamState& vk_stream = vulkan_stream_pool().unwrap(stream);
  const uint64_t value =
      vk_stream.last_submitted_value.load(std::memory_order_acquire);
  return vulkan_stream_pool().query_complete(vk_stream, value);
}

void Context::synchronize_stream(const c10::Stream& stream) {
  std::unique_lock<std::mutex> context_lock(dispatch_lock());
  if (stream == current_c10_stream()) {
    submit_cmd_to_gpu(
        VK_NULL_HANDLE, false, VulkanSubmitOrigin::ExplicitSynchronize);
  }
  context_lock.unlock();
  VulkanStreamState& vk_stream = vulkan_stream_pool().unwrap(stream);
  const uint64_t value =
      vk_stream.last_submitted_value.load(std::memory_order_acquire);
  vulkan_stream_pool().wait_complete(vk_stream, value);
  poll_retire_queue();
}

void Context::synchronize_device() {
  {
    std::unique_lock<std::mutex> context_lock(dispatch_lock());
    submit_cmd_to_gpu(
        /*fence_handle=*/VK_NULL_HANDLE,
        /*final_use=*/true,
        VulkanSubmitOrigin::ExplicitSynchronize);
  }
  vulkan_stream_pool().wait_all(device_index_);
  retire_queue_.drain(device_);
  command_pool_.flush();
  descriptor_pool_.flush();
  if (cmd_) {
    cmd_.invalidate();
  }
  submit_count_ = 0u;
  command_buffer_recording_id_ = 0u;
  clear_pending_retire_resources_locked();
}

std::string Context::format_submit_failure_diagnostics(
    const VulkanStreamState& stream_state,
    const VulkanSubmitOrigin origin,
    const uint64_t signal_value,
    const size_t wait_count,
    const VkFence fence_handle,
    const bool final_use) {
  std::vector<std::string> pending_samples;
  pending_samples.reserve(4u);
  uint64_t pending_buffer_count = 0u;
  uint64_t pending_image_count = 0u;

  const auto append_stack_provenance =
      [](std::ostringstream& out,
         const VulkanStackRetireProvenance& provenance) {
        if (!provenance.defined) {
          return;
        }
        out << " stack_phase=" << vision_stack_phase_name(provenance.phase)
            << " block=" << provenance.block_index
            << " proof=" << (provenance.has_last_use_proof ? 1 : 0)
            << " escapes=" << (provenance.escapes_stack ? 1 : 0)
            << " requested_intermediate="
            << (provenance.requested_intermediate ? 1 : 0);
      };

  {
    std::lock_guard<std::mutex> lock(pending_retire_buffers_mutex_);
    pending_buffer_count = pending_retire_buffers_.size();
    for (const PendingRetireBuffer& pending : pending_retire_buffers_) {
      if (pending_samples.size() >= 4u) {
        break;
      }
      std::ostringstream sample;
      sample << "buffer{kind=" << retired_resource_kind_name(pending.kind)
             << " role=" << retired_resource_role_name(pending.role)
             << " phase=" << submit_phase_name(pending.phase)
             << " callsite=" << retire_call_site_name(pending.callsite)
             << " bytes=" << pending.bytes
             << " label=" << pending.buffer.allocation_label();
      append_stack_provenance(sample, pending.stack_provenance);
      sample << "}";
      pending_samples.emplace_back(sample.str());
    }
  }
  {
    std::lock_guard<std::mutex> lock(pending_retire_images_mutex_);
    pending_image_count = pending_retire_images_.size();
    for (const PendingRetireImage& pending : pending_retire_images_) {
      if (pending_samples.size() >= 4u) {
        break;
      }
      std::ostringstream sample;
      sample << "image{kind=" << retired_resource_kind_name(pending.kind)
             << " role=" << retired_resource_role_name(pending.role)
             << " phase=" << submit_phase_name(pending.phase)
             << " callsite=" << retire_call_site_name(pending.callsite)
             << " bytes=" << pending.bytes
             << " label=" << pending.image.allocation_label();
      append_stack_provenance(sample, pending.stack_provenance);
      sample << "}";
      pending_samples.emplace_back(sample.str());
    }
  }

  std::ostringstream out;
  out << " submit_breadcrumbs origin=" << submit_origin_name(origin)
      << " caller=" << current_allocation_label()
      << " stream_id=" << stream_state.id
      << " stream_device=" << stream_state.device_index
      << " queue_family=" << stream_state.queue.family_index
      << " queue_index=" << stream_state.queue.queue_index
      << " signal_value=" << signal_value
      << " last_submitted_value="
      << stream_state.last_submitted_value.load(std::memory_order_relaxed)
      << " wait_count=" << wait_count
      << " fence=" << (fence_handle != VK_NULL_HANDLE ? 1 : 0)
      << " final_use=" << (final_use ? 1 : 0)
      << " pending_retire_count="
      << (pending_buffer_count + pending_image_count)
      << " pending_retire_buffers=" << pending_buffer_count
      << " pending_retire_images=" << pending_image_count
      << " pending_retire_bytes=" << pending_retire_bytes();
  if (!current_runtime_label().empty()) {
    out << " runtime_label=" << current_runtime_label();
  }
  if (!recent_op_label().empty()) {
    out << " recent_op=" << recent_op_label();
  }
  if (!pending_samples.empty()) {
    out << " pending_retire_samples=[";
    for (size_t i = 0; i < pending_samples.size(); ++i) {
      if (i > 0u) {
        out << ";";
      }
      out << pending_samples[i];
    }
    out << "]";
  }
  return out.str();
}

VulkanSubmission Context::submit_cmd_handle_to_gpu(
    VulkanStreamState& stream,
    VkCommandBuffer cmd,
    VulkanSubmitOrigin origin,
    VkFence fence_handle,
    const bool final_use) {
  std::vector<VkSemaphore> wait_semaphores;
  std::vector<uint64_t> wait_values;
  std::vector<VkPipelineStageFlags> wait_stages;
  {
    std::lock_guard<std::mutex> lock(stream.mutex);
    wait_semaphores.reserve(stream.pending_waits.size());
    wait_values.reserve(stream.pending_waits.size());
    wait_stages.reserve(stream.pending_waits.size());
    for (const auto& wait : stream.pending_waits) {
      wait_semaphores.push_back(wait.semaphore);
      wait_values.push_back(wait.value);
      wait_stages.push_back(wait.wait_stage);
    }
    stream.pending_waits.clear();
  }

  const uint64_t signal_value = stream.reserve_signal_value();
  VK_CHECK_COND(
      stream.queue.family_index == queue_.family_index,
      "Vulkan stream queue family does not match command buffer queue family");
  try {
    adapter_p_->submit_cmd_timeline(
        stream.queue,
        cmd,
        wait_semaphores,
        wait_values,
        wait_stages,
        stream.timeline,
        signal_value,
        fence_handle);
  } catch (const Error& error) {
    VK_THROW(
        error.msg(),
        format_submit_failure_diagnostics(
            stream,
            origin,
            signal_value,
            wait_values.size(),
            fence_handle,
            final_use));
  }
  note_vulkan_queue_submit(origin);
  vulkan_sync_counters().stream_submit_count.fetch_add(
      1u, std::memory_order_relaxed);
  return VulkanSubmission{stream.id, stream.timeline, signal_value};
}

void Context::retire_deferred_cleanup(
    VulkanSubmission submission,
    VulkanSubmitOrigin origin) {
  if (submission.timeline == VK_NULL_HANDLE || submission.timeline_value == 0u) {
    clear_pending_retire_resources_locked();
    return;
  }
  if (origin == VulkanSubmitOrigin::ConvPrepackUpload) {
    uint64_t pending_resource_count = 0u;
    {
      std::lock_guard<std::mutex> bufferlist_lock(
          pending_retire_buffers_mutex_);
      pending_resource_count += pending_retire_buffers_.size();
    }
    {
      std::lock_guard<std::mutex> imagelist_lock(pending_retire_images_mutex_);
      pending_resource_count += pending_retire_images_.size();
    }
    if (should_defer_tiny_old_path_retire_drain(
            PendingWorkRetireDrainPolicy::DeferTinyOldPathPending,
            pending_resource_count,
            pending_retire_bytes())) {
      return;
    }
  }
  const VulkanRetireCallSite callsite =
      origin == VulkanSubmitOrigin::StackPlannedRecordingSubmit
      ? VulkanRetireCallSite::StackPlannedRecordingEnd
      : retire_call_site_for_current_phase();
  {
    std::lock_guard<std::mutex> bufferlist_lock(
        pending_retire_buffers_mutex_);
    for (PendingRetireBuffer& pending : pending_retire_buffers_) {
      note_vulkan_retired_resource(
          pending.kind,
          pending.role,
          pending.phase,
          pending.callsite == VulkanRetireCallSite::Unknown ? callsite
                                                            : pending.callsite,
          pending.bytes,
          /*queue_submit=*/true,
          /*blocking_wait=*/false,
          /*poll_only=*/false,
          pending.stack_provenance);
      retire_queue_.retire(RetiredResource{
          submission.stream_id,
          submission.timeline,
          submission.timeline_value,
          [buffer = std::make_shared<VulkanBuffer>(
               std::move(pending.buffer))]() mutable {
            buffer.reset();
          },
      });
    }
    pending_retire_buffers_.clear();
  }
  {
    std::lock_guard<std::mutex> imagelist_lock(pending_retire_images_mutex_);
    for (PendingRetireImage& pending : pending_retire_images_) {
      note_vulkan_retired_resource(
          pending.kind,
          pending.role,
          pending.phase,
          pending.callsite == VulkanRetireCallSite::Unknown ? callsite
                                                            : pending.callsite,
          pending.bytes,
          /*queue_submit=*/true,
          /*blocking_wait=*/false,
          /*poll_only=*/false,
          pending.stack_provenance);
      retire_queue_.retire(RetiredResource{
          submission.stream_id,
          submission.timeline,
          submission.timeline_value,
          [image = std::make_shared<VulkanImage>(
               std::move(pending.image))]() mutable {
            image.reset();
          },
      });
    }
    pending_retire_images_.clear();
  }
  pending_retire_bytes_.store(0u, std::memory_order_relaxed);
}

void Context::poll_retire_queue() {
  retire_queue_.poll(device_);
}

void Context::submit_pending_work_and_poll_retire(
    const PendingWorkRetireDrainPolicy policy) {
  const uint64_t pending_bytes = pending_retire_bytes();
  uint64_t pending_resource_count = 0u;
  uint64_t qkv_hypothetical_count = 0u;
  uint64_t qkv_hypothetical_bytes = 0u;
  bool blocked_requested_intermediate = false;
  bool blocked_missing_proof = false;
  bool blocked_generic_stack_internal_temp = false;
  bool blocked_metadata_or_uniform = false;
  bool blocked_other_roles = false;
  std::map<std::string, std::pair<uint64_t, uint64_t>> copresent_resources;
  std::set<std::string> copresent_blockers;
  std::vector<RegionLifetimeSubmitResourceAttribution>
      region_lifetime_resource_attributions;
  const VulkanRetireCallSite callsite = retire_call_site_for_current_phase();
  const VulkanSubmitPhase phase = current_submit_phase();
  const bool record_subresource_lifetime_dry_run =
      phase == VulkanSubmitPhase::StackOwner &&
      callsite == VulkanRetireCallSite::StackOwnerNorm2;
  uint64_t dry_run_safe_candidate_count = 0u;
  uint64_t dry_run_safe_candidate_bytes = 0u;
  bool dry_run_has_large_backing = false;
  std::map<std::string, std::pair<uint64_t, uint64_t>>
      dry_run_resource_classes;
  std::set<std::string> dry_run_blockers;
  std::string dry_run_budget_reject = "not_stack_owner_norm2";
  bool dry_run_all_safe_group_eligible = false;
  std::string dry_run_signature;
  std::string dry_run_blocker_signature;
  const auto inspect_pending_resource =
      [&](const auto& pending) {
        const bool qkv_would_batch =
            is_qkv_stack_temp_retire_batch_candidate(pending.stack_provenance);
        const VulkanRetireCallSite effective_callsite =
            pending.callsite == VulkanRetireCallSite::Unknown ? callsite
                                                              : pending.callsite;
        const char* const blocker_reason =
            stack_retire_drain_blocker_reason(
                pending.kind,
                pending.role,
                pending.stack_provenance,
                qkv_would_batch);
        const VulkanStackTempLifetimeSafety safety =
            stack_retire_lifetime_safety_for_resource(
                pending.role, pending.stack_provenance);
        const VulkanStackRawResourceAllocationProof allocation_proof =
            stack_raw_allocation_proof(pending);
        const std::string& allocation_label =
            pending_retire_allocation_label(pending);
        std::ostringstream resource_key;
        resource_key << retired_resource_role_name(pending.role) << ":"
                     << blocker_reason << ":"
                     << stack_temp_lifetime_safety_name(safety);
        auto& resource_value = copresent_resources[resource_key.str()];
        resource_value.first += 1u;
        resource_value.second += pending.bytes;
        copresent_blockers.insert(blocker_reason);
        region_lifetime_resource_attributions.emplace_back(
            make_region_lifetime_submit_resource_attribution(
                pending, phase, effective_callsite));
        if (qkv_would_batch) {
          qkv_hypothetical_count++;
          qkv_hypothetical_bytes += pending.bytes;
        } else if (
            pending.stack_provenance.defined &&
            (pending.stack_provenance.requested_intermediate ||
             pending.stack_provenance.escapes_stack)) {
          blocked_requested_intermediate = true;
        } else if (
            pending.stack_provenance.defined &&
            is_stack_temp_retired_resource_role(pending.role) &&
            !pending.stack_provenance.has_last_use_proof) {
          blocked_missing_proof = true;
        } else if (pending.role == VulkanRetiredResourceRole::StackInternalTemp) {
          blocked_generic_stack_internal_temp = true;
        } else {
          switch (pending.role) {
            case VulkanRetiredResourceRole::NativeLayerNormUniform:
            case VulkanRetiredResourceRole::NativeLayerNormMetadata:
            case VulkanRetiredResourceRole::AttentionMetadata:
            case VulkanRetiredResourceRole::LinearMetadata:
            case VulkanRetiredResourceRole::ConvMetadata:
            case VulkanRetiredResourceRole::ResidualAddMetadata:
              blocked_metadata_or_uniform = true;
              break;
            default:
              if (
                  pending.kind == VulkanRetiredResourceKind::UniformBuffer ||
                  pending.kind == VulkanRetiredResourceKind::MetadataBuffer) {
                blocked_metadata_or_uniform = true;
              } else {
                blocked_other_roles = true;
              }
              break;
          }
        }
        note_stack_retire_drain_blocker_resource(
            pending.kind,
            pending.role,
            pending.phase,
            effective_callsite,
            pending.bytes,
            qkv_would_batch,
            pending.stack_provenance,
            allocation_proof,
            allocation_label);
        if (record_subresource_lifetime_dry_run) {
          const char* const resource_class =
              stack_subresource_lifetime_dry_run_resource_class(
                  pending.kind,
                  pending.role,
                  pending.stack_provenance,
                  qkv_would_batch,
                  allocation_proof);
          const bool formal_last_use_proof =
              stack_subresource_lifetime_dry_run_has_formal_norm2_last_use_proof(
                  pending.kind,
                  pending.role,
                  resource_class,
                  pending.stack_provenance,
                  allocation_proof,
                  allocation_label);
          const bool safe_candidate =
              stack_subresource_lifetime_dry_run_resource_is_safe(
                  resource_class) ||
              formal_last_use_proof;
          const bool large_backing =
              stack_subresource_lifetime_dry_run_is_large_backing(
                  pending.role, pending.bytes, pending.stack_provenance);
          auto& class_value = dry_run_resource_classes[resource_class];
          class_value.first += 1u;
          class_value.second += pending.bytes;
          if (safe_candidate && !large_backing) {
            dry_run_safe_candidate_count++;
            dry_run_safe_candidate_bytes += pending.bytes;
          } else {
            dry_run_blockers.insert(resource_class);
          }
          if (large_backing) {
            dry_run_has_large_backing = true;
            dry_run_blockers.insert("large_backing_excluded");
          }
          note_stack_subresource_lifetime_dry_run_resource(
              pending.kind,
              pending.role,
              pending.phase,
              effective_callsite,
              pending.bytes,
              resource_class,
              safe_candidate,
              large_backing,
              formal_last_use_proof,
              pending.stack_provenance,
              allocation_proof,
              allocation_label);
        }
      };
  {
    std::lock_guard<std::mutex> bufferlist_lock(pending_retire_buffers_mutex_);
    pending_resource_count += pending_retire_buffers_.size();
    for (const PendingRetireBuffer& pending : pending_retire_buffers_) {
      inspect_pending_resource(pending);
    }
  }
  {
    std::lock_guard<std::mutex> imagelist_lock(pending_retire_images_mutex_);
    pending_resource_count += pending_retire_images_.size();
    for (const PendingRetireImage& pending : pending_retire_images_) {
      inspect_pending_resource(pending);
    }
  }
  if (record_subresource_lifetime_dry_run) {
    std::ostringstream dry_run_signature_stream;
    for (const auto& entry : dry_run_resource_classes) {
      if (dry_run_signature_stream.tellp() > 0) {
        dry_run_signature_stream << ",";
      }
      dry_run_signature_stream << entry.first << "#" << entry.second.first
                               << "#" << entry.second.second;
    }
    std::ostringstream dry_run_blocker_signature_stream;
    for (const auto& blocker : dry_run_blockers) {
      if (dry_run_blocker_signature_stream.tellp() > 0) {
        dry_run_blocker_signature_stream << ",";
      }
      dry_run_blocker_signature_stream << blocker;
    }
    dry_run_signature = dry_run_signature_stream.str();
    dry_run_blocker_signature = dry_run_blocker_signature_stream.str();
    const bool all_safe_without_budget =
        pending_resource_count > 0u &&
        pending_resource_count == dry_run_safe_candidate_count &&
        !dry_run_has_large_backing;
    if (pending_resource_count == 0u) {
      dry_run_budget_reject = "no_old_path_pending";
    } else if (dry_run_has_large_backing) {
      dry_run_budget_reject = "large_backing_excluded";
    } else if (!all_safe_without_budget) {
      dry_run_budget_reject = "unsafe_resource_class";
    } else if (
        dry_run_safe_candidate_bytes >
        kStackSubresourceLifetimeDryRunBlockBudgetBytes) {
      dry_run_budget_reject = "over_block_budget";
    } else if (
        dry_run_safe_candidate_bytes >
        kStackSubresourceLifetimeDryRunScopeBudgetBytes) {
      dry_run_budget_reject = "over_scope_budget";
    } else {
      dry_run_budget_reject = "none";
    }
    dry_run_all_safe_group_eligible = dry_run_budget_reject == "none";
  }
  bool had_pending_work = false;
  const bool has_old_path_pending_retire = pending_resource_count > 0u;
  const bool should_defer_tiny_old_path_pending =
      should_defer_tiny_old_path_retire_drain(
          policy, pending_resource_count, pending_bytes);
  const bool old_path_pending_would_submit =
      has_old_path_pending_retire && !should_defer_tiny_old_path_pending;
  const bool should_coalesce_norm2_retire_submit =
      record_subresource_lifetime_dry_run && dry_run_all_safe_group_eligible &&
      old_path_pending_would_submit;
  const bool should_submit_old_path_pending =
      old_path_pending_would_submit && !should_coalesce_norm2_retire_submit;
  {
    std::unique_lock<std::mutex> context_lock(dispatch_lock());
    had_pending_work = has_pending_work_for_current_stream();
    if (should_submit_old_path_pending) {
      submit_cmd_to_gpu(
          VK_NULL_HANDLE, false, VulkanSubmitOrigin::RetireQueueDrain);
    }
  }
  const bool skipped_no_old_path_pending = !has_old_path_pending_retire;
  const bool skipped_no_pending_command_work =
      skipped_no_old_path_pending && !had_pending_work;
  const bool submitted_for_retire_drain =
      should_submit_old_path_pending && had_pending_work;
  const bool coalesced_retire_drain_submit =
      should_coalesce_norm2_retire_submit && had_pending_work;
  const std::string region_lifetime_signature =
      format_region_lifetime_submit_signature(copresent_resources);
  const std::string region_lifetime_blockers =
      format_region_lifetime_submit_blockers(copresent_blockers);
  note_region_lifetime_submit_attribution_group(
      VulkanSubmitOrigin::RetireQueueDrain,
      phase,
      callsite,
      submitted_for_retire_drain,
      had_pending_work,
      pending_resource_count,
      pending_bytes,
      region_lifetime_signature,
      region_lifetime_blockers);
  for (const auto& attribution : region_lifetime_resource_attributions) {
    note_region_lifetime_submit_attribution_resource(
        VulkanSubmitOrigin::RetireQueueDrain,
        attribution.phase,
        attribution.callsite,
        attribution.kind,
        attribution.role,
        attribution.bytes,
        attribution.reason,
        attribution.safety,
        submitted_for_retire_drain,
        had_pending_work,
        attribution.provenance,
        attribution.allocation_proof,
        attribution.allocation_label);
  }
  note_vulkan_retire_drain(
      retire_drain_reason_for_current_phase(),
      callsite,
      submitted_for_retire_drain,
      /*blocking_wait=*/false,
      pending_resource_count,
      pending_bytes);
  note_stack_retire_drain_blocker_summary(
      phase,
      callsite,
      submitted_for_retire_drain,
      pending_resource_count,
      pending_bytes,
      qkv_hypothetical_count,
      qkv_hypothetical_bytes,
      pending_resource_count > 0u &&
          pending_resource_count == qkv_hypothetical_count,
      /*only_already_batched=*/false,
      blocked_requested_intermediate,
      blocked_missing_proof,
      blocked_generic_stack_internal_temp,
      blocked_metadata_or_uniform,
      blocked_other_roles,
      skipped_no_old_path_pending,
      skipped_no_pending_command_work);
  note_stack_retire_drain_copresent_group(
      phase,
      callsite,
      submitted_for_retire_drain,
      pending_resource_count,
      pending_bytes,
      qkv_hypothetical_count,
      pending_resource_count > 0u &&
          pending_resource_count == qkv_hypothetical_count,
      skipped_no_old_path_pending,
      region_lifetime_signature,
      region_lifetime_blockers);
  if (record_subresource_lifetime_dry_run) {
    const bool would_remove_submit_drain =
        dry_run_all_safe_group_eligible &&
        old_path_pending_would_submit && had_pending_work;
    note_stack_subresource_lifetime_dry_run_group(
        phase,
        callsite,
        submitted_for_retire_drain,
        pending_resource_count,
        pending_bytes,
        dry_run_safe_candidate_count,
        dry_run_safe_candidate_bytes,
        dry_run_all_safe_group_eligible,
        would_remove_submit_drain,
        coalesced_retire_drain_submit,
        dry_run_budget_reject,
        dry_run_signature,
        dry_run_blocker_signature);
  }
  if (!had_pending_work) {
    std::lock_guard<std::mutex> bufferlist_lock(pending_retire_buffers_mutex_);
    for (const PendingRetireBuffer& pending : pending_retire_buffers_) {
      note_vulkan_retired_resource(
          pending.kind,
          pending.role,
          pending.phase,
          pending.callsite == VulkanRetireCallSite::Unknown ? callsite
                                                            : pending.callsite,
          pending.bytes,
          had_pending_work,
          /*blocking_wait=*/false,
          !had_pending_work,
          pending.stack_provenance);
    }
  }
  if (!had_pending_work) {
    std::lock_guard<std::mutex> imagelist_lock(pending_retire_images_mutex_);
    for (const PendingRetireImage& pending : pending_retire_images_) {
      note_vulkan_retired_resource(
          pending.kind,
          pending.role,
          pending.phase,
          pending.callsite == VulkanRetireCallSite::Unknown ? callsite
                                                            : pending.callsite,
          pending.bytes,
          had_pending_work,
          /*blocking_wait=*/false,
          !had_pending_work,
          pending.stack_provenance);
    }
  }
  poll_retire_queue();
}

VulkanSubmission Context::submit_cmd_to_gpu(
    VkFence fence_handle,
    const bool final_use,
    VulkanSubmitOrigin origin) {
  const bool cpu_timeline = cpu_timeline_logging_enabled();
  const uint64_t cpu_start_us =
      cpu_timeline ? cpu_timeline_now_us() : 0u;
  const bool had_cmd = static_cast<bool>(cmd_);
  const uint64_t command_buffer_recording_id = command_buffer_recording_id_;
  const uint64_t submit_epoch_before =
      current_stream().last_submitted_value.load(std::memory_order_relaxed);
  const uint64_t submit_epoch_after = had_cmd ? submit_epoch_before + 1u
                                              : submit_epoch_before;
  const uint64_t pending_dispatch_count = submit_count_;
  constexpr bool kCoalescePhaseBoundaryExplicitSync = true;
  constexpr uint64_t kStackActivationPhaseBoundaryLifetimeBlockBudgetBytes =
      5u * 1024u * 1024u;
  constexpr uint64_t kStackActivationPhaseBoundaryLifetimeScopeBudgetBytes =
      kStackSubresourceLifetimeDryRunScopeBudgetBytes;
  if (had_cmd && origin == VulkanSubmitOrigin::ExplicitSynchronize) {
    const VulkanSubmitPhase phase = current_submit_phase();
    const VulkanRetireCallSite callsite = retire_call_site_for_current_phase();
    uint64_t pending_resource_count = 0u;
    const uint64_t pending_bytes = pending_retire_bytes();
    uint64_t dry_run_safe_candidate_count = 0u;
    uint64_t dry_run_safe_candidate_bytes = 0u;
    bool dry_run_has_large_backing = false;
    std::map<std::string, std::pair<uint64_t, uint64_t>> resources;
    std::map<std::string, std::pair<uint64_t, uint64_t>>
        dry_run_resource_classes;
    std::map<std::string, std::pair<uint64_t, uint64_t>>
        dry_run_allocation_ranges;
    std::map<std::string, std::pair<uint64_t, uint64_t>>
        dry_run_raw_provenance;
    std::set<std::string> blockers;
    std::set<std::string> dry_run_blockers;
    std::vector<RegionLifetimeSubmitResourceAttribution>
        region_lifetime_resource_attributions;
    std::string dry_run_budget_reject = "not_stack_owner_phase_boundary";
    std::string dry_run_signature;
    std::string dry_run_allocation_signature;
    std::string dry_run_raw_provenance_signature;
    std::string dry_run_blocker_signature;
    const bool record_phase_boundary_dry_run =
        phase == VulkanSubmitPhase::StackOwner &&
        (callsite == VulkanRetireCallSite::StackOwnerPhaseBoundary ||
         callsite == VulkanRetireCallSite::StackOwnerNorm1 ||
         callsite == VulkanRetireCallSite::StackOwnerNorm2);
    const auto inspect_pending_resource = [&](const auto& pending) {
      const bool qkv_would_batch =
          is_qkv_stack_temp_retire_batch_candidate(pending.stack_provenance);
      append_region_lifetime_submit_signature(
          pending, callsite, resources, blockers);
      region_lifetime_resource_attributions.emplace_back(
          make_region_lifetime_submit_resource_attribution(
              pending, phase, callsite));
      if (!record_phase_boundary_dry_run) {
        return;
      }
      const VulkanStackRawResourceAllocationProof allocation_proof =
          stack_raw_allocation_proof(pending);
      const std::string& allocation_label =
          pending_retire_allocation_label(pending);
      const char* const resource_class =
          stack_subresource_lifetime_dry_run_resource_class(
              pending.kind,
              pending.role,
              pending.stack_provenance,
              qkv_would_batch,
              allocation_proof);
      const bool formal_last_use_proof =
          stack_subresource_lifetime_dry_run_has_formal_stack_owner_last_use_proof(
              pending.kind,
              pending.role,
              resource_class,
              pending.stack_provenance,
              allocation_proof,
              allocation_label,
              callsite);
      const bool safe_candidate =
          stack_subresource_lifetime_dry_run_resource_is_safe(
              resource_class) ||
          formal_last_use_proof;
      const bool large_backing =
          stack_subresource_lifetime_dry_run_is_large_backing(
              pending.role, pending.bytes, pending.stack_provenance);
      auto& class_value = dry_run_resource_classes[resource_class];
      class_value.first += 1u;
      class_value.second += pending.bytes;
      if (allocation_proof.has_generation && allocation_proof.has_byte_range) {
        std::ostringstream allocation_key;
        allocation_key << allocation_proof.allocation_id << "#"
                       << allocation_proof.allocation_generation << "#"
                       << allocation_proof.byte_offset << "#"
                       << allocation_proof.byte_range << "#"
                       << resource_class;
        auto& allocation_value =
            dry_run_allocation_ranges[allocation_key.str()];
        allocation_value.first += 1u;
        allocation_value.second += pending.bytes;
      }
      if (stack_region_raw_provenance_diagnostic_class(resource_class)) {
        const VulkanStackRetireProvenance& provenance =
            pending.stack_provenance;
        const std::string allocation_status =
            allocation_proof.has_generation && allocation_proof.has_byte_range
            ? "generation_and_range"
            : (allocation_proof.has_generation ? "generation_only"
                                               : "missing_allocation_proof");
        std::ostringstream allocation_key;
        if (allocation_proof.has_generation && allocation_proof.has_byte_range) {
          allocation_key << allocation_proof.allocation_id << "-"
                         << allocation_proof.allocation_generation << "-"
                         << allocation_proof.byte_offset << "-"
                         << allocation_proof.byte_range;
        } else {
          allocation_key << allocation_status;
        }
        const std::string raw_status = stack_region_raw_provenance_status(
            safe_candidate,
            large_backing,
            formal_last_use_proof,
            provenance,
            allocation_proof);
        std::ostringstream raw_key;
        raw_key
            << stack_region_diagnostic_token(resource_class) << "|"
            << retired_resource_kind_name(pending.kind) << "|"
            << retired_resource_role_name(pending.role) << "|"
            << stack_region_diagnostic_token(allocation_label) << "|"
            << (provenance.defined ? vision_stack_phase_name(provenance.phase)
                                   : "missing") << "|"
            << (provenance.defined ? provenance.block_index : -1) << "|"
            << (provenance.defined
                    ? vision_stack_phase_name(provenance.expected_consumer_phase)
                    : "missing")
            << "|"
            << (provenance.defined ? provenance.expected_consumer_block_index
                                   : -1)
            << "|"
            << (provenance.defined ? "stack_provenance_present"
                                   : "stack_provenance_missing")
            << "|"
            << (provenance.has_last_use_proof ? "last_use_present"
                                              : "last_use_missing")
            << "|"
            << (provenance.internal_non_escaping ? "non_escape_present"
                                                 : "non_escape_missing")
            << "|" << allocation_status << "|"
            << stack_region_diagnostic_token(allocation_key.str()) << "|"
            << raw_status;
        auto& raw_value = dry_run_raw_provenance[raw_key.str()];
        raw_value.first += 1u;
        raw_value.second += pending.bytes;
      }
      if (safe_candidate && !large_backing) {
        dry_run_safe_candidate_count++;
        dry_run_safe_candidate_bytes += pending.bytes;
      } else {
        dry_run_blockers.insert(resource_class);
      }
      if (large_backing) {
        dry_run_has_large_backing = true;
        dry_run_blockers.insert("large_backing_excluded");
      }
      note_stack_subresource_lifetime_dry_run_resource(
          pending.kind,
          pending.role,
          pending.phase,
          pending.callsite == VulkanRetireCallSite::Unknown ? callsite
                                                            : pending.callsite,
          pending.bytes,
          resource_class,
          safe_candidate,
          large_backing,
          formal_last_use_proof,
          pending.stack_provenance,
          allocation_proof,
          allocation_label);
      note_stack_owner_dispatch_dependency_dry_run(
          pending.kind,
          pending.role,
          phase,
          pending.callsite == VulkanRetireCallSite::Unknown ? callsite
                                                            : pending.callsite,
          /*queue_submit=*/true,
          pending.bytes,
          resource_class,
          formal_last_use_proof,
          pending.stack_provenance,
          allocation_proof,
          allocation_label);
    };
    {
      std::lock_guard<std::mutex> bufferlist_lock(
          pending_retire_buffers_mutex_);
      pending_resource_count += pending_retire_buffers_.size();
      for (const PendingRetireBuffer& pending : pending_retire_buffers_) {
        inspect_pending_resource(pending);
      }
    }
    {
      std::lock_guard<std::mutex> imagelist_lock(pending_retire_images_mutex_);
      pending_resource_count += pending_retire_images_.size();
      for (const PendingRetireImage& pending : pending_retire_images_) {
        inspect_pending_resource(pending);
      }
    }
    bool dry_run_all_safe_group_eligible = false;
    if (record_phase_boundary_dry_run) {
      std::ostringstream dry_run_signature_stream;
      for (const auto& entry : dry_run_resource_classes) {
        if (dry_run_signature_stream.tellp() > 0) {
          dry_run_signature_stream << ",";
        }
        dry_run_signature_stream << entry.first << "#" << entry.second.first
                                 << "#" << entry.second.second;
      }
      std::ostringstream dry_run_blocker_signature_stream;
      for (const auto& blocker : dry_run_blockers) {
        if (dry_run_blocker_signature_stream.tellp() > 0) {
          dry_run_blocker_signature_stream << ",";
        }
        dry_run_blocker_signature_stream << blocker;
      }
      dry_run_signature = dry_run_signature_stream.str();
      dry_run_blocker_signature = dry_run_blocker_signature_stream.str();
      std::ostringstream dry_run_allocation_signature_stream;
      for (const auto& entry : dry_run_allocation_ranges) {
        if (dry_run_allocation_signature_stream.tellp() > 0) {
          dry_run_allocation_signature_stream << ",";
        }
        dry_run_allocation_signature_stream << entry.first << "#"
                                            << entry.second.first << "#"
                                            << entry.second.second;
      }
      dry_run_allocation_signature = dry_run_allocation_signature_stream.str();
      std::ostringstream dry_run_raw_provenance_signature_stream;
      for (const auto& entry : dry_run_raw_provenance) {
        if (dry_run_raw_provenance_signature_stream.tellp() > 0) {
          dry_run_raw_provenance_signature_stream << ",";
        }
        dry_run_raw_provenance_signature_stream << entry.first << "#"
                                                << entry.second.first << "#"
                                                << entry.second.second;
      }
      dry_run_raw_provenance_signature =
          dry_run_raw_provenance_signature_stream.str();
      const bool all_safe_without_budget =
          pending_resource_count > 0u &&
          pending_resource_count == dry_run_safe_candidate_count &&
          !dry_run_has_large_backing;
      if (pending_resource_count == 0u) {
        dry_run_budget_reject = "no_old_path_pending";
      } else if (dry_run_has_large_backing) {
        dry_run_budget_reject = "large_backing_excluded";
      } else if (!all_safe_without_budget) {
        dry_run_budget_reject = "unsafe_resource_class";
      } else if (
          dry_run_safe_candidate_bytes >
          kStackActivationPhaseBoundaryLifetimeBlockBudgetBytes) {
        dry_run_budget_reject = "over_block_budget";
      } else if (
          dry_run_safe_candidate_bytes >
          kStackActivationPhaseBoundaryLifetimeScopeBudgetBytes) {
        dry_run_budget_reject = "over_scope_budget";
      } else {
        dry_run_budget_reject = "none";
      }
      dry_run_all_safe_group_eligible = dry_run_budget_reject == "none";
    }
    const bool should_elide_stack_region_boundary_submit =
        record_phase_boundary_dry_run &&
        maybe_elide_stack_region_boundary_submit_canary(phase, callsite);
    const bool should_coalesce_phase_boundary_explicit_sync =
        (kCoalescePhaseBoundaryExplicitSync &&
         dry_run_all_safe_group_eligible) ||
        should_elide_stack_region_boundary_submit;
    note_region_lifetime_submit_attribution_group(
        origin,
        phase,
        callsite,
        /*queue_submit=*/!should_coalesce_phase_boundary_explicit_sync,
        /*had_pending_work=*/true,
        pending_resource_count,
        pending_bytes,
        format_region_lifetime_submit_signature(resources),
        format_region_lifetime_submit_blockers(blockers));
    for (const auto& attribution : region_lifetime_resource_attributions) {
      note_region_lifetime_submit_attribution_resource(
          origin,
          attribution.phase,
          attribution.callsite,
          attribution.kind,
          attribution.role,
          attribution.bytes,
          attribution.reason,
          attribution.safety,
          /*queue_submit=*/!should_coalesce_phase_boundary_explicit_sync,
          /*had_pending_work=*/true,
          attribution.provenance,
          attribution.allocation_proof,
          attribution.allocation_label);
    }
    if (record_phase_boundary_dry_run) {
      note_stack_phase_boundary_lifetime_dry_run_group(
          phase,
          callsite,
          /*queue_submit=*/!should_coalesce_phase_boundary_explicit_sync,
          pending_resource_count,
          pending_bytes,
          dry_run_safe_candidate_count,
          dry_run_safe_candidate_bytes,
          dry_run_all_safe_group_eligible,
          dry_run_all_safe_group_eligible,
          /*actual_removed_explicit_synchronize=*/
          should_coalesce_phase_boundary_explicit_sync,
          kStackActivationPhaseBoundaryLifetimeBlockBudgetBytes,
          kStackActivationPhaseBoundaryLifetimeScopeBudgetBytes,
          dry_run_budget_reject,
          dry_run_signature,
          dry_run_blocker_signature);
      note_stack_region_boundary_submit_plan(
          phase,
          callsite,
          /*queue_submit=*/!should_coalesce_phase_boundary_explicit_sync,
          pending_resource_count,
          pending_bytes,
          dry_run_safe_candidate_count,
          dry_run_safe_candidate_bytes,
          command_buffer_recording_id,
          submit_epoch_before,
          should_coalesce_phase_boundary_explicit_sync ? submit_epoch_before
                                                       : submit_epoch_after,
          pending_dispatch_count,
          dry_run_budget_reject,
          dry_run_signature,
          dry_run_allocation_signature,
          dry_run_raw_provenance_signature,
          dry_run_blocker_signature);
    }
    if (should_coalesce_phase_boundary_explicit_sync) {
      VulkanSubmission submission{};
      poll_retire_queue();
      if (cpu_timeline) {
        std::ostringstream stream;
        stream << "event=submit_cmd_to_gpu had_cmd=1"
               << " coalesced_phase_boundary_sync=1"
               << " duration_us=" << (cpu_timeline_now_us() - cpu_start_us)
               << " fence=0 final_use=0";
        append_cpu_timeline_log_line(stream.str());
      }
      return submission;
    }
  }
  VulkanSubmission submission{};
  if (cmd_) {
    cmd_.end();
    submission = submit_cmd_handle_to_gpu(
        current_stream(),
        cmd_.get_submit_handle(final_use),
        origin,
        fence_handle,
        final_use);
    last_submission_ = submission;

    submit_count_ = 0u;
    command_buffer_recording_id_ = 0u;
    retire_deferred_cleanup(submission, origin);
  }
  poll_retire_queue();
  if (cpu_timeline) {
    std::ostringstream stream;
    stream << "event=submit_cmd_to_gpu had_cmd=" << (had_cmd ? 1 : 0)
           << " duration_us=" << (cpu_timeline_now_us() - cpu_start_us)
           << " fence=" << (fence_handle != VK_NULL_HANDLE ? 1 : 0)
           << " final_use=" << (final_use ? 1 : 0);
    append_cpu_timeline_log_line(stream.str());
  }
  return submission;
}

void Context::flush_pending_cmds(VkFence fence_handle) {
  const bool cpu_timeline = cpu_timeline_logging_enabled();
  const uint64_t cpu_start_us =
      cpu_timeline ? cpu_timeline_now_us() : 0u;
  std::unique_lock<std::mutex> context_lock(dispatch_lock());
  submit_cmd_to_gpu(
      fence_handle, false, VulkanSubmitOrigin::TensorCpuReadback);
  if (cpu_timeline) {
    std::ostringstream stream;
    stream << "event=flush_pending_cmds duration_us="
           << (cpu_timeline_now_us() - cpu_start_us)
           << " fence=" << (fence_handle != VK_NULL_HANDLE ? 1 : 0);
    append_cpu_timeline_log_line(stream.str());
  }
}

void Context::begin_stack_planned_recording() {
  std::unique_lock<std::mutex> context_lock(dispatch_lock());
  VK_CHECK_COND(
      !is_inside_owned_program_recording(),
      "Cannot begin stack planned recording inside external command recording");
  VK_CHECK_COND(
      !is_stack_planned_recording_active(),
      "Vulkan stack planned recording is already active");
  submit_cmd_to_gpu(VK_NULL_HANDLE, false, VulkanSubmitOrigin::PreStackFlush);
  stack_planned_recording_owner_ = std::this_thread::get_id();
  stack_planned_recording_stats_ = StackPlannedRecordingStats{};
  stack_planned_recording_active_.store(true, std::memory_order_release);
}

StackPlannedRecordingStats Context::end_stack_planned_recording_and_submit() {
  std::unique_lock<std::mutex> context_lock(dispatch_lock());
  VK_CHECK_COND(
      is_stack_planned_recording_active(),
      "Vulkan stack planned recording is not active");
  VK_CHECK_COND(
      stack_planned_recording_owner_ == std::this_thread::get_id(),
      "Vulkan stack planned recording ended from the wrong thread");
  StackPlannedRecordingStats stats = stack_planned_recording_stats_;
  VulkanSubmission submission = submit_cmd_to_gpu(
      VK_NULL_HANDLE, false, VulkanSubmitOrigin::StackPlannedRecordingSubmit);
  retire_stack_internal_temp_retire_batch_locked(submission);
  stack_planned_recording_active_.store(false, std::memory_order_release);
  stack_planned_recording_owner_ = std::thread::id{};
  stack_planned_recording_stats_ = StackPlannedRecordingStats{};
  return stats;
}

StackPlannedRecordingStats Context::cancel_stack_planned_recording() {
  std::unique_lock<std::mutex> context_lock(dispatch_lock());
  VK_CHECK_COND(
      is_stack_planned_recording_active(),
      "Vulkan stack planned recording is not active");
  StackPlannedRecordingStats stats = stack_planned_recording_stats_;
  restore_stack_internal_temp_retire_batch_to_pending_locked();
  stack_planned_recording_active_.store(false, std::memory_order_release);
  stack_planned_recording_owner_ = std::thread::id{};
  stack_planned_recording_stats_ = StackPlannedRecordingStats{};
  submit_cmd_to_gpu(VK_NULL_HANDLE, false, VulkanSubmitOrigin::PostStackFlush);
  return stats;
}

CommandBuffer Context::acquire_persistent_command_buffer() {
  CommandBuffer cmd = persistent_command_pool_.get_new_cmd(/*reusable=*/true);
  cmd.begin();
  return cmd;
}

void Context::submit_prepared_command_buffer(
    CommandBuffer& cmd,
    VkFence fence_handle,
    const bool final_use,
    const char* profile_label) {
  VK_CHECK_COND(
      !is_inside_owned_program_recording(),
      "submit_prepared_command_buffer cannot be called from inside an owned "
      "Vulkan replay/program recording scope. Nested phases must be lowered "
      "as first-class program or executable-region stages.");
  const bool cpu_timeline = cpu_timeline_logging_enabled();
  const uint64_t cpu_start_us =
      cpu_timeline ? cpu_timeline_now_us() : 0u;
  std::unique_lock<std::mutex> context_lock(dispatch_lock());

  const bool profile_submit =
      enable_op_profiling_ && querypool_.is_enabled();
  const uint32_t log_idx = [&]() -> uint32_t {
    if (!profile_submit) {
      return UINT32_MAX;
    }
    CommandBuffer begin_cmd = command_pool_.get_new_cmd(/*reusable=*/false);
    begin_cmd.begin();
    const std::string label =
        (profile_label && profile_label[0] != '\0')
        ? std::string("prepared.") + profile_label
        : std::string("prepared_command_buffer");
    const uint32_t idx = gpu_profile_begin(
        begin_cmd,
        label,
        create_extent3d({0, 0, 0}),
        create_extent3d({0, 0, 0}));
    begin_cmd.end();
    adapter_p_->submit_cmd(queue_, begin_cmd.get_submit_handle(/*final_use=*/true));
    note_vulkan_queue_submit(VulkanSubmitOrigin::ProfilingTimestampReset);
    return idx;
  }();

  cmd.end();
  VulkanSubmission submission = submit_cmd_handle_to_gpu(
      current_stream(),
      cmd.get_submit_handle(final_use),
      VulkanSubmitOrigin::DebugValidation,
      profile_submit ? VK_NULL_HANDLE : fence_handle,
      final_use);
  last_submission_ = submission;

  if (profile_submit) {
    CommandBuffer end_cmd = command_pool_.get_new_cmd(/*reusable=*/false);
    end_cmd.begin();
    gpu_profile_end(end_cmd, log_idx);
    end_cmd.end();
    adapter_p_->submit_cmd(
        queue_, end_cmd.get_submit_handle(/*final_use=*/true), fence_handle);
    note_vulkan_queue_submit(VulkanSubmitOrigin::ProfilingTimestampReadback);
  }

  if (profile_submit) {
    querypool_.mark_results_pending();
  }
  retire_deferred_cleanup(
      submission,
      profile_submit ? VulkanSubmitOrigin::ProfilingTimestampReadback
                     : VulkanSubmitOrigin::Unknown);
  poll_retire_queue();
  if (cpu_timeline) {
    std::ostringstream stream;
    stream << "event=submit_prepared_command_buffer duration_us="
           << (cpu_timeline_now_us() - cpu_start_us)
           << " fence=" << (fence_handle != VK_NULL_HANDLE ? 1 : 0)
           << " final_use=" << (final_use ? 1 : 0)
           << " profile_label="
           << (profile_label && profile_label[0] != '\0' ? profile_label : "");
    append_cpu_timeline_log_line(stream.str());
  }
}

void Context::take_external_recording_cleanup_resources(
    std::vector<VulkanBuffer>& buffers,
    std::vector<VulkanImage>& images) {
  buffers = std::move(g_external_command_recording_state.buffers_to_keep_alive);
  images = std::move(g_external_command_recording_state.images_to_keep_alive);
  g_external_command_recording_state.buffers_to_keep_alive.clear();
  g_external_command_recording_state.images_to_keep_alive.clear();
}

void Context::clear_pending_retire_resources_locked() {
  std::lock_guard<std::mutex> bufferlist_lock(pending_retire_buffers_mutex_);
  std::lock_guard<std::mutex> imagelist_lock(pending_retire_images_mutex_);
  pending_retire_buffers_.clear();
  pending_retire_images_.clear();
  pending_retire_bytes_.store(0u, std::memory_order_relaxed);
}

void Context::clear_stack_internal_temp_retire_batch_locked() {
  std::lock_guard<std::mutex> batch_lock(
      stack_internal_temp_retire_batch_mutex_);
  stack_internal_temp_retire_batch_buffers_.clear();
  stack_internal_temp_retire_batch_images_.clear();
}

void Context::restore_stack_internal_temp_retire_batch_to_pending_locked() {
  std::lock_guard<std::mutex> batch_lock(
      stack_internal_temp_retire_batch_mutex_);
  if (
      stack_internal_temp_retire_batch_buffers_.empty() &&
      stack_internal_temp_retire_batch_images_.empty()) {
    return;
  }
  {
    std::lock_guard<std::mutex> bufferlist_lock(pending_retire_buffers_mutex_);
    for (auto& pending : stack_internal_temp_retire_batch_buffers_) {
      pending_retire_bytes_.fetch_add(
          pending.bytes, std::memory_order_relaxed);
      pending_retire_buffers_.push_back(std::move(pending));
    }
    stack_internal_temp_retire_batch_buffers_.clear();
  }
  {
    std::lock_guard<std::mutex> imagelist_lock(pending_retire_images_mutex_);
    for (auto& pending : stack_internal_temp_retire_batch_images_) {
      pending_retire_bytes_.fetch_add(
          pending.bytes, std::memory_order_relaxed);
      pending_retire_images_.push_back(std::move(pending));
    }
    stack_internal_temp_retire_batch_images_.clear();
  }
}

void Context::retire_stack_internal_temp_retire_batch_locked(
    const VulkanSubmission& submission) {
  if (submission.timeline == VK_NULL_HANDLE || submission.timeline_value == 0u) {
    restore_stack_internal_temp_retire_batch_to_pending_locked();
    return;
  }
  std::lock_guard<std::mutex> batch_lock(
      stack_internal_temp_retire_batch_mutex_);
  uint64_t batch_bytes = 0u;
  for (PendingRetireBuffer& pending :
       stack_internal_temp_retire_batch_buffers_) {
    note_vulkan_retired_resource(
        pending.kind,
        pending.role,
        pending.phase,
        VulkanRetireCallSite::StackPlannedRecordingEnd,
        pending.bytes,
        /*queue_submit=*/true,
        /*blocking_wait=*/false,
        /*poll_only=*/false,
        pending.stack_provenance);
    batch_bytes += pending.bytes;
    retire_queue_.retire(RetiredResource{
        submission.stream_id,
        submission.timeline,
        submission.timeline_value,
        [buffer = std::make_shared<VulkanBuffer>(
             std::move(pending.buffer))]() {},
    });
  }
  stack_internal_temp_retire_batch_buffers_.clear();
  for (PendingRetireImage& pending :
       stack_internal_temp_retire_batch_images_) {
    note_vulkan_retired_resource(
        pending.kind,
        pending.role,
        pending.phase,
        VulkanRetireCallSite::StackPlannedRecordingEnd,
        pending.bytes,
        /*queue_submit=*/true,
        /*blocking_wait=*/false,
        /*poll_only=*/false,
        pending.stack_provenance);
    batch_bytes += pending.bytes;
    retire_queue_.retire(RetiredResource{
        submission.stream_id,
        submission.timeline,
        submission.timeline_value,
        [image = std::make_shared<VulkanImage>(
             std::move(pending.image))]() {},
    });
  }
  stack_internal_temp_retire_batch_images_.clear();
  if (batch_bytes > 0u) {
    note_stack_internal_temp_retire_batch_submitted(batch_bytes);
  }
}

void Context::flush() {
  const bool cpu_timeline = cpu_timeline_logging_enabled();
  const uint64_t cpu_start_us =
      cpu_timeline ? cpu_timeline_now_us() : 0u;
  if (sync_logging_enabled()) {
    std::ostringstream stream;
    stream << "flush: pending=" << format_sync_bytes(pending_retire_bytes())
           << " submit_count=" << submit_count_;
    append_sync_log_line(stream.str());
  }

  synchronize_device();
  dump_gpu_profile_log("flush");
  if (cpu_timeline) {
    std::ostringstream stream;
    stream << "event=flush duration_us=" << (cpu_timeline_now_us() - cpu_start_us);
    append_cpu_timeline_log_line(stream.str());
  }
  dump_cpu_timeline_summary_log();
}

void Context::retire_after_fence_wait() {
  const bool cpu_timeline = cpu_timeline_logging_enabled();
  const uint64_t cpu_start_us =
      cpu_timeline ? cpu_timeline_now_us() : 0u;
  const bool flush_pools = true;

  if (sync_logging_enabled()) {
    std::ostringstream stream;
    stream << "retire_after_fence_wait: pending="
           << format_sync_bytes(pending_retire_bytes())
           << " submit_count=" << submit_count_
           << " caller=" << current_allocation_label()
           << " flush_pools=" << (flush_pools ? "1" : "0");
    append_sync_log_line(stream.str());
  }

  if (flush_pools) {
    command_pool_.flush();
    descriptor_pool_.flush();
  }
  if (cmd_) {
    cmd_.invalidate();
  }

  submit_count_ = 0u;
  command_buffer_recording_id_ = 0u;
  clear_pending_retire_resources_locked();
  dump_gpu_profile_log("retire_after_fence_wait");
  if (cpu_timeline) {
    std::ostringstream stream;
    stream << "event=retire_after_fence_wait duration_us="
           << (cpu_timeline_now_us() - cpu_start_us)
           << " flush_pools=" << (flush_pools ? 1 : 0);
    append_cpu_timeline_log_line(stream.str());
  }
}

void Context::flush_after_fence_wait() {
  const bool cpu_timeline = cpu_timeline_logging_enabled();
  const uint64_t cpu_start_us =
      cpu_timeline ? cpu_timeline_now_us() : 0u;
  if (sync_logging_enabled()) {
    std::ostringstream stream;
    stream << "flush_after_fence_wait: pending="
           << format_sync_bytes(pending_retire_bytes())
           << " submit_count=" << submit_count_
           << " caller=" << current_allocation_label();
    append_sync_log_line(stream.str());
  }

  command_pool_.flush();
  descriptor_pool_.flush();

  if (cmd_) {
    cmd_.invalidate();
  }

  submit_count_ = 0u;
  command_buffer_recording_id_ = 0u;
  clear_pending_retire_resources_locked();
  dump_gpu_profile_log("flush_after_fence_wait");
  if (cpu_timeline) {
    std::ostringstream stream;
    stream << "event=flush_after_fence_wait duration_us="
           << (cpu_timeline_now_us() - cpu_start_us);
    append_cpu_timeline_log_line(stream.str());
  }
}

bool available() {
  return runtime()->device_count() > 0u;
}

c10::DeviceIndex device_count() {
  return utils::safe_downcast<c10::DeviceIndex>(runtime()->device_count());
}

c10::DeviceIndex current_device() {
  if (runtime()->device_count() == 0u) {
    return -1;
  }

  if (g_current_device_index < 0) {
    g_current_device_index = runtime()->default_device_index();
  }

  validate_device_index(g_current_device_index);
  return g_current_device_index;
}

void set_current_device(c10::DeviceIndex device_index) {
  validate_device_index(device_index);
  g_current_device_index = device_index;
}

c10::DeviceIndex exchange_device(c10::DeviceIndex device_index) {
  const c10::DeviceIndex previous_device = current_device();
  set_current_device(device_index);
  return previous_device;
}

Context* context(c10::DeviceIndex device_index) {
  validate_device_index(device_index);

  static std::mutex* const contexts_mutex = new std::mutex();
  static std::vector<Context*>* const contexts = new std::vector<Context*>();

  Context* device_context = nullptr;
  {
    std::lock_guard<std::mutex> lock(*contexts_mutex);
    const size_t required_size = runtime()->device_count();
    if (contexts->size() < required_size) {
      contexts->resize(required_size, nullptr);
    }

    Context*& stored_context =
        contexts->at(utils::safe_downcast<size_t>(device_index));
    if (!stored_context) {
      stored_context = new Context(device_index, default_context_config());
    }
    device_context = stored_context;
  }

  return device_context;
}

Context* context() {
  const c10::DeviceIndex device_index = current_device();
  if (device_index < 0) {
    return nullptr;
  }
  return context(device_index);
}

//
// UniformParamsBuffer
//

namespace {

void memcpy_to_buffer(const VulkanBuffer& src, VulkanBuffer& dst) {
  MemoryMap dst_mapping(dst, MemoryAccessType::WRITE);

  MemoryMap src_mapping(src, MemoryAccessType::READ);
  src_mapping.invalidate();

  void* dst_ptr = dst_mapping.template data<void>();
  void* src_ptr = src_mapping.template data<void>();

  // @lint-ignore CLANGTIDY facebook-security-vulnerable-memcpy
  memcpy(dst_ptr, src_ptr, src.mem_size());
}

} // namespace

UniformParamsBuffer::UniformParamsBuffer(const UniformParamsBuffer& other)
    : context_p_(other.context_p_),
      nbytes_(other.nbytes_),
      vulkan_buffer_{},
      retire_kind_(other.retire_kind_),
      retire_role_(other.retire_role_),
      retire_phase_(other.retire_phase_),
      retire_callsite_(other.retire_callsite_) {
  if (other.vulkan_buffer_) {
    vulkan_buffer_ = context_p_->adapter_ptr()->vma().create_uniform_buffer(
        other.vulkan_buffer_.mem_size());

    memcpy_to_buffer(other.vulkan_buffer_, vulkan_buffer_);
  }
}

UniformParamsBuffer& UniformParamsBuffer::operator=(
    const UniformParamsBuffer& other) {
  if (&other != this) {
    context_p_ = other.context_p_;
    nbytes_ = other.nbytes_;
    retire_kind_ = other.retire_kind_;
    retire_role_ = other.retire_role_;
    retire_phase_ = other.retire_phase_;
    retire_callsite_ = other.retire_callsite_;

    // Move vulkan_buffer_ to another VulkanBuffer for cleanup
    if (vulkan_buffer_) {
      VulkanBuffer temp_buffer(std::move(vulkan_buffer_));
      context_p_->register_buffer_cleanup(temp_buffer);
    }
    // vulkan_buffer_ should now be empty

    if (other.vulkan_buffer_) {
      vulkan_buffer_ = context_p_->adapter_ptr()->vma().create_uniform_buffer(
          other.vulkan_buffer_.mem_size());

      memcpy_to_buffer(other.vulkan_buffer_, vulkan_buffer_);
    }
  }

  return *this;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
