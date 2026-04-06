#include <ATen/native/vulkan/planning/ExecutionPrograms.h>

#include <ATen/native/vulkan/ops/InferenceCache.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

constexpr size_t kExecutionProgramCacheSize = 64u;

std::string normalize_program_label(
    const std::string& allocation_label,
    const char* fallback) {
  if (!allocation_label.empty()) {
    return allocation_label;
  }
  return std::string(fallback);
}

std::string program_object_label(
    const std::string& allocation_label,
    const char* suffix) {
  return normalize_program_label(allocation_label, suffix) + "." + suffix;
}

const std::string& execution_program_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_EXECUTION_PROGRAM_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool execution_program_logging_enabled() {
  return !execution_program_log_path().empty();
}

void log_execution_program_event(
    const VulkanExecutionProgramKind kind,
    const char* event,
    const std::string& allocation_label,
    const void* identity) {
  if (!execution_program_logging_enabled()) {
    return;
  }

  std::ofstream out(execution_program_log_path(), std::ios::app);
  out << "execution_program event=" << event << " kind="
      << execution_program_kind_name(kind) << " allocation_label="
      << allocation_label << " identity=" << identity << '\n';
}

bool same_sizes(
    const std::vector<int64_t>& lhs,
    const std::vector<int64_t>& rhs) {
  return lhs.size() == rhs.size() &&
      std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool same_kv_cache_spec(
    const std::optional<VulkanKVCacheSpec>& lhs,
    const std::optional<VulkanKVCacheSpec>& rhs) {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  if (!lhs.has_value()) {
    return true;
  }
  return lhs->dtype == rhs->dtype && same_sizes(lhs->sizes, rhs->sizes) &&
      lhs->sequence_dim == rhs->sequence_dim &&
      lhs->execution_layout == rhs->execution_layout &&
      lhs->memory_layout == rhs->memory_layout &&
      lhs->storage_type == rhs->storage_type &&
      lhs->persistent == rhs->persistent;
}

bool same_scratch_spec(
    const std::optional<VulkanScratchArenaSpec>& lhs,
    const std::optional<VulkanScratchArenaSpec>& rhs) {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  if (!lhs.has_value()) {
    return true;
  }
  return lhs->dtype == rhs->dtype && lhs->num_bytes == rhs->num_bytes &&
      lhs->alignment == rhs->alignment &&
      lhs->execution_layout == rhs->execution_layout &&
      lhs->memory_layout == rhs->memory_layout &&
      lhs->storage_type == rhs->storage_type &&
      lhs->persistent == rhs->persistent;
}

struct AttentionRuntimeProgramKey final {
  std::string allocation_label;
  VulkanAttentionKernelFamily kernel_family{
      VulkanAttentionKernelFamily::TextureMath};
  std::optional<VulkanKVCacheSpec> key_cache_spec;
  std::optional<VulkanKVCacheSpec> value_cache_spec;
  std::optional<VulkanScratchArenaSpec> scratch_spec;
  bool persistent{true};
};

bool operator==(
    const AttentionRuntimeProgramKey& lhs,
    const AttentionRuntimeProgramKey& rhs) {
  return lhs.allocation_label == rhs.allocation_label &&
      lhs.kernel_family == rhs.kernel_family &&
      same_kv_cache_spec(lhs.key_cache_spec, rhs.key_cache_spec) &&
      same_kv_cache_spec(lhs.value_cache_spec, rhs.value_cache_spec) &&
      same_scratch_spec(lhs.scratch_spec, rhs.scratch_spec) &&
      lhs.persistent == rhs.persistent;
}

InferenceLruCache<AttentionRuntimeProgramKey, AttentionRuntimeProgram>&
attention_runtime_program_cache() {
  static InferenceLruCache<AttentionRuntimeProgramKey, AttentionRuntimeProgram>
      cache{kExecutionProgramCacheSize};
  return cache;
}

struct GatedDeltaSplitProgramKey final {
  std::string allocation_label;
  VulkanBoundaryPlan boundary_plan{};
  std::optional<VulkanScratchArenaSpec> scratch_spec;
  bool persistent{true};
};

bool operator==(
    const GatedDeltaSplitProgramKey& lhs,
    const GatedDeltaSplitProgramKey& rhs) {
  return lhs.allocation_label == rhs.allocation_label &&
      lhs.boundary_plan.kind == rhs.boundary_plan.kind &&
      lhs.boundary_plan.input_transfer_layout ==
          rhs.boundary_plan.input_transfer_layout &&
      lhs.boundary_plan.output_transfer_layout ==
          rhs.boundary_plan.output_transfer_layout &&
      lhs.boundary_plan.prefer_backend_owned_execution ==
          rhs.boundary_plan.prefer_backend_owned_execution &&
      lhs.boundary_plan.requires_scratch_arena ==
          rhs.boundary_plan.requires_scratch_arena &&
      lhs.boundary_plan.preferred_cpu_threads ==
          rhs.boundary_plan.preferred_cpu_threads &&
      same_scratch_spec(lhs.scratch_spec, rhs.scratch_spec) &&
      lhs.persistent == rhs.persistent;
}

InferenceLruCache<GatedDeltaSplitProgramKey, GatedDeltaSplitProgram>&
gated_delta_split_program_cache() {
  static InferenceLruCache<GatedDeltaSplitProgramKey, GatedDeltaSplitProgram>
      cache{kExecutionProgramCacheSize};
  return cache;
}

} // namespace

struct AttentionRuntimeProgram::State final {
  VulkanAttentionKernelFamily kernel_family_{
      VulkanAttentionKernelFamily::TextureMath};
  std::optional<KVCacheObject> key_cache_;
  std::optional<KVCacheObject> value_cache_;
  std::optional<ScratchArena> scratch_arena_;
  bool persistent_{true};

  State(
      const VulkanAttentionKernelFamily kernel_family,
      std::optional<KVCacheObject> key_cache,
      std::optional<KVCacheObject> value_cache,
      std::optional<ScratchArena> scratch_arena,
      const bool persistent)
      : kernel_family_(kernel_family),
        key_cache_(std::move(key_cache)),
        value_cache_(std::move(value_cache)),
        scratch_arena_(std::move(scratch_arena)),
        persistent_(persistent) {}
};

struct GatedDeltaSplitProgram::State final {
  VulkanBoundaryPlan boundary_plan_{};
  std::optional<ScratchArena> scratch_arena_;
  bool persistent_{true};

  State(
      VulkanBoundaryPlan boundary_plan,
      std::optional<ScratchArena> scratch_arena,
      const bool persistent)
      : boundary_plan_(std::move(boundary_plan)),
        scratch_arena_(std::move(scratch_arena)),
        persistent_(persistent) {}
};

bool AttentionRuntimeProgram::defined() const {
  return static_cast<bool>(state_);
}

VulkanAttentionKernelFamily AttentionRuntimeProgram::kernel_family() const {
  return state_ ? state_->kernel_family_
                : VulkanAttentionKernelFamily::TextureMath;
}

const std::optional<KVCacheObject>& AttentionRuntimeProgram::key_cache() const {
  static const std::optional<KVCacheObject> empty;
  return state_ ? state_->key_cache_ : empty;
}

const std::optional<KVCacheObject>& AttentionRuntimeProgram::value_cache()
    const {
  static const std::optional<KVCacheObject> empty;
  return state_ ? state_->value_cache_ : empty;
}

const std::optional<ScratchArena>& AttentionRuntimeProgram::scratch_arena()
    const {
  static const std::optional<ScratchArena> empty;
  return state_ ? state_->scratch_arena_ : empty;
}

bool AttentionRuntimeProgram::persistent() const {
  return state_ && state_->persistent_;
}

void AttentionRuntimeProgram::set_sequence_lengths(
    const int64_t key_sequence_length,
    const int64_t value_sequence_length) const {
  if (!state_) {
    return;
  }
  if (state_->key_cache_.has_value()) {
    state_->key_cache_->set_sequence_length(key_sequence_length);
  }
  if (state_->value_cache_.has_value()) {
    state_->value_cache_->set_sequence_length(value_sequence_length);
  }
}

const void* AttentionRuntimeProgram::identity() const {
  return state_.get();
}

bool GatedDeltaSplitProgram::defined() const {
  return static_cast<bool>(state_);
}

const VulkanBoundaryPlan& GatedDeltaSplitProgram::boundary_plan() const {
  TORCH_INTERNAL_ASSERT(state_, "Undefined GatedDeltaSplitProgram");
  return state_->boundary_plan_;
}

const std::optional<ScratchArena>& GatedDeltaSplitProgram::scratch_arena()
    const {
  static const std::optional<ScratchArena> empty;
  return state_ ? state_->scratch_arena_ : empty;
}

bool GatedDeltaSplitProgram::persistent() const {
  return state_ && state_->persistent_;
}

const void* GatedDeltaSplitProgram::identity() const {
  return state_.get();
}

AttentionRuntimeProgram lookup_or_create_labeled_attention_runtime_program(
    const std::string& allocation_label,
    const VulkanAttentionKernelFamily kernel_family,
    const std::optional<VulkanKVCacheSpec>& key_cache_spec,
    const std::optional<VulkanKVCacheSpec>& value_cache_spec,
    const std::optional<VulkanScratchArenaSpec>& scratch_spec,
    const int64_t key_sequence_length,
    const int64_t value_sequence_length,
    const VulkanExecutionProgramPlanningDesc& program_plan) {
  const AttentionRuntimeProgramKey query{
      normalize_program_label(allocation_label, "attention_runtime"),
      kernel_family,
      key_cache_spec,
      value_cache_spec,
      scratch_spec,
      program_plan.persistent};
  if (const auto cached = attention_runtime_program_cache().lookup(
          query,
          [](const AttentionRuntimeProgramKey& lhs,
             const AttentionRuntimeProgramKey& rhs) { return lhs == rhs; })) {
    cached->set_sequence_lengths(key_sequence_length, value_sequence_length);
    log_execution_program_event(
        VulkanExecutionProgramKind::AttentionRuntime,
        "hit",
        query.allocation_label,
        cached->identity());
    return *cached;
  }

  std::optional<KVCacheObject> key_cache;
  if (key_cache_spec.has_value()) {
    key_cache = lookup_or_create_labeled_kv_cache_object(
        program_object_label(query.allocation_label, "key_cache"),
        *key_cache_spec);
    key_cache->set_sequence_length(key_sequence_length);
  }

  std::optional<KVCacheObject> value_cache;
  if (value_cache_spec.has_value()) {
    value_cache = lookup_or_create_labeled_kv_cache_object(
        program_object_label(query.allocation_label, "value_cache"),
        *value_cache_spec);
    value_cache->set_sequence_length(value_sequence_length);
  }

  std::optional<ScratchArena> scratch_arena;
  if (scratch_spec.has_value()) {
    scratch_arena = lookup_or_create_labeled_scratch_arena(
        program_object_label(query.allocation_label, "scratch"),
        *scratch_spec);
  }

  AttentionRuntimeProgram created{std::make_shared<AttentionRuntimeProgram::State>(
      kernel_family,
      std::move(key_cache),
      std::move(value_cache),
      std::move(scratch_arena),
      program_plan.persistent)};
  attention_runtime_program_cache().store(
      query,
      created,
      [](const AttentionRuntimeProgramKey& lhs,
         const AttentionRuntimeProgramKey& rhs) { return lhs == rhs; });
  log_execution_program_event(
      VulkanExecutionProgramKind::AttentionRuntime,
      "store",
      query.allocation_label,
      created.identity());
  return created;
}

std::optional<GatedDeltaSplitProgram>
lookup_or_create_labeled_gated_delta_split_program(
    const std::string& allocation_label,
    const VulkanBoundaryPlan& boundary_plan,
    const std::optional<VulkanScratchArenaSpec>& scratch_spec,
    const VulkanExecutionProgramPlanningDesc& program_plan) {
  const GatedDeltaSplitProgramKey query{
      normalize_program_label(allocation_label, "gated_delta_split"),
      boundary_plan,
      scratch_spec,
      program_plan.persistent};
  if (const auto cached = gated_delta_split_program_cache().lookup(
          query,
          [](const GatedDeltaSplitProgramKey& lhs,
             const GatedDeltaSplitProgramKey& rhs) { return lhs == rhs; })) {
    log_execution_program_event(
        VulkanExecutionProgramKind::GatedDeltaSplit,
        "hit",
        query.allocation_label,
        cached->identity());
    return *cached;
  }

  std::optional<ScratchArena> scratch_arena;
  if (scratch_spec.has_value()) {
    scratch_arena = lookup_or_create_labeled_scratch_arena(
        program_object_label(query.allocation_label, "scratch"),
        *scratch_spec);
  }

  GatedDeltaSplitProgram created{
      std::make_shared<GatedDeltaSplitProgram::State>(
          boundary_plan, std::move(scratch_arena), program_plan.persistent)};
  gated_delta_split_program_cache().store(
      query,
      created,
      [](const GatedDeltaSplitProgramKey& lhs,
         const GatedDeltaSplitProgramKey& rhs) { return lhs == rhs; });
  log_execution_program_event(
      VulkanExecutionProgramKind::GatedDeltaSplit,
      "store",
      query.allocation_label,
      created.identity());
  return created;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
