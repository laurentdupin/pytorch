#include <ATen/native/vulkan/planning/InferenceGraphs.h>

#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/ops/InferenceCache.h>
#include <ATen/native/vulkan/ops/Utils.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

constexpr size_t kInferenceGraphCacheSize = 32u;

const std::string& inference_graph_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_INFERENCE_GRAPH_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool inference_graph_logging_enabled() {
  return !inference_graph_log_path().empty();
}

std::mutex& inference_graph_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

void log_inference_graph_event(
    const VulkanInferenceGraphKind kind,
    const char* event,
    const std::string& allocation_label,
    const void* identity,
    const size_t bytes = 0u) {
  if (!inference_graph_logging_enabled()) {
    return;
  }

  std::lock_guard<std::mutex> lock(inference_graph_log_mutex());
  std::ofstream out(inference_graph_log_path(), std::ios::app);
  out << "inference_graph event=" << event << " kind="
      << inference_graph_kind_name(kind) << " allocation_label="
      << allocation_label;
  if (identity) {
    out << " identity=" << identity;
  }
  if (bytes > 0u) {
    out << " bytes=" << bytes;
  }
  out << '\n';
}

void log_inference_replay_event(
    const VulkanInferenceGraphKind kind,
    const char* event,
    const std::string& allocation_label,
    const void* identity) {
  if (!inference_graph_logging_enabled()) {
    return;
  }

  std::lock_guard<std::mutex> lock(inference_graph_log_mutex());
  std::ofstream out(inference_graph_log_path(), std::ios::app);
  out << "inference_replay event=" << event << " kind="
      << inference_graph_kind_name(kind) << " allocation_label="
      << allocation_label;
  if (identity) {
    out << " identity=" << identity;
  }
  out << '\n';
}

void log_execution_graph_plan_event(
    const VulkanInferenceGraphKind kind,
    const char* event,
    const std::string& allocation_label,
    const void* identity,
    const char* phase_key = nullptr) {
  if (!inference_graph_logging_enabled()) {
    return;
  }

  std::lock_guard<std::mutex> lock(inference_graph_log_mutex());
  std::ofstream out(inference_graph_log_path(), std::ios::app);
  out << "execution_graph_plan event=" << event << " kind="
      << inference_graph_kind_name(kind) << " allocation_label="
      << allocation_label;
  if (identity) {
    out << " identity=" << identity;
  }
  if (phase_key && *phase_key) {
    out << " phase_key=" << phase_key;
  }
  out << '\n';
}

void log_execution_graph_root_event(
    const char* event,
    const std::string& allocation_label,
    const void* identity,
    const VulkanInferenceGraphKind* kind = nullptr,
    const char* phase_key = nullptr) {
  if (!inference_graph_logging_enabled()) {
    return;
  }

  std::lock_guard<std::mutex> lock(inference_graph_log_mutex());
  std::ofstream out(inference_graph_log_path(), std::ios::app);
  out << "execution_graph_root event=" << event
      << " allocation_label=" << allocation_label;
  if (identity) {
    out << " identity=" << identity;
  }
  if (kind) {
    out << " kind=" << inference_graph_kind_name(*kind);
  }
  if (phase_key && *phase_key) {
    out << " phase_key=" << phase_key;
  }
  out << '\n';
}

std::string default_inference_graph_label(const VulkanInferenceGraphKind kind) {
  switch (kind) {
    case VulkanInferenceGraphKind::AttentionRuntime:
      return "attention_runtime.graph";
    case VulkanInferenceGraphKind::VisionBackbone:
      return "vision_backbone.graph";
    case VulkanInferenceGraphKind::VisionDecoder:
      return "vision_decoder.graph";
    case VulkanInferenceGraphKind::ExecutionGraphBundle:
      return "execution_graph_bundle.graph";
  }
  return "inference_graph";
}

struct InferenceGraphKey final {
  VulkanInferenceGraphKind kind{VulkanInferenceGraphKind::VisionBackbone};
  std::string allocation_label;
  ScalarType dtype{kFloat};
  bool persistent{true};
};

bool operator==(const InferenceGraphKey& lhs, const InferenceGraphKey& rhs) {
  return lhs.kind == rhs.kind &&
      lhs.allocation_label == rhs.allocation_label && lhs.dtype == rhs.dtype &&
      lhs.persistent == rhs.persistent;
}

InferenceLruCache<InferenceGraphKey, InferenceGraph>& inference_graph_cache() {
  static InferenceLruCache<InferenceGraphKey, InferenceGraph>
      cache{kInferenceGraphCacheSize};
  return cache;
}

InferenceLruCache<InferenceGraphKey, InferenceReplay>& inference_replay_cache() {
  static InferenceLruCache<InferenceGraphKey, InferenceReplay>
      cache{kInferenceGraphCacheSize};
  return cache;
}

InferenceLruCache<InferenceGraphKey, ExecutionGraphPlan>&
execution_graph_plan_cache() {
  static InferenceLruCache<InferenceGraphKey, ExecutionGraphPlan>
      cache{kInferenceGraphCacheSize};
  return cache;
}

struct ExecutionGraphRootKey final {
  std::string allocation_label;
  ScalarType dtype{kFloat};
  bool persistent{true};
};

bool operator==(
    const ExecutionGraphRootKey& lhs,
    const ExecutionGraphRootKey& rhs) {
  return lhs.allocation_label == rhs.allocation_label &&
      lhs.dtype == rhs.dtype && lhs.persistent == rhs.persistent;
}

InferenceLruCache<ExecutionGraphRootKey, ExecutionGraphRoot>&
execution_graph_root_cache() {
  static InferenceLruCache<ExecutionGraphRootKey, ExecutionGraphRoot>
      cache{kInferenceGraphCacheSize};
  return cache;
}

std::string format_size_vector_key(const std::vector<int64_t>& sizes) {
  std::string key = "[";
  for (size_t idx = 0u; idx < sizes.size(); ++idx) {
    if (idx > 0u) {
      key += "x";
    }
    key += std::to_string(sizes[idx]);
  }
  key += "]";
  return key;
}

std::string format_optional_size_vector_key(
    const std::optional<std::vector<int64_t>>& sizes) {
  return sizes.has_value() ? format_size_vector_key(*sizes) : "none";
}

std::string format_attention_kernel_family_key(
    const VulkanAttentionKernelFamily family) {
  return std::to_string(static_cast<int>(family));
}

std::string format_scalar_type_key(const ScalarType dtype) {
  return std::to_string(static_cast<int>(dtype));
}

std::string format_execution_layout_key(
    const api::ExecutionLayout execution_layout) {
  return std::to_string(static_cast<int>(execution_layout));
}

std::string format_memory_layout_key(
    const api::GPUMemoryLayout memory_layout) {
  return std::to_string(static_cast<int>(memory_layout));
}

std::string format_storage_type_key(const api::StorageType storage_type) {
  return std::to_string(static_cast<int>(storage_type));
}

std::string format_bool_key(const bool value) {
  return value ? "1" : "0";
}

std::string format_optional_scratch_spec_key(
    const std::optional<VulkanScratchArenaSpec>& scratch_spec) {
  if (!scratch_spec.has_value()) {
    return "none";
  }
  return std::to_string(scratch_spec->num_bytes) + "@" +
      std::to_string(scratch_spec->alignment) + "." +
      format_scalar_type_key(scratch_spec->dtype) + "." +
      format_execution_layout_key(scratch_spec->execution_layout) + "." +
      format_memory_layout_key(scratch_spec->memory_layout) + "." +
      format_storage_type_key(scratch_spec->storage_type) + "." +
      format_bool_key(scratch_spec->persistent);
}

std::string format_optional_kv_cache_spec_key(
    const std::optional<VulkanKVCacheSpec>& cache_spec) {
  if (!cache_spec.has_value()) {
    return "none";
  }
  return format_size_vector_key(cache_spec->sizes) + "." +
      std::to_string(cache_spec->sequence_dim) + "." +
      format_scalar_type_key(cache_spec->dtype) + "." +
      format_execution_layout_key(cache_spec->execution_layout) + "." +
      format_memory_layout_key(cache_spec->memory_layout) + "." +
      format_storage_type_key(cache_spec->storage_type) + "." +
      format_bool_key(cache_spec->persistent);
}

} // namespace

struct InferenceGraph::State final {
  VulkanInferenceGraphKind kind_{VulkanInferenceGraphKind::VisionBackbone};
  std::string allocation_label_;
  ScalarType dtype_{kFloat};
  std::optional<ScratchArena> shared_scratch_arena_;
  size_t shared_scratch_bytes_{0u};
  size_t planned_shared_scratch_bytes_{0u};
  bool persistent_{true};
  mutable std::mutex mutex_;

  State(
      const VulkanInferenceGraphKind kind,
      std::string allocation_label,
      const ScalarType dtype,
      const bool persistent)
      : kind_(kind),
        allocation_label_(std::move(allocation_label)),
        dtype_(dtype),
        persistent_(persistent) {}
};

struct InferenceReplay::State final {
  VulkanInferenceGraphKind kind_{VulkanInferenceGraphKind::VisionBackbone};
  std::string allocation_label_;
  ScalarType dtype_{kFloat};
  bool persistent_{true};
  std::optional<api::CommandBuffer> command_buffer_;
  std::vector<api::VulkanBuffer> retained_buffers_;
  std::vector<api::VulkanImage> retained_images_;
  mutable std::mutex mutex_;

  State(
      const VulkanInferenceGraphKind kind,
      std::string allocation_label,
      const ScalarType dtype,
      const bool persistent)
      : kind_(kind),
        allocation_label_(std::move(allocation_label)),
        dtype_(dtype),
        persistent_(persistent) {}

  void release_retained_resources() {
    if (retained_buffers_.empty() && retained_images_.empty()) {
      return;
    }

    api::Context* const context = api::context();
    if (!context) {
      retained_buffers_.clear();
      retained_images_.clear();
      return;
    }

    for (api::VulkanBuffer& buffer : retained_buffers_) {
      context->register_buffer_cleanup(buffer);
    }
    retained_buffers_.clear();

    for (api::VulkanImage& image : retained_images_) {
      context->register_image_cleanup(image);
    }
    retained_images_.clear();
  }

  ~State() {
    release_retained_resources();
  }
};

struct ExecutionGraphTensorSlots::State final {
  std::vector<Tensor> tensors_;
  std::vector<std::optional<Tensor>> optional_tensors_;

  State(
      std::vector<Tensor> tensors,
      std::vector<std::optional<Tensor>> optional_tensors)
      : tensors_(std::move(tensors)),
        optional_tensors_(std::move(optional_tensors)) {}
};

struct ExecutionGraphProgramSlots::State final {
  std::vector<ExecutionGraphProgramHandle> programs_;

  explicit State(std::vector<ExecutionGraphProgramHandle> programs)
      : programs_(std::move(programs)) {}
};

struct ExecutionGraphReplay::State final {
  InferenceReplay replay_;
  ExecutionGraphTensorSlots tensor_slots_;
  ExecutionGraphProgramSlots program_slots_;

  State(
      InferenceReplay replay,
      ExecutionGraphTensorSlots tensor_slots,
      ExecutionGraphProgramSlots program_slots)
      : replay_(std::move(replay)),
        tensor_slots_(std::move(tensor_slots)),
        program_slots_(std::move(program_slots)) {}
};

struct ExecutionGraphReplayBundle::State final {
  InferenceReplay replay_;
  std::vector<ExecutionGraphReplayStep> steps_;
  std::shared_ptr<std::vector<Tensor>> tensor_slots_;

  State(
      InferenceReplay replay,
      std::vector<ExecutionGraphReplayStep> steps,
      std::shared_ptr<std::vector<Tensor>> tensor_slots)
      : replay_(std::move(replay)),
        steps_(std::move(steps)),
        tensor_slots_(tensor_slots ? std::move(tensor_slots)
                                   : std::make_shared<std::vector<Tensor>>()) {}
};

struct ExecutionGraphPlan::State final {
  struct ProgramEntry final {
    std::string phase_key;
    ExecutionGraphProgramHandle program;
  };

  struct ReplayEntry final {
    std::string phase_key;
    ExecutionGraphReplay replay;
  };

  InferenceGraph graph_;
  std::vector<ProgramEntry> programs_;
  std::vector<ReplayEntry> replays_;
  mutable std::mutex mutex_;

  explicit State(InferenceGraph graph) : graph_(std::move(graph)) {}
};

struct ExecutionGraphRoot::State final {
  struct PhasePlanEntry final {
    VulkanInferenceGraphKind kind{VulkanInferenceGraphKind::VisionBackbone};
    std::string phase_key;
    ExecutionGraphPlan plan;
  };

  struct BundleEntry final {
    std::string bundle_key;
    ExecutionGraphReplayBundle bundle;
  };

  std::string allocation_label_;
  ScalarType dtype_{kFloat};
  bool persistent_{true};
  std::vector<PhasePlanEntry> phase_plans_;
  std::vector<BundleEntry> bundles_;
  mutable std::mutex mutex_;

  State(std::string allocation_label, ScalarType dtype, bool persistent)
      : allocation_label_(std::move(allocation_label)),
        dtype_(dtype),
        persistent_(persistent) {}
};

struct AttentionRuntimeInferenceGraph::State final {
  ExecutionGraphPlan plan_;

  explicit State(ExecutionGraphPlan plan) : plan_(std::move(plan)) {}
};

struct VisionBackboneInferenceGraph::State final {
  ExecutionGraphPlan plan_;

  explicit State(ExecutionGraphPlan plan) : plan_(std::move(plan)) {}
};

struct VisionDecoderInferenceGraph::State final {
  ExecutionGraphPlan plan_;

  explicit State(ExecutionGraphPlan plan) : plan_(std::move(plan)) {}
};

const char* inference_graph_kind_name(const VulkanInferenceGraphKind kind) {
  switch (kind) {
    case VulkanInferenceGraphKind::AttentionRuntime:
      return "AttentionRuntime";
    case VulkanInferenceGraphKind::VisionBackbone:
      return "VisionBackbone";
    case VulkanInferenceGraphKind::VisionDecoder:
      return "VisionDecoder";
    case VulkanInferenceGraphKind::ExecutionGraphBundle:
      return "ExecutionGraphBundle";
  }
  return "VisionBackbone";
}

namespace {

AttentionRuntimeProgram& expect_attention_runtime_program(
    ExecutionGraphProgramHandle& handle) {
  auto* program = std::get_if<AttentionRuntimeProgram>(&handle);
  TORCH_INTERNAL_ASSERT(
      program != nullptr,
      "ExecutionGraphReplay expected an AttentionRuntimeProgram slot");
  return *program;
}

const AttentionRuntimeProgram& expect_attention_runtime_program(
    const ExecutionGraphProgramHandle& handle) {
  const auto* program = std::get_if<AttentionRuntimeProgram>(&handle);
  TORCH_INTERNAL_ASSERT(
      program != nullptr,
      "ExecutionGraphReplay expected an AttentionRuntimeProgram slot");
  return *program;
}

VisionDecoderProgram& expect_vision_decoder_program(
    ExecutionGraphProgramHandle& handle) {
  auto* program = std::get_if<VisionDecoderProgram>(&handle);
  TORCH_INTERNAL_ASSERT(
      program != nullptr,
      "ExecutionGraphReplay expected a VisionDecoderProgram slot");
  return *program;
}

const VisionDecoderProgram& expect_vision_decoder_program(
    const ExecutionGraphProgramHandle& handle) {
  const auto* program = std::get_if<VisionDecoderProgram>(&handle);
  TORCH_INTERNAL_ASSERT(
      program != nullptr,
      "ExecutionGraphReplay expected a VisionDecoderProgram slot");
  return *program;
}

VisionBackboneProgram& expect_vision_backbone_program(
    ExecutionGraphProgramHandle& handle) {
  auto* program = std::get_if<VisionBackboneProgram>(&handle);
  TORCH_INTERNAL_ASSERT(
      program != nullptr,
      "ExecutionGraphReplay expected a VisionBackboneProgram slot");
  return *program;
}

const VisionBackboneProgram& expect_vision_backbone_program(
    const ExecutionGraphProgramHandle& handle) {
  const auto* program = std::get_if<VisionBackboneProgram>(&handle);
  TORCH_INTERNAL_ASSERT(
      program != nullptr,
      "ExecutionGraphReplay expected a VisionBackboneProgram slot");
  return *program;
}

ExecutionGraphReplay make_execution_graph_replay(
    const std::string& allocation_label,
    const VulkanInferenceGraphKind kind,
    const ScalarType dtype,
    const bool persistent,
    std::vector<Tensor> tensors,
    std::vector<std::optional<Tensor>> optional_tensors,
    std::vector<ExecutionGraphProgramHandle> programs) {
  return ExecutionGraphReplay{
      std::make_shared<ExecutionGraphReplay::State>(
          lookup_or_create_labeled_inference_replay(
              allocation_label,
              kind,
              dtype,
              persistent),
          ExecutionGraphTensorSlots{
              std::make_shared<ExecutionGraphTensorSlots::State>(
                  std::move(tensors), std::move(optional_tensors))},
          ExecutionGraphProgramSlots{
              std::make_shared<ExecutionGraphProgramSlots::State>(
                  std::move(programs))})};
}

template <typename Graph, typename State>
Graph make_typed_inference_graph_from_plan(
    ExecutionGraphPlan plan) {
  return Graph{std::make_shared<State>(std::move(plan))};
}

std::string phase_plan_label(
    const std::string& allocation_label,
    const std::string& phase_key) {
  return phase_key.empty() ? allocation_label
                           : allocation_label + "." + phase_key;
}

std::string phase_replay_label(
    const std::string& allocation_label,
    const char* replay_suffix,
    const std::string& phase_key) {
  std::string label = allocation_label.empty() ? std::string(replay_suffix)
                                               : allocation_label + replay_suffix;
  if (phase_key.empty()) {
    return label;
  }

  label += ".phase.";
  label.reserve(label.size() + phase_key.size());
  for (const char ch : phase_key) {
    const bool safe_char =
        (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
        (ch >= '0' && ch <= '9') || ch == '.' || ch == '_' || ch == '-';
    label.push_back(safe_char ? ch : '_');
  }
  return label;
}

template <typename Graph, typename State>
Graph lookup_or_create_typed_inference_graph(
    const std::string& allocation_label,
    const VulkanInferenceGraphKind kind,
    const ScalarType dtype,
    const bool persistent) {
  const std::string resolved_label = allocation_label.empty()
      ? default_inference_graph_label(kind)
      : allocation_label;
  auto root = lookup_or_create_labeled_execution_graph_root(
      resolved_label, dtype, persistent);
  return make_typed_inference_graph_from_plan<Graph, State>(
      root.lookup_or_create_phase_plan(kind));
}

} // namespace

ExecutionGraphReplayStep make_execution_graph_replay_step(
    ExecutionGraphReplay replay,
    std::function<void()> record_step) {
  TORCH_INTERNAL_ASSERT(
      replay.defined(),
      "ExecutionGraphReplayStep requires a defined replay");
  TORCH_INTERNAL_ASSERT(
      static_cast<bool>(record_step),
      "ExecutionGraphReplayStep requires a defined record step");
  return ExecutionGraphReplayStep{
      std::move(replay),
      std::move(record_step),
  };
}

ExecutionGraphReplayBundle make_execution_graph_replay_bundle(
    const std::string& allocation_label,
    const ScalarType dtype,
    const bool persistent,
    std::vector<ExecutionGraphReplayStep> steps,
    std::shared_ptr<std::vector<Tensor>> tensor_slots) {
  TORCH_INTERNAL_ASSERT(
      !steps.empty(),
      "ExecutionGraphReplayBundle requires at least one step");
  for (const auto& step : steps) {
    TORCH_INTERNAL_ASSERT(
        step.replay.defined(),
        "ExecutionGraphReplayBundle step requires a defined replay");
    TORCH_INTERNAL_ASSERT(
        static_cast<bool>(step.record_step),
        "ExecutionGraphReplayBundle step requires a record step");
  }
  return ExecutionGraphReplayBundle{
      std::make_shared<ExecutionGraphReplayBundle::State>(
          lookup_or_create_labeled_inference_replay(
              allocation_label,
              VulkanInferenceGraphKind::ExecutionGraphBundle,
              dtype,
              persistent),
          std::move(steps),
          std::move(tensor_slots))};
}

bool InferenceGraph::defined() const {
  return static_cast<bool>(state_);
}

bool InferenceReplay::defined() const {
  return static_cast<bool>(state_);
}

bool ExecutionGraphTensorSlots::defined() const {
  return static_cast<bool>(state_);
}

bool ExecutionGraphProgramSlots::defined() const {
  return static_cast<bool>(state_);
}

bool ExecutionGraphReplay::defined() const {
  return static_cast<bool>(state_);
}

bool ExecutionGraphReplayBundle::defined() const {
  return static_cast<bool>(state_);
}

bool ExecutionGraphPlan::defined() const {
  return static_cast<bool>(state_);
}

bool ExecutionGraphRoot::defined() const {
  return static_cast<bool>(state_);
}

VulkanInferenceGraphKind InferenceGraph::kind() const {
  return state_ ? state_->kind_ : VulkanInferenceGraphKind::VisionBackbone;
}

VulkanInferenceGraphKind InferenceReplay::kind() const {
  return state_ ? state_->kind_ : VulkanInferenceGraphKind::VisionBackbone;
}

VulkanInferenceGraphKind ExecutionGraphPlan::kind() const {
  return state_ ? state_->graph_.kind() : VulkanInferenceGraphKind::VisionBackbone;
}

const std::string& InferenceGraph::allocation_label() const {
  static const std::string empty;
  return state_ ? state_->allocation_label_ : empty;
}

const std::string& InferenceReplay::allocation_label() const {
  static const std::string empty;
  return state_ ? state_->allocation_label_ : empty;
}

const std::string& ExecutionGraphPlan::allocation_label() const {
  static const std::string empty;
  return state_ ? state_->graph_.allocation_label() : empty;
}

const std::string& ExecutionGraphRoot::allocation_label() const {
  static const std::string empty;
  return state_ ? state_->allocation_label_ : empty;
}

bool InferenceReplay::recorded() const {
  if (!state_) {
    return false;
  }
  std::lock_guard<std::mutex> lock(state_->mutex_);
  return state_->command_buffer_.has_value();
}

void InferenceReplay::record(const std::function<void()>& recorder) const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined InferenceReplay");
  std::lock_guard<std::mutex> lock(state_->mutex_);
  if (state_->command_buffer_.has_value()) {
    return;
  }

  api::Context* const context = api::context();
  api::CommandBuffer command_buffer =
      context->acquire_persistent_command_buffer();
  std::vector<api::VulkanBuffer> retained_buffers;
  std::vector<api::VulkanImage> retained_images;
  {
    api::Context::ScopedExternalCommandRecording recording_scope(
        *context, command_buffer);
    recorder();
  }
  context->take_external_recording_cleanup_resources(
      retained_buffers, retained_images);
  command_buffer.end();
  state_->command_buffer_.emplace(std::move(command_buffer));
  state_->retained_buffers_ = std::move(retained_buffers);
  state_->retained_images_ = std::move(retained_images);
  log_inference_replay_event(
      state_->kind_, "record", state_->allocation_label_, identity());
}

void InferenceReplay::submit(
    VkFence fence_handle,
    const bool final_use) const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined InferenceReplay");
  std::lock_guard<std::mutex> lock(state_->mutex_);
  TORCH_INTERNAL_ASSERT(
      state_->command_buffer_.has_value(),
      "Attempted to submit an unrecorded InferenceReplay");
  api::context()->submit_prepared_command_buffer(
      *state_->command_buffer_, fence_handle, final_use);
  log_inference_replay_event(
      state_->kind_, "submit", state_->allocation_label_, identity());
  if (final_use) {
    state_->command_buffer_.reset();
    state_->release_retained_resources();
  }
}

void InferenceGraph::note_shared_scratch_requirement(
    const size_t requested_bytes,
    const uint32_t alignment,
    const bool persistent) const {
  if (!state_ || requested_bytes == 0u) {
    return;
  }

  std::lock_guard<std::mutex> lock(state_->mutex_);
  state_->planned_shared_scratch_bytes_ =
      std::max(state_->planned_shared_scratch_bytes_, requested_bytes);
  (void)alignment;
  (void)persistent;
}

std::optional<ScratchArena> InferenceGraph::ensure_shared_scratch(
    const size_t requested_bytes,
    const uint32_t alignment,
    const bool persistent) const {
  if (!state_ || requested_bytes == 0u) {
    return std::nullopt;
  }

  std::lock_guard<std::mutex> lock(state_->mutex_);
  const size_t required_bytes = std::max(
      requested_bytes, state_->planned_shared_scratch_bytes_);
  if (
      state_->shared_scratch_arena_.has_value() &&
      state_->shared_scratch_bytes_ >= required_bytes) {
    return state_->shared_scratch_arena_;
  }

  const size_t num_bytes =
      std::max(state_->shared_scratch_bytes_, required_bytes);
  state_->shared_scratch_arena_ = lookup_or_create_labeled_scratch_arena(
      state_->allocation_label_ + ".scratch",
      VulkanScratchArenaSpec{
          kByte,
          num_bytes,
          alignment,
          api::ExecutionLayout::BUFFER_DIRECT,
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
          api::StorageType::BUFFER,
          persistent,
      });
  state_->shared_scratch_bytes_ = num_bytes;
  log_inference_graph_event(
      state_->kind_,
      "scratch_resize",
      state_->allocation_label_,
      identity(),
      num_bytes);
  return state_->shared_scratch_arena_;
}

const void* InferenceGraph::identity() const {
  return state_.get();
}

const void* InferenceReplay::identity() const {
  return state_.get();
}

size_t ExecutionGraphTensorSlots::tensor_count() const {
  return state_ ? state_->tensors_.size() : 0u;
}

size_t ExecutionGraphTensorSlots::optional_tensor_count() const {
  return state_ ? state_->optional_tensors_.size() : 0u;
}

Tensor& ExecutionGraphTensorSlots::tensor(const size_t idx) {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphTensorSlots");
  TORCH_INTERNAL_ASSERT(
      idx < state_->tensors_.size(),
      "ExecutionGraphTensorSlots tensor index out of range");
  return state_->tensors_.at(idx);
}

const Tensor& ExecutionGraphTensorSlots::tensor(const size_t idx) const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphTensorSlots");
  TORCH_INTERNAL_ASSERT(
      idx < state_->tensors_.size(),
      "ExecutionGraphTensorSlots tensor index out of range");
  return state_->tensors_.at(idx);
}

std::optional<Tensor>& ExecutionGraphTensorSlots::optional_tensor(
    const size_t idx) {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphTensorSlots");
  TORCH_INTERNAL_ASSERT(
      idx < state_->optional_tensors_.size(),
      "ExecutionGraphTensorSlots optional tensor index out of range");
  return state_->optional_tensors_.at(idx);
}

const std::optional<Tensor>& ExecutionGraphTensorSlots::optional_tensor(
    const size_t idx) const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphTensorSlots");
  TORCH_INTERNAL_ASSERT(
      idx < state_->optional_tensors_.size(),
      "ExecutionGraphTensorSlots optional tensor index out of range");
  return state_->optional_tensors_.at(idx);
}

const void* ExecutionGraphTensorSlots::identity() const {
  return state_.get();
}

size_t ExecutionGraphProgramSlots::size() const {
  return state_ ? state_->programs_.size() : 0u;
}

ExecutionGraphProgramHandle& ExecutionGraphProgramSlots::program(
    const size_t idx) {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphProgramSlots");
  TORCH_INTERNAL_ASSERT(
      idx < state_->programs_.size(),
      "ExecutionGraphProgramSlots program index out of range");
  return state_->programs_.at(idx);
}

const ExecutionGraphProgramHandle& ExecutionGraphProgramSlots::program(
    const size_t idx) const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphProgramSlots");
  TORCH_INTERNAL_ASSERT(
      idx < state_->programs_.size(),
      "ExecutionGraphProgramSlots program index out of range");
  return state_->programs_.at(idx);
}

const void* ExecutionGraphProgramSlots::identity() const {
  return state_.get();
}

bool ExecutionGraphReplay::recorded() const {
  return state_ && state_->replay_.recorded();
}

const InferenceReplay& ExecutionGraphReplay::replay() const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphReplay");
  return state_->replay_;
}

ExecutionGraphTensorSlots& ExecutionGraphReplay::tensor_slots() {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphReplay");
  return state_->tensor_slots_;
}

const ExecutionGraphTensorSlots& ExecutionGraphReplay::tensor_slots() const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphReplay");
  return state_->tensor_slots_;
}

ExecutionGraphProgramSlots& ExecutionGraphReplay::program_slots() {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphReplay");
  return state_->program_slots_;
}

const ExecutionGraphProgramSlots& ExecutionGraphReplay::program_slots() const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphReplay");
  return state_->program_slots_;
}

const void* ExecutionGraphReplay::identity() const {
  return state_.get();
}

size_t ExecutionGraphReplayBundle::size() const {
  return state_ ? state_->steps_.size() : 0u;
}

bool ExecutionGraphReplayBundle::recorded() const {
  if (!state_) {
    return false;
  }
  if (state_->replay_.defined()) {
    return state_->replay_.recorded();
  }
  if (state_->steps_.empty()) {
    return false;
  }
  return std::all_of(
      state_->steps_.cbegin(),
      state_->steps_.cend(),
      [](const ExecutionGraphReplayStep& step) {
        return step.replay.recorded();
      });
}

void ExecutionGraphReplayBundle::record() const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphReplayBundle");
  TORCH_INTERNAL_ASSERT(
      state_->replay_.defined(),
      "ExecutionGraphReplayBundle does not own a bundle replay");
  TORCH_INTERNAL_ASSERT(
      !state_->steps_.empty(),
      "ExecutionGraphReplayBundle does not define bundle record steps");
  const auto state = state_;
  state_->replay_.record([state]() {
    for (const auto& step : state->steps_) {
      step.record_step();
    }
  });
}

ExecutionGraphReplay& ExecutionGraphReplayBundle::replay(const size_t idx) {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphReplayBundle");
  TORCH_INTERNAL_ASSERT(
      idx < state_->steps_.size(),
      "ExecutionGraphReplayBundle replay index out of range");
  return state_->steps_.at(idx).replay;
}

const ExecutionGraphReplay& ExecutionGraphReplayBundle::replay(
    const size_t idx) const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphReplayBundle");
  TORCH_INTERNAL_ASSERT(
      idx < state_->steps_.size(),
      "ExecutionGraphReplayBundle replay index out of range");
  return state_->steps_.at(idx).replay;
}

size_t ExecutionGraphReplayBundle::tensor_slot_count() const {
  return state_ ? state_->tensor_slots_->size() : 0u;
}

Tensor& ExecutionGraphReplayBundle::tensor_slot(const size_t idx) {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphReplayBundle");
  TORCH_INTERNAL_ASSERT(
      idx < state_->tensor_slots_->size(),
      "ExecutionGraphReplayBundle tensor slot index out of range");
  return state_->tensor_slots_->at(idx);
}

const Tensor& ExecutionGraphReplayBundle::tensor_slot(const size_t idx) const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphReplayBundle");
  TORCH_INTERNAL_ASSERT(
      idx < state_->tensor_slots_->size(),
      "ExecutionGraphReplayBundle tensor slot index out of range");
  return state_->tensor_slots_->at(idx);
}

void ExecutionGraphReplayBundle::submit(
    VkFence fence_handle,
    const bool final_use) const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphReplayBundle");
  if (state_->replay_.defined() && state_->replay_.recorded()) {
    state_->replay_.submit(fence_handle, final_use);
    return;
  }
  TORCH_INTERNAL_ASSERT(
      !state_->steps_.empty(),
      "ExecutionGraphReplayBundle must contain at least one replay");
  for (size_t idx = 0u; idx < state_->steps_.size(); ++idx) {
    const bool is_last = (idx + 1u) == state_->steps_.size();
    state_->steps_.at(idx).replay.replay().submit(
        is_last ? fence_handle : VK_NULL_HANDLE, final_use);
  }
}

const void* ExecutionGraphReplayBundle::identity() const {
  return state_.get();
}

void ExecutionGraphPlan::note_shared_scratch_requirement(
    const size_t requested_bytes,
    const uint32_t alignment,
    const bool persistent) const {
  if (!state_) {
    return;
  }
  state_->graph_.note_shared_scratch_requirement(
      requested_bytes, alignment, persistent);
}

std::optional<ScratchArena> ExecutionGraphPlan::ensure_shared_scratch(
    const size_t requested_bytes,
    const uint32_t alignment,
    const bool persistent) const {
  if (!state_) {
    return std::nullopt;
  }
  return state_->graph_.ensure_shared_scratch(
      requested_bytes, alignment, persistent);
}

ExecutionGraphProgramHandle ExecutionGraphPlan::lookup_or_create_program(
    const std::string& phase_key,
    const std::function<ExecutionGraphProgramHandle()>& builder) const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphPlan");
  {
    std::lock_guard<std::mutex> lock(state_->mutex_);
    for (const State::ProgramEntry& entry : state_->programs_) {
      if (entry.phase_key == phase_key) {
        log_execution_graph_plan_event(
            kind(), "program_hit", allocation_label(), identity(), phase_key.c_str());
        return entry.program;
      }
    }
  }

  ExecutionGraphProgramHandle created = builder();
  TORCH_INTERNAL_ASSERT(
      !std::holds_alternative<std::monostate>(created),
      "ExecutionGraphPlan program builder returned an undefined program handle");
  {
    std::lock_guard<std::mutex> lock(state_->mutex_);
    for (const State::ProgramEntry& entry : state_->programs_) {
      if (entry.phase_key == phase_key) {
        log_execution_graph_plan_event(
            kind(), "program_hit", allocation_label(), identity(), phase_key.c_str());
        return entry.program;
      }
    }
    state_->programs_.push_back(State::ProgramEntry{phase_key, created});
  }
  log_execution_graph_plan_event(
      kind(), "program_store", allocation_label(), identity(), phase_key.c_str());
  return created;
}

ExecutionGraphReplay ExecutionGraphPlan::lookup_or_create_replay(
    const std::string& phase_key,
    const std::function<ExecutionGraphReplay()>& builder) const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphPlan");
  {
    std::lock_guard<std::mutex> lock(state_->mutex_);
    for (const State::ReplayEntry& entry : state_->replays_) {
      if (entry.phase_key == phase_key) {
        log_execution_graph_plan_event(
            kind(), "replay_hit", allocation_label(), identity(), phase_key.c_str());
        return entry.replay;
      }
    }
  }

  ExecutionGraphReplay created = builder();
  TORCH_INTERNAL_ASSERT(
      created.defined(),
      "ExecutionGraphPlan replay builder returned an undefined replay");
  {
    std::lock_guard<std::mutex> lock(state_->mutex_);
    for (const State::ReplayEntry& entry : state_->replays_) {
      if (entry.phase_key == phase_key) {
        log_execution_graph_plan_event(
            kind(), "replay_hit", allocation_label(), identity(), phase_key.c_str());
        return entry.replay;
      }
    }
    state_->replays_.push_back(State::ReplayEntry{phase_key, created});
  }
  log_execution_graph_plan_event(
      kind(), "replay_store", allocation_label(), identity(), phase_key.c_str());
  return created;
}

const void* ExecutionGraphPlan::identity() const {
  return state_.get();
}

ExecutionGraphPlan ExecutionGraphRoot::lookup_or_create_phase_plan(
    const VulkanInferenceGraphKind kind,
    const std::string& phase_key) const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphRoot");
  {
    std::lock_guard<std::mutex> lock(state_->mutex_);
    for (const State::PhasePlanEntry& entry : state_->phase_plans_) {
      if (entry.kind == kind && entry.phase_key == phase_key) {
        log_execution_graph_root_event(
            "phase_hit",
            allocation_label(),
            identity(),
            &kind,
            phase_key.c_str());
        return entry.plan;
      }
    }
  }

  ExecutionGraphPlan created = lookup_or_create_labeled_execution_graph_plan(
      phase_plan_label(allocation_label(), phase_key),
      kind,
      state_->dtype_,
      state_->persistent_);
  {
    std::lock_guard<std::mutex> lock(state_->mutex_);
    for (const State::PhasePlanEntry& entry : state_->phase_plans_) {
      if (entry.kind == kind && entry.phase_key == phase_key) {
        log_execution_graph_root_event(
            "phase_hit",
            allocation_label(),
            identity(),
            &kind,
            phase_key.c_str());
        return entry.plan;
      }
    }
    state_->phase_plans_.push_back(State::PhasePlanEntry{kind, phase_key, created});
  }
  log_execution_graph_root_event(
      "phase_store",
      allocation_label(),
      identity(),
      &kind,
      phase_key.c_str());
  return created;
}

ExecutionGraphReplayBundle ExecutionGraphRoot::lookup_or_create_replay_bundle(
    const std::string& bundle_key,
    const std::function<ExecutionGraphReplayBundle()>& builder) const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphRoot");
  {
    std::lock_guard<std::mutex> lock(state_->mutex_);
    for (const State::BundleEntry& entry : state_->bundles_) {
      if (entry.bundle_key == bundle_key) {
        log_execution_graph_root_event(
            "bundle_hit",
            allocation_label(),
            identity(),
            nullptr,
            bundle_key.c_str());
        return entry.bundle;
      }
    }
  }

  ExecutionGraphReplayBundle created = builder();
  TORCH_INTERNAL_ASSERT(
      created.defined(),
      "ExecutionGraphRoot replay bundle builder returned an undefined bundle");
  {
    std::lock_guard<std::mutex> lock(state_->mutex_);
    for (const State::BundleEntry& entry : state_->bundles_) {
      if (entry.bundle_key == bundle_key) {
        log_execution_graph_root_event(
            "bundle_hit",
            allocation_label(),
            identity(),
            nullptr,
            bundle_key.c_str());
        return entry.bundle;
      }
    }
    state_->bundles_.push_back(State::BundleEntry{bundle_key, created});
  }
  log_execution_graph_root_event(
      "bundle_store",
      allocation_label(),
      identity(),
      nullptr,
      bundle_key.c_str());
  return created;
}

const void* ExecutionGraphRoot::identity() const {
  return state_.get();
}

InferenceGraph lookup_or_create_labeled_inference_graph(
    const std::string& allocation_label,
    const VulkanInferenceGraphKind kind,
    const ScalarType dtype,
    const bool persistent) {
  const InferenceGraphKey query{
      kind,
      allocation_label.empty() ? default_inference_graph_label(kind)
                               : allocation_label,
      dtype,
      persistent,
  };
  if (const auto cached = inference_graph_cache().lookup(
          query,
          [](const InferenceGraphKey& lhs, const InferenceGraphKey& rhs) {
            return lhs == rhs;
          })) {
    log_inference_graph_event(
        query.kind, "hit", query.allocation_label, cached->identity());
    return *cached;
  }

  InferenceGraph created{std::make_shared<InferenceGraph::State>(
      query.kind, query.allocation_label, query.dtype, query.persistent)};
  inference_graph_cache().store(
      query,
      created,
      [](const InferenceGraphKey& lhs, const InferenceGraphKey& rhs) {
        return lhs == rhs;
      });
  log_inference_graph_event(
      query.kind, "store", query.allocation_label, created.identity());
  return created;
}

InferenceReplay lookup_or_create_labeled_inference_replay(
    const std::string& allocation_label,
    const VulkanInferenceGraphKind kind,
    const ScalarType dtype,
    const bool persistent) {
  const InferenceGraphKey query{
      kind,
      allocation_label.empty() ? default_inference_graph_label(kind) + ".replay"
                               : allocation_label,
      dtype,
      persistent,
  };
  if (const auto cached = inference_replay_cache().lookup(
          query,
          [](const InferenceGraphKey& lhs, const InferenceGraphKey& rhs) {
            return lhs == rhs;
          })) {
    log_inference_replay_event(
        query.kind, "hit", query.allocation_label, cached->identity());
    return *cached;
  }

  InferenceReplay created{std::make_shared<InferenceReplay::State>(
      query.kind, query.allocation_label, query.dtype, query.persistent)};
  inference_replay_cache().store(
      query,
      created,
      [](const InferenceGraphKey& lhs, const InferenceGraphKey& rhs) {
        return lhs == rhs;
      });
  log_inference_replay_event(
      query.kind, "store", query.allocation_label, created.identity());
  return created;
}

ExecutionGraphPlan lookup_or_create_labeled_execution_graph_plan(
    const std::string& allocation_label,
    const VulkanInferenceGraphKind kind,
    const ScalarType dtype,
    const bool persistent) {
  const InferenceGraphKey query{
      kind,
      allocation_label.empty() ? default_inference_graph_label(kind) + ".plan"
                               : allocation_label,
      dtype,
      persistent,
  };
  if (const auto cached = execution_graph_plan_cache().lookup(
          query,
          [](const InferenceGraphKey& lhs, const InferenceGraphKey& rhs) {
            return lhs == rhs;
          })) {
    log_execution_graph_plan_event(
        query.kind, "hit", query.allocation_label, cached->identity());
    return *cached;
  }

ExecutionGraphPlan created{std::make_shared<ExecutionGraphPlan::State>(
      lookup_or_create_labeled_inference_graph(
          query.allocation_label, query.kind, query.dtype, query.persistent))};
  execution_graph_plan_cache().store(
      query,
      created,
      [](const InferenceGraphKey& lhs, const InferenceGraphKey& rhs) {
        return lhs == rhs;
      });
  log_execution_graph_plan_event(
      query.kind, "store", query.allocation_label, created.identity());
  return created;
}

ExecutionGraphRoot lookup_or_create_labeled_execution_graph_root(
    const std::string& allocation_label,
    const ScalarType dtype,
    const bool persistent) {
  const ExecutionGraphRootKey query{
      allocation_label.empty() ? std::string("execution_graph.root")
                               : allocation_label,
      dtype,
      persistent,
  };
  if (const auto cached = execution_graph_root_cache().lookup(
          query,
          [](const ExecutionGraphRootKey& lhs, const ExecutionGraphRootKey& rhs) {
            return lhs == rhs;
          })) {
    log_execution_graph_root_event(
        "hit", query.allocation_label, cached->identity());
    return *cached;
  }

  ExecutionGraphRoot created{std::make_shared<ExecutionGraphRoot::State>(
      query.allocation_label, query.dtype, query.persistent)};
  execution_graph_root_cache().store(
      query,
      created,
      [](const ExecutionGraphRootKey& lhs, const ExecutionGraphRootKey& rhs) {
        return lhs == rhs;
      });
  log_execution_graph_root_event(
      "store", query.allocation_label, created.identity());
  return created;
}

bool AttentionRuntimeInferenceGraph::defined() const {
  return state_ && state_->plan_.defined();
}

void AttentionRuntimeInferenceGraph::note_shared_scratch_requirement(
    const size_t requested_bytes,
    const uint32_t alignment,
    const bool persistent) const {
  if (!state_) {
    return;
  }
  state_->plan_.note_shared_scratch_requirement(
      requested_bytes, alignment, persistent);
}

std::optional<ScratchArena> AttentionRuntimeInferenceGraph::ensure_shared_scratch(
    const size_t requested_bytes,
    const uint32_t alignment,
    const bool persistent) const {
  if (!state_) {
    return std::nullopt;
  }
  return state_->plan_.ensure_shared_scratch(
      requested_bytes, alignment, persistent);
}

AttentionRuntimeProgram AttentionRuntimeInferenceGraph::lookup_or_create_program(
    const std::string& allocation_label,
    const VulkanAttentionKernelFamily kernel_family,
    const std::optional<VulkanKVCacheSpec>& key_cache_spec,
    const std::optional<VulkanKVCacheSpec>& value_cache_spec,
    const std::optional<VulkanScratchArenaSpec>& scratch_spec,
    const int64_t key_sequence_length,
    const int64_t value_sequence_length,
    const VulkanExecutionProgramPlanningDesc& program_plan) const {
  TORCH_INTERNAL_ASSERT(
      defined(), "Undefined AttentionRuntimeInferenceGraph");
  const std::string phase_key = allocation_label + "|family=" +
      format_attention_kernel_family_key(kernel_family) + "|key_cache=" +
      format_optional_kv_cache_spec_key(key_cache_spec) + "|value_cache=" +
      format_optional_kv_cache_spec_key(value_cache_spec) + "|scratch=" +
      format_optional_scratch_spec_key(scratch_spec);
  AttentionRuntimeProgram program = expect_attention_runtime_program(
      state_->plan_.lookup_or_create_program(
          phase_key,
          [&]() -> ExecutionGraphProgramHandle {
            return lookup_or_create_labeled_attention_runtime_program(
                allocation_label,
                kernel_family,
                key_cache_spec,
                value_cache_spec,
                scratch_spec,
                key_sequence_length,
                value_sequence_length,
                program_plan);
          }));
  program.set_sequence_lengths(key_sequence_length, value_sequence_length);
  return program;
}

AttentionRuntimeInferenceReplay
AttentionRuntimeInferenceGraph::lookup_or_create_replay(
    const std::string& allocation_label,
    IntArrayRef query_sizes,
    IntArrayRef key_sizes,
    IntArrayRef value_sizes,
    const VulkanAttentionKernelFamily kernel_family,
    const std::optional<VulkanKVCacheSpec>& key_cache_spec,
    const std::optional<VulkanKVCacheSpec>& value_cache_spec,
    const std::optional<VulkanScratchArenaSpec>& scratch_spec,
    const int64_t key_sequence_length,
    const int64_t value_sequence_length,
    const VulkanExecutionProgramPlanningDesc& program_plan) const {
  TORCH_INTERNAL_ASSERT(
      defined(), "Undefined AttentionRuntimeInferenceGraph");
  const std::vector<int64_t> query_sizes_vec = query_sizes.vec();
  const std::vector<int64_t> key_sizes_vec = key_sizes.vec();
  const std::vector<int64_t> value_sizes_vec = value_sizes.vec();
  const std::string phase_key = allocation_label + "|family=" +
      format_attention_kernel_family_key(kernel_family) + "|query=" +
      format_size_vector_key(query_sizes_vec) + "|key=" +
      format_size_vector_key(key_sizes_vec) + "|value=" +
      format_size_vector_key(value_sizes_vec) + "|key_cache=" +
      format_optional_kv_cache_spec_key(key_cache_spec) + "|value_cache=" +
      format_optional_kv_cache_spec_key(value_cache_spec) + "|scratch=" +
      format_optional_scratch_spec_key(scratch_spec);

  ExecutionGraphReplay graph_replay = state_->plan_.lookup_or_create_replay(
      phase_key,
      [&]() -> ExecutionGraphReplay {
        AttentionRuntimeProgram program = lookup_or_create_program(
            allocation_label + ".replay.program",
            kernel_family,
            key_cache_spec,
            value_cache_spec,
            scratch_spec,
            key_sequence_length,
            value_sequence_length,
            program_plan);
        std::vector<Tensor> tensors;
        tensors.reserve(4u);
        tensors.push_back(ops::utils::create_buffer_tensor(
            query_sizes, kFloat, program_plan.persistent));
        tensors.push_back(ops::utils::create_buffer_tensor(
            key_sizes, kFloat, program_plan.persistent));
        tensors.push_back(ops::utils::create_buffer_tensor(
            value_sizes, kFloat, program_plan.persistent));
        tensors.push_back(ops::utils::create_buffer_tensor(
            {
                query_sizes_vec.at(0),
                query_sizes_vec.at(1),
                value_sizes_vec.at(2),
            },
            kFloat,
            program_plan.persistent));
        std::vector<ExecutionGraphProgramHandle> programs;
        programs.emplace_back(program);
        return make_execution_graph_replay(
            phase_replay_label(allocation_label, ".replay", phase_key),
            VulkanInferenceGraphKind::AttentionRuntime,
            kFloat,
            program_plan.persistent,
            std::move(tensors),
            std::vector<std::optional<Tensor>>{},
            std::move(programs));
      });

  AttentionRuntimeInferenceReplay replay{std::move(graph_replay)};
  replay.program().set_sequence_lengths(
      key_sequence_length, value_sequence_length);
  return replay;
}

const void* AttentionRuntimeInferenceGraph::identity() const {
  return state_ ? state_->plan_.identity() : nullptr;
}

bool AttentionRuntimeInferenceReplay::defined() const {
  return graph_replay_.defined();
}

bool AttentionRuntimeInferenceReplay::recorded() const {
  return graph_replay_.recorded();
}

const InferenceReplay& AttentionRuntimeInferenceReplay::replay() const {
  return graph_replay_.replay();
}

const ExecutionGraphReplay& AttentionRuntimeInferenceReplay::graph_replay()
    const {
  return graph_replay_;
}

ExecutionGraphReplayStep AttentionRuntimeInferenceReplay::phase_step(
    std::function<void()> record_step) const {
  return make_execution_graph_replay_step(graph_replay_, std::move(record_step));
}

const AttentionRuntimeProgram& AttentionRuntimeInferenceReplay::program() const {
  return expect_attention_runtime_program(graph_replay_.program_slots().program(0u));
}

AttentionRuntimeProgram& AttentionRuntimeInferenceReplay::program() {
  return expect_attention_runtime_program(graph_replay_.program_slots().program(0u));
}

Tensor& AttentionRuntimeInferenceReplay::query_slot() {
  return graph_replay_.tensor_slots().tensor(0u);
}

Tensor& AttentionRuntimeInferenceReplay::key_slot() {
  return graph_replay_.tensor_slots().tensor(1u);
}

Tensor& AttentionRuntimeInferenceReplay::value_slot() {
  return graph_replay_.tensor_slots().tensor(2u);
}

Tensor& AttentionRuntimeInferenceReplay::output_slot() {
  return graph_replay_.tensor_slots().tensor(3u);
}

const void* AttentionRuntimeInferenceReplay::identity() const {
  return graph_replay_.identity();
}

bool VisionBackboneInferenceGraph::defined() const {
  return state_ && state_->plan_.defined();
}

void VisionBackboneInferenceGraph::note_shared_scratch_requirement(
    const size_t requested_bytes,
    const uint32_t alignment,
    const bool persistent) const {
  if (!state_) {
    return;
  }
  state_->plan_.note_shared_scratch_requirement(
      requested_bytes, alignment, persistent);
}

std::optional<ScratchArena> VisionBackboneInferenceGraph::ensure_shared_scratch(
    const size_t requested_bytes,
    const uint32_t alignment,
    const bool persistent) const {
  if (!state_) {
    return std::nullopt;
  }
  return state_->plan_.ensure_shared_scratch(
      requested_bytes, alignment, persistent);
}

VisionBackboneProgram VisionBackboneInferenceGraph::lookup_or_create_program(
    const std::string& allocation_label,
    const ScalarType dtype,
    const int64_t batch_size,
    const int64_t token_count,
    const int64_t embed_dim,
    const int64_t hidden_dim,
    const int64_t num_heads,
    const VulkanExecutionProgramPlanningDesc& program_plan) const {
  TORCH_INTERNAL_ASSERT(
      defined(), "Undefined VisionBackboneInferenceGraph");
  const std::string phase_key = allocation_label + "|dtype=" +
      std::to_string(static_cast<int>(dtype)) + "|batch=" +
      std::to_string(batch_size) + "|tokens=" + std::to_string(token_count) +
      "|embed=" + std::to_string(embed_dim) + "|hidden=" +
      std::to_string(hidden_dim) + "|heads=" + std::to_string(num_heads);
  return expect_vision_backbone_program(state_->plan_.lookup_or_create_program(
      phase_key,
      [&]() -> ExecutionGraphProgramHandle {
        return lookup_or_create_labeled_vision_backbone_program(
            allocation_label,
            dtype,
            batch_size,
            token_count,
            embed_dim,
            hidden_dim,
            num_heads,
            std::nullopt,
            program_plan);
      }));
}

VisionBackboneInferenceReplay VisionBackboneInferenceGraph::lookup_or_create_replay(
    const std::string& allocation_label,
    IntArrayRef input_sizes,
    const int64_t token_count,
    const int64_t embed_dim,
    const int64_t hidden_dim,
    const int64_t num_heads,
    const VulkanExecutionProgramPlanningDesc& program_plan) const {
  TORCH_INTERNAL_ASSERT(
      defined(), "Undefined VisionBackboneInferenceGraph");
  const std::vector<int64_t> input_sizes_vec = input_sizes.vec();
  const std::string phase_key = allocation_label + "|batch=" +
      std::to_string(input_sizes.size() == 2 ? 1 : input_sizes[0]) + "|input=" +
      format_size_vector_key(input_sizes_vec) + "|tokens=" + std::to_string(token_count) +
      "|embed=" + std::to_string(embed_dim) + "|hidden=" +
      std::to_string(hidden_dim) + "|heads=" + std::to_string(num_heads);

  ExecutionGraphReplay graph_replay = state_->plan_.lookup_or_create_replay(
      phase_key,
      [&]() -> ExecutionGraphReplay {
        VisionBackboneProgram program = lookup_or_create_program(
            allocation_label + ".replay.program",
            kFloat,
            input_sizes.size() == 2 ? 1 : input_sizes[0],
            token_count,
            embed_dim,
            hidden_dim,
            num_heads,
            program_plan);
        std::vector<Tensor> tensors;
        tensors.reserve(2u);
        tensors.push_back(ops::utils::create_buffer_tensor(
            input_sizes,
            kFloat,
            program_plan.persistent));
        tensors.push_back(ops::utils::create_buffer_tensor(
            input_sizes,
            kFloat,
            program_plan.persistent));
        std::vector<ExecutionGraphProgramHandle> programs;
        programs.emplace_back(program);
        return make_execution_graph_replay(
            phase_replay_label(allocation_label, ".replay", phase_key),
            VulkanInferenceGraphKind::VisionBackbone,
            kFloat,
            program_plan.persistent,
            std::move(tensors),
            std::vector<std::optional<Tensor>>{},
            std::move(programs));
      });

  return VisionBackboneInferenceReplay{std::move(graph_replay)};
}

const void* VisionBackboneInferenceGraph::identity() const {
  return state_ ? state_->plan_.identity() : nullptr;
}

bool VisionBackboneInferenceReplay::defined() const {
  return graph_replay_.defined();
}

bool VisionBackboneInferenceReplay::recorded() const {
  return graph_replay_.recorded();
}

const InferenceReplay& VisionBackboneInferenceReplay::replay() const {
  return graph_replay_.replay();
}

const ExecutionGraphReplay& VisionBackboneInferenceReplay::graph_replay() const {
  return graph_replay_;
}

ExecutionGraphReplayStep VisionBackboneInferenceReplay::phase_step(
    std::function<void()> record_step) const {
  return make_execution_graph_replay_step(graph_replay_, std::move(record_step));
}

const VisionBackboneProgram& VisionBackboneInferenceReplay::program() const {
  return expect_vision_backbone_program(graph_replay_.program_slots().program(0u));
}

VisionBackboneProgram& VisionBackboneInferenceReplay::program() {
  return expect_vision_backbone_program(graph_replay_.program_slots().program(0u));
}

Tensor& VisionBackboneInferenceReplay::input_slot() {
  return graph_replay_.tensor_slots().tensor(0u);
}

Tensor& VisionBackboneInferenceReplay::output_slot() {
  return graph_replay_.tensor_slots().tensor(1u);
}

const void* VisionBackboneInferenceReplay::identity() const {
  return graph_replay_.identity();
}

bool VisionDecoderInferenceGraph::defined() const {
  return state_ && state_->plan_.defined();
}

void VisionDecoderInferenceGraph::note_shared_scratch_requirement(
    const size_t requested_bytes,
    const uint32_t alignment,
    const bool persistent) const {
  if (!state_) {
    return;
  }
  state_->plan_.note_shared_scratch_requirement(
      requested_bytes, alignment, persistent);
}

std::optional<ScratchArena> VisionDecoderInferenceGraph::ensure_shared_scratch(
    const size_t requested_bytes,
    const uint32_t alignment,
    const bool persistent) const {
  if (!state_) {
    return std::nullopt;
  }
  return state_->plan_.ensure_shared_scratch(
      requested_bytes, alignment, persistent);
}

VisionDecoderProgram VisionDecoderInferenceGraph::lookup_or_create_program(
    const std::string& allocation_label,
    IntArrayRef input_sizes,
    const std::optional<std::vector<int64_t>>& skip_sizes,
    IntArrayRef target_sizes,
    const int64_t out_channels,
    const bool allocate_intermediate_outputs,
    const VulkanExecutionProgramPlanningDesc& program_plan) const {
  TORCH_INTERNAL_ASSERT(
      defined(), "Undefined VisionDecoderInferenceGraph");
  const std::string phase_key = allocation_label + "|input=" +
      format_size_vector_key(input_sizes.vec()) + "|skip=" +
      format_optional_size_vector_key(skip_sizes) + "|target=" +
      format_size_vector_key(target_sizes.vec()) + "|out=" +
      std::to_string(out_channels) + "|allocate=" +
      std::to_string(static_cast<int>(allocate_intermediate_outputs));
  return expect_vision_decoder_program(state_->plan_.lookup_or_create_program(
      phase_key,
      [&]() -> ExecutionGraphProgramHandle {
        return lookup_or_create_labeled_vision_decoder_program(
            allocation_label,
            input_sizes,
            skip_sizes,
            target_sizes,
            out_channels,
            std::nullopt,
            program_plan,
            allocate_intermediate_outputs);
      }));
}

VisionDecoderInferenceReplay VisionDecoderInferenceGraph::lookup_or_create_replay(
    const std::string& allocation_label,
    IntArrayRef input_sizes,
    const std::optional<std::vector<int64_t>>& skip_sizes,
    IntArrayRef target_sizes,
    const int64_t out_channels,
    const VulkanExecutionProgramPlanningDesc& program_plan) const {
  TORCH_INTERNAL_ASSERT(
      defined(), "Undefined VisionDecoderInferenceGraph");
  const std::string phase_key = allocation_label + "|input=" +
      format_size_vector_key(input_sizes.vec()) + "|skip=" +
      format_optional_size_vector_key(skip_sizes) + "|target=" +
      format_size_vector_key(target_sizes.vec()) + "|out=" +
      std::to_string(out_channels);

  ExecutionGraphReplay graph_replay = state_->plan_.lookup_or_create_replay(
      phase_key,
      [&]() -> ExecutionGraphReplay {
        VisionDecoderProgram program = lookup_or_create_program(
            allocation_label + ".replay.program",
            input_sizes,
            skip_sizes,
            target_sizes,
            out_channels,
            /*allocate_intermediate_outputs=*/true,
            program_plan);
        std::vector<Tensor> tensors;
        tensors.reserve(1u);
        tensors.push_back(ops::utils::create_buffer_tensor(
            input_sizes, kFloat, program_plan.persistent));
        std::vector<std::optional<Tensor>> optional_tensors;
        optional_tensors.push_back(skip_sizes.has_value()
            ? std::optional<Tensor>(ops::utils::create_buffer_tensor(
                  *skip_sizes, kFloat, program_plan.persistent))
            : std::nullopt);
        std::vector<ExecutionGraphProgramHandle> programs;
        programs.emplace_back(program);
        return make_execution_graph_replay(
            phase_replay_label(allocation_label, ".replay", phase_key),
            VulkanInferenceGraphKind::VisionDecoder,
            kFloat,
            program_plan.persistent,
            std::move(tensors),
            std::move(optional_tensors),
            std::move(programs));
      });

  return VisionDecoderInferenceReplay{std::move(graph_replay)};
}

VisionDecoderHeadInferenceReplay
VisionDecoderInferenceGraph::lookup_or_create_head_replay(
    const std::string& allocation_label,
    IntArrayRef layer1_sizes,
    IntArrayRef layer2_sizes,
    IntArrayRef layer3_sizes,
    IntArrayRef layer4_sizes,
    IntArrayRef output_sizes,
    const int64_t output_conv1_channels,
    const int64_t output_conv2_channels,
    const int64_t final_channels,
    const VulkanExecutionProgramPlanningDesc& program_plan) const {
  TORCH_INTERNAL_ASSERT(
      defined(), "Undefined VisionDecoderInferenceGraph");
  const std::vector<int64_t> layer1_sizes_vec = layer1_sizes.vec();
  const std::vector<int64_t> layer2_sizes_vec = layer2_sizes.vec();
  const std::vector<int64_t> layer3_sizes_vec = layer3_sizes.vec();
  const std::vector<int64_t> layer4_sizes_vec = layer4_sizes.vec();
  const std::vector<int64_t> output_sizes_vec = output_sizes.vec();
  const std::string phase_key = allocation_label + "|layer1=" +
      format_size_vector_key(layer1_sizes_vec) + "|layer2=" +
      format_size_vector_key(layer2_sizes_vec) + "|layer3=" +
      format_size_vector_key(layer3_sizes_vec) + "|layer4=" +
      format_size_vector_key(layer4_sizes_vec) + "|output=" +
      format_size_vector_key(output_sizes_vec) + "|conv1=" +
      std::to_string(output_conv1_channels) + "|conv2=" +
      std::to_string(output_conv2_channels) + "|final=" +
      std::to_string(final_channels);

  ExecutionGraphReplay graph_replay = state_->plan_.lookup_or_create_replay(
      phase_key,
      [&]() -> ExecutionGraphReplay {
        const std::vector<int64_t> refinenet4_target_sizes{
            layer3_sizes_vec.at(2), layer3_sizes_vec.at(3)};
        const std::vector<int64_t> refinenet3_target_sizes{
            layer2_sizes_vec.at(2), layer2_sizes_vec.at(3)};
        const std::vector<int64_t> refinenet2_target_sizes{
            layer1_sizes_vec.at(2), layer1_sizes_vec.at(3)};
        const std::vector<int64_t> refinenet1_target_sizes{
            layer1_sizes_vec.at(2) * 2, layer1_sizes_vec.at(3) * 2};

        VisionDecoderProgram refinenet4_program = lookup_or_create_program(
            allocation_label + ".refinenet4.program",
            layer4_sizes,
            std::nullopt,
            refinenet4_target_sizes,
            layer3_sizes_vec.at(1),
            /*allocate_intermediate_outputs=*/true,
            program_plan);
        VisionDecoderProgram refinenet3_program = lookup_or_create_program(
            allocation_label + ".refinenet3.program",
            layer3_sizes,
            layer3_sizes_vec,
            refinenet3_target_sizes,
            layer2_sizes_vec.at(1),
            /*allocate_intermediate_outputs=*/true,
            program_plan);
        VisionDecoderProgram refinenet2_program = lookup_or_create_program(
            allocation_label + ".refinenet2.program",
            layer2_sizes,
            layer2_sizes_vec,
            refinenet2_target_sizes,
            layer1_sizes_vec.at(1),
            /*allocate_intermediate_outputs=*/true,
            program_plan);
        VisionDecoderProgram refinenet1_program = lookup_or_create_program(
            allocation_label + ".refinenet1.program",
            layer1_sizes,
            layer1_sizes_vec,
            refinenet1_target_sizes,
            layer1_sizes_vec.at(1),
            /*allocate_intermediate_outputs=*/true,
            program_plan);
        const int64_t batch_size = output_sizes_vec.at(0);
        const int64_t output_height = output_sizes_vec.at(2);
        const int64_t output_width = output_sizes_vec.at(3);

        std::vector<Tensor> tensors;
        tensors.reserve(10u);
        tensors.push_back(ops::utils::create_buffer_tensor(
            layer1_sizes, kFloat, program_plan.persistent));
        tensors.push_back(ops::utils::create_buffer_tensor(
            layer2_sizes, kFloat, program_plan.persistent));
        tensors.push_back(ops::utils::create_buffer_tensor(
            layer3_sizes, kFloat, program_plan.persistent));
        tensors.push_back(ops::utils::create_buffer_tensor(
            layer4_sizes, kFloat, program_plan.persistent));
        tensors.push_back(ops::utils::create_buffer_tensor(
            {
                batch_size,
                output_conv1_channels,
                refinenet1_target_sizes.at(0),
                refinenet1_target_sizes.at(1),
            },
            kFloat,
            program_plan.persistent));
        tensors.push_back(ops::utils::create_buffer_tensor(
            {
                batch_size,
                output_conv1_channels,
                output_height,
                output_width,
            },
            kFloat,
            program_plan.persistent));
        tensors.push_back(ops::utils::create_buffer_tensor(
            {batch_size, output_conv2_channels, output_height, output_width},
            kFloat,
            program_plan.persistent));
        tensors.push_back(ops::utils::create_buffer_tensor(
            {batch_size, output_conv2_channels, output_height, output_width},
            kFloat,
            program_plan.persistent));
        tensors.push_back(ops::utils::create_buffer_tensor(
            {batch_size, final_channels, output_height, output_width},
            kFloat,
            program_plan.persistent));
        tensors.push_back(ops::utils::create_buffer_tensor(
            output_sizes, kFloat, program_plan.persistent));
        std::vector<ExecutionGraphProgramHandle> programs;
        programs.emplace_back(refinenet4_program);
        programs.emplace_back(refinenet3_program);
        programs.emplace_back(refinenet2_program);
        programs.emplace_back(refinenet1_program);
        return make_execution_graph_replay(
            phase_replay_label(allocation_label, ".head.replay", phase_key),
            VulkanInferenceGraphKind::VisionDecoder,
            kFloat,
            program_plan.persistent,
            std::move(tensors),
            std::vector<std::optional<Tensor>>{},
            std::move(programs));
      });

  return VisionDecoderHeadInferenceReplay{std::move(graph_replay)};
}

const void* VisionDecoderInferenceGraph::identity() const {
  return state_ ? state_->plan_.identity() : nullptr;
}

bool VisionDecoderInferenceReplay::defined() const {
  return graph_replay_.defined();
}

bool VisionDecoderInferenceReplay::recorded() const {
  return graph_replay_.recorded();
}

const InferenceReplay& VisionDecoderInferenceReplay::replay() const {
  return graph_replay_.replay();
}

const ExecutionGraphReplay& VisionDecoderInferenceReplay::graph_replay() const {
  return graph_replay_;
}

ExecutionGraphReplayStep VisionDecoderInferenceReplay::phase_step(
    std::function<void()> record_step) const {
  return make_execution_graph_replay_step(graph_replay_, std::move(record_step));
}

const VisionDecoderProgram& VisionDecoderInferenceReplay::program() const {
  return expect_vision_decoder_program(graph_replay_.program_slots().program(0u));
}

VisionDecoderProgram& VisionDecoderInferenceReplay::program() {
  return expect_vision_decoder_program(graph_replay_.program_slots().program(0u));
}

Tensor& VisionDecoderInferenceReplay::input_slot() {
  return graph_replay_.tensor_slots().tensor(0u);
}

std::optional<Tensor>& VisionDecoderInferenceReplay::skip_slot() {
  return graph_replay_.tensor_slots().optional_tensor(0u);
}

const std::optional<Tensor>& VisionDecoderInferenceReplay::skip_slot() const {
  static const std::optional<Tensor> empty;
  return defined() ? graph_replay_.tensor_slots().optional_tensor(0u) : empty;
}

Tensor& VisionDecoderInferenceReplay::output_slot() {
  return program().out_conv_output();
}

const void* VisionDecoderInferenceReplay::identity() const {
  return graph_replay_.identity();
}

bool VisionDecoderHeadInferenceReplay::defined() const {
  return graph_replay_.defined();
}

bool VisionDecoderHeadInferenceReplay::recorded() const {
  return graph_replay_.recorded();
}

const InferenceReplay& VisionDecoderHeadInferenceReplay::replay() const {
  return graph_replay_.replay();
}

const ExecutionGraphReplay& VisionDecoderHeadInferenceReplay::graph_replay()
    const {
  return graph_replay_;
}

ExecutionGraphReplayStep VisionDecoderHeadInferenceReplay::phase_step(
    std::function<void()> record_step) const {
  return make_execution_graph_replay_step(graph_replay_, std::move(record_step));
}

Tensor& VisionDecoderHeadInferenceReplay::layer1_slot() {
  return graph_replay_.tensor_slots().tensor(0u);
}

Tensor& VisionDecoderHeadInferenceReplay::layer2_slot() {
  return graph_replay_.tensor_slots().tensor(1u);
}

Tensor& VisionDecoderHeadInferenceReplay::layer3_slot() {
  return graph_replay_.tensor_slots().tensor(2u);
}

Tensor& VisionDecoderHeadInferenceReplay::layer4_slot() {
  return graph_replay_.tensor_slots().tensor(3u);
}

const VisionDecoderProgram&
VisionDecoderHeadInferenceReplay::refinenet4_program() const {
  return expect_vision_decoder_program(graph_replay_.program_slots().program(0u));
}

VisionDecoderProgram& VisionDecoderHeadInferenceReplay::refinenet4_program() {
  return expect_vision_decoder_program(graph_replay_.program_slots().program(0u));
}

const VisionDecoderProgram&
VisionDecoderHeadInferenceReplay::refinenet3_program() const {
  return expect_vision_decoder_program(graph_replay_.program_slots().program(1u));
}

VisionDecoderProgram& VisionDecoderHeadInferenceReplay::refinenet3_program() {
  return expect_vision_decoder_program(graph_replay_.program_slots().program(1u));
}

const VisionDecoderProgram&
VisionDecoderHeadInferenceReplay::refinenet2_program() const {
  return expect_vision_decoder_program(graph_replay_.program_slots().program(2u));
}

VisionDecoderProgram& VisionDecoderHeadInferenceReplay::refinenet2_program() {
  return expect_vision_decoder_program(graph_replay_.program_slots().program(2u));
}

const VisionDecoderProgram&
VisionDecoderHeadInferenceReplay::refinenet1_program() const {
  return expect_vision_decoder_program(graph_replay_.program_slots().program(3u));
}

VisionDecoderProgram& VisionDecoderHeadInferenceReplay::refinenet1_program() {
  return expect_vision_decoder_program(graph_replay_.program_slots().program(3u));
}

Tensor& VisionDecoderHeadInferenceReplay::output_conv1_output() {
  return graph_replay_.tensor_slots().tensor(4u);
}

Tensor& VisionDecoderHeadInferenceReplay::upsample_output() {
  return graph_replay_.tensor_slots().tensor(5u);
}

Tensor& VisionDecoderHeadInferenceReplay::output_conv2_conv1_output() {
  return graph_replay_.tensor_slots().tensor(6u);
}

Tensor& VisionDecoderHeadInferenceReplay::output_conv2_relu1_output() {
  return graph_replay_.tensor_slots().tensor(7u);
}

Tensor& VisionDecoderHeadInferenceReplay::output_conv2_conv2_output() {
  return graph_replay_.tensor_slots().tensor(8u);
}

Tensor& VisionDecoderHeadInferenceReplay::output_slot() {
  return graph_replay_.tensor_slots().tensor(9u);
}

const void* VisionDecoderHeadInferenceReplay::identity() const {
  return graph_replay_.identity();
}

AttentionRuntimeInferenceGraph
lookup_or_create_labeled_attention_runtime_inference_graph(
    const std::string& allocation_label,
    const ScalarType dtype,
    const bool persistent) {
  return lookup_or_create_typed_inference_graph<
      AttentionRuntimeInferenceGraph,
      AttentionRuntimeInferenceGraph::State>(
      allocation_label,
      VulkanInferenceGraphKind::AttentionRuntime,
      dtype,
      persistent);
}

VisionBackboneInferenceGraph
lookup_or_create_labeled_vision_backbone_inference_graph(
    const std::string& allocation_label,
    const ScalarType dtype,
    const bool persistent) {
  return lookup_or_create_typed_inference_graph<
      VisionBackboneInferenceGraph,
      VisionBackboneInferenceGraph::State>(
      allocation_label,
      VulkanInferenceGraphKind::VisionBackbone,
      dtype,
      persistent);
}

VisionDecoderInferenceGraph lookup_or_create_labeled_vision_decoder_inference_graph(
    const std::string& allocation_label,
    const ScalarType dtype,
    const bool persistent) {
  return lookup_or_create_typed_inference_graph<
      VisionDecoderInferenceGraph,
      VisionDecoderInferenceGraph::State>(
      allocation_label,
      VulkanInferenceGraphKind::VisionDecoder,
      dtype,
      persistent);
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
