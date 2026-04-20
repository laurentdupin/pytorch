#include <ATen/native/vulkan/planning/InferenceGraphs.h>

#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/ops/InferenceCache.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/planning/CompiledSession.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

constexpr size_t kInferenceGraphCacheSize = 32u;

template <typename T>
void hash_combine(size_t& seed, const T& value) {
  seed ^= std::hash<T>{}(value) + size_t{0x9e3779b97f4a7c15ull} + (seed << 6u) +
      (seed >> 2u);
}

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

size_t hash_inference_graph_key(const InferenceGraphKey& key) {
  size_t seed = 0u;
  hash_combine(seed, static_cast<int>(key.kind));
  hash_combine(seed, key.allocation_label);
  hash_combine(seed, static_cast<int>(key.dtype));
  hash_combine(seed, key.persistent);
  return seed;
}

InferenceLruCache<InferenceGraphKey, InferenceGraph>& inference_graph_cache() {
  static auto* cache =
      new InferenceLruCache<InferenceGraphKey, InferenceGraph>{
          kInferenceGraphCacheSize};
  return *cache;
}

InferenceLruCache<InferenceGraphKey, InferenceReplay>& inference_replay_cache() {
  static auto* cache =
      new InferenceLruCache<InferenceGraphKey, InferenceReplay>{
          kInferenceGraphCacheSize};
  return *cache;
}

InferenceLruCache<InferenceGraphKey, ExecutionGraphPlan>&
execution_graph_plan_cache() {
  static auto* cache =
      new InferenceLruCache<InferenceGraphKey, ExecutionGraphPlan>{
          kInferenceGraphCacheSize};
  return *cache;
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

size_t hash_execution_graph_root_key(const ExecutionGraphRootKey& key) {
  size_t seed = 0u;
  hash_combine(seed, key.allocation_label);
  hash_combine(seed, static_cast<int>(key.dtype));
  hash_combine(seed, key.persistent);
  return seed;
}

InferenceLruCache<ExecutionGraphRootKey, ExecutionGraphRoot>&
execution_graph_root_cache() {
  static auto* cache =
      new InferenceLruCache<ExecutionGraphRootKey, ExecutionGraphRoot>{
          kInferenceGraphCacheSize};
  return *cache;
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
  InferenceGraph graph_;
  std::unordered_map<std::string, ExecutionGraphProgramHandle> programs_;
  std::unordered_map<std::string, ExecutionGraphReplay> replays_;
  std::unordered_map<
      std::string,
      std::shared_future<ExecutionGraphProgramHandle>>
      in_flight_programs_;
  std::unordered_map<std::string, std::shared_future<ExecutionGraphReplay>>
      in_flight_replays_;
  mutable std::mutex mutex_;

  explicit State(InferenceGraph graph) : graph_(std::move(graph)) {}
};

struct ExecutionGraphRoot::State final {
  struct PhasePlanKey final {
    VulkanInferenceGraphKind kind{VulkanInferenceGraphKind::VisionBackbone};
    std::string phase_key;
  };

  struct PhasePlanKeyHash final {
    size_t operator()(const PhasePlanKey& key) const {
      size_t seed = 0u;
      hash_combine(seed, static_cast<int>(key.kind));
      hash_combine(seed, key.phase_key);
      return seed;
    }
  };

  struct PhasePlanKeyEqual final {
    bool operator()(const PhasePlanKey& lhs, const PhasePlanKey& rhs) const {
      return lhs.kind == rhs.kind && lhs.phase_key == rhs.phase_key;
    }
  };

  std::string allocation_label_;
  ScalarType dtype_{kFloat};
  bool persistent_{true};
  std::unordered_map<
      PhasePlanKey,
      ExecutionGraphPlan,
      PhasePlanKeyHash,
      PhasePlanKeyEqual>
      phase_plans_;
  std::unordered_map<std::string, ExecutionGraphReplayBundle> bundles_;
  std::unordered_map<
      PhasePlanKey,
      std::shared_future<ExecutionGraphPlan>,
      PhasePlanKeyHash,
      PhasePlanKeyEqual>
      in_flight_phase_plans_;
  std::unordered_map<std::string, std::shared_future<ExecutionGraphReplayBundle>>
      in_flight_bundles_;
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

ExecutionGraphReplay make_execution_graph_replay_impl(
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

ExecutionGraphReplay make_execution_graph_replay(
    const std::string& allocation_label,
    const VulkanInferenceGraphKind kind,
    const ScalarType dtype,
    const bool persistent,
    std::vector<Tensor> tensors,
    std::vector<std::optional<Tensor>> optional_tensors,
    std::vector<ExecutionGraphProgramHandle> programs) {
  return make_execution_graph_replay_impl(
      allocation_label,
      kind,
      dtype,
      persistent,
      std::move(tensors),
      std::move(optional_tensors),
      std::move(programs));
}

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
      *state_->command_buffer_,
      fence_handle,
      final_use,
      state_->allocation_label_.c_str());
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

void ExecutionGraphReplayBundle::warmup() const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphReplayBundle");
  TORCH_INTERNAL_ASSERT(
      !state_->steps_.empty(),
      "ExecutionGraphReplayBundle does not define warmup steps");
  for (const auto& step : state_->steps_) {
    step.record_step();
  }
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

void ExecutionGraphReplayBundle::record_steps_individually() const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphReplayBundle");
  TORCH_INTERNAL_ASSERT(
      !state_->steps_.empty(),
      "ExecutionGraphReplayBundle does not define replay steps");
  for (const auto& step : state_->steps_) {
    if (!step.replay.recorded()) {
      step.replay.replay().record(step.record_step);
    }
  }
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
  std::shared_future<ExecutionGraphProgramHandle> pending;
  std::optional<std::promise<ExecutionGraphProgramHandle>> owner_promise;
  {
    std::lock_guard<std::mutex> lock(state_->mutex_);
    if (const auto found = state_->programs_.find(phase_key);
        found != state_->programs_.end()) {
      log_execution_graph_plan_event(
          kind(), "program_hit", allocation_label(), identity(), phase_key.c_str());
      return found->second;
    }
    if (const auto in_flight = state_->in_flight_programs_.find(phase_key);
        in_flight != state_->in_flight_programs_.end()) {
      pending = in_flight->second;
    } else {
      std::promise<ExecutionGraphProgramHandle> promise;
      pending = promise.get_future().share();
      state_->in_flight_programs_.emplace(phase_key, pending);
      owner_promise.emplace(std::move(promise));
    }
  }

  if (!owner_promise.has_value()) {
    log_execution_graph_plan_event(
        kind(),
        "program_wait",
        allocation_label(),
        identity(),
        phase_key.c_str());
    ExecutionGraphProgramHandle awaited = pending.get();
    log_execution_graph_plan_event(
        kind(), "program_hit", allocation_label(), identity(), phase_key.c_str());
    return awaited;
  }

  ExecutionGraphProgramHandle created;
  try {
    log_execution_graph_plan_event(
        kind(),
        "program_build_start",
        allocation_label(),
        identity(),
        phase_key.c_str());
    created = builder();
    TORCH_INTERNAL_ASSERT(
        !std::holds_alternative<std::monostate>(created),
        "ExecutionGraphPlan program builder returned an undefined program handle");
    {
      std::lock_guard<std::mutex> lock(state_->mutex_);
      auto [it, inserted] = state_->programs_.emplace(phase_key, created);
      if (!inserted) {
        created = it->second;
      }
      state_->in_flight_programs_.erase(phase_key);
    }
    owner_promise->set_value(created);
    log_execution_graph_plan_event(
        kind(),
        "program_build_finish",
        allocation_label(),
        identity(),
        phase_key.c_str());
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(state_->mutex_);
      state_->in_flight_programs_.erase(phase_key);
    }
    owner_promise->set_exception(std::current_exception());
    throw;
  }

  log_execution_graph_plan_event(
      kind(), "program_store", allocation_label(), identity(), phase_key.c_str());
  return created;
}

ExecutionGraphReplay ExecutionGraphPlan::lookup_or_create_replay(
    const std::string& phase_key,
    const std::function<ExecutionGraphReplay()>& builder) const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphPlan");
  std::shared_future<ExecutionGraphReplay> pending;
  std::optional<std::promise<ExecutionGraphReplay>> owner_promise;
  {
    std::lock_guard<std::mutex> lock(state_->mutex_);
    if (const auto found = state_->replays_.find(phase_key);
        found != state_->replays_.end()) {
      log_execution_graph_plan_event(
          kind(), "replay_hit", allocation_label(), identity(), phase_key.c_str());
      return found->second;
    }
    if (const auto in_flight = state_->in_flight_replays_.find(phase_key);
        in_flight != state_->in_flight_replays_.end()) {
      pending = in_flight->second;
    } else {
      std::promise<ExecutionGraphReplay> promise;
      pending = promise.get_future().share();
      state_->in_flight_replays_.emplace(phase_key, pending);
      owner_promise.emplace(std::move(promise));
    }
  }

  if (!owner_promise.has_value()) {
    log_execution_graph_plan_event(
        kind(),
        "replay_wait",
        allocation_label(),
        identity(),
        phase_key.c_str());
    ExecutionGraphReplay awaited = pending.get();
    log_execution_graph_plan_event(
        kind(), "replay_hit", allocation_label(), identity(), phase_key.c_str());
    return awaited;
  }

  ExecutionGraphReplay created;
  try {
    log_execution_graph_plan_event(
        kind(),
        "replay_build_start",
        allocation_label(),
        identity(),
        phase_key.c_str());
    created = builder();
    TORCH_INTERNAL_ASSERT(
        created.defined(),
        "ExecutionGraphPlan replay builder returned an undefined replay");
    {
      std::lock_guard<std::mutex> lock(state_->mutex_);
      auto [it, inserted] = state_->replays_.emplace(phase_key, created);
      if (!inserted) {
        created = it->second;
      }
      state_->in_flight_replays_.erase(phase_key);
    }
    owner_promise->set_value(created);
    log_execution_graph_plan_event(
        kind(),
        "replay_build_finish",
        allocation_label(),
        identity(),
        phase_key.c_str());
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(state_->mutex_);
      state_->in_flight_replays_.erase(phase_key);
    }
    owner_promise->set_exception(std::current_exception());
    throw;
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
  const State::PhasePlanKey key{kind, phase_key};
  std::shared_future<ExecutionGraphPlan> pending;
  std::optional<std::promise<ExecutionGraphPlan>> owner_promise;
  {
    std::lock_guard<std::mutex> lock(state_->mutex_);
    if (const auto found = state_->phase_plans_.find(key);
        found != state_->phase_plans_.end()) {
      log_execution_graph_root_event(
          "phase_hit",
          allocation_label(),
          identity(),
          &kind,
          phase_key.c_str());
      return found->second;
    }
    if (const auto in_flight = state_->in_flight_phase_plans_.find(key);
        in_flight != state_->in_flight_phase_plans_.end()) {
      pending = in_flight->second;
    } else {
      std::promise<ExecutionGraphPlan> promise;
      pending = promise.get_future().share();
      state_->in_flight_phase_plans_.emplace(key, pending);
      owner_promise.emplace(std::move(promise));
    }
  }

  if (!owner_promise.has_value()) {
    log_execution_graph_root_event(
        "phase_wait",
        allocation_label(),
        identity(),
        &kind,
        phase_key.c_str());
    ExecutionGraphPlan awaited = pending.get();
    log_execution_graph_root_event(
        "phase_hit",
        allocation_label(),
        identity(),
        &kind,
        phase_key.c_str());
    return awaited;
  }

  ExecutionGraphPlan created;
  try {
    log_execution_graph_root_event(
        "phase_build_start",
        allocation_label(),
        identity(),
        &kind,
        phase_key.c_str());
    created = lookup_or_create_labeled_execution_graph_plan(
        phase_plan_label(allocation_label(), phase_key),
        kind,
        state_->dtype_,
        state_->persistent_);
    {
      std::lock_guard<std::mutex> lock(state_->mutex_);
      auto [it, inserted] = state_->phase_plans_.emplace(key, created);
      if (!inserted) {
        created = it->second;
      }
      state_->in_flight_phase_plans_.erase(key);
    }
    owner_promise->set_value(created);
    log_execution_graph_root_event(
        "phase_build_finish",
        allocation_label(),
        identity(),
        &kind,
        phase_key.c_str());
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(state_->mutex_);
      state_->in_flight_phase_plans_.erase(key);
    }
    owner_promise->set_exception(std::current_exception());
    throw;
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
  std::shared_future<ExecutionGraphReplayBundle> pending;
  std::optional<std::promise<ExecutionGraphReplayBundle>> owner_promise;
  {
    std::lock_guard<std::mutex> lock(state_->mutex_);
    if (const auto found = state_->bundles_.find(bundle_key);
        found != state_->bundles_.end()) {
      log_execution_graph_root_event(
          "bundle_hit",
          allocation_label(),
          identity(),
          nullptr,
          bundle_key.c_str());
      return found->second;
    }
    if (const auto in_flight = state_->in_flight_bundles_.find(bundle_key);
        in_flight != state_->in_flight_bundles_.end()) {
      pending = in_flight->second;
    } else {
      std::promise<ExecutionGraphReplayBundle> promise;
      pending = promise.get_future().share();
      state_->in_flight_bundles_.emplace(bundle_key, pending);
      owner_promise.emplace(std::move(promise));
    }
  }

  if (!owner_promise.has_value()) {
    log_execution_graph_root_event(
        "bundle_wait",
        allocation_label(),
        identity(),
        nullptr,
        bundle_key.c_str());
    ExecutionGraphReplayBundle awaited = pending.get();
    log_execution_graph_root_event(
        "bundle_hit",
        allocation_label(),
        identity(),
        nullptr,
        bundle_key.c_str());
    return awaited;
  }

  ExecutionGraphReplayBundle created;
  try {
    log_execution_graph_root_event(
        "bundle_build_start",
        allocation_label(),
        identity(),
        nullptr,
        bundle_key.c_str());
    created = builder();
    TORCH_INTERNAL_ASSERT(
        created.defined(),
        "ExecutionGraphRoot replay bundle builder returned an undefined bundle");
    {
      std::lock_guard<std::mutex> lock(state_->mutex_);
      auto [it, inserted] = state_->bundles_.emplace(bundle_key, created);
      if (!inserted) {
        created = it->second;
      }
      state_->in_flight_bundles_.erase(bundle_key);
    }
    owner_promise->set_value(created);
    log_execution_graph_root_event(
        "bundle_build_finish",
        allocation_label(),
        identity(),
        nullptr,
        bundle_key.c_str());
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(state_->mutex_);
      state_->in_flight_bundles_.erase(bundle_key);
    }
    owner_promise->set_exception(std::current_exception());
    throw;
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
          hash_inference_graph_key,
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
      hash_inference_graph_key,
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
          hash_inference_graph_key,
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
      hash_inference_graph_key,
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
          hash_inference_graph_key,
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
      hash_inference_graph_key,
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
          hash_execution_graph_root_key,
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
      hash_execution_graph_root_key,
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
        std::vector<Tensor> tensors;
        tensors.reserve(5u);
        tensors.push_back(ops::utils::create_buffer_tensor(
            layer1_sizes, kFloat, program_plan.persistent));
        tensors.push_back(ops::utils::create_buffer_tensor(
            layer2_sizes, kFloat, program_plan.persistent));
        tensors.push_back(ops::utils::create_buffer_tensor(
            layer3_sizes, kFloat, program_plan.persistent));
        tensors.push_back(ops::utils::create_buffer_tensor(
            layer4_sizes, kFloat, program_plan.persistent));
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

Tensor& VisionDecoderHeadInferenceReplay::output_slot() {
  return graph_replay_.tensor_slots().tensor(4u);
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

struct VulkanCompiledSession::State final {
  VulkanCompiledSessionKey key;
  VulkanBackendIR ir;
  VulkanGlobalLayoutPlan layout_plan;
  VulkanIRMemoryPlan memory_plan;
  bool executable{false};

  State(
      VulkanCompiledSessionKey key_in,
      VulkanBackendIR ir_in,
      VulkanGlobalLayoutPlan layout_plan_in,
      VulkanIRMemoryPlan memory_plan_in,
      const bool executable_in)
      : key(std::move(key_in)),
        ir(std::move(ir_in)),
        layout_plan(std::move(layout_plan_in)),
        memory_plan(std::move(memory_plan_in)),
        executable(executable_in) {}
};

namespace compiled_session_impl {

constexpr size_t kCompiledSessionCacheSize = 16u;

template <typename T>
void hash_combine_session(size_t& seed, const T& value) {
  seed ^= std::hash<T>{}(value) + size_t{0x9e3779b97f4a7c15ull} + (seed << 6u) +
      (seed >> 2u);
}

void hash_shape(size_t& seed, const std::vector<int64_t>& shape) {
  hash_combine_session(seed, shape.size());
  for (const int64_t value : shape) {
    hash_combine_session(seed, value);
  }
}

void hash_shapes(size_t& seed, const std::vector<std::vector<int64_t>>& shapes) {
  hash_combine_session(seed, shapes.size());
  for (const auto& shape : shapes) {
    hash_shape(seed, shape);
  }
}

std::string shape_key(const std::vector<int64_t>& shape) {
  std::ostringstream out;
  for (size_t idx = 0u; idx < shape.size(); ++idx) {
    if (idx > 0u) {
      out << 'x';
    }
    out << shape[idx];
  }
  return out.str();
}

std::string optional_shape_key(
    const std::optional<std::vector<int64_t>>& shape) {
  return shape.has_value() ? shape_key(*shape) : std::string("none");
}

std::string vector_key(const std::vector<int64_t>& values) {
  std::ostringstream out;
  for (size_t idx = 0u; idx < values.size(); ++idx) {
    if (idx > 0u) {
      out << ',';
    }
    out << values[idx];
  }
  return out.str();
}

int64_t round_up_to_multiple(const int64_t value, const int64_t alignment) {
  if (value <= 0 || alignment <= 1) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

size_t tensor_spec_numel(const VulkanIRTensorSpec& spec) {
  if (spec.padded_sizes.empty()) {
    return 0u;
  }
  size_t numel = 1u;
  for (const int64_t size : spec.padded_sizes) {
    if (size <= 0) {
      return 0u;
    }
    numel *= static_cast<size_t>(size);
  }
  return numel;
}

size_t tensor_spec_nbytes(const VulkanIRTensorSpec& spec) {
  return tensor_spec_numel(spec) * c10::elementSize(spec.dtype);
}

bool requires_dedicated_slot(const VulkanIRTensorSpec& spec) {
  return spec.role == VulkanIRTensorRole::Input ||
      spec.role == VulkanIRTensorRole::Output ||
      spec.role == VulkanIRTensorRole::Constant ||
      spec.external;
}

VulkanIRTensorSpec make_tensor_spec(
    const std::vector<int64_t>& logical_sizes,
    const ScalarType dtype,
    const VulkanIRTensorRole role,
    const bool persistent,
    const bool external) {
  VulkanIRTensorSpec spec;
  spec.dtype = dtype;
  spec.logical_sizes = logical_sizes;
  spec.padded_sizes = logical_sizes;
  spec.role = role;
  spec.persistent = persistent;
  spec.external = external;
  return spec;
}

VulkanIRTensorSpec make_constant_spec(
    const ScalarType dtype,
    const bool persistent) {
  return make_tensor_spec(
      std::vector<int64_t>{},
      dtype,
      VulkanIRTensorRole::Constant,
      persistent,
      false);
}

const std::string& compiled_session_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_COMPILED_SESSION_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool compiled_session_logging_enabled() {
  return !compiled_session_log_path().empty();
}

std::mutex& compiled_session_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

void log_compiled_session_event(
    const char* event,
    const VulkanCompiledSessionKey& key,
    const void* identity,
    const size_t value_count = 0u,
    const size_t op_count = 0u,
    const VulkanGlobalLayoutPlan* layout_plan = nullptr,
    const VulkanIRMemoryPlan* memory_plan = nullptr) {
  if (!compiled_session_logging_enabled()) {
    return;
  }

  std::lock_guard<std::mutex> lock(compiled_session_log_mutex());
  std::ofstream out(compiled_session_log_path(), std::ios::app);
  out << "compiled_session event=" << event << " kind="
      << compiled_session_kind_name(key.kind) << " model_key=" << key.model_key
      << " config=" << key.configuration_key << " dtype="
      << static_cast<int>(key.dtype) << " capability_key="
      << key.capability_key << " persistent=" << (key.persistent ? 1 : 0);
  if (identity) {
    out << " identity=" << identity;
  }
  if (value_count > 0u || op_count > 0u) {
    out << " values=" << value_count << " ops=" << op_count;
  }
  if (layout_plan) {
    out << " storage=buffer"
        << " width_alignment=" << layout_plan->width_alignment
        << " pad_width=" << (layout_plan->pad_width ? 1 : 0)
        << " reason=" << layout_plan->reason;
  }
  if (memory_plan) {
    out << " slots=" << memory_plan->slots.size()
        << " planned_bytes=" << memory_plan->planned_bytes
        << " reusable_bytes=" << memory_plan->reusable_bytes
        << " dedicated_bytes=" << memory_plan->dedicated_bytes
        << " external_bytes=" << memory_plan->external_bytes;
  }
  out << '\n';
}

struct VulkanCompiledSessionCache final {
  std::mutex mutex;
  std::unordered_map<
      VulkanCompiledSessionKey,
      VulkanCompiledSession,
      VulkanCompiledSessionKeyHash>
      sessions;
};

VulkanCompiledSessionCache& compiled_session_cache() {
  static VulkanCompiledSessionCache cache;
  return cache;
}

std::string make_backbone_configuration_key(
    const DepthAnythingV2BackboneStackSessionDesc& desc) {
  std::ostringstream out;
  out << "patch_tokens=" << shape_key(desc.patch_token_sizes)
      << "|blocks=" << desc.backbone_block_count
      << "|capture=" << vector_key(desc.capture_indices)
      << "|hidden=" << vector_key(desc.block_hidden_dims)
      << "|heads=" << vector_key(desc.block_num_heads)
      << "|norm_shape=" << optional_shape_key(desc.normalized_shape);
  return out.str();
}

std::string make_decoder_configuration_key(
    const DepthAnythingV2DecoderPreprocessHeadSessionDesc& desc) {
  std::ostringstream out;
  out << "patch=" << desc.patch_h << 'x' << desc.patch_w;
  for (size_t idx = 0u; idx < desc.layer_token_sizes.size(); ++idx) {
    out << "|layer" << (idx + 1u)
        << "_tokens=" << shape_key(desc.layer_token_sizes[idx])
        << "_feature=" << shape_key(desc.layer_feature_sizes[idx])
        << "_project=" << shape_key(desc.project_layer_sizes[idx])
        << "_resize="
        << (desc.apply_resize[idx] ? shape_key(desc.resize_layer_sizes[idx])
                                   : std::string("none"))
        << "_decoder=" << shape_key(desc.decoder_layer_sizes[idx]);
  }
  out << "|output=" << shape_key(desc.output_sizes);
  return out.str();
}

std::string make_full_session_configuration_key(
    const DepthAnythingV2SessionDesc& desc) {
  std::ostringstream out;
  out << "patch_tokens=" << shape_key(desc.patch_token_sizes)
      << "|blocks=" << desc.backbone_block_count
      << "|capture=" << vector_key(desc.capture_indices)
      << "|hidden=" << vector_key(desc.block_hidden_dims)
      << "|heads=" << vector_key(desc.block_num_heads)
      << "|norm_shape=" << shape_key(desc.normalized_shape)
      << "|patch=" << desc.patch_h << 'x' << desc.patch_w;
  for (size_t idx = 0u; idx < desc.layer_feature_sizes.size(); ++idx) {
    out << "|layer" << (idx + 1u)
        << "_feature=" << shape_key(desc.layer_feature_sizes[idx])
        << "_project=" << shape_key(desc.project_layer_sizes[idx])
        << "_resize="
        << (desc.apply_resize[idx] ? shape_key(desc.resize_layer_sizes[idx])
                                   : std::string("none"))
        << "_decoder=" << shape_key(desc.decoder_layer_sizes[idx]);
  }
  out << "|output=" << shape_key(desc.output_sizes);
  return out.str();
}

std::string make_image_session_configuration_key(
    const DepthAnythingV2ImageSessionDesc& desc) {
  std::ostringstream out;
  out << "image=" << shape_key(desc.image_sizes)
      << "|patch_tokens=" << shape_key(desc.patch_token_sizes)
      << "|prefix=" << shape_key(desc.prefix_token_sizes)
      << "|patch_pos=" << shape_key(desc.patch_pos_encoding_sizes)
      << "|blocks=" << desc.backbone_block_count
      << "|capture=" << vector_key(desc.capture_indices)
      << "|hidden=" << vector_key(desc.block_hidden_dims)
      << "|heads=" << vector_key(desc.block_num_heads)
      << "|norm_shape=" << shape_key(desc.normalized_shape)
      << "|patch=" << desc.patch_h << 'x' << desc.patch_w;
  for (size_t idx = 0u; idx < desc.layer_feature_sizes.size(); ++idx) {
    out << "|layer" << (idx + 1u)
        << "_feature=" << shape_key(desc.layer_feature_sizes[idx])
        << "_project=" << shape_key(desc.project_layer_sizes[idx])
        << "_resize="
        << (desc.apply_resize[idx] ? shape_key(desc.resize_layer_sizes[idx])
                                   : std::string("none"))
        << "_decoder=" << shape_key(desc.decoder_layer_sizes[idx]);
  }
  out << "|output=" << shape_key(desc.output_sizes);
  return out.str();
}

struct DepthAnythingV2BackboneIRHandles final {
  VulkanValueId patch_tokens{0u};
  std::vector<VulkanValueId> block_outputs;
  std::vector<VulkanValueId> capture_outputs;
};

struct DepthAnythingV2DecoderIRDesc final {
  ScalarType dtype{kFloat};
  bool persistent{true};
  int64_t patch_h{0};
  int64_t patch_w{0};
  std::array<std::vector<int64_t>, 4u> layer_feature_sizes;
  std::array<std::vector<int64_t>, 4u> project_layer_sizes;
  std::array<std::vector<int64_t>, 4u> resize_layer_sizes;
  std::array<bool, 4u> apply_resize{{true, true, false, true}};
  std::array<std::vector<int64_t>, 4u> decoder_layer_sizes;
  std::vector<int64_t> output_sizes;
};

struct DepthAnythingV2DecoderIRHandles final {
  std::array<VulkanValueId, 4u> feature_values{};
  std::array<VulkanValueId, 4u> project_values{};
  std::array<std::optional<VulkanValueId>, 4u> resize_values{};
  std::array<VulkanValueId, 4u> decoder_values{};
  VulkanValueId head_output{0u};
  VulkanValueId final_output{0u};
};

DepthAnythingV2BackboneIRHandles append_depth_anything_v2_backbone_region(
    VulkanBackendIR& ir,
    const VulkanValueId patch_tokens_value,
    const std::vector<int64_t>& patch_token_sizes,
    const ScalarType dtype,
    const bool persistent,
    const int64_t backbone_block_count,
    const std::vector<int64_t>& capture_indices,
    const std::vector<int64_t>& block_hidden_dims,
    const std::vector<int64_t>& block_num_heads,
    const std::optional<std::vector<int64_t>>& normalized_shape,
    const VulkanIRTensorRole capture_role,
    const bool capture_external) {
  DepthAnythingV2BackboneIRHandles handles;
  handles.patch_tokens = patch_tokens_value;
  handles.block_outputs.reserve(static_cast<size_t>(std::max<int64_t>(backbone_block_count, 0)));
  handles.capture_outputs.resize(capture_indices.size());

  VulkanValueId current = handles.patch_tokens;
  for (int64_t block_idx = 0; block_idx < backbone_block_count; ++block_idx) {
    const VulkanValueId constants = ir.add_value(
        "backbone.block." + std::to_string(block_idx) + ".constants",
        make_constant_spec(dtype, persistent));
    const VulkanValueId output = ir.add_value(
        "backbone.block." + std::to_string(block_idx) + ".tokens",
        make_tensor_spec(
            patch_token_sizes,
            dtype,
            VulkanIRTensorRole::Intermediate,
            persistent,
            false));

    std::ostringstream attrs;
    attrs << "block=" << block_idx;
    if (static_cast<size_t>(block_idx) < block_hidden_dims.size()) {
      attrs << "|hidden_dim=" << block_hidden_dims[block_idx];
    }
    if (static_cast<size_t>(block_idx) < block_num_heads.size()) {
      attrs << "|num_heads=" << block_num_heads[block_idx];
    }
    ir.add_op(VulkanIROpNode{
        VulkanIROpKind::BackboneBlock,
        "backbone.block." + std::to_string(block_idx),
        {current},
        {output},
        {constants},
        attrs.str()});
    current = output;
    handles.block_outputs.push_back(output);

    const auto capture_it =
        std::find(capture_indices.begin(), capture_indices.end(), block_idx);
    if (capture_it == capture_indices.end()) {
      continue;
    }

    const size_t capture_pos = static_cast<size_t>(
        std::distance(capture_indices.begin(), capture_it));
    const VulkanValueId capture = ir.add_value(
        "capture." + std::to_string(capture_pos) + ".tokens",
        make_tensor_spec(
            patch_token_sizes,
            dtype,
            capture_role,
            persistent,
            capture_external));
    if (normalized_shape.has_value()) {
      const VulkanValueId norm_constants = ir.add_value(
          "capture." + std::to_string(capture_pos) + ".norm.constants",
          make_constant_spec(dtype, persistent));
      ir.add_op(VulkanIROpNode{
          VulkanIROpKind::CaptureNormedPatchTokens,
          "capture." + std::to_string(capture_pos) + ".norm",
          {current},
          {capture},
          {norm_constants},
          "normalized_shape=" + shape_key(*normalized_shape)});
    } else {
      ir.add_op(VulkanIROpNode{
          VulkanIROpKind::OutputAlias,
          "capture." + std::to_string(capture_pos) + ".alias",
          {current},
          {capture},
          {},
          std::string()});
      ir.add_output_alias(capture, current);
    }
    handles.capture_outputs[capture_pos] = capture;
  }

  return handles;
}

DepthAnythingV2DecoderIRHandles append_depth_anything_v2_decoder_region(
    VulkanBackendIR& ir,
    const std::array<VulkanValueId, 4u>& token_values,
    const DepthAnythingV2DecoderIRDesc& desc) {
  DepthAnythingV2DecoderIRHandles handles;
  for (size_t idx = 0u; idx < token_values.size(); ++idx) {
    handles.feature_values[idx] = ir.add_value(
        "decoder.layer" + std::to_string(idx + 1u) + ".feature_map",
        make_tensor_spec(
            desc.layer_feature_sizes[idx],
            desc.dtype,
            VulkanIRTensorRole::Intermediate,
            desc.persistent,
            false));
    ir.add_op(VulkanIROpNode{
        VulkanIROpKind::TokensToFeatureMap,
        "decoder.layer" + std::to_string(idx + 1u) + ".tokens_to_feature_map",
        {token_values[idx]},
        {handles.feature_values[idx]},
        {},
        "patch=" + std::to_string(desc.patch_h) + "x" +
            std::to_string(desc.patch_w)});

    const VulkanValueId project_constants = ir.add_value(
        "decoder.layer" + std::to_string(idx + 1u) + ".project.constants",
        make_constant_spec(desc.dtype, desc.persistent));
    handles.project_values[idx] = ir.add_value(
        "decoder.layer" + std::to_string(idx + 1u) + ".projected",
        make_tensor_spec(
            desc.project_layer_sizes[idx],
            desc.dtype,
            VulkanIRTensorRole::Intermediate,
            desc.persistent,
            false));
    ir.add_op(VulkanIROpNode{
        VulkanIROpKind::DecoderProject,
        "decoder.layer" + std::to_string(idx + 1u) + ".project",
        {handles.feature_values[idx]},
        {handles.project_values[idx]},
        {project_constants},
        std::string()});

    VulkanValueId preprocess_input = handles.project_values[idx];
    if (desc.apply_resize[idx]) {
      const VulkanValueId resize_constants = ir.add_value(
          "decoder.layer" + std::to_string(idx + 1u) + ".resize.constants",
          make_constant_spec(desc.dtype, desc.persistent));
      const VulkanValueId resize_output = ir.add_value(
          "decoder.layer" + std::to_string(idx + 1u) + ".resized",
          make_tensor_spec(
              desc.resize_layer_sizes[idx],
              desc.dtype,
              VulkanIRTensorRole::Intermediate,
              desc.persistent,
              false));
      handles.resize_values[idx] = resize_output;
      ir.add_op(VulkanIROpNode{
          VulkanIROpKind::DecoderResize,
          "decoder.layer" + std::to_string(idx + 1u) + ".resize",
          {handles.project_values[idx]},
          {resize_output},
          {resize_constants},
          std::string()});
      preprocess_input = resize_output;
    }

    const VulkanValueId preprocess_constants = ir.add_value(
        "decoder.layer" + std::to_string(idx + 1u) + ".preprocess.constants",
        make_constant_spec(desc.dtype, desc.persistent));
    handles.decoder_values[idx] = ir.add_value(
        "decoder.layer" + std::to_string(idx + 1u) + ".preprocessed",
        make_tensor_spec(
            desc.decoder_layer_sizes[idx],
            desc.dtype,
            VulkanIRTensorRole::Intermediate,
            desc.persistent,
            false));
    ir.add_op(VulkanIROpNode{
        VulkanIROpKind::DecoderPreprocess,
        "decoder.layer" + std::to_string(idx + 1u) + ".preprocess",
        {preprocess_input},
        {handles.decoder_values[idx]},
        {preprocess_constants},
        std::string()});
  }

  const VulkanValueId head_constants = ir.add_value(
      "decoder.head.constants",
      make_constant_spec(desc.dtype, desc.persistent));
  handles.head_output = ir.add_value(
      "decoder.head.output",
      make_tensor_spec(
          desc.output_sizes,
          desc.dtype,
          VulkanIRTensorRole::Intermediate,
          desc.persistent,
          false));
  ir.add_op(VulkanIROpNode{
      VulkanIROpKind::DecoderHead,
      "decoder.head",
      {handles.decoder_values[0],
       handles.decoder_values[1],
       handles.decoder_values[2],
       handles.decoder_values[3]},
      {handles.head_output},
      {head_constants},
      std::string()});

  handles.final_output = ir.add_value(
      "final_output",
      make_tensor_spec(
          desc.output_sizes,
          desc.dtype,
          VulkanIRTensorRole::Output,
          desc.persistent,
          true));
  ir.add_op(VulkanIROpNode{
      VulkanIROpKind::OutputAlias,
      "final_output.alias",
      {handles.head_output},
      {handles.final_output},
      {},
      std::string()});
  ir.add_output_alias(handles.final_output, handles.head_output);

  return handles;
}

VulkanBackendIR make_depth_anything_v2_backbone_ir(
    const DepthAnythingV2BackboneStackSessionDesc& desc) {
  VulkanBackendIR ir;
  const VulkanValueId patch_tokens = ir.add_value(
      "patch_tokens",
      make_tensor_spec(
          desc.patch_token_sizes,
          desc.dtype,
          VulkanIRTensorRole::Input,
          desc.persistent,
          true));
  (void)append_depth_anything_v2_backbone_region(
      ir,
      patch_tokens,
      desc.patch_token_sizes,
      desc.dtype,
      desc.persistent,
      desc.backbone_block_count,
      desc.capture_indices,
      desc.block_hidden_dims,
      desc.block_num_heads,
      desc.normalized_shape,
      VulkanIRTensorRole::Output,
      true);
  ir.recompute_lifetimes();
  return ir;
}

VulkanBackendIR make_depth_anything_v2_decoder_ir(
    const DepthAnythingV2DecoderPreprocessHeadSessionDesc& desc) {
  VulkanBackendIR ir;
  std::array<VulkanValueId, 4u> token_values{};
  for (size_t idx = 0u; idx < token_values.size(); ++idx) {
    token_values[idx] = ir.add_value(
        "decoder.layer" + std::to_string(idx + 1u) + ".tokens",
        make_tensor_spec(
            desc.layer_token_sizes[idx],
            desc.dtype,
            VulkanIRTensorRole::Input,
            desc.persistent,
            true));
  }
  (void)append_depth_anything_v2_decoder_region(
      ir,
      token_values,
      DepthAnythingV2DecoderIRDesc{
          desc.dtype,
          desc.persistent,
          desc.patch_h,
          desc.patch_w,
          desc.layer_feature_sizes,
          desc.project_layer_sizes,
          desc.resize_layer_sizes,
          desc.apply_resize,
          desc.decoder_layer_sizes,
          desc.output_sizes});
  ir.recompute_lifetimes();
  return ir;
}

VulkanBackendIR make_depth_anything_v2_full_ir(
    const DepthAnythingV2SessionDesc& desc) {
  TORCH_INTERNAL_ASSERT(
      desc.capture_indices.size() == 4u,
      "DepthAnythingV2 full session expects exactly four capture indices");
  VulkanBackendIR ir;
  const VulkanValueId patch_tokens = ir.add_value(
      "patch_tokens",
      make_tensor_spec(
          desc.patch_token_sizes,
          desc.dtype,
          VulkanIRTensorRole::Input,
          desc.persistent,
          true));
  const std::optional<std::vector<int64_t>> normalized_shape =
      desc.normalized_shape.empty() ? std::nullopt
                                    : std::make_optional(desc.normalized_shape);
  const auto backbone_handles = append_depth_anything_v2_backbone_region(
      ir,
      patch_tokens,
      desc.patch_token_sizes,
      desc.dtype,
      desc.persistent,
      desc.backbone_block_count,
      desc.capture_indices,
      desc.block_hidden_dims,
      desc.block_num_heads,
      normalized_shape,
      VulkanIRTensorRole::Intermediate,
      false);
  std::array<VulkanValueId, 4u> capture_tokens{};
  for (size_t idx = 0u; idx < capture_tokens.size(); ++idx) {
    capture_tokens[idx] = backbone_handles.capture_outputs[idx];
  }
  (void)append_depth_anything_v2_decoder_region(
      ir,
      capture_tokens,
      DepthAnythingV2DecoderIRDesc{
          desc.dtype,
          desc.persistent,
          desc.patch_h,
          desc.patch_w,
          desc.layer_feature_sizes,
          desc.project_layer_sizes,
          desc.resize_layer_sizes,
          desc.apply_resize,
          desc.decoder_layer_sizes,
          desc.output_sizes});
  ir.recompute_lifetimes();
  return ir;
}

VulkanBackendIR make_depth_anything_v2_image_full_ir(
    const DepthAnythingV2ImageSessionDesc& desc) {
  TORCH_INTERNAL_ASSERT(
      desc.capture_indices.size() == 4u,
      "DepthAnythingV2 image session expects exactly four capture indices");
  TORCH_INTERNAL_ASSERT(
      desc.image_sizes.size() == 4u,
      "DepthAnythingV2 image session expects a rank-4 image input");
  TORCH_INTERNAL_ASSERT(
      desc.patch_token_sizes.size() == 2u || desc.patch_token_sizes.size() == 3u,
      "DepthAnythingV2 image session expects rank-2 or rank-3 patch tokens");
  TORCH_INTERNAL_ASSERT(
      desc.prefix_token_sizes.size() == 3u &&
          desc.patch_pos_encoding_sizes.size() == 3u,
      "DepthAnythingV2 image session expects rank-3 prefix and positional "
      "encoding tensors");

  const std::vector<int64_t> backbone_patch_token_sizes =
      desc.patch_token_sizes.size() == 2u
      ? std::vector<int64_t>{
            std::max<int64_t>(desc.image_sizes[0], 1),
            desc.patch_token_sizes[0],
            desc.patch_token_sizes[1]}
      : desc.patch_token_sizes;
  const std::vector<int64_t> patch_body_token_sizes{
      backbone_patch_token_sizes[0],
      desc.patch_h * desc.patch_w,
      backbone_patch_token_sizes[2],
  };

  VulkanBackendIR ir;
  const VulkanValueId image_input = ir.add_value(
      "input_image",
      make_tensor_spec(
          desc.image_sizes,
          desc.dtype,
          VulkanIRTensorRole::Input,
          desc.persistent,
          true));
  const VulkanValueId patch_embed_constants = ir.add_value(
      "patch_embed.constants",
      make_constant_spec(desc.dtype, desc.persistent));
  const VulkanValueId patch_feature_map = ir.add_value(
      "patch_embed.feature_map",
      make_tensor_spec(
          std::vector<int64_t>{
              desc.image_sizes[0],
              backbone_patch_token_sizes.back(),
              desc.patch_h,
              desc.patch_w},
          desc.dtype,
          VulkanIRTensorRole::Intermediate,
          desc.persistent,
          false));
  ir.add_op(VulkanIROpNode{
      VulkanIROpKind::PatchEmbed,
      "patch_embed",
      {image_input},
      {patch_feature_map},
      {patch_embed_constants},
      std::string()});

  const VulkanValueId prefix_token = ir.add_value(
      "patch_tokens.prefix",
      make_tensor_spec(
          desc.prefix_token_sizes,
          desc.dtype,
          VulkanIRTensorRole::Constant,
          desc.persistent,
          true));
  const VulkanValueId patch_pos_encoding = ir.add_value(
      "patch_tokens.pos_encoding",
      make_tensor_spec(
          desc.patch_pos_encoding_sizes,
          desc.dtype,
          VulkanIRTensorRole::Constant,
          desc.persistent,
          true));
  const VulkanValueId feature_map_tokens = ir.add_value(
      "patch_tokens.body",
      make_tensor_spec(
          patch_body_token_sizes,
          desc.dtype,
          VulkanIRTensorRole::Intermediate,
          desc.persistent,
          false));
  ir.add_op(VulkanIROpNode{
      VulkanIROpKind::FeatureMapToTokens,
      "patch_tokens.feature_map_to_tokens",
      {patch_feature_map},
      {feature_map_tokens},
      {},
      std::string()});
  const VulkanValueId positioned_patch_tokens = ir.add_value(
      "patch_tokens.positioned",
      make_tensor_spec(
          patch_body_token_sizes,
          desc.dtype,
          VulkanIRTensorRole::Intermediate,
          desc.persistent,
          false));
  ir.add_op(VulkanIROpNode{
      VulkanIROpKind::ElementwiseAdd,
      "patch_tokens.add_pos_encoding",
      {feature_map_tokens, patch_pos_encoding},
      {positioned_patch_tokens},
      {},
      std::string()});
  const VulkanValueId patch_tokens = ir.add_value(
      "patch_tokens",
      make_tensor_spec(
          backbone_patch_token_sizes,
          desc.dtype,
          VulkanIRTensorRole::Intermediate,
          desc.persistent,
          false));
  ir.add_op(VulkanIROpNode{
      VulkanIROpKind::Concat,
      "patch_tokens.concat_prefix",
      {prefix_token, positioned_patch_tokens},
      {patch_tokens},
      {},
      "dim=1|special_tokens=" +
          std::to_string(desc.prefix_token_sizes.size() > 1u
                             ? desc.prefix_token_sizes[1]
                             : 0)});

  const std::optional<std::vector<int64_t>> normalized_shape =
      desc.normalized_shape.empty() ? std::nullopt
                                    : std::make_optional(desc.normalized_shape);
  const auto backbone_handles = append_depth_anything_v2_backbone_region(
      ir,
      patch_tokens,
      backbone_patch_token_sizes,
      desc.dtype,
      desc.persistent,
      desc.backbone_block_count,
      desc.capture_indices,
      desc.block_hidden_dims,
      desc.block_num_heads,
      normalized_shape,
      VulkanIRTensorRole::Intermediate,
      false);
  std::array<VulkanValueId, 4u> capture_tokens{};
  for (size_t idx = 0u; idx < capture_tokens.size(); ++idx) {
    capture_tokens[idx] = backbone_handles.capture_outputs[idx];
  }
  (void)append_depth_anything_v2_decoder_region(
      ir,
      capture_tokens,
      DepthAnythingV2DecoderIRDesc{
          desc.dtype,
          desc.persistent,
          desc.patch_h,
          desc.patch_w,
          desc.layer_feature_sizes,
          desc.project_layer_sizes,
          desc.resize_layer_sizes,
          desc.apply_resize,
          desc.decoder_layer_sizes,
          desc.output_sizes});
  ir.recompute_lifetimes();
  return ir;
}

VulkanIRMemoryPlan make_memory_plan(const VulkanBackendIR& ir) {
  VulkanIRMemoryPlan plan;
  const auto& values = ir.values();
  const auto& lifetimes = ir.lifetimes();

  const auto lifetime_for = [&](const VulkanValueId id) -> VulkanIRLifetime {
    if (id < lifetimes.size()) {
      return lifetimes[id];
    }
    return VulkanIRLifetime{id, 0u, 0u, false};
  };

  const auto make_slot = [&](
                             const VulkanIRValue& value,
                             const VulkanIRLifetime& lifetime,
                             const size_t bytes,
                             const bool dedicated) {
    VulkanIRAllocationSlot slot;
    slot.slot_id = plan.slots.size();
    slot.bytes = bytes;
    slot.first_op = lifetime.first_op;
    slot.last_op = lifetime.last_op;
    slot.dedicated = dedicated;
    slot.values.push_back(value.id);
    plan.slots.push_back(std::move(slot));
  };

  for (const auto& value : values) {
    const size_t bytes = tensor_spec_nbytes(value.spec);
    if (bytes == 0u) {
      continue;
    }

    const VulkanIRLifetime lifetime = lifetime_for(value.id);
    if (value.spec.external) {
      plan.external_bytes += bytes;
      continue;
    }

    if (requires_dedicated_slot(value.spec) || lifetime.may_alias) {
      plan.dedicated_bytes += bytes;
      make_slot(value, lifetime, bytes, true);
      continue;
    }

    auto reusable = std::find_if(
        plan.slots.begin(),
        plan.slots.end(),
        [&](const VulkanIRAllocationSlot& slot) {
          if (slot.dedicated || slot.last_op >= lifetime.first_op ||
              slot.values.empty() || slot.values[0] >= values.size()) {
            return false;
          }
          const auto& slot_spec = values[slot.values[0]].spec;
          return slot_spec.logical_sizes == value.spec.logical_sizes &&
              slot_spec.dtype == value.spec.dtype &&
              slot_spec.execution_layout == value.spec.execution_layout &&
              slot_spec.memory_layout == value.spec.memory_layout &&
              slot_spec.storage_type == value.spec.storage_type;
        });
    if (reusable == plan.slots.end()) {
      make_slot(value, lifetime, bytes, false);
      plan.reusable_bytes += bytes;
    } else {
      if (bytes > reusable->bytes) {
        plan.reusable_bytes += bytes - reusable->bytes;
        reusable->bytes = bytes;
      }
      reusable->first_op = std::min(reusable->first_op, lifetime.first_op);
      reusable->last_op = std::max(reusable->last_op, lifetime.last_op);
      reusable->values.push_back(value.id);
    }
  }

  plan.planned_bytes = plan.reusable_bytes + plan.dedicated_bytes;
  return plan;
}

VulkanCompiledSession make_compiled_session(
    VulkanCompiledSessionKey key,
    VulkanBackendIR ir,
    VulkanGlobalLayoutPlan layout_plan,
    const bool executable) {
  apply_global_layout_plan(ir, layout_plan);
  ir.recompute_lifetimes();
  VulkanIRMemoryPlan memory_plan = make_memory_plan(ir);
  return VulkanCompiledSession{std::make_shared<VulkanCompiledSession::State>(
      std::move(key),
      std::move(ir),
      std::move(layout_plan),
      std::move(memory_plan),
      executable)};
}

} // namespace compiled_session_impl

const char* compiled_session_kind_name(const VulkanCompiledSessionKind kind) {
  switch (kind) {
    case VulkanCompiledSessionKind::DepthAnythingV2:
      return "DepthAnythingV2";
    case VulkanCompiledSessionKind::DepthAnythingV2Image:
      return "DepthAnythingV2Image";
    case VulkanCompiledSessionKind::DepthAnythingV2BackboneStack:
      return "DepthAnythingV2BackboneStack";
    case VulkanCompiledSessionKind::DepthAnythingV2DecoderPreprocessHead:
      return "DepthAnythingV2DecoderPreprocessHead";
  }
  return "UnknownCompiledSession";
}

const char* ir_op_kind_name(const VulkanIROpKind kind) {
  switch (kind) {
    case VulkanIROpKind::InputImage:
      return "InputImage";
    case VulkanIROpKind::PatchEmbed:
      return "PatchEmbed";
    case VulkanIROpKind::FeatureMapToTokens:
      return "FeatureMapToTokens";
    case VulkanIROpKind::ElementwiseAdd:
      return "ElementwiseAdd";
    case VulkanIROpKind::Concat:
      return "Concat";
    case VulkanIROpKind::PatchTokenInput:
      return "PatchTokenInput";
    case VulkanIROpKind::BackboneBlock:
      return "BackboneBlock";
    case VulkanIROpKind::CaptureNormedPatchTokens:
      return "CaptureNormedPatchTokens";
    case VulkanIROpKind::TokensToFeatureMap:
      return "TokensToFeatureMap";
    case VulkanIROpKind::DecoderProject:
      return "DecoderProject";
    case VulkanIROpKind::DecoderResize:
      return "DecoderResize";
    case VulkanIROpKind::DecoderPreprocess:
      return "DecoderPreprocess";
    case VulkanIROpKind::DecoderHead:
      return "DecoderHead";
    case VulkanIROpKind::OutputAlias:
      return "OutputAlias";
  }
  return "UnknownIROp";
}

bool operator==(
    const VulkanCompiledSessionKey& lhs,
    const VulkanCompiledSessionKey& rhs) {
  return lhs.kind == rhs.kind && lhs.model_key == rhs.model_key &&
      lhs.configuration_key == rhs.configuration_key &&
      lhs.input_shapes == rhs.input_shapes &&
      lhs.output_shapes == rhs.output_shapes && lhs.dtype == rhs.dtype &&
      lhs.capability_key == rhs.capability_key &&
      lhs.persistent == rhs.persistent;
}

size_t VulkanCompiledSessionKeyHash::operator()(
    const VulkanCompiledSessionKey& key) const {
  size_t seed = 0u;
  compiled_session_impl::hash_combine_session(
      seed, static_cast<uint8_t>(key.kind));
  compiled_session_impl::hash_combine_session(seed, key.model_key);
  compiled_session_impl::hash_combine_session(seed, key.configuration_key);
  compiled_session_impl::hash_shapes(seed, key.input_shapes);
  compiled_session_impl::hash_shapes(seed, key.output_shapes);
  compiled_session_impl::hash_combine_session(seed, static_cast<int>(key.dtype));
  compiled_session_impl::hash_combine_session(seed, key.capability_key);
  compiled_session_impl::hash_combine_session(seed, key.persistent);
  return seed;
}

VulkanValueId VulkanBackendIR::add_value(
    std::string name,
    VulkanIRTensorSpec spec) {
  const VulkanValueId id = static_cast<VulkanValueId>(values_.size());
  values_.push_back(VulkanIRValue{id, std::move(name), std::move(spec)});
  return id;
}

void VulkanBackendIR::add_op(VulkanIROpNode op) {
  ops_.push_back(std::move(op));
}

void VulkanBackendIR::add_output_alias(
    const VulkanValueId output,
    const VulkanValueId source) {
  output_aliases_.push_back(VulkanIROutputAlias{output, source});
}

void VulkanBackendIR::recompute_lifetimes() {
  lifetimes_.clear();
  lifetimes_.reserve(values_.size());
  for (const auto& value : values_) {
    lifetimes_.push_back(VulkanIRLifetime{
        value.id,
        std::numeric_limits<size_t>::max(),
        0u,
        false});
  }

  const auto note_use = [&](const VulkanValueId id, const size_t op_idx) {
    if (id >= lifetimes_.size()) {
      return;
    }
    auto& lifetime = lifetimes_[id];
    lifetime.first_op = std::min(lifetime.first_op, op_idx);
    lifetime.last_op = std::max(lifetime.last_op, op_idx);
  };

  for (size_t op_idx = 0u; op_idx < ops_.size(); ++op_idx) {
    const auto& op = ops_[op_idx];
    for (const VulkanValueId id : op.inputs) {
      note_use(id, op_idx);
    }
    for (const VulkanValueId id : op.outputs) {
      note_use(id, op_idx);
    }
    for (const VulkanValueId id : op.constants) {
      note_use(id, op_idx);
    }
  }

  for (const auto& alias : output_aliases_) {
    note_use(alias.output, ops_.size());
    note_use(alias.source, ops_.size());
    if (alias.output < lifetimes_.size()) {
      lifetimes_[alias.output].may_alias = true;
    }
  }

  for (auto& lifetime : lifetimes_) {
    if (lifetime.first_op == std::numeric_limits<size_t>::max()) {
      lifetime.first_op = 0u;
      lifetime.last_op = 0u;
    }
  }
}

const std::vector<VulkanIRValue>& VulkanBackendIR::values() const {
  return values_;
}

std::vector<VulkanIRValue>& VulkanBackendIR::mutable_values() {
  return values_;
}

const std::vector<VulkanIROpNode>& VulkanBackendIR::ops() const {
  return ops_;
}

const std::vector<VulkanIRLifetime>& VulkanBackendIR::lifetimes() const {
  return lifetimes_;
}

const std::vector<VulkanIROutputAlias>& VulkanBackendIR::output_aliases()
    const {
  return output_aliases_;
}

bool VulkanCompiledSession::defined() const {
  return static_cast<bool>(state_);
}

const VulkanCompiledSessionKey& VulkanCompiledSession::key() const {
  TORCH_INTERNAL_ASSERT(state_, "VulkanCompiledSession is not defined");
  return state_->key;
}

const VulkanBackendIR& VulkanCompiledSession::ir() const {
  TORCH_INTERNAL_ASSERT(state_, "VulkanCompiledSession is not defined");
  return state_->ir;
}

const VulkanGlobalLayoutPlan& VulkanCompiledSession::layout_plan() const {
  TORCH_INTERNAL_ASSERT(state_, "VulkanCompiledSession is not defined");
  return state_->layout_plan;
}

const VulkanIRMemoryPlan& VulkanCompiledSession::memory_plan() const {
  TORCH_INTERNAL_ASSERT(state_, "VulkanCompiledSession is not defined");
  return state_->memory_plan;
}

bool VulkanCompiledSession::executable() const {
  return state_ && state_->executable;
}

const void* VulkanCompiledSession::identity() const {
  return state_.get();
}

std::optional<VulkanCompiledSessionTensorBindings>
make_compiled_session_tensor_bindings(const VulkanCompiledSession& session) {
  if (!session.defined() || !session.executable()) {
    return std::nullopt;
  }

  const auto& ir = session.ir();
  const auto& values = ir.values();
  const auto& memory_plan = session.memory_plan();

  VulkanCompiledSessionTensorBindings bindings;
  bindings.value_tensor_slots.resize(values.size());

  const auto bind_value = [&](const VulkanValueId value_id, const size_t slot_idx) {
    TORCH_INTERNAL_ASSERT(
        value_id < bindings.value_tensor_slots.size(),
        "Compiled session tensor binding references an invalid value id");
    auto& mapped_slot = bindings.value_tensor_slots[value_id];
    TORCH_INTERNAL_ASSERT(
        !mapped_slot.has_value() || *mapped_slot == slot_idx,
        "Compiled session value was assigned conflicting tensor slots");
    mapped_slot = slot_idx;
  };

  for (const auto& value : values) {
    if (
        value.spec.role != VulkanIRTensorRole::Input || !value.spec.external ||
        value.spec.logical_sizes.empty()) {
      continue;
    }
    const size_t slot_idx = bindings.slot_values.size();
    bindings.slot_values.push_back(value.id);
    bindings.input_values.push_back(value.id);
    bind_value(value.id, slot_idx);
  }

  for (const auto& slot : memory_plan.slots) {
    TORCH_INTERNAL_ASSERT(
        !slot.values.empty(),
        "Compiled session memory slot requires at least one value");
    TORCH_INTERNAL_ASSERT(
        slot.values[0] < values.size(),
        "Compiled session memory slot references an invalid value");
    const size_t slot_idx = bindings.slot_values.size();
    bindings.slot_values.push_back(slot.values[0]);
    for (const VulkanValueId value_id : slot.values) {
      bind_value(value_id, slot_idx);
    }
  }

  std::unordered_set<VulkanValueId> aliased_outputs;
  aliased_outputs.reserve(ir.output_aliases().size());
  for (const auto& alias : ir.output_aliases()) {
    aliased_outputs.insert(alias.output);
  }

  for (const auto& value : values) {
    if (
        !value.spec.external || value.spec.logical_sizes.empty() ||
        value.spec.role == VulkanIRTensorRole::Input ||
        aliased_outputs.count(value.id) > 0u ||
        bindings.value_tensor_slots[value.id].has_value()) {
      continue;
    }
    const size_t slot_idx = bindings.slot_values.size();
    bindings.slot_values.push_back(value.id);
    bind_value(value.id, slot_idx);
  }

  for (const auto& alias : ir.output_aliases()) {
    TORCH_INTERNAL_ASSERT(
        alias.output < bindings.value_tensor_slots.size() &&
            alias.source < bindings.value_tensor_slots.size(),
        "Compiled session output alias references an invalid value");
    if (!bindings.value_tensor_slots[alias.source].has_value()) {
      return std::nullopt;
    }
    bind_value(alias.output, *bindings.value_tensor_slots[alias.source]);
  }

  for (const auto& value : values) {
    if (
        compiled_session_impl::tensor_spec_nbytes(value.spec) == 0u ||
        bindings.value_tensor_slots[value.id].has_value()) {
      continue;
    }
    return std::nullopt;
  }

  return bindings;
}

std::string make_vulkan_compiled_session_capability_key(
    const VulkanRuntimeCapabilityProfile& profile) {
  std::ostringstream out;
  out << "umem=" << (profile.has_unified_memory ? 1 : 0)
      << "|ts=" << (profile.has_timestamps ? 1 : 0)
      << "|bf16=" << (profile.has_shader_bfloat16 ? 1 : 0)
      << "|i8=" << (profile.has_shader_int8 ? 1 : 0)
      << "|sb8=" << (profile.has_storage_buffer_8bit ? 1 : 0)
      << "|coop=" << (profile.has_cooperative_matrix ? 1 : 0)
      << "|coop_bf16="
      << (profile.has_subgroup_bfloat16_cooperative_matrix_inputs ? 1 : 0)
      << "|coop_f16="
      << (profile.has_subgroup_float16_cooperative_matrix_inputs ? 1 : 0)
      << "|coop_f32="
      << (profile.has_subgroup_float32_cooperative_matrix_inputs ? 1 : 0)
      << "|subgroup=" << profile.min_subgroup_size << '-'
      << profile.max_subgroup_size
      << "|coop_mnk=" << profile.cooperative_matrix_max_m << 'x'
      << profile.cooperative_matrix_max_n << 'x'
      << profile.cooperative_matrix_max_k
      << "|queues=" << profile.num_compute_queues;
  return out.str();
}

VulkanGlobalLayoutPlan make_buffer_first_width_packed_layout_plan(
    const VulkanCompiledSessionKey& key,
    const VulkanRuntimeCapabilityProfile& profile) {
  VulkanGlobalLayoutPlan plan;
  plan.execution_layout = api::ExecutionLayout::BUFFER_DIRECT;
  plan.memory_layout = api::GPUMemoryLayout::TENSOR_WIDTH_PACKED;
  plan.storage_type = api::StorageType::BUFFER;
  if (
      profile.has_cooperative_matrix &&
      (key.dtype == kFloat || key.dtype == kBFloat16 || key.dtype == kHalf)) {
    plan.width_alignment = 16;
    plan.pad_width = true;
    plan.reason = "buffer_first_width_packed_cooperative_matrix_ready";
  } else {
    plan.width_alignment = 1;
    plan.pad_width = false;
    plan.reason = "buffer_first_width_packed";
  }
  return plan;
}

void apply_global_layout_plan(
    VulkanBackendIR& ir,
    const VulkanGlobalLayoutPlan& plan) {
  for (auto& value : ir.mutable_values()) {
    if (
        value.spec.role == VulkanIRTensorRole::Constant &&
        !plan.apply_to_constants) {
      continue;
    }
    value.spec.execution_layout = plan.execution_layout;
    value.spec.memory_layout = plan.memory_layout;
    value.spec.storage_type = plan.storage_type;
    value.spec.padded_sizes = value.spec.logical_sizes;
    if (
        plan.pad_width && plan.width_alignment > 1 &&
        !value.spec.padded_sizes.empty()) {
      value.spec.padded_sizes.back() =
          compiled_session_impl::round_up_to_multiple(
              value.spec.padded_sizes.back(), plan.width_alignment);
    }
  }
}

VulkanCompiledSession lookup_or_create_vulkan_compiled_session(
    const VulkanCompiledSessionKey& key,
    const std::function<VulkanCompiledSession()>& builder) {
  auto& cache = compiled_session_impl::compiled_session_cache();
  {
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto found = cache.sessions.find(key);
    if (found != cache.sessions.end()) {
      compiled_session_impl::log_compiled_session_event(
          "hit",
          key,
          found->second.identity(),
          found->second.ir().values().size(),
          found->second.ir().ops().size(),
          &found->second.layout_plan(),
          &found->second.memory_plan());
      return found->second;
    }
  }

  VulkanCompiledSession created = builder();
  if (!created.defined()) {
    return created;
  }

  std::lock_guard<std::mutex> lock(cache.mutex);
  const auto found = cache.sessions.find(key);
  if (found != cache.sessions.end()) {
    return found->second;
  }
  if (cache.sessions.size() >= compiled_session_impl::kCompiledSessionCacheSize) {
    cache.sessions.erase(cache.sessions.begin());
  }
  auto [it, inserted] = cache.sessions.emplace(key, created);
  (void)inserted;
  compiled_session_impl::log_compiled_session_event(
      "store",
      key,
      it->second.identity(),
      it->second.ir().values().size(),
      it->second.ir().ops().size(),
      &it->second.layout_plan(),
      &it->second.memory_plan());
  return it->second;
}

VulkanCompiledSession lookup_or_create_depth_anything_v2_session(
    const DepthAnythingV2SessionDesc& desc) {
  const VulkanRuntimeCapabilityProfile profile =
      query_vulkan_runtime_capability_profile();
  VulkanCompiledSessionKey key;
  key.kind = VulkanCompiledSessionKind::DepthAnythingV2;
  key.model_key =
      desc.model_key.empty() ? "depth_anything_v2.compiled_session"
                             : desc.model_key;
  key.configuration_key =
      compiled_session_impl::make_full_session_configuration_key(desc);
  key.input_shapes = {desc.patch_token_sizes};
  key.output_shapes = {desc.output_sizes};
  key.dtype = desc.dtype;
  key.capability_key = make_vulkan_compiled_session_capability_key(profile);
  key.persistent = desc.persistent;

  return lookup_or_create_vulkan_compiled_session(key, [&]() {
    VulkanBackendIR ir =
        compiled_session_impl::make_depth_anything_v2_full_ir(desc);
    VulkanGlobalLayoutPlan layout_plan =
        make_buffer_first_width_packed_layout_plan(key, profile);
    return compiled_session_impl::make_compiled_session(
        key,
        std::move(ir),
        std::move(layout_plan),
        /*executable=*/true);
  });
}

VulkanCompiledSession lookup_or_create_depth_anything_v2_image_session(
    const DepthAnythingV2ImageSessionDesc& desc) {
  const VulkanRuntimeCapabilityProfile profile =
      query_vulkan_runtime_capability_profile();
  VulkanCompiledSessionKey key;
  key.kind = VulkanCompiledSessionKind::DepthAnythingV2Image;
  key.model_key = desc.model_key.empty()
      ? "depth_anything_v2.image_compiled_session"
      : desc.model_key;
  key.configuration_key =
      compiled_session_impl::make_image_session_configuration_key(desc);
  key.input_shapes = {desc.image_sizes};
  key.output_shapes = {desc.output_sizes};
  key.dtype = desc.dtype;
  key.capability_key = make_vulkan_compiled_session_capability_key(profile);
  key.persistent = desc.persistent;

  return lookup_or_create_vulkan_compiled_session(key, [&]() {
    VulkanBackendIR ir =
        compiled_session_impl::make_depth_anything_v2_image_full_ir(desc);
    VulkanGlobalLayoutPlan layout_plan =
        make_buffer_first_width_packed_layout_plan(key, profile);
    return compiled_session_impl::make_compiled_session(
        key,
        std::move(ir),
        std::move(layout_plan),
        /*executable=*/true);
  });
}

VulkanCompiledSession lookup_or_create_depth_anything_v2_backbone_stack_session(
    const DepthAnythingV2BackboneStackSessionDesc& desc) {
  const VulkanRuntimeCapabilityProfile profile =
      query_vulkan_runtime_capability_profile();
  VulkanCompiledSessionKey key;
  key.kind = VulkanCompiledSessionKind::DepthAnythingV2BackboneStack;
  key.model_key = desc.model_key.empty() ? "depth_anything_v2.backbone_stack"
                                         : desc.model_key;
  key.configuration_key =
      compiled_session_impl::make_backbone_configuration_key(desc);
  key.input_shapes = {desc.patch_token_sizes};
  for (size_t idx = 0u; idx < desc.capture_indices.size(); ++idx) {
    key.output_shapes.push_back(desc.patch_token_sizes);
  }
  key.dtype = desc.dtype;
  key.capability_key = make_vulkan_compiled_session_capability_key(profile);
  key.persistent = desc.persistent;

  return lookup_or_create_vulkan_compiled_session(key, [&]() {
    VulkanBackendIR ir =
        compiled_session_impl::make_depth_anything_v2_backbone_ir(desc);
    VulkanGlobalLayoutPlan layout_plan =
        make_buffer_first_width_packed_layout_plan(key, profile);
    return compiled_session_impl::make_compiled_session(
        key,
        std::move(ir),
        std::move(layout_plan),
        /*executable=*/true);
  });
}

VulkanCompiledSession
lookup_or_create_depth_anything_v2_decoder_preprocess_head_session(
    const DepthAnythingV2DecoderPreprocessHeadSessionDesc& desc) {
  const VulkanRuntimeCapabilityProfile profile =
      query_vulkan_runtime_capability_profile();
  VulkanCompiledSessionKey key;
  key.kind = VulkanCompiledSessionKind::DepthAnythingV2DecoderPreprocessHead;
  key.model_key = desc.model_key.empty()
      ? "depth_anything_v2.decoder_preprocess_head"
      : desc.model_key;
  key.configuration_key =
      compiled_session_impl::make_decoder_configuration_key(desc);
  for (const auto& shape : desc.layer_token_sizes) {
    key.input_shapes.push_back(shape);
  }
  key.output_shapes = {desc.output_sizes};
  key.dtype = desc.dtype;
  key.capability_key = make_vulkan_compiled_session_capability_key(profile);
  key.persistent = desc.persistent;

  return lookup_or_create_vulkan_compiled_session(key, [&]() {
    VulkanBackendIR ir = compiled_session_impl::make_depth_anything_v2_decoder_ir(
        desc);
    VulkanGlobalLayoutPlan layout_plan =
        make_buffer_first_width_packed_layout_plan(key, profile);
    return compiled_session_impl::make_compiled_session(
        key,
        std::move(ir),
        std::move(layout_plan),
        /*executable=*/true);
  });
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
