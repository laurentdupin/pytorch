#include <ATen/native/vulkan/planning/InferenceGraphs.h>

#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/ops/InferenceCache.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/planning/CompiledSession.h>
#include <ATen/native/vulkan/planning/ExecutableRegions.h>
#include <ATen/native/vulkan/planning/ReplayTensorState.h>

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
thread_local size_t g_inference_replay_callback_depth = 0u;

class InferenceReplayCallbackDepthScope final {
 public:
  InferenceReplayCallbackDepthScope(
      const char* const phase,
      const char* const allocation_label,
      const void* const identity)
      : active_(true) {
    if (g_inference_replay_callback_depth != 0u) {
      std::ostringstream detail;
      detail << "action=reject_nested_replay_callback"
             << " phase=" << (phase ? phase : "unknown")
             << " allocation_label="
             << (allocation_label ? allocation_label : "unknown")
             << " identity=" << identity
             << " depth=" << g_inference_replay_callback_depth;
      TORCH_CHECK(
          false,
          "Nested Vulkan inference replay callbacks are unsupported; ",
          "compiled regions must be flattened into first-class replay steps. ",
          detail.str());
    }
    ++g_inference_replay_callback_depth;
  }

  InferenceReplayCallbackDepthScope(
      const InferenceReplayCallbackDepthScope&) = delete;
  InferenceReplayCallbackDepthScope& operator=(
      const InferenceReplayCallbackDepthScope&) = delete;

  ~InferenceReplayCallbackDepthScope() {
    if (active_) {
      TORCH_INTERNAL_ASSERT(g_inference_replay_callback_depth > 0u);
      --g_inference_replay_callback_depth;
    }
  }

 private:
  bool active_;
};

template <typename T>
void hash_combine(size_t& seed, const T& value) {
  seed ^= std::hash<T>{}(value) + size_t{0x9e3779b97f4a7c15ull} + (seed << 6u) +
      (seed >> 2u);
}

std::string inference_graph_log_path() {
  const char* env = std::getenv("PYTORCH_VULKAN_INFERENCE_GRAPH_LOG");
  return env ? std::string(env) : std::string();
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
  InferenceReplayCallbackDepthScope depth_scope(
      "record", state_->allocation_label_.c_str(), identity());
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
  log_replay_event(
      "record",
      identity(),
      current_replay_epoch(identity()).run_id,
      state_->allocation_label_.c_str());
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
  const ReplayEpoch epoch =
      begin_replay_epoch(identity(), state_->allocation_label_.c_str());
  api::context()->submit_prepared_command_buffer(
      *state_->command_buffer_,
      fence_handle,
      final_use,
      state_->allocation_label_.c_str());
  log_replay_event(
      "submit",
      identity(),
      epoch.run_id,
      state_->allocation_label_.c_str(),
      final_use ? "final_use=1" : "final_use=0");
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
  InferenceReplayCallbackDepthScope depth_scope(
      "warmup",
      state_->replay_.defined() ? state_->replay_.allocation_label().c_str()
                                : "execution_graph_replay_bundle",
      state_->replay_.defined() ? state_->replay_.identity() : identity());
  for (const auto& step : state_->steps_) {
    step.record_step();
  }
  if (state_->replay_.defined()) {
    log_inference_replay_event(
        state_->replay_.kind(),
        "warmup",
        state_->replay_.allocation_label(),
        state_->replay_.identity());
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

void ExecutionGraphReplayBundle::record_empty() const {
  TORCH_INTERNAL_ASSERT(defined(), "Undefined ExecutionGraphReplayBundle");
  TORCH_INTERNAL_ASSERT(
      state_->replay_.defined(),
      "ExecutionGraphReplayBundle does not own a bundle replay");
  state_->replay_.record([]() {});
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
    if (!created.defined()) {
      {
        std::lock_guard<std::mutex> lock(state_->mutex_);
        state_->in_flight_bundles_.erase(bundle_key);
      }
      owner_promise->set_value(created);
      log_execution_graph_root_event(
          "bundle_build_skip",
          allocation_label(),
          identity(),
          nullptr,
          bundle_key.c_str());
      return created;
    }
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

void log_inference_replay_lifecycle_event(
    const InferenceReplay& replay,
    const char* event) {
  if (!replay.defined()) {
    return;
  }
  log_inference_replay_event(
      replay.kind(), event, replay.allocation_label(), replay.identity());
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
  std::shared_ptr<VulkanExecutableRegion> executable_region;
  bool executable{false};

  State(
      VulkanCompiledSessionKey key_in,
      VulkanBackendIR ir_in,
      VulkanGlobalLayoutPlan layout_plan_in,
      VulkanIRMemoryPlan memory_plan_in,
      std::shared_ptr<VulkanExecutableRegion> executable_region_in,
      const bool executable_in)
      : key(std::move(key_in)),
        ir(std::move(ir_in)),
        layout_plan(std::move(layout_plan_in)),
        memory_plan(std::move(memory_plan_in)),
        executable_region(std::move(executable_region_in)),
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

std::string compiled_session_log_path() {
  const char* env = std::getenv("PYTORCH_VULKAN_COMPILED_SESSION_LOG");
  return env ? std::string(env) : std::string();
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
      << key.capability_key << " layout_policy_version="
      << key.layout_policy_version << " model_lane_policy_version="
      << key.model_lane_policy_version
      << " persistent=" << (key.persistent ? 1 : 0);
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

std::string make_vision_transformer_depth_backbone_configuration_key(
    const VisionTransformerDepthBackboneSessionDesc& desc) {
  std::ostringstream out;
  out << "patch_tokens=" << shape_key(desc.patch_token_sizes)
      << "|blocks=" << desc.backbone_block_count
      << "|capture=" << vector_key(desc.capture_indices)
      << "|hidden=" << vector_key(desc.block_hidden_dims)
      << "|heads=" << vector_key(desc.block_num_heads)
      << "|norm_shape=" << optional_shape_key(desc.normalized_shape);
  return out.str();
}

std::string make_vision_transformer_depth_decoder_configuration_key(
    const VisionTransformerDepthDecoderSessionDesc& desc) {
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

std::string make_vision_transformer_depth_full_configuration_key(
    const VisionTransformerDepthSessionDesc& desc) {
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

std::string make_vision_transformer_depth_image_configuration_key(
    const VisionTransformerDepthImageSessionDesc& desc) {
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

struct VisionTransformerDepthBackboneIRHandles final {
  VulkanValueId patch_tokens{0u};
  std::vector<VulkanValueId> block_outputs;
  std::vector<VulkanValueId> capture_outputs;
};

struct VisionTransformerDepthDecoderIRDesc final {
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

struct VisionTransformerDepthDecoderIRHandles final {
  std::array<VulkanValueId, 4u> feature_values{};
  std::array<VulkanValueId, 4u> project_values{};
  std::array<std::optional<VulkanValueId>, 4u> resize_values{};
  std::array<VulkanValueId, 4u> decoder_values{};
  VulkanValueId head_output{0u};
  VulkanValueId final_output{0u};
};

VisionTransformerDepthBackboneIRHandles
append_vision_transformer_depth_backbone_region(
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
  VisionTransformerDepthBackboneIRHandles handles;
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
          VulkanIROpKind::CapturePatchTokens,
          "capture." + std::to_string(capture_pos) + ".materialize",
          {current},
          {capture},
          {},
          std::string()});
    }
    handles.capture_outputs[capture_pos] = capture;
  }

  return handles;
}

VisionTransformerDepthDecoderIRHandles
append_vision_transformer_depth_decoder_region(
    VulkanBackendIR& ir,
    const std::array<VulkanValueId, 4u>& token_values,
    const VisionTransformerDepthDecoderIRDesc& desc) {
  VisionTransformerDepthDecoderIRHandles handles;
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

VulkanBackendIR make_vision_transformer_depth_backbone_ir(
    const VisionTransformerDepthBackboneSessionDesc& desc) {
  VulkanBackendIR ir;
  const VulkanValueId patch_tokens = ir.add_value(
      "patch_tokens",
      make_tensor_spec(
          desc.patch_token_sizes,
          desc.dtype,
          VulkanIRTensorRole::Input,
          desc.persistent,
          true));
  (void)append_vision_transformer_depth_backbone_region(
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

VulkanBackendIR make_vision_transformer_depth_decoder_ir(
    const VisionTransformerDepthDecoderSessionDesc& desc) {
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
  (void)append_vision_transformer_depth_decoder_region(
      ir,
      token_values,
      VisionTransformerDepthDecoderIRDesc{
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

VulkanBackendIR make_vision_transformer_depth_full_ir(
    const VisionTransformerDepthSessionDesc& desc) {
  TORCH_INTERNAL_ASSERT(
      desc.capture_indices.size() == 4u,
      "VisionTransformerDepth full session expects exactly four capture indices");
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
  const auto backbone_handles = append_vision_transformer_depth_backbone_region(
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
  (void)append_vision_transformer_depth_decoder_region(
      ir,
      capture_tokens,
      VisionTransformerDepthDecoderIRDesc{
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

VulkanBackendIR make_vision_transformer_depth_image_full_ir(
    const VisionTransformerDepthImageSessionDesc& desc) {
  TORCH_INTERNAL_ASSERT(
      desc.capture_indices.size() == 4u,
      "VisionTransformerDepth image session expects exactly four capture indices");
  TORCH_INTERNAL_ASSERT(
      desc.image_sizes.size() == 4u,
      "VisionTransformerDepth image session expects a rank-4 image input");
  TORCH_INTERNAL_ASSERT(
      desc.patch_token_sizes.size() == 2u || desc.patch_token_sizes.size() == 3u,
      "VisionTransformerDepth image session expects rank-2 or rank-3 patch tokens");
  TORCH_INTERNAL_ASSERT(
      desc.prefix_token_sizes.size() == 3u &&
          desc.patch_pos_encoding_sizes.size() == 3u,
      "VisionTransformerDepth image session expects rank-3 prefix and positional "
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
  const auto backbone_handles = append_vision_transformer_depth_backbone_region(
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
  (void)append_vision_transformer_depth_decoder_region(
      ir,
      capture_tokens,
      VisionTransformerDepthDecoderIRDesc{
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

StageKind executable_stage_kind_for_op(const VulkanIROpNode& op) {
  switch (op.kind) {
    case VulkanIROpKind::PatchEmbed:
    case VulkanIROpKind::FeatureMapToTokens:
    case VulkanIROpKind::ElementwiseAdd:
    case VulkanIROpKind::Concat:
    case VulkanIROpKind::PatchTokenInput:
      return StageKind::ImageEntry;
    case VulkanIROpKind::BackboneBlock:
      return StageKind::Backbone;
    case VulkanIROpKind::CapturePatchTokens:
    case VulkanIROpKind::CaptureNormedPatchTokens:
      return StageKind::Capture;
    case VulkanIROpKind::TokensToFeatureMap:
    case VulkanIROpKind::DecoderProject:
    case VulkanIROpKind::DecoderResize:
    case VulkanIROpKind::DecoderPreprocess:
    case VulkanIROpKind::DecoderHead:
      return StageKind::Decoder;
    case VulkanIROpKind::InputImage:
    case VulkanIROpKind::OutputAlias:
      return StageKind::Export;
  }
  return StageKind::Unknown;
}

DispatchKind dispatch_kind_for_ir_op_kind(const VulkanIROpKind kind) {
  switch (kind) {
    case VulkanIROpKind::PatchEmbed:
      return DispatchKind::PatchEmbed;
    case VulkanIROpKind::FeatureMapToTokens:
      return DispatchKind::FeatureMapToTokens;
    case VulkanIROpKind::ElementwiseAdd:
      return DispatchKind::ElementwiseAdd;
    case VulkanIROpKind::Concat:
      return DispatchKind::Concat;
    case VulkanIROpKind::PatchTokenInput:
      return DispatchKind::PatchTokenInput;
    case VulkanIROpKind::BackboneBlock:
      return DispatchKind::BackboneBlock;
    case VulkanIROpKind::CapturePatchTokens:
      return DispatchKind::CapturePatchTokens;
    case VulkanIROpKind::CaptureNormedPatchTokens:
      return DispatchKind::CaptureNormedPatchTokens;
    case VulkanIROpKind::TokensToFeatureMap:
      return DispatchKind::TokensToFeatureMap;
    case VulkanIROpKind::DecoderProject:
      return DispatchKind::DecoderProject;
    case VulkanIROpKind::DecoderResize:
      return DispatchKind::DecoderResize;
    case VulkanIROpKind::DecoderPreprocess:
      return DispatchKind::DecoderPreprocess;
    case VulkanIROpKind::DecoderHead:
      return DispatchKind::DecoderHead;
    case VulkanIROpKind::InputImage:
    case VulkanIROpKind::OutputAlias:
      return DispatchKind::Unknown;
  }
  return DispatchKind::Unknown;
}

struct FusedExecutableImageEntryPattern final {
  DispatchStep step;
  std::array<VulkanValueId, 2u> virtual_values{};
};

struct FusedExecutableImagePatchEntryPattern final {
  DispatchStep step;
  std::array<VulkanValueId, 3u> virtual_values{};
  size_t consumed_ops{0u};
};

std::optional<FusedExecutableImagePatchEntryPattern>
try_make_fused_image_patch_token_input_step(
    const std::vector<VulkanIROpNode>& ops,
    const size_t op_idx) {
  if (op_idx + 3u >= ops.size()) {
    return std::nullopt;
  }

  const auto& patch_embed = ops[op_idx];
  const auto& feature_to_tokens = ops[op_idx + 1u];
  const auto& add = ops[op_idx + 2u];
  const auto& concat = ops[op_idx + 3u];
  if (
      patch_embed.kind != VulkanIROpKind::PatchEmbed ||
      feature_to_tokens.kind != VulkanIROpKind::FeatureMapToTokens ||
      add.kind != VulkanIROpKind::ElementwiseAdd ||
      concat.kind != VulkanIROpKind::Concat) {
    return std::nullopt;
  }
  if (
      patch_embed.inputs.size() != 1u || patch_embed.outputs.size() != 1u ||
      patch_embed.constants.size() != 1u ||
      feature_to_tokens.inputs.size() != 1u ||
      feature_to_tokens.outputs.size() != 1u ||
      !feature_to_tokens.constants.empty() ||
      add.inputs.size() != 2u || add.outputs.size() != 1u || !add.constants.empty() ||
      concat.inputs.size() != 2u || concat.outputs.size() != 1u ||
      !concat.constants.empty()) {
    return std::nullopt;
  }

  const VulkanValueId patch_feature_map_value = patch_embed.outputs[0];
  const VulkanValueId feature_tokens_value = feature_to_tokens.outputs[0];
  const VulkanValueId positioned_tokens_value = add.outputs[0];
  if (
      feature_to_tokens.inputs[0] != patch_feature_map_value ||
      add.inputs[0] != feature_tokens_value ||
      concat.inputs[1] != positioned_tokens_value) {
    return std::nullopt;
  }

  DispatchStep step;
  step.ir_op_index = static_cast<uint32_t>(op_idx);
  step.name = concat.name.empty() ? "patch_tokens.image_input" : concat.name;
  step.program_key = "ImagePatchTokenInput";
  step.dispatch_kind = DispatchKind::ImagePatchTokenInput;
  step.attributes_key = concat.attributes_key;
  step.reads = {patch_embed.inputs[0]};
  step.constants = {concat.inputs[0], add.inputs[1]};
  step.writes = {concat.outputs[0]};
  return FusedExecutableImagePatchEntryPattern{
      std::move(step),
      {patch_feature_map_value, feature_tokens_value, positioned_tokens_value},
      4u};
}

std::optional<FusedExecutableImageEntryPattern>
try_make_fused_patch_token_input_step(
    const std::vector<VulkanIROpNode>& ops,
    const size_t op_idx) {
  if (op_idx + 2u >= ops.size()) {
    return std::nullopt;
  }

  const auto& feature_to_tokens = ops[op_idx];
  const auto& add = ops[op_idx + 1u];
  const auto& concat = ops[op_idx + 2u];
  if (
      feature_to_tokens.kind != VulkanIROpKind::FeatureMapToTokens ||
      add.kind != VulkanIROpKind::ElementwiseAdd ||
      concat.kind != VulkanIROpKind::Concat) {
    return std::nullopt;
  }
  if (
      feature_to_tokens.inputs.size() != 1u ||
      feature_to_tokens.outputs.size() != 1u || !feature_to_tokens.constants.empty() ||
      add.inputs.size() != 2u || add.outputs.size() != 1u || !add.constants.empty() ||
      concat.inputs.size() != 2u || concat.outputs.size() != 1u ||
      !concat.constants.empty()) {
    return std::nullopt;
  }

  const VulkanValueId feature_tokens_value = feature_to_tokens.outputs[0];
  const VulkanValueId positioned_tokens_value = add.outputs[0];
  if (
      add.inputs[0] != feature_tokens_value ||
      concat.inputs[1] != positioned_tokens_value) {
    return std::nullopt;
  }

  DispatchStep step;
  step.ir_op_index = static_cast<uint32_t>(op_idx);
  step.name = concat.name.empty() ? "patch_tokens.input" : concat.name;
  step.program_key = "PatchTokenInput";
  step.dispatch_kind = DispatchKind::PatchTokenInput;
  step.attributes_key = concat.attributes_key;
  step.reads = {feature_to_tokens.inputs[0]};
  step.constants = {concat.inputs[0], add.inputs[1]};
  step.writes = {concat.outputs[0]};
  return FusedExecutableImageEntryPattern{
      std::move(step), {feature_tokens_value, positioned_tokens_value}};
}

struct FusedExecutableDecoderLayerPattern final {
  DispatchStep step;
  std::vector<VulkanValueId> virtual_values;
  size_t consumed_ops{0u};
};

struct FusedExecutableCaptureDecoderLayerPattern final {
  DispatchStep step;
  std::vector<VulkanValueId> virtual_values;
  size_t decoder_op_idx{0u};
  size_t decoder_consumed_ops{0u};
};

std::optional<FusedExecutableDecoderLayerPattern>
try_make_fused_decoder_layer_step(
    const std::vector<VulkanIROpNode>& ops,
    const size_t op_idx) {
  if (op_idx + 2u >= ops.size()) {
    return std::nullopt;
  }

  const auto& tokens_to_feature_map = ops[op_idx];
  const auto& project = ops[op_idx + 1u];
  if (
      tokens_to_feature_map.kind != VulkanIROpKind::TokensToFeatureMap ||
      project.kind != VulkanIROpKind::DecoderProject) {
    return std::nullopt;
  }
  if (
      tokens_to_feature_map.inputs.size() != 1u ||
      tokens_to_feature_map.outputs.size() != 1u ||
      !tokens_to_feature_map.constants.empty() ||
      project.inputs.size() != 1u || project.outputs.size() != 1u ||
      project.constants.size() != 1u ||
      project.inputs[0] != tokens_to_feature_map.outputs[0]) {
    return std::nullopt;
  }

  size_t cursor = op_idx + 2u;
  bool apply_resize = false;
  const VulkanIROpNode* resize = nullptr;
  if (cursor < ops.size() && ops[cursor].kind == VulkanIROpKind::DecoderResize) {
    resize = &ops[cursor];
    apply_resize = true;
    if (
        resize->inputs.size() != 1u || resize->outputs.size() != 1u ||
        resize->constants.size() != 1u ||
        resize->inputs[0] != project.outputs[0]) {
      return std::nullopt;
    }
    ++cursor;
  }
  if (cursor >= ops.size()) {
    return std::nullopt;
  }

  const auto& preprocess = ops[cursor];
  if (
      preprocess.kind != VulkanIROpKind::DecoderPreprocess ||
      preprocess.inputs.size() != 1u || preprocess.outputs.size() != 1u ||
      preprocess.constants.size() != 1u ||
      preprocess.inputs[0] !=
          (apply_resize ? resize->outputs[0] : project.outputs[0])) {
    return std::nullopt;
  }

  DispatchStep step;
  step.ir_op_index = static_cast<uint32_t>(op_idx);
  step.name = preprocess.name.empty() ? "decoder.layer.preprocess" : preprocess.name;
  step.program_key = "DecoderLayerPreprocess";
  step.dispatch_kind = DispatchKind::DecoderLayerPreprocess;
  step.attributes_key = preprocess.attributes_key;
  step.reads = {tokens_to_feature_map.inputs[0]};
  step.constants = {project.constants[0]};
  if (apply_resize) {
    step.constants.push_back(resize->constants[0]);
  }
  step.constants.push_back(preprocess.constants[0]);
  step.temporaries = {tokens_to_feature_map.outputs[0], project.outputs[0]};
  if (apply_resize) {
    step.temporaries.push_back(resize->outputs[0]);
  }
  step.writes = {preprocess.outputs[0]};

  std::vector<VulkanValueId> virtual_values = {
      tokens_to_feature_map.outputs[0], project.outputs[0]};
  if (apply_resize) {
    virtual_values.push_back(resize->outputs[0]);
  }
  return FusedExecutableDecoderLayerPattern{
      std::move(step), std::move(virtual_values), cursor - op_idx + 1u};
}

std::optional<FusedExecutableCaptureDecoderLayerPattern>
try_make_fused_capture_decoder_layer_step(
    const std::vector<VulkanIROpNode>& ops,
    const size_t op_idx) {
  if (op_idx >= ops.size()) {
    return std::nullopt;
  }

  const auto& capture = ops[op_idx];
  const bool capture_is_normed =
      capture.kind == VulkanIROpKind::CaptureNormedPatchTokens;
  const bool capture_is_plain =
      capture.kind == VulkanIROpKind::CapturePatchTokens;
  if (
      (!capture_is_normed && !capture_is_plain) ||
      capture.inputs.size() != 1u || capture.outputs.size() != 1u ||
      capture.constants.size() != (capture_is_normed ? 1u : 0u)) {
    return std::nullopt;
  }

  for (size_t decoder_op_idx = op_idx + 1u; decoder_op_idx < ops.size();
       ++decoder_op_idx) {
    if (
        ops[decoder_op_idx].kind != VulkanIROpKind::TokensToFeatureMap ||
        ops[decoder_op_idx].inputs.size() != 1u ||
        ops[decoder_op_idx].inputs[0] != capture.outputs[0]) {
      continue;
    }
    auto fused_decoder = try_make_fused_decoder_layer_step(ops, decoder_op_idx);
    if (
        !fused_decoder.has_value() ||
        fused_decoder->step.reads.size() != 1u ||
        fused_decoder->step.reads[0] != capture.outputs[0]) {
      return std::nullopt;
    }

    DispatchStep step;
    step.ir_op_index = static_cast<uint32_t>(op_idx);
    step.name = fused_decoder->step.name.empty()
        ? (capture_is_normed ? "capture.decoder.layer.preprocess"
                             : "decoder.layer.preprocess")
        : fused_decoder->step.name;
    step.program_key = capture_is_normed ? "CaptureDecoderLayerPreprocess"
                                         : fused_decoder->step.program_key;
    step.dispatch_kind = capture_is_normed
        ? DispatchKind::CaptureDecoderLayerPreprocess
        : fused_decoder->step.dispatch_kind;
    step.attributes_key = fused_decoder->step.attributes_key;
    step.reads = {capture.inputs[0]};
    if (capture_is_normed) {
      step.constants = {capture.constants[0]};
      step.constants.insert(
          step.constants.end(),
          fused_decoder->step.constants.cbegin(),
          fused_decoder->step.constants.cend());
      step.temporaries = {capture.outputs[0]};
      step.temporaries.insert(
          step.temporaries.end(),
          fused_decoder->step.temporaries.cbegin(),
          fused_decoder->step.temporaries.cend());
    } else {
      step.constants = fused_decoder->step.constants;
      step.temporaries = fused_decoder->step.temporaries;
    }
    step.writes = fused_decoder->step.writes;

    std::vector<VulkanValueId> virtual_values = {capture.outputs[0]};
    virtual_values.insert(
        virtual_values.end(),
        fused_decoder->virtual_values.cbegin(),
        fused_decoder->virtual_values.cend());
    return FusedExecutableCaptureDecoderLayerPattern{
        std::move(step),
        std::move(virtual_values),
        decoder_op_idx,
        fused_decoder->consumed_ops};
  }

  return std::nullopt;
}

ViewTransformKind infer_view_transform_kind(
    const VulkanIRValue& value,
    const std::optional<VulkanIROutputAlias>& output_alias) {
  if (output_alias.has_value()) {
    return ViewTransformKind::Reinterpret;
  }
  if (value.spec.logical_sizes != value.spec.padded_sizes) {
    return ViewTransformKind::Reshape;
  }
  return ViewTransformKind::Identity;
}

std::optional<std::shared_ptr<VulkanExecutableRegion>>
maybe_lower_compiled_session_to_executable_region(
    const VulkanCompiledSession& session) {
  if (!session.defined() || !session.executable()) {
    return std::nullopt;
  }

  switch (session.key().kind) {
    case VulkanCompiledSessionKind::VisionTransformerDepth:
    case VulkanCompiledSessionKind::VisionTransformerDepthImage:
    case VulkanCompiledSessionKind::VisionTransformerDepthBackbone:
    case VulkanCompiledSessionKind::VisionTransformerDepthDecoderPreprocessHead:
      break;
  }

  const auto bindings = make_compiled_session_tensor_bindings(session);
  if (!bindings.has_value()) {
    return std::nullopt;
  }

  const auto& ir = session.ir();
  const auto& values = ir.values();
  const auto& lifetimes = ir.lifetimes();
  const auto& memory_plan = session.memory_plan();
  const auto& binding_table = *bindings;

  auto region = std::make_shared<VulkanExecutableRegion>();
  region->key = session.key().model_key + "|" + session.key().configuration_key +
      "|" + session.key().capability_key;
  region->contract.dtype = session.key().dtype;
  region->contract.storage_type = session.layout_plan().storage_type;
  region->contract.memory_layout = session.layout_plan().memory_layout;
  region->contract.execution_layout = session.layout_plan().execution_layout;
  region->contract.width_alignment = session.layout_plan().width_alignment;
  region->contract.pad_width = session.layout_plan().pad_width;
  region->contract.capability_key = session.key().capability_key;
  region->contract.debug_name = std::string(
      compiled_session_family_name(
          compiled_session_family_for_kind(session.key().kind))) +
      "." + compiled_session_kind_name(session.key().kind);

  region->slots.reserve(binding_table.slot_values.size());
  const auto memory_slot_for_value = [&](const VulkanValueId value_id)
      -> std::optional<size_t> {
    for (size_t idx = 0u; idx < memory_plan.slots.size(); ++idx) {
      const auto& slot = memory_plan.slots[idx];
      if (std::find(slot.values.begin(), slot.values.end(), value_id) !=
          slot.values.end()) {
        return idx;
      }
    }
    return std::nullopt;
  };

  for (size_t slot_idx = 0u; slot_idx < binding_table.slot_values.size();
       ++slot_idx) {
    const VulkanValueId value_id = binding_table.slot_values[slot_idx];
    TORCH_INTERNAL_ASSERT(
        value_id < values.size(),
        "Executable region slot lowering references an invalid value");
    const auto& value = values[value_id];
    const auto source_memory_slot = memory_slot_for_value(value_id);
    PhysicalSlot slot;
    slot.id = slot_idx;
    slot.source_memory_slot = source_memory_slot;
    slot.storage_dtype = value.spec.dtype;
    slot.storage_type = value.spec.storage_type;
    slot.storage_layout = value.spec.memory_layout;
    slot.physical_sizes = value.spec.padded_sizes.empty()
        ? value.spec.logical_sizes
        : value.spec.padded_sizes;
    slot.byte_size = compiled_session_impl::tensor_spec_nbytes(value.spec);
    slot.memory_class = value.spec.external ? MemoryClass::External
                                            : MemoryClass::DeviceLocal;
    slot.alignment = 1u;
    slot.dedicated = source_memory_slot.has_value()
        ? memory_plan.slots[*source_memory_slot].dedicated
        : value.spec.external;
    slot.external = value.spec.external;
    region->slots.push_back(std::move(slot));
  }

  region->values.reserve(values.size());
  const auto alias_for_value = [&](const VulkanValueId value_id)
      -> std::optional<VulkanIROutputAlias> {
    for (const auto& alias : ir.output_aliases()) {
      if (alias.output == value_id) {
        return alias;
      }
    }
    return std::nullopt;
  };

  std::unordered_set<VulkanValueId> virtualized_values;
  std::unordered_map<size_t, FusedExecutableCaptureDecoderLayerPattern>
      fused_capture_decoder_patterns;
  std::unordered_map<size_t, size_t> consumed_decoder_fusion_starts;
  for (size_t op_idx = 0u; op_idx < ir.ops().size(); ++op_idx) {
    if (const auto fused_image_patch =
            try_make_fused_image_patch_token_input_step(ir.ops(), op_idx);
        fused_image_patch.has_value()) {
      virtualized_values.insert(
          fused_image_patch->virtual_values.cbegin(),
          fused_image_patch->virtual_values.cend());
      op_idx += fused_image_patch->consumed_ops - 1u;
      continue;
    }
    if (const auto fused_capture_decoder =
            try_make_fused_capture_decoder_layer_step(ir.ops(), op_idx);
        fused_capture_decoder.has_value()) {
      virtualized_values.insert(
          fused_capture_decoder->virtual_values.cbegin(),
          fused_capture_decoder->virtual_values.cend());
      consumed_decoder_fusion_starts.emplace(
          fused_capture_decoder->decoder_op_idx,
          fused_capture_decoder->decoder_consumed_ops);
      fused_capture_decoder_patterns.emplace(
          op_idx, std::move(*fused_capture_decoder));
      continue;
    }
    if (const auto consumed = consumed_decoder_fusion_starts.find(op_idx);
        consumed != consumed_decoder_fusion_starts.end()) {
      op_idx += consumed->second - 1u;
      continue;
    }
    if (const auto fused_patch =
            try_make_fused_patch_token_input_step(ir.ops(), op_idx);
        fused_patch.has_value()) {
      virtualized_values.insert(fused_patch->virtual_values[0]);
      virtualized_values.insert(fused_patch->virtual_values[1]);
      op_idx += 2u;
      continue;
    }
    if (const auto fused_decoder =
            try_make_fused_decoder_layer_step(ir.ops(), op_idx);
        fused_decoder.has_value()) {
      virtualized_values.insert(
          fused_decoder->virtual_values.begin(),
          fused_decoder->virtual_values.end());
      op_idx += fused_decoder->consumed_ops - 1u;
    }
  }

  for (const auto& value : values) {
    LoweredValue lowered;
    lowered.ir_value = value.id;
    lowered.name = value.name;
    lowered.boundary_role =
        (value.spec.role == VulkanIRTensorRole::Output && value.spec.external)
        ? BoundaryRole::RegionOutput
        : BoundaryRole::Internal;

    const auto alias = alias_for_value(value.id);
    if (value.spec.role == VulkanIRTensorRole::Input && value.spec.external) {
      lowered.realization = RealizationKind::ExternalInput;
    } else if (value.spec.role == VulkanIRTensorRole::Constant) {
      lowered.realization = RealizationKind::Constant;
    } else if (virtualized_values.count(value.id) > 0u) {
      lowered.realization = RealizationKind::Virtual;
    } else if (alias.has_value()) {
      lowered.realization = RealizationKind::View;
      lowered.base = alias->source;
    } else {
      lowered.realization = RealizationKind::Materialized;
    }

    if (
        lowered.realization != RealizationKind::Virtual &&
        value.id < binding_table.value_tensor_slots.size() &&
        binding_table.value_tensor_slots[value.id].has_value()) {
      lowered.slot = *binding_table.value_tensor_slots[value.id];
      lowered.view.slot = lowered.slot;
    }
    lowered.view.logical_dtype = value.spec.dtype;
    lowered.view.logical_sizes = value.spec.logical_sizes;
    const auto logical_strides =
        c10::contiguous_strides(value.spec.logical_sizes);
    lowered.view.logical_strides.assign(
        logical_strides.begin(), logical_strides.end());
    lowered.view.storage_offset = 0;
    lowered.view.transform = infer_view_transform_kind(value, alias);
    lowered.first_use_step = value.id < lifetimes.size()
        ? static_cast<uint32_t>(lifetimes[value.id].first_op)
        : 0u;
    lowered.last_use_step = value.id < lifetimes.size()
        ? static_cast<uint32_t>(lifetimes[value.id].last_op)
        : 0u;
    region->values.push_back(std::move(lowered));
  }

  region->steps.reserve(ir.ops().size() + values.size());
  std::optional<StageRange> current_stage = std::nullopt;
  for (size_t op_idx = 0u; op_idx < ir.ops().size(); ++op_idx) {
    if (const auto consumed = consumed_decoder_fusion_starts.find(op_idx);
        consumed != consumed_decoder_fusion_starts.end()) {
      op_idx += consumed->second - 1u;
      continue;
    }
    const auto& op = ir.ops()[op_idx];
    const StageKind stage_kind = executable_stage_kind_for_op(op);
    if (
        !current_stage.has_value() || current_stage->kind != stage_kind) {
      if (current_stage.has_value()) {
        current_stage->end_step =
            static_cast<uint32_t>(region->steps.size());
        region->stages.push_back(*current_stage);
      }
      current_stage = StageRange{
          stage_kind,
          static_cast<uint32_t>(region->steps.size()),
          static_cast<uint32_t>(region->steps.size()),
          std::nullopt};
    }

    if (op.kind == VulkanIROpKind::OutputAlias) {
      continue;
    }

    if (const auto fused_capture_decoder =
            fused_capture_decoder_patterns.find(op_idx);
        fused_capture_decoder != fused_capture_decoder_patterns.end()) {
      region->steps.push_back(ExecStep{
          ExecOpcode::Dispatch, fused_capture_decoder->second.step});
      continue;
    }
    if (const auto fused_image_patch =
            try_make_fused_image_patch_token_input_step(ir.ops(), op_idx);
        fused_image_patch.has_value()) {
      region->steps.push_back(
          ExecStep{ExecOpcode::Dispatch, std::move(fused_image_patch->step)});
      op_idx += fused_image_patch->consumed_ops - 1u;
      continue;
    }
    if (const auto fused =
            try_make_fused_patch_token_input_step(ir.ops(), op_idx);
        fused.has_value()) {
      region->steps.push_back(
          ExecStep{ExecOpcode::Dispatch, std::move(fused->step)});
      op_idx += 2u;
      continue;
    }
    if (const auto fused_decoder =
            try_make_fused_decoder_layer_step(ir.ops(), op_idx);
        fused_decoder.has_value()) {
      region->steps.push_back(
          ExecStep{ExecOpcode::Dispatch, std::move(fused_decoder->step)});
      op_idx += fused_decoder->consumed_ops - 1u;
      continue;
    }

    DispatchStep step;
    step.ir_op_index = static_cast<uint32_t>(op_idx);
    step.name = op.name;
    step.program_key = ir_op_kind_name(op.kind);
    step.dispatch_kind = dispatch_kind_for_ir_op_kind(op.kind);
    step.attributes_key = op.attributes_key;
    step.reads.reserve(op.inputs.size());
    for (const VulkanValueId input : op.inputs) {
      step.reads.push_back(input);
    }
    step.constants.reserve(op.constants.size());
    for (const VulkanValueId constant : op.constants) {
      step.constants.push_back(constant);
    }
    step.writes.reserve(op.outputs.size());
    for (const VulkanValueId output : op.outputs) {
      step.writes.push_back(output);
    }
    region->steps.push_back(
        ExecStep{ExecOpcode::Dispatch, std::move(step)});
  }

  if (current_stage.has_value()) {
    current_stage->end_step = static_cast<uint32_t>(region->steps.size());
    region->stages.push_back(*current_stage);
  }

  size_t output_index = 0u;
  for (const auto& value : region->values) {
    if (value.boundary_role != BoundaryRole::RegionOutput) {
      continue;
    }
    region->outputs.push_back(RegionOutputBinding{
        value.ir_value,
        output_index,
        value.name});
    region->steps.push_back(ExecStep{
        ExecOpcode::Export,
        ExportStep{value.ir_value, output_index, value.name}});
    ++output_index;
  }
  if (!region->outputs.empty()) {
    region->stages.push_back(StageRange{
        StageKind::Export,
        static_cast<uint32_t>(region->steps.size() - region->outputs.size()),
        static_cast<uint32_t>(region->steps.size()),
        std::nullopt});
  }

  return region;
}

VulkanCompiledSession make_compiled_session(
    VulkanCompiledSessionKey key,
    VulkanBackendIR ir,
    VulkanGlobalLayoutPlan layout_plan,
    const bool executable) {
  apply_global_layout_plan(ir, layout_plan);
  ir.recompute_lifetimes();
  VulkanIRMemoryPlan memory_plan = make_memory_plan(ir);
  auto state = std::make_shared<VulkanCompiledSession::State>(
      std::move(key),
      std::move(ir),
      std::move(layout_plan),
      std::move(memory_plan),
      nullptr,
      executable);
  VulkanCompiledSession session{state};
  if (executable) {
    const auto region =
        compiled_session_impl::maybe_lower_compiled_session_to_executable_region(
            session);
    if (region.has_value()) {
      state->executable_region = *region;
    }
  }
  return session;
}

} // namespace compiled_session_impl

const char* compiled_session_family_name(
    const VulkanCompiledSessionFamily family) {
  switch (family) {
    case VulkanCompiledSessionFamily::VisionTransformerDepth:
      return "VisionTransformerDepth";
    case VulkanCompiledSessionFamily::DiffusionUNet:
      return "DiffusionUNet";
    case VulkanCompiledSessionFamily::HybridPipeline:
      return "HybridPipeline";
  }
  return "UnknownCompiledSessionFamily";
}

VulkanCompiledSessionFamily compiled_session_family_for_kind(
    const VulkanCompiledSessionKind kind) {
  switch (kind) {
    case VulkanCompiledSessionKind::VisionTransformerDepth:
    case VulkanCompiledSessionKind::VisionTransformerDepthImage:
    case VulkanCompiledSessionKind::VisionTransformerDepthBackbone:
    case VulkanCompiledSessionKind::VisionTransformerDepthDecoderPreprocessHead:
      return VulkanCompiledSessionFamily::VisionTransformerDepth;
  }
  return VulkanCompiledSessionFamily::VisionTransformerDepth;
}

const char* compiled_session_kind_name(const VulkanCompiledSessionKind kind) {
  switch (kind) {
    case VulkanCompiledSessionKind::VisionTransformerDepth:
      return "VisionTransformerDepth";
    case VulkanCompiledSessionKind::VisionTransformerDepthImage:
      return "VisionTransformerDepthImage";
    case VulkanCompiledSessionKind::VisionTransformerDepthBackbone:
      return "VisionTransformerDepthBackbone";
    case VulkanCompiledSessionKind::VisionTransformerDepthDecoderPreprocessHead:
      return "VisionTransformerDepthDecoderPreprocessHead";
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
    case VulkanIROpKind::CapturePatchTokens:
      return "CapturePatchTokens";
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
      lhs.layout_policy_version == rhs.layout_policy_version &&
      lhs.model_lane_policy_version == rhs.model_lane_policy_version &&
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
  compiled_session_impl::hash_combine_session(
      seed, key.layout_policy_version);
  compiled_session_impl::hash_combine_session(
      seed, key.model_lane_policy_version);
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

const VulkanExecutableRegion* VulkanCompiledSession::executable_region() const {
  TORCH_INTERNAL_ASSERT(state_, "VulkanCompiledSession is not defined");
  return state_->executable_region.get();
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

std::optional<VulkanCompiledSessionTensorBindings>
make_compiled_executable_region_tensor_bindings(
    const VulkanCompiledSession& session,
    const VulkanExecutableRegion& region) {
  const auto& ir = session.ir();
  auto base_bindings = make_compiled_session_tensor_bindings(session);
  if (!base_bindings.has_value()) {
    return std::nullopt;
  }
  if (!region.defined()) {
    return base_bindings;
  }

  VulkanCompiledSessionTensorBindings bindings;
  bindings.value_tensor_slots.resize(base_bindings->value_tensor_slots.size());
  std::unordered_map<VulkanValueId, VulkanValueId> output_alias_sources;
  output_alias_sources.reserve(ir.output_aliases().size());
  for (const auto& alias : ir.output_aliases()) {
    output_alias_sources.emplace(alias.output, alias.source);
  }
  std::unordered_set<VulkanValueId> decoder_head_inputs;
  for (const auto& op : ir.ops()) {
    if (op.kind != VulkanIROpKind::DecoderHead) {
      continue;
    }
    decoder_head_inputs.insert(op.inputs.cbegin(), op.inputs.cend());
  }

  const auto is_virtual = [&](const VulkanValueId value_id) {
    return value_id < region.values.size() &&
        region.values[value_id].realization == RealizationKind::Virtual;
  };
  const auto alias_source_for_value = [&](const VulkanValueId value_id)
      -> std::optional<VulkanValueId> {
    const auto it = output_alias_sources.find(value_id);
    if (it == output_alias_sources.end()) {
      return std::nullopt;
    }
    return it->second;
  };
  const auto requires_dedicated_tensor_slot = [&](const VulkanValueId value_id) {
    if (value_id >= region.values.size() || is_virtual(value_id)) {
      return false;
    }
    const auto& lowered_value = region.values[value_id];
    if (decoder_head_inputs.count(value_id) > 0u) {
      return true;
    }
    if (alias_source_for_value(value_id).has_value()) {
      return lowered_value.realization == RealizationKind::ExternalInput;
    }
    return lowered_value.boundary_role == BoundaryRole::RegionOutput ||
        lowered_value.realization == RealizationKind::ExternalInput;
  };
  const auto bind_value = [&](const VulkanValueId value_id, const size_t slot_idx) {
    TORCH_INTERNAL_ASSERT(
        value_id < bindings.value_tensor_slots.size(),
        "Executable region tensor binding references an invalid value id");
    auto& mapped_slot = bindings.value_tensor_slots[value_id];
    TORCH_INTERNAL_ASSERT(
        !mapped_slot.has_value() || *mapped_slot == slot_idx,
        "Executable region value was assigned conflicting tensor slots");
    mapped_slot = slot_idx;
  };

  std::vector<std::vector<VulkanValueId>> old_slot_members(
      base_bindings->tensor_slot_count());
  for (size_t value_id = 0u; value_id < base_bindings->value_tensor_slots.size();
       ++value_id) {
    if (!base_bindings->value_tensor_slots[value_id].has_value()) {
      continue;
    }
    old_slot_members[*base_bindings->value_tensor_slots[value_id]].push_back(
        static_cast<VulkanValueId>(value_id));
  }

  for (size_t old_slot_idx = 0u; old_slot_idx < old_slot_members.size();
       ++old_slot_idx) {
    std::vector<VulkanValueId> shared_members;
    for (const VulkanValueId value_id : old_slot_members[old_slot_idx]) {
      if (is_virtual(value_id)) {
        continue;
      }
      if (requires_dedicated_tensor_slot(value_id)) {
        const size_t dedicated_slot_idx = bindings.slot_values.size();
        bindings.slot_values.push_back(value_id);
        bind_value(value_id, dedicated_slot_idx);
        continue;
      }
      shared_members.push_back(value_id);
    }

    if (shared_members.empty()) {
      continue;
    }

    const size_t new_slot_idx = bindings.slot_values.size();
    bindings.slot_values.push_back(shared_members.front());
    for (const VulkanValueId value_id : shared_members) {
      bind_value(value_id, new_slot_idx);
    }
  }

  for (const VulkanValueId input_value : base_bindings->input_values) {
    if (is_virtual(input_value)) {
      continue;
    }
    bindings.input_values.push_back(input_value);
  }

  for (const auto& alias : ir.output_aliases()) {
    if (is_virtual(alias.output)) {
      continue;
    }
    if (alias.output >= bindings.value_tensor_slots.size() ||
        alias.source >= bindings.value_tensor_slots.size() ||
        !bindings.value_tensor_slots[alias.source].has_value()) {
      return std::nullopt;
    }
    bind_value(alias.output, *bindings.value_tensor_slots[alias.source]);
  }

  const auto& values = ir.values();
  for (const auto& value : values) {
    if (
        compiled_session_impl::tensor_spec_nbytes(value.spec) == 0u ||
        is_virtual(value.id) || bindings.value_tensor_slots[value.id].has_value()) {
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
      << "|vk13=" << (profile.has_vulkan_1_3 ? 1 : 0)
      << "|m4=" << (profile.has_maintenance4 ? 1 : 0)
      << "|sync2=" << (profile.has_synchronization2 ? 1 : 0)
      << "|zero_wg="
      << (profile.has_shader_zero_initialize_workgroup_memory ? 1 : 0)
      << "|dot=" << (profile.has_shader_integer_dot_product ? 1 : 0)
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

VulkanCompiledSession lookup_or_create_vision_transformer_depth_session(
    const VisionTransformerDepthSessionDesc& desc) {
  const VulkanRuntimeCapabilityProfile profile =
      query_vulkan_runtime_capability_profile();
  VulkanCompiledSessionKey key;
  key.kind = VulkanCompiledSessionKind::VisionTransformerDepth;
  key.model_key =
      desc.model_key.empty() ? "vision_transformer_depth.compiled_session"
                             : desc.model_key;
  key.configuration_key =
      compiled_session_impl::
          make_vision_transformer_depth_full_configuration_key(desc);
  key.input_shapes = {desc.patch_token_sizes};
  key.output_shapes = {desc.output_sizes};
  key.dtype = desc.dtype;
  key.capability_key = make_vulkan_compiled_session_capability_key(profile);
  key.persistent = desc.persistent;

  return lookup_or_create_vulkan_compiled_session(key, [&]() {
    VulkanBackendIR ir =
        compiled_session_impl::make_vision_transformer_depth_full_ir(desc);
    VulkanGlobalLayoutPlan layout_plan =
        make_buffer_first_width_packed_layout_plan(key, profile);
    return compiled_session_impl::make_compiled_session(
        key,
        std::move(ir),
        std::move(layout_plan),
        /*executable=*/true);
  });
}

VulkanCompiledSession lookup_or_create_vision_transformer_depth_image_session(
    const VisionTransformerDepthImageSessionDesc& desc) {
  const VulkanRuntimeCapabilityProfile profile =
      query_vulkan_runtime_capability_profile();
  VulkanCompiledSessionKey key;
  key.kind = VulkanCompiledSessionKind::VisionTransformerDepthImage;
  key.model_key = desc.model_key.empty()
      ? "vision_transformer_depth.image_compiled_session"
      : desc.model_key;
  key.configuration_key =
      compiled_session_impl::
          make_vision_transformer_depth_image_configuration_key(desc);
  key.input_shapes = {desc.image_sizes};
  key.output_shapes = {desc.output_sizes};
  key.dtype = desc.dtype;
  key.capability_key = make_vulkan_compiled_session_capability_key(profile);
  key.persistent = desc.persistent;

  return lookup_or_create_vulkan_compiled_session(key, [&]() {
    VulkanBackendIR ir =
        compiled_session_impl::make_vision_transformer_depth_image_full_ir(desc);
    VulkanGlobalLayoutPlan layout_plan =
        make_buffer_first_width_packed_layout_plan(key, profile);
    return compiled_session_impl::make_compiled_session(
        key,
        std::move(ir),
        std::move(layout_plan),
        /*executable=*/true);
  });
}

VulkanCompiledSession lookup_or_create_vision_transformer_depth_backbone_session(
    const VisionTransformerDepthBackboneSessionDesc& desc) {
  const VulkanRuntimeCapabilityProfile profile =
      query_vulkan_runtime_capability_profile();
  VulkanCompiledSessionKey key;
  key.kind = VulkanCompiledSessionKind::VisionTransformerDepthBackbone;
  key.model_key = desc.model_key.empty() ? "vision_transformer_depth.backbone"
                                         : desc.model_key;
  key.configuration_key =
      compiled_session_impl::
          make_vision_transformer_depth_backbone_configuration_key(desc);
  key.input_shapes = {desc.patch_token_sizes};
  for (size_t idx = 0u; idx < desc.capture_indices.size(); ++idx) {
    key.output_shapes.push_back(desc.patch_token_sizes);
  }
  key.dtype = desc.dtype;
  key.capability_key = make_vulkan_compiled_session_capability_key(profile);
  key.persistent = desc.persistent;

  return lookup_or_create_vulkan_compiled_session(key, [&]() {
    VulkanBackendIR ir =
        compiled_session_impl::make_vision_transformer_depth_backbone_ir(desc);
    VulkanGlobalLayoutPlan layout_plan =
        make_buffer_first_width_packed_layout_plan(key, profile);
    return compiled_session_impl::make_compiled_session(
        key,
        std::move(ir),
        std::move(layout_plan),
        /*executable=*/true);
  });
}

VulkanCompiledSession lookup_or_create_vision_transformer_depth_decoder_session(
    const VisionTransformerDepthDecoderSessionDesc& desc) {
  const VulkanRuntimeCapabilityProfile profile =
      query_vulkan_runtime_capability_profile();
  VulkanCompiledSessionKey key;
  key.kind =
      VulkanCompiledSessionKind::VisionTransformerDepthDecoderPreprocessHead;
  key.model_key = desc.model_key.empty()
      ? "vision_transformer_depth.decoder_preprocess_head"
      : desc.model_key;
  key.configuration_key =
      compiled_session_impl::
          make_vision_transformer_depth_decoder_configuration_key(desc);
  for (const auto& shape : desc.layer_token_sizes) {
    key.input_shapes.push_back(shape);
  }
  key.output_shapes = {desc.output_sizes};
  key.dtype = desc.dtype;
  key.capability_key = make_vulkan_compiled_session_capability_key(profile);
  key.persistent = desc.persistent;

  return lookup_or_create_vulkan_compiled_session(key, [&]() {
    VulkanBackendIR ir = compiled_session_impl::
        make_vision_transformer_depth_decoder_ir(
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

VulkanCompiledSession lookup_or_create_depth_anything_v2_session(
    const DepthAnythingV2SessionDesc& desc) {
  auto generic_desc = desc;
  if (generic_desc.model_key.empty()) {
    generic_desc.model_key = "depth_anything_v2.compiled_session";
  }
  return lookup_or_create_vision_transformer_depth_session(generic_desc);
}

VulkanCompiledSession lookup_or_create_depth_anything_v2_image_session(
    const DepthAnythingV2ImageSessionDesc& desc) {
  auto generic_desc = desc;
  if (generic_desc.model_key.empty()) {
    generic_desc.model_key = "depth_anything_v2.image_compiled_session";
  }
  return lookup_or_create_vision_transformer_depth_image_session(generic_desc);
}

VulkanCompiledSession lookup_or_create_depth_anything_v2_backbone_stack_session(
    const DepthAnythingV2BackboneStackSessionDesc& desc) {
  auto generic_desc = desc;
  if (generic_desc.model_key.empty()) {
    generic_desc.model_key = "depth_anything_v2.backbone_stack";
  }
  return lookup_or_create_vision_transformer_depth_backbone_session(
      generic_desc);
}

VulkanCompiledSession
lookup_or_create_depth_anything_v2_decoder_preprocess_head_session(
    const DepthAnythingV2DecoderPreprocessHeadSessionDesc& desc) {
  auto generic_desc = desc;
  if (generic_desc.model_key.empty()) {
    generic_desc.model_key = "depth_anything_v2.decoder_preprocess_head";
  }
  return lookup_or_create_vision_transformer_depth_decoder_session(
      generic_desc);
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
