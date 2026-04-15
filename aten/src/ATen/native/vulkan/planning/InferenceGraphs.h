#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/planning/ExecutionPrograms.h>

#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

enum class VulkanInferenceGraphKind : uint8_t {
  AttentionRuntime = 0u,
  VisionBackbone,
  VisionDecoder,
  ExecutionGraphBundle,
};

const char* inference_graph_kind_name(VulkanInferenceGraphKind);

class InferenceGraph final {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;

 public:
  InferenceGraph() = default;
  explicit InferenceGraph(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  VulkanInferenceGraphKind kind() const;
  const std::string& allocation_label() const;
  void note_shared_scratch_requirement(
      size_t requested_bytes,
      uint32_t alignment,
      bool persistent) const;
  std::optional<ScratchArena> ensure_shared_scratch(
      size_t requested_bytes,
      uint32_t alignment,
      bool persistent) const;
  const void* identity() const;
};

class InferenceReplay final {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;

 public:
  InferenceReplay() = default;
  explicit InferenceReplay(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  VulkanInferenceGraphKind kind() const;
  const std::string& allocation_label() const;
  bool recorded() const;
  void record(const std::function<void()>& recorder) const;
  void submit(
      VkFence fence_handle = VK_NULL_HANDLE,
      bool final_use = false) const;
  const void* identity() const;
};

using ExecutionGraphProgramHandle = std::variant<
    std::monostate,
    AttentionRuntimeProgram,
    GatedDeltaSplitProgram,
    VisionBackboneProgram,
    VisionDecoderProgram>;

class ExecutionGraphTensorSlots final {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;

 public:
  ExecutionGraphTensorSlots() = default;
  explicit ExecutionGraphTensorSlots(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  size_t tensor_count() const;
  size_t optional_tensor_count() const;
  Tensor& tensor(size_t idx);
  const Tensor& tensor(size_t idx) const;
  std::optional<Tensor>& optional_tensor(size_t idx);
  const std::optional<Tensor>& optional_tensor(size_t idx) const;
  const void* identity() const;
};

class ExecutionGraphProgramSlots final {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;

 public:
  ExecutionGraphProgramSlots() = default;
  explicit ExecutionGraphProgramSlots(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  size_t size() const;
  ExecutionGraphProgramHandle& program(size_t idx);
  const ExecutionGraphProgramHandle& program(size_t idx) const;
  const void* identity() const;
};

class ExecutionGraphReplay final {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;

 public:
  ExecutionGraphReplay() = default;
  explicit ExecutionGraphReplay(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  bool recorded() const;
  const InferenceReplay& replay() const;
  ExecutionGraphTensorSlots& tensor_slots();
  const ExecutionGraphTensorSlots& tensor_slots() const;
  ExecutionGraphProgramSlots& program_slots();
  const ExecutionGraphProgramSlots& program_slots() const;
  const void* identity() const;
};

struct ExecutionGraphReplayStep final {
  ExecutionGraphReplay replay;
  std::function<void()> record_step;
};

ExecutionGraphReplayStep make_execution_graph_replay_step(
    ExecutionGraphReplay replay,
    std::function<void()> record_step);

class ExecutionGraphReplayBundle final {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;

 public:
  ExecutionGraphReplayBundle() = default;
  explicit ExecutionGraphReplayBundle(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  size_t size() const;
  bool recorded() const;
  void record() const;
  ExecutionGraphReplay& replay(size_t idx);
  const ExecutionGraphReplay& replay(size_t idx) const;
  void submit(
      VkFence fence_handle = VK_NULL_HANDLE,
      bool final_use = false) const;
  const void* identity() const;
};

class ExecutionGraphPlan final {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;

 public:
  ExecutionGraphPlan() = default;
  explicit ExecutionGraphPlan(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  VulkanInferenceGraphKind kind() const;
  const std::string& allocation_label() const;
  void note_shared_scratch_requirement(
      size_t requested_bytes,
      uint32_t alignment,
      bool persistent) const;
  std::optional<ScratchArena> ensure_shared_scratch(
      size_t requested_bytes,
      uint32_t alignment,
      bool persistent) const;
  ExecutionGraphProgramHandle lookup_or_create_program(
      const std::string& phase_key,
      const std::function<ExecutionGraphProgramHandle()>& builder) const;
  ExecutionGraphReplay lookup_or_create_replay(
      const std::string& phase_key,
      const std::function<ExecutionGraphReplay()>& builder) const;
  const void* identity() const;
};

class ExecutionGraphRoot final {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;

 public:
  ExecutionGraphRoot() = default;
  explicit ExecutionGraphRoot(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  const std::string& allocation_label() const;
  ExecutionGraphPlan lookup_or_create_phase_plan(
      VulkanInferenceGraphKind kind,
      const std::string& phase_key = std::string()) const;
  ExecutionGraphReplayBundle lookup_or_create_replay_bundle(
      const std::string& bundle_key,
      const std::function<ExecutionGraphReplayBundle()>& builder) const;
  const void* identity() const;
};

ExecutionGraphReplayBundle make_execution_graph_replay_bundle(
    const std::string& allocation_label,
    ScalarType dtype,
    bool persistent,
    std::vector<ExecutionGraphReplayStep> steps);

InferenceGraph lookup_or_create_labeled_inference_graph(
    const std::string& allocation_label,
    VulkanInferenceGraphKind kind,
    ScalarType dtype,
    bool persistent);

InferenceReplay lookup_or_create_labeled_inference_replay(
    const std::string& allocation_label,
    VulkanInferenceGraphKind kind,
    ScalarType dtype,
    bool persistent);

ExecutionGraphPlan lookup_or_create_labeled_execution_graph_plan(
    const std::string& allocation_label,
    VulkanInferenceGraphKind kind,
    ScalarType dtype,
    bool persistent);

ExecutionGraphRoot lookup_or_create_labeled_execution_graph_root(
    const std::string& allocation_label,
    ScalarType dtype,
    bool persistent);

class VisionBackboneInferenceReplay;
class VisionDecoderInferenceReplay;
class VisionDecoderHeadInferenceReplay;
class AttentionRuntimeInferenceReplay;

class AttentionRuntimeInferenceGraph final {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;

 public:
  AttentionRuntimeInferenceGraph() = default;
  explicit AttentionRuntimeInferenceGraph(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  void note_shared_scratch_requirement(
      size_t requested_bytes,
      uint32_t alignment,
      bool persistent) const;
  std::optional<ScratchArena> ensure_shared_scratch(
      size_t requested_bytes,
      uint32_t alignment,
      bool persistent) const;
  AttentionRuntimeProgram lookup_or_create_program(
      const std::string& allocation_label,
      VulkanAttentionKernelFamily kernel_family,
      const std::optional<VulkanKVCacheSpec>& key_cache_spec,
      const std::optional<VulkanKVCacheSpec>& value_cache_spec,
      const std::optional<VulkanScratchArenaSpec>& scratch_spec,
      int64_t key_sequence_length,
      int64_t value_sequence_length,
      const VulkanExecutionProgramPlanningDesc& program_plan) const;
  AttentionRuntimeInferenceReplay lookup_or_create_replay(
      const std::string& allocation_label,
      IntArrayRef query_sizes,
      IntArrayRef key_sizes,
      IntArrayRef value_sizes,
      VulkanAttentionKernelFamily kernel_family,
      const std::optional<VulkanKVCacheSpec>& key_cache_spec,
      const std::optional<VulkanKVCacheSpec>& value_cache_spec,
      const std::optional<VulkanScratchArenaSpec>& scratch_spec,
      int64_t key_sequence_length,
      int64_t value_sequence_length,
      const VulkanExecutionProgramPlanningDesc& program_plan) const;
  const void* identity() const;
};

class AttentionRuntimeInferenceReplay final {
 private:
  ExecutionGraphReplay graph_replay_;

 public:
  AttentionRuntimeInferenceReplay() = default;
  explicit AttentionRuntimeInferenceReplay(ExecutionGraphReplay graph_replay)
      : graph_replay_(std::move(graph_replay)) {}

  bool defined() const;
  bool recorded() const;
  const InferenceReplay& replay() const;
  const ExecutionGraphReplay& graph_replay() const;
  ExecutionGraphReplayStep phase_step(std::function<void()> record_step) const;
  const AttentionRuntimeProgram& program() const;
  AttentionRuntimeProgram& program();
  Tensor& query_slot();
  Tensor& key_slot();
  Tensor& value_slot();
  Tensor& output_slot();
  const void* identity() const;
};

class VisionBackboneInferenceGraph final {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;

 public:
  VisionBackboneInferenceGraph() = default;
  explicit VisionBackboneInferenceGraph(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  void note_shared_scratch_requirement(
      size_t requested_bytes,
      uint32_t alignment,
      bool persistent) const;
  std::optional<ScratchArena> ensure_shared_scratch(
      size_t requested_bytes,
      uint32_t alignment,
      bool persistent) const;
  VisionBackboneProgram lookup_or_create_program(
      const std::string& allocation_label,
      ScalarType dtype,
      int64_t batch_size,
      int64_t token_count,
      int64_t embed_dim,
      int64_t hidden_dim,
      int64_t num_heads,
      const VulkanExecutionProgramPlanningDesc& program_plan) const;
  VisionBackboneInferenceReplay lookup_or_create_replay(
      const std::string& allocation_label,
      IntArrayRef input_sizes,
      int64_t token_count,
      int64_t embed_dim,
      int64_t hidden_dim,
      int64_t num_heads,
      const VulkanExecutionProgramPlanningDesc& program_plan) const;
  const void* identity() const;
};

class VisionBackboneInferenceReplay final {
 private:
  ExecutionGraphReplay graph_replay_;

 public:
  VisionBackboneInferenceReplay() = default;
  explicit VisionBackboneInferenceReplay(ExecutionGraphReplay graph_replay)
      : graph_replay_(std::move(graph_replay)) {}

  bool defined() const;
  bool recorded() const;
  const InferenceReplay& replay() const;
  const ExecutionGraphReplay& graph_replay() const;
  ExecutionGraphReplayStep phase_step(std::function<void()> record_step) const;
  const VisionBackboneProgram& program() const;
  VisionBackboneProgram& program();
  Tensor& input_slot();
  Tensor& output_slot();
  const void* identity() const;
};

class VisionDecoderInferenceGraph final {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;

 public:
  VisionDecoderInferenceGraph() = default;
  explicit VisionDecoderInferenceGraph(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  void note_shared_scratch_requirement(
      size_t requested_bytes,
      uint32_t alignment,
      bool persistent) const;
  std::optional<ScratchArena> ensure_shared_scratch(
      size_t requested_bytes,
      uint32_t alignment,
      bool persistent) const;
  VisionDecoderProgram lookup_or_create_program(
      const std::string& allocation_label,
      IntArrayRef input_sizes,
      const std::optional<std::vector<int64_t>>& skip_sizes,
      IntArrayRef target_sizes,
      int64_t out_channels,
      bool allocate_intermediate_outputs,
      const VulkanExecutionProgramPlanningDesc& program_plan) const;
  VisionDecoderInferenceReplay lookup_or_create_replay(
      const std::string& allocation_label,
      IntArrayRef input_sizes,
      const std::optional<std::vector<int64_t>>& skip_sizes,
      IntArrayRef target_sizes,
      int64_t out_channels,
      const VulkanExecutionProgramPlanningDesc& program_plan) const;
  VisionDecoderHeadInferenceReplay lookup_or_create_head_replay(
      const std::string& allocation_label,
      IntArrayRef layer1_sizes,
      IntArrayRef layer2_sizes,
      IntArrayRef layer3_sizes,
      IntArrayRef layer4_sizes,
      IntArrayRef output_sizes,
      int64_t output_conv1_channels,
      int64_t output_conv2_channels,
      int64_t final_channels,
      const VulkanExecutionProgramPlanningDesc& program_plan) const;
  const void* identity() const;
};

class VisionDecoderInferenceReplay final {
 private:
  ExecutionGraphReplay graph_replay_;

 public:
  VisionDecoderInferenceReplay() = default;
  explicit VisionDecoderInferenceReplay(ExecutionGraphReplay graph_replay)
      : graph_replay_(std::move(graph_replay)) {}

  bool defined() const;
  bool recorded() const;
  const InferenceReplay& replay() const;
  const ExecutionGraphReplay& graph_replay() const;
  ExecutionGraphReplayStep phase_step(std::function<void()> record_step) const;
  const VisionDecoderProgram& program() const;
  VisionDecoderProgram& program();
  Tensor& input_slot();
  std::optional<Tensor>& skip_slot();
  const std::optional<Tensor>& skip_slot() const;
  Tensor& output_slot();
  const void* identity() const;
};

class VisionDecoderHeadInferenceReplay final {
 private:
  ExecutionGraphReplay graph_replay_;

 public:
  VisionDecoderHeadInferenceReplay() = default;
  explicit VisionDecoderHeadInferenceReplay(ExecutionGraphReplay graph_replay)
      : graph_replay_(std::move(graph_replay)) {}

  bool defined() const;
  bool recorded() const;
  const InferenceReplay& replay() const;
  const ExecutionGraphReplay& graph_replay() const;
  ExecutionGraphReplayStep phase_step(std::function<void()> record_step) const;
  Tensor& layer1_slot();
  Tensor& layer2_slot();
  Tensor& layer3_slot();
  Tensor& layer4_slot();
  const VisionDecoderProgram& refinenet4_program() const;
  VisionDecoderProgram& refinenet4_program();
  const VisionDecoderProgram& refinenet3_program() const;
  VisionDecoderProgram& refinenet3_program();
  const VisionDecoderProgram& refinenet2_program() const;
  VisionDecoderProgram& refinenet2_program();
  const VisionDecoderProgram& refinenet1_program() const;
  VisionDecoderProgram& refinenet1_program();
  Tensor& output_conv1_output();
  Tensor& upsample_output();
  Tensor& output_conv2_conv1_output();
  Tensor& output_conv2_relu1_output();
  Tensor& output_conv2_conv2_output();
  Tensor& output_slot();
  const void* identity() const;
};

AttentionRuntimeInferenceGraph
lookup_or_create_labeled_attention_runtime_inference_graph(
    const std::string& allocation_label,
    ScalarType dtype,
    bool persistent);

VisionBackboneInferenceGraph
lookup_or_create_labeled_vision_backbone_inference_graph(
    const std::string& allocation_label,
    ScalarType dtype,
    bool persistent);

VisionDecoderInferenceGraph lookup_or_create_labeled_vision_decoder_inference_graph(
    const std::string& allocation_label,
    ScalarType dtype,
    bool persistent);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
