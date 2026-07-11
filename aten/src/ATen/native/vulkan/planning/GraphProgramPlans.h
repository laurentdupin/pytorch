#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/Mm.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <torch/custom_class.h>
#include <tuple>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

struct StaticLinearGeluPlanSchema final {
  const char* program_name{"StaticLinearGeluRegion"};
  const char* version{"v1"};
  uint32_t instruction_count{1u};
  uint32_t input_ssa{0u};
  uint32_t output_ssa{1u};
  uint32_t input_use_count{1u};
  uint32_t input_last_use{0u};
  uint32_t static_context_slot{0u};
  bool direct_transition_only{true};
  bool replay_state_empty{true};
};

class GraphLinearGeluPlan final : public torch::jit::CustomClassHolder {
 private:
  StaticLinearGeluPlanSchema schema_{};
  c10::intrusive_ptr<LinearPackedContext> linear_context_;
  std::atomic_flag invocation_active_ = ATOMIC_FLAG_INIT;

 public:
  explicit GraphLinearGeluPlan(
      c10::intrusive_ptr<LinearPackedContext> linear_context);

  const StaticLinearGeluPlanSchema& schema() const;
  const c10::intrusive_ptr<LinearPackedContext>& linear_context() const;
  bool try_begin_invocation();
  void end_invocation();
};

c10::intrusive_ptr<GraphLinearGeluPlan> create_graph_linear_gelu_plan(
    const c10::intrusive_ptr<LinearPackedContext>& linear_context);

Tensor run_graph_linear_gelu_plan(
    const Tensor& input,
    const c10::intrusive_ptr<GraphLinearGeluPlan>& plan);

struct StaticAddLayernormPlanSchema final {
  const char* program_name{"StaticAddLayernormRegion"};
  const char* version{"v1"};
  const char* fused_instruction{"add_layernorm_fused_or_composed_vulkan"};
  uint32_t instruction_count{1u};
  uint32_t residual_input_ssa{0u};
  uint32_t addend_input_ssa{1u};
  uint32_t residual_output_ssa{2u};
  uint32_t normalized_output_ssa{3u};
  uint32_t residual_input_use_count{1u};
  uint32_t residual_input_last_use{0u};
  uint32_t addend_input_use_count{1u};
  uint32_t addend_input_last_use{0u};
  uint32_t static_context_slot{0u};
  bool direct_transition_only{true};
  bool replay_state_empty{true};
  bool persistent_output_state{false};
};

class GraphAddLayernormPlan final : public torch::jit::CustomClassHolder {
 private:
  StaticAddLayernormPlanSchema schema_{};
  c10::intrusive_ptr<LayernormPackedContext> layernorm_context_;
  std::vector<int64_t> normalized_shape_;
  std::atomic_flag invocation_active_ = ATOMIC_FLAG_INIT;

 public:
  GraphAddLayernormPlan(
      c10::intrusive_ptr<LayernormPackedContext> layernorm_context,
      std::vector<int64_t> normalized_shape);

  const StaticAddLayernormPlanSchema& schema() const;
  const c10::intrusive_ptr<LayernormPackedContext>& layernorm_context() const;
  const std::vector<int64_t>& normalized_shape() const;
  bool try_begin_invocation();
  void end_invocation();
};

c10::intrusive_ptr<GraphAddLayernormPlan> create_graph_add_layernorm_plan(
    const c10::intrusive_ptr<LayernormPackedContext>& layernorm_context,
    IntArrayRef normalized_shape);

std::tuple<Tensor, Tensor> run_graph_add_layernorm_plan(
    const Tensor& residual,
    const Tensor& addend,
    const c10::intrusive_ptr<GraphAddLayernormPlan>& plan);

struct StaticConv2dReluPlanSchema final {
  const char* program_name{"StaticConv2dReluRegion"};
  const char* version{"v1"};
  uint32_t instruction_count{1u};
  uint32_t input_ssa{0u};
  uint32_t output_ssa{1u};
  uint32_t input_use_count{1u};
  uint32_t input_last_use{0u};
  uint32_t static_context_slot{0u};
  bool direct_transition_only{true};
  bool replay_state_empty{true};
};

class GraphConv2dReluPlan final : public torch::jit::CustomClassHolder {
 private:
  StaticConv2dReluPlanSchema schema_{};
  c10::intrusive_ptr<Conv2dPackedContext> conv_context_;
  std::atomic_flag invocation_active_ = ATOMIC_FLAG_INIT;

 public:
  explicit GraphConv2dReluPlan(
      c10::intrusive_ptr<Conv2dPackedContext> conv_context);

  const StaticConv2dReluPlanSchema& schema() const;
  const c10::intrusive_ptr<Conv2dPackedContext>& conv_context() const;
  bool try_begin_invocation();
  void end_invocation();
};

c10::intrusive_ptr<GraphConv2dReluPlan> create_graph_conv2d_relu_plan(
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context);

Tensor run_graph_conv2d_relu_plan(
    const Tensor& input,
    const c10::intrusive_ptr<GraphConv2dReluPlan>& plan);

struct StaticConv2dReluConv2dPlanSchema final {
  const char* program_name{"StaticConv2dReluConv2dRegion"};
  const char* version{"v3"};
  uint32_t instruction_count{2u};
  uint32_t input_ssa{0u};
  uint32_t intermediate_ssa{1u};
  uint32_t output_ssa{2u};
  uint32_t input_use_count{1u};
  uint32_t input_last_use{0u};
  uint32_t intermediate_use_count{1u};
  uint32_t intermediate_last_use{1u};
  uint32_t first_static_context_slot{0u};
  uint32_t second_static_context_slot{1u};
  bool bounded_submission_owned{true};
  bool program_private_scratch{true};
  uint32_t scratch_ring_capacity{2u};
  bool timeline_gated_release{true};
  bool direct_transition_only{true};
  bool replay_state_empty{true};
};

struct GraphConv2dReluConv2dScratchDescriptor final {
  std::vector<int64_t> sizes;
  ScalarType dtype{ScalarType::Undefined};
  api::StorageType storage_type{api::StorageType::UNKNOWN};
  api::GPUMemoryLayout memory_layout{
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED};
  api::ExecutionLayout execution_layout{api::ExecutionLayout::TEXTURE};
  bool direct_buffer{false};

  bool matches(const GraphConv2dReluConv2dScratchDescriptor&) const;
};

struct GraphConv2dReluConv2dScratchSlot final {
  Tensor tensor;
  GraphConv2dReluConv2dScratchDescriptor descriptor;
  api::VulkanSubmission submission;
};

class GraphConv2dReluConv2dPlan final
    : public torch::jit::CustomClassHolder {
 private:
  StaticConv2dReluConv2dPlanSchema schema_{};
  c10::intrusive_ptr<Conv2dPackedContext> first_conv_context_;
  c10::intrusive_ptr<Conv2dPackedContext> second_conv_context_;
  std::array<GraphConv2dReluConv2dScratchSlot, 2u> scratch_slots_{};
  std::atomic_flag invocation_active_ = ATOMIC_FLAG_INIT;

 public:
  GraphConv2dReluConv2dPlan(
      c10::intrusive_ptr<Conv2dPackedContext> first_conv_context,
      c10::intrusive_ptr<Conv2dPackedContext> second_conv_context);
  ~GraphConv2dReluConv2dPlan() noexcept;

  const StaticConv2dReluConv2dPlanSchema& schema() const;
  const c10::intrusive_ptr<Conv2dPackedContext>& first_conv_context() const;
  const c10::intrusive_ptr<Conv2dPackedContext>& second_conv_context() const;
  int64_t find_reusable_scratch_slot(
      const GraphConv2dReluConv2dScratchDescriptor&,
      api::Context&);
  int64_t find_capture_scratch_slot(api::Context&);
  Tensor& scratch_tensor(size_t);
  void adopt_scratch_tensor(
      size_t,
      Tensor,
      GraphConv2dReluConv2dScratchDescriptor,
      api::VulkanSubmission);
  void mark_scratch_submission(size_t, api::VulkanSubmission);
  bool try_begin_invocation();
  void end_invocation();
};

c10::intrusive_ptr<GraphConv2dReluConv2dPlan>
create_graph_conv2d_relu_conv2d_plan(
    const c10::intrusive_ptr<Conv2dPackedContext>& first_conv_context,
    const c10::intrusive_ptr<Conv2dPackedContext>& second_conv_context);

Tensor run_graph_conv2d_relu_conv2d_plan(
    const Tensor& input,
    const c10::intrusive_ptr<GraphConv2dReluConv2dPlan>& plan);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif // USE_VULKAN_API
