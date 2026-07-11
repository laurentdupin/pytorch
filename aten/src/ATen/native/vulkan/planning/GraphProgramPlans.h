#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Mm.h>

#include <atomic>
#include <cstdint>
#include <torch/custom_class.h>

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
  const char* version{"v1"};
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
  bool direct_transition_only{true};
  bool replay_state_empty{true};
};

class GraphConv2dReluConv2dPlan final
    : public torch::jit::CustomClassHolder {
 private:
  StaticConv2dReluConv2dPlanSchema schema_{};
  c10::intrusive_ptr<Conv2dPackedContext> first_conv_context_;
  c10::intrusive_ptr<Conv2dPackedContext> second_conv_context_;
  std::atomic_flag invocation_active_ = ATOMIC_FLAG_INIT;

 public:
  GraphConv2dReluConv2dPlan(
      c10::intrusive_ptr<Conv2dPackedContext> first_conv_context,
      c10::intrusive_ptr<Conv2dPackedContext> second_conv_context);

  const StaticConv2dReluConv2dPlanSchema& schema() const;
  const c10::intrusive_ptr<Conv2dPackedContext>& first_conv_context() const;
  const c10::intrusive_ptr<Conv2dPackedContext>& second_conv_context() const;
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
