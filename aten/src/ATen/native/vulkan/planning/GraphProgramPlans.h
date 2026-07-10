#pragma once

#ifdef USE_VULKAN_API

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

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif // USE_VULKAN_API
