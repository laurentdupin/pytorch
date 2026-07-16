#pragma once

#ifdef USE_VULKAN_API

#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>

#include <torch/custom_class.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace api {
struct VulkanSubmission;
}
namespace ops {
namespace utils {

class VulkanGraphPlan final : public torch::jit::CustomClassHolder {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;
  std::atomic_flag invocation_active_ = ATOMIC_FLAG_INIT;

 public:
  explicit VulkanGraphPlan(std::shared_ptr<State> state);

  int64_t input_count() const;
  int64_t instruction_count() const;
  int64_t effect_instruction_count() const;
  int64_t graph_scalar_instruction_count() const;
  int64_t list_projection_instruction_count() const;
  int64_t list_argument_count() const;
  int64_t invocation_value_slot_count() const;
  int64_t invocation_list_slot_count() const;
  int64_t invocation_stack_capacity() const;
  int64_t dead_input_reuse_instruction_count() const;
  int64_t dead_input_reuse_count() const;
  int64_t value_count() const;
  int64_t output_count() const;
  bool submission_owned() const;
  int64_t planning_model_domain() const;
  int64_t planning_execution_phase() const;
  bool planning_prefer_packed_layout_propagation() const;
  std::optional<std::vector<int64_t>>
  planning_fixed_shape_graph_input_sizes() const;
  int64_t invocation_generation() const;
  int64_t last_submission_value() const;
  bool last_submission_complete() const;
  std::vector<int64_t> value_use_counts() const;
  std::vector<int64_t> value_last_uses() const;
  bool valid() const;
  bool try_begin_invocation();
  void end_invocation();
  void record_submission(
      c10::DeviceIndex device_index,
      const api::VulkanSubmission& submission);

  const State& state() const;
};

c10::intrusive_ptr<VulkanGraphPlan> create_vulkan_graph_plan(
    std::vector<std::string> node_names,
    std::vector<std::string> operator_names,
    std::vector<std::string> overload_names,
    std::vector<std::vector<std::vector<int64_t>>> argument_refs,
    std::vector<std::vector<int64_t>> argument_kinds,
    std::vector<std::vector<int64_t>> instruction_output_value_ids,
    const c10::List<c10::IValue>& constants,
    int64_t input_count,
    std::vector<int64_t> output_value_ids,
    int64_t planning_model_domain,
    int64_t planning_execution_phase,
    bool planning_prefer_packed_layout_propagation,
    std::optional<std::vector<int64_t>>
        planning_fixed_shape_graph_input_sizes);

std::vector<Tensor> run_vulkan_graph_plan(
    const std::vector<Tensor>& inputs,
    const c10::intrusive_ptr<VulkanGraphPlan>& plan);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif // USE_VULKAN_API
