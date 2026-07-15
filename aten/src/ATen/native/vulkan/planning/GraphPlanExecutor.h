#pragma once

#ifdef USE_VULKAN_API

#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>

#include <torch/custom_class.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
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
  int64_t value_count() const;
  int64_t output_count() const;
  std::vector<int64_t> value_use_counts() const;
  std::vector<int64_t> value_last_uses() const;
  bool valid() const;
  bool try_begin_invocation();
  void end_invocation();

  const State& state() const;
};

c10::intrusive_ptr<VulkanGraphPlan> create_vulkan_graph_plan(
    std::vector<std::string> node_names,
    std::vector<std::string> operator_names,
    std::vector<std::string> overload_names,
    std::vector<std::vector<int64_t>> argument_refs,
    const c10::List<c10::IValue>& constants,
    int64_t input_count,
    std::vector<int64_t> output_value_ids);

std::vector<Tensor> run_vulkan_graph_plan(
    const std::vector<Tensor>& inputs,
    const c10::intrusive_ptr<VulkanGraphPlan>& plan);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif // USE_VULKAN_API
