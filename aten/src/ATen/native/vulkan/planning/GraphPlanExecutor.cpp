#include <ATen/native/vulkan/planning/GraphPlanExecutor.h>

#ifdef USE_VULKAN_API

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>

#include <c10/core/DispatchKey.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <utility>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

struct VulkanGraphPlanValue final {
  int64_t use_count{0};
  int64_t last_use{-1};
  bool escapes{false};
};

struct VulkanGraphPlanInstruction final {
  std::string node_name;
  c10::OperatorHandle operator_handle;
  std::vector<int64_t> argument_refs;
  int64_t output_value_id{-1};
};

bool has_plan_dispatch(const c10::OperatorHandle& operator_handle) {
  return operator_handle.hasKernelForDispatchKey(c10::DispatchKey::Vulkan) ||
      operator_handle.hasKernelForDispatchKey(
          c10::DispatchKey::CompositeImplicitAutograd) ||
      operator_handle.hasKernelForDispatchKey(
          c10::DispatchKey::CompositeExplicitAutograd) ||
      operator_handle.hasKernelForDispatchKey(
          c10::DispatchKey::CompositeExplicitAutogradNonFunctional);
}

int64_t constant_index(const int64_t argument_ref) {
  TORCH_INTERNAL_ASSERT(argument_ref < 0);
  TORCH_CHECK(
      argument_ref != std::numeric_limits<int64_t>::min(),
      "VulkanGraphPlan.v2 constant reference underflow");
  return -argument_ref - 1;
}

bool any_implicit_boundary(const std::vector<int64_t>& counters) {
  return std::any_of(counters.begin(), counters.end(), [](const int64_t value) {
    return value != 0;
  });
}

void check_implicit_boundary(
    const VulkanGraphPlanInstruction& instruction,
    const std::vector<int64_t>& counters) {
  TORCH_CHECK(
      !any_implicit_boundary(counters),
      "VulkanGraphPlan.v2 node '",
      instruction.node_name,
      "' (",
      instruction.operator_handle.schema().operator_name(),
      ") crossed an implicit host boundary: cpu_fallback=",
      counters.at(0),
      ", sync_readback=",
      counters.at(1),
      ", deferred_values_created=",
      counters.at(2));
}

class VulkanGraphPlanInvocation final {
 private:
  VulkanGraphPlan& plan_;

 public:
  explicit VulkanGraphPlanInvocation(VulkanGraphPlan& plan) : plan_(plan) {
    TORCH_CHECK(
        plan_.try_begin_invocation(),
        "VulkanGraphPlan.v2 rejects concurrent invocation");
  }

  ~VulkanGraphPlanInvocation() {
    plan_.end_invocation();
  }
};

} // namespace

struct VulkanGraphPlan::State final {
  std::vector<VulkanGraphPlanValue> values;
  std::vector<VulkanGraphPlanInstruction> instructions;
  std::vector<c10::IValue> constants;
  std::vector<int64_t> output_value_ids;
  int64_t input_count{0};
};

VulkanGraphPlan::VulkanGraphPlan(std::shared_ptr<State> state)
    : state_(std::move(state)) {
  TORCH_CHECK(valid(), "VulkanGraphPlan.v2 has an invalid schema");
}

int64_t VulkanGraphPlan::input_count() const {
  return state_ ? state_->input_count : 0;
}

int64_t VulkanGraphPlan::instruction_count() const {
  return state_ ? static_cast<int64_t>(state_->instructions.size()) : 0;
}

int64_t VulkanGraphPlan::effect_instruction_count() const {
  if (!state_) {
    return 0;
  }
  return static_cast<int64_t>(std::count_if(
      state_->instructions.begin(),
      state_->instructions.end(),
      [](const VulkanGraphPlanInstruction& instruction) {
        return instruction.output_value_id < 0;
      }));
}

int64_t VulkanGraphPlan::value_count() const {
  return state_ ? static_cast<int64_t>(state_->values.size()) : 0;
}

int64_t VulkanGraphPlan::output_count() const {
  return state_ ? static_cast<int64_t>(state_->output_value_ids.size()) : 0;
}

std::vector<int64_t> VulkanGraphPlan::value_use_counts() const {
  std::vector<int64_t> counts;
  if (!state_) {
    return counts;
  }
  counts.reserve(state_->values.size());
  for (const VulkanGraphPlanValue& value : state_->values) {
    counts.push_back(value.use_count);
  }
  return counts;
}

std::vector<int64_t> VulkanGraphPlan::value_last_uses() const {
  std::vector<int64_t> last_uses;
  if (!state_) {
    return last_uses;
  }
  last_uses.reserve(state_->values.size());
  for (const VulkanGraphPlanValue& value : state_->values) {
    last_uses.push_back(value.last_use);
  }
  return last_uses;
}

bool VulkanGraphPlan::valid() const {
  if (
      !state_ || state_->input_count <= 0 || state_->instructions.empty() ||
      state_->output_value_ids.empty() ||
      state_->values.size() < static_cast<size_t>(state_->input_count)) {
    return false;
  }
  int64_t next_value_id = state_->input_count;
  for (const auto instruction_index :
       c10::irange(state_->instructions.size())) {
    const VulkanGraphPlanInstruction& instruction =
        state_->instructions[instruction_index];
    if (instruction.node_name.empty() || instruction.output_value_id < -1) {
      return false;
    }
    if (instruction.output_value_id >= 0) {
      if (instruction.output_value_id != next_value_id) {
        return false;
      }
      ++next_value_id;
    }
  }
  if (next_value_id != static_cast<int64_t>(state_->values.size())) {
    return false;
  }
  return std::all_of(
      state_->output_value_ids.begin(),
      state_->output_value_ids.end(),
      [this](const int64_t value_id) {
        return value_id >= 0 &&
            value_id < static_cast<int64_t>(state_->values.size());
      });
}

bool VulkanGraphPlan::try_begin_invocation() {
  return !invocation_active_.test_and_set(std::memory_order_acquire);
}

void VulkanGraphPlan::end_invocation() {
  invocation_active_.clear(std::memory_order_release);
}

const VulkanGraphPlan::State& VulkanGraphPlan::state() const {
  TORCH_INTERNAL_ASSERT(state_);
  return *state_;
}

c10::intrusive_ptr<VulkanGraphPlan> create_vulkan_graph_plan(
    std::vector<std::string> node_names,
    std::vector<std::string> operator_names,
    std::vector<std::string> overload_names,
    std::vector<std::vector<int64_t>> argument_refs,
    std::vector<int64_t> instruction_output_value_ids,
    const c10::List<c10::IValue>& constants,
    const int64_t input_count,
    std::vector<int64_t> output_value_ids) {
  TORCH_CHECK(
      input_count > 0,
      "VulkanGraphPlan.v2 requires at least one tensor input");
  const size_t instruction_count = node_names.size();
  TORCH_CHECK(
      instruction_count > 0 && operator_names.size() == instruction_count &&
          overload_names.size() == instruction_count &&
          argument_refs.size() == instruction_count &&
          instruction_output_value_ids.size() == instruction_count,
      "VulkanGraphPlan.v2 requires aligned non-empty instruction fields");
  TORCH_CHECK(
      !output_value_ids.empty(),
      "VulkanGraphPlan.v2 requires at least one output value");

  int64_t next_value_id = input_count;
  for (const int64_t output_value_id : instruction_output_value_ids) {
    TORCH_CHECK(
        output_value_id == -1 || output_value_id == next_value_id,
        "VulkanGraphPlan.v2 instruction output IDs must follow IValue SSA order");
    if (output_value_id >= 0) {
      ++next_value_id;
    }
  }

  auto state = std::make_shared<VulkanGraphPlan::State>();
  state->input_count = input_count;
  state->constants.assign(constants.begin(), constants.end());
  state->output_value_ids = std::move(output_value_ids);
  state->values.resize(static_cast<size_t>(next_value_id));
  state->instructions.reserve(instruction_count);
  int64_t defined_value_count = input_count;
  for (const auto instruction_index : c10::irange(instruction_count)) {
    TORCH_CHECK(
        !node_names[instruction_index].empty() &&
            !operator_names[instruction_index].empty(),
        "VulkanGraphPlan.v2 instruction names must be non-empty");
    c10::OperatorHandle operator_handle =
        c10::Dispatcher::singleton().findSchemaOrThrow(
            operator_names[instruction_index].c_str(),
            overload_names[instruction_index].c_str());
    const c10::FunctionSchema& schema = operator_handle.schema();
    TORCH_CHECK(
        !schema.is_mutable(),
        "VulkanGraphPlan.v2 rejects mutable operator ",
        schema.operator_name());
    TORCH_CHECK(
        has_plan_dispatch(operator_handle),
        "VulkanGraphPlan.v2 requires a Vulkan or composite kernel for ",
        schema.operator_name());
    const int64_t output_value_id =
        instruction_output_value_ids[instruction_index];
    TORCH_CHECK(
        schema.returns().size() <= 1u,
        "VulkanGraphPlan.v2 does not support multiple dispatcher returns from ",
        schema.operator_name());
    TORCH_CHECK(
        (schema.returns().empty() && output_value_id < 0) ||
            (schema.returns().size() == 1u && output_value_id >= 0),
        "VulkanGraphPlan.v2 output schema does not match ",
        schema.operator_name());
    TORCH_CHECK(
        argument_refs[instruction_index].size() == schema.arguments().size(),
        "VulkanGraphPlan.v2 argument count does not match ",
        schema.operator_name());

    for (const int64_t argument_ref : argument_refs[instruction_index]) {
      if (argument_ref >= 0) {
        TORCH_CHECK(
            argument_ref < defined_value_count,
            "VulkanGraphPlan.v2 instruction '",
            node_names[instruction_index],
            "' references a value before it is defined");
        VulkanGraphPlanValue& value =
            state->values[static_cast<size_t>(argument_ref)];
        ++value.use_count;
        value.last_use = static_cast<int64_t>(instruction_index);
      } else {
        const int64_t index = constant_index(argument_ref);
        TORCH_CHECK(
            index < static_cast<int64_t>(state->constants.size()),
            "VulkanGraphPlan.v2 instruction '",
            node_names[instruction_index],
            "' has an invalid constant reference");
      }
    }
    if (output_value_id >= 0) {
      state->values[static_cast<size_t>(output_value_id)].last_use =
          static_cast<int64_t>(instruction_index);
    }
    state->instructions.push_back(VulkanGraphPlanInstruction{
        std::move(node_names[instruction_index]),
        std::move(operator_handle),
        std::move(argument_refs[instruction_index]),
        output_value_id});
    if (output_value_id >= 0) {
      ++defined_value_count;
    }
  }

  for (const int64_t output_value_id : state->output_value_ids) {
    TORCH_CHECK(
        output_value_id >= 0 &&
            output_value_id < static_cast<int64_t>(state->values.size()),
        "VulkanGraphPlan.v2 output value is out of range");
    state->values[static_cast<size_t>(output_value_id)].escapes = true;
  }
  return c10::make_intrusive<VulkanGraphPlan>(std::move(state));
}

std::vector<Tensor> run_vulkan_graph_plan(
    const std::vector<Tensor>& inputs,
    const c10::intrusive_ptr<VulkanGraphPlan>& plan) {
  TORCH_CHECK(plan, "VulkanGraphPlan.v2 requires a plan");
  TORCH_CHECK(plan->valid(), "VulkanGraphPlan.v2 has an invalid schema");
  const VulkanGraphPlan::State& state = plan->state();
  TORCH_CHECK(
      inputs.size() == static_cast<size_t>(state.input_count),
      "VulkanGraphPlan.v2 input count mismatch");
  TORCH_CHECK(
      std::all_of(inputs.begin(), inputs.end(), [](const Tensor& input) {
        return input.is_vulkan();
      }),
      "VulkanGraphPlan.v2 requires Vulkan tensor inputs");
  VulkanGraphPlanInvocation invocation(*plan);

  std::vector<c10::IValue> values(state.values.size());
  std::vector<bool> value_live(state.values.size(), false);
  for (const auto input_index : c10::irange(inputs.size())) {
    values[input_index] = inputs[input_index];
    value_live[input_index] = true;
  }
  for (const auto instruction_index : c10::irange(state.instructions.size())) {
    const VulkanGraphPlanInstruction& instruction =
        state.instructions[instruction_index];
    std::vector<c10::IValue> stack;
    stack.reserve(instruction.argument_refs.size());
    for (const int64_t argument_ref : instruction.argument_refs) {
      if (argument_ref >= 0) {
        TORCH_CHECK(
            value_live[static_cast<size_t>(argument_ref)],
            "VulkanGraphPlan.v2 node '",
            instruction.node_name,
            "' references a released value");
        stack.push_back(values[static_cast<size_t>(argument_ref)]);
      } else {
        stack.push_back(
            state.constants[static_cast<size_t>(constant_index(argument_ref))]);
      }
    }

    const int64_t scope_token = begin_vulkan_graph_execution_scope();
    try {
      c10::Dispatcher::singleton().callBoxed(
          instruction.operator_handle, &stack);
    } catch (const c10::Error& error) {
      const std::vector<int64_t> counters =
          end_vulkan_graph_execution_scope(scope_token);
      check_implicit_boundary(instruction, counters);
      TORCH_CHECK(
          false,
          "VulkanGraphPlan.v2 node '",
          instruction.node_name,
          "' (",
          instruction.operator_handle.schema().operator_name(),
          ") failed: ",
          error.what_without_backtrace());
    } catch (const std::exception& error) {
      const std::vector<int64_t> counters =
          end_vulkan_graph_execution_scope(scope_token);
      check_implicit_boundary(instruction, counters);
      TORCH_CHECK(
          false,
          "VulkanGraphPlan.v2 node '",
          instruction.node_name,
          "' (",
          instruction.operator_handle.schema().operator_name(),
          ") failed: ",
          error.what());
    } catch (...) {
      const std::vector<int64_t> counters =
          end_vulkan_graph_execution_scope(scope_token);
      check_implicit_boundary(instruction, counters);
      TORCH_CHECK(
          false,
          "VulkanGraphPlan.v2 node '",
          instruction.node_name,
          "' (",
          instruction.operator_handle.schema().operator_name(),
          ") failed with a non-standard exception");
    }
    const std::vector<int64_t> counters =
        end_vulkan_graph_execution_scope(scope_token);
    check_implicit_boundary(instruction, counters);
    if (instruction.output_value_id < 0) {
      TORCH_CHECK(
          stack.empty(),
          "VulkanGraphPlan.v2 effect node '",
          instruction.node_name,
          "' produced an undeclared value");
    } else {
      TORCH_CHECK(
          stack.size() == 1u,
          "VulkanGraphPlan.v2 node '",
          instruction.node_name,
          "' did not produce its declared value");
      c10::IValue output = std::move(stack.back());
      TORCH_CHECK(
          !output.isTensor() || output.toTensor().is_vulkan(),
          "VulkanGraphPlan.v2 node '",
          instruction.node_name,
          "' produced a non-Vulkan tensor");
      values[static_cast<size_t>(instruction.output_value_id)] =
          std::move(output);
      value_live[static_cast<size_t>(instruction.output_value_id)] = true;
    }

    for (const int64_t argument_ref : instruction.argument_refs) {
      if (argument_ref < 0) {
        continue;
      }
      const VulkanGraphPlanValue& value =
          state.values[static_cast<size_t>(argument_ref)];
      if (
          !value.escapes &&
          value.last_use == static_cast<int64_t>(instruction_index)) {
        values[static_cast<size_t>(argument_ref)] = c10::IValue();
        value_live[static_cast<size_t>(argument_ref)] = false;
      }
    }
  }

  std::vector<Tensor> outputs;
  outputs.reserve(state.output_value_ids.size());
  for (const int64_t output_value_id : state.output_value_ids) {
    c10::IValue& output = values[static_cast<size_t>(output_value_id)];
    TORCH_CHECK(
        value_live[static_cast<size_t>(output_value_id)] && output.isTensor(),
        "VulkanGraphPlan.v2 output references a released or non-Tensor value");
    outputs.push_back(output.toTensor());
  }
  return outputs;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif // USE_VULKAN_API
