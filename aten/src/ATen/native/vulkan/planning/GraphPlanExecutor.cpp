#include <ATen/native/vulkan/planning/GraphPlanExecutor.h>

#ifdef USE_VULKAN_API

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>

#include <c10/core/DispatchKey.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <c10/util/safe_numerics.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <optional>
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

enum class VulkanGraphPlanArgumentKind : int64_t {
  Value = 0,
  List = 1,
};

enum class VulkanGraphPlanInstructionKind : int64_t {
  Dispatcher = 0,
  IntAdd = 1,
  IntSubtract = 2,
  IntMultiply = 3,
  IntFloorDivide = 4,
  ListGetItem = 5,
};

struct VulkanGraphPlanArgument final {
  VulkanGraphPlanArgumentKind kind{VulkanGraphPlanArgumentKind::Value};
  std::vector<int64_t> refs;
  c10::TypePtr list_element_type;
};

struct VulkanGraphPlanInstruction final {
  std::string node_name;
  std::string operator_name;
  VulkanGraphPlanInstructionKind kind{
      VulkanGraphPlanInstructionKind::Dispatcher};
  std::optional<c10::OperatorHandle> operator_handle;
  std::vector<VulkanGraphPlanArgument> arguments;
  std::vector<int64_t> output_value_ids;
};

std::optional<VulkanGraphPlanInstructionKind> graph_scalar_instruction_kind(
    const std::string& operator_name) {
  if (operator_name == "vulkan_graph::int_add") {
    return VulkanGraphPlanInstructionKind::IntAdd;
  }
  if (operator_name == "vulkan_graph::int_subtract") {
    return VulkanGraphPlanInstructionKind::IntSubtract;
  }
  if (operator_name == "vulkan_graph::int_multiply") {
    return VulkanGraphPlanInstructionKind::IntMultiply;
  }
  if (operator_name == "vulkan_graph::int_floor_divide") {
    return VulkanGraphPlanInstructionKind::IntFloorDivide;
  }
  return std::nullopt;
}

std::optional<VulkanGraphPlanInstructionKind> graph_instruction_kind(
    const std::string& operator_name) {
  if (operator_name == "vulkan_graph::list_getitem") {
    return VulkanGraphPlanInstructionKind::ListGetItem;
  }
  return graph_scalar_instruction_kind(operator_name);
}

bool is_graph_scalar_instruction_kind(
    const VulkanGraphPlanInstructionKind kind) {
  return kind == VulkanGraphPlanInstructionKind::IntAdd ||
      kind == VulkanGraphPlanInstructionKind::IntSubtract ||
      kind == VulkanGraphPlanInstructionKind::IntMultiply ||
      kind == VulkanGraphPlanInstructionKind::IntFloorDivide;
}

bool subtract_overflows(
    const int64_t left,
    const int64_t right,
    int64_t* output) {
  if (
      (right > 0 && left < std::numeric_limits<int64_t>::min() + right) ||
      (right < 0 && left > std::numeric_limits<int64_t>::max() + right)) {
    return true;
  }
  *output = left - right;
  return false;
}

void execute_graph_scalar_instruction(
    const VulkanGraphPlanInstruction& instruction,
    std::vector<c10::IValue>& stack) {
  TORCH_CHECK(
      stack.size() == 2u && stack[0].isInt() && stack[1].isInt(),
      "VulkanGraphPlan.v6 graph scalar node '",
      instruction.node_name,
      "' requires two integer operands");
  const int64_t left = stack[0].toInt();
  const int64_t right = stack[1].toInt();
  int64_t result = 0;
  bool overflowed = false;
  switch (instruction.kind) {
    case VulkanGraphPlanInstructionKind::IntAdd:
      overflowed = c10::add_overflows(left, right, &result);
      break;
    case VulkanGraphPlanInstructionKind::IntSubtract:
      overflowed = subtract_overflows(left, right, &result);
      break;
    case VulkanGraphPlanInstructionKind::IntMultiply:
      overflowed = c10::mul_overflows(left, right, &result);
      break;
    case VulkanGraphPlanInstructionKind::IntFloorDivide: {
      TORCH_CHECK(
          right != 0,
          "VulkanGraphPlan.v6 graph scalar node '",
          instruction.node_name,
          "' divides by zero");
      TORCH_CHECK(
          left != std::numeric_limits<int64_t>::min() || right != -1,
          "VulkanGraphPlan.v6 graph scalar node '",
          instruction.node_name,
          "' overflows integer floor division");
      result = left / right;
      const int64_t remainder = left % right;
      if (remainder != 0 && ((remainder < 0) != (right < 0))) {
        --result;
      }
      break;
    }
    case VulkanGraphPlanInstructionKind::Dispatcher:
    case VulkanGraphPlanInstructionKind::ListGetItem:
      TORCH_INTERNAL_ASSERT(false);
  }
  TORCH_CHECK(
      !overflowed,
      "VulkanGraphPlan.v6 graph scalar node '",
      instruction.node_name,
      "' overflows int64");
  stack.clear();
  stack.emplace_back(result);
}

void execute_list_getitem_instruction(
    const VulkanGraphPlanInstruction& instruction,
    std::vector<c10::IValue>& stack) {
  TORCH_CHECK(
      stack.size() == 2u && stack[0].isList() && stack[1].isInt(),
      "VulkanGraphPlan.v6 list projection node '",
      instruction.node_name,
      "' requires a list and integer index");
  const c10::List<c10::IValue> list = stack[0].toList();
  const int64_t list_size = static_cast<int64_t>(list.size());
  int64_t index = stack[1].toInt();
  if (index < 0) {
    index += list_size;
  }
  TORCH_CHECK(
      index >= 0 && index < list_size,
      "VulkanGraphPlan.v6 list projection node '",
      instruction.node_name,
      "' index ",
      stack[1].toInt(),
      " is out of range for length ",
      list_size);
  c10::IValue output = list.get(static_cast<size_t>(index));
  stack.clear();
  stack.push_back(std::move(output));
}

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
      "VulkanGraphPlan.v6 constant reference underflow");
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
      "VulkanGraphPlan.v6 node '",
      instruction.node_name,
      "' (",
      instruction.operator_name,
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
        "VulkanGraphPlan.v6 rejects concurrent invocation");
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
  TORCH_CHECK(valid(), "VulkanGraphPlan.v6 has an invalid schema");
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
        return instruction.output_value_ids.empty();
      }));
}

int64_t VulkanGraphPlan::graph_scalar_instruction_count() const {
  if (!state_) {
    return 0;
  }
  return static_cast<int64_t>(std::count_if(
      state_->instructions.begin(),
      state_->instructions.end(),
      [](const VulkanGraphPlanInstruction& instruction) {
        return is_graph_scalar_instruction_kind(instruction.kind);
      }));
}

int64_t VulkanGraphPlan::list_projection_instruction_count() const {
  if (!state_) {
    return 0;
  }
  return static_cast<int64_t>(std::count_if(
      state_->instructions.begin(),
      state_->instructions.end(),
      [](const VulkanGraphPlanInstruction& instruction) {
        return instruction.kind == VulkanGraphPlanInstructionKind::ListGetItem;
      }));
}

int64_t VulkanGraphPlan::list_argument_count() const {
  if (!state_) {
    return 0;
  }
  int64_t count = 0;
  for (const VulkanGraphPlanInstruction& instruction : state_->instructions) {
    count += static_cast<int64_t>(std::count_if(
        instruction.arguments.begin(),
        instruction.arguments.end(),
        [](const VulkanGraphPlanArgument& argument) {
          return argument.kind == VulkanGraphPlanArgumentKind::List;
        }));
  }
  return count;
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
    if (
        instruction.node_name.empty() || instruction.operator_name.empty() ||
        ((instruction.kind == VulkanGraphPlanInstructionKind::Dispatcher) !=
         instruction.operator_handle.has_value())) {
      return false;
    }
    if (
        instruction.kind != VulkanGraphPlanInstructionKind::Dispatcher &&
        (instruction.output_value_ids.size() != 1u ||
         instruction.arguments.size() != 2u)) {
      return false;
    }
    for (const VulkanGraphPlanArgument& argument : instruction.arguments) {
      if (
          argument.refs.empty() ||
          (argument.kind == VulkanGraphPlanArgumentKind::Value &&
           argument.refs.size() != 1u) ||
          (argument.kind == VulkanGraphPlanArgumentKind::List &&
           !argument.list_element_type)) {
        return false;
      }
      if (
          instruction.kind != VulkanGraphPlanInstructionKind::Dispatcher &&
          argument.kind != VulkanGraphPlanArgumentKind::Value) {
        return false;
      }
    }
    for (const int64_t output_value_id : instruction.output_value_ids) {
      if (output_value_id != next_value_id) {
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
    std::vector<std::vector<std::vector<int64_t>>> argument_refs,
    std::vector<std::vector<int64_t>> argument_kinds,
    std::vector<std::vector<int64_t>> instruction_output_value_ids,
    const c10::List<c10::IValue>& constants,
    const int64_t input_count,
    std::vector<int64_t> output_value_ids) {
  TORCH_CHECK(
      input_count > 0,
      "VulkanGraphPlan.v6 requires at least one tensor input");
  const size_t instruction_count = node_names.size();
  TORCH_CHECK(
      instruction_count > 0 && operator_names.size() == instruction_count &&
          overload_names.size() == instruction_count &&
          argument_refs.size() == instruction_count &&
          argument_kinds.size() == instruction_count &&
          instruction_output_value_ids.size() == instruction_count,
      "VulkanGraphPlan.v6 requires aligned non-empty instruction fields");
  TORCH_CHECK(
      !output_value_ids.empty(),
      "VulkanGraphPlan.v6 requires at least one output value");

  int64_t next_value_id = input_count;
  for (const auto& instruction_output_ids : instruction_output_value_ids) {
    for (const int64_t output_value_id : instruction_output_ids) {
      TORCH_CHECK(
          output_value_id == next_value_id,
          "VulkanGraphPlan.v6 instruction output IDs must follow IValue SSA order");
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
        "VulkanGraphPlan.v6 instruction names must be non-empty");
    const auto internal_instruction_kind =
        graph_instruction_kind(operator_names[instruction_index]);
    const VulkanGraphPlanInstructionKind instruction_kind =
        internal_instruction_kind.value_or(
            VulkanGraphPlanInstructionKind::Dispatcher);
    std::optional<c10::OperatorHandle> operator_handle;
    const c10::FunctionSchema* schema = nullptr;
    if (internal_instruction_kind) {
      TORCH_CHECK(
          overload_names[instruction_index].empty(),
          "VulkanGraphPlan.v6 internal instruction '",
          node_names[instruction_index],
          "' must not declare an overload");
    } else {
      operator_handle.emplace(
          c10::Dispatcher::singleton().findSchemaOrThrow(
              operator_names[instruction_index].c_str(),
              overload_names[instruction_index].c_str()));
      schema = &operator_handle->schema();
      TORCH_CHECK(
          !schema->is_mutable(),
          "VulkanGraphPlan.v6 rejects mutable operator ",
          schema->operator_name());
      TORCH_CHECK(
          has_plan_dispatch(*operator_handle),
          "VulkanGraphPlan.v6 requires a Vulkan or composite kernel for ",
          schema->operator_name());
    }
    std::vector<int64_t>& output_value_ids_for_instruction =
        instruction_output_value_ids[instruction_index];
    if (schema) {
      TORCH_CHECK(
          schema->returns().size() ==
              output_value_ids_for_instruction.size(),
          "VulkanGraphPlan.v6 output schema does not match ",
          schema->operator_name());
    } else {
      TORCH_CHECK(
          output_value_ids_for_instruction.size() == 1u,
          "VulkanGraphPlan.v6 internal instruction '",
          node_names[instruction_index],
          "' must define one value");
    }
    const size_t expected_argument_count =
        schema ? schema->arguments().size() : 2u;
    TORCH_CHECK(
        argument_refs[instruction_index].size() == expected_argument_count &&
            argument_kinds[instruction_index].size() == expected_argument_count,
        "VulkanGraphPlan.v6 argument count does not match ",
        operator_names[instruction_index]);

    std::vector<VulkanGraphPlanArgument> arguments;
    arguments.reserve(expected_argument_count);
    for (const auto argument_index : c10::irange(expected_argument_count)) {
      const int64_t kind_value =
          argument_kinds[instruction_index][argument_index];
      TORCH_CHECK(
          kind_value ==
                  static_cast<int64_t>(VulkanGraphPlanArgumentKind::Value) ||
              kind_value ==
                  static_cast<int64_t>(VulkanGraphPlanArgumentKind::List),
          "VulkanGraphPlan.v6 instruction '",
          node_names[instruction_index],
          "' has an invalid argument kind");
      const auto kind = static_cast<VulkanGraphPlanArgumentKind>(kind_value);
      std::vector<int64_t>& refs =
          argument_refs[instruction_index][argument_index];
      TORCH_CHECK(
          !refs.empty() &&
              (kind != VulkanGraphPlanArgumentKind::Value ||
               refs.size() == 1u),
          "VulkanGraphPlan.v6 instruction '",
          node_names[instruction_index],
          "' has an invalid argument recipe");

      c10::TypePtr list_element_type;
      if (kind == VulkanGraphPlanArgumentKind::List) {
        TORCH_CHECK(
            schema,
            "VulkanGraphPlan.v6 internal instruction '",
            node_names[instruction_index],
            "' requires value arguments");
        c10::TypePtr argument_type =
            schema->arguments()[argument_index].type();
        if (const auto optional_type = argument_type->cast<c10::OptionalType>()) {
          argument_type = optional_type->getElementType();
        }
        const auto list_type = argument_type->cast<c10::ListType>();
        TORCH_CHECK(
            list_type,
            "VulkanGraphPlan.v6 instruction '",
            node_names[instruction_index],
            "' declares a list recipe for non-list argument '",
            schema->arguments()[argument_index].name(),
            "'");
        list_element_type = list_type->getElementType();
      }

      for (const int64_t argument_ref : refs) {
        if (argument_ref >= 0) {
          TORCH_CHECK(
              argument_ref < defined_value_count,
              "VulkanGraphPlan.v6 instruction '",
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
              "VulkanGraphPlan.v6 instruction '",
              node_names[instruction_index],
              "' has an invalid constant reference");
        }
      }
      arguments.push_back(VulkanGraphPlanArgument{
          kind, std::move(refs), std::move(list_element_type)});
    }
    for (const int64_t output_value_id : output_value_ids_for_instruction) {
      state->values[static_cast<size_t>(output_value_id)].last_use =
          static_cast<int64_t>(instruction_index);
    }
    std::string diagnostic_operator_name = operator_names[instruction_index];
    if (!overload_names[instruction_index].empty()) {
      diagnostic_operator_name.append(".").append(
          overload_names[instruction_index]);
    }
    state->instructions.push_back(VulkanGraphPlanInstruction{
        std::move(node_names[instruction_index]),
        std::move(diagnostic_operator_name),
        instruction_kind,
        std::move(operator_handle),
        std::move(arguments),
        std::move(output_value_ids_for_instruction)});
    defined_value_count += static_cast<int64_t>(
        state->instructions.back().output_value_ids.size());
  }

  for (const int64_t output_value_id : state->output_value_ids) {
    TORCH_CHECK(
        output_value_id >= 0 &&
            output_value_id < static_cast<int64_t>(state->values.size()),
        "VulkanGraphPlan.v6 output value is out of range");
    state->values[static_cast<size_t>(output_value_id)].escapes = true;
  }
  return c10::make_intrusive<VulkanGraphPlan>(std::move(state));
}

std::vector<Tensor> run_vulkan_graph_plan(
    const std::vector<Tensor>& inputs,
    const c10::intrusive_ptr<VulkanGraphPlan>& plan) {
  TORCH_CHECK(plan, "VulkanGraphPlan.v6 requires a plan");
  TORCH_CHECK(plan->valid(), "VulkanGraphPlan.v6 has an invalid schema");
  const VulkanGraphPlan::State& state = plan->state();
  TORCH_CHECK(
      inputs.size() == static_cast<size_t>(state.input_count),
      "VulkanGraphPlan.v6 input count mismatch");
  TORCH_CHECK(
      std::all_of(inputs.begin(), inputs.end(), [](const Tensor& input) {
        return input.is_vulkan();
      }),
      "VulkanGraphPlan.v6 requires Vulkan tensor inputs");
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
    stack.reserve(instruction.arguments.size());
    const auto load_argument_ref = [&](const int64_t argument_ref) {
      if (argument_ref >= 0) {
        TORCH_CHECK(
            value_live[static_cast<size_t>(argument_ref)],
            "VulkanGraphPlan.v6 node '",
            instruction.node_name,
            "' references a released value");
        return values[static_cast<size_t>(argument_ref)];
      }
      return state.constants[static_cast<size_t>(constant_index(argument_ref))];
    };
    for (const VulkanGraphPlanArgument& argument : instruction.arguments) {
      if (argument.kind == VulkanGraphPlanArgumentKind::Value) {
        stack.push_back(load_argument_ref(argument.refs.front()));
        continue;
      }
      c10::impl::GenericList list(argument.list_element_type);
      list.reserve(argument.refs.size());
      for (const int64_t argument_ref : argument.refs) {
        list.emplace_back(load_argument_ref(argument_ref));
      }
      stack.emplace_back(std::move(list));
    }

    const int64_t scope_token = begin_vulkan_graph_execution_scope();
    try {
      if (instruction.kind == VulkanGraphPlanInstructionKind::Dispatcher) {
        TORCH_INTERNAL_ASSERT(instruction.operator_handle);
        c10::Dispatcher::singleton().callBoxed(
            *instruction.operator_handle, &stack);
      } else if (
          instruction.kind == VulkanGraphPlanInstructionKind::ListGetItem) {
        execute_list_getitem_instruction(instruction, stack);
      } else {
        execute_graph_scalar_instruction(instruction, stack);
      }
    } catch (const c10::Error& error) {
      const std::vector<int64_t> counters =
          end_vulkan_graph_execution_scope(scope_token);
      check_implicit_boundary(instruction, counters);
      TORCH_CHECK(
          false,
          "VulkanGraphPlan.v6 node '",
          instruction.node_name,
          "' (",
          instruction.operator_name,
          ") failed: ",
          error.what_without_backtrace());
    } catch (const std::exception& error) {
      const std::vector<int64_t> counters =
          end_vulkan_graph_execution_scope(scope_token);
      check_implicit_boundary(instruction, counters);
      TORCH_CHECK(
          false,
          "VulkanGraphPlan.v6 node '",
          instruction.node_name,
          "' (",
          instruction.operator_name,
          ") failed: ",
          error.what());
    } catch (...) {
      const std::vector<int64_t> counters =
          end_vulkan_graph_execution_scope(scope_token);
      check_implicit_boundary(instruction, counters);
      TORCH_CHECK(
          false,
          "VulkanGraphPlan.v6 node '",
          instruction.node_name,
          "' (",
          instruction.operator_name,
          ") failed with a non-standard exception");
    }
    const std::vector<int64_t> counters =
        end_vulkan_graph_execution_scope(scope_token);
    check_implicit_boundary(instruction, counters);
    if (instruction.output_value_ids.empty()) {
      TORCH_CHECK(
          stack.empty(),
          "VulkanGraphPlan.v6 effect node '",
          instruction.node_name,
          "' produced an undeclared value");
    } else {
      TORCH_CHECK(
          stack.size() == instruction.output_value_ids.size(),
          "VulkanGraphPlan.v6 node '",
          instruction.node_name,
          "' did not produce its declared values");
      for (const auto output_index :
           c10::irange(instruction.output_value_ids.size())) {
        c10::IValue output = std::move(stack[output_index]);
        TORCH_CHECK(
            !output.isTensor() || output.toTensor().is_vulkan(),
            "VulkanGraphPlan.v6 node '",
            instruction.node_name,
            "' produced a non-Vulkan tensor");
        const int64_t output_value_id =
            instruction.output_value_ids[output_index];
        values[static_cast<size_t>(output_value_id)] = std::move(output);
        value_live[static_cast<size_t>(output_value_id)] = true;
      }
    }

    for (const VulkanGraphPlanArgument& argument : instruction.arguments) {
      for (const int64_t argument_ref : argument.refs) {
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
    for (const int64_t output_value_id : instruction.output_value_ids) {
      const VulkanGraphPlanValue& value =
          state.values[static_cast<size_t>(output_value_id)];
      if (
          !value.escapes &&
          value.last_use == static_cast<int64_t>(instruction_index)) {
        values[static_cast<size_t>(output_value_id)] = c10::IValue();
        value_live[static_cast<size_t>(output_value_id)] = false;
      }
    }
  }

  std::vector<Tensor> outputs;
  outputs.reserve(state.output_value_ids.size());
  for (const int64_t output_value_id : state.output_value_ids) {
    c10::IValue& output = values[static_cast<size_t>(output_value_id)];
    TORCH_CHECK(
        value_live[static_cast<size_t>(output_value_id)] && output.isTensor(),
        "VulkanGraphPlan.v6 output references a released or non-Tensor value");
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
