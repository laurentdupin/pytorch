#include <ATen/native/vulkan/planning/GraphPlanExecutor.h>

#ifdef USE_VULKAN_API

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/SyncCounters.h>
#include <ATen/native/vulkan/ops/Convert.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/Softmax.h>
#include <ATen/native/vulkan/ops/TensorState.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/planning/ExecutionObjects.h>
#include <ATen/native/vulkan/planning/GraphProgramPlans.h>
#include <ATen/native/vulkan/planning/Request.h>

#include <c10/core/DispatchKey.h>
#include <c10/util/Exception.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/irange.h>
#include <c10/util/safe_numerics.h>

#include <algorithm>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
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
  int64_t definition{-1};
  int64_t last_use{-1};
  bool escapes{false};
  int64_t resource_slot_id{-1};
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

enum class VulkanGraphPlanResourceWriterKind : uint8_t {
  None,
  LinearContext,
  LayernormContext,
  AddLayernormPlan,
  ScaledAdd,
  ScaledAddLayernormPlan,
  LinearGeluRegionPlan,
  AttentionMath,
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
  std::optional<c10::OperatorHandle> dead_input_reuse_operator_handle;
  VulkanGraphPlanResourceWriterKind resource_writer_kind{
      VulkanGraphPlanResourceWriterKind::None};
  bool reusable_list_arguments{false};
  std::vector<VulkanGraphPlanArgument> arguments;
  std::vector<int64_t> output_value_ids;
  std::vector<int64_t> scratch_resource_slot_ids;
  std::vector<int64_t> release_value_ids;
};

struct VulkanGraphPlanResourceSlot final {
  std::vector<int64_t> sizes;
  ScalarType dtype{kFloat};
  api::StorageType storage_type{api::StorageType::BUFFER};
  api::GPUMemoryLayout memory_layout{
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  api::ExecutionLayout execution_layout{api::ExecutionLayout::BUFFER_DIRECT};
};

enum class VulkanGraphPlanRecordedPartitionState : uint8_t {
  Empty,
  Primed,
  Ready,
  Failed,
};

enum class VulkanGraphPlanRecordedPartitionMode : uint8_t {
  None,
  Prime,
  Capture,
  Replay,
};

struct VulkanGraphPlanRecordedTensorStamp final {
  uint64_t storage_id{0u};
  uint64_t logical_desc_hash{0u};
};

struct VulkanGraphPlanRecordedPartition final {
  int64_t start{0};
  int64_t end{0};
  VulkanGraphPlanRecordedPartitionState state{
      VulkanGraphPlanRecordedPartitionState::Empty};
  std::optional<api::CommandBuffer> command;
  std::vector<VulkanGraphPlanRecordedTensorStamp> tensor_stamps;
  std::vector<std::vector<std::vector<int64_t>>> output_sizes;
  std::vector<api::VulkanBuffer> retained_buffers;
  std::vector<api::VulkanImage> retained_images;
  uint32_t represented_dispatch_count{0u};
};

struct VulkanGraphPlanRecordingArena final {
  std::unique_ptr<api::CommandPool> command_pool;
  std::unique_ptr<api::DescriptorPool> descriptor_pool;
  std::vector<VulkanGraphPlanRecordedPartition> partitions;
};

struct VulkanGraphPlanResourceArena final {
  std::vector<Tensor> tensors;
  std::unique_ptr<VulkanGraphPlanRecordingArena> recording;
  c10::DeviceIndex device_index{-1};
  api::VulkanSubmission submission{};
  bool poisoned{false};
};

struct VulkanGraphPlanArenaRetirementBundle final {
  std::vector<api::VulkanBuffer> tensor_buffers;
  std::unique_ptr<VulkanGraphPlanRecordingArena> recording;
};

struct VulkanGraphPlanInvocationWorkspace final {
  std::vector<c10::IValue> values;
  std::vector<uint8_t> value_live;
  std::vector<c10::IValue> stack;
  std::vector<std::vector<std::optional<c10::impl::GenericList>>>
      list_arguments;

  void reset() {
    stack.clear();
    for (auto& instruction_lists : list_arguments) {
      for (auto& list : instruction_lists) {
        if (list) {
          list->clear();
        }
      }
    }
    for (const auto value_index : c10::irange(values.size())) {
      if (value_live[value_index]) {
        values[value_index] = c10::IValue();
        value_live[value_index] = 0u;
      }
    }
  }
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

VulkanGraphPlanResourceWriterKind graph_resource_writer_kind(
    const std::string& operator_name) {
  if (operator_name == "vulkan_prepack::run_linear_context") {
    return VulkanGraphPlanResourceWriterKind::LinearContext;
  }
  if (operator_name == "vulkan_prepack::run_layernorm_context") {
    return VulkanGraphPlanResourceWriterKind::LayernormContext;
  }
  if (operator_name == "vulkan_prepack::run_graph_add_layernorm_plan") {
    return VulkanGraphPlanResourceWriterKind::AddLayernormPlan;
  }
  if (operator_name == "vulkan_prepack::run_graph_scaled_add") {
    return VulkanGraphPlanResourceWriterKind::ScaledAdd;
  }
  if (operator_name ==
      "vulkan_prepack::run_graph_scaled_add_layernorm_plan") {
    return VulkanGraphPlanResourceWriterKind::ScaledAddLayernormPlan;
  }
  if (operator_name == "vulkan_prepack::run_vulkan_graph_region_plan") {
    return VulkanGraphPlanResourceWriterKind::LinearGeluRegionPlan;
  }
  if (operator_name == "vulkan_prepack::run_graph_attention_math") {
    return VulkanGraphPlanResourceWriterKind::AttentionMath;
  }
  return VulkanGraphPlanResourceWriterKind::None;
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
      "VulkanGraphPlan.v9 graph scalar node '",
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
          "VulkanGraphPlan.v9 graph scalar node '",
          instruction.node_name,
          "' divides by zero");
      TORCH_CHECK(
          left != std::numeric_limits<int64_t>::min() || right != -1,
          "VulkanGraphPlan.v9 graph scalar node '",
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
      "VulkanGraphPlan.v9 graph scalar node '",
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
      "VulkanGraphPlan.v9 list projection node '",
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
      "VulkanGraphPlan.v9 list projection node '",
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

bool schema_has_list_return(const c10::FunctionSchema& schema) {
  return std::any_of(
      schema.returns().begin(),
      schema.returns().end(),
      [](const c10::Argument& argument) {
        c10::TypePtr type = argument.type();
        if (const auto optional_type = type->cast<c10::OptionalType>()) {
          type = optional_type->getElementType();
        }
        return type->cast<c10::ListType>() != nullptr;
      });
}

int64_t constant_index(const int64_t argument_ref) {
  TORCH_INTERNAL_ASSERT(argument_ref < 0);
  TORCH_CHECK(
      argument_ref != std::numeric_limits<int64_t>::min(),
      "VulkanGraphPlan.v9 constant reference underflow");
  return -argument_ref - 1;
}

bool any_implicit_boundary(const VulkanGraphExecutionScopeCounts& counters) {
  return std::any_of(counters.begin(), counters.end(), [](const int64_t value) {
    return value != 0;
  });
}

bool ivalue_references_tensor_impl(
    const c10::IValue& value,
    const TensorImpl* const candidate_impl) {
  if (value.isTensor()) {
    const Tensor tensor = value.toTensor();
    return tensor.defined() && tensor.unsafeGetTensorImpl() == candidate_impl;
  }
  if (value.isList()) {
    for (const c10::IValue& element : value.toList()) {
      if (ivalue_references_tensor_impl(element, candidate_impl)) {
        return true;
      }
    }
  }
  if (value.isTuple()) {
    for (const c10::IValue& element : value.toTupleRef().elements()) {
      if (ivalue_references_tensor_impl(element, candidate_impl)) {
        return true;
      }
    }
  }
  return false;
}

bool can_reuse_dead_input(
    const VulkanGraphPlanInstruction& instruction,
    const size_t instruction_index,
    const std::vector<VulkanGraphPlanValue>& value_plan,
    const std::vector<c10::IValue>& values,
    const std::vector<uint8_t>& value_live,
    const std::vector<Tensor>& inputs,
    const std::vector<c10::IValue>& constants) {
  if (
      !instruction.dead_input_reuse_operator_handle ||
      instruction.arguments.empty() ||
      instruction.arguments.front().kind != VulkanGraphPlanArgumentKind::Value) {
    return false;
  }
  const int64_t value_id = instruction.arguments.front().refs.front();
  if (
      value_id < static_cast<int64_t>(inputs.size()) ||
      value_id >= static_cast<int64_t>(values.size()) ||
      !value_live[static_cast<size_t>(value_id)]) {
    return false;
  }
  const VulkanGraphPlanValue& planned_value =
      value_plan[static_cast<size_t>(value_id)];
  if (
      planned_value.escapes ||
      planned_value.last_use != static_cast<int64_t>(instruction_index) ||
      !values[static_cast<size_t>(value_id)].isTensor()) {
    return false;
  }
  const Tensor candidate = values[static_cast<size_t>(value_id)].toTensor();
  if (!candidate.defined() || !candidate.is_vulkan()) {
    return false;
  }
  if (!convert(candidate).owns_unique_storage()) {
    return false;
  }
  const TensorImpl* const candidate_impl = candidate.unsafeGetTensorImpl();
  for (const Tensor& input : inputs) {
    if (input.unsafeGetTensorImpl() == candidate_impl) {
      return false;
    }
  }
  for (const c10::IValue& constant : constants) {
    if (ivalue_references_tensor_impl(constant, candidate_impl)) {
      return false;
    }
  }
  for (const auto live_value_id : c10::irange(values.size())) {
    if (
        live_value_id != static_cast<size_t>(value_id) &&
        value_live[live_value_id] &&
        ivalue_references_tensor_impl(values[live_value_id], candidate_impl)) {
      return false;
    }
  }
  return true;
}

bool same_vulkan_resource(const Tensor& left, const Tensor& right) {
  return left.defined() && right.defined() && left.is_vulkan() &&
      right.is_vulkan() && left.sizes().equals(right.sizes()) &&
      left.scalar_type() == right.scalar_type() &&
      convert(left).storage_identity() == convert(right).storage_identity();
}

bool same_vulkan_storage(const Tensor& left, const Tensor& right) {
  return left.defined() && right.defined() && left.is_vulkan() &&
      right.is_vulkan() && left.scalar_type() == right.scalar_type() &&
      convert(left).storage_identity() == convert(right).storage_identity();
}

std::vector<VulkanGraphPlanRecordedTensorStamp> recorded_arena_stamps(
    const VulkanGraphPlanResourceArena& arena) {
  std::vector<VulkanGraphPlanRecordedTensorStamp> stamps;
  stamps.reserve(arena.tensors.size());
  for (const Tensor& tensor : arena.tensors) {
    stamps.push_back(
        {tensor_storage_identity(tensor), tensor_logical_desc_hash(tensor)});
  }
  return stamps;
}

bool recorded_arena_stamps_match(
    const VulkanGraphPlanResourceArena& arena,
    const std::vector<VulkanGraphPlanRecordedTensorStamp>& stamps) {
  if (arena.tensors.size() != stamps.size()) {
    return false;
  }
  for (const auto index : c10::irange(arena.tensors.size())) {
    if (
        tensor_storage_identity(arena.tensors[index]) !=
            stamps[index].storage_id ||
        tensor_logical_desc_hash(arena.tensors[index]) !=
            stamps[index].logical_desc_hash) {
      return false;
    }
  }
  return true;
}

api::PipelineBarrier prepare_recorded_partition_entry(
    VulkanGraphPlanResourceArena& arena) {
  api::PipelineBarrier barrier{};
  const auto read_write = static_cast<api::MemoryAccessType>(
      api::MemoryAccessType::READ | api::MemoryAccessType::WRITE);
  for (Tensor& tensor : arena.tensors) {
    convert(tensor).buffer(barrier, api::PipelineStage::COMPUTE, read_write);
  }
  return barrier;
}

void record_partition_exit_state(VulkanGraphPlanResourceArena& arena) {
  api::PipelineBarrier ignored{};
  const auto read_write = static_cast<api::MemoryAccessType>(
      api::MemoryAccessType::READ | api::MemoryAccessType::WRITE);
  for (Tensor& tensor : arena.tensors) {
    convert(tensor).buffer(ignored, api::PipelineStage::COMPUTE, read_write);
  }
}

enum class VulkanGraphPlanResourceWriteResult : uint8_t {
  NotApplicable,
  Written,
  ProducedUnowned,
  NeedsDispatcher,
};

api::StorageType parse_resource_storage_type(const int64_t value) {
  if (
      value >= static_cast<int64_t>(api::StorageType::BUFFER) &&
      value <= static_cast<int64_t>(api::StorageType::TEXTURE_2D)) {
    return static_cast<api::StorageType>(value);
  }
  TORCH_CHECK(
      false, "VulkanGraphPlan.v9 has invalid resource storage type ", value);
}

api::GPUMemoryLayout parse_resource_memory_layout(const int64_t value) {
  if (
      value >=
          static_cast<int64_t>(api::GPUMemoryLayout::TENSOR_WIDTH_PACKED) &&
      value <=
          static_cast<int64_t>(api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED)) {
    return static_cast<api::GPUMemoryLayout>(value);
  }
  TORCH_CHECK(
      false, "VulkanGraphPlan.v9 has invalid resource memory layout ", value);
}

api::ExecutionLayout parse_resource_execution_layout(const int64_t value) {
  if (
      value >= static_cast<int64_t>(api::ExecutionLayout::TEXTURE) &&
      value <= static_cast<int64_t>(api::ExecutionLayout::PACKED_WEIGHT)) {
    return static_cast<api::ExecutionLayout>(value);
  }
  TORCH_CHECK(
      false, "VulkanGraphPlan.v9 has invalid resource execution layout ", value);
}

VulkanGraphPlanResourceWriteResult execute_resource_writer(
    VulkanGraphPlan& plan,
    const VulkanGraphPlanInstruction& instruction,
    const std::vector<VulkanGraphPlanValue>& value_plan,
    const int64_t arena_index,
    std::vector<c10::IValue>& stack) {
  if (
      arena_index < 0 ||
      instruction.resource_writer_kind ==
          VulkanGraphPlanResourceWriterKind::None) {
    return VulkanGraphPlanResourceWriteResult::NotApplicable;
  }
  std::vector<Tensor*> targets;
  targets.reserve(instruction.output_value_ids.size());
  for (const int64_t output_value_id : instruction.output_value_ids) {
    const int64_t slot_id =
        value_plan.at(static_cast<size_t>(output_value_id)).resource_slot_id;
    TORCH_INTERNAL_ASSERT(slot_id >= 0);
    targets.push_back(&plan.resource_tensor(arena_index, slot_id));
  }

  switch (instruction.resource_writer_kind) {
    case VulkanGraphPlanResourceWriterKind::LinearContext: {
      TORCH_CHECK(
          targets.size() == 1u && stack.size() == 2u && stack[0].isTensor(),
          "VulkanGraphPlan.v9 linear resource writer has invalid arguments");
      const auto context = stack[1].toCustomClass<LinearPackedContext>();
      Tensor output_candidate = *targets[0];
      Tensor result = run_linear_context_out(
          stack[0].toTensor(), context, output_candidate);
      stack.clear();
      stack.emplace_back(std::move(result));
      return same_vulkan_resource(stack[0].toTensor(), *targets[0])
          ? VulkanGraphPlanResourceWriteResult::Written
          : VulkanGraphPlanResourceWriteResult::ProducedUnowned;
    }
    case VulkanGraphPlanResourceWriterKind::AddLayernormPlan: {
      TORCH_CHECK(
          targets.size() == 2u && stack.size() == 3u && stack[0].isTensor() &&
              stack[1].isTensor(),
          "VulkanGraphPlan.v9 add-layernorm resource writer has invalid arguments");
      const auto region_plan =
          stack[2].toCustomClass<GraphAddLayernormPlan>();
      auto result = try_run_graph_add_layernorm_plan_out(
          stack[0].toTensor(),
          stack[1].toTensor(),
          region_plan,
          *targets[0],
          *targets[1]);
      if (!result) {
        return VulkanGraphPlanResourceWriteResult::NeedsDispatcher;
      }
      Tensor residual_output = std::get<0>(*result);
      Tensor normalized_output = std::get<1>(*result);
      TORCH_CHECK(
          same_vulkan_resource(residual_output, *targets[0]) &&
              same_vulkan_resource(normalized_output, *targets[1]),
          "VulkanGraphPlan.v9 add-layernorm resource writer rebound a stable slot");
      stack.clear();
      stack.emplace_back(std::move(residual_output));
      stack.emplace_back(std::move(normalized_output));
      return VulkanGraphPlanResourceWriteResult::Written;
    }
    case VulkanGraphPlanResourceWriterKind::LayernormContext: {
      TORCH_CHECK(
          targets.size() == 1u && stack.size() == 3u && stack[0].isTensor() &&
              stack[1].isIntList(),
          "VulkanGraphPlan.v9 layernorm resource writer has invalid arguments");
      const auto context = stack[2].toCustomClass<LayernormPackedContext>();
      Tensor output_candidate = *targets[0];
      Tensor result = run_layernorm_context_out(
          stack[0].toTensor(),
          stack[1].toIntVector(),
          context,
          output_candidate);
      stack.clear();
      stack.emplace_back(std::move(result));
      return same_vulkan_resource(stack[0].toTensor(), *targets[0])
          ? VulkanGraphPlanResourceWriteResult::Written
          : VulkanGraphPlanResourceWriteResult::ProducedUnowned;
    }
    case VulkanGraphPlanResourceWriterKind::ScaledAdd: {
      TORCH_CHECK(
          targets.size() == 1u && stack.size() == 3u &&
              stack[0].isTensor() && stack[1].isTensor() &&
              stack[2].isTensor() &&
              instruction.scratch_resource_slot_ids.size() == 1u,
          "VulkanGraphPlan.v9 scaled-add resource writer has invalid arguments");
      Tensor& scaled_scratch = plan.resource_tensor(
          arena_index, instruction.scratch_resource_slot_ids[0]);
      auto result = try_run_graph_scaled_add_out(
          stack[0].toTensor(),
          stack[1].toTensor(),
          stack[2].toTensor(),
          scaled_scratch,
          *targets[0]);
      if (!result) {
        return VulkanGraphPlanResourceWriteResult::NeedsDispatcher;
      }
      TORCH_CHECK(
          same_vulkan_resource(*result, *targets[0]),
          "VulkanGraphPlan.v9 scaled-add resource writer rebound a stable slot");
      stack.clear();
      stack.emplace_back(std::move(*result));
      return VulkanGraphPlanResourceWriteResult::Written;
    }
    case VulkanGraphPlanResourceWriterKind::ScaledAddLayernormPlan: {
      TORCH_CHECK(
          targets.size() == 2u && stack.size() == 4u &&
              stack[0].isTensor() && stack[1].isTensor() &&
              stack[2].isTensor() &&
              instruction.scratch_resource_slot_ids.size() == 1u,
          "VulkanGraphPlan.v9 scaled-add-layernorm resource writer has invalid arguments");
      Tensor& scaled_scratch = plan.resource_tensor(
          arena_index, instruction.scratch_resource_slot_ids[0]);
      const auto region_plan =
          stack[3].toCustomClass<GraphAddLayernormPlan>();
      auto result = try_run_graph_scaled_add_layernorm_plan_out(
          stack[0].toTensor(),
          stack[1].toTensor(),
          stack[2].toTensor(),
          region_plan,
          scaled_scratch,
          *targets[0],
          *targets[1]);
      if (!result) {
        return VulkanGraphPlanResourceWriteResult::NeedsDispatcher;
      }
      Tensor residual_output = std::get<0>(*result);
      Tensor normalized_output = std::get<1>(*result);
      TORCH_CHECK(
          same_vulkan_resource(residual_output, *targets[0]) &&
              same_vulkan_resource(normalized_output, *targets[1]),
          "VulkanGraphPlan.v9 scaled-add-layernorm writer rebound a stable slot");
      stack.clear();
      stack.emplace_back(std::move(residual_output));
      stack.emplace_back(std::move(normalized_output));
      return VulkanGraphPlanResourceWriteResult::Written;
    }
    case VulkanGraphPlanResourceWriterKind::LinearGeluRegionPlan: {
      TORCH_CHECK(
          targets.size() == 1u && stack.size() == 2u && stack[0].isTensor(),
          "VulkanGraphPlan.v9 linear-GELU resource writer has invalid arguments");
      const auto region_plan =
          stack[1].toCustomClass<VulkanGraphRegionPlan>();
      const VulkanGraphRegionFamily family = region_plan->schema().family;
      if (
          family != VulkanGraphRegionFamily::LinearGeluTanh &&
          family != VulkanGraphRegionFamily::LinearGeluNone) {
        return VulkanGraphPlanResourceWriteResult::NeedsDispatcher;
      }
      auto result = try_run_vulkan_graph_region_plan_out(
          stack[0].toTensor(), region_plan, *targets[0]);
      if (!result) {
        return VulkanGraphPlanResourceWriteResult::NeedsDispatcher;
      }
      stack.clear();
      stack.emplace_back(std::move(*result));
      return same_vulkan_storage(stack[0].toTensor(), *targets[0])
          ? VulkanGraphPlanResourceWriteResult::Written
          : VulkanGraphPlanResourceWriteResult::ProducedUnowned;
    }
    case VulkanGraphPlanResourceWriterKind::AttentionMath: {
      TORCH_CHECK(
          targets.size() == 1u && stack.size() == 4u &&
              stack[0].isTensor() && stack[1].isTensor() &&
              stack[2].isTensor() && stack[3].isDouble() &&
              instruction.scratch_resource_slot_ids.size() == 3u,
          "VulkanGraphPlan.v9 attention-math resource writer has invalid arguments");
      Tensor& scaled_query = plan.resource_tensor(
          arena_index, instruction.scratch_resource_slot_ids[0]);
      Tensor& scores = plan.resource_tensor(
          arena_index, instruction.scratch_resource_slot_ids[1]);
      Tensor& probability = plan.resource_tensor(
          arena_index, instruction.scratch_resource_slot_ids[2]);
      Tensor result =
          at::native::vulkan::ops::run_graph_attention_math_out_vulkan(
              stack[0].toTensor(),
              stack[1].toTensor(),
              stack[2].toTensor(),
              stack[3].toDouble(),
              *targets[0],
              scaled_query,
              scores,
              probability);
      stack.clear();
      stack.emplace_back(std::move(result));
      return same_vulkan_storage(stack[0].toTensor(), *targets[0])
          ? VulkanGraphPlanResourceWriteResult::Written
          : VulkanGraphPlanResourceWriteResult::ProducedUnowned;
    }
    case VulkanGraphPlanResourceWriterKind::None:
      break;
  }
  return VulkanGraphPlanResourceWriteResult::NotApplicable;
}

void check_implicit_boundary(
    const VulkanGraphPlanInstruction& instruction,
    const VulkanGraphExecutionScopeCounts& counters) {
  TORCH_CHECK(
      !any_implicit_boundary(counters),
      "VulkanGraphPlan.v9 node '",
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
        "VulkanGraphPlan.v9 rejects concurrent invocation");
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
  std::vector<VulkanGraphPlanResourceSlot> resource_slots;
  std::vector<VulkanGraphPlanResourceArena> resource_arenas;
  std::vector<std::pair<int64_t, int64_t>> recorded_partition_ranges;
  int64_t resource_arena_flight_depth{0};
  int64_t input_count{0};
  bool submission_owned{true};
  VulkanPlanningRequest planning_request;
  mutable VulkanGraphPlanInvocationWorkspace invocation_workspace;
  mutable std::mutex submission_mutex;
  c10::DeviceIndex submission_device_index{-1};
  api::VulkanSubmission last_submission{};
  uint64_t invocation_generation{0u};
  mutable std::atomic<uint64_t> dead_input_reuse_count{0u};
  mutable std::atomic<uint64_t> resource_arena_generation_count{0u};
  mutable std::atomic<uint64_t> resource_arena_capture_count{0u};
  mutable std::atomic<uint64_t> resource_arena_reuse_count{0u};
  mutable std::atomic<uint64_t> resource_arena_spill_count{0u};
  mutable std::atomic<uint64_t> resource_write_count{0u};
  mutable std::atomic<uint64_t> resource_writer_bypass_count{0u};
  mutable std::atomic<uint64_t> recorded_partition_prime_count{0u};
  mutable std::atomic<uint64_t> recorded_partition_capture_count{0u};
  mutable std::atomic<uint64_t> recorded_partition_replay_count{0u};
  mutable std::atomic<uint64_t> recorded_partition_failure_count{0u};
  mutable std::atomic<uint64_t> recorded_partition_represented_dispatch_count{
      0u};
};

VulkanGraphPlan::VulkanGraphPlan(std::shared_ptr<State> state)
    : state_(std::move(state)) {
  TORCH_CHECK(valid(), "VulkanGraphPlan.v9 has an invalid schema");
}

VulkanGraphPlan::~VulkanGraphPlan() noexcept {
  if (!state_) {
    return;
  }
  api::VulkanGraphProgramInvocationCounters& counters =
      api::vulkan_graph_program_invocation_counters();
  for (VulkanGraphPlanResourceArena& arena : state_->resource_arenas) {
    if (arena.tensors.empty()) {
      continue;
    }
    if (arena.poisoned) {
      for (Tensor& tensor : arena.tensors) {
        if (tensor.defined()) {
          (void)tensor.unsafeReleaseTensorImpl();
          counters.resource_arena_unsafe_slot_leak_count.fetch_add(
              1u, std::memory_order_relaxed);
        }
      }
      (void)arena.recording.release();
      continue;
    }
    api::Context* context = nullptr;
    auto retirement =
        std::make_unique<VulkanGraphPlanArenaRetirementBundle>();
    retirement->recording = std::move(arena.recording);
    for (Tensor& slot : arena.tensors) {
      if (!slot.defined()) {
        continue;
      }
      Tensor tensor = std::move(slot);
      try {
        TORCH_CHECK(
            tensor.use_count() == 1u,
            "VulkanGraphPlan.v9 resource slot escaped its program arena");
        vTensor& v_tensor = convert(tensor);
        if (!context) {
          context = v_tensor.context();
        }
        std::vector<api::VulkanBuffer> buffers =
            v_tensor.release_graph_program_owned_buffers();
        retirement->tensor_buffers.insert(
            retirement->tensor_buffers.end(),
            std::make_move_iterator(buffers.begin()),
            std::make_move_iterator(buffers.end()));
      } catch (...) {
        (void)tensor.unsafeReleaseTensorImpl();
        counters.resource_arena_unsafe_slot_leak_count.fetch_add(
            1u, std::memory_order_relaxed);
      }
    }
    if (retirement->tensor_buffers.empty() && !retirement->recording) {
      continue;
    }
    if (!context) {
      counters.resource_arena_retirement_failure_count.fetch_add(
          1u, std::memory_order_relaxed);
      (void)retirement.release();
      continue;
    }
    try {
      if (
          arena.submission.timeline_value == 0u ||
          context->graph_program_submission_complete(arena.submission)) {
        retirement.reset();
        counters.resource_arena_immediate_release_count.fetch_add(
            1u, std::memory_order_relaxed);
        continue;
      }
      auto* const resources = retirement.release();
      context->retire_graph_program_resource(
          arena.submission, [resources]() { delete resources; });
      counters.resource_arena_retire_enqueued_count.fetch_add(
          1u, std::memory_order_relaxed);
    } catch (...) {
      counters.resource_arena_retirement_failure_count.fetch_add(
          1u, std::memory_order_relaxed);
      (void)retirement.release();
    }
  }
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

int64_t VulkanGraphPlan::invocation_value_slot_count() const {
  return state_
      ? static_cast<int64_t>(state_->invocation_workspace.values.size())
      : 0;
}

int64_t VulkanGraphPlan::invocation_list_slot_count() const {
  if (!state_) {
    return 0;
  }
  int64_t count = 0;
  for (const auto& instruction_lists :
       state_->invocation_workspace.list_arguments) {
    count += static_cast<int64_t>(std::count_if(
        instruction_lists.begin(),
        instruction_lists.end(),
        [](const auto& list) { return list.has_value(); }));
  }
  return count;
}

int64_t VulkanGraphPlan::invocation_stack_capacity() const {
  return state_
      ? static_cast<int64_t>(state_->invocation_workspace.stack.capacity())
      : 0;
}

int64_t VulkanGraphPlan::dead_input_reuse_instruction_count() const {
  if (!state_) {
    return 0;
  }
  return static_cast<int64_t>(std::count_if(
      state_->instructions.begin(),
      state_->instructions.end(),
      [](const VulkanGraphPlanInstruction& instruction) {
        return instruction.dead_input_reuse_operator_handle.has_value();
      }));
}

int64_t VulkanGraphPlan::dead_input_reuse_count() const {
  if (!state_) {
    return 0;
  }
  const uint64_t count =
      state_->dead_input_reuse_count.load(std::memory_order_relaxed);
  TORCH_CHECK(
      count <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "VulkanGraphPlan.v9 dead-input reuse count overflow");
  return static_cast<int64_t>(count);
}

int64_t VulkanGraphPlan::resource_slot_count() const {
  return state_ ? static_cast<int64_t>(state_->resource_slots.size()) : 0;
}

int64_t VulkanGraphPlan::resource_value_count() const {
  if (!state_) {
    return 0;
  }
  return static_cast<int64_t>(std::count_if(
      state_->values.begin(),
      state_->values.end(),
      [](const VulkanGraphPlanValue& value) {
        return value.resource_slot_id >= 0;
      }));
}

int64_t VulkanGraphPlan::resource_writer_instruction_count() const {
  if (!state_) {
    return 0;
  }
  return static_cast<int64_t>(std::count_if(
      state_->instructions.begin(),
      state_->instructions.end(),
      [](const VulkanGraphPlanInstruction& instruction) {
        return instruction.resource_writer_kind !=
            VulkanGraphPlanResourceWriterKind::None;
      }));
}

int64_t VulkanGraphPlan::resource_arena_flight_depth() const {
  return state_ ? state_->resource_arena_flight_depth : 0;
}

int64_t VulkanGraphPlan::resource_arena_generation_count() const {
  return state_
      ? static_cast<int64_t>(state_->resource_arena_generation_count.load(
            std::memory_order_relaxed))
      : 0;
}

int64_t VulkanGraphPlan::resource_arena_capture_count() const {
  return state_
      ? static_cast<int64_t>(state_->resource_arena_capture_count.load(
            std::memory_order_relaxed))
      : 0;
}

int64_t VulkanGraphPlan::resource_arena_reuse_count() const {
  return state_
      ? static_cast<int64_t>(state_->resource_arena_reuse_count.load(
            std::memory_order_relaxed))
      : 0;
}

int64_t VulkanGraphPlan::resource_arena_spill_count() const {
  return state_
      ? static_cast<int64_t>(state_->resource_arena_spill_count.load(
            std::memory_order_relaxed))
      : 0;
}

int64_t VulkanGraphPlan::resource_write_count() const {
  return state_
      ? static_cast<int64_t>(
            state_->resource_write_count.load(std::memory_order_relaxed))
      : 0;
}

int64_t VulkanGraphPlan::resource_writer_bypass_count() const {
  return state_
      ? static_cast<int64_t>(state_->resource_writer_bypass_count.load(
            std::memory_order_relaxed))
      : 0;
}

int64_t VulkanGraphPlan::recorded_partition_count() const {
  return state_
      ? static_cast<int64_t>(state_->recorded_partition_ranges.size())
      : 0;
}

int64_t VulkanGraphPlan::recorded_partition_instruction_count() const {
  if (!state_) {
    return 0;
  }
  int64_t count = 0;
  for (const auto& range : state_->recorded_partition_ranges) {
    count += range.second - range.first;
  }
  return count;
}

int64_t VulkanGraphPlan::recorded_partition_prime_count() const {
  return state_ ? static_cast<int64_t>(
                      state_->recorded_partition_prime_count.load(
                          std::memory_order_relaxed))
                : 0;
}

int64_t VulkanGraphPlan::recorded_partition_capture_count() const {
  return state_ ? static_cast<int64_t>(
                      state_->recorded_partition_capture_count.load(
                          std::memory_order_relaxed))
                : 0;
}

int64_t VulkanGraphPlan::recorded_partition_replay_count() const {
  return state_ ? static_cast<int64_t>(
                      state_->recorded_partition_replay_count.load(
                          std::memory_order_relaxed))
                : 0;
}

int64_t VulkanGraphPlan::recorded_partition_failure_count() const {
  return state_ ? static_cast<int64_t>(
                      state_->recorded_partition_failure_count.load(
                          std::memory_order_relaxed))
                : 0;
}

int64_t VulkanGraphPlan::recorded_partition_represented_dispatch_count() const {
  return state_ ? static_cast<int64_t>(
                      state_->recorded_partition_represented_dispatch_count.load(
                          std::memory_order_relaxed))
                : 0;
}

int64_t VulkanGraphPlan::value_count() const {
  return state_ ? static_cast<int64_t>(state_->values.size()) : 0;
}

int64_t VulkanGraphPlan::output_count() const {
  return state_ ? static_cast<int64_t>(state_->output_value_ids.size()) : 0;
}

bool VulkanGraphPlan::submission_owned() const {
  return state_ && state_->submission_owned;
}

int64_t VulkanGraphPlan::planning_model_domain() const {
  return state_ ? static_cast<int64_t>(state_->planning_request.model_domain)
                : 0;
}

int64_t VulkanGraphPlan::planning_execution_phase() const {
  return state_ ? static_cast<int64_t>(state_->planning_request.execution_phase)
                : 0;
}

bool VulkanGraphPlan::planning_prefer_packed_layout_propagation() const {
  return state_ &&
      state_->planning_request.prefer_packed_layout_propagation;
}

std::optional<std::vector<int64_t>>
VulkanGraphPlan::planning_fixed_shape_graph_input_sizes() const {
  return state_ ? state_->planning_request.fixed_shape_graph_input_sizes
                : std::nullopt;
}

int64_t VulkanGraphPlan::invocation_generation() const {
  if (!state_) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(state_->submission_mutex);
  return static_cast<int64_t>(state_->invocation_generation);
}

int64_t VulkanGraphPlan::last_submission_value() const {
  if (!state_) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(state_->submission_mutex);
  return static_cast<int64_t>(state_->last_submission.timeline_value);
}

bool VulkanGraphPlan::last_submission_complete() const {
  if (!state_) {
    return true;
  }
  std::lock_guard<std::mutex> lock(state_->submission_mutex);
  if (state_->last_submission.timeline_value == 0u) {
    return true;
  }
  TORCH_CHECK(
      state_->submission_device_index >= 0,
      "VulkanGraphPlan.v9 submission has no device");
  return api::context(state_->submission_device_index)
      ->graph_program_submission_complete(state_->last_submission);
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
      state_->values.size() < static_cast<size_t>(state_->input_count) ||
      state_->invocation_workspace.values.size() != state_->values.size() ||
      state_->invocation_workspace.value_live.size() != state_->values.size() ||
      state_->invocation_workspace.list_arguments.size() !=
          state_->instructions.size() ||
      (state_->resource_slots.empty() !=
       (state_->resource_arena_flight_depth == 0))) {
    return false;
  }
  int64_t next_value_id = state_->input_count;
  size_t maximum_argument_count = 0u;
  std::vector<uint8_t> release_scheduled(state_->values.size(), uint8_t{0u});
  for (const auto instruction_index :
       c10::irange(state_->instructions.size())) {
    const VulkanGraphPlanInstruction& instruction =
        state_->instructions[instruction_index];
    const auto& instruction_lists =
        state_->invocation_workspace.list_arguments[instruction_index];
    if (
        instruction.node_name.empty() || instruction.operator_name.empty() ||
        ((instruction.kind == VulkanGraphPlanInstructionKind::Dispatcher) !=
         instruction.operator_handle.has_value()) ||
        instruction_lists.size() != instruction.arguments.size()) {
      return false;
    }
    maximum_argument_count =
        std::max(maximum_argument_count, instruction.arguments.size());
    if (
        instruction.dead_input_reuse_operator_handle &&
        (instruction.kind != VulkanGraphPlanInstructionKind::Dispatcher ||
         instruction.arguments.size() != 1u ||
         instruction.arguments.front().kind !=
             VulkanGraphPlanArgumentKind::Value ||
         instruction.output_value_ids.size() != 1u)) {
      return false;
    }
    const bool any_resource_output = std::any_of(
        instruction.output_value_ids.begin(),
        instruction.output_value_ids.end(),
        [this](const int64_t value_id) {
          return state_->values[static_cast<size_t>(value_id)].resource_slot_id >=
              0;
        });
    const bool all_resource_outputs = !instruction.output_value_ids.empty() &&
        std::all_of(
            instruction.output_value_ids.begin(),
            instruction.output_value_ids.end(),
            [this](const int64_t value_id) {
              return state_->values[static_cast<size_t>(value_id)]
                         .resource_slot_id >= 0;
            });
    if (
        any_resource_output != all_resource_outputs ||
        (any_resource_output !=
         (instruction.resource_writer_kind !=
          VulkanGraphPlanResourceWriterKind::None))) {
      return false;
    }
    if (
        ((instruction.resource_writer_kind ==
              VulkanGraphPlanResourceWriterKind::LinearContext ||
          instruction.resource_writer_kind ==
              VulkanGraphPlanResourceWriterKind::LinearGeluRegionPlan ||
          instruction.resource_writer_kind ==
              VulkanGraphPlanResourceWriterKind::ScaledAdd) &&
         instruction.output_value_ids.size() != 1u) ||
        ((instruction.resource_writer_kind ==
              VulkanGraphPlanResourceWriterKind::AddLayernormPlan ||
          instruction.resource_writer_kind ==
              VulkanGraphPlanResourceWriterKind::ScaledAddLayernormPlan) &&
         instruction.output_value_ids.size() != 2u)) {
      return false;
    }
    if (
        instruction.kind != VulkanGraphPlanInstructionKind::Dispatcher &&
        (instruction.output_value_ids.size() != 1u ||
         instruction.arguments.size() != 2u)) {
      return false;
    }
    for (const auto argument_index :
         c10::irange(instruction.arguments.size())) {
      const VulkanGraphPlanArgument& argument =
          instruction.arguments[argument_index];
      if (
          (argument.kind == VulkanGraphPlanArgumentKind::Value &&
           argument.refs.size() != 1u) ||
          (argument.kind == VulkanGraphPlanArgumentKind::List &&
           !argument.list_element_type) ||
          instruction_lists[argument_index].has_value() !=
              (argument.kind == VulkanGraphPlanArgumentKind::List &&
               instruction.reusable_list_arguments)) {
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
    for (const int64_t release_value_id : instruction.release_value_ids) {
      if (
          release_value_id < 0 ||
          release_value_id >= static_cast<int64_t>(state_->values.size()) ||
          release_scheduled[static_cast<size_t>(release_value_id)] ||
          state_->values[static_cast<size_t>(release_value_id)].escapes ||
          state_->values[static_cast<size_t>(release_value_id)].last_use !=
              static_cast<int64_t>(instruction_index)) {
        return false;
      }
      release_scheduled[static_cast<size_t>(release_value_id)] = 1u;
    }
  }
  if (next_value_id != static_cast<int64_t>(state_->values.size())) {
    return false;
  }
  if (state_->invocation_workspace.stack.capacity() < maximum_argument_count) {
    return false;
  }
  for (const auto value_index : c10::irange(state_->values.size())) {
    const VulkanGraphPlanValue& value = state_->values[value_index];
    const bool should_release = !value.escapes && value.last_use >= 0;
    if (static_cast<bool>(release_scheduled[value_index]) != should_release) {
      return false;
    }
    if (
        value.resource_slot_id < -1 ||
        value.resource_slot_id >=
            static_cast<int64_t>(state_->resource_slots.size()) ||
        (value.resource_slot_id >= 0 &&
         (value.escapes || value.definition < 0 ||
          value.last_use < value.definition))) {
      return false;
    }
  }
  for (const VulkanGraphPlanResourceSlot& slot : state_->resource_slots) {
    if (
        slot.dtype != kFloat || slot.sizes.empty() ||
        (api::uses_buffer_execution(slot.execution_layout) !=
         (slot.storage_type == api::StorageType::BUFFER)) ||
        slot.execution_layout == api::ExecutionLayout::BUFFER_VIEW ||
        !std::all_of(slot.sizes.begin(), slot.sizes.end(), [](const int64_t size) {
          return size > 0;
        })) {
      return false;
    }
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

void VulkanGraphPlan::record_submission(
    const c10::DeviceIndex device_index,
    const api::VulkanSubmission& submission) {
  TORCH_INTERNAL_ASSERT(state_ && state_->submission_owned);
  std::lock_guard<std::mutex> lock(state_->submission_mutex);
  TORCH_CHECK(
      state_->invocation_generation <
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "VulkanGraphPlan.v9 invocation generation overflow");
  state_->submission_device_index = device_index;
  state_->last_submission = submission;
  ++state_->invocation_generation;
}

int64_t VulkanGraphPlan::acquire_resource_arena(
    const c10::DeviceIndex device_index) {
  TORCH_INTERNAL_ASSERT(state_);
  if (state_->resource_slots.empty()) {
    return -1;
  }
  api::Context* const context = api::context(device_index);
  for (const auto arena_index : c10::irange(state_->resource_arenas.size())) {
    VulkanGraphPlanResourceArena& arena =
        state_->resource_arenas[arena_index];
    if (
        arena.poisoned || arena.device_index != device_index ||
        !context->graph_program_submission_complete(arena.submission)) {
      continue;
    }
    const bool exclusively_owned = std::all_of(
        arena.tensors.begin(), arena.tensors.end(), [](const Tensor& tensor) {
          return tensor.defined() && tensor.use_count() == 1u &&
              convert(tensor).owns_unique_storage();
        });
    if (!exclusively_owned) {
      continue;
    }
    state_->resource_arena_reuse_count.fetch_add(
        1u, std::memory_order_relaxed);
    return static_cast<int64_t>(arena_index);
  }
  if (
      state_->resource_arenas.size() >=
      static_cast<size_t>(state_->resource_arena_flight_depth)) {
    state_->resource_arena_spill_count.fetch_add(
        1u, std::memory_order_relaxed);
    return -1;
  }

  api::set_current_device(device_index);
  VulkanGraphPlanResourceArena arena;
  arena.device_index = device_index;
  arena.tensors.reserve(state_->resource_slots.size());
  for (const VulkanGraphPlanResourceSlot& slot : state_->resource_slots) {
    arena.tensors.push_back(create_vulkan_execution_tensor(
        slot.sizes,
        slot.dtype,
        slot.execution_layout,
        slot.memory_layout,
        slot.storage_type,
        /*persistent=*/true));
  }
  if (!state_->recorded_partition_ranges.empty()) {
    arena.recording = std::make_unique<VulkanGraphPlanRecordingArena>();
    arena.recording->command_pool =
        context->create_graph_program_command_pool();
    arena.recording->descriptor_pool =
        context->create_graph_program_descriptor_pool();
    arena.recording->partitions.reserve(
        state_->recorded_partition_ranges.size());
    for (const auto& range : state_->recorded_partition_ranges) {
      VulkanGraphPlanRecordedPartition partition;
      partition.start = range.first;
      partition.end = range.second;
      partition.output_sizes.resize(
          static_cast<size_t>(range.second - range.first));
      arena.recording->partitions.push_back(std::move(partition));
    }
  }
  state_->resource_arenas.push_back(std::move(arena));
  state_->resource_arena_generation_count.fetch_add(
      1u, std::memory_order_relaxed);
  state_->resource_arena_capture_count.fetch_add(
      1u, std::memory_order_relaxed);
  return static_cast<int64_t>(state_->resource_arenas.size() - 1u);
}

Tensor& VulkanGraphPlan::resource_tensor(
    const int64_t arena_index,
    const int64_t resource_slot_id) {
  TORCH_CHECK(
      state_ && arena_index >= 0 && resource_slot_id >= 0 &&
          arena_index < static_cast<int64_t>(state_->resource_arenas.size()) &&
          resource_slot_id <
              static_cast<int64_t>(state_->resource_slots.size()),
      "VulkanGraphPlan.v9 resource slot is out of range");
  return state_->resource_arenas[static_cast<size_t>(arena_index)]
      .tensors[static_cast<size_t>(resource_slot_id)];
}

void VulkanGraphPlan::record_resource_arena_submission(
    const int64_t arena_index,
    const api::VulkanSubmission& submission) {
  if (arena_index < 0) {
    return;
  }
  TORCH_CHECK(
      state_ && arena_index < static_cast<int64_t>(state_->resource_arenas.size()),
      "VulkanGraphPlan.v9 resource arena is out of range");
  state_->resource_arenas[static_cast<size_t>(arena_index)].submission =
      submission;
}

void VulkanGraphPlan::poison_resource_arena(const int64_t arena_index) noexcept {
  if (
      state_ && arena_index >= 0 &&
      arena_index < static_cast<int64_t>(state_->resource_arenas.size())) {
    state_->resource_arenas[static_cast<size_t>(arena_index)].poisoned = true;
  }
}

VulkanGraphPlan::State& VulkanGraphPlan::state() {
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
    std::vector<int64_t> output_value_ids,
    const int64_t planning_model_domain,
    const int64_t planning_execution_phase,
    const bool planning_prefer_packed_layout_propagation,
    std::optional<std::vector<int64_t>>
        planning_fixed_shape_graph_input_sizes,
    std::vector<int64_t> value_resource_slot_ids,
    std::vector<int64_t> resource_slot_sizes,
    std::vector<int64_t> resource_slot_ranks,
    const int64_t resource_arena_flight_depth,
    std::vector<int64_t> resource_slot_storage_types,
    std::vector<int64_t> resource_slot_memory_layouts,
    std::vector<int64_t> resource_slot_execution_layouts,
    std::vector<int64_t> instruction_scratch_resource_slot_ids,
    std::vector<int64_t> recorded_partition_ranges) {
  TORCH_CHECK(
      input_count > 0,
      "VulkanGraphPlan.v9 requires at least one tensor input");
  const size_t instruction_count = node_names.size();
  TORCH_CHECK(
      instruction_count > 0 && operator_names.size() == instruction_count &&
          overload_names.size() == instruction_count &&
          argument_refs.size() == instruction_count &&
          argument_kinds.size() == instruction_count &&
          instruction_output_value_ids.size() == instruction_count &&
          (instruction_scratch_resource_slot_ids.empty() ||
           instruction_scratch_resource_slot_ids.size() ==
               instruction_count * 3u),
      "VulkanGraphPlan.v9 requires aligned non-empty instruction fields");
  if (instruction_scratch_resource_slot_ids.empty()) {
    instruction_scratch_resource_slot_ids.resize(instruction_count * 3u, -1);
  }
  TORCH_CHECK(
      recorded_partition_ranges.size() % 2u == 0u,
      "VulkanGraphPlan.v9 recorded partition ranges must be start/end pairs");
  int64_t prior_partition_end = 0;
  for (size_t index = 0; index < recorded_partition_ranges.size(); index += 2u) {
    const int64_t start = recorded_partition_ranges[index];
    const int64_t end = recorded_partition_ranges[index + 1u];
    TORCH_CHECK(
        start >= prior_partition_end && start < end &&
            end <= static_cast<int64_t>(instruction_count),
        "VulkanGraphPlan.v9 recorded partition ranges must be ordered, "
        "non-overlapping, and inside the instruction sequence");
    prior_partition_end = end;
  }
  TORCH_CHECK(
      !output_value_ids.empty(),
      "VulkanGraphPlan.v9 requires at least one output value");
  TORCH_CHECK(
      planning_model_domain >=
              static_cast<int64_t>(VulkanModelDomain::Generic) &&
          planning_model_domain <= static_cast<int64_t>(VulkanModelDomain::LLM),
      "VulkanGraphPlan.v9 has an invalid planning model domain");
  TORCH_CHECK(
      planning_execution_phase >=
              static_cast<int64_t>(VulkanExecutionPhase::None) &&
          planning_execution_phase <=
              static_cast<int64_t>(VulkanExecutionPhase::Decoder),
      "VulkanGraphPlan.v9 has an invalid planning execution phase");
  const auto model_domain =
      static_cast<VulkanModelDomain>(planning_model_domain);
  const auto execution_phase =
      static_cast<VulkanExecutionPhase>(planning_execution_phase);
  TORCH_CHECK(
      is_valid_vulkan_planning_context(model_domain, execution_phase),
      "VulkanGraphPlan.v9 has incompatible planning semantics");
  TORCH_CHECK(
      !planning_fixed_shape_graph_input_sizes.has_value() ||
          (!planning_fixed_shape_graph_input_sizes->empty() &&
           std::all_of(
               planning_fixed_shape_graph_input_sizes->begin(),
               planning_fixed_shape_graph_input_sizes->end(),
               [](const int64_t size) { return size > 0; })),
      "VulkanGraphPlan.v9 fixed graph input sizes must be positive");

  int64_t next_value_id = input_count;
  for (const auto& instruction_output_ids : instruction_output_value_ids) {
    for (const int64_t output_value_id : instruction_output_ids) {
      TORCH_CHECK(
          output_value_id == next_value_id,
          "VulkanGraphPlan.v9 instruction output IDs must follow IValue SSA order");
      ++next_value_id;
    }
  }

  if (value_resource_slot_ids.empty()) {
    value_resource_slot_ids.resize(static_cast<size_t>(next_value_id), -1);
  }
  TORCH_CHECK(
      value_resource_slot_ids.size() == static_cast<size_t>(next_value_id),
      "VulkanGraphPlan.v9 resource value map must cover every SSA value");
  TORCH_CHECK(
      resource_slot_ranks.empty() ||
          (resource_arena_flight_depth >= 1 &&
           resource_arena_flight_depth <= 4),
      "VulkanGraphPlan.v9 resource arena flight depth must be in [1, 4]");
  size_t flattened_rank_sum = 0u;
  for (const int64_t rank : resource_slot_ranks) {
    TORCH_CHECK(
        rank > 0 &&
            static_cast<uint64_t>(rank) <=
                resource_slot_sizes.size() - flattened_rank_sum,
        "VulkanGraphPlan.v9 resource slot ranks must partition the flat sizes");
    flattened_rank_sum += static_cast<size_t>(rank);
  }
  TORCH_CHECK(
      flattened_rank_sum == resource_slot_sizes.size(),
      "VulkanGraphPlan.v9 resource slot ranks must partition the flat sizes");

  const size_t resource_slot_count = resource_slot_ranks.size();
  if (resource_slot_storage_types.empty()) {
    resource_slot_storage_types.assign(
        resource_slot_count, static_cast<int64_t>(api::StorageType::BUFFER));
  }
  if (resource_slot_memory_layouts.empty()) {
    resource_slot_memory_layouts.assign(
        resource_slot_count,
        static_cast<int64_t>(api::GPUMemoryLayout::TENSOR_WIDTH_PACKED));
  }
  if (resource_slot_execution_layouts.empty()) {
    resource_slot_execution_layouts.assign(
        resource_slot_count,
        static_cast<int64_t>(api::ExecutionLayout::BUFFER_DIRECT));
  }
  TORCH_CHECK(
      resource_slot_storage_types.size() == resource_slot_count &&
          resource_slot_memory_layouts.size() == resource_slot_count &&
          resource_slot_execution_layouts.size() == resource_slot_count,
      "VulkanGraphPlan.v9 resource layout descriptors must match slot count");

  auto state = std::make_shared<VulkanGraphPlan::State>();
  state->input_count = input_count;
  state->planning_request = make_vulkan_planning_request(
      VulkanWorkloadClass::Generic,
      VulkanTensorRole::Input,
      model_domain,
      execution_phase);
  state->planning_request.prefer_packed_layout_propagation =
      planning_prefer_packed_layout_propagation;
  state->planning_request.fixed_shape_graph_input_sizes =
      std::move(planning_fixed_shape_graph_input_sizes);
  state->constants.assign(constants.begin(), constants.end());
  state->output_value_ids = std::move(output_value_ids);
  state->values.resize(static_cast<size_t>(next_value_id));
  state->resource_arena_flight_depth =
      resource_slot_ranks.empty() ? 0 : resource_arena_flight_depth;
  state->resource_slots.reserve(resource_slot_ranks.size());
  size_t resource_size_offset = 0u;
  for (const auto slot_index : c10::irange(resource_slot_ranks.size())) {
    const int64_t rank = resource_slot_ranks[slot_index];
    std::vector<int64_t> slot_sizes(
        resource_slot_sizes.begin() + resource_size_offset,
        resource_slot_sizes.begin() + resource_size_offset + rank);
    TORCH_CHECK(
        std::all_of(
            slot_sizes.begin(),
            slot_sizes.end(),
            [](const int64_t size) { return size > 0; }),
        "VulkanGraphPlan.v9 resource slots require positive fp32 buffer shapes");
    const api::StorageType storage_type =
        parse_resource_storage_type(resource_slot_storage_types[slot_index]);
    const api::GPUMemoryLayout memory_layout =
        parse_resource_memory_layout(resource_slot_memory_layouts[slot_index]);
    const api::ExecutionLayout execution_layout =
        parse_resource_execution_layout(
            resource_slot_execution_layouts[slot_index]);
    TORCH_CHECK(
        execution_layout != api::ExecutionLayout::BUFFER_VIEW,
        "VulkanGraphPlan.v9 resource slots cannot materialize buffer views");
    TORCH_CHECK(
        api::uses_buffer_execution(execution_layout) ==
            (storage_type == api::StorageType::BUFFER),
        "VulkanGraphPlan.v9 resource storage and execution layouts disagree");
    state->resource_slots.push_back(
        VulkanGraphPlanResourceSlot{
            std::move(slot_sizes),
            kFloat,
            storage_type,
            memory_layout,
            execution_layout});
    resource_size_offset += static_cast<size_t>(rank);
  }
  for (const auto value_index : c10::irange(state->values.size())) {
    const int64_t slot_id = value_resource_slot_ids[value_index];
    TORCH_CHECK(
        slot_id >= -1 &&
            slot_id < static_cast<int64_t>(state->resource_slots.size()) &&
            (value_index >= static_cast<size_t>(input_count) || slot_id == -1),
        "VulkanGraphPlan.v9 has an invalid resource value map");
    state->values[value_index].resource_slot_id = slot_id;
  }
  state->instructions.reserve(instruction_count);
  int64_t defined_value_count = input_count;
  for (const auto instruction_index : c10::irange(instruction_count)) {
    TORCH_CHECK(
        !node_names[instruction_index].empty() &&
            !operator_names[instruction_index].empty(),
        "VulkanGraphPlan.v9 instruction names must be non-empty");
    const auto internal_instruction_kind =
        graph_instruction_kind(operator_names[instruction_index]);
    const VulkanGraphPlanInstructionKind instruction_kind =
        internal_instruction_kind.value_or(
            VulkanGraphPlanInstructionKind::Dispatcher);
    std::optional<c10::OperatorHandle> operator_handle;
    std::optional<c10::OperatorHandle> dead_input_reuse_operator_handle;
    VulkanGraphPlanResourceWriterKind resource_writer_kind =
        VulkanGraphPlanResourceWriterKind::None;
    const c10::FunctionSchema* schema = nullptr;
    if (internal_instruction_kind) {
      TORCH_CHECK(
          overload_names[instruction_index].empty(),
          "VulkanGraphPlan.v9 internal instruction '",
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
          "VulkanGraphPlan.v9 rejects mutable operator ",
          schema->operator_name());
      TORCH_CHECK(
          has_plan_dispatch(*operator_handle),
          "VulkanGraphPlan.v9 requires a Vulkan or composite kernel for ",
          schema->operator_name());
      if (
          operator_names[instruction_index] == "aten::relu" &&
          overload_names[instruction_index].empty()) {
        dead_input_reuse_operator_handle.emplace(
            c10::Dispatcher::singleton().findSchemaOrThrow("aten::relu_", ""));
        TORCH_CHECK(
            has_plan_dispatch(*dead_input_reuse_operator_handle),
            "VulkanGraphPlan.v9 requires a Vulkan in-place reuse kernel for ",
            schema->operator_name());
      }
    }
    std::vector<int64_t>& output_value_ids_for_instruction =
        instruction_output_value_ids[instruction_index];
    if (schema) {
      TORCH_CHECK(
          schema->returns().size() ==
              output_value_ids_for_instruction.size(),
          "VulkanGraphPlan.v9 output schema does not match ",
          schema->operator_name());
    } else {
      TORCH_CHECK(
          output_value_ids_for_instruction.size() == 1u,
          "VulkanGraphPlan.v9 internal instruction '",
          node_names[instruction_index],
          "' must define one value");
    }
    const bool any_resource_output = std::any_of(
        output_value_ids_for_instruction.begin(),
        output_value_ids_for_instruction.end(),
        [&state](const int64_t value_id) {
          return state->values[static_cast<size_t>(value_id)].resource_slot_id >=
              0;
        });
    const bool all_resource_outputs = !output_value_ids_for_instruction.empty() &&
        std::all_of(
            output_value_ids_for_instruction.begin(),
            output_value_ids_for_instruction.end(),
            [&state](const int64_t value_id) {
              return state->values[static_cast<size_t>(value_id)]
                         .resource_slot_id >= 0;
            });
    if (any_resource_output) {
      resource_writer_kind = graph_resource_writer_kind(
          operator_names[instruction_index]);
      TORCH_CHECK(
          all_resource_outputs &&
              resource_writer_kind != VulkanGraphPlanResourceWriterKind::None,
          "VulkanGraphPlan.v9 resource outputs require a supported complete writer");
    }
    std::vector<int64_t> scratch_resource_slot_ids;
    scratch_resource_slot_ids.reserve(3u);
    const size_t scratch_offset = instruction_index * 3u;
    for (const auto scratch_index : c10::irange(3u)) {
      const int64_t slot_id =
          instruction_scratch_resource_slot_ids[scratch_offset + scratch_index];
      TORCH_CHECK(
          slot_id >= -1,
          "VulkanGraphPlan.v9 instruction scratch slot must be -1 or a "
          "resource-slot index");
      if (slot_id >= 0) {
        scratch_resource_slot_ids.push_back(slot_id);
      }
    }
    TORCH_CHECK(
        scratch_resource_slot_ids.empty() ||
            (resource_writer_kind ==
                 VulkanGraphPlanResourceWriterKind::AttentionMath &&
             scratch_resource_slot_ids.size() == 3u) ||
            ((resource_writer_kind ==
                  VulkanGraphPlanResourceWriterKind::ScaledAdd ||
              resource_writer_kind ==
                  VulkanGraphPlanResourceWriterKind::ScaledAddLayernormPlan) &&
             scratch_resource_slot_ids.size() == 1u),
        "VulkanGraphPlan.v9 instruction scratch slots require the "
        "attention-math or scaled-add resource writer");
    for (const int64_t slot_id : scratch_resource_slot_ids) {
      TORCH_CHECK(
          slot_id >= 0 &&
              slot_id < static_cast<int64_t>(state->resource_slots.size()),
          "VulkanGraphPlan.v9 instruction scratch slot is out of range");
      TORCH_CHECK(
          std::count(
              scratch_resource_slot_ids.begin(),
              scratch_resource_slot_ids.end(),
              slot_id) == 1,
          "VulkanGraphPlan.v9 instruction scratch slots must be distinct");
      for (const int64_t output_value_id : output_value_ids_for_instruction) {
        TORCH_CHECK(
            state->values[static_cast<size_t>(output_value_id)]
                    .resource_slot_id != slot_id,
            "VulkanGraphPlan.v9 instruction scratch and output slots overlap");
      }
    }
    const size_t expected_argument_count =
        schema ? schema->arguments().size() : 2u;
    TORCH_CHECK(
        argument_refs[instruction_index].size() == expected_argument_count &&
            argument_kinds[instruction_index].size() == expected_argument_count,
        "VulkanGraphPlan.v9 argument count does not match ",
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
          "VulkanGraphPlan.v9 instruction '",
          node_names[instruction_index],
          "' has an invalid argument kind");
      const auto kind = static_cast<VulkanGraphPlanArgumentKind>(kind_value);
      std::vector<int64_t>& refs =
          argument_refs[instruction_index][argument_index];
      TORCH_CHECK(
          kind == VulkanGraphPlanArgumentKind::List || refs.size() == 1u,
          "VulkanGraphPlan.v9 instruction '",
          node_names[instruction_index],
          "' has an invalid argument recipe");

      c10::TypePtr list_element_type;
      if (kind == VulkanGraphPlanArgumentKind::List) {
        TORCH_CHECK(
            schema,
            "VulkanGraphPlan.v9 internal instruction '",
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
            "VulkanGraphPlan.v9 instruction '",
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
              "VulkanGraphPlan.v9 instruction '",
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
              "VulkanGraphPlan.v9 instruction '",
              node_names[instruction_index],
              "' has an invalid constant reference");
        }
      }
      arguments.push_back(VulkanGraphPlanArgument{
          kind, std::move(refs), std::move(list_element_type)});
    }
    for (const int64_t output_value_id : output_value_ids_for_instruction) {
      VulkanGraphPlanValue& output_value =
          state->values[static_cast<size_t>(output_value_id)];
      output_value.definition = static_cast<int64_t>(instruction_index);
      output_value.last_use = static_cast<int64_t>(instruction_index);
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
        std::move(dead_input_reuse_operator_handle),
        resource_writer_kind,
        schema == nullptr || !schema_has_list_return(*schema),
        std::move(arguments),
        std::move(output_value_ids_for_instruction),
        std::move(scratch_resource_slot_ids),
        {}});
    defined_value_count += static_cast<int64_t>(
        state->instructions.back().output_value_ids.size());
  }

  for (size_t index = 0; index < recorded_partition_ranges.size(); index += 2u) {
    const int64_t start = recorded_partition_ranges[index];
    const int64_t end = recorded_partition_ranges[index + 1u];
    for (int64_t instruction_index = start; instruction_index < end;
         ++instruction_index) {
      const VulkanGraphPlanInstruction& instruction =
          state->instructions[static_cast<size_t>(instruction_index)];
      const bool host_recipe =
          instruction.operator_name == "aten::permute" ||
          instruction.operator_name == "aten::reshape" ||
          instruction.operator_name == "aten::select.int" ||
          instruction.operator_name == "aten::transpose.int";
      TORCH_CHECK(
          instruction.resource_writer_kind !=
                  VulkanGraphPlanResourceWriterKind::None ||
              host_recipe,
          "VulkanGraphPlan.v9 recorded partition contains unsupported node '",
          instruction.node_name,
          "' (",
          instruction.operator_name,
          ")");
    }
    state->recorded_partition_ranges.emplace_back(start, end);
  }

  for (const int64_t output_value_id : state->output_value_ids) {
    TORCH_CHECK(
        output_value_id >= 0 &&
            output_value_id < static_cast<int64_t>(state->values.size()),
        "VulkanGraphPlan.v9 output value is out of range");
    VulkanGraphPlanValue& output_value =
        state->values[static_cast<size_t>(output_value_id)];
    TORCH_CHECK(
        output_value.resource_slot_id < 0,
        "VulkanGraphPlan.v9 escaping outputs cannot use an internal resource slot");
    output_value.escapes = true;
  }
  std::vector<std::vector<size_t>> resource_slot_values(
      state->resource_slots.size());
  for (const auto value_index : c10::irange(state->values.size())) {
    const VulkanGraphPlanValue& value = state->values[value_index];
    if (value.resource_slot_id < 0) {
      continue;
    }
    TORCH_CHECK(
        value.definition >= 0 && value.last_use >= value.definition &&
            !value.escapes,
        "VulkanGraphPlan.v9 resource slots require non-escaping defined values");
    auto& assigned =
        resource_slot_values[static_cast<size_t>(value.resource_slot_id)];
    for (const size_t prior_index : assigned) {
      const VulkanGraphPlanValue& prior = state->values[prior_index];
      TORCH_CHECK(
          prior.last_use < value.definition || value.last_use < prior.definition,
          "VulkanGraphPlan.v9 resource slot lifetimes overlap");
    }
    assigned.push_back(value_index);
  }
  for (const auto instruction_index : c10::irange(state->instructions.size())) {
    const VulkanGraphPlanInstruction& instruction =
        state->instructions[instruction_index];
    for (const int64_t slot_id : instruction.scratch_resource_slot_ids) {
      for (const size_t value_index :
           resource_slot_values[static_cast<size_t>(slot_id)]) {
        const VulkanGraphPlanValue& value = state->values[value_index];
        TORCH_CHECK(
            value.last_use < static_cast<int64_t>(instruction_index) ||
                value.definition > static_cast<int64_t>(instruction_index),
            "VulkanGraphPlan.v9 scratch and value resource-slot lifetimes "
            "overlap");
      }
    }
  }
  for (const auto value_index : c10::irange(state->values.size())) {
    const VulkanGraphPlanValue& value = state->values[value_index];
    if (value.escapes || value.last_use < 0) {
      continue;
    }
    TORCH_INTERNAL_ASSERT(
        value.last_use < static_cast<int64_t>(state->instructions.size()));
    state->instructions[static_cast<size_t>(value.last_use)]
        .release_value_ids.push_back(static_cast<int64_t>(value_index));
  }
  VulkanGraphPlanInvocationWorkspace& workspace = state->invocation_workspace;
  workspace.values.resize(state->values.size());
  workspace.value_live.resize(state->values.size(), uint8_t{0u});
  workspace.list_arguments.reserve(state->instructions.size());
  size_t maximum_argument_count = 0u;
  for (const VulkanGraphPlanInstruction& instruction : state->instructions) {
    maximum_argument_count =
        std::max(maximum_argument_count, instruction.arguments.size());
    auto& instruction_lists = workspace.list_arguments.emplace_back();
    instruction_lists.reserve(instruction.arguments.size());
    for (const VulkanGraphPlanArgument& argument : instruction.arguments) {
      if (
          argument.kind == VulkanGraphPlanArgumentKind::List &&
          instruction.reusable_list_arguments) {
        instruction_lists.emplace_back(
            c10::impl::GenericList(argument.list_element_type));
        instruction_lists.back()->reserve(argument.refs.size());
      } else {
        instruction_lists.emplace_back(std::nullopt);
      }
    }
  }
  workspace.stack.reserve(maximum_argument_count);
  return c10::make_intrusive<VulkanGraphPlan>(std::move(state));
}

std::vector<Tensor> run_vulkan_graph_plan(
    const std::vector<Tensor>& inputs,
    const c10::intrusive_ptr<VulkanGraphPlan>& plan) {
  TORCH_CHECK(plan, "VulkanGraphPlan.v9 requires a plan");
  VulkanGraphPlan::State& state = plan->state();
  TORCH_CHECK(
      inputs.size() == static_cast<size_t>(state.input_count),
      "VulkanGraphPlan.v9 input count mismatch");
  TORCH_CHECK(
      std::all_of(inputs.begin(), inputs.end(), [](const Tensor& input) {
        return input.is_vulkan();
      }),
      "VulkanGraphPlan.v9 requires Vulkan tensor inputs");
  const c10::DeviceIndex device_index = inputs.front().device().index();
  TORCH_CHECK(
      std::all_of(
          inputs.begin(),
          inputs.end(),
          [device_index](const Tensor& input) {
            return input.device().index() == device_index;
          }),
      "VulkanGraphPlan.v9 requires inputs on one Vulkan device");
  VulkanGraphPlanInvocation invocation(*plan);
  VulkanPlanningRequestScope planning_scope(state.planning_request);
  const int64_t resource_arena_index =
      plan->acquire_resource_arena(device_index);
  std::optional<api::Context::GraphProgramInvocationScope> submission_scope;
  if (state.submission_owned) {
    submission_scope.emplace(*api::context(device_index));
  }
  bool resource_arena_finalized = false;
  auto finalize_resource_arena = c10::make_scope_exit(
      [&plan,
       &submission_scope,
       resource_arena_index,
       &resource_arena_finalized]() {
        if (resource_arena_index < 0 || resource_arena_finalized) {
          return;
        }
        try {
          if (submission_scope && submission_scope->active()) {
            submission_scope->abort();
          }
          if (submission_scope) {
            plan->record_resource_arena_submission(
                resource_arena_index, submission_scope->submission());
          }
        } catch (...) {
          plan->poison_resource_arena(resource_arena_index);
        }
      });

  VulkanGraphPlanInvocationWorkspace& workspace = state.invocation_workspace;
  auto reset_workspace = c10::make_scope_exit([&workspace]() {
    workspace.reset();
  });
  std::vector<c10::IValue>& values = workspace.values;
  std::vector<uint8_t>& value_live = workspace.value_live;
  std::vector<c10::IValue>& stack = workspace.stack;
  for (const auto input_index : c10::irange(inputs.size())) {
    values[input_index] = inputs[input_index];
    value_live[input_index] = 1u;
  }
  VulkanGraphPlanResourceArena* resource_arena = resource_arena_index >= 0
      ? &state.resource_arenas[static_cast<size_t>(resource_arena_index)]
      : nullptr;
  api::Context* const context = api::context(device_index);
  size_t recorded_partition_index = 0u;
  VulkanGraphPlanRecordedPartition* active_partition = nullptr;
  VulkanGraphPlanRecordedPartitionMode active_partition_mode =
      VulkanGraphPlanRecordedPartitionMode::None;
  std::unique_ptr<api::Context::ScopedExternalCommandRecording>
      partition_recording_scope;
  api::PipelineBarrier partition_entry_barrier{};
  bool partition_capture_active = false;
  auto fail_partial_partition_capture = c10::make_scope_exit(
      [&plan,
       &state,
       &active_partition,
       &partition_recording_scope,
       &partition_capture_active,
       resource_arena_index,
       context]() {
        if (!partition_capture_active || !active_partition) {
          return;
        }
        partition_recording_scope.reset();
        try {
          context->take_external_recording_cleanup_resources(
              active_partition->retained_buffers,
              active_partition->retained_images);
          (void)context->take_external_recording_dispatch_count();
        } catch (...) {
        }
        active_partition->state =
            VulkanGraphPlanRecordedPartitionState::Failed;
        state.recorded_partition_failure_count.fetch_add(
            1u, std::memory_order_relaxed);
        plan->poison_resource_arena(resource_arena_index);
      });
  for (const auto instruction_index : c10::irange(state.instructions.size())) {
    const VulkanGraphPlanInstruction& instruction =
        state.instructions[instruction_index];
    if (
        resource_arena && resource_arena->recording &&
        recorded_partition_index <
            resource_arena->recording->partitions.size() &&
        instruction_index == static_cast<size_t>(
                                 resource_arena->recording
                                     ->partitions[recorded_partition_index]
                                     .start)) {
      active_partition = &resource_arena->recording
                              ->partitions[recorded_partition_index];
      TORCH_CHECK(
          active_partition->state !=
              VulkanGraphPlanRecordedPartitionState::Failed,
          "VulkanGraphPlan.v9 recorded partition is poisoned");
      if (
          active_partition->state ==
          VulkanGraphPlanRecordedPartitionState::Empty) {
        active_partition_mode =
            VulkanGraphPlanRecordedPartitionMode::Prime;
      } else if (
          active_partition->state ==
          VulkanGraphPlanRecordedPartitionState::Primed) {
        active_partition_mode =
            VulkanGraphPlanRecordedPartitionMode::Capture;
        partition_entry_barrier =
            prepare_recorded_partition_entry(*resource_arena);
        active_partition->retained_buffers.clear();
        active_partition->retained_images.clear();
        for (auto& output_sizes : active_partition->output_sizes) {
          output_sizes.clear();
        }
        active_partition->command.emplace(
            resource_arena->recording->command_pool->get_new_cmd(
                /*reusable=*/true, VK_COMMAND_BUFFER_LEVEL_SECONDARY));
        active_partition->command->begin();
        partition_recording_scope = std::make_unique<
            api::Context::ScopedExternalCommandRecording>(
            *context,
            *active_partition->command,
            *resource_arena->recording->descriptor_pool);
        partition_capture_active = true;
      } else {
        active_partition_mode =
            VulkanGraphPlanRecordedPartitionMode::Replay;
        if (!recorded_arena_stamps_match(
                *resource_arena, active_partition->tensor_stamps)) {
          active_partition->state =
              VulkanGraphPlanRecordedPartitionState::Failed;
          state.recorded_partition_failure_count.fetch_add(
              1u, std::memory_order_relaxed);
          plan->poison_resource_arena(resource_arena_index);
          TORCH_CHECK(
              false,
              "VulkanGraphPlan.v9 recorded partition resource identity "
              "changed");
        }
        partition_entry_barrier =
            prepare_recorded_partition_entry(*resource_arena);
      }
    }
    stack.clear();
    stack.reserve(instruction.arguments.size());
    std::vector<c10::impl::GenericList> transient_lists;
    const auto load_argument_ref = [&](const int64_t argument_ref) {
      if (argument_ref >= 0) {
        TORCH_CHECK(
            value_live[static_cast<size_t>(argument_ref)],
            "VulkanGraphPlan.v9 node '",
            instruction.node_name,
            "' references a released value");
        return values[static_cast<size_t>(argument_ref)];
      }
      return state.constants[static_cast<size_t>(constant_index(argument_ref))];
    };
    for (const auto argument_index :
         c10::irange(instruction.arguments.size())) {
      const VulkanGraphPlanArgument& argument =
          instruction.arguments[argument_index];
      if (argument.kind == VulkanGraphPlanArgumentKind::Value) {
        stack.push_back(load_argument_ref(argument.refs.front()));
        continue;
      }
      auto& reusable_list =
          workspace.list_arguments[instruction_index][argument_index];
      c10::impl::GenericList* list = nullptr;
      if (reusable_list) {
        TORCH_INTERNAL_ASSERT(reusable_list->empty());
        list = &*reusable_list;
      } else {
        transient_lists.emplace_back(argument.list_element_type);
        list = &transient_lists.back();
      }
      list->reserve(argument.refs.size());
      for (const int64_t argument_ref : argument.refs) {
        list->emplace_back(load_argument_ref(argument_ref));
      }
      stack.emplace_back(*list);
    }

    const bool reuse_dead_input = can_reuse_dead_input(
        instruction,
        instruction_index,
        state.values,
        values,
        value_live,
        inputs,
        state.constants);
    const int64_t scope_token = begin_vulkan_graph_execution_scope();
    try {
      VulkanGraphPlanResourceWriteResult resource_write =
          VulkanGraphPlanResourceWriteResult::NotApplicable;
      if (
          active_partition_mode ==
              VulkanGraphPlanRecordedPartitionMode::Replay &&
          active_partition &&
          instruction.resource_writer_kind !=
              VulkanGraphPlanResourceWriterKind::None) {
        const size_t partition_offset = instruction_index -
            static_cast<size_t>(active_partition->start);
        const auto& saved_output_sizes =
            active_partition->output_sizes[partition_offset];
        TORCH_CHECK(
            saved_output_sizes.size() == instruction.output_value_ids.size(),
            "VulkanGraphPlan.v9 recorded partition output recipe is invalid");
        stack.clear();
        for (const auto output_index :
             c10::irange(instruction.output_value_ids.size())) {
          const int64_t output_value_id =
              instruction.output_value_ids[output_index];
          const int64_t slot_id = state.values[static_cast<size_t>(
                                                   output_value_id)]
                                      .resource_slot_id;
          TORCH_INTERNAL_ASSERT(slot_id >= 0);
          Tensor& target =
              plan->resource_tensor(resource_arena_index, slot_id);
          stack.emplace_back(
              target.sizes().equals(saved_output_sizes[output_index])
                  ? target
                  : target.view(saved_output_sizes[output_index]));
        }
        resource_write = VulkanGraphPlanResourceWriteResult::Written;
      } else {
        resource_write = execute_resource_writer(
            *plan,
            instruction,
            state.values,
            resource_arena_index,
            stack);
      }
      if (resource_write == VulkanGraphPlanResourceWriteResult::Written) {
        state.resource_write_count.fetch_add(1u, std::memory_order_relaxed);
      } else if (
          resource_write ==
          VulkanGraphPlanResourceWriteResult::ProducedUnowned) {
        state.resource_writer_bypass_count.fetch_add(
            1u, std::memory_order_relaxed);
      } else if (
          instruction.kind == VulkanGraphPlanInstructionKind::Dispatcher) {
        if (
            resource_write ==
            VulkanGraphPlanResourceWriteResult::NeedsDispatcher) {
          state.resource_writer_bypass_count.fetch_add(
              1u, std::memory_order_relaxed);
        }
        TORCH_INTERNAL_ASSERT(instruction.operator_handle);
        c10::Dispatcher::singleton().callBoxed(
            reuse_dead_input ? *instruction.dead_input_reuse_operator_handle
                             : *instruction.operator_handle,
            &stack);
      } else if (
          instruction.kind == VulkanGraphPlanInstructionKind::ListGetItem) {
        execute_list_getitem_instruction(instruction, stack);
      } else {
        execute_graph_scalar_instruction(instruction, stack);
      }
    } catch (const c10::Error& error) {
      const VulkanGraphExecutionScopeCounts counters =
          end_vulkan_graph_execution_scope_counts(scope_token);
      check_implicit_boundary(instruction, counters);
      TORCH_CHECK(
          false,
          "VulkanGraphPlan.v9 node '",
          instruction.node_name,
          "' (",
          instruction.operator_name,
          ") failed: ",
          error.what_without_backtrace());
    } catch (const std::exception& error) {
      const VulkanGraphExecutionScopeCounts counters =
          end_vulkan_graph_execution_scope_counts(scope_token);
      check_implicit_boundary(instruction, counters);
      TORCH_CHECK(
          false,
          "VulkanGraphPlan.v9 node '",
          instruction.node_name,
          "' (",
          instruction.operator_name,
          ") failed: ",
          error.what());
    } catch (...) {
      const VulkanGraphExecutionScopeCounts counters =
          end_vulkan_graph_execution_scope_counts(scope_token);
      check_implicit_boundary(instruction, counters);
      TORCH_CHECK(
          false,
          "VulkanGraphPlan.v9 node '",
          instruction.node_name,
          "' (",
          instruction.operator_name,
          ") failed with a non-standard exception");
    }
    const VulkanGraphExecutionScopeCounts counters =
        end_vulkan_graph_execution_scope_counts(scope_token);
    check_implicit_boundary(instruction, counters);
    for (auto& list : workspace.list_arguments[instruction_index]) {
      if (list) {
        list->clear();
      }
    }
    if (reuse_dead_input) {
      state.dead_input_reuse_count.fetch_add(1u, std::memory_order_relaxed);
    }
    if (instruction.output_value_ids.empty()) {
      TORCH_CHECK(
          stack.empty(),
          "VulkanGraphPlan.v9 effect node '",
          instruction.node_name,
          "' produced an undeclared value");
    } else {
      TORCH_CHECK(
          stack.size() == instruction.output_value_ids.size(),
          "VulkanGraphPlan.v9 node '",
          instruction.node_name,
          "' did not produce its declared values");
      for (const auto output_index :
           c10::irange(instruction.output_value_ids.size())) {
        c10::IValue output = std::move(stack[output_index]);
        TORCH_CHECK(
            !output.isTensor() || output.toTensor().is_vulkan(),
            "VulkanGraphPlan.v9 node '",
            instruction.node_name,
            "' produced a non-Vulkan tensor");
        const int64_t output_value_id =
            instruction.output_value_ids[output_index];
        values[static_cast<size_t>(output_value_id)] = std::move(output);
        value_live[static_cast<size_t>(output_value_id)] = 1u;
      }
    }

    if (
        active_partition_mode ==
            VulkanGraphPlanRecordedPartitionMode::Capture &&
        active_partition) {
      const size_t partition_offset = instruction_index -
          static_cast<size_t>(active_partition->start);
      auto& saved_output_sizes =
          active_partition->output_sizes[partition_offset];
      saved_output_sizes.clear();
      saved_output_sizes.reserve(instruction.output_value_ids.size());
      for (const int64_t output_value_id : instruction.output_value_ids) {
        const c10::IValue& output =
            values[static_cast<size_t>(output_value_id)];
        TORCH_CHECK(
            output.isTensor(),
            "VulkanGraphPlan.v9 recorded partition requires tensor outputs");
        saved_output_sizes.emplace_back(output.toTensor().sizes().vec());
      }
    }

    for (const int64_t release_value_id : instruction.release_value_ids) {
      const size_t value_index = static_cast<size_t>(release_value_id);
      TORCH_INTERNAL_ASSERT(value_live[value_index]);
      values[value_index] = c10::IValue();
      value_live[value_index] = 0u;
    }
    if (
        active_partition &&
        instruction_index + 1u ==
            static_cast<size_t>(active_partition->end)) {
      if (
          active_partition_mode ==
          VulkanGraphPlanRecordedPartitionMode::Prime) {
        active_partition->state =
            VulkanGraphPlanRecordedPartitionState::Primed;
        state.recorded_partition_prime_count.fetch_add(
            1u, std::memory_order_relaxed);
      } else if (
          active_partition_mode ==
          VulkanGraphPlanRecordedPartitionMode::Capture) {
        partition_recording_scope.reset();
        context->take_external_recording_cleanup_resources(
            active_partition->retained_buffers,
            active_partition->retained_images);
        active_partition->represented_dispatch_count =
            context->take_external_recording_dispatch_count();
        TORCH_CHECK(
            active_partition->represented_dispatch_count > 0u,
            "VulkanGraphPlan.v9 recorded partition captured no dispatches");
        TORCH_INTERNAL_ASSERT(active_partition->command);
        active_partition->command->end();
        active_partition->tensor_stamps =
            recorded_arena_stamps(*resource_arena);
        context->execute_secondary_command_buffer(
            partition_entry_barrier,
            *active_partition->command,
            active_partition->represented_dispatch_count);
        record_partition_exit_state(*resource_arena);
        active_partition->state =
            VulkanGraphPlanRecordedPartitionState::Ready;
        partition_capture_active = false;
        state.recorded_partition_capture_count.fetch_add(
            1u, std::memory_order_relaxed);
        state.recorded_partition_represented_dispatch_count.fetch_add(
            active_partition->represented_dispatch_count,
            std::memory_order_relaxed);
      } else if (
          active_partition_mode ==
          VulkanGraphPlanRecordedPartitionMode::Replay) {
        TORCH_INTERNAL_ASSERT(active_partition->command);
        context->execute_secondary_command_buffer(
            partition_entry_barrier,
            *active_partition->command,
            active_partition->represented_dispatch_count);
        record_partition_exit_state(*resource_arena);
        state.recorded_partition_replay_count.fetch_add(
            1u, std::memory_order_relaxed);
        state.recorded_partition_represented_dispatch_count.fetch_add(
            active_partition->represented_dispatch_count,
            std::memory_order_relaxed);
      }
      active_partition = nullptr;
      active_partition_mode =
          VulkanGraphPlanRecordedPartitionMode::None;
      ++recorded_partition_index;
    }
    if (submission_scope && submission_scope->checkpoint_requested()) {
      submission_scope->checkpoint();
    }
  }

  std::vector<Tensor> outputs;
  outputs.reserve(state.output_value_ids.size());
  for (const int64_t output_value_id : state.output_value_ids) {
    c10::IValue& output = values[static_cast<size_t>(output_value_id)];
    TORCH_CHECK(
        value_live[static_cast<size_t>(output_value_id)] && output.isTensor(),
        "VulkanGraphPlan.v9 output references a released or non-Tensor value");
    outputs.push_back(output.toTensor());
  }
  if (submission_scope) {
    const api::VulkanSubmission submission = submission_scope->submit();
    plan->record_resource_arena_submission(resource_arena_index, submission);
    plan->record_submission(device_index, submission);
  }
  resource_arena_finalized = true;
  return outputs;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif // USE_VULKAN_API
