from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch.utils import _pytree as pytree


@dataclass(frozen=True)
class VulkanGraphPlanReport:
    status: str
    reason: str
    plan_class: str
    plan_version: str
    input_count: int
    instruction_count: int
    value_count: int
    output_count: int
    node_names: tuple[str, ...]
    value_use_counts: tuple[int, ...]
    value_last_uses: tuple[int, ...]


@dataclass(frozen=True)
class _VulkanGraphPlanCompilation:
    plan: Any | None
    report: VulkanGraphPlanReport


def _rejected(reason: str) -> _VulkanGraphPlanCompilation:
    return _VulkanGraphPlanCompilation(
        plan=None,
        report=VulkanGraphPlanReport(
            status="python_correctness_executor",
            reason=reason,
            plan_class="VulkanGraphPlan",
            plan_version="v1",
            input_count=0,
            instruction_count=0,
            value_count=0,
            output_count=0,
            node_names=(),
            value_use_counts=(),
            value_last_uses=(),
        ),
    )


def _fetch_attr(module: torch.nn.Module, target: str) -> Any:
    value: Any = module
    for atom in target.split("."):
        value = getattr(value, atom)
    return value


def _contains_node(value: Any) -> bool:
    if isinstance(value, torch.fx.Node):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_node(item) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_node(key) or _contains_node(item)
            for key, item in value.items()
        )
    if isinstance(value, slice):
        return any(
            _contains_node(item)
            for item in (value.start, value.stop, value.step)
        )
    return False


def _bound_operator_arguments(node: torch.fx.Node) -> tuple[Any, ...] | str:
    schema = node.target._schema
    positional = iter(node.args)
    kwargs = dict(node.kwargs)
    values: list[Any] = []
    for argument in schema.arguments:
        if not argument.kwarg_only:
            try:
                values.append(next(positional))
                continue
            except StopIteration:
                pass
        if argument.name in kwargs:
            values.append(kwargs.pop(argument.name))
        elif argument.has_default_value():
            values.append(argument.default_value)
        else:
            return f"missing_argument:{node.name}:{argument.name}"
    try:
        next(positional)
    except StopIteration:
        pass
    else:
        return f"too_many_positional_arguments:{node.name}"
    if kwargs:
        return f"unknown_keyword_arguments:{node.name}:{','.join(sorted(kwargs))}"
    return tuple(values)


def _argument_type_matches(
    graph_module: torch.fx.GraphModule,
    value: Any,
    expected_type: Any,
) -> bool:
    if isinstance(value, torch.fx.Node):
        if value.op == "get_attr":
            value = _fetch_attr(graph_module, str(value.target))
        else:
            return torch._C.TensorType.get().isSubtypeOf(expected_type)
    inferred_type = torch._C._jit_try_infer_type(value)
    return inferred_type.success() and inferred_type.type().isSubtypeOf(
        expected_type
    )


def compile_vulkan_graph_plan(
    graph_module: torch.fx.GraphModule,
    classifications: Mapping[str, str],
) -> _VulkanGraphPlanCompilation:
    placeholders = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    if not placeholders:
        return _rejected("v1_requires_tensor_inputs")
    for node in placeholders:
        if not isinstance(node.meta.get("val"), torch.Tensor):
            return _rejected(f"non_tensor_input:{node.name}")

    value_ids = {node: index for index, node in enumerate(placeholders)}
    constants: list[Any] = []
    constant_ids: dict[torch.fx.Node, int] = {}
    node_names: list[str] = []
    operator_names: list[str] = []
    overload_names: list[str] = []
    argument_refs: list[list[int]] = []

    def encode_argument(value: Any, consumer: torch.fx.Node) -> int | str:
        if isinstance(value, torch.fx.Node):
            if value in value_ids:
                return value_ids[value]
            if value.op != "get_attr":
                return f"unrepresented_value:{consumer.name}:{value.name}"
            if value not in constant_ids:
                constant_ids[value] = len(constants)
                constants.append(_fetch_attr(graph_module, str(value.target)))
            return -constant_ids[value] - 1
        if _contains_node(value):
            return f"nested_dynamic_argument:{consumer.name}"
        constants.append(value)
        return -len(constants)

    for node in graph_module.graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            return _rejected(f"unsupported_node_kind:{node.name}:{node.op}")
        if classifications.get(node.name) not in (
            "direct_vulkan",
            "lowered_vulkan",
            "composite",
        ):
            return _rejected(
                f"node_not_vulkan_admitted:{node.name}:"
                f"{classifications.get(node.name, 'unknown')}"
            )
        schema = node.target._schema
        if schema.is_mutable:
            return _rejected(f"mutable_operator:{node.name}:{schema.name}")
        if len(schema.returns) != 1 or str(schema.returns[0].type) != "Tensor":
            return _rejected(f"non_tensor_return:{node.name}:{schema.name}")
        bound_arguments = _bound_operator_arguments(node)
        if isinstance(bound_arguments, str):
            return _rejected(bound_arguments)
        refs: list[int] = []
        for schema_argument, argument in zip(schema.arguments, bound_arguments):
            if not _argument_type_matches(graph_module, argument, schema_argument.type):
                return _rejected(
                    f"argument_type_mismatch:{node.name}:"
                    f"{schema_argument.name}"
                )
            ref = encode_argument(argument, node)
            if isinstance(ref, str):
                return _rejected(ref)
            refs.append(ref)
        value_ids[node] = len(placeholders) + len(node_names)
        node_names.append(node.name)
        operator_names.append(schema.name)
        overload_names.append(schema.overload_name)
        argument_refs.append(refs)

    if not node_names:
        return _rejected("v1_requires_at_least_one_instruction")
    output_node = next(
        node for node in graph_module.graph.nodes if node.op == "output"
    )
    output_leaves, _ = pytree.tree_flatten(output_node.args[0])
    output_value_ids: list[int] = []
    for leaf in output_leaves:
        if not isinstance(leaf, torch.fx.Node) or leaf not in value_ids:
            return _rejected("v1_requires_tensor_value_outputs")
        output_value_ids.append(value_ids[leaf])
    if not output_value_ids:
        return _rejected("v1_requires_tensor_value_outputs")

    input_count = len(placeholders)
    value_count = input_count + len(node_names)
    use_counts = [0] * value_count
    last_uses = [-1] * value_count
    for instruction_index, refs in enumerate(argument_refs):
        output_value_id = input_count + instruction_index
        last_uses[output_value_id] = instruction_index
        for ref in refs:
            if ref >= 0:
                use_counts[ref] += 1
                last_uses[ref] = instruction_index

    plan = torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
        node_names,
        operator_names,
        overload_names,
        argument_refs,
        constants,
        input_count,
        output_value_ids,
    )
    report = VulkanGraphPlanReport(
        status="compiled",
        reason="immutable_tensor_ssa_plan",
        plan_class="VulkanGraphPlan",
        plan_version="v1",
        input_count=input_count,
        instruction_count=len(node_names),
        value_count=value_count,
        output_count=len(output_value_ids),
        node_names=tuple(node_names),
        value_use_counts=tuple(use_counts),
        value_last_uses=tuple(last_uses),
    )
    if (
        plan.input_count() != report.input_count
        or plan.instruction_count() != report.instruction_count
        or plan.value_count() != report.value_count
        or plan.output_count() != report.output_count
        or tuple(plan.value_use_counts()) != report.value_use_counts
        or tuple(plan.value_last_uses()) != report.value_last_uses
    ):
        raise RuntimeError("VulkanGraphPlan.v1 C++ schema disagrees with lowering")
    return _VulkanGraphPlanCompilation(plan, report)


__all__ = ["VulkanGraphPlanReport", "compile_vulkan_graph_plan"]
