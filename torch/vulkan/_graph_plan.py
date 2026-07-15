from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.utils import _pytree as pytree

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class VulkanGraphPlanReport:
    status: str
    reason: str
    plan_class: str
    plan_version: str
    input_count: int
    instruction_count: int
    effect_instruction_count: int
    list_argument_count: int
    value_count: int
    output_count: int
    node_names: tuple[str, ...]
    value_use_counts: tuple[int, ...]
    value_last_uses: tuple[int, ...]


@dataclass(frozen=True)
class _VulkanGraphPlanCompilation:
    plan: Any | None
    report: VulkanGraphPlanReport


_VALUE_ARGUMENT = 0
_LIST_ARGUMENT = 1


def _rejected(reason: str) -> _VulkanGraphPlanCompilation:
    return _VulkanGraphPlanCompilation(
        plan=None,
        report=VulkanGraphPlanReport(
            status="python_correctness_executor",
            reason=reason,
            plan_class="VulkanGraphPlan",
            plan_version="v3",
            input_count=0,
            instruction_count=0,
            effect_instruction_count=0,
            list_argument_count=0,
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
    value_types: Mapping[torch.fx.Node, Any],
) -> bool:
    if isinstance(value, torch.fx.Node):
        if value.op == "get_attr":
            value = _fetch_attr(graph_module, str(value.target))
        else:
            actual_type = value_types.get(value)
            return actual_type is not None and actual_type.isSubtypeOf(
                expected_type
            )
    inferred_type = torch._C._jit_try_infer_type(value)
    return inferred_type.success() and inferred_type.type().isSubtypeOf(
        expected_type
    )


def _unwrapped_optional_type(value_type: Any) -> Any:
    if value_type.kind() == "OptionalType":
        return value_type.getElementType()
    return value_type


def _canonicalize_argument(value: Any, expected_type: Any) -> Any:
    value_type = _unwrapped_optional_type(expected_type)
    if isinstance(value, str) and value_type.kind() == "DeviceObjType":
        try:
            return torch.device(value)
        except RuntimeError:
            pass
    return value


def compile_vulkan_graph_plan(
    graph_module: torch.fx.GraphModule,
    classifications: Mapping[str, str],
) -> _VulkanGraphPlanCompilation:
    placeholders = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    if not placeholders:
        return _rejected("v3_requires_tensor_inputs")
    for node in placeholders:
        if not isinstance(node.meta.get("val"), torch.Tensor):
            return _rejected(f"non_tensor_input:{node.name}")

    value_ids = {node: index for index, node in enumerate(placeholders)}
    value_types = {
        node: torch._C.TensorType.get() for node in placeholders
    }
    constants: list[Any] = []
    constant_ids: dict[torch.fx.Node, int] = {}
    node_names: list[str] = []
    operator_names: list[str] = []
    overload_names: list[str] = []
    argument_refs: list[list[list[int]]] = []
    argument_kinds: list[list[int]] = []
    instruction_output_value_ids: list[int] = []
    next_value_id = len(placeholders)

    def encode_leaf(value: Any, consumer: torch.fx.Node) -> int | str:
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

    def encode_bound_argument(
        value: Any,
        expected_type: Any,
        consumer: torch.fx.Node,
        argument_name: str,
    ) -> tuple[int, list[int]] | str:
        value = _canonicalize_argument(value, expected_type)
        if not isinstance(value, torch.fx.Node) and _contains_node(value):
            list_type = _unwrapped_optional_type(expected_type)
            if (
                not isinstance(value, (tuple, list))
                or list_type.kind() != "ListType"
            ):
                return (
                    f"unsupported_dynamic_container:{consumer.name}:"
                    f"{argument_name}:{type(value).__name__}"
                )
            element_type = list_type.getElementType()
            refs: list[int] = []
            for item in value:
                item = _canonicalize_argument(item, element_type)
                if not isinstance(item, torch.fx.Node) and _contains_node(item):
                    return (
                        f"nested_dynamic_container:{consumer.name}:"
                        f"{argument_name}"
                    )
                if not _argument_type_matches(
                    graph_module,
                    item,
                    element_type,
                    value_types,
                ):
                    return (
                        f"argument_type_mismatch:{consumer.name}:"
                        f"{argument_name}"
                    )
                ref = encode_leaf(item, consumer)
                if isinstance(ref, str):
                    return ref
                refs.append(ref)
            return _LIST_ARGUMENT, refs
        if not _argument_type_matches(
            graph_module,
            value,
            expected_type,
            value_types,
        ):
            return f"argument_type_mismatch:{consumer.name}:{argument_name}"
        ref = encode_leaf(value, consumer)
        if isinstance(ref, str):
            return ref
        return _VALUE_ARGUMENT, [ref]

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
        if len(schema.returns) > 1:
            return _rejected(
                f"multiple_dispatch_returns:{node.name}:{schema.name}"
            )
        bound_arguments = _bound_operator_arguments(node)
        if isinstance(bound_arguments, str):
            return _rejected(bound_arguments)
        refs: list[list[int]] = []
        kinds: list[int] = []
        for schema_argument, argument in zip(schema.arguments, bound_arguments):
            encoded = encode_bound_argument(
                argument,
                schema_argument.type,
                node,
                schema_argument.name,
            )
            if isinstance(encoded, str):
                return _rejected(encoded)
            kind, encoded_refs = encoded
            kinds.append(kind)
            refs.append(encoded_refs)
        output_value_id = -1
        if schema.returns:
            output_value_id = next_value_id
            next_value_id += 1
            value_ids[node] = output_value_id
            value_types[node] = schema.returns[0].type
        node_names.append(node.name)
        operator_names.append(schema.name)
        overload_names.append(schema.overload_name)
        argument_refs.append(refs)
        argument_kinds.append(kinds)
        instruction_output_value_ids.append(output_value_id)

    if not node_names:
        return _rejected("v3_requires_at_least_one_instruction")
    output_node = next(
        node for node in graph_module.graph.nodes if node.op == "output"
    )
    output_leaves, _ = pytree.tree_flatten(output_node.args[0])
    output_value_ids: list[int] = []
    for leaf in output_leaves:
        if not isinstance(leaf, torch.fx.Node) or leaf not in value_ids:
            return _rejected("v3_requires_tensor_value_outputs")
        if not value_types[leaf].isSubtypeOf(torch._C.TensorType.get()):
            return _rejected("v3_requires_tensor_value_outputs")
        output_value_ids.append(value_ids[leaf])
    if not output_value_ids:
        return _rejected("v3_requires_tensor_value_outputs")

    input_count = len(placeholders)
    value_count = next_value_id
    use_counts = [0] * value_count
    last_uses = [-1] * value_count
    for instruction_index, (refs, output_value_id) in enumerate(
        zip(argument_refs, instruction_output_value_ids)
    ):
        if output_value_id >= 0:
            last_uses[output_value_id] = instruction_index
        for argument_refs_for_value in refs:
            for ref in argument_refs_for_value:
                if ref >= 0:
                    use_counts[ref] += 1
                    last_uses[ref] = instruction_index

    plan = torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
        node_names,
        operator_names,
        overload_names,
        argument_refs,
        argument_kinds,
        instruction_output_value_ids,
        constants,
        input_count,
        output_value_ids,
    )
    report = VulkanGraphPlanReport(
        status="compiled",
        reason="immutable_ivalue_ssa_plan",
        plan_class="VulkanGraphPlan",
        plan_version="v3",
        input_count=input_count,
        instruction_count=len(node_names),
        effect_instruction_count=sum(
            output_value_id < 0
            for output_value_id in instruction_output_value_ids
        ),
        list_argument_count=sum(
            kind == _LIST_ARGUMENT
            for instruction_kinds in argument_kinds
            for kind in instruction_kinds
        ),
        value_count=value_count,
        output_count=len(output_value_ids),
        node_names=tuple(node_names),
        value_use_counts=tuple(use_counts),
        value_last_uses=tuple(last_uses),
    )
    if (
        plan.input_count() != report.input_count
        or plan.instruction_count() != report.instruction_count
        or plan.effect_instruction_count()
        != report.effect_instruction_count
        or plan.list_argument_count() != report.list_argument_count
        or plan.value_count() != report.value_count
        or plan.output_count() != report.output_count
        or tuple(plan.value_use_counts()) != report.value_use_counts
        or tuple(plan.value_last_uses()) != report.value_last_uses
    ):
        raise RuntimeError("VulkanGraphPlan.v3 C++ schema disagrees with lowering")
    return _VulkanGraphPlanCompilation(plan, report)


__all__ = ["VulkanGraphPlanReport", "compile_vulkan_graph_plan"]
