from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.utils import _pytree as pytree

if TYPE_CHECKING:
    from collections.abc import Mapping


_MODEL_DOMAIN_VALUES = {"generic": 0, "vision": 1, "llm": 2}
_EXECUTION_PHASE_VALUES = {
    "none": 0,
    "prefill": 1,
    "decode": 2,
    "backbone": 3,
    "decoder": 4,
}
_MODEL_DOMAIN_PHASES = {
    "generic": frozenset(("none",)),
    "vision": frozenset(("none", "backbone", "decoder")),
    "llm": frozenset(("prefill", "decode")),
}


@dataclass(frozen=True)
class VulkanGraphPlanningContext:
    model_domain: str = "generic"
    execution_phase: str = "none"
    prefer_packed_layout_propagation: bool = False
    fixed_shape_graph_input_sizes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.model_domain not in _MODEL_DOMAIN_VALUES:
            raise ValueError(
                "Vulkan graph model_domain must be generic, vision, or llm"
            )
        if self.execution_phase not in _EXECUTION_PHASE_VALUES:
            raise ValueError(
                "Vulkan graph execution_phase must be none, prefill, decode, "
                "backbone, or decoder"
            )
        if self.execution_phase not in _MODEL_DOMAIN_PHASES[self.model_domain]:
            raise ValueError(
                "Vulkan graph execution_phase is incompatible with model_domain"
            )
        if type(self.prefer_packed_layout_propagation) is not bool:
            raise TypeError("prefer_packed_layout_propagation must be bool")
        sizes = self.fixed_shape_graph_input_sizes
        if sizes is not None and (
            type(sizes) is not tuple
            or not sizes
            or any(type(size) is not int or size <= 0 for size in sizes)
        ):
            raise ValueError(
                "fixed_shape_graph_input_sizes must be a non-empty tuple of "
                "positive integers"
            )

    @property
    def model_domain_value(self) -> int:
        return _MODEL_DOMAIN_VALUES[self.model_domain]

    @property
    def execution_phase_value(self) -> int:
        return _EXECUTION_PHASE_VALUES[self.execution_phase]


@dataclass(frozen=True)
class VulkanGraphPlanReport:
    status: str
    reason: str
    plan_class: str
    plan_version: str
    planning_model_domain: str
    planning_execution_phase: str
    planning_prefer_packed_layout_propagation: bool
    planning_fixed_shape_graph_input_sizes: tuple[int, ...] | None
    input_count: int
    instruction_count: int
    effect_instruction_count: int
    graph_scalar_instruction_count: int
    list_projection_instruction_count: int
    list_argument_count: int
    invocation_value_slot_count: int
    invocation_list_slot_count: int
    invocation_stack_capacity: int
    dead_input_reuse_instruction_count: int
    resource_slot_count: int
    resource_value_count: int
    resource_writer_instruction_count: int
    resource_arena_flight_depth: int
    resource_alias_extended_lifetime_count: int
    resource_alias_escape_rejection_count: int
    value_count: int
    output_count: int
    submission_owned: bool
    node_names: tuple[str, ...]
    value_use_counts: tuple[int, ...]
    value_last_uses: tuple[int, ...]


@dataclass(frozen=True)
class _VulkanGraphPlanCompilation:
    plan: Any | None
    report: VulkanGraphPlanReport
    input_indices: tuple[int, ...]


_VALUE_ARGUMENT = 0
_LIST_ARGUMENT = 1
_GRAPH_INT_OPERATOR_NAMES = {
    operator.add: "vulkan_graph::int_add",
    operator.sub: "vulkan_graph::int_subtract",
    operator.mul: "vulkan_graph::int_multiply",
    operator.floordiv: "vulkan_graph::int_floor_divide",
}
_GRAPH_LIST_GETITEM_OPERATOR_NAME = "vulkan_graph::list_getitem"
_RESOURCE_WRITER_OPERATOR_NAMES = frozenset(
    (
        "vulkan_prepack::run_linear_context",
        "vulkan_prepack::run_graph_add_layernorm_plan",
        "vulkan_prepack::run_vulkan_graph_region_plan",
        "vulkan_prepack::run_graph_attention_math",
    )
)
_RESOURCE_ARENA_FLIGHT_DEPTH = 2
_ATTENTION_SCRATCH_RESOURCE_COUNT = 3


@dataclass(frozen=True)
class _VulkanGraphResourceDescriptor:
    sizes: tuple[int, ...]
    dtype: torch.dtype
    storage_type: int
    memory_layout: int
    execution_layout: int


_STORAGE_TYPE_BUFFER = 0
_MEMORY_LAYOUT_TENSOR_WIDTH_PACKED = 0
_EXECUTION_LAYOUT_BUFFER_DIRECT = 1
_DIRECT_WIDTH_BUFFER_RESOURCE = (
    _STORAGE_TYPE_BUFFER,
    _MEMORY_LAYOUT_TENSOR_WIDTH_PACKED,
    _EXECUTION_LAYOUT_BUFFER_DIRECT,
)


def _rejected(
    reason: str,
    planning_context: VulkanGraphPlanningContext,
) -> _VulkanGraphPlanCompilation:
    return _VulkanGraphPlanCompilation(
        plan=None,
        report=VulkanGraphPlanReport(
            status="python_correctness_executor",
            reason=reason,
            plan_class="VulkanGraphPlan",
            plan_version="v9",
            planning_model_domain=planning_context.model_domain,
            planning_execution_phase=planning_context.execution_phase,
            planning_prefer_packed_layout_propagation=(
                planning_context.prefer_packed_layout_propagation
            ),
            planning_fixed_shape_graph_input_sizes=(
                planning_context.fixed_shape_graph_input_sizes
            ),
            input_count=0,
            instruction_count=0,
            effect_instruction_count=0,
            graph_scalar_instruction_count=0,
            list_projection_instruction_count=0,
            list_argument_count=0,
            invocation_value_slot_count=0,
            invocation_list_slot_count=0,
            invocation_stack_capacity=0,
            dead_input_reuse_instruction_count=0,
            resource_slot_count=0,
            resource_value_count=0,
            resource_writer_instruction_count=0,
            resource_arena_flight_depth=0,
            resource_alias_extended_lifetime_count=0,
            resource_alias_escape_rejection_count=0,
            value_count=0,
            output_count=0,
            submission_owned=False,
            node_names=(),
            value_use_counts=(),
            value_last_uses=(),
        ),
        input_indices=(),
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
    if value_type.kind() == "TensorType" and isinstance(
        value, (bool, int, float, complex)
    ):
        return torch.tensor(value, device="cpu")
    return value


def _resource_descriptor(
    value: Any,
    planning_context: VulkanGraphPlanningContext,
) -> _VulkanGraphResourceDescriptor | None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        return None
    sizes: list[int] = []
    for size in value.shape:
        if type(size) is int:
            concrete_size = size
        elif planning_context.fixed_shape_graph_input_sizes is not None:
            try:
                concrete_size = int(size)
            except (TypeError, ValueError, RuntimeError):
                return None
        else:
            return None
        if concrete_size <= 0:
            return None
        sizes.append(concrete_size)
    return _VulkanGraphResourceDescriptor(
        tuple(sizes), value.dtype, *_DIRECT_WIDTH_BUFFER_RESOURCE
    )


def _resource_output_descriptors(
    node: torch.fx.Node,
    output_count: int,
    planning_context: VulkanGraphPlanningContext,
) -> tuple[_VulkanGraphResourceDescriptor | None, ...]:
    if output_count == 1:
        descriptor = _resource_descriptor(node.meta.get("val"), planning_context)
        if (
            descriptor is not None
            and node.meta.get("vulkan_graph_region_family")
            in ("linear_gelu_tanh", "linear_gelu_none")
            and len(descriptor.sizes) != 2
        ):
            row_count = 1
            for size in descriptor.sizes[:-1]:
                row_count *= size
            descriptor = _VulkanGraphResourceDescriptor(
                (row_count, descriptor.sizes[-1]),
                descriptor.dtype,
                descriptor.storage_type,
                descriptor.memory_layout,
                descriptor.execution_layout,
            )
        return (descriptor,)

    descriptors: list[_VulkanGraphResourceDescriptor | None] = [
        None
    ] * output_count
    node_value = node.meta.get("val")
    if isinstance(node_value, (tuple, list)) and len(node_value) == output_count:
        descriptors = [
            _resource_descriptor(value, planning_context) for value in node_value
        ]
    for user in node.users:
        if (
            user.op != "call_function"
            or user.target is not operator.getitem
            or len(user.args) != 2
            or user.args[0] is not node
            or type(user.args[1]) is not int
        ):
            continue
        index = user.args[1]
        if index < 0:
            index += output_count
        if 0 <= index < output_count:
            descriptors[index] = _resource_descriptor(
                user.meta.get("val"), planning_context
            )
    if node.target._schema.name == "vulkan_prepack::run_graph_add_layernorm_plan":
        known = next((item for item in descriptors if item is not None), None)
        if known is not None:
            descriptors = [item if item is not None else known for item in descriptors]
    return tuple(descriptors)


def _attention_scratch_descriptors(
    node: torch.fx.Node,
    planning_context: VulkanGraphPlanningContext,
) -> tuple[_VulkanGraphResourceDescriptor, ...] | None:
    if node.target._schema.name != "vulkan_prepack::run_graph_attention_math":
        return ()
    bound_arguments = _bound_operator_arguments(node)
    if isinstance(bound_arguments, str) or len(bound_arguments) < 3:
        return None
    query_node, key_node = bound_arguments[:2]
    if not isinstance(query_node, torch.fx.Node) or not isinstance(
        key_node, torch.fx.Node
    ):
        return None
    query = query_node.meta.get("val")
    key = key_node.meta.get("val")
    query_descriptor = _resource_descriptor(query, planning_context)
    key_descriptor = _resource_descriptor(key, planning_context)
    if (
        query_descriptor is None
        or key_descriptor is None
        or len(query_descriptor.sizes) not in (3, 4)
        or len(key_descriptor.sizes) != len(query_descriptor.sizes)
    ):
        return None
    scores_descriptor = _VulkanGraphResourceDescriptor(
        (*query_descriptor.sizes[:-1], key_descriptor.sizes[-2]),
        torch.float32,
        *_DIRECT_WIDTH_BUFFER_RESOURCE,
    )
    return (query_descriptor, scores_descriptor, scores_descriptor)


def _alias_info_labels(alias_info: Any) -> set[str]:
    if alias_info is None:
        return set()
    return set(alias_info.before_set) | set(alias_info.after_set)


def _alias_infos_overlap(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    left_labels = _alias_info_labels(left)
    right_labels = _alias_info_labels(right)
    if not left_labels or not right_labels:
        return True
    return (
        "*" in left_labels
        or "*" in right_labels
        or not left_labels.isdisjoint(right_labels)
    )


def compile_vulkan_graph_plan(
    graph_module: torch.fx.GraphModule,
    classifications: Mapping[str, str],
    planning_context: VulkanGraphPlanningContext | None = None,
) -> _VulkanGraphPlanCompilation:
    if planning_context is None:
        planning_context = VulkanGraphPlanningContext()

    def rejected(reason: str) -> _VulkanGraphPlanCompilation:
        return _rejected(reason, planning_context)

    graph_placeholders = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    if not graph_placeholders:
        return rejected("v9_requires_tensor_inputs")
    for node in graph_placeholders:
        if (
            not isinstance(node.meta.get("val"), torch.Tensor)
            and node.users
        ):
            return rejected(f"non_tensor_input:{node.name}")
    input_indices = tuple(
        index
        for index, node in enumerate(graph_placeholders)
        if isinstance(node.meta.get("val"), torch.Tensor)
    )
    placeholders = [graph_placeholders[index] for index in input_indices]
    if not placeholders:
        return rejected("v9_requires_tensor_inputs")

    value_ids = {node: index for index, node in enumerate(placeholders)}
    value_types = {node: torch._C.TensorType.get() for node in placeholders}
    multi_value_ids: dict[torch.fx.Node, tuple[int, ...]] = {}
    multi_value_types: dict[torch.fx.Node, tuple[Any, ...]] = {}
    constants: list[Any] = []
    constant_ids: dict[torch.fx.Node, int] = {}
    node_names: list[str] = []
    operator_names: list[str] = []
    overload_names: list[str] = []
    argument_refs: list[list[list[int]]] = []
    argument_kinds: list[list[int]] = []
    instruction_output_value_ids: list[list[int]] = []
    instruction_nodes: list[torch.fx.Node] = []
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
        list_type = _unwrapped_optional_type(expected_type)
        if (
            isinstance(value, (tuple, list))
            and not value
            and list_type.kind() == "ListType"
        ):
            return _LIST_ARGUMENT, []
        if not isinstance(value, torch.fx.Node) and _contains_node(value):
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
        graph_int_operator = (
            _GRAPH_INT_OPERATOR_NAMES.get(node.target)
            if node.op == "call_function"
            else None
        )
        if graph_int_operator is not None:
            if classifications.get(node.name) != "graph":
                return rejected(
                    f"graph_scalar_not_admitted:{node.name}:"
                    f"{classifications.get(node.name, 'unknown')}"
                )
            if len(node.args) != 2 or node.kwargs:
                return rejected(f"invalid_graph_scalar_arguments:{node.name}")
            refs: list[list[int]] = []
            for argument_index, argument in enumerate(node.args):
                encoded = encode_bound_argument(
                    argument,
                    torch._C.IntType.get(),
                    node,
                    f"operand_{argument_index}",
                )
                if isinstance(encoded, str):
                    return rejected(encoded)
                kind, encoded_refs = encoded
                if kind != _VALUE_ARGUMENT:
                    return rejected(
                        f"graph_scalar_container_argument:{node.name}:"
                        f"operand_{argument_index}"
                    )
                refs.append(encoded_refs)
            output_value_id = next_value_id
            next_value_id += 1
            value_ids[node] = output_value_id
            value_types[node] = torch._C.IntType.get()
            node_names.append(node.name)
            operator_names.append(graph_int_operator)
            overload_names.append("")
            argument_refs.append(refs)
            argument_kinds.append([_VALUE_ARGUMENT, _VALUE_ARGUMENT])
            instruction_output_value_ids.append([output_value_id])
            instruction_nodes.append(node)
            continue
        if (
            node.op == "call_function"
            and node.target is operator.getitem
            and node.args
            and isinstance(node.args[0], torch.fx.Node)
            and node.args[0] in multi_value_ids
        ):
            if classifications.get(node.name) != "graph":
                return rejected(
                    f"multi_return_getitem_not_admitted:{node.name}:"
                    f"{classifications.get(node.name, 'unknown')}"
                )
            if len(node.args) != 2 or node.kwargs:
                return rejected(f"invalid_multi_return_getitem:{node.name}")
            producer = node.args[0]
            index = node.args[1]
            if type(index) is not int:
                return rejected(f"invalid_multi_return_index:{node.name}")
            output_ids = multi_value_ids[producer]
            normalized_index = index if index >= 0 else len(output_ids) + index
            if normalized_index < 0 or normalized_index >= len(output_ids):
                return rejected(f"multi_return_index_out_of_range:{node.name}")
            value_ids[node] = output_ids[normalized_index]
            value_types[node] = multi_value_types[producer][normalized_index]
            continue
        if (
            node.op == "call_function"
            and node.target is operator.getitem
            and node.args
            and isinstance(node.args[0], torch.fx.Node)
            and node.args[0] in value_ids
            and value_types[node.args[0]].kind() == "ListType"
        ):
            if classifications.get(node.name) != "graph":
                return rejected(
                    f"list_projection_not_admitted:{node.name}:"
                    f"{classifications.get(node.name, 'unknown')}"
                )
            if len(node.args) != 2 or node.kwargs:
                return rejected(f"invalid_list_projection:{node.name}")
            producer = node.args[0]
            index = node.args[1]
            if type(index) is not int:
                return rejected(f"invalid_list_projection_index:{node.name}")
            constants.append(index)
            output_value_id = next_value_id
            next_value_id += 1
            value_ids[node] = output_value_id
            value_types[node] = value_types[producer].getElementType()
            node_names.append(node.name)
            operator_names.append(_GRAPH_LIST_GETITEM_OPERATOR_NAME)
            overload_names.append("")
            argument_refs.append([[value_ids[producer]], [-len(constants)]])
            argument_kinds.append([_VALUE_ARGUMENT, _VALUE_ARGUMENT])
            instruction_output_value_ids.append([output_value_id])
            instruction_nodes.append(node)
            continue
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            return rejected(f"unsupported_node_kind:{node.name}:{node.op}")
        if classifications.get(node.name) not in (
            "direct_vulkan",
            "lowered_vulkan",
            "composite",
        ):
            return rejected(
                f"node_not_vulkan_admitted:{node.name}:"
                f"{classifications.get(node.name, 'unknown')}"
            )
        schema = node.target._schema
        if schema.is_mutable:
            return rejected(f"mutable_operator:{node.name}:{schema.name}")
        bound_arguments = _bound_operator_arguments(node)
        if isinstance(bound_arguments, str):
            return rejected(bound_arguments)
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
                return rejected(encoded)
            kind, encoded_refs = encoded
            kinds.append(kind)
            refs.append(encoded_refs)
        output_value_ids: list[int] = []
        if schema.returns:
            output_value_ids = list(
                range(next_value_id, next_value_id + len(schema.returns))
            )
            next_value_id += len(schema.returns)
            if len(schema.returns) == 1:
                value_ids[node] = output_value_ids[0]
                value_types[node] = schema.returns[0].type
            else:
                multi_value_ids[node] = tuple(output_value_ids)
                multi_value_types[node] = tuple(
                    result.type for result in schema.returns
                )
        node_names.append(node.name)
        operator_names.append(schema.name)
        overload_names.append(schema.overload_name)
        argument_refs.append(refs)
        argument_kinds.append(kinds)
        instruction_output_value_ids.append(output_value_ids)
        instruction_nodes.append(node)

    if not node_names:
        return rejected("v9_requires_at_least_one_instruction")
    output_node = next(
        node for node in graph_module.graph.nodes if node.op == "output"
    )
    output_leaves, _ = pytree.tree_flatten(output_node.args[0])
    output_value_ids: list[int] = []
    for leaf in output_leaves:
        if not isinstance(leaf, torch.fx.Node) or leaf not in value_ids:
            return rejected("v9_requires_tensor_value_outputs")
        if not value_types[leaf].isSubtypeOf(torch._C.TensorType.get()):
            return rejected("v9_requires_tensor_value_outputs")
        output_value_ids.append(value_ids[leaf])
    if not output_value_ids:
        return rejected("v9_requires_tensor_value_outputs")

    input_count = len(placeholders)
    value_count = next_value_id
    use_counts = [0] * value_count
    last_uses = [-1] * value_count
    for instruction_index, (refs, instruction_output_ids) in enumerate(
        zip(argument_refs, instruction_output_value_ids)
    ):
        for output_value_id in instruction_output_ids:
            last_uses[output_value_id] = instruction_index
        for argument_refs_for_value in refs:
            for ref in argument_refs_for_value:
                if ref >= 0:
                    use_counts[ref] += 1
                    last_uses[ref] = instruction_index

    alias_parents = list(range(value_count))

    def find_alias_root(value_id: int) -> int:
        root = value_id
        while alias_parents[root] != root:
            root = alias_parents[root]
        while alias_parents[value_id] != value_id:
            parent = alias_parents[value_id]
            alias_parents[value_id] = root
            value_id = parent
        return root

    def union_aliases(left: int, right: int) -> None:
        left_root = find_alias_root(left)
        right_root = find_alias_root(right)
        if left_root != right_root:
            alias_parents[right_root] = left_root

    for node, refs, output_ids in zip(
        instruction_nodes, argument_refs, instruction_output_value_ids
    ):
        if not isinstance(node.target, torch._ops.OpOverload):
            continue
        schema = node.target._schema
        for return_schema, output_id in zip(schema.returns, output_ids):
            if return_schema.alias_info is None:
                continue
            for argument_schema, encoded_refs in zip(schema.arguments, refs):
                if not _alias_infos_overlap(
                    return_schema.alias_info, argument_schema.alias_info
                ):
                    continue
                for ref in encoded_refs:
                    if ref >= 0:
                        union_aliases(output_id, ref)

    alias_components: dict[int, list[int]] = {}
    for value_id in range(value_count):
        alias_components.setdefault(find_alias_root(value_id), []).append(value_id)
    alias_last_uses = list(last_uses)
    alias_escapes = [False] * value_count
    escaping_value_ids = set(output_value_ids)
    for component in alias_components.values():
        component_last_use = max(last_uses[value_id] for value_id in component)
        component_escapes = any(
            value_id in escaping_value_ids for value_id in component
        )
        for value_id in component:
            alias_last_uses[value_id] = component_last_use
            alias_escapes[value_id] = component_escapes

    value_resource_slot_ids = [-1] * value_count
    instruction_scratch_resource_slot_ids = [-1] * (
        len(instruction_nodes) * _ATTENTION_SCRATCH_RESOURCE_COUNT
    )
    resource_descriptors: list[_VulkanGraphResourceDescriptor] = []
    resource_slot_last_uses: list[int] = []
    resource_writer_instruction_count = 0
    resource_alias_extended_lifetime_count = 0
    resource_alias_escape_rejection_count = 0

    def reserve_resource_slot(
        descriptor: _VulkanGraphResourceDescriptor,
        instruction_index: int,
        last_use: int,
    ) -> int:
        slot_id = next(
            (
                index
                for index, (slot_descriptor, slot_last_use) in enumerate(
                    zip(resource_descriptors, resource_slot_last_uses)
                )
                if slot_descriptor == descriptor
                and slot_last_use < instruction_index
            ),
            -1,
        )
        if slot_id < 0:
            slot_id = len(resource_descriptors)
            resource_descriptors.append(descriptor)
            resource_slot_last_uses.append(last_use)
        else:
            resource_slot_last_uses[slot_id] = last_use
        return slot_id

    for instruction_index, (node, operator_name, instruction_output_ids) in enumerate(
        zip(instruction_nodes, operator_names, instruction_output_value_ids)
    ):
        if (
            operator_name not in _RESOURCE_WRITER_OPERATOR_NAMES
            or not instruction_output_ids
        ):
            continue
        if (
            operator_name == "vulkan_prepack::run_vulkan_graph_region_plan"
            and node.meta.get("vulkan_graph_region_family")
            not in ("linear_gelu_tanh", "linear_gelu_none")
        ):
            continue
        if any(alias_escapes[value_id] for value_id in instruction_output_ids):
            resource_alias_escape_rejection_count += 1
            continue
        descriptors = _resource_output_descriptors(
            node, len(instruction_output_ids), planning_context
        )
        if any(descriptor is None for descriptor in descriptors):
            continue
        if operator_name.startswith("aten::") and any(
            descriptor is not None and descriptor.dtype is not torch.float32
            for descriptor in descriptors
        ):
            continue
        scratch_descriptors = _attention_scratch_descriptors(node, planning_context)
        if scratch_descriptors is None:
            continue
        if scratch_descriptors:
            scratch_slot_ids = [
                reserve_resource_slot(
                    descriptor, instruction_index, instruction_index
                )
                for descriptor in scratch_descriptors
            ]
            if len(set(scratch_slot_ids)) != _ATTENTION_SCRATCH_RESOURCE_COUNT:
                raise AssertionError(
                    "attention scratch resources must have distinct live slots"
                )
            scratch_offset = (
                instruction_index * _ATTENTION_SCRATCH_RESOURCE_COUNT
            )
            instruction_scratch_resource_slot_ids[
                scratch_offset : scratch_offset
                + _ATTENTION_SCRATCH_RESOURCE_COUNT
            ] = scratch_slot_ids
        resource_writer_instruction_count += 1
        for value_id, descriptor in zip(instruction_output_ids, descriptors):
            assert descriptor is not None
            slot_id = reserve_resource_slot(
                descriptor, instruction_index, alias_last_uses[value_id]
            )
            value_resource_slot_ids[value_id] = slot_id
            if alias_last_uses[value_id] > last_uses[value_id]:
                resource_alias_extended_lifetime_count += 1

    resource_slot_sizes = [
        size for descriptor in resource_descriptors for size in descriptor.sizes
    ]
    resource_slot_ranks = [
        len(descriptor.sizes) for descriptor in resource_descriptors
    ]
    resource_slot_storage_types = [
        descriptor.storage_type for descriptor in resource_descriptors
    ]
    resource_slot_memory_layouts = [
        descriptor.memory_layout for descriptor in resource_descriptors
    ]
    resource_slot_execution_layouts = [
        descriptor.execution_layout for descriptor in resource_descriptors
    ]
    resource_value_count = sum(slot_id >= 0 for slot_id in value_resource_slot_ids)

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
        planning_context.model_domain_value,
        planning_context.execution_phase_value,
        planning_context.prefer_packed_layout_propagation,
        planning_context.fixed_shape_graph_input_sizes,
        value_resource_slot_ids,
        resource_slot_sizes,
        resource_slot_ranks,
        _RESOURCE_ARENA_FLIGHT_DEPTH,
        resource_slot_storage_types,
        resource_slot_memory_layouts,
        resource_slot_execution_layouts,
        instruction_scratch_resource_slot_ids,
    )
    report = VulkanGraphPlanReport(
        status="compiled",
        reason="immutable_ivalue_ssa_resource_plan",
        plan_class="VulkanGraphPlan",
        plan_version="v9",
        planning_model_domain=planning_context.model_domain,
        planning_execution_phase=planning_context.execution_phase,
        planning_prefer_packed_layout_propagation=(
            planning_context.prefer_packed_layout_propagation
        ),
        planning_fixed_shape_graph_input_sizes=(
            planning_context.fixed_shape_graph_input_sizes
        ),
        input_count=input_count,
        instruction_count=len(node_names),
        effect_instruction_count=sum(
            not output_value_ids
            for output_value_ids in instruction_output_value_ids
        ),
        graph_scalar_instruction_count=sum(
            operator_name in _GRAPH_INT_OPERATOR_NAMES.values()
            for operator_name in operator_names
        ),
        list_projection_instruction_count=sum(
            operator_name == _GRAPH_LIST_GETITEM_OPERATOR_NAME
            for operator_name in operator_names
        ),
        list_argument_count=sum(
            kind == _LIST_ARGUMENT
            for instruction_kinds in argument_kinds
            for kind in instruction_kinds
        ),
        invocation_value_slot_count=plan.invocation_value_slot_count(),
        invocation_list_slot_count=plan.invocation_list_slot_count(),
        invocation_stack_capacity=plan.invocation_stack_capacity(),
        dead_input_reuse_instruction_count=(
            plan.dead_input_reuse_instruction_count()
        ),
        resource_slot_count=len(resource_descriptors),
        resource_value_count=resource_value_count,
        resource_writer_instruction_count=resource_writer_instruction_count,
        resource_arena_flight_depth=(
            _RESOURCE_ARENA_FLIGHT_DEPTH if resource_descriptors else 0
        ),
        resource_alias_extended_lifetime_count=(
            resource_alias_extended_lifetime_count
        ),
        resource_alias_escape_rejection_count=(
            resource_alias_escape_rejection_count
        ),
        value_count=value_count,
        output_count=len(output_value_ids),
        submission_owned=plan.submission_owned(),
        node_names=tuple(node_names),
        value_use_counts=tuple(use_counts),
        value_last_uses=tuple(last_uses),
    )
    if (
        plan.input_count() != report.input_count
        or plan.instruction_count() != report.instruction_count
        or plan.effect_instruction_count()
        != report.effect_instruction_count
        or plan.graph_scalar_instruction_count()
        != report.graph_scalar_instruction_count
        or plan.list_projection_instruction_count()
        != report.list_projection_instruction_count
        or plan.list_argument_count() != report.list_argument_count
        or plan.invocation_value_slot_count()
        != report.invocation_value_slot_count
        or plan.invocation_list_slot_count()
        != report.invocation_list_slot_count
        or plan.invocation_stack_capacity()
        != report.invocation_stack_capacity
        or report.invocation_value_slot_count != report.value_count
        or report.invocation_list_slot_count > report.list_argument_count
        or plan.dead_input_reuse_instruction_count()
        != report.dead_input_reuse_instruction_count
        or plan.resource_slot_count() != report.resource_slot_count
        or plan.resource_value_count() != report.resource_value_count
        or plan.resource_writer_instruction_count()
        != report.resource_writer_instruction_count
        or plan.resource_arena_flight_depth()
        != report.resource_arena_flight_depth
        or plan.value_count() != report.value_count
        or plan.output_count() != report.output_count
        or plan.submission_owned() != report.submission_owned
        or plan.planning_model_domain()
        != planning_context.model_domain_value
        or plan.planning_execution_phase()
        != planning_context.execution_phase_value
        or plan.planning_prefer_packed_layout_propagation()
        != planning_context.prefer_packed_layout_propagation
        or (
            tuple(plan.planning_fixed_shape_graph_input_sizes())
            if plan.planning_fixed_shape_graph_input_sizes() is not None
            else None
        )
        != planning_context.fixed_shape_graph_input_sizes
        or tuple(plan.value_use_counts()) != report.value_use_counts
        or tuple(plan.value_last_uses()) != report.value_last_uses
    ):
        raise RuntimeError("VulkanGraphPlan.v9 C++ schema disagrees with lowering")
    return _VulkanGraphPlanCompilation(plan, report, input_indices)


__all__ = [
    "VulkanGraphPlanningContext",
    "VulkanGraphPlanReport",
    "compile_vulkan_graph_plan",
]
