from __future__ import annotations

import dataclasses
import hashlib
import math
import operator
from collections.abc import Mapping
from typing import Any

import torch
import torch.fx


@dataclasses.dataclass(frozen=True)
class VulkanLinearLoweringNodeReport:
    node_name: str
    status: str
    reason: str
    weight_attr: str | None
    bias_attr: str | None
    context_attr: str | None
    context_status: str | None


@dataclasses.dataclass(frozen=True)
class VulkanLinearLoweringReport:
    linear_node_count: int
    lowered_count: int
    rejected_count: int
    created_context_count: int
    reused_context_count: int
    context_factory: str
    nodes: tuple[VulkanLinearLoweringNodeReport, ...]


@dataclasses.dataclass(frozen=True)
class VulkanConv2dLoweringNodeReport:
    node_name: str
    status: str
    reason: str
    weight_attr: str | None
    bias_attr: str | None
    context_attr: str | None
    context_status: str | None


@dataclasses.dataclass(frozen=True)
class VulkanConv2dLoweringReport:
    conv2d_node_count: int
    lowered_count: int
    rejected_count: int
    created_context_count: int
    reused_context_count: int
    context_factory: str
    nodes: tuple[VulkanConv2dLoweringNodeReport, ...]


@dataclasses.dataclass(frozen=True)
class VulkanLayernormLoweringNodeReport:
    node_name: str
    status: str
    reason: str
    weight_attr: str | None
    bias_attr: str | None
    normalized_shape: tuple[int, ...] | None
    eps: float | None
    context_attr: str | None
    context_status: str | None


@dataclasses.dataclass(frozen=True)
class VulkanLayernormLoweringReport:
    layer_norm_node_count: int
    lowered_count: int
    rejected_count: int
    skipped_count: int
    created_context_count: int
    reused_context_count: int
    context_factory: str
    nodes: tuple[VulkanLayernormLoweringNodeReport, ...]


@dataclasses.dataclass(frozen=True)
class VulkanStaticAddLayernormRegionNodeReport:
    node_name: str
    status: str
    reason: str
    add_node_name: str | None
    layernorm_node_name: str | None
    context_attr: str | None
    plan_attr: str | None
    normalized_shape: tuple[int, ...] | None
    program_name: str | None
    program_version: str | None
    fused_instruction: str | None
    instruction_count: int
    residual_input_ssa: int | None
    addend_input_ssa: int | None
    residual_output_ssa: int | None
    normalized_output_ssa: int | None
    residual_input_use_count: int | None
    residual_input_last_use: int | None
    addend_input_use_count: int | None
    addend_input_last_use: int | None
    static_context_slot: int | None
    context_ownership_outcome: str | None
    direct_transition_only: bool | None
    replay_state_empty: bool | None
    persistent_output_state: bool | None


@dataclasses.dataclass(frozen=True)
class VulkanStaticAddLayernormRegionReport:
    candidate_count: int
    lowered_count: int
    rejected_count: int
    skipped_count: int
    plan_factory: str
    nodes: tuple[VulkanStaticAddLayernormRegionNodeReport, ...]


@dataclasses.dataclass(frozen=True)
class VulkanStaticLinearGeluRegionNodeReport:
    node_name: str
    status: str
    reason: str
    linear_node_name: str | None
    context_attr: str | None
    plan_attr: str | None
    program_name: str | None
    program_version: str | None
    instruction_count: int
    input_ssa: int | None
    output_ssa: int | None
    input_use_count: int | None
    input_last_use: int | None
    static_context_slot: int | None
    direct_transition_only: bool | None
    replay_state_empty: bool | None
    region_family: str | None = None
    intermediate_ssa: int | None = None
    intermediate_use_count: int | None = None
    intermediate_last_use: int | None = None


@dataclasses.dataclass(frozen=True)
class VulkanStaticLinearGeluRegionReport:
    candidate_count: int
    lowered_count: int
    rejected_count: int
    skipped_count: int
    plan_factory: str
    nodes: tuple[VulkanStaticLinearGeluRegionNodeReport, ...]


@dataclasses.dataclass(frozen=True)
class VulkanGraphRegionFamilyDiagnostics:
    family: str
    candidate_count: int
    lowered_count: int
    rejected_count: int
    skipped_count: int
    plan_factory: str
    nodes: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class VulkanGraphRegionLoweringReport:
    plan_class: str
    plan_version: str
    families: tuple[VulkanGraphRegionFamilyDiagnostics, ...]


@dataclasses.dataclass(frozen=True)
class VulkanStaticConv2dReluRegionNodeReport:
    node_name: str
    status: str
    reason: str
    conv2d_node_name: str | None
    context_attr: str | None
    plan_attr: str | None
    program_name: str | None
    program_version: str | None
    instruction_count: int
    input_ssa: int | None
    output_ssa: int | None
    input_use_count: int | None
    input_last_use: int | None
    static_context_slot: int | None
    direct_transition_only: bool | None
    replay_state_empty: bool | None


@dataclasses.dataclass(frozen=True)
class VulkanStaticConv2dReluRegionReport:
    candidate_count: int
    lowered_count: int
    rejected_count: int
    skipped_count: int
    plan_factory: str
    nodes: tuple[VulkanStaticConv2dReluRegionNodeReport, ...]


@dataclasses.dataclass(frozen=True)
class VulkanStaticConv2dReluConv2dRegionNodeReport:
    node_name: str
    status: str
    reason: str
    first_conv2d_node_name: str | None
    relu_node_name: str | None
    second_conv2d_node_name: str | None
    first_context_attr: str | None
    second_context_attr: str | None
    plan_attr: str | None
    program_name: str | None
    program_version: str | None
    instruction_count: int
    input_ssa: int | None
    intermediate_ssa: int | None
    output_ssa: int | None
    input_use_count: int | None
    input_last_use: int | None
    intermediate_use_count: int | None
    intermediate_last_use: int | None
    first_static_context_slot: int | None
    second_static_context_slot: int | None
    bounded_submission_owned: bool | None
    program_private_scratch: bool | None
    scratch_ring_capacity: int | None
    timeline_gated_release: bool | None
    direct_transition_only: bool | None
    replay_state_empty: bool | None


@dataclasses.dataclass(frozen=True)
class VulkanStaticConv2dReluConv2dRegionReport:
    candidate_count: int
    lowered_count: int
    rejected_count: int
    skipped_count: int
    plan_factory: str
    nodes: tuple[VulkanStaticConv2dReluConv2dRegionNodeReport, ...]
    excluded_relu_node_names: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class VulkanGraphInputNormalizationNodeReport:
    node_name: str
    status: str
    reason: str
    placeholder_name: str | None
    source_dtype: torch.dtype | None
    target_dtype: torch.dtype | None
    erased_node_name: str | None
    chain_node_names: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class VulkanGraphInputNormalizationReport:
    candidate_count: int
    lowered_count: int
    rejected_count: int
    nodes: tuple[VulkanGraphInputNormalizationNodeReport, ...]


@dataclasses.dataclass(frozen=True)
class VulkanFreshDetachFunctionalizationNodeReport:
    node_name: str
    status: str
    reason: str
    source_node_name: str | None
    replacement_target: str | None


@dataclasses.dataclass(frozen=True)
class VulkanFreshDetachFunctionalizationReport:
    candidate_count: int
    functionalized_count: int
    rejected_count: int
    nodes: tuple[VulkanFreshDetachFunctionalizationNodeReport, ...]


@dataclasses.dataclass(frozen=True)
class VulkanFreshReluFunctionalizationNodeReport:
    node_name: str
    status: str
    reason: str
    source_node_name: str | None
    source_operator_name: str | None
    replacement_target: str | None


@dataclasses.dataclass(frozen=True)
class VulkanFreshReluFunctionalizationReport:
    candidate_count: int
    functionalized_count: int
    rejected_count: int
    nodes: tuple[VulkanFreshReluFunctionalizationNodeReport, ...]


@dataclasses.dataclass(frozen=True)
class VulkanStaticFactoryConstantNodeReport:
    node_name: str
    status: str
    reason: str
    operator_name: str
    constant_attr: str | None
    constant_status: str | None
    dtype: torch.dtype | None
    shape: tuple[int, ...] | None


@dataclasses.dataclass(frozen=True)
class VulkanStaticFactoryConstantReport:
    candidate_count: int
    lowered_count: int
    rejected_count: int
    skipped_count: int
    created_constant_count: int
    reused_constant_count: int
    nodes: tuple[VulkanStaticFactoryConstantNodeReport, ...]


@dataclasses.dataclass(frozen=True)
class VulkanLiftedTensorConstantNodeReport:
    node_name: str
    status: str
    reason: str
    source_attr: str | None
    constant_attr: str | None
    constant_status: str | None
    dtype: torch.dtype | None
    shape: tuple[int, ...] | None


@dataclasses.dataclass(frozen=True)
class VulkanLiftedTensorConstantReport:
    candidate_count: int
    lowered_count: int
    rejected_count: int
    created_constant_count: int
    reused_constant_count: int
    nodes: tuple[VulkanLiftedTensorConstantNodeReport, ...]


@dataclasses.dataclass(frozen=True)
class VulkanStaticIdentityAdvancedIndexNodeReport:
    node_name: str
    status: str
    reason: str
    source_shape: tuple[int, ...] | None
    output_shape: tuple[int, ...] | None
    index_attrs: tuple[str, ...]
    replacement_node_name: str | None


@dataclasses.dataclass(frozen=True)
class VulkanStaticIdentityAdvancedIndexReport:
    candidate_count: int
    lowered_count: int
    rejected_count: int
    skipped_count: int
    nodes: tuple[VulkanStaticIdentityAdvancedIndexNodeReport, ...]


@dataclasses.dataclass(frozen=True)
class VulkanStaticGQARepeatNodeReport:
    node_name: str
    status: str
    reason: str
    source_shape: tuple[int, ...] | None
    output_shape: tuple[int, ...] | None
    repeat_factor: int | None
    replacement_node_name: str | None


@dataclasses.dataclass(frozen=True)
class VulkanStaticGQARepeatReport:
    candidate_count: int
    lowered_count: int
    rejected_count: int
    nodes: tuple[VulkanStaticGQARepeatNodeReport, ...]
    operator: str


@dataclasses.dataclass(frozen=True)
class VulkanGraphTensorPlacementNodeReport:
    source_kind: str
    source_name: str
    dtype: torch.dtype
    storage_type: str
    execution_layout: str
    reason: str


@dataclasses.dataclass(frozen=True)
class VulkanGraphTensorPlacementReport:
    buffer_direct_count: int
    upload_operator: str
    nodes: tuple[VulkanGraphTensorPlacementNodeReport, ...]

    @property
    def buffer_placeholder_names(self) -> tuple[str, ...]:
        return tuple(
            node.source_name
            for node in self.nodes
            if node.source_kind == "placeholder"
        )

    @property
    def buffer_constant_attrs(self) -> tuple[str, ...]:
        return tuple(
            node.source_name
            for node in self.nodes
            if node.source_kind == "constant"
        )


def _get_attr_target(value: Any) -> str | None:
    if isinstance(value, torch.fx.Node) and value.op == "get_attr":
        return str(value.target)
    return None


def _snapshot_tensor(
    state_dict_snapshot: Mapping[str, torch.Tensor],
    target: str,
) -> torch.Tensor | None:
    value = state_dict_snapshot.get(target)
    return value if isinstance(value, torch.Tensor) else None


def _delete_graph_attr_if_unreferenced(
    graph_module: torch.fx.GraphModule,
    target: str,
) -> None:
    if any(
        node.op == "get_attr" and str(node.target) == target
        for node in graph_module.graph.nodes
    ):
        return
    owner: torch.nn.Module = graph_module
    path = target.split(".")
    for name in path[:-1]:
        if not hasattr(owner, name):
            return
        child = getattr(owner, name)
        if not isinstance(child, torch.nn.Module):
            return
        owner = child
    if hasattr(owner, path[-1]):
        delattr(owner, path[-1])


def _context_attr_name(
    weight_attr: str,
    bias_attr: str | None,
    dtype: torch.dtype,
) -> str:
    identity = "\x00".join((weight_attr, bias_attr or "<none>", str(dtype)))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"_vulkan_linear_context_{digest}"


def _conv2d_context_attr_name(
    weight_attr: str,
    bias_attr: str | None,
    dtype: torch.dtype,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> str:
    identity = "\x00".join(
        (
            weight_attr,
            bias_attr or "<none>",
            str(dtype),
            repr(stride),
            repr(padding),
            repr(dilation),
            str(groups),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"_vulkan_conv2d_context_{digest}"


def _layernorm_context_attr_name(
    weight_attr: str,
    bias_attr: str,
    dtype: torch.dtype,
    normalized_shape: tuple[int, ...],
    eps: float,
) -> str:
    identity = "\x00".join(
        (
            weight_attr,
            bias_attr,
            str(dtype),
            repr(normalized_shape),
            repr(eps),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"_vulkan_layernorm_context_{digest}"


def _static_add_layernorm_plan_attr_name(
    context_attr: str,
    add_node_name: str,
    layernorm_node_name: str,
    normalized_shape: tuple[int, ...],
) -> str:
    identity = "\x00".join(
        (
            context_attr,
            add_node_name,
            layernorm_node_name,
            repr(normalized_shape),
            "StaticAddLayernormRegion.v1",
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"_vulkan_static_add_layernorm_plan_{digest}"


def _vulkan_graph_region_plan_attr_name(
    family: str,
    *identity_parts: str,
) -> str:
    identity = "\x00".join((*identity_parts, family, "VulkanGraphRegionPlan.v1"))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"_vulkan_graph_region_plan_{digest}"


def _static_conv2d_relu_plan_attr_name(
    context_attr: str,
    relu_node_name: str,
) -> str:
    identity = "\x00".join(
        (context_attr, relu_node_name, "StaticConv2dReluRegion.v1")
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"_vulkan_static_conv2d_relu_plan_{digest}"


def _snapshot_for_context(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone(memory_format=torch.contiguous_format)


def _node_tensor_dtype(node: torch.fx.Node) -> torch.dtype | None:
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        return value.dtype
    tensor_meta = node.meta.get("tensor_meta")
    dtype = getattr(tensor_meta, "dtype", None)
    return dtype if isinstance(dtype, torch.dtype) else None


def _set_node_tensor_dtype(node: torch.fx.Node, dtype: torch.dtype) -> None:
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        node.meta["val"] = value.to(dtype=dtype)
    tensor_meta = node.meta.get("tensor_meta")
    if hasattr(tensor_meta, "_replace"):
        node.meta["tensor_meta"] = tensor_meta._replace(dtype=dtype)


def _is_static_unsqueeze(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops.aten.unsqueeze.default
        and len(node.args) == 2
        and not node.kwargs
        and isinstance(node.args[0], torch.fx.Node)
        and isinstance(node.args[1], int)
        and not isinstance(node.args[1], bool)
    )


def _node_static_tensor_shape(node: torch.fx.Node) -> tuple[int, ...] | None:
    value = node.meta.get("val")
    shape = value.shape if isinstance(value, torch.Tensor) else None
    if shape is None:
        tensor_meta = node.meta.get("tensor_meta")
        shape = getattr(tensor_meta, "shape", None)
    if shape is None:
        return None
    static_shape: list[int] = []
    for size in shape:
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            return None
        static_shape.append(size)
    return tuple(static_shape)


def _static_shape_argument(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, tuple | list) or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 0
        for size in value
    ):
        return None
    return tuple(value)


def _static_identity_expand_rejection_reason(node: torch.fx.Node) -> str | None:
    if (
        node.op != "call_function"
        or node.target != torch.ops.aten.expand.default
        or len(node.args) != 2
        or node.kwargs
        or not isinstance(node.args[0], torch.fx.Node)
    ):
        return "expand_signature_not_static_default"
    requested_shape = _static_shape_argument(node.args[1])
    if requested_shape is None:
        return "expand_shape_not_static"
    input_shape = _node_static_tensor_shape(node.args[0])
    output_shape = _node_static_tensor_shape(node)
    if input_shape is None or output_shape is None:
        return "expand_tensor_metadata_not_static"
    if requested_shape != input_shape or output_shape != input_shape:
        return "expand_not_identity"
    return None


def _is_metadata_assertion(
    node: torch.fx.Node,
    value: torch.fx.Node,
) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops.aten._assert_tensor_metadata.default
        and len(node.args) == 1
        and node.args[0] is value
        and "dtype" in node.kwargs
        and set(node.kwargs).issubset({"dtype", "device", "layout"})
        and not node.users
    )


def _graph_placeholder_nodes(
    graph_module: torch.fx.GraphModule,
) -> tuple[torch.fx.Node, ...]:
    return tuple(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )


def is_verified_exported_input_guard_call(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
) -> bool:
    placeholders = _graph_placeholder_nodes(graph_module)
    if (
        node.op != "call_module"
        or node.target != "_guards_fn"
        or node.kwargs
        or node.users
        or not placeholders
        or len(node.args) != len(placeholders)
        or any(
            argument is not placeholder
            for argument, placeholder in zip(node.args, placeholders)
        )
    ):
        return False
    try:
        guard_module = graph_module.get_submodule(str(node.target))
    except AttributeError:
        return False
    from torch.export._unlift import GuardsFn

    return type(guard_module) is GuardsFn


def extract_verified_exported_input_guard(
    graph_module: torch.fx.GraphModule,
) -> torch.nn.Module | None:
    guard_nodes = tuple(
        node
        for node in graph_module.graph.nodes
        if is_verified_exported_input_guard_call(graph_module, node)
    )
    if len(guard_nodes) != 1:
        return None
    guard_node = guard_nodes[0]
    guard_module = graph_module.get_submodule(str(guard_node.target))
    graph_module.graph.erase_node(guard_node)
    delattr(graph_module, str(guard_node.target))
    graph_module.graph.lint()
    graph_module.recompile()
    return guard_module


def _is_export_guard_user(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
    value: torch.fx.Node,
) -> bool:
    return (
        is_verified_exported_input_guard_call(graph_module, node)
        and any(argument is value for argument in node.args)
    )


def _input_normalization_report(
    node: torch.fx.Node,
    status: str,
    reason: str,
    placeholder: torch.fx.Node | None = None,
    source_dtype: torch.dtype | None = None,
    target_dtype: torch.dtype | None = None,
    chain: tuple[torch.fx.Node, ...] = (),
) -> VulkanGraphInputNormalizationNodeReport:
    return VulkanGraphInputNormalizationNodeReport(
        node_name=node.name,
        status=status,
        reason=reason,
        placeholder_name=None if placeholder is None else str(placeholder.target),
        source_dtype=source_dtype,
        target_dtype=target_dtype,
        erased_node_name=node.name if status == "lowered" else None,
        chain_node_names=tuple(chain_node.name for chain_node in chain),
    )


def lower_graph_input_dtype_normalizations(
    graph_module: torch.fx.GraphModule,
) -> VulkanGraphInputNormalizationReport:
    graph = graph_module.graph
    reports: list[VulkanGraphInputNormalizationNodeReport] = []
    changed = False
    candidate_count = 0

    for node in tuple(graph.nodes):
        if node.op != "call_function" or node.target not in (
            torch.ops.aten.to.dtype,
            torch.ops.aten.to.device,
        ):
            continue
        candidate_count += 1
        if node.target == torch.ops.aten.to.dtype:
            if len(node.args) != 2 or node.kwargs:
                reports.append(
                    _input_normalization_report(
                        node,
                        "rejected",
                        "to_dtype_signature_not_static_default",
                    )
                )
                continue
            input_node, target_dtype = node.args
            if target_dtype != torch.float32:
                reports.append(
                    _input_normalization_report(
                        node,
                        "rejected",
                        "to_dtype_target_not_supported_floating_dtype",
                        target_dtype=target_dtype
                        if isinstance(target_dtype, torch.dtype)
                        else None,
                    )
                )
                continue
            operation_name = "to_dtype"
            requires_view_chain = True
            lowered_reason = "isolated_int64_placeholder_view_chain_to_float32"
        else:
            if len(node.args) != 3 or node.kwargs:
                reports.append(
                    _input_normalization_report(
                        node,
                        "rejected",
                        "to_device_signature_not_static_default",
                    )
                )
                continue
            input_node, target_device, target_dtype = node.args
            if (
                not isinstance(target_device, torch.device)
                or target_device.type != "cpu"
            ):
                reports.append(
                    _input_normalization_report(
                        node,
                        "rejected",
                        "to_device_target_not_cpu_capture_device",
                        target_dtype=target_dtype
                        if isinstance(target_dtype, torch.dtype)
                        else None,
                    )
                )
                continue
            if target_dtype != torch.bool:
                reports.append(
                    _input_normalization_report(
                        node,
                        "rejected",
                        "to_device_target_not_supported_input_dtype",
                        target_dtype=target_dtype
                        if isinstance(target_dtype, torch.dtype)
                        else None,
                    )
                )
                continue
            operation_name = "to_device"
            requires_view_chain = False
            lowered_reason = "isolated_int64_placeholder_to_bool"
        if not isinstance(input_node, torch.fx.Node):
            reports.append(
                _input_normalization_report(
                    node,
                    "rejected",
                    f"{operation_name}_input_not_graph_node",
                    target_dtype=target_dtype
                    if isinstance(target_dtype, torch.dtype)
                    else None,
                )
            )
            continue

        chain_reversed: list[torch.fx.Node] = []
        current = input_node
        seen_path_nodes: set[torch.fx.Node] = set()
        path_rejection_reason: str | None = None
        while isinstance(current, torch.fx.Node):
            if current in seen_path_nodes:
                path_rejection_reason = "placeholder_path_cycle"
                break
            seen_path_nodes.add(current)
            if _is_static_unsqueeze(current):
                chain_reversed.append(current)
                current = current.args[0]
                continue
            if (
                current.op == "call_function"
                and current.target == torch.ops.aten.expand.default
            ):
                path_rejection_reason = _static_identity_expand_rejection_reason(
                    current
                )
                if path_rejection_reason is not None:
                    break
                chain_reversed.append(current)
                current = current.args[0]
                continue
            break
        if path_rejection_reason is not None:
            reports.append(
                _input_normalization_report(
                    node,
                    "rejected",
                    path_rejection_reason,
                    target_dtype=target_dtype,
                    chain=tuple(reversed(chain_reversed)),
                )
            )
            continue
        if current.op != "placeholder" or (
            requires_view_chain and not chain_reversed
        ):
            reports.append(
                _input_normalization_report(
                    node,
                    "rejected",
                    (
                        "to_dtype_input_not_isolated_unsqueeze_from_placeholder"
                        if operation_name == "to_dtype"
                        else "to_device_input_not_isolated_placeholder"
                    ),
                    target_dtype=target_dtype,
                )
            )
            continue

        placeholder = current
        chain = tuple(reversed(chain_reversed))
        source_dtype = _node_tensor_dtype(placeholder)
        if source_dtype != torch.int64:
            reports.append(
                _input_normalization_report(
                    node,
                    "rejected",
                    "placeholder_source_dtype_not_int64",
                    placeholder,
                    source_dtype,
                    target_dtype,
                    chain,
                )
            )
            continue

        assertions: list[torch.fx.Node] = []
        path = (placeholder, *chain)
        path_isolated = True
        for index, value in enumerate(path):
            expected_user = chain[index] if index < len(chain) else node
            for user in value.users:
                if user is expected_user:
                    continue
                if _is_metadata_assertion(user, value):
                    assertions.append(user)
                    continue
                if value is placeholder and _is_export_guard_user(
                    graph_module, user, value
                ):
                    continue
                path_isolated = False
                break
            if not path_isolated:
                break
        if not path_isolated:
            reports.append(
                _input_normalization_report(
                    node,
                    "rejected",
                    "placeholder_path_has_observable_consumer_or_alias",
                    placeholder,
                    source_dtype,
                    target_dtype,
                    chain,
                )
            )
            continue
        if any(assertion.kwargs["dtype"] != source_dtype for assertion in assertions):
            reports.append(
                _input_normalization_report(
                    node,
                    "rejected",
                    "placeholder_path_metadata_assertion_dtype_mismatch",
                    placeholder,
                    source_dtype,
                    target_dtype,
                    chain,
                )
            )
            continue

        for value in path:
            _set_node_tensor_dtype(value, target_dtype)
        for assertion in assertions:
            assertion.kwargs = {**assertion.kwargs, "dtype": target_dtype}
        normalized_input = chain[-1] if chain else placeholder
        node.replace_all_uses_with(normalized_input)
        graph.erase_node(node)
        reports.append(
            _input_normalization_report(
                node,
                "lowered",
                lowered_reason,
                placeholder,
                source_dtype,
                target_dtype,
                chain,
            )
        )
        changed = True

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

    return VulkanGraphInputNormalizationReport(
        candidate_count=candidate_count,
        lowered_count=sum(report.status == "lowered" for report in reports),
        rejected_count=sum(report.status == "rejected" for report in reports),
        nodes=tuple(reports),
    )


_STATIC_ARANGE_OPERATOR_NAMES = frozenset(
    (
        "aten::arange",
        "aten::arange.start",
        "aten::arange.start_step",
    )
)
_STATIC_FACTORY_EXPRESSION_OPERATOR_NAMES = frozenset(
    (
        "aten::add.Tensor",
        "aten::le.Tensor",
        "aten::new_ones",
        "aten::to.dtype",
        "aten::unsqueeze",
    )
)


def _is_static_arange_target(target: Any) -> bool:
    return (
        isinstance(target, torch._ops.OpOverload)
        and target.name() in _STATIC_ARANGE_OPERATOR_NAMES
    )


def _is_static_factory_value(value: Any) -> bool:
    if value is None or isinstance(
        value,
        (bool, int, float, complex, str, torch.device, torch.dtype, torch.layout),
    ):
        return True
    if isinstance(value, tuple | list):
        return all(_is_static_factory_value(item) for item in value)
    return False


def _static_factory_constant_attr(value: torch.Tensor) -> str:
    fingerprint = hashlib.sha256()
    fingerprint.update(str(value.dtype).encode("utf-8"))
    fingerprint.update(b"\x00")
    fingerprint.update(repr(tuple(value.shape)).encode("utf-8"))
    fingerprint.update(b"\x00")
    fingerprint.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return f"_vulkan_static_factory_constant_{fingerprint.hexdigest()[:16]}"


def _resolve_static_factory_expression_value(
    graph_module: torch.fx.GraphModule,
    value: Any,
) -> tuple[Any, bool, bool]:
    if isinstance(value, torch.fx.Node):
        target = str(value.target)
        if (
            value.op == "get_attr"
            and target.startswith("_vulkan_static_factory_constant_")
            and hasattr(graph_module, target)
        ):
            constant = getattr(graph_module, target)
            if isinstance(constant, torch.Tensor) and constant.device.type == "cpu":
                return constant, True, True
        return None, False, False
    if isinstance(value, tuple | list):
        resolved: list[Any] = []
        contains_constant = False
        for item in value:
            resolved_item, is_static, item_contains_constant = (
                _resolve_static_factory_expression_value(graph_module, item)
            )
            if not is_static:
                return None, False, False
            resolved.append(resolved_item)
            contains_constant = contains_constant or item_contains_constant
        return type(value)(resolved), True, contains_constant
    return value, _is_static_factory_value(value), False


def _contains_static_factory_constant_node(value: Any) -> bool:
    if isinstance(value, torch.fx.Node):
        return value.op == "get_attr" and str(value.target).startswith(
            "_vulkan_static_factory_constant_"
        )
    if isinstance(value, tuple | list):
        return any(_contains_static_factory_constant_node(item) for item in value)
    return False


def lower_static_factory_constants(
    graph_module: torch.fx.GraphModule,
) -> VulkanStaticFactoryConstantReport:
    graph = graph_module.graph
    reports: list[VulkanStaticFactoryConstantNodeReport] = []
    constant_attrs: set[str] = set()
    created_constant_count = 0
    reused_constant_count = 0

    for node in tuple(graph.nodes):
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            continue
        operator_name = node.target.name()
        is_factory = _is_static_arange_target(node.target)
        is_expression = operator_name in _STATIC_FACTORY_EXPRESSION_OPERATOR_NAMES
        if not is_factory and not is_expression:
            continue
        if is_expression and not any(
            _contains_static_factory_constant_node(argument)
            for argument in (*node.args, *node.kwargs.values())
        ):
            continue

        resolved_args: list[Any] = []
        args_are_static = True
        contains_constant = False
        for argument in node.args:
            resolved, is_static, argument_contains_constant = (
                _resolve_static_factory_expression_value(graph_module, argument)
            )
            if not is_static:
                args_are_static = False
                break
            resolved_args.append(resolved)
            contains_constant = contains_constant or argument_contains_constant
        resolved_kwargs: dict[str, Any] = {}
        if args_are_static:
            for name, argument in node.kwargs.items():
                resolved, is_static, argument_contains_constant = (
                    _resolve_static_factory_expression_value(graph_module, argument)
                )
                if not is_static:
                    args_are_static = False
                    break
                resolved_kwargs[name] = resolved
                contains_constant = contains_constant or argument_contains_constant
        if not args_are_static or (is_expression and not contains_constant):
            reports.append(
                VulkanStaticFactoryConstantNodeReport(
                    node_name=node.name,
                    status="skipped",
                    reason="factory_arguments_not_static",
                    operator_name=operator_name,
                    constant_attr=None,
                    constant_status=None,
                    dtype=_node_tensor_dtype(node),
                    shape=None,
                )
            )
            continue

        if is_factory:
            resolved_kwargs["device"] = torch.device("cpu")
            if "pin_memory" in resolved_kwargs:
                resolved_kwargs["pin_memory"] = False
        try:
            with torch.inference_mode():
                value = node.target(*resolved_args, **resolved_kwargs)
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                raise RuntimeError("static factory did not produce a CPU tensor")
            value = value.detach().contiguous()
        except (RuntimeError, TypeError, ValueError) as error:
            reports.append(
                VulkanStaticFactoryConstantNodeReport(
                    node_name=node.name,
                    status="rejected",
                    reason=f"static_factory_evaluation_failed:{type(error).__name__}",
                    operator_name=operator_name,
                    constant_attr=None,
                    constant_status=None,
                    dtype=_node_tensor_dtype(node),
                    shape=None,
                )
            )
            continue

        constant_attr = _static_factory_constant_attr(value)
        constant_attrs.add(constant_attr)
        if hasattr(graph_module, constant_attr):
            constant_status = "reused"
            reused_constant_count += 1
        else:
            graph_module.register_buffer(constant_attr, value, persistent=False)
            constant_status = "created"
            created_constant_count += 1
        with graph.inserting_before(node):
            constant_node = graph.get_attr(constant_attr)
        constant_node.meta = dict(node.meta)
        node.replace_all_uses_with(constant_node)
        graph.erase_node(node)
        reports.append(
            VulkanStaticFactoryConstantNodeReport(
                node_name=node.name,
                status="lowered",
                reason=(
                    "static_arange_graph_owned_constant"
                    if is_factory
                    else "static_factory_expression_graph_owned_constant"
                ),
                operator_name=operator_name,
                constant_attr=constant_attr,
                constant_status=constant_status,
                dtype=value.dtype,
                shape=tuple(value.shape),
            )
        )

    lowered_count = sum(report.status == "lowered" for report in reports)
    if lowered_count:
        graph.eliminate_dead_code()
        for target in constant_attrs:
            _delete_graph_attr_if_unreferenced(graph_module, target)
        graph.lint()
        graph_module.recompile()

    return VulkanStaticFactoryConstantReport(
        candidate_count=len(reports),
        lowered_count=lowered_count,
        rejected_count=sum(report.status == "rejected" for report in reports),
        skipped_count=sum(report.status == "skipped" for report in reports),
        created_constant_count=created_constant_count,
        reused_constant_count=reused_constant_count,
        nodes=tuple(reports),
    )


def _fresh_detach_chain_rejection(
    producer: Any,
    consumer: torch.fx.Node,
) -> str | None:
    if not isinstance(producer, torch.fx.Node):
        return "input_is_not_a_graph_value"
    if producer.op != "call_function" or producer.target not in (
        torch.ops.aten.lift_fresh_copy.default,
        torch.ops.aten.detach.default,
    ):
        return "input_is_not_a_fresh_detach_chain"
    if len(producer.args) != 1 or producer.kwargs:
        return "fresh_chain_node_has_invalid_arguments"
    if len(producer.users) != 1 or consumer not in producer.users:
        return "fresh_chain_value_has_other_users"
    if producer.target == torch.ops.aten.lift_fresh_copy.default:
        return None
    return _fresh_detach_chain_rejection(producer.args[0], producer)


def functionalize_fresh_detach_mutations(
    graph_module: torch.fx.GraphModule,
) -> VulkanFreshDetachFunctionalizationReport:
    reports: list[VulkanFreshDetachFunctionalizationNodeReport] = []
    for node in tuple(graph_module.graph.nodes):
        if (
            node.op != "call_function"
            or node.target != torch.ops.aten.detach_.default
        ):
            continue
        source = node.args[0] if len(node.args) == 1 and not node.kwargs else None
        rejection = _fresh_detach_chain_rejection(source, node)
        if rejection is not None:
            reports.append(
                VulkanFreshDetachFunctionalizationNodeReport(
                    node_name=node.name,
                    status="rejected",
                    reason=rejection,
                    source_node_name=(
                        source.name if isinstance(source, torch.fx.Node) else None
                    ),
                    replacement_target=None,
                )
            )
            continue
        node.target = torch.ops.aten.detach.default
        reports.append(
            VulkanFreshDetachFunctionalizationNodeReport(
                node_name=node.name,
                status="functionalized",
                reason="fresh_single_user_detach_chain",
                source_node_name=source.name,
                replacement_target="aten::detach",
            )
        )

    functionalized_count = sum(
        report.status == "functionalized" for report in reports
    )
    if functionalized_count:
        graph_module.graph.lint()
        graph_module.recompile()
    return VulkanFreshDetachFunctionalizationReport(
        candidate_count=len(reports),
        functionalized_count=functionalized_count,
        rejected_count=len(reports) - functionalized_count,
        nodes=tuple(reports),
    )


def _fresh_single_user_tensor_rejection(
    source: Any,
    consumer: torch.fx.Node,
) -> str | None:
    if not isinstance(source, torch.fx.Node):
        return "input_is_not_a_graph_value"
    if source.op != "call_function" or not isinstance(
        source.target, torch._ops.OpOverload
    ):
        return "source_is_not_an_operator_result"
    schema = source.target._schema
    if schema.is_mutable:
        return "source_operator_is_mutable"
    if len(schema.returns) != 1:
        return "source_return_count_is_not_one"
    result = schema.returns[0]
    if result.alias_info is not None:
        return "source_result_may_alias"
    if not result.type.isSubtypeOf(torch._C.TensorType.get()):
        return "source_result_is_not_a_tensor"
    if len(source.users) != 1 or consumer not in source.users:
        return "fresh_value_has_other_users"
    return None


def functionalize_fresh_relu_mutations(
    graph_module: torch.fx.GraphModule,
) -> VulkanFreshReluFunctionalizationReport:
    reports: list[VulkanFreshReluFunctionalizationNodeReport] = []
    for node in tuple(graph_module.graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.relu_.default:
            continue
        source = node.args[0] if len(node.args) == 1 and not node.kwargs else None
        rejection = _fresh_single_user_tensor_rejection(source, node)
        if rejection is not None:
            reports.append(
                VulkanFreshReluFunctionalizationNodeReport(
                    node_name=node.name,
                    status="rejected",
                    reason=rejection,
                    source_node_name=(
                        source.name if isinstance(source, torch.fx.Node) else None
                    ),
                    source_operator_name=(
                        str(source.target)
                        if isinstance(source, torch.fx.Node)
                        else None
                    ),
                    replacement_target=None,
                )
            )
            continue
        node.target = torch.ops.aten.relu.default
        reports.append(
            VulkanFreshReluFunctionalizationNodeReport(
                node_name=node.name,
                status="functionalized",
                reason="fresh_single_user_non_aliasing_tensor_result",
                source_node_name=source.name,
                source_operator_name=str(source.target),
                replacement_target="aten::relu",
            )
        )

    functionalized_count = sum(
        report.status == "functionalized" for report in reports
    )
    if functionalized_count:
        graph_module.graph.lint()
        graph_module.recompile()
    return VulkanFreshReluFunctionalizationReport(
        candidate_count=len(reports),
        functionalized_count=functionalized_count,
        rejected_count=len(reports) - functionalized_count,
        nodes=tuple(reports),
    )


def _resolve_graph_attr(
    graph_module: torch.fx.GraphModule,
    target: str,
) -> Any:
    value: Any = graph_module
    for name in target.split("."):
        if not hasattr(value, name):
            return None
        value = getattr(value, name)
    return value


def _lifted_tensor_constant_attr(value: torch.Tensor) -> str:
    suffix = _static_factory_constant_attr(value).removeprefix(
        "_vulkan_static_factory_constant_"
    )
    return f"_vulkan_lifted_tensor_constant_{suffix}"


def lower_lifted_tensor_constants(
    graph_module: torch.fx.GraphModule,
) -> VulkanLiftedTensorConstantReport:
    graph = graph_module.graph
    reports: list[VulkanLiftedTensorConstantNodeReport] = []
    removed_source_attrs: set[str] = set()
    created_constant_count = 0
    reused_constant_count = 0

    for node in tuple(graph.nodes):
        if (
            node.op != "call_function"
            or node.target != torch.ops.aten.lift_fresh_copy.default
        ):
            continue
        source = node.args[0] if len(node.args) == 1 and not node.kwargs else None
        source_attr = _get_attr_target(source)
        value = (
            _resolve_graph_attr(graph_module, source_attr)
            if source_attr is not None
            else None
        )
        if (
            source_attr is None
            or not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
        ):
            reports.append(
                VulkanLiftedTensorConstantNodeReport(
                    node_name=node.name,
                    status="rejected",
                    reason="lifted_source_is_not_a_cpu_tensor_attr",
                    source_attr=source_attr,
                    constant_attr=None,
                    constant_status=None,
                    dtype=value.dtype if isinstance(value, torch.Tensor) else None,
                    shape=tuple(value.shape) if isinstance(value, torch.Tensor) else None,
                )
            )
            continue
        try:
            constant = value.detach().clone(memory_format=torch.contiguous_format)
        except RuntimeError:
            reports.append(
                VulkanLiftedTensorConstantNodeReport(
                    node_name=node.name,
                    status="rejected",
                    reason="lifted_source_cannot_be_frozen_contiguously",
                    source_attr=source_attr,
                    constant_attr=None,
                    constant_status=None,
                    dtype=value.dtype,
                    shape=tuple(value.shape),
                )
            )
            continue

        constant_attr = _lifted_tensor_constant_attr(constant)
        if hasattr(graph_module, constant_attr):
            constant_status = "reused"
            reused_constant_count += 1
        else:
            graph_module.register_buffer(constant_attr, constant, persistent=False)
            constant_status = "created"
            created_constant_count += 1
        with graph.inserting_before(node):
            constant_node = graph.get_attr(constant_attr)
        if isinstance(source, torch.fx.Node):
            constant_node.meta = dict(source.meta)
        node.args = (constant_node,)
        removed_source_attrs.add(source_attr)
        reports.append(
            VulkanLiftedTensorConstantNodeReport(
                node_name=node.name,
                status="lowered",
                reason="lifted_tensor_registered_as_graph_owned_constant",
                source_attr=source_attr,
                constant_attr=constant_attr,
                constant_status=constant_status,
                dtype=constant.dtype,
                shape=tuple(constant.shape),
            )
        )

    lowered_count = sum(report.status == "lowered" for report in reports)
    if lowered_count:
        graph.eliminate_dead_code()
        for target in removed_source_attrs:
            _delete_graph_attr_if_unreferenced(graph_module, target)
        graph.lint()
        graph_module.recompile()

    return VulkanLiftedTensorConstantReport(
        candidate_count=len(reports),
        lowered_count=lowered_count,
        rejected_count=sum(report.status == "rejected" for report in reports),
        created_constant_count=created_constant_count,
        reused_constant_count=reused_constant_count,
        nodes=tuple(reports),
    )


def _static_factory_tensor_attr(
    graph_module: torch.fx.GraphModule,
    value: Any,
) -> tuple[str, torch.Tensor] | None:
    if not isinstance(value, torch.fx.Node) or value.op != "get_attr":
        return None
    target = str(value.target)
    if not target.startswith("_vulkan_static_factory_constant_") or not hasattr(
        graph_module, target
    ):
        return None
    tensor = getattr(graph_module, target)
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        return None
    return target, tensor


def _static_identity_advanced_index_offsets(
    source_shape: tuple[int, ...],
    indices: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor | None, str]:
    if not source_shape or not indices:
        return None, "static_advanced_index_requires_nonzero_rank"
    try:
        broadcast_indices = torch.broadcast_tensors(*indices)
    except RuntimeError:
        return None, "static_indices_do_not_broadcast"

    output_shape = tuple(broadcast_indices[0].shape)
    source_numel = math.prod(source_shape)
    if math.prod(output_shape) != source_numel:
        return None, "static_index_output_numel_differs_from_source"

    linear_offsets = torch.zeros(output_shape, dtype=torch.int64)
    stride = source_numel
    for size, index in zip(source_shape, broadcast_indices):
        stride //= size if size else 1
        normalized = index.to(dtype=torch.int64)
        if size:
            normalized = torch.where(normalized < 0, normalized + size, normalized)
            if bool(torch.any((normalized < 0) | (normalized >= size))):
                return None, "static_index_out_of_bounds"
        elif normalized.numel():
            return None, "static_index_into_empty_dimension"
        linear_offsets.add_(normalized * stride)

    expected = torch.arange(source_numel, dtype=torch.int64).reshape(output_shape)
    if not torch.equal(linear_offsets, expected):
        return None, "static_index_is_not_identity_order"
    return linear_offsets, "static_full_rank_identity_advanced_index"


def lower_static_identity_advanced_indices(
    graph_module: torch.fx.GraphModule,
) -> VulkanStaticIdentityAdvancedIndexReport:
    graph = graph_module.graph
    reports: list[VulkanStaticIdentityAdvancedIndexNodeReport] = []
    lowered_attrs: set[str] = set()

    def record(
        node: torch.fx.Node,
        status: str,
        reason: str,
        source_shape: tuple[int, ...] | None,
        output_shape: tuple[int, ...] | None,
        index_attrs: tuple[str, ...] = (),
        replacement_node_name: str | None = None,
    ) -> None:
        reports.append(
            VulkanStaticIdentityAdvancedIndexNodeReport(
                node_name=node.name,
                status=status,
                reason=reason,
                source_shape=source_shape,
                output_shape=output_shape,
                index_attrs=index_attrs,
                replacement_node_name=replacement_node_name,
            )
        )

    for node in tuple(graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.index.Tensor:
            continue
        source = node.args[0] if node.args else None
        source_shape = (
            _node_static_tensor_shape(source)
            if isinstance(source, torch.fx.Node)
            else None
        )
        output_shape = _node_static_tensor_shape(node)
        if (
            len(node.args) != 2
            or node.kwargs
            or not isinstance(source, torch.fx.Node)
            or not isinstance(node.args[1], tuple | list)
        ):
            record(
                node,
                "skipped",
                "unsupported_advanced_index_signature",
                source_shape,
                output_shape,
            )
            continue
        if source_shape is None or output_shape is None:
            record(
                node,
                "skipped",
                "advanced_index_shape_not_static",
                source_shape,
                output_shape,
            )
            continue

        source_value = source.meta.get("val")
        if not isinstance(source_value, torch.Tensor) or not source_value.is_contiguous():
            record(
                node,
                "rejected",
                "advanced_index_source_not_contiguous",
                source_shape,
                output_shape,
            )
            continue

        index_nodes = tuple(node.args[1])
        if len(index_nodes) != len(source_shape):
            record(
                node,
                "skipped",
                "advanced_index_does_not_cover_every_source_dimension",
                source_shape,
                output_shape,
            )
            continue
        resolved = tuple(
            _static_factory_tensor_attr(graph_module, index) for index in index_nodes
        )
        if any(item is None for item in resolved):
            record(
                node,
                "skipped",
                "advanced_indices_are_not_graph_owned_static_constants",
                source_shape,
                output_shape,
            )
            continue
        static_indices = tuple(item for item in resolved if item is not None)
        index_attrs = tuple(item[0] for item in static_indices)
        index_tensors = tuple(item[1] for item in static_indices)
        if any(index.dtype not in (torch.int32, torch.int64) for index in index_tensors):
            record(
                node,
                "rejected",
                "static_advanced_index_dtype_not_integral",
                source_shape,
                output_shape,
                index_attrs,
            )
            continue

        _, reason = _static_identity_advanced_index_offsets(
            source_shape, index_tensors
        )
        if reason != "static_full_rank_identity_advanced_index":
            record(
                node,
                "rejected",
                reason,
                source_shape,
                output_shape,
                index_attrs,
            )
            continue
        broadcast_shape = tuple(torch.broadcast_shapes(*(x.shape for x in index_tensors)))
        if output_shape != broadcast_shape:
            record(
                node,
                "rejected",
                "advanced_index_metadata_shape_mismatch",
                source_shape,
                output_shape,
                index_attrs,
            )
            continue

        with graph.inserting_before(node):
            replacement = graph.call_function(
                torch.ops.aten.view.default,
                (source, list(output_shape)),
            )
        replacement.meta = dict(node.meta)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        lowered_attrs.update(index_attrs)
        record(
            node,
            "lowered",
            reason,
            source_shape,
            output_shape,
            index_attrs,
            replacement.name,
        )

    lowered_count = sum(report.status == "lowered" for report in reports)
    if lowered_count:
        graph.eliminate_dead_code()
        for target in lowered_attrs:
            _delete_graph_attr_if_unreferenced(graph_module, target)
        graph.lint()
        graph_module.recompile()

    return VulkanStaticIdentityAdvancedIndexReport(
        candidate_count=len(reports),
        lowered_count=lowered_count,
        rejected_count=sum(report.status == "rejected" for report in reports),
        skipped_count=sum(report.status == "skipped" for report in reports),
        nodes=tuple(reports),
    )


def lower_static_gqa_repeats(
    graph_module: torch.fx.GraphModule,
) -> VulkanStaticGQARepeatReport:
    graph = graph_module.graph
    reports: list[VulkanStaticGQARepeatNodeReport] = []

    def record(
        node: torch.fx.Node,
        status: str,
        reason: str,
        source_shape: tuple[int, ...] | None,
        output_shape: tuple[int, ...] | None,
        repeat_factor: int | None,
        replacement_node_name: str | None = None,
    ) -> None:
        reports.append(
            VulkanStaticGQARepeatNodeReport(
                node_name=node.name,
                status=status,
                reason=reason,
                source_shape=source_shape,
                output_shape=output_shape,
                repeat_factor=repeat_factor,
                replacement_node_name=replacement_node_name,
            )
        )

    for node in tuple(graph.nodes):
        if (
            node.op != "call_function"
            or node.target != torch.ops.aten.reshape.default
            or len(node.args) != 2
            or node.kwargs
        ):
            continue
        expand = node.args[0]
        if not (
            isinstance(expand, torch.fx.Node)
            and expand.op == "call_function"
            and expand.target == torch.ops.aten.expand.default
            and len(expand.args) == 2
            and not expand.kwargs
        ):
            continue
        expanded_shape = _node_static_tensor_shape(expand)
        if expanded_shape is None or len(expanded_shape) != 5:
            continue
        requested_expanded_shape = _static_shape_argument(expand.args[1])
        requested_output_shape = _static_shape_argument(node.args[1])

        unsqueeze = expand.args[0]
        source = (
            unsqueeze.args[0]
            if isinstance(unsqueeze, torch.fx.Node)
            and _is_static_unsqueeze(unsqueeze)
            else None
        )
        source_shape = (
            _node_static_tensor_shape(source)
            if isinstance(source, torch.fx.Node)
            else None
        )
        output_shape = _node_static_tensor_shape(node)
        repeat_factor = expanded_shape[2]
        if (
            not isinstance(source, torch.fx.Node)
            or unsqueeze.args[1] != 2
            or len(unsqueeze.users) != 1
            or len(expand.users) != 1
            or source_shape is None
            or len(source_shape) != 4
            or output_shape is None
        ):
            record(
                node,
                "rejected",
                "gqa_repeat_graph_structure_mismatch",
                source_shape,
                output_shape,
                repeat_factor,
            )
            continue
        if requested_expanded_shape is None or requested_output_shape is None:
            record(
                node,
                "rejected",
                "gqa_repeat_shape_arguments_not_static",
                source_shape,
                output_shape,
                repeat_factor,
            )
            continue
        expected_expanded_shape = (
            source_shape[0],
            source_shape[1],
            repeat_factor,
            source_shape[2],
            source_shape[3],
        )
        expected_output_shape = (
            source_shape[0],
            source_shape[1] * repeat_factor,
            source_shape[2],
            source_shape[3],
        )
        if (
            repeat_factor <= 1
            or requested_expanded_shape != expanded_shape
            or requested_output_shape != output_shape
            or expanded_shape != expected_expanded_shape
            or output_shape != expected_output_shape
        ):
            record(
                node,
                "rejected",
                "gqa_repeat_shape_contract_mismatch",
                source_shape,
                output_shape,
                repeat_factor,
            )
            continue
        if _node_tensor_dtype(source) != torch.float32:
            record(
                node,
                "rejected",
                "gqa_repeat_requires_float32",
                source_shape,
                output_shape,
                repeat_factor,
            )
            continue

        with graph.inserting_before(unsqueeze):
            replacement = graph.call_function(
                torch.ops.vulkan_prepack.repeat_attention_heads_for_gqa.default,
                (source, repeat_factor),
            )
        replacement.meta = dict(node.meta)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        graph.erase_node(expand)
        graph.erase_node(unsqueeze)
        record(
            node,
            "lowered",
            "static_gqa_repeat_kernel_family",
            source_shape,
            output_shape,
            repeat_factor,
            replacement.name,
        )

    lowered_count = sum(report.status == "lowered" for report in reports)
    if lowered_count:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

    return VulkanStaticGQARepeatReport(
        candidate_count=len(reports),
        lowered_count=lowered_count,
        rejected_count=sum(report.status == "rejected" for report in reports),
        nodes=tuple(reports),
        operator="vulkan_prepack::repeat_attention_heads_for_gqa",
    )


def plan_graph_tensor_placements(
    graph_module: torch.fx.GraphModule,
    input_normalization: VulkanGraphInputNormalizationReport,
    static_factory_constants: VulkanStaticFactoryConstantReport,
    lifted_tensor_constants: VulkanLiftedTensorConstantReport,
) -> VulkanGraphTensorPlacementReport:
    normalized_placeholders = {
        node.placeholder_name
        for node in input_normalization.nodes
        if node.status == "lowered" and node.placeholder_name is not None
    }
    reports: list[VulkanGraphTensorPlacementNodeReport] = []
    for placeholder in _graph_placeholder_nodes(graph_module):
        if _node_tensor_dtype(placeholder) != torch.bool:
            continue
        placeholder_name = str(placeholder.target)
        reports.append(
            VulkanGraphTensorPlacementNodeReport(
                source_kind="placeholder",
                source_name=placeholder_name,
                dtype=torch.bool,
                storage_type="buffer",
                execution_layout="buffer_direct",
                reason=(
                    "normalized_bool_graph_input"
                    if placeholder_name in normalized_placeholders
                    else "captured_bool_graph_input"
                ),
            )
        )

    referenced_attrs = {
        str(node.target)
        for node in graph_module.graph.nodes
        if node.op == "get_attr"
    }
    seen_attrs: set[str] = set()
    constant_candidates: list[
        tuple[
            VulkanStaticFactoryConstantNodeReport
            | VulkanLiftedTensorConstantNodeReport,
            str,
        ]
    ] = [
        (node, "graph_owned_bool_factory_constant")
        for node in static_factory_constants.nodes
    ]
    constant_candidates.extend(
        (node, "graph_owned_bool_lifted_constant")
        for node in lifted_tensor_constants.nodes
    )
    for node, reason in constant_candidates:
        if (
            node.status != "lowered"
            or node.dtype != torch.bool
            or node.constant_attr is None
            or node.constant_attr not in referenced_attrs
            or node.constant_attr in seen_attrs
        ):
            continue
        seen_attrs.add(node.constant_attr)
        reports.append(
            VulkanGraphTensorPlacementNodeReport(
                source_kind="constant",
                source_name=node.constant_attr,
                dtype=torch.bool,
                storage_type="buffer",
                execution_layout="buffer_direct",
                reason=reason,
            )
        )

    return VulkanGraphTensorPlacementReport(
        buffer_direct_count=len(reports),
        upload_operator="vulkan_prepack::upload_graph_tensor_to_buffer",
        nodes=tuple(reports),
    )


def _reject(
    node: torch.fx.Node,
    reason: str,
    weight_attr: str | None,
    bias_attr: str | None,
) -> VulkanLinearLoweringNodeReport:
    return VulkanLinearLoweringNodeReport(
        node_name=node.name,
        status="rejected",
        reason=reason,
        weight_attr=weight_attr,
        bias_attr=bias_attr,
        context_attr=None,
        context_status=None,
    )


def _reject_conv2d(
    node: torch.fx.Node,
    reason: str,
    weight_attr: str | None,
    bias_attr: str | None,
) -> VulkanConv2dLoweringNodeReport:
    return VulkanConv2dLoweringNodeReport(
        node_name=node.name,
        status="rejected",
        reason=reason,
        weight_attr=weight_attr,
        bias_attr=bias_attr,
        context_attr=None,
        context_status=None,
    )


def _static_int_pair(
    value: Any,
    name: str,
    minimum: int,
) -> tuple[tuple[int, int] | None, str | None]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None, f"{name}_not_static_int_pair"
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        return None, f"{name}_not_static_int_pair"
    result = (int(value[0]), int(value[1]))
    if any(item < minimum for item in result):
        return None, f"{name}_outside_supported_range"
    return result, None


def _static_positive_int_sequence(
    value: Any,
    name: str,
) -> tuple[tuple[int, ...] | None, str | None]:
    if not isinstance(value, (list, tuple)) or not value:
        return None, f"{name}_not_static_positive_int_sequence"
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        return None, f"{name}_not_static_positive_int_sequence"
    result = tuple(int(item) for item in value)
    if any(item <= 0 for item in result):
        return None, f"{name}_outside_supported_range"
    return result, None


def _gelu_approximate_mode(node: torch.fx.Node) -> str | None:
    if not (
        node.op == "call_function"
        and node.target == torch.ops.aten.gelu.default
        and len(node.args) == 1
    ):
        return None
    if node.kwargs == {"approximate": "tanh"}:
        return "tanh"
    if node.kwargs in ({}, {"approximate": "none"}):
        return "none"
    return None


def _is_relu(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops.aten.relu.default
        and len(node.args) == 1
        and not node.kwargs
    )


def _is_add_tensor_alpha_one(node: torch.fx.Node) -> tuple[bool, str]:
    if node.op != "call_function" or node.target != torch.ops.aten.add.Tensor:
        return False, "input_not_aten_add_tensor"
    if len(node.args) not in (2, 3) or set(node.kwargs).difference({"alpha"}):
        return False, "unsupported_add_tensor_signature"
    if len(node.args) == 3 and "alpha" in node.kwargs:
        return False, "unsupported_add_tensor_signature"
    alpha = node.args[2] if len(node.args) == 3 else node.kwargs.get("alpha", 1)
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        return False, "add_alpha_not_static_one"
    if float(alpha) != 1.0:
        return False, "add_alpha_not_one"
    return True, ""


def _graph_owned_linear_context_attr(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
) -> str | None:
    if (
        node.op != "call_function"
        or node.target != torch.ops.vulkan_prepack.run_linear_context.default
        or len(node.args) != 2
        or node.kwargs
    ):
        return None
    context_attr = _get_attr_target(node.args[1])
    if context_attr is None or not context_attr.startswith(
        "_vulkan_linear_context_"
    ):
        return None
    return context_attr if hasattr(graph_module, context_attr) else None


def _graph_owned_conv2d_context_attr(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
) -> str | None:
    if (
        node.op != "call_function"
        or node.target != torch.ops.vulkan_prepack.run_conv2d_context.default
        or len(node.args) != 2
        or node.kwargs
    ):
        return None
    context_attr = _get_attr_target(node.args[1])
    if context_attr is None or not context_attr.startswith(
        "_vulkan_conv2d_context_"
    ):
        return None
    return context_attr if hasattr(graph_module, context_attr) else None


def _graph_owned_layernorm_context_attr(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
) -> str | None:
    if (
        node.op != "call_function"
        or node.target != torch.ops.vulkan_prepack.run_layernorm_context.default
        or len(node.args) != 3
        or node.kwargs
    ):
        return None
    context_attr = _get_attr_target(node.args[2])
    if context_attr is None or not context_attr.startswith(
        "_vulkan_layernorm_context_"
    ):
        return None
    if not hasattr(graph_module, context_attr):
        return None
    return (
        context_attr
        if isinstance(getattr(graph_module, context_attr), torch.ScriptObject)
        else None
    )


def _skip_layernorm(
    node: torch.fx.Node,
    reason: str,
    weight_attr: str | None,
    bias_attr: str | None,
    normalized_shape: tuple[int, ...] | None,
    eps: float | None,
) -> VulkanLayernormLoweringNodeReport:
    return VulkanLayernormLoweringNodeReport(
        node_name=node.name,
        status="skipped",
        reason=reason,
        weight_attr=weight_attr,
        bias_attr=bias_attr,
        normalized_shape=normalized_shape,
        eps=eps,
        context_attr=None,
        context_status=None,
    )


def lower_static_linear_to_vulkan_contexts(
    graph_module: torch.fx.GraphModule,
    state_dict_snapshot: Mapping[str, torch.Tensor],
) -> VulkanLinearLoweringReport:
    graph = graph_module.graph
    reports: list[VulkanLinearLoweringNodeReport] = []
    context_attrs: dict[tuple[str, str | None, torch.dtype], str] = {}
    linear_node_count = 0
    created_context_count = 0
    reused_context_count = 0

    for node in tuple(graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.linear.default:
            continue

        linear_node_count += 1
        if len(node.args) not in (2, 3) or node.kwargs:
            reports.append(_reject(node, "unsupported_linear_signature", None, None))
            continue

        _, weight_arg = node.args[:2]
        bias_arg = None if len(node.args) == 2 else node.args[2]
        weight_attr = _get_attr_target(weight_arg)
        bias_attr = None if bias_arg is None else _get_attr_target(bias_arg)
        if weight_attr is None:
            reports.append(_reject(node, "weight_not_static_get_attr", None, bias_attr))
            continue
        if bias_arg is not None and bias_attr is None:
            reports.append(
                _reject(node, "bias_not_static_get_attr", weight_attr, None)
            )
            continue
        weight = _snapshot_tensor(state_dict_snapshot, weight_attr)
        if weight is None:
            reports.append(
                _reject(
                    node,
                    "weight_missing_from_cpu_snapshot",
                    weight_attr,
                    bias_attr,
                )
            )
            continue
        if weight.device.type != "cpu":
            reports.append(
                _reject(node, "weight_snapshot_not_cpu", weight_attr, bias_attr)
            )
            continue
        if weight.dim() != 2:
            reports.append(
                _reject(node, "weight_snapshot_rank_not_two", weight_attr, bias_attr)
            )
            continue
        if not torch.is_floating_point(weight):
            reports.append(
                _reject(node, "weight_snapshot_not_floating", weight_attr, bias_attr)
            )
            continue

        bias = None
        if bias_attr is not None:
            bias = _snapshot_tensor(state_dict_snapshot, bias_attr)
            if bias is None:
                reports.append(
                    _reject(
                        node,
                        "bias_missing_from_cpu_snapshot",
                        weight_attr,
                        bias_attr,
                    )
                )
                continue
            if bias.device.type != "cpu":
                reports.append(
                    _reject(node, "bias_snapshot_not_cpu", weight_attr, bias_attr)
                )
                continue
            if bias.dim() != 1:
                reports.append(
                    _reject(node, "bias_snapshot_rank_not_one", weight_attr, bias_attr)
                )
                continue
            if not torch.is_floating_point(bias):
                reports.append(
                    _reject(node, "bias_snapshot_not_floating", weight_attr, bias_attr)
                )
                continue
            if bias.dtype != weight.dtype:
                reports.append(
                    _reject(node, "bias_weight_dtype_mismatch", weight_attr, bias_attr)
                )
                continue
            if bias.numel() != weight.size(0):
                reports.append(
                    _reject(
                        node,
                        "bias_output_features_mismatch",
                        weight_attr,
                        bias_attr,
                    )
                )
                continue

        context_key = (weight_attr, bias_attr, weight.dtype)
        context_attr = context_attrs.get(context_key)
        context_status = "reused" if context_attr is not None else "created"
        if context_attr is None:
            context_attr = _context_attr_name(weight_attr, bias_attr, weight.dtype)
            if hasattr(graph_module, context_attr):
                reports.append(
                    _reject(
                        node,
                        "deterministic_context_attribute_collision",
                        weight_attr,
                        bias_attr,
                    )
                )
                continue
            try:
                context = torch.ops.vulkan_prepack.create_graph_linear_context.default(
                    _snapshot_for_context(weight).t().contiguous(),
                    None if bias is None else _snapshot_for_context(bias),
                )
                setattr(graph_module, context_attr, context)
            except (RuntimeError, TypeError) as error:
                reports.append(
                    _reject(
                        node,
                        f"linear_context_creation_failed:{type(error).__name__}",
                        weight_attr,
                        bias_attr,
                    )
                )
                continue
            context_attrs[context_key] = context_attr
            created_context_count += 1
        else:
            reused_context_count += 1

        with graph.inserting_before(node):
            context_node = graph.create_node("get_attr", context_attr, (), {})
            lowered_node = graph.call_function(
                torch.ops.vulkan_prepack.run_linear_context.default,
                args=(node.args[0], context_node),
            )
        lowered_node.meta = dict(node.meta)
        node.replace_all_uses_with(lowered_node)
        graph.erase_node(node)
        reports.append(
            VulkanLinearLoweringNodeReport(
                node_name=node.name,
                status="lowered",
                reason="static_cpu_snapshot_weight_and_bias",
                weight_attr=weight_attr,
                bias_attr=bias_attr,
                context_attr=context_attr,
                context_status=context_status,
            )
        )

    if any(report.status == "lowered" for report in reports):
        graph.eliminate_dead_code()
        graph_module.delete_all_unused_submodules()
        graph.lint()
        graph_module.recompile()

    return VulkanLinearLoweringReport(
        linear_node_count=linear_node_count,
        lowered_count=sum(report.status == "lowered" for report in reports),
        rejected_count=sum(report.status == "rejected" for report in reports),
        created_context_count=created_context_count,
        reused_context_count=reused_context_count,
        context_factory="vulkan_prepack::create_graph_linear_context",
        nodes=tuple(reports),
    )


def lower_static_linear_gelu_regions(
    graph_module: torch.fx.GraphModule,
) -> VulkanStaticLinearGeluRegionReport:
    graph = graph_module.graph
    reports: list[VulkanStaticLinearGeluRegionNodeReport] = []
    removed_context_attrs: set[str] = set()
    candidate_count = 0
    lowered_count = 0
    rejected_count = 0
    skipped_count = 0

    for gelu_node in tuple(graph.nodes):
        approximate = _gelu_approximate_mode(gelu_node)
        if approximate is None:
            continue
        region_family = f"linear_gelu_{approximate}"
        linear_node = gelu_node.args[0]
        if not isinstance(linear_node, torch.fx.Node):
            continue
        context_attr = _graph_owned_linear_context_attr(
            graph_module, linear_node
        )
        if context_attr is None:
            continue
        context_node = linear_node.args[1]
        context_attr_reference_count = sum(
            node.op == "get_attr" and str(node.target) == context_attr
            for node in graph.nodes
        )
        if len(linear_node.users) != 1:
            reports.append(
                VulkanStaticLinearGeluRegionNodeReport(
                    node_name=gelu_node.name,
                    status="skipped",
                    reason="linear_output_has_multiple_users",
                    linear_node_name=linear_node.name,
                    context_attr=context_attr,
                    plan_attr=None,
                    program_name=None,
                    program_version=None,
                    instruction_count=0,
                    input_ssa=None,
                    output_ssa=None,
                    input_use_count=None,
                    input_last_use=None,
                    static_context_slot=None,
                    direct_transition_only=None,
                    replay_state_empty=None,
                    region_family=region_family,
                )
            )
            skipped_count += 1
            continue
        if context_attr_reference_count != 1:
            reports.append(
                VulkanStaticLinearGeluRegionNodeReport(
                    node_name=gelu_node.name,
                    status="skipped",
                    reason="context_attr_has_multiple_references",
                    linear_node_name=linear_node.name,
                    context_attr=context_attr,
                    plan_attr=None,
                    program_name=None,
                    program_version=None,
                    instruction_count=0,
                    input_ssa=None,
                    output_ssa=None,
                    input_use_count=None,
                    input_last_use=None,
                    static_context_slot=None,
                    direct_transition_only=None,
                    replay_state_empty=None,
                    region_family=region_family,
                )
            )
            skipped_count += 1
            continue

        candidate_count += 1
        plan_attr = _vulkan_graph_region_plan_attr_name(
            region_family, context_attr, gelu_node.name
        )
        if hasattr(graph_module, plan_attr):
            reports.append(
                VulkanStaticLinearGeluRegionNodeReport(
                    node_name=gelu_node.name,
                    status="rejected",
                    reason="deterministic_plan_attribute_collision",
                    linear_node_name=linear_node.name,
                    context_attr=context_attr,
                    plan_attr=plan_attr,
                    program_name="VulkanGraphRegionPlan",
                    program_version="v1",
                    instruction_count=2,
                    input_ssa=0,
                    output_ssa=2,
                    input_use_count=1,
                    input_last_use=0,
                    static_context_slot=0,
                    direct_transition_only=True,
                    replay_state_empty=True,
                    region_family=region_family,
                    intermediate_ssa=1,
                    intermediate_use_count=1,
                    intermediate_last_use=1,
                )
            )
            rejected_count += 1
            continue
        try:
            plan = (
                torch.ops.vulkan_prepack.create_vulkan_graph_region_plan_linear_gelu.default(
                    getattr(graph_module, context_attr)
                )
                if approximate == "tanh"
                else torch.ops.vulkan_prepack.create_vulkan_graph_region_plan_linear_gelu_none.default(
                    getattr(graph_module, context_attr)
                )
            )
            setattr(graph_module, plan_attr, plan)
        except (RuntimeError, TypeError, AttributeError) as error:
            reports.append(
                VulkanStaticLinearGeluRegionNodeReport(
                    node_name=gelu_node.name,
                    status="rejected",
                    reason=(
                        "vulkan_graph_region_plan_creation_failed:"
                        f"{type(error).__name__}"
                    ),
                    linear_node_name=linear_node.name,
                    context_attr=context_attr,
                    plan_attr=plan_attr,
                    program_name="VulkanGraphRegionPlan",
                    program_version="v1",
                    instruction_count=2,
                    input_ssa=0,
                    output_ssa=2,
                    input_use_count=1,
                    input_last_use=0,
                    static_context_slot=0,
                    direct_transition_only=True,
                    replay_state_empty=True,
                    region_family=region_family,
                    intermediate_ssa=1,
                    intermediate_use_count=1,
                    intermediate_last_use=1,
                )
            )
            rejected_count += 1
            continue

        with graph.inserting_before(gelu_node):
            plan_node = graph.create_node("get_attr", plan_attr, (), {})
            region_node = graph.call_function(
                torch.ops.vulkan_prepack.run_vulkan_graph_region_plan.default,
                args=([linear_node.args[0]], plan_node),
            )
            lowered_node = graph.call_function(operator.getitem, (region_node, 0))
        lowered_node.meta = dict(gelu_node.meta)
        gelu_node.replace_all_uses_with(lowered_node)
        graph.erase_node(gelu_node)
        graph.erase_node(linear_node)
        graph.erase_node(context_node)
        removed_context_attrs.add(context_attr)
        reports.append(
            VulkanStaticLinearGeluRegionNodeReport(
                node_name=lowered_node.name,
                status="lowered",
                reason=f"graph_owned_static_linear_{approximate}_gelu",
                linear_node_name=linear_node.name,
                context_attr=context_attr,
                plan_attr=plan_attr,
                program_name="VulkanGraphRegionPlan",
                program_version="v1",
                instruction_count=2,
                input_ssa=0,
                output_ssa=2,
                input_use_count=1,
                input_last_use=0,
                static_context_slot=0,
                direct_transition_only=True,
                replay_state_empty=True,
                region_family=region_family,
                intermediate_ssa=1,
                intermediate_use_count=1,
                intermediate_last_use=1,
            )
        )
        lowered_count += 1

    if lowered_count:
        graph.eliminate_dead_code()
        for context_attr in removed_context_attrs:
            if hasattr(graph_module, context_attr):
                delattr(graph_module, context_attr)
        graph_module.delete_all_unused_submodules()
        graph.lint()
        graph_module.recompile()

    region_families = {
        node.region_family for node in reports if node.region_family is not None
    }
    if region_families == {"linear_gelu_none"}:
        plan_factory = (
            "vulkan_prepack::create_vulkan_graph_region_plan_linear_gelu_none"
        )
    elif region_families == {"linear_gelu_tanh"} or not region_families:
        plan_factory = "vulkan_prepack::create_vulkan_graph_region_plan_linear_gelu"
    else:
        plan_factory = (
            "vulkan_prepack::create_vulkan_graph_region_plan_linear_gelu{,_none}"
        )

    return VulkanStaticLinearGeluRegionReport(
        candidate_count=candidate_count,
        lowered_count=lowered_count,
        rejected_count=rejected_count,
        skipped_count=skipped_count,
        plan_factory=plan_factory,
        nodes=tuple(reports),
    )


def lower_static_conv2d_to_vulkan_contexts(
    graph_module: torch.fx.GraphModule,
    state_dict_snapshot: Mapping[str, torch.Tensor],
) -> VulkanConv2dLoweringReport:
    graph = graph_module.graph
    reports: list[VulkanConv2dLoweringNodeReport] = []
    context_attrs: dict[
        tuple[
            str,
            str | None,
            torch.dtype,
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
            int,
        ],
        str,
    ] = {}
    conv2d_node_count = 0
    created_context_count = 0
    reused_context_count = 0

    for node in tuple(graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.conv2d.default:
            continue

        conv2d_node_count += 1
        optional_argument_names = ("bias", "stride", "padding", "dilation", "groups")
        if (
            len(node.args) < 2
            or len(node.args) > 7
            or set(node.kwargs).difference(optional_argument_names)
        ):
            reports.append(
                _reject_conv2d(node, "unsupported_conv2d_signature", None, None)
            )
            continue
        positional_names = (
            "input",
            "weight",
            "bias",
            "stride",
            "padding",
            "dilation",
            "groups",
        )
        if any(
            name in node.kwargs and name in positional_names[: len(node.args)]
            for name in optional_argument_names
        ):
            reports.append(
                _reject_conv2d(node, "unsupported_conv2d_signature", None, None)
            )
            continue
        input_arg, weight_arg = node.args[:2]
        bias_arg = (
            node.args[2]
            if len(node.args) > 2
            else node.kwargs.get("bias", None)
        )
        stride_arg = (
            node.args[3]
            if len(node.args) > 3
            else node.kwargs.get("stride", (1, 1))
        )
        padding_arg = (
            node.args[4]
            if len(node.args) > 4
            else node.kwargs.get("padding", (0, 0))
        )
        dilation_arg = (
            node.args[5]
            if len(node.args) > 5
            else node.kwargs.get("dilation", (1, 1))
        )
        groups_arg = (
            node.args[6]
            if len(node.args) > 6
            else node.kwargs.get("groups", 1)
        )
        weight_attr = _get_attr_target(weight_arg)
        bias_attr = None if bias_arg is None else _get_attr_target(bias_arg)
        if weight_attr is None:
            reports.append(
                _reject_conv2d(node, "weight_not_static_get_attr", None, bias_attr)
            )
            continue
        if bias_arg is not None and bias_attr is None:
            reports.append(
                _reject_conv2d(node, "bias_not_static_get_attr", weight_attr, None)
            )
            continue
        input_meta = (
            input_arg.meta.get("val")
            if isinstance(input_arg, torch.fx.Node)
            else None
        )
        if not isinstance(input_meta, torch.Tensor):
            reports.append(
                _reject_conv2d(
                    node,
                    "input_metadata_not_tensor",
                    weight_attr,
                    bias_attr,
                )
            )
            continue
        if input_meta.dim() != 4:
            reports.append(
                _reject_conv2d(
                    node,
                    "input_metadata_rank_not_four",
                    weight_attr,
                    bias_attr,
                )
            )
            continue
        if input_meta.dtype != torch.float32:
            reports.append(
                _reject_conv2d(
                    node,
                    "input_metadata_dtype_not_float32",
                    weight_attr,
                    bias_attr,
                )
            )
            continue
        stride, reason = _static_int_pair(stride_arg, "stride", 1)
        if reason:
            reports.append(_reject_conv2d(node, reason, weight_attr, bias_attr))
            continue
        padding, reason = _static_int_pair(padding_arg, "padding", 0)
        if reason:
            reports.append(_reject_conv2d(node, reason, weight_attr, bias_attr))
            continue
        dilation, reason = _static_int_pair(dilation_arg, "dilation", 1)
        if reason:
            reports.append(_reject_conv2d(node, reason, weight_attr, bias_attr))
            continue
        if not isinstance(groups_arg, int) or isinstance(groups_arg, bool):
            reports.append(
                _reject_conv2d(
                    node,
                    "groups_not_static_positive_int",
                    weight_attr,
                    bias_attr,
                )
            )
            continue
        groups = int(groups_arg)
        if groups <= 0:
            reports.append(
                _reject_conv2d(
                    node,
                    "groups_not_static_positive_int",
                    weight_attr,
                    bias_attr,
                )
            )
            continue

        weight = _snapshot_tensor(state_dict_snapshot, weight_attr)
        if weight is None:
            reports.append(
                _reject_conv2d(
                    node,
                    "weight_missing_from_cpu_snapshot",
                    weight_attr,
                    bias_attr,
                )
            )
            continue
        if weight.device.type != "cpu":
            reports.append(
                _reject_conv2d(
                    node,
                    "weight_snapshot_not_cpu",
                    weight_attr,
                    bias_attr,
                )
            )
            continue
        if weight.dim() != 4:
            reports.append(
                _reject_conv2d(
                    node,
                    "weight_snapshot_rank_not_four",
                    weight_attr,
                    bias_attr,
                )
            )
            continue
        if weight.dtype != torch.float32:
            reports.append(
                _reject_conv2d(
                    node,
                    "weight_snapshot_dtype_not_float32",
                    weight_attr,
                    bias_attr,
                )
            )
            continue
        if any(size <= 0 for size in weight.shape):
            reports.append(
                _reject_conv2d(
                    node,
                    "weight_snapshot_has_empty_dimension",
                    weight_attr,
                    bias_attr,
                )
            )
            continue
        if weight.size(0) % groups != 0:
            reports.append(
                _reject_conv2d(
                    node,
                    "weight_output_channels_not_divisible_by_groups",
                    weight_attr,
                    bias_attr,
                )
            )
            continue
        input_channels = input_meta.size(1)
        if (
            not isinstance(input_channels, int)
            or input_channels != weight.size(1) * groups
        ):
            reports.append(
                _reject_conv2d(
                    node,
                    "input_channel_group_relation_not_static_or_invalid",
                    weight_attr,
                    bias_attr,
                )
            )
            continue

        bias = None
        if bias_attr is not None:
            bias = _snapshot_tensor(state_dict_snapshot, bias_attr)
            if bias is None:
                reports.append(
                    _reject_conv2d(
                        node,
                        "bias_missing_from_cpu_snapshot",
                        weight_attr,
                        bias_attr,
                    )
                )
                continue
            if bias.device.type != "cpu":
                reports.append(
                    _reject_conv2d(
                        node,
                        "bias_snapshot_not_cpu",
                        weight_attr,
                        bias_attr,
                    )
                )
                continue
            if bias.dim() != 1:
                reports.append(
                    _reject_conv2d(
                        node,
                        "bias_snapshot_rank_not_one",
                        weight_attr,
                        bias_attr,
                    )
                )
                continue
            if bias.dtype != torch.float32:
                reports.append(
                    _reject_conv2d(
                        node,
                        "bias_snapshot_dtype_not_float32",
                        weight_attr,
                        bias_attr,
                    )
                )
                continue
            if bias.dtype != weight.dtype:
                reports.append(
                    _reject_conv2d(
                        node,
                        "bias_weight_dtype_mismatch",
                        weight_attr,
                        bias_attr,
                    )
                )
                continue
            if bias.numel() != weight.size(0):
                reports.append(
                    _reject_conv2d(
                        node,
                        "bias_output_channels_mismatch",
                        weight_attr,
                        bias_attr,
                    )
                )
                continue

        context_key = (
            weight_attr,
            bias_attr,
            weight.dtype,
            stride,
            padding,
            dilation,
            groups,
        )
        context_attr = context_attrs.get(context_key)
        context_status = "reused" if context_attr is not None else "created"
        if context_attr is None:
            context_attr = _conv2d_context_attr_name(
                weight_attr,
                bias_attr,
                weight.dtype,
                stride,
                padding,
                dilation,
                groups,
            )
            if hasattr(graph_module, context_attr):
                reports.append(
                    _reject_conv2d(
                        node,
                        "deterministic_context_attribute_collision",
                        weight_attr,
                        bias_attr,
                    )
                )
                continue
            try:
                context = torch.ops.vulkan_prepack.create_graph_conv2d_context.default(
                    _snapshot_for_context(weight),
                    None if bias is None else _snapshot_for_context(bias),
                    list(stride),
                    list(padding),
                    list(dilation),
                    groups,
                )
                setattr(graph_module, context_attr, context)
            except (RuntimeError, TypeError) as error:
                reports.append(
                    _reject_conv2d(
                        node,
                        f"conv2d_context_creation_failed:{type(error).__name__}",
                        weight_attr,
                        bias_attr,
                    )
                )
                continue
            context_attrs[context_key] = context_attr
            created_context_count += 1
        else:
            reused_context_count += 1

        with graph.inserting_before(node):
            context_node = graph.create_node("get_attr", context_attr, (), {})
            lowered_node = graph.call_function(
                torch.ops.vulkan_prepack.run_conv2d_context.default,
                args=(node.args[0], context_node),
            )
        lowered_node.meta = dict(node.meta)
        node.replace_all_uses_with(lowered_node)
        graph.erase_node(node)
        reports.append(
            VulkanConv2dLoweringNodeReport(
                node_name=node.name,
                status="lowered",
                reason="static_cpu_snapshot_weight_and_bias",
                weight_attr=weight_attr,
                bias_attr=bias_attr,
                context_attr=context_attr,
                context_status=context_status,
            )
        )

    if any(report.status == "lowered" for report in reports):
        graph.eliminate_dead_code()
        graph_module.delete_all_unused_submodules()
        graph.lint()
        graph_module.recompile()

    return VulkanConv2dLoweringReport(
        conv2d_node_count=conv2d_node_count,
        lowered_count=sum(report.status == "lowered" for report in reports),
        rejected_count=sum(report.status == "rejected" for report in reports),
        created_context_count=created_context_count,
        reused_context_count=reused_context_count,
        context_factory="vulkan_prepack::create_graph_conv2d_context",
        nodes=tuple(reports),
    )


def lower_static_layernorm_to_vulkan_contexts(
    graph_module: torch.fx.GraphModule,
    state_dict_snapshot: Mapping[str, torch.Tensor],
) -> VulkanLayernormLoweringReport:
    graph = graph_module.graph
    reports: list[VulkanLayernormLoweringNodeReport] = []
    context_attrs: dict[
        tuple[str, str, torch.dtype, tuple[int, ...], float], str
    ] = {}
    layer_norm_node_count = 0
    created_context_count = 0
    reused_context_count = 0
    replaced_affine_attrs: set[str] = set()

    for node in tuple(graph.nodes):
        if (
            node.op != "call_function"
            or node.target != torch.ops.aten.layer_norm.default
        ):
            continue

        layer_norm_node_count += 1
        optional_argument_names = ("weight", "bias", "eps", "cudnn_enable")
        if (
            len(node.args) < 2
            or len(node.args) > 6
            or set(node.kwargs).difference(optional_argument_names)
        ):
            reports.append(
                _skip_layernorm(
                    node,
                    "unsupported_layer_norm_signature",
                    None,
                    None,
                    None,
                    None,
                )
            )
            continue
        positional_names = (
            "input",
            "normalized_shape",
            "weight",
            "bias",
            "eps",
            "cudnn_enable",
        )
        if any(
            name in node.kwargs and name in positional_names[: len(node.args)]
            for name in optional_argument_names
        ):
            reports.append(
                _skip_layernorm(
                    node,
                    "unsupported_layer_norm_signature",
                    None,
                    None,
                    None,
                    None,
                )
            )
            continue

        input_arg, normalized_shape_arg = node.args[:2]
        weight_arg = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("weight")
        )
        bias_arg = (
            node.args[3] if len(node.args) > 3 else node.kwargs.get("bias")
        )
        eps_arg = (
            node.args[4] if len(node.args) > 4 else node.kwargs.get("eps", 1e-5)
        )
        cudnn_enable_arg = (
            node.args[5]
            if len(node.args) > 5
            else node.kwargs.get("cudnn_enable", True)
        )
        normalized_shape, normalized_shape_error = _static_positive_int_sequence(
            normalized_shape_arg, "normalized_shape"
        )
        if normalized_shape_error is not None:
            reports.append(
                _skip_layernorm(
                    node,
                    normalized_shape_error,
                    _get_attr_target(weight_arg),
                    _get_attr_target(bias_arg),
                    None,
                    None,
                )
            )
            continue
        if not isinstance(eps_arg, (int, float)) or isinstance(eps_arg, bool):
            reports.append(
                _skip_layernorm(
                    node,
                    "eps_not_finite_nonnegative_static_float",
                    _get_attr_target(weight_arg),
                    _get_attr_target(bias_arg),
                    normalized_shape,
                    None,
                )
            )
            continue
        try:
            eps = float(eps_arg)
        except OverflowError:
            reports.append(
                _skip_layernorm(
                    node,
                    "eps_not_finite_nonnegative_static_float",
                    _get_attr_target(weight_arg),
                    _get_attr_target(bias_arg),
                    normalized_shape,
                    None,
                )
            )
            continue
        if not math.isfinite(eps) or eps < 0.0:
            reports.append(
                _skip_layernorm(
                    node,
                    "eps_not_finite_nonnegative_static_float",
                    _get_attr_target(weight_arg),
                    _get_attr_target(bias_arg),
                    normalized_shape,
                    None,
                )
            )
            continue
        if not isinstance(cudnn_enable_arg, bool):
            reports.append(
                _skip_layernorm(
                    node,
                    "cudnn_enable_not_static_bool",
                    _get_attr_target(weight_arg),
                    _get_attr_target(bias_arg),
                    normalized_shape,
                    eps,
                )
            )
            continue

        input_meta = (
            input_arg.meta.get("val")
            if isinstance(input_arg, torch.fx.Node)
            else None
        )
        if not isinstance(input_meta, torch.Tensor):
            reports.append(
                _skip_layernorm(
                    node,
                    "input_meta_not_tensor",
                    _get_attr_target(weight_arg),
                    _get_attr_target(bias_arg),
                    normalized_shape,
                    eps,
                )
            )
            continue
        if not torch.is_floating_point(input_meta):
            reports.append(
                _skip_layernorm(
                    node,
                    "input_meta_not_floating",
                    _get_attr_target(weight_arg),
                    _get_attr_target(bias_arg),
                    normalized_shape,
                    eps,
                )
            )
            continue
        if input_meta.dim() < len(normalized_shape):
            reports.append(
                _skip_layernorm(
                    node,
                    "input_rank_smaller_than_normalized_shape",
                    _get_attr_target(weight_arg),
                    _get_attr_target(bias_arg),
                    normalized_shape,
                    eps,
                )
            )
            continue
        input_tail = tuple(input_meta.shape[-len(normalized_shape) :])
        if any(
            isinstance(actual, int) and actual != expected
            for actual, expected in zip(input_tail, normalized_shape)
        ):
            reports.append(
                _skip_layernorm(
                    node,
                    "input_normalized_shape_mismatch",
                    _get_attr_target(weight_arg),
                    _get_attr_target(bias_arg),
                    normalized_shape,
                    eps,
                )
            )
            continue

        weight_attr = _get_attr_target(weight_arg)
        bias_attr = _get_attr_target(bias_arg)
        if weight_arg is None:
            reports.append(
                _skip_layernorm(
                    node,
                    "affine_weight_missing",
                    None,
                    bias_attr,
                    normalized_shape,
                    eps,
                )
            )
            continue
        if bias_arg is None:
            reports.append(
                _skip_layernorm(
                    node,
                    "affine_bias_missing",
                    weight_attr,
                    None,
                    normalized_shape,
                    eps,
                )
            )
            continue
        if weight_attr is None:
            reports.append(
                _skip_layernorm(
                    node,
                    "weight_not_static_get_attr",
                    None,
                    bias_attr,
                    normalized_shape,
                    eps,
                )
            )
            continue
        if bias_attr is None:
            reports.append(
                _skip_layernorm(
                    node,
                    "bias_not_static_get_attr",
                    weight_attr,
                    None,
                    normalized_shape,
                    eps,
                )
            )
            continue
        weight = _snapshot_tensor(state_dict_snapshot, weight_attr)
        bias = _snapshot_tensor(state_dict_snapshot, bias_attr)
        if weight is None or bias is None:
            reports.append(
                _skip_layernorm(
                    node,
                    "affine_state_missing_from_cpu_snapshot",
                    weight_attr,
                    bias_attr,
                    normalized_shape,
                    eps,
                )
            )
            continue
        if weight.device.type != "cpu" or bias.device.type != "cpu":
            reports.append(
                _skip_layernorm(
                    node,
                    "affine_state_snapshot_not_cpu",
                    weight_attr,
                    bias_attr,
                    normalized_shape,
                    eps,
                )
            )
            continue
        if (
            not torch.is_floating_point(weight)
            or not torch.is_floating_point(bias)
        ):
            reports.append(
                _skip_layernorm(
                    node,
                    "affine_state_not_floating",
                    weight_attr,
                    bias_attr,
                    normalized_shape,
                    eps,
                )
            )
            continue
        if (
            weight.dtype != bias.dtype
            or weight.dtype != input_meta.dtype
            or tuple(weight.shape) != normalized_shape
            or tuple(bias.shape) != normalized_shape
        ):
            reports.append(
                _skip_layernorm(
                    node,
                    "affine_state_incompatible_with_normalized_shape",
                    weight_attr,
                    bias_attr,
                    normalized_shape,
                    eps,
                )
            )
            continue

        context_key = (
            weight_attr,
            bias_attr,
            weight.dtype,
            normalized_shape,
            eps,
        )
        context_attr = context_attrs.get(context_key)
        context_status = "reused" if context_attr is not None else "created"
        if context_attr is None:
            context_attr = _layernorm_context_attr_name(
                weight_attr, bias_attr, weight.dtype, normalized_shape, eps
            )
            if hasattr(graph_module, context_attr):
                reports.append(
                    VulkanLayernormLoweringNodeReport(
                        node_name=node.name,
                        status="rejected",
                        reason="deterministic_context_attribute_collision",
                        weight_attr=weight_attr,
                        bias_attr=bias_attr,
                        normalized_shape=normalized_shape,
                        eps=eps,
                        context_attr=None,
                        context_status=None,
                    )
                )
                continue
            try:
                context = torch.ops.vulkan_prepack.create_layernorm_context.default(
                    _snapshot_for_context(weight),
                    _snapshot_for_context(bias),
                    eps,
                )
                setattr(graph_module, context_attr, context)
            except (RuntimeError, TypeError, AttributeError) as error:
                reports.append(
                    VulkanLayernormLoweringNodeReport(
                        node_name=node.name,
                        status="rejected",
                        reason=(
                            "layernorm_context_creation_failed:"
                            f"{type(error).__name__}"
                        ),
                        weight_attr=weight_attr,
                        bias_attr=bias_attr,
                        normalized_shape=normalized_shape,
                        eps=eps,
                        context_attr=None,
                        context_status=None,
                    )
                )
                continue
            context_attrs[context_key] = context_attr
            created_context_count += 1
        else:
            reused_context_count += 1

        with graph.inserting_before(node):
            context_node = graph.create_node("get_attr", context_attr, (), {})
            lowered_node = graph.call_function(
                torch.ops.vulkan_prepack.run_layernorm_context.default,
                args=(node.args[0], list(normalized_shape), context_node),
            )
        lowered_node.meta = dict(node.meta)
        node.replace_all_uses_with(lowered_node)
        graph.erase_node(node)
        replaced_affine_attrs.add(weight_attr)
        replaced_affine_attrs.add(bias_attr)
        reports.append(
            VulkanLayernormLoweringNodeReport(
                node_name=node.name,
                status="lowered",
                reason="static_cpu_snapshot_affine_parameters",
                weight_attr=weight_attr,
                bias_attr=bias_attr,
                normalized_shape=normalized_shape,
                eps=eps,
                context_attr=context_attr,
                context_status=context_status,
            )
        )

    if any(report.status == "lowered" for report in reports):
        graph.eliminate_dead_code()
        for affine_attr in replaced_affine_attrs:
            _delete_graph_attr_if_unreferenced(graph_module, affine_attr)
        graph_module.delete_all_unused_submodules()
        graph.lint()
        graph_module.recompile()

    return VulkanLayernormLoweringReport(
        layer_norm_node_count=layer_norm_node_count,
        lowered_count=sum(report.status == "lowered" for report in reports),
        rejected_count=sum(report.status == "rejected" for report in reports),
        skipped_count=sum(report.status == "skipped" for report in reports),
        created_context_count=created_context_count,
        reused_context_count=reused_context_count,
        context_factory="vulkan_prepack::create_layernorm_context",
        nodes=tuple(reports),
    )


def lower_static_add_layernorm_regions(
    graph_module: torch.fx.GraphModule,
) -> VulkanStaticAddLayernormRegionReport:
    graph = graph_module.graph
    reports: list[VulkanStaticAddLayernormRegionNodeReport] = []
    replaced_context_attrs: set[str] = set()
    candidate_count = 0
    lowered_count = 0
    rejected_count = 0
    skipped_count = 0

    def append_report(
        *,
        node_name: str,
        status: str,
        reason: str,
        add_node: torch.fx.Node | None,
        layernorm_node: torch.fx.Node,
        context_attr: str | None,
        plan_attr: str | None,
        normalized_shape: tuple[int, ...] | None,
        context_ownership_outcome: str | None,
        has_plan_schema: bool,
    ) -> None:
        reports.append(
            VulkanStaticAddLayernormRegionNodeReport(
                node_name=node_name,
                status=status,
                reason=reason,
                add_node_name=add_node.name if add_node is not None else None,
                layernorm_node_name=layernorm_node.name,
                context_attr=context_attr,
                plan_attr=plan_attr,
                normalized_shape=normalized_shape,
                program_name=(
                    "StaticAddLayernormRegion" if has_plan_schema else None
                ),
                program_version="v1" if has_plan_schema else None,
                fused_instruction=(
                    "add_layernorm_fused_or_composed_vulkan"
                    if has_plan_schema
                    else None
                ),
                instruction_count=1 if has_plan_schema else 0,
                residual_input_ssa=0 if has_plan_schema else None,
                addend_input_ssa=1 if has_plan_schema else None,
                residual_output_ssa=2 if has_plan_schema else None,
                normalized_output_ssa=3 if has_plan_schema else None,
                residual_input_use_count=1 if has_plan_schema else None,
                residual_input_last_use=0 if has_plan_schema else None,
                addend_input_use_count=1 if has_plan_schema else None,
                addend_input_last_use=0 if has_plan_schema else None,
                static_context_slot=0 if has_plan_schema else None,
                context_ownership_outcome=context_ownership_outcome,
                direct_transition_only=True if has_plan_schema else None,
                replay_state_empty=True if has_plan_schema else None,
                persistent_output_state=False if has_plan_schema else None,
            )
        )

    for layernorm_node in tuple(graph.nodes):
        if (
            layernorm_node.op != "call_function"
            or layernorm_node.target
            != torch.ops.vulkan_prepack.run_layernorm_context.default
        ):
            continue
        context_attr = _graph_owned_layernorm_context_attr(
            graph_module, layernorm_node
        )
        add_node = layernorm_node.args[0]
        if context_attr is None:
            append_report(
                node_name=layernorm_node.name,
                status="skipped",
                reason="layernorm_context_not_graph_owned",
                add_node=add_node if isinstance(add_node, torch.fx.Node) else None,
                layernorm_node=layernorm_node,
                context_attr=None,
                plan_attr=None,
                normalized_shape=None,
                context_ownership_outcome=None,
                has_plan_schema=False,
            )
            skipped_count += 1
            continue
        normalized_shape, normalized_shape_error = _static_positive_int_sequence(
            layernorm_node.args[1], "normalized_shape"
        )
        if normalized_shape_error is not None:
            append_report(
                node_name=layernorm_node.name,
                status="skipped",
                reason=normalized_shape_error,
                add_node=add_node if isinstance(add_node, torch.fx.Node) else None,
                layernorm_node=layernorm_node,
                context_attr=context_attr,
                plan_attr=None,
                normalized_shape=None,
                context_ownership_outcome=None,
                has_plan_schema=False,
            )
            skipped_count += 1
            continue
        if not isinstance(add_node, torch.fx.Node):
            append_report(
                node_name=layernorm_node.name,
                status="skipped",
                reason="layernorm_input_not_graph_add",
                add_node=None,
                layernorm_node=layernorm_node,
                context_attr=context_attr,
                plan_attr=None,
                normalized_shape=normalized_shape,
                context_ownership_outcome=None,
                has_plan_schema=False,
            )
            skipped_count += 1
            continue
        add_is_legal, add_reason = _is_add_tensor_alpha_one(add_node)
        if not add_is_legal:
            append_report(
                node_name=layernorm_node.name,
                status="skipped",
                reason=add_reason,
                add_node=add_node,
                layernorm_node=layernorm_node,
                context_attr=context_attr,
                plan_attr=None,
                normalized_shape=normalized_shape,
                context_ownership_outcome=None,
                has_plan_schema=False,
            )
            skipped_count += 1
            continue
        layernorm_consumers = [
            user
            for user in add_node.users
            if user.op == "call_function"
            and user.target == torch.ops.vulkan_prepack.run_layernorm_context.default
        ]
        if len(layernorm_consumers) != 1:
            append_report(
                node_name=layernorm_node.name,
                status="skipped",
                reason="add_has_multiple_layernorm_consumers",
                add_node=add_node,
                layernorm_node=layernorm_node,
                context_attr=context_attr,
                plan_attr=None,
                normalized_shape=normalized_shape,
                context_ownership_outcome=None,
                has_plan_schema=False,
            )
            skipped_count += 1
            continue

        candidate_count += 1
        plan_attr = _static_add_layernorm_plan_attr_name(
            context_attr,
            add_node.name,
            layernorm_node.name,
            normalized_shape,
        )
        if hasattr(graph_module, plan_attr):
            append_report(
                node_name=layernorm_node.name,
                status="rejected",
                reason="deterministic_plan_attribute_collision",
                add_node=add_node,
                layernorm_node=layernorm_node,
                context_attr=context_attr,
                plan_attr=plan_attr,
                normalized_shape=normalized_shape,
                context_ownership_outcome=None,
                has_plan_schema=True,
            )
            rejected_count += 1
            continue
        try:
            plan = torch.ops.vulkan_prepack.create_graph_add_layernorm_plan.default(
                getattr(graph_module, context_attr),
                list(normalized_shape),
            )
            setattr(graph_module, plan_attr, plan)
        except (RuntimeError, TypeError, AttributeError) as error:
            append_report(
                node_name=layernorm_node.name,
                status="rejected",
                reason=(
                    "static_add_layernorm_plan_creation_failed:"
                    f"{type(error).__name__}"
                ),
                add_node=add_node,
                layernorm_node=layernorm_node,
                context_attr=context_attr,
                plan_attr=plan_attr,
                normalized_shape=normalized_shape,
                context_ownership_outcome=None,
                has_plan_schema=True,
            )
            rejected_count += 1
            continue

        context_node = layernorm_node.args[2]
        context_reference_count = sum(
            node.op == "get_attr" and str(node.target) == context_attr
            for node in graph.nodes
        )
        context_has_other_users = any(
            user is not layernorm_node for user in context_node.users
        )
        context_ownership_outcome = (
            "transferred_removed_original_context_attr"
            if context_reference_count == 1 and not context_has_other_users
            else "shared_context_retained_original_attr"
        )
        with graph.inserting_before(add_node):
            plan_node = graph.create_node("get_attr", plan_attr, (), {})
            region_node = graph.call_function(
                torch.ops.vulkan_prepack.run_graph_add_layernorm_plan.default,
                args=(add_node.args[0], add_node.args[1], plan_node),
            )
            residual_node = graph.call_function(operator.getitem, (region_node, 0))
            normalized_node = graph.call_function(operator.getitem, (region_node, 1))
        residual_node.meta = dict(add_node.meta)
        normalized_node.meta = dict(layernorm_node.meta)
        add_node.replace_all_uses_with(residual_node)
        layernorm_node.replace_all_uses_with(normalized_node)
        graph.erase_node(layernorm_node)
        if not context_node.users:
            graph.erase_node(context_node)
        graph.erase_node(add_node)
        replaced_context_attrs.add(context_attr)
        append_report(
            node_name=region_node.name,
            status="lowered",
            reason="graph_owned_static_add_layernorm",
            add_node=add_node,
            layernorm_node=layernorm_node,
            context_attr=context_attr,
            plan_attr=plan_attr,
            normalized_shape=normalized_shape,
            context_ownership_outcome=context_ownership_outcome,
            has_plan_schema=True,
        )
        lowered_count += 1

    if lowered_count:
        graph.eliminate_dead_code()
        for context_attr in replaced_context_attrs:
            _delete_graph_attr_if_unreferenced(graph_module, context_attr)
        graph_module.delete_all_unused_submodules()
        graph.lint()
        graph_module.recompile()

    return VulkanStaticAddLayernormRegionReport(
        candidate_count=candidate_count,
        lowered_count=lowered_count,
        rejected_count=rejected_count,
        skipped_count=skipped_count,
        plan_factory="vulkan_prepack::create_graph_add_layernorm_plan",
        nodes=tuple(reports),
    )


def lower_static_conv2d_relu_conv2d_regions(
    graph_module: torch.fx.GraphModule,
) -> VulkanStaticConv2dReluConv2dRegionReport:
    graph = graph_module.graph
    reports: list[VulkanStaticConv2dReluConv2dRegionNodeReport] = []
    removed_context_attrs: set[str] = set()
    candidate_count = 0
    lowered_count = 0
    rejected_count = 0
    skipped_count = 0
    excluded_relu_node_names: set[str] = set()

    def append_report(
        *,
        node_name: str,
        status: str,
        reason: str,
        first_conv2d_node: torch.fx.Node | None,
        relu_node: torch.fx.Node | None,
        second_conv2d_node: torch.fx.Node | None,
        first_context_attr: str | None,
        second_context_attr: str | None,
        plan_attr: str | None,
        has_plan_schema: bool,
    ) -> None:
        reports.append(
            VulkanStaticConv2dReluConv2dRegionNodeReport(
                node_name=node_name,
                status=status,
                reason=reason,
                first_conv2d_node_name=(
                    first_conv2d_node.name if first_conv2d_node else None
                ),
                relu_node_name=relu_node.name if relu_node else None,
                second_conv2d_node_name=(
                    second_conv2d_node.name if second_conv2d_node else None
                ),
                first_context_attr=first_context_attr,
                second_context_attr=second_context_attr,
                plan_attr=plan_attr,
                program_name="VulkanGraphRegionPlan" if has_plan_schema else None,
                program_version="v1" if has_plan_schema else None,
                instruction_count=2 if has_plan_schema else 0,
                input_ssa=0 if has_plan_schema else None,
                intermediate_ssa=1 if has_plan_schema else None,
                output_ssa=2 if has_plan_schema else None,
                input_use_count=1 if has_plan_schema else None,
                input_last_use=0 if has_plan_schema else None,
                intermediate_use_count=1 if has_plan_schema else None,
                intermediate_last_use=1 if has_plan_schema else None,
                first_static_context_slot=0 if has_plan_schema else None,
                second_static_context_slot=1 if has_plan_schema else None,
                bounded_submission_owned=True if has_plan_schema else None,
                program_private_scratch=True if has_plan_schema else None,
                scratch_ring_capacity=2 if has_plan_schema else None,
                timeline_gated_release=True if has_plan_schema else None,
                direct_transition_only=True if has_plan_schema else None,
                replay_state_empty=True if has_plan_schema else None,
            )
        )

    for relu_node in tuple(graph.nodes):
        if not _is_relu(relu_node):
            continue
        first_conv2d_node = relu_node.args[0]
        if not isinstance(first_conv2d_node, torch.fx.Node):
            continue
        first_context_attr = _graph_owned_conv2d_context_attr(
            graph_module, first_conv2d_node
        )
        if first_context_attr is None:
            continue
        second_conv2d_node = next(
            (
                node
                for node in relu_node.users
                if _graph_owned_conv2d_context_attr(graph_module, node) is not None
            ),
            None,
        )
        second_context_attr = (
            _graph_owned_conv2d_context_attr(graph_module, second_conv2d_node)
            if second_conv2d_node is not None
            else None
        )
        is_three_node_candidate = second_conv2d_node is not None
        if len(first_conv2d_node.users) != 1:
            append_report(
                node_name=relu_node.name,
                status="skipped",
                reason="first_conv2d_output_has_multiple_users",
                first_conv2d_node=first_conv2d_node,
                relu_node=relu_node,
                second_conv2d_node=second_conv2d_node,
                first_context_attr=first_context_attr,
                second_context_attr=second_context_attr,
                plan_attr=None,
                has_plan_schema=False,
            )
            if is_three_node_candidate:
                excluded_relu_node_names.add(relu_node.name)
            skipped_count += 1
            continue
        if len(relu_node.users) != 1:
            append_report(
                node_name=relu_node.name,
                status="skipped",
                reason="relu_output_has_multiple_users",
                first_conv2d_node=first_conv2d_node,
                relu_node=relu_node,
                second_conv2d_node=second_conv2d_node,
                first_context_attr=first_context_attr,
                second_context_attr=second_context_attr,
                plan_attr=None,
                has_plan_schema=False,
            )
            if is_three_node_candidate:
                excluded_relu_node_names.add(relu_node.name)
            skipped_count += 1
            continue
        if second_context_attr is None:
            append_report(
                node_name=relu_node.name,
                status="skipped",
                reason="relu_output_not_graph_owned_conv2d",
                first_conv2d_node=first_conv2d_node,
                relu_node=relu_node,
                second_conv2d_node=second_conv2d_node,
                first_context_attr=first_context_attr,
                second_context_attr=None,
                plan_attr=None,
                has_plan_schema=False,
            )
            skipped_count += 1
            continue
        if first_context_attr == second_context_attr or (
            getattr(graph_module, first_context_attr)
            is getattr(graph_module, second_context_attr)
        ):
            append_report(
                node_name=relu_node.name,
                status="skipped",
                reason="conv2d_contexts_must_be_distinct",
                first_conv2d_node=first_conv2d_node,
                relu_node=relu_node,
                second_conv2d_node=second_conv2d_node,
                first_context_attr=first_context_attr,
                second_context_attr=second_context_attr,
                plan_attr=None,
                has_plan_schema=False,
            )
            excluded_relu_node_names.add(relu_node.name)
            skipped_count += 1
            continue
        first_context_reference_count = sum(
            node.op == "get_attr" and str(node.target) == first_context_attr
            for node in graph.nodes
        )
        second_context_reference_count = sum(
            node.op == "get_attr" and str(node.target) == second_context_attr
            for node in graph.nodes
        )
        if first_context_reference_count != 1:
            append_report(
                node_name=relu_node.name,
                status="skipped",
                reason="first_context_attr_has_multiple_references",
                first_conv2d_node=first_conv2d_node,
                relu_node=relu_node,
                second_conv2d_node=second_conv2d_node,
                first_context_attr=first_context_attr,
                second_context_attr=second_context_attr,
                plan_attr=None,
                has_plan_schema=False,
            )
            excluded_relu_node_names.add(relu_node.name)
            skipped_count += 1
            continue
        if second_context_reference_count != 1:
            append_report(
                node_name=relu_node.name,
                status="skipped",
                reason="second_context_attr_has_multiple_references",
                first_conv2d_node=first_conv2d_node,
                relu_node=relu_node,
                second_conv2d_node=second_conv2d_node,
                first_context_attr=first_context_attr,
                second_context_attr=second_context_attr,
                plan_attr=None,
                has_plan_schema=False,
            )
            excluded_relu_node_names.add(relu_node.name)
            skipped_count += 1
            continue

        candidate_count += 1
        plan_attr = _vulkan_graph_region_plan_attr_name(
            "conv2d_relu_conv2d",
            first_context_attr,
            second_context_attr,
            second_conv2d_node.name,
        )
        if hasattr(graph_module, plan_attr):
            append_report(
                node_name=relu_node.name,
                status="rejected",
                reason="deterministic_plan_attribute_collision",
                first_conv2d_node=first_conv2d_node,
                relu_node=relu_node,
                second_conv2d_node=second_conv2d_node,
                first_context_attr=first_context_attr,
                second_context_attr=second_context_attr,
                plan_attr=plan_attr,
                has_plan_schema=True,
            )
            excluded_relu_node_names.add(relu_node.name)
            rejected_count += 1
            continue
        try:
            plan = (
                torch.ops.vulkan_prepack.create_vulkan_graph_region_plan_conv2d_relu_conv2d.default(
                    getattr(graph_module, first_context_attr),
                    getattr(graph_module, second_context_attr),
                )
            )
            setattr(graph_module, plan_attr, plan)
        except (RuntimeError, TypeError, AttributeError) as error:
            append_report(
                node_name=relu_node.name,
                status="rejected",
                reason=(
                    "vulkan_graph_region_plan_creation_failed:"
                    f"{type(error).__name__}"
                ),
                first_conv2d_node=first_conv2d_node,
                relu_node=relu_node,
                second_conv2d_node=second_conv2d_node,
                first_context_attr=first_context_attr,
                second_context_attr=second_context_attr,
                plan_attr=plan_attr,
                has_plan_schema=True,
            )
            excluded_relu_node_names.add(relu_node.name)
            rejected_count += 1
            continue

        first_context_node = first_conv2d_node.args[1]
        second_context_node = second_conv2d_node.args[1]
        with graph.inserting_before(second_conv2d_node):
            plan_node = graph.create_node("get_attr", plan_attr, (), {})
            region_node = graph.call_function(
                torch.ops.vulkan_prepack.run_vulkan_graph_region_plan.default,
                args=([first_conv2d_node.args[0]], plan_node),
            )
            lowered_node = graph.call_function(operator.getitem, (region_node, 0))
        lowered_node.meta = dict(second_conv2d_node.meta)
        second_conv2d_node.replace_all_uses_with(lowered_node)
        graph.erase_node(second_conv2d_node)
        graph.erase_node(relu_node)
        graph.erase_node(first_conv2d_node)
        graph.erase_node(second_context_node)
        graph.erase_node(first_context_node)
        removed_context_attrs.add(first_context_attr)
        removed_context_attrs.add(second_context_attr)
        append_report(
            node_name=lowered_node.name,
            status="lowered",
            reason="graph_owned_static_conv2d_relu_conv2d",
            first_conv2d_node=first_conv2d_node,
            relu_node=relu_node,
            second_conv2d_node=second_conv2d_node,
            first_context_attr=first_context_attr,
            second_context_attr=second_context_attr,
            plan_attr=plan_attr,
            has_plan_schema=True,
        )
        lowered_count += 1

    if lowered_count:
        graph.eliminate_dead_code()
        for context_attr in removed_context_attrs:
            if hasattr(graph_module, context_attr):
                delattr(graph_module, context_attr)
        graph_module.delete_all_unused_submodules()
        graph.lint()
        graph_module.recompile()

    return VulkanStaticConv2dReluConv2dRegionReport(
        candidate_count=candidate_count,
        lowered_count=lowered_count,
        rejected_count=rejected_count,
        skipped_count=skipped_count,
        plan_factory=(
            "vulkan_prepack::create_vulkan_graph_region_plan_conv2d_relu_conv2d"
        ),
        nodes=tuple(reports),
        excluded_relu_node_names=tuple(sorted(excluded_relu_node_names)),
    )


def lower_static_conv2d_relu_regions(
    graph_module: torch.fx.GraphModule,
    excluded_relu_node_names: frozenset[str] | set[str] | tuple[str, ...] = (),
) -> VulkanStaticConv2dReluRegionReport:
    graph = graph_module.graph
    reports: list[VulkanStaticConv2dReluRegionNodeReport] = []
    removed_context_attrs: set[str] = set()
    candidate_count = 0
    lowered_count = 0
    rejected_count = 0
    skipped_count = 0
    excluded_relu_node_names = frozenset(excluded_relu_node_names)

    for relu_node in tuple(graph.nodes):
        if not _is_relu(relu_node):
            continue
        conv2d_node = relu_node.args[0]
        if not isinstance(conv2d_node, torch.fx.Node):
            continue
        context_attr = _graph_owned_conv2d_context_attr(
            graph_module, conv2d_node
        )
        if context_attr is None:
            continue
        if relu_node.name in excluded_relu_node_names:
            reports.append(
                VulkanStaticConv2dReluRegionNodeReport(
                    node_name=relu_node.name,
                    status="skipped",
                    reason="excluded_by_static_conv2d_relu_conv2d_region",
                    conv2d_node_name=conv2d_node.name,
                    context_attr=context_attr,
                    plan_attr=None,
                    program_name=None,
                    program_version=None,
                    instruction_count=0,
                    input_ssa=None,
                    output_ssa=None,
                    input_use_count=None,
                    input_last_use=None,
                    static_context_slot=None,
                    direct_transition_only=None,
                    replay_state_empty=None,
                )
            )
            skipped_count += 1
            continue
        context_node = conv2d_node.args[1]
        context_attr_reference_count = sum(
            node.op == "get_attr" and str(node.target) == context_attr
            for node in graph.nodes
        )
        if len(conv2d_node.users) != 1:
            reports.append(
                VulkanStaticConv2dReluRegionNodeReport(
                    node_name=relu_node.name,
                    status="skipped",
                    reason="conv2d_output_has_multiple_users",
                    conv2d_node_name=conv2d_node.name,
                    context_attr=context_attr,
                    plan_attr=None,
                    program_name=None,
                    program_version=None,
                    instruction_count=0,
                    input_ssa=None,
                    output_ssa=None,
                    input_use_count=None,
                    input_last_use=None,
                    static_context_slot=None,
                    direct_transition_only=None,
                    replay_state_empty=None,
                )
            )
            skipped_count += 1
            continue
        if context_attr_reference_count != 1:
            reports.append(
                VulkanStaticConv2dReluRegionNodeReport(
                    node_name=relu_node.name,
                    status="skipped",
                    reason="context_attr_has_multiple_references",
                    conv2d_node_name=conv2d_node.name,
                    context_attr=context_attr,
                    plan_attr=None,
                    program_name=None,
                    program_version=None,
                    instruction_count=0,
                    input_ssa=None,
                    output_ssa=None,
                    input_use_count=None,
                    input_last_use=None,
                    static_context_slot=None,
                    direct_transition_only=None,
                    replay_state_empty=None,
                )
            )
            skipped_count += 1
            continue

        candidate_count += 1
        plan_attr = _static_conv2d_relu_plan_attr_name(context_attr, relu_node.name)
        if hasattr(graph_module, plan_attr):
            reports.append(
                VulkanStaticConv2dReluRegionNodeReport(
                    node_name=relu_node.name,
                    status="rejected",
                    reason="deterministic_plan_attribute_collision",
                    conv2d_node_name=conv2d_node.name,
                    context_attr=context_attr,
                    plan_attr=plan_attr,
                    program_name="StaticConv2dReluRegion",
                    program_version="v1",
                    instruction_count=1,
                    input_ssa=0,
                    output_ssa=1,
                    input_use_count=1,
                    input_last_use=0,
                    static_context_slot=0,
                    direct_transition_only=True,
                    replay_state_empty=True,
                )
            )
            rejected_count += 1
            continue
        try:
            plan = torch.ops.vulkan_prepack.create_graph_conv2d_relu_plan.default(
                getattr(graph_module, context_attr)
            )
            setattr(graph_module, plan_attr, plan)
        except (RuntimeError, TypeError, AttributeError) as error:
            reports.append(
                VulkanStaticConv2dReluRegionNodeReport(
                    node_name=relu_node.name,
                    status="rejected",
                    reason=(
                        "static_conv2d_relu_plan_creation_failed:"
                        f"{type(error).__name__}"
                    ),
                    conv2d_node_name=conv2d_node.name,
                    context_attr=context_attr,
                    plan_attr=plan_attr,
                    program_name="StaticConv2dReluRegion",
                    program_version="v1",
                    instruction_count=1,
                    input_ssa=0,
                    output_ssa=1,
                    input_use_count=1,
                    input_last_use=0,
                    static_context_slot=0,
                    direct_transition_only=True,
                    replay_state_empty=True,
                )
            )
            rejected_count += 1
            continue

        with graph.inserting_before(relu_node):
            plan_node = graph.create_node("get_attr", plan_attr, (), {})
            lowered_node = graph.call_function(
                torch.ops.vulkan_prepack.run_graph_conv2d_relu_plan.default,
                args=(conv2d_node.args[0], plan_node),
            )
        lowered_node.meta = dict(relu_node.meta)
        relu_node.replace_all_uses_with(lowered_node)
        graph.erase_node(relu_node)
        graph.erase_node(conv2d_node)
        graph.erase_node(context_node)
        removed_context_attrs.add(context_attr)
        reports.append(
            VulkanStaticConv2dReluRegionNodeReport(
                node_name=lowered_node.name,
                status="lowered",
                reason="graph_owned_static_conv2d_relu",
                conv2d_node_name=conv2d_node.name,
                context_attr=context_attr,
                plan_attr=plan_attr,
                program_name="StaticConv2dReluRegion",
                program_version="v1",
                instruction_count=1,
                input_ssa=0,
                output_ssa=1,
                input_use_count=1,
                input_last_use=0,
                static_context_slot=0,
                direct_transition_only=True,
                replay_state_empty=True,
            )
        )
        lowered_count += 1

    if lowered_count:
        graph.eliminate_dead_code()
        for context_attr in removed_context_attrs:
            if hasattr(graph_module, context_attr):
                delattr(graph_module, context_attr)
        graph_module.delete_all_unused_submodules()
        graph.lint()
        graph_module.recompile()

    return VulkanStaticConv2dReluRegionReport(
        candidate_count=candidate_count,
        lowered_count=lowered_count,
        rejected_count=rejected_count,
        skipped_count=skipped_count,
        plan_factory="vulkan_prepack::create_graph_conv2d_relu_plan",
        nodes=tuple(reports),
    )


def make_vulkan_graph_region_lowering_report(
    static_linear_gelu_regions: VulkanStaticLinearGeluRegionReport,
    static_conv2d_relu_conv2d_regions: VulkanStaticConv2dReluConv2dRegionReport,
) -> VulkanGraphRegionLoweringReport:
    def linear_gelu_family_diagnostics(
        family: str,
        plan_factory: str,
    ) -> VulkanGraphRegionFamilyDiagnostics:
        nodes = tuple(
            node
            for node in static_linear_gelu_regions.nodes
            if node.region_family == family
        )
        return VulkanGraphRegionFamilyDiagnostics(
            family=family,
            candidate_count=sum(
                node.status in {"lowered", "rejected"} for node in nodes
            ),
            lowered_count=sum(node.status == "lowered" for node in nodes),
            rejected_count=sum(node.status == "rejected" for node in nodes),
            skipped_count=sum(node.status == "skipped" for node in nodes),
            plan_factory=plan_factory,
            nodes=nodes,
        )

    return VulkanGraphRegionLoweringReport(
        plan_class="VulkanGraphRegionPlan",
        plan_version="v1",
        families=(
            linear_gelu_family_diagnostics(
                "linear_gelu_tanh",
                "vulkan_prepack::create_vulkan_graph_region_plan_linear_gelu",
            ),
            linear_gelu_family_diagnostics(
                "linear_gelu_none",
                "vulkan_prepack::create_vulkan_graph_region_plan_linear_gelu_none",
            ),
            VulkanGraphRegionFamilyDiagnostics(
                family="conv2d_relu_conv2d",
                candidate_count=static_conv2d_relu_conv2d_regions.candidate_count,
                lowered_count=static_conv2d_relu_conv2d_regions.lowered_count,
                rejected_count=static_conv2d_relu_conv2d_regions.rejected_count,
                skipped_count=static_conv2d_relu_conv2d_regions.skipped_count,
                plan_factory=static_conv2d_relu_conv2d_regions.plan_factory,
                nodes=static_conv2d_relu_conv2d_regions.nodes,
            ),
        ),
    )


__all__ = [
    "VulkanConv2dLoweringNodeReport",
    "VulkanConv2dLoweringReport",
    "VulkanGraphInputNormalizationNodeReport",
    "VulkanGraphInputNormalizationReport",
    "VulkanGraphTensorPlacementNodeReport",
    "VulkanGraphTensorPlacementReport",
    "VulkanLiftedTensorConstantNodeReport",
    "VulkanLiftedTensorConstantReport",
    "VulkanStaticFactoryConstantNodeReport",
    "VulkanStaticFactoryConstantReport",
    "VulkanStaticIdentityAdvancedIndexNodeReport",
    "VulkanStaticIdentityAdvancedIndexReport",
    "VulkanStaticGQARepeatNodeReport",
    "VulkanStaticGQARepeatReport",
    "VulkanLayernormLoweringNodeReport",
    "VulkanLayernormLoweringReport",
    "VulkanStaticAddLayernormRegionNodeReport",
    "VulkanStaticAddLayernormRegionReport",
    "VulkanLinearLoweringNodeReport",
    "VulkanLinearLoweringReport",
    "VulkanStaticConv2dReluRegionNodeReport",
    "VulkanStaticConv2dReluRegionReport",
    "VulkanStaticConv2dReluConv2dRegionNodeReport",
    "VulkanStaticConv2dReluConv2dRegionReport",
    "VulkanStaticLinearGeluRegionNodeReport",
    "VulkanStaticLinearGeluRegionReport",
    "VulkanGraphRegionFamilyDiagnostics",
    "VulkanGraphRegionLoweringReport",
    "make_vulkan_graph_region_lowering_report",
    "lower_static_conv2d_to_vulkan_contexts",
    "lower_static_conv2d_relu_regions",
    "lower_static_conv2d_relu_conv2d_regions",
    "lower_static_add_layernorm_regions",
    "lower_static_layernorm_to_vulkan_contexts",
    "lower_static_linear_to_vulkan_contexts",
    "lower_static_linear_gelu_regions",
    "lower_graph_input_dtype_normalizations",
    "lower_lifted_tensor_constants",
    "lower_static_factory_constants",
    "lower_static_identity_advanced_indices",
    "lower_static_gqa_repeats",
    "plan_graph_tensor_placements",
]
