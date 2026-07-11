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


@dataclasses.dataclass(frozen=True)
class VulkanStaticLinearGeluRegionReport:
    candidate_count: int
    lowered_count: int
    rejected_count: int
    skipped_count: int
    plan_factory: str
    nodes: tuple[VulkanStaticLinearGeluRegionNodeReport, ...]


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


def _static_linear_gelu_plan_attr_name(
    context_attr: str,
    gelu_node_name: str,
) -> str:
    identity = "\x00".join(
        (context_attr, gelu_node_name, "StaticLinearGeluRegion.v1")
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"_vulkan_static_linear_gelu_plan_{digest}"


def _static_conv2d_relu_plan_attr_name(
    context_attr: str,
    relu_node_name: str,
) -> str:
    identity = "\x00".join(
        (context_attr, relu_node_name, "StaticConv2dReluRegion.v1")
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"_vulkan_static_conv2d_relu_plan_{digest}"


def _static_conv2d_relu_conv2d_plan_attr_name(
    first_context_attr: str,
    second_context_attr: str,
    second_conv2d_node_name: str,
) -> str:
    identity = "\x00".join(
        (
            first_context_attr,
            second_context_attr,
            second_conv2d_node_name,
            "StaticConv2dReluConv2dRegion.v2",
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"_vulkan_static_conv2d_relu_conv2d_plan_{digest}"


def _snapshot_for_context(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone(memory_format=torch.contiguous_format)


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


def _is_tanh_gelu(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops.aten.gelu.default
        and len(node.args) == 1
        and node.kwargs == {"approximate": "tanh"}
    )


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
        if not _is_tanh_gelu(gelu_node):
            continue
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
                )
            )
            skipped_count += 1
            continue

        candidate_count += 1
        plan_attr = _static_linear_gelu_plan_attr_name(context_attr, gelu_node.name)
        if hasattr(graph_module, plan_attr):
            reports.append(
                VulkanStaticLinearGeluRegionNodeReport(
                    node_name=gelu_node.name,
                    status="rejected",
                    reason="deterministic_plan_attribute_collision",
                    linear_node_name=linear_node.name,
                    context_attr=context_attr,
                    plan_attr=plan_attr,
                    program_name="StaticLinearGeluRegion",
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
            plan = torch.ops.vulkan_prepack.create_graph_linear_gelu_plan.default(
                getattr(graph_module, context_attr)
            )
            setattr(graph_module, plan_attr, plan)
        except (RuntimeError, TypeError, AttributeError) as error:
            reports.append(
                VulkanStaticLinearGeluRegionNodeReport(
                    node_name=gelu_node.name,
                    status="rejected",
                    reason=(
                        "static_linear_gelu_plan_creation_failed:"
                        f"{type(error).__name__}"
                    ),
                    linear_node_name=linear_node.name,
                    context_attr=context_attr,
                    plan_attr=plan_attr,
                    program_name="StaticLinearGeluRegion",
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

        with graph.inserting_before(gelu_node):
            plan_node = graph.create_node("get_attr", plan_attr, (), {})
            lowered_node = graph.call_function(
                torch.ops.vulkan_prepack.run_graph_linear_gelu_plan.default,
                args=(linear_node.args[0], plan_node),
            )
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
                reason="graph_owned_static_linear_tanh_gelu",
                linear_node_name=linear_node.name,
                context_attr=context_attr,
                plan_attr=plan_attr,
                program_name="StaticLinearGeluRegion",
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

    return VulkanStaticLinearGeluRegionReport(
        candidate_count=candidate_count,
        lowered_count=lowered_count,
        rejected_count=rejected_count,
        skipped_count=skipped_count,
        plan_factory="vulkan_prepack::create_graph_linear_gelu_plan",
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
                program_name=(
                    "StaticConv2dReluConv2dRegion" if has_plan_schema else None
                ),
                program_version="v2" if has_plan_schema else None,
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
        plan_attr = _static_conv2d_relu_conv2d_plan_attr_name(
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
                torch.ops.vulkan_prepack.create_graph_conv2d_relu_conv2d_plan.default(
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
                    "static_conv2d_relu_conv2d_plan_creation_failed:"
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
            lowered_node = graph.call_function(
                torch.ops.vulkan_prepack.run_graph_conv2d_relu_conv2d_plan.default,
                args=(first_conv2d_node.args[0], plan_node),
            )
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
        plan_factory="vulkan_prepack::create_graph_conv2d_relu_conv2d_plan",
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


__all__ = [
    "VulkanConv2dLoweringNodeReport",
    "VulkanConv2dLoweringReport",
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
    "lower_static_conv2d_to_vulkan_contexts",
    "lower_static_conv2d_relu_regions",
    "lower_static_conv2d_relu_conv2d_regions",
    "lower_static_add_layernorm_regions",
    "lower_static_layernorm_to_vulkan_contexts",
    "lower_static_linear_to_vulkan_contexts",
    "lower_static_linear_gelu_regions",
]
