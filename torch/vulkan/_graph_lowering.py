from __future__ import annotations

import dataclasses
import hashlib
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


def _static_linear_gelu_plan_attr_name(
    context_attr: str,
    gelu_node_name: str,
) -> str:
    identity = "\x00".join(
        (context_attr, gelu_node_name, "StaticLinearGeluRegion.v1")
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"_vulkan_static_linear_gelu_plan_{digest}"


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


def _is_tanh_gelu(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops.aten.gelu.default
        and len(node.args) == 1
        and node.kwargs == {"approximate": "tanh"}
    )


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


__all__ = [
    "VulkanConv2dLoweringNodeReport",
    "VulkanConv2dLoweringReport",
    "VulkanLinearLoweringNodeReport",
    "VulkanLinearLoweringReport",
    "VulkanStaticLinearGeluRegionNodeReport",
    "VulkanStaticLinearGeluRegionReport",
    "lower_static_conv2d_to_vulkan_contexts",
    "lower_static_linear_to_vulkan_contexts",
    "lower_static_linear_gelu_regions",
]
