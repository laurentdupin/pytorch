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
    nodes: tuple[VulkanLinearLoweringNodeReport, ...]


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
                context = torch.ops.vulkan_prepack.create_linear_context.default(
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
        nodes=tuple(reports),
    )


__all__ = [
    "VulkanLinearLoweringNodeReport",
    "VulkanLinearLoweringReport",
    "lower_static_linear_to_vulkan_contexts",
]
