# mypy: allow-untyped-defs

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import operator
import re
import threading
from typing import TYPE_CHECKING, Any

import torch
import torch.utils._pytree as pytree

from ._graph_plan import VulkanGraphPlanReport, compile_vulkan_graph_plan
from ._graph_lowering import (
    VulkanConv2dLoweringReport,
    VulkanGraphInputNormalizationReport,
    VulkanGraphRegionLoweringReport,
    VulkanGraphTensorPlacementReport,
    VulkanLayernormLoweringReport,
    VulkanLiftedTensorConstantReport,
    VulkanLinearLoweringReport,
    VulkanStaticAddLayernormRegionReport,
    VulkanStaticConv2dReluConv2dRegionReport,
    VulkanStaticConv2dReluRegionReport,
    VulkanStaticFactoryConstantReport,
    VulkanStaticGQARepeatReport,
    VulkanStaticIdentityAdvancedIndexReport,
    VulkanStaticLinearGeluRegionReport,
    extract_verified_exported_input_guard,
    is_verified_exported_input_guard_call,
    lower_lifted_tensor_constants,
    lower_static_conv2d_relu_regions,
    lower_static_conv2d_relu_conv2d_regions,
    lower_static_conv2d_to_vulkan_contexts,
    lower_static_factory_constants,
    lower_static_gqa_repeats,
    lower_static_identity_advanced_indices,
    lower_static_add_layernorm_regions,
    lower_static_layernorm_to_vulkan_contexts,
    lower_static_linear_to_vulkan_contexts,
    lower_static_linear_gelu_regions,
    lower_graph_input_dtype_normalizations,
    make_vulkan_graph_region_lowering_report,
    plan_graph_tensor_placements,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclasses.dataclass(frozen=True)
class VulkanGraphNodeRecord:
    index: int
    name: str
    op: str
    target: str
    classification: str
    reason: str


@dataclasses.dataclass(frozen=True)
class VulkanGraphCensus:
    captured_node_count: int
    call_function_node_count: int
    direct_vulkan_node_count: int
    lowered_vulkan_node_count: int
    composite_node_count: int
    graph_node_count: int
    unsupported_node_count: int
    nodes: tuple[VulkanGraphNodeRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class VulkanGraphProgramKey:
    graph_hash: str
    state_fingerprint: str
    input_signature: tuple[str, ...]
    device_index: int
    vendor_id: int
    device_id: int
    driver_version: int
    api_version: str


@dataclasses.dataclass(frozen=True)
class VulkanGraphImplicitBoundaryAttribution:
    node_name: str
    target: str
    cpu_fallback_count: int
    sync_readback_count: int
    deferred_values_created: int


class VulkanGraphExecutionError(RuntimeError):
    pass


_CPP_PLAN_IMPLICIT_BOUNDARY_PATTERN = re.compile(
    r"VulkanGraphPlan\.v[0-9]+ node '([^']+)' \(([^)]+)\) crossed an "
    r"implicit host boundary: cpu_fallback=([0-9]+), sync_readback=([0-9]+), "
    r"deferred_values_created=([0-9]+)"
)


def _implicit_boundary_error(
    attribution: VulkanGraphImplicitBoundaryAttribution,
) -> VulkanGraphExecutionError:
    return VulkanGraphExecutionError(
        f"Vulkan graph node {attribution.node_name!r} ({attribution.target}) "
        "crossed an implicit host boundary: "
        f"cpu_fallback={attribution.cpu_fallback_count}, "
        f"sync_readback={attribution.sync_readback_count}, "
        f"deferred_values_created={attribution.deferred_values_created}. "
        "Explicit CPU partitions and deferred values are not implemented"
    )


def _cpp_plan_implicit_boundary(
    error: Exception,
) -> VulkanGraphImplicitBoundaryAttribution | None:
    match = _CPP_PLAN_IMPLICIT_BOUNDARY_PATTERN.search(str(error))
    if match is None:
        return None
    return VulkanGraphImplicitBoundaryAttribution(
        node_name=match.group(1),
        target=match.group(2),
        cpu_fallback_count=int(match.group(3)),
        sync_readback_count=int(match.group(4)),
        deferred_values_created=int(match.group(5)),
    )


_PYTHON_SCALAR_ARITHMETIC_TARGETS = frozenset(
    (operator.add, operator.sub, operator.mul, operator.floordiv)
)
_SYMBOLIC_SCALAR_TYPES = tuple(
    scalar_type
    for scalar_type in (
        getattr(torch, "SymInt", None),
        getattr(torch, "SymFloat", None),
        getattr(torch, "SymBool", None),
    )
    if isinstance(scalar_type, type)
)


def _target_name(target: Any) -> str:
    name = getattr(target, "name", None)
    if callable(name):
        return str(name())
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return str(target)


def _inline_inference_grad_wrappers(graph_module: torch.fx.GraphModule) -> int:
    from torch._higher_order_ops.wrap import wrap_with_set_grad_enabled

    graph = graph_module.graph
    inlined = 0
    for node in tuple(graph.nodes):
        if (
            node.op != "call_function"
            or node.target is not wrap_with_set_grad_enabled
            or len(node.args) < 2
            or node.args[0] is not False
        ):
            continue
        if node.kwargs:
            raise VulkanGraphExecutionError(
                "Inference grad wrapper lowering does not accept keyword arguments"
            )
        submodule_node = node.args[1]
        if (
            not isinstance(submodule_node, torch.fx.Node)
            or submodule_node.op != "get_attr"
            or not isinstance(submodule_node.target, str)
        ):
            raise VulkanGraphExecutionError(
                "Inference grad wrapper requires a graph-owned submodule"
            )
        submodule = graph_module.get_submodule(submodule_node.target)
        if not isinstance(submodule, torch.fx.GraphModule):
            raise VulkanGraphExecutionError(
                "Inference grad wrapper requires an FX GraphModule body"
            )
        body_nodes = tuple(submodule.graph.nodes)
        if any(
            body_node.op not in ("placeholder", "call_function", "output")
            for body_node in body_nodes
        ):
            raise VulkanGraphExecutionError(
                "Inference grad wrapper body contains unsupported graph state"
            )
        placeholders = tuple(
            body_node for body_node in body_nodes if body_node.op == "placeholder"
        )
        operands = node.args[2:]
        if len(placeholders) != len(operands) or any(
            not isinstance(operand, torch.fx.Node) for operand in operands
        ):
            raise VulkanGraphExecutionError(
                "Inference grad wrapper inputs do not match its graph body"
            )
        users = tuple(node.users)
        if any(
            user.op != "call_function"
            or user.target is not operator.getitem
            or len(user.args) != 2
            or user.args[0] is not node
            or not isinstance(user.args[1], int)
            or isinstance(user.args[1], bool)
            for user in users
        ):
            raise VulkanGraphExecutionError(
                "Inference grad wrapper outputs require static tuple extraction"
            )
        with graph.inserting_before(node):
            output = graph.graph_copy(
                submodule.graph,
                dict(zip(placeholders, operands)),
            )
        if not isinstance(output, tuple | list) or any(
            not isinstance(value, torch.fx.Node) for value in output
        ):
            raise VulkanGraphExecutionError(
                "Inference grad wrapper body must return a tensor tuple"
            )
        for user in users:
            index = user.args[1]
            if index < 0 or index >= len(output):
                raise VulkanGraphExecutionError(
                    "Inference grad wrapper output index is out of range"
                )
            user.replace_all_uses_with(output[index])
            graph.erase_node(user)
        graph.erase_node(node)
        if not submodule_node.users:
            graph.erase_node(submodule_node)
        inlined += 1
    if inlined:
        graph.eliminate_dead_code()
        graph_module.delete_all_unused_submodules()
        graph.lint()
        graph_module.recompile()
    return inlined


def _is_python_scalar_or_symbolic(value: Any) -> bool:
    return isinstance(value, (bool, int, float, complex, *_SYMBOLIC_SCALAR_TYPES))


def _node_argument_is_python_scalar_or_symbolic(value: Any) -> bool:
    if isinstance(value, torch.fx.Node):
        return "val" in value.meta and _is_python_scalar_or_symbolic(
            value.meta["val"]
        )
    if isinstance(value, tuple | list):
        return all(_node_argument_is_python_scalar_or_symbolic(item) for item in value)
    return _is_python_scalar_or_symbolic(value)


def _is_python_scalar_arithmetic_node(node: torch.fx.Node) -> bool:
    return (
        node.target in _PYTHON_SCALAR_ARITHMETIC_TARGETS
        and "val" in node.meta
        and _is_python_scalar_or_symbolic(node.meta["val"])
        and all(
            _node_argument_is_python_scalar_or_symbolic(value)
            for value in (*node.args, *node.kwargs.values())
        )
    )


def _has_dispatch_kernel(operator_name: str, dispatch_key: str) -> bool:
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(
            operator_name, dispatch_key
        )
    except RuntimeError:
        return False


def _is_graph_owned_linear_context(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
) -> bool:
    if len(node.args) != 2:
        return False
    context_node = node.args[1]
    if not isinstance(context_node, torch.fx.Node) or context_node.op != "get_attr":
        return False
    context_attr = str(context_node.target)
    return context_attr.startswith("_vulkan_linear_context_") and hasattr(
        graph_module, context_attr
    )


def _is_graph_owned_conv2d_context(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
) -> bool:
    if len(node.args) != 2:
        return False
    context_node = node.args[1]
    if not isinstance(context_node, torch.fx.Node) or context_node.op != "get_attr":
        return False
    context_attr = str(context_node.target)
    return context_attr.startswith("_vulkan_conv2d_context_") and hasattr(
        graph_module, context_attr
    )


def _is_graph_owned_layernorm_context(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
) -> bool:
    if len(node.args) != 3 or node.kwargs:
        return False
    context_node = node.args[2]
    if not isinstance(context_node, torch.fx.Node) or context_node.op != "get_attr":
        return False
    context_attr = str(context_node.target)
    if not context_attr.startswith("_vulkan_layernorm_context_") or not hasattr(
        graph_module, context_attr
    ):
        return False
    return isinstance(getattr(graph_module, context_attr), torch.ScriptObject)


def _is_graph_owned_vulkan_graph_region_plan(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
) -> bool:
    if len(node.args) != 2 or node.kwargs:
        return False
    inputs, plan_node = node.args
    if not isinstance(inputs, tuple | list) or not inputs:
        return False
    if not isinstance(plan_node, torch.fx.Node) or plan_node.op != "get_attr":
        return False
    plan_attr = str(plan_node.target)
    if not plan_attr.startswith("_vulkan_graph_region_plan_") or not hasattr(
        graph_module, plan_attr
    ):
        return False
    plan = getattr(graph_module, plan_attr)
    return (
        isinstance(plan, torch.ScriptObject)
        and plan._type().qualified_name()
        == "__torch__.torch.classes.vulkan.VulkanGraphRegionPlan"
    )


def _is_graph_owned_static_add_layernorm_plan(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
) -> bool:
    if len(node.args) != 3 or node.kwargs:
        return False
    plan_node = node.args[2]
    if not isinstance(plan_node, torch.fx.Node) or plan_node.op != "get_attr":
        return False
    plan_attr = str(plan_node.target)
    if not plan_attr.startswith("_vulkan_static_add_layernorm_plan_") or not hasattr(
        graph_module, plan_attr
    ):
        return False
    return isinstance(getattr(graph_module, plan_attr), torch.ScriptObject)


def _is_graph_owned_static_conv2d_relu_plan(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
) -> bool:
    if len(node.args) != 2:
        return False
    plan_node = node.args[1]
    if not isinstance(plan_node, torch.fx.Node) or plan_node.op != "get_attr":
        return False
    plan_attr = str(plan_node.target)
    return plan_attr.startswith("_vulkan_static_conv2d_relu_plan_") and hasattr(
        graph_module, plan_attr
    )


def _classify_node(
    graph_module: torch.fx.GraphModule,
    index: int,
    node: torch.fx.Node,
) -> VulkanGraphNodeRecord:
    target = _target_name(node.target)
    if node.op in ("placeholder", "get_attr", "output"):
        return VulkanGraphNodeRecord(
            index,
            node.name,
            node.op,
            target,
            "graph",
            "graph_boundary_or_state",
        )
    if node.op == "call_function" and node.target is operator.getitem:
        return VulkanGraphNodeRecord(
            index,
            node.name,
            node.op,
            target,
            "graph",
            "python_graph_bookkeeping",
        )
    if node.op == "call_function" and _is_python_scalar_arithmetic_node(node):
        return VulkanGraphNodeRecord(
            index,
            node.name,
            node.op,
            target,
            "graph",
            "python_scalar_shape_arithmetic",
        )
    if node.op == "call_function" and isinstance(
        node.target, torch._ops.OpOverload
    ):
        operator_name = node.target.name()
        if operator_name == "aten::sym_size.int":
            return VulkanGraphNodeRecord(
                index,
                node.name,
                node.op,
                operator_name,
                "graph",
                "symbolic_shape_query",
            )
        if operator_name == "vulkan_prepack::run_linear_context":
            if not _is_graph_owned_linear_context(graph_module, node):
                return VulkanGraphNodeRecord(
                    index,
                    node.name,
                    node.op,
                    operator_name,
                    "unsupported",
                    "run_linear_context_missing_graph_owned_context",
                )
            return VulkanGraphNodeRecord(
                index,
                node.name,
                node.op,
                operator_name,
                "lowered_vulkan",
                "graph_owned_linear_context",
            )
        if operator_name == "vulkan_prepack::run_conv2d_context":
            if not _is_graph_owned_conv2d_context(graph_module, node):
                return VulkanGraphNodeRecord(
                    index,
                    node.name,
                    node.op,
                    operator_name,
                    "unsupported",
                    "run_conv2d_context_missing_graph_owned_context",
                )
            return VulkanGraphNodeRecord(
                index,
                node.name,
                node.op,
                operator_name,
                "lowered_vulkan",
                "graph_owned_conv2d_context",
            )
        if operator_name == "vulkan_prepack::run_layernorm_context":
            if not _is_graph_owned_layernorm_context(graph_module, node):
                return VulkanGraphNodeRecord(
                    index,
                    node.name,
                    node.op,
                    operator_name,
                    "unsupported",
                    "run_layernorm_context_missing_graph_owned_context",
                )
            return VulkanGraphNodeRecord(
                index,
                node.name,
                node.op,
                operator_name,
                "lowered_vulkan",
                "graph_owned_layernorm_context",
            )
        if operator_name == "vulkan_prepack::run_vulkan_graph_region_plan":
            if not _is_graph_owned_vulkan_graph_region_plan(graph_module, node):
                return VulkanGraphNodeRecord(
                    index,
                    node.name,
                    node.op,
                    operator_name,
                    "unsupported",
                    "run_vulkan_graph_region_plan_missing_graph_owned_plan",
                )
            return VulkanGraphNodeRecord(
                index,
                node.name,
                node.op,
                operator_name,
                "lowered_vulkan",
                "graph_owned_vulkan_graph_region_plan",
            )
        if operator_name == "vulkan_prepack::run_graph_add_layernorm_plan":
            if not _is_graph_owned_static_add_layernorm_plan(
                graph_module, node
            ):
                return VulkanGraphNodeRecord(
                    index,
                    node.name,
                    node.op,
                    operator_name,
                    "unsupported",
                    "run_graph_add_layernorm_plan_missing_graph_owned_plan",
                )
            return VulkanGraphNodeRecord(
                index,
                node.name,
                node.op,
                operator_name,
                "lowered_vulkan",
                "graph_owned_static_add_layernorm_plan",
            )
        if operator_name == "vulkan_prepack::run_graph_conv2d_relu_plan":
            if not _is_graph_owned_static_conv2d_relu_plan(graph_module, node):
                return VulkanGraphNodeRecord(
                    index,
                    node.name,
                    node.op,
                    operator_name,
                    "unsupported",
                    "run_graph_conv2d_relu_plan_missing_graph_owned_plan",
                )
            return VulkanGraphNodeRecord(
                index,
                node.name,
                node.op,
                operator_name,
                "lowered_vulkan",
                "graph_owned_static_conv2d_relu_plan",
            )
        if _has_dispatch_kernel(operator_name, "Vulkan"):
            return VulkanGraphNodeRecord(
                index,
                node.name,
                node.op,
                operator_name,
                "direct_vulkan",
                "registered_vulkan_kernel",
            )
        for dispatch_key in (
            "CompositeImplicitAutograd",
            "CompositeExplicitAutograd",
            "CompositeExplicitAutogradNonFunctional",
        ):
            if _has_dispatch_kernel(operator_name, dispatch_key):
                return VulkanGraphNodeRecord(
                    index,
                    node.name,
                    node.op,
                    operator_name,
                    "composite",
                    f"registered_{dispatch_key}",
                )
        return VulkanGraphNodeRecord(
            index,
            node.name,
            node.op,
            operator_name,
            "unsupported",
            "no_vulkan_or_composite_dispatch_kernel",
        )
    if node.op == "call_module" and node.target == "_guards_fn":
        if is_verified_exported_input_guard_call(graph_module, node):
            return VulkanGraphNodeRecord(
                index,
                node.name,
                node.op,
                target,
                "graph",
                "exported_input_guard",
            )
        return VulkanGraphNodeRecord(
            index,
            node.name,
            node.op,
            target,
            "unsupported",
            "unverified_exported_input_guard",
        )
    return VulkanGraphNodeRecord(
        index,
        node.name,
        node.op,
        target,
        "unsupported",
        "unsupported_graph_node_kind",
    )


def _build_census(graph_module: torch.fx.GraphModule) -> VulkanGraphCensus:
    nodes = tuple(
        _classify_node(graph_module, index, node)
        for index, node in enumerate(graph_module.graph.nodes)
    )
    return VulkanGraphCensus(
        captured_node_count=len(nodes),
        call_function_node_count=sum(node.op == "call_function" for node in nodes),
        direct_vulkan_node_count=sum(
            node.classification == "direct_vulkan" for node in nodes
        ),
        lowered_vulkan_node_count=sum(
            node.classification == "lowered_vulkan" for node in nodes
        ),
        composite_node_count=sum(
            node.classification == "composite" for node in nodes
        ),
        graph_node_count=sum(node.classification == "graph" for node in nodes),
        unsupported_node_count=sum(
            node.classification == "unsupported" for node in nodes
        ),
        nodes=nodes,
    )


def _normalize_example_inputs(example_inputs: Any) -> tuple[Any, ...]:
    if isinstance(example_inputs, tuple):
        return example_inputs
    return (example_inputs,)


def _validate_cpu_capture_inputs(args: Any, kwargs: Any) -> None:
    for value in pytree.tree_leaves((args, kwargs)):
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            raise ValueError(
                "torch.vulkan.export_and_lower expects CPU example tensors; "
                f"got {value.device}"
            )


def _export_input_signature(
    exported_program: torch.export.ExportedProgram,
) -> tuple[str, ...]:
    from torch.export.graph_signature import InputKind

    placeholders = {
        node.name: node
        for node in exported_program.graph.nodes
        if node.op == "placeholder"
    }
    signature: list[str] = []
    for input_spec in exported_program.graph_signature.input_specs:
        if input_spec.kind != InputKind.USER_INPUT:
            continue
        node = placeholders.get(input_spec.arg.name)
        value = None if node is None else node.meta.get("val")
        if isinstance(value, torch.Tensor):
            signature.append(
                "tensor:"
                f"dtype={value.dtype}:rank={value.dim()}:"
                f"shape={tuple(map(str, value.shape))}"
            )
        else:
            signature.append(
                f"value:{input_spec.arg.name}:{type(value).__qualname__}:{value!r}"
            )
    return tuple(signature)


def _freeze_cpu_state_dict_snapshot(
    exported_program: torch.export.ExportedProgram,
) -> tuple[dict[str, torch.Tensor], str]:
    snapshot: dict[str, torch.Tensor] = {}
    fingerprint = hashlib.sha256()
    for name, value in sorted(exported_program.state_dict.items()):
        if not isinstance(value, torch.Tensor):
            continue
        if value.device.type != "cpu":
            raise VulkanGraphExecutionError(
                "torch.vulkan.export_and_lower requires CPU state tensors; "
                f"state {name!r} is on {value.device}"
            )
        frozen = value.detach().contiguous().clone()
        snapshot[name] = frozen
        fingerprint.update(name.encode("utf-8"))
        fingerprint.update(b"\x00")
        fingerprint.update(str(frozen.dtype).encode("utf-8"))
        fingerprint.update(b"\x00")
        fingerprint.update(repr(tuple(frozen.shape)).encode("utf-8"))
        fingerprint.update(b"\x00")
        fingerprint.update(
            frozen.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return snapshot, fingerprint.hexdigest()


def _move_runtime_value(
    value: Any,
    device: torch.device,
    *,
    buffer_direct: bool = False,
) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    if value.device == device:
        return value
    if value.device.type != "cpu":
        raise VulkanGraphExecutionError(
            f"VulkanGraphProgram cannot move an input from {value.device}; "
            "runtime tensors must be on CPU or the program Vulkan device"
        )
    if buffer_direct:
        try:
            return torch.ops.vulkan_prepack.upload_graph_tensor_to_buffer(
                value, device
            )
        except (AttributeError, RuntimeError, TypeError) as error:
            raise VulkanGraphExecutionError(
                "VulkanGraphProgram direct-buffer input upload failed: "
                f"{error}"
            ) from error
    return value.to(device)


def _move_lowered_graph_module_to_device(
    graph_module: torch.fx.GraphModule,
    device: torch.device,
    tensor_placement: VulkanGraphTensorPlacementReport,
) -> None:
    for attr in tensor_placement.buffer_constant_attrs:
        value = getattr(graph_module, attr, None)
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise VulkanGraphExecutionError(
                "Vulkan graph direct-buffer constant placement requires a CPU "
                f"tensor at {attr!r}"
            )
        try:
            setattr(
                graph_module,
                attr,
                torch.ops.vulkan_prepack.upload_graph_tensor_to_buffer(value, device),
            )
        except (AttributeError, RuntimeError, TypeError) as error:
            raise VulkanGraphExecutionError(
                f"Vulkan graph direct-buffer constant upload failed for {attr!r}: "
                f"{error}"
            ) from error
    graph_module.to(device)
    for module in graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if "device" in node.kwargs:
                kwargs = dict(node.kwargs)
                kwargs["device"] = str(device)
                node.kwargs = kwargs
            if node.op == "call_function" and node.target is torch.ops.aten.to.device:
                args = list(node.args)
                args[1] = str(device)
                node.args = tuple(args)
            if "val" in node.meta:
                node.meta["val"] = pytree.tree_map(
                    lambda value: value.to(device)
                    if isinstance(value, torch.Tensor)
                    else value,
                    node.meta["val"],
                )
        module.graph.lint()
        module.recompile()


def _graph_placeholder_names(graph_module: torch.fx.GraphModule) -> tuple[str, ...]:
    return tuple(
        str(node.target)
        for node in graph_module.graph.nodes
        if node.op == "placeholder"
    )


def _move_graph_runtime_inputs_to_device(
    graph_module: torch.fx.GraphModule,
    values: tuple[Any, ...],
    device: torch.device,
    tensor_placement: VulkanGraphTensorPlacementReport,
) -> tuple[Any, ...]:
    buffer_placeholders = set(tensor_placement.buffer_placeholder_names)
    return tuple(
        pytree.tree_map(
            lambda leaf: _move_runtime_value(
                leaf,
                device,
                buffer_direct=placeholder_name in buffer_placeholders,
            ),
            value,
        )
        for placeholder_name, value in zip(
            _graph_placeholder_names(graph_module), values
        )
    )


def _normalize_graph_runtime_inputs(
    graph_module: torch.fx.GraphModule,
    values: tuple[Any, ...],
    report: VulkanGraphInputNormalizationReport,
) -> tuple[Any, ...]:
    rules = {
        node.placeholder_name: node
        for node in report.nodes
        if node.status == "lowered" and node.placeholder_name is not None
    }
    if not rules:
        return values
    normalized: list[Any] = []
    for placeholder_name, value in zip(_graph_placeholder_names(graph_module), values):
        rule = rules.get(placeholder_name)
        if rule is None:
            normalized.append(value)
            continue
        if (
            not isinstance(value, torch.Tensor)
            or rule.source_dtype is None
            or rule.target_dtype is None
        ):
            raise VulkanGraphExecutionError(
                "VulkanGraphProgram input normalization requires a tensor with "
                f"recorded dtypes for placeholder {placeholder_name!r}"
            )
        if value.device.type != "cpu":
            raise VulkanGraphExecutionError(
                f"VulkanGraphProgram input {placeholder_name!r} requires the "
                f"original CPU dtype {rule.source_dtype} before graph input "
                f"normalization to {rule.target_dtype}; got {value.dtype} on "
                f"{value.device}"
            )
        if value.dtype != rule.source_dtype:
            raise VulkanGraphExecutionError(
                f"VulkanGraphProgram input {placeholder_name!r} requires CPU "
                f"dtype {rule.source_dtype} before graph input normalization to "
                f"{rule.target_dtype}; got {value.dtype}"
            )
        normalized.append(value.to(dtype=rule.target_dtype))
    return tuple(normalized)


def _tensor_devices(value: Any) -> set[torch.device]:
    return {
        leaf.device
        for leaf in pytree.tree_leaves(value)
        if isinstance(leaf, torch.Tensor)
    }


def _begin_graph_execution_scope() -> int:
    try:
        return int(torch.ops.vulkan_prepack.begin_graph_execution_scope())
    except (AttributeError, RuntimeError, TypeError) as error:
        raise VulkanGraphExecutionError(
            f"Vulkan graph execution scope begin failed: {error}"
        ) from error


def _end_graph_execution_scope(token: int) -> tuple[int, int, int]:
    try:
        counters = tuple(
            int(value)
            for value in torch.ops.vulkan_prepack.end_graph_execution_scope(token)
        )
    except (AttributeError, RuntimeError, TypeError) as error:
        raise VulkanGraphExecutionError(
            f"Vulkan graph execution scope end failed: {error}"
        ) from error
    if len(counters) != 3 or any(value < 0 for value in counters):
        raise VulkanGraphExecutionError(
            "Vulkan graph execution scope returned invalid fallback counters: "
            f"{counters}"
        )
    return counters


def _bind_runtime_inputs(
    graph_module: torch.fx.GraphModule,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[Any, ...]:
    signature = inspect.signature(graph_module.forward)
    parameters = tuple(signature.parameters.values())
    if any(
        parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in parameters
    ):
        raise VulkanGraphExecutionError(
            "VulkanGraphProgram cannot prove runtime binding for variadic "
            "GraphModule.forward"
        )
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError as error:
        raise VulkanGraphExecutionError(
            f"VulkanGraphProgram input binding failed: {error}"
        ) from error
    bound.apply_defaults()
    parameter_names = tuple(parameter.name for parameter in parameters)
    placeholder_names = _graph_placeholder_names(graph_module)
    if placeholder_names != parameter_names:
        raise VulkanGraphExecutionError(
            "VulkanGraphProgram cannot prove GraphModule placeholder order: "
            f"forward={parameter_names}, graph={placeholder_names}"
        )
    return tuple(bound.arguments[name] for name in parameter_names)


def _linear_lowering_rejection_message(
    report: VulkanLinearLoweringReport,
) -> str:
    rejected = [node for node in report.nodes if node.status == "rejected"]
    return "\n".join(
        f"{node.node_name}: {node.reason}"
        for node in rejected
    )


def _conv2d_lowering_rejection_message(
    report: VulkanConv2dLoweringReport,
) -> str:
    rejected = [node for node in report.nodes if node.status == "rejected"]
    return "\n".join(
        f"{node.node_name}: {node.reason}"
        for node in rejected
    )


def _layernorm_lowering_rejection_message(
    report: VulkanLayernormLoweringReport,
) -> str:
    rejected = [node for node in report.nodes if node.status == "rejected"]
    return "\n".join(
        f"{node.node_name}: {node.reason}"
        for node in rejected
    )


def _static_linear_gelu_region_rejection_message(
    report: VulkanStaticLinearGeluRegionReport,
) -> str:
    rejected = [node for node in report.nodes if node.status == "rejected"]
    return "\n".join(
        f"{node.node_name}: {node.reason}"
        for node in rejected
    )


def _static_add_layernorm_region_rejection_message(
    report: VulkanStaticAddLayernormRegionReport,
) -> str:
    rejected = [node for node in report.nodes if node.status == "rejected"]
    return "\n".join(
        f"{node.node_name}: {node.reason}"
        for node in rejected
    )


def _static_conv2d_relu_region_rejection_message(
    report: VulkanStaticConv2dReluRegionReport,
) -> str:
    rejected = [node for node in report.nodes if node.status == "rejected"]
    return "\n".join(
        f"{node.node_name}: {node.reason}"
        for node in rejected
    )


def _static_conv2d_relu_conv2d_region_rejection_message(
    report: VulkanStaticConv2dReluConv2dRegionReport,
) -> str:
    rejected = [node for node in report.nodes if node.status == "rejected"]
    return "\n".join(
        f"{node.node_name}: {node.reason}"
        for node in rejected
    )


def _static_factory_constant_rejection_message(
    report: VulkanStaticFactoryConstantReport,
) -> str:
    rejected = [node for node in report.nodes if node.status == "rejected"]
    return "\n".join(
        f"{node.node_name}: {node.reason}"
        for node in rejected
    )


def _tensor_leaves(value: Any) -> tuple[torch.Tensor, ...]:
    return tuple(
        leaf
        for leaf in pytree.tree_leaves(value)
        if isinstance(leaf, torch.Tensor)
    )


class _VulkanGraphInterpreter(torch.fx.Interpreter):
    def __init__(self, module: torch.fx.GraphModule, device: torch.device) -> None:
        super().__init__(module)
        self.device = device
        self.executed_nodes: list[str] = []
        self.last_implicit_boundary: VulkanGraphImplicitBoundaryAttribution | None = (
            None
        )

    def run_node(self, node: torch.fx.Node) -> Any:
        is_dispatch = node.op in ("call_function", "call_method", "call_module")
        scope_token = _begin_graph_execution_scope() if is_dispatch else None
        node_error: Exception | None = None
        result: Any = None
        try:
            result = super().run_node(node)
        except Exception as error:
            node_error = error
        finally:
            counters = (
                _end_graph_execution_scope(scope_token)
                if scope_token is not None
                else (0, 0, 0)
            )
        if is_dispatch and any(counters):
            attribution = VulkanGraphImplicitBoundaryAttribution(
                node_name=node.name,
                target=_target_name(node.target),
                cpu_fallback_count=counters[0],
                sync_readback_count=counters[1],
                deferred_values_created=counters[2],
            )
            self.last_implicit_boundary = attribution
            error = _implicit_boundary_error(attribution)
            if node_error is not None:
                raise error from node_error
            raise error
        if node_error is not None:
            raise VulkanGraphExecutionError(
                f"Vulkan graph node {node.name!r} ({_target_name(node.target)}) "
                f"failed: {node_error}"
            ) from node_error
        if is_dispatch:
            devices = _tensor_devices(result)
            non_vulkan = {device for device in devices if device.type != "vulkan"}
            if non_vulkan:
                raise VulkanGraphExecutionError(
                    f"Vulkan graph node {node.name!r} ({_target_name(node.target)}) "
                    f"produced tensors on {sorted(map(str, non_vulkan))}; explicit "
                    "CPU partitions are not implemented"
                )
            self.executed_nodes.append(node.name)
        return result


class VulkanGraphProgram:
    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        exported_input_guard: torch.nn.Module | None,
        device: torch.device,
        key: VulkanGraphProgramKey,
        census: VulkanGraphCensus,
        input_normalization: VulkanGraphInputNormalizationReport,
        static_factory_constants: VulkanStaticFactoryConstantReport,
        lifted_tensor_constants: VulkanLiftedTensorConstantReport,
        static_identity_advanced_indices: VulkanStaticIdentityAdvancedIndexReport,
        static_gqa_repeats: VulkanStaticGQARepeatReport,
        tensor_placement: VulkanGraphTensorPlacementReport,
        linear_lowering: VulkanLinearLoweringReport,
        static_linear_gelu_regions: VulkanStaticLinearGeluRegionReport,
        conv2d_lowering: VulkanConv2dLoweringReport,
        layernorm_lowering: VulkanLayernormLoweringReport,
        static_add_layernorm_regions: VulkanStaticAddLayernormRegionReport,
        static_conv2d_relu_conv2d_regions: VulkanStaticConv2dReluConv2dRegionReport,
        static_conv2d_relu_regions: VulkanStaticConv2dReluRegionReport,
        vulkan_graph_regions: VulkanGraphRegionLoweringReport,
        cpp_plan: Any | None,
        cpp_plan_report: VulkanGraphPlanReport,
    ) -> None:
        self._graph_module = graph_module
        self._exported_input_guard = exported_input_guard
        self._device = device
        self._key = key
        self._census = census
        self._input_normalization = input_normalization
        self._static_factory_constants = static_factory_constants
        self._lifted_tensor_constants = lifted_tensor_constants
        self._static_identity_advanced_indices = static_identity_advanced_indices
        self._static_gqa_repeats = static_gqa_repeats
        self._tensor_placement = tensor_placement
        self._linear_lowering = linear_lowering
        self._static_linear_gelu_regions = static_linear_gelu_regions
        self._conv2d_lowering = conv2d_lowering
        self._layernorm_lowering = layernorm_lowering
        self._static_add_layernorm_regions = static_add_layernorm_regions
        self._static_conv2d_relu_conv2d_regions = (
            static_conv2d_relu_conv2d_regions
        )
        self._static_conv2d_relu_regions = static_conv2d_relu_regions
        self._vulkan_graph_regions = vulkan_graph_regions
        self._cpp_plan = cpp_plan
        self._cpp_plan_report = cpp_plan_report
        self._run_count = 0
        self._last_executed_nodes: tuple[str, ...] = ()
        self._last_cpu_fallback_count = 0
        self._last_sync_readback_count = 0
        self._last_deferred_values_created = 0
        self._last_implicit_boundary: VulkanGraphImplicitBoundaryAttribution | None = (
            None
        )
        self._execution_lock = threading.RLock()

    @property
    def graph_module(self) -> torch.fx.GraphModule:
        return self._graph_module

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def key(self) -> VulkanGraphProgramKey:
        return self._key

    @property
    def census(self) -> VulkanGraphCensus:
        return self._census

    @property
    def input_normalization(self) -> VulkanGraphInputNormalizationReport:
        return self._input_normalization

    @property
    def static_factory_constants(self) -> VulkanStaticFactoryConstantReport:
        return self._static_factory_constants

    @property
    def lifted_tensor_constants(self) -> VulkanLiftedTensorConstantReport:
        return self._lifted_tensor_constants

    @property
    def static_identity_advanced_indices(
        self,
    ) -> VulkanStaticIdentityAdvancedIndexReport:
        return self._static_identity_advanced_indices

    @property
    def static_gqa_repeats(self) -> VulkanStaticGQARepeatReport:
        return self._static_gqa_repeats

    @property
    def tensor_placement(self) -> VulkanGraphTensorPlacementReport:
        return self._tensor_placement

    @property
    def linear_lowering(self) -> VulkanLinearLoweringReport:
        return self._linear_lowering

    @property
    def static_linear_gelu_regions(self) -> VulkanStaticLinearGeluRegionReport:
        return self._static_linear_gelu_regions

    @property
    def conv2d_lowering(self) -> VulkanConv2dLoweringReport:
        return self._conv2d_lowering

    @property
    def layernorm_lowering(self) -> VulkanLayernormLoweringReport:
        return self._layernorm_lowering

    @property
    def static_add_layernorm_regions(self) -> VulkanStaticAddLayernormRegionReport:
        return self._static_add_layernorm_regions

    @property
    def static_conv2d_relu_conv2d_regions(
        self,
    ) -> VulkanStaticConv2dReluConv2dRegionReport:
        return self._static_conv2d_relu_conv2d_regions

    @property
    def static_conv2d_relu_regions(self) -> VulkanStaticConv2dReluRegionReport:
        return self._static_conv2d_relu_regions

    @property
    def vulkan_graph_regions(self) -> VulkanGraphRegionLoweringReport:
        return self._vulkan_graph_regions

    @property
    def cpp_plan(self) -> Any | None:
        return self._cpp_plan

    @property
    def cpp_plan_report(self) -> VulkanGraphPlanReport:
        return self._cpp_plan_report

    @property
    def execution_mode(self) -> str:
        return (
            "cpp_plan"
            if self._cpp_plan_report.status == "compiled"
            else "python_correctness_executor"
        )

    @property
    def run_count(self) -> int:
        return self._run_count

    @property
    def last_executed_nodes(self) -> tuple[str, ...]:
        return self._last_executed_nodes

    @property
    def last_cpu_fallback_count(self) -> int:
        return self._last_cpu_fallback_count

    @property
    def last_sync_readback_count(self) -> int:
        return self._last_sync_readback_count

    @property
    def last_deferred_values_created(self) -> int:
        return self._last_deferred_values_created

    @property
    def last_implicit_boundary(
        self,
    ) -> VulkanGraphImplicitBoundaryAttribution | None:
        return self._last_implicit_boundary

    def _reset_last_run_diagnostics(self) -> None:
        self._last_executed_nodes = ()
        self._last_cpu_fallback_count = 0
        self._last_sync_readback_count = 0
        self._last_deferred_values_created = 0
        self._last_implicit_boundary = None

    def _run_exported_input_guard(self, bound_args: tuple[Any, ...]) -> None:
        if self._exported_input_guard is None:
            return
        try:
            with torch.inference_mode():
                self._exported_input_guard(*bound_args)
        except Exception as error:
            raise VulkanGraphExecutionError(
                f"VulkanGraphProgram exported input guard failed: {error}"
            ) from error

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self._execution_lock:
            self._reset_last_run_diagnostics()
            bound_args = _bind_runtime_inputs(self._graph_module, args, kwargs)
            self._run_exported_input_guard(bound_args)
            normalized_args = _normalize_graph_runtime_inputs(
                self._graph_module,
                bound_args,
                self._input_normalization,
            )
            with torch.vulkan.device(self._device):
                moved_args = _move_graph_runtime_inputs_to_device(
                    self._graph_module,
                    normalized_args,
                    self._device,
                    self._tensor_placement,
                )
                interpreter: _VulkanGraphInterpreter | None = None
                scope_token = _begin_graph_execution_scope()
                try:
                    with torch.inference_mode():
                        if self._cpp_plan is not None:
                            try:
                                flat_output = (
                                    torch.ops.vulkan_prepack.run_vulkan_graph_plan.default(
                                        list(moved_args), self._cpp_plan
                                    )
                                )
                            except Exception as error:
                                attribution = _cpp_plan_implicit_boundary(error)
                                if attribution is not None:
                                    self._last_implicit_boundary = attribution
                                    raise _implicit_boundary_error(
                                        attribution
                                    ) from error
                                raise VulkanGraphExecutionError(
                                    "Vulkan C++ graph plan execution failed: "
                                    f"{error}"
                                ) from error
                            output = self._graph_module.graph.process_outputs(
                                tuple(flat_output)
                            )
                            self._last_executed_nodes = (
                                self._cpp_plan_report.node_names
                            )
                        else:
                            interpreter = _VulkanGraphInterpreter(
                                self._graph_module, self._device
                            )
                            output = interpreter.run(*moved_args)
                finally:
                    (
                        self._last_cpu_fallback_count,
                        self._last_sync_readback_count,
                        self._last_deferred_values_created,
                    ) = _end_graph_execution_scope(scope_token)
                    if interpreter is not None:
                        self._last_executed_nodes = tuple(
                            interpreter.executed_nodes
                        )
                        self._last_implicit_boundary = (
                            interpreter.last_implicit_boundary
                        )

            if (
                self._last_cpu_fallback_count
                or self._last_sync_readback_count
                or self._last_deferred_values_created
            ):
                raise VulkanGraphExecutionError(
                    "Vulkan graph execution crossed an implicit host boundary: "
                    f"cpu_fallback={self._last_cpu_fallback_count}, "
                    f"sync_readback={self._last_sync_readback_count}, "
                    f"deferred_values_created={self._last_deferred_values_created}. "
                    "Explicit CPU partitions and deferred values are not implemented"
                )
            output_tensors = _tensor_leaves(output)
            if not output_tensors:
                raise VulkanGraphExecutionError(
                    "VulkanGraphProgram output must contain at least one Vulkan tensor"
                )
            non_vulkan = {
                tensor.device
                for tensor in output_tensors
                if tensor.device.type != "vulkan"
            }
            if non_vulkan:
                raise VulkanGraphExecutionError(
                    "VulkanGraphProgram produced non-Vulkan tensor outputs on "
                    f"{sorted(map(str, non_vulkan))}"
                )
            self._run_count += 1
            return output


def export_and_lower(
    model: torch.nn.Module,
    example_inputs: Any,
    *,
    example_kwargs: Mapping[str, Any] | None = None,
    dynamic_shapes: Any = None,
    device: Any = None,
    fallback_policy: str = "error",
) -> VulkanGraphProgram:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("torch.vulkan.export_and_lower expects an nn.Module")
    if model.training:
        raise ValueError("torch.vulkan.export_and_lower is inference-only")
    if fallback_policy != "error":
        raise ValueError(
            "Only fallback_policy='error' is implemented; CPU partitions must "
            "be explicit before another policy is exposed"
        )
    if not torch.vulkan.is_available():
        raise RuntimeError("No Vulkan devices are available")

    args = _normalize_example_inputs(example_inputs)
    kwargs = dict(example_kwargs or {})
    _validate_cpu_capture_inputs(args, kwargs)
    device_index = torch.vulkan._get_device_index(device, optional=True)
    target_device = torch.device("vulkan", device_index)

    exported_program = torch.export.export(
        model,
        args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    cpu_state_snapshot, state_fingerprint = _freeze_cpu_state_dict_snapshot(
        exported_program
    )
    graph_module = exported_program.module()
    _inline_inference_grad_wrappers(graph_module)
    exported_input_guard = extract_verified_exported_input_guard(graph_module)
    input_normalization = lower_graph_input_dtype_normalizations(graph_module)
    static_factory_constants = lower_static_factory_constants(graph_module)
    lifted_tensor_constants = lower_lifted_tensor_constants(graph_module)
    static_identity_advanced_indices = lower_static_identity_advanced_indices(
        graph_module
    )
    static_gqa_repeats = lower_static_gqa_repeats(graph_module)
    tensor_placement = plan_graph_tensor_placements(
        graph_module,
        input_normalization,
        static_factory_constants,
        lifted_tensor_constants,
    )
    with torch.vulkan.device(target_device):
        linear_lowering = lower_static_linear_to_vulkan_contexts(
            graph_module, cpu_state_snapshot
        )
        static_linear_gelu_regions = lower_static_linear_gelu_regions(
            graph_module
        )
        conv2d_lowering = lower_static_conv2d_to_vulkan_contexts(
            graph_module, cpu_state_snapshot
        )
        layernorm_lowering = lower_static_layernorm_to_vulkan_contexts(
            graph_module, cpu_state_snapshot
        )
        static_add_layernorm_regions = lower_static_add_layernorm_regions(
            graph_module
        )
        static_conv2d_relu_conv2d_regions = (
            lower_static_conv2d_relu_conv2d_regions(graph_module)
        )
        static_conv2d_relu_regions = lower_static_conv2d_relu_regions(
            graph_module,
            static_conv2d_relu_conv2d_regions.excluded_relu_node_names,
        )
        vulkan_graph_regions = make_vulkan_graph_region_lowering_report(
            static_linear_gelu_regions,
            static_conv2d_relu_conv2d_regions,
        )
        _move_lowered_graph_module_to_device(
            graph_module,
            target_device,
            tensor_placement,
        )
    lowering_rejections: list[str] = []
    if static_factory_constants.rejected_count:
        lowering_rejections.append(
            "static_factory_constants:\n"
            + _static_factory_constant_rejection_message(static_factory_constants)
        )
    if linear_lowering.rejected_count:
        lowering_rejections.append(
            "linear:\n" + _linear_lowering_rejection_message(linear_lowering)
        )
    if static_linear_gelu_regions.rejected_count:
        lowering_rejections.append(
            "static_linear_gelu_regions:\n"
            + _static_linear_gelu_region_rejection_message(
                static_linear_gelu_regions
            )
        )
    if conv2d_lowering.rejected_count:
        lowering_rejections.append(
            "conv2d:\n" + _conv2d_lowering_rejection_message(conv2d_lowering)
        )
    if layernorm_lowering.rejected_count:
        lowering_rejections.append(
            "layernorm:\n" + _layernorm_lowering_rejection_message(layernorm_lowering)
        )
    if static_add_layernorm_regions.rejected_count:
        lowering_rejections.append(
            "static_add_layernorm_regions:\n"
            + _static_add_layernorm_region_rejection_message(
                static_add_layernorm_regions
            )
        )
    if static_conv2d_relu_conv2d_regions.rejected_count:
        lowering_rejections.append(
            "static_conv2d_relu_conv2d_regions:\n"
            + _static_conv2d_relu_conv2d_region_rejection_message(
                static_conv2d_relu_conv2d_regions
            )
        )
    if static_conv2d_relu_regions.rejected_count:
        lowering_rejections.append(
            "static_conv2d_relu_regions:\n"
            + _static_conv2d_relu_region_rejection_message(
                static_conv2d_relu_regions
            )
        )
    if lowering_rejections:
        raise VulkanGraphExecutionError(
            "Exported graph contains rejected static lowerings:\n"
            + "\n".join(lowering_rejections)
        )

    census = _build_census(graph_module)
    if census.unsupported_node_count:
        unsupported = [
            f"{node.name} ({node.target}): {node.reason}"
            for node in census.nodes
            if node.classification == "unsupported"
        ]
        raise VulkanGraphExecutionError(
            "Exported graph contains unsupported Vulkan nodes:\n"
            + "\n".join(unsupported)
        )
    cpp_plan_compilation = compile_vulkan_graph_plan(
        graph_module,
        {node.name: node.classification for node in census.nodes},
    )

    graph_fingerprint = "\n".join(
        (
            graph_module.code,
            str(exported_program.graph_signature),
            str(exported_program.range_constraints),
            repr(input_normalization),
            repr(static_factory_constants),
            repr(lifted_tensor_constants),
            repr(static_identity_advanced_indices),
            repr(static_gqa_repeats),
            repr(tensor_placement),
            repr(linear_lowering),
            repr(static_linear_gelu_regions),
            repr(conv2d_lowering),
            repr(layernorm_lowering),
            repr(static_add_layernorm_regions),
            repr(static_conv2d_relu_conv2d_regions),
            repr(static_conv2d_relu_regions),
            repr(vulkan_graph_regions),
            repr(cpp_plan_compilation.report),
        )
    )
    properties = torch.vulkan.get_device_properties(target_device)
    key = VulkanGraphProgramKey(
        graph_hash=hashlib.sha256(graph_fingerprint.encode("utf-8")).hexdigest(),
        state_fingerprint=state_fingerprint,
        input_signature=_export_input_signature(exported_program),
        device_index=device_index,
        vendor_id=properties.vendor_id,
        device_id=properties.device_id,
        driver_version=properties.driver_version,
        api_version=properties.api_version,
    )
    return VulkanGraphProgram(
        graph_module,
        exported_input_guard,
        target_device,
        key,
        census,
        input_normalization,
        static_factory_constants,
        lifted_tensor_constants,
        static_identity_advanced_indices,
        static_gqa_repeats,
        tensor_placement,
        linear_lowering,
        static_linear_gelu_regions,
        conv2d_lowering,
        layernorm_lowering,
        static_add_layernorm_regions,
        static_conv2d_relu_conv2d_regions,
        static_conv2d_relu_regions,
        vulkan_graph_regions,
        cpp_plan_compilation.plan,
        cpp_plan_compilation.report,
    )


__all__ = [
    "VulkanGraphCensus",
    "VulkanGraphExecutionError",
    "VulkanGraphImplicitBoundaryAttribution",
    "VulkanGraphInputNormalizationReport",
    "VulkanGraphTensorPlacementReport",
    "VulkanLiftedTensorConstantReport",
    "VulkanStaticIdentityAdvancedIndexReport",
    "VulkanStaticGQARepeatReport",
    "VulkanGraphNodeRecord",
    "VulkanGraphProgram",
    "VulkanGraphProgramKey",
    "VulkanGraphPlanReport",
    "VulkanConv2dLoweringReport",
    "VulkanLayernormLoweringReport",
    "VulkanStaticAddLayernormRegionReport",
    "VulkanLinearLoweringReport",
    "VulkanStaticConv2dReluConv2dRegionReport",
    "VulkanStaticConv2dReluRegionReport",
    "VulkanStaticFactoryConstantReport",
    "VulkanStaticLinearGeluRegionReport",
    "export_and_lower",
]
