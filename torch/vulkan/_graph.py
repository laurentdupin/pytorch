# mypy: allow-untyped-defs

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import operator
import threading
from collections.abc import Mapping
from typing import Any

import torch
import torch.utils._pytree as pytree

from ._graph_lowering import (
    VulkanLinearLoweringReport,
    lower_static_linear_to_vulkan_contexts,
)


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


class VulkanGraphExecutionError(RuntimeError):
    pass


_VULKAN_GRAPH_EXECUTION_LOCK = threading.RLock()


def _target_name(target: Any) -> str:
    name = getattr(target, "name", None)
    if callable(name):
        return str(name())
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return str(target)


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
    if node.op == "call_function" and isinstance(
        node.target, torch._ops.OpOverload
    ):
        operator_name = node.target.name()
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


def _move_runtime_value(value: Any, device: torch.device) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    if value.device == device:
        return value
    if value.device.type != "cpu":
        raise VulkanGraphExecutionError(
            f"VulkanGraphProgram cannot move an input from {value.device}; "
            "runtime tensors must be on CPU or the program Vulkan device"
        )
    return value.to(device)


def _tensor_devices(value: Any) -> set[torch.device]:
    return {
        leaf.device
        for leaf in pytree.tree_leaves(value)
        if isinstance(leaf, torch.Tensor)
    }


def _fallback_counters() -> tuple[int, int]:
    return (
        torch.ops.vulkan_prepack.cpu_fallback_count(),
        torch.ops.vulkan_prepack.sync_readback_count(),
    )


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
    placeholder_names = tuple(
        str(node.target)
        for node in graph_module.graph.nodes
        if node.op == "placeholder"
    )
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

    def run_node(self, node: torch.fx.Node) -> Any:
        try:
            result = super().run_node(node)
        except Exception as error:
            raise VulkanGraphExecutionError(
                f"Vulkan graph node {node.name!r} ({_target_name(node.target)}) "
                f"failed: {error}"
            ) from error
        if node.op in ("call_function", "call_method", "call_module"):
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
        device: torch.device,
        key: VulkanGraphProgramKey,
        census: VulkanGraphCensus,
        linear_lowering: VulkanLinearLoweringReport,
    ) -> None:
        self._graph_module = graph_module
        self._device = device
        self._key = key
        self._census = census
        self._linear_lowering = linear_lowering
        self._run_count = 0
        self._last_executed_nodes: tuple[str, ...] = ()
        self._last_cpu_fallback_count = 0
        self._last_sync_readback_count = 0

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
    def linear_lowering(self) -> VulkanLinearLoweringReport:
        return self._linear_lowering

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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        bound_args = _bind_runtime_inputs(self._graph_module, args, kwargs)
        moved_args = pytree.tree_map(
            lambda value: _move_runtime_value(value, self._device), bound_args
        )
        interpreter = _VulkanGraphInterpreter(self._graph_module, self._device)
        with _VULKAN_GRAPH_EXECUTION_LOCK:
            fallback_before, readback_before = _fallback_counters()
            try:
                with torch.vulkan.device(self._device), torch.inference_mode():
                    output = interpreter.run(*moved_args)
            finally:
                fallback_after, readback_after = _fallback_counters()
                self._last_cpu_fallback_count = fallback_after - fallback_before
                self._last_sync_readback_count = readback_after - readback_before
                self._last_executed_nodes = tuple(interpreter.executed_nodes)

            if (
                self._last_cpu_fallback_count < 0
                or self._last_sync_readback_count < 0
            ):
                raise VulkanGraphExecutionError(
                    "Vulkan graph fallback counters were reset during execution"
                )
            if self._last_cpu_fallback_count or self._last_sync_readback_count:
                raise VulkanGraphExecutionError(
                    "Vulkan graph execution crossed an implicit host boundary: "
                    f"cpu_fallback={self._last_cpu_fallback_count}, "
                    f"sync_readback={self._last_sync_readback_count}. Explicit CPU "
                    "partitions are not implemented"
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
    from torch.export.passes import move_to_device_pass

    with torch.vulkan.device(target_device):
        moved_exported_program = move_to_device_pass(
            exported_program, target_device
        )
        graph_module = moved_exported_program.module()
        linear_lowering = lower_static_linear_to_vulkan_contexts(
            graph_module, cpu_state_snapshot
        )
    if linear_lowering.rejected_count:
        raise VulkanGraphExecutionError(
            "Exported graph contains rejected static linear lowerings:\n"
            + _linear_lowering_rejection_message(linear_lowering)
        )

    graph_fingerprint = "\n".join(
        (
            graph_module.code,
            str(exported_program.graph_signature),
            str(exported_program.range_constraints),
            repr(linear_lowering),
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
    return VulkanGraphProgram(
        graph_module,
        target_device,
        key,
        census,
        linear_lowering,
    )


__all__ = [
    "VulkanGraphCensus",
    "VulkanGraphExecutionError",
    "VulkanGraphNodeRecord",
    "VulkanGraphProgram",
    "VulkanGraphProgramKey",
    "VulkanLinearLoweringReport",
    "export_and_lower",
]
