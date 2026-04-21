# mypy: allow-untyped-defs
r"""
Composite multi-device helpers for the Vulkan backend.

These APIs deliberately sit above real tensor devices. Tensors still live on a
single ``vulkan:i`` device, while composite execution is expressed through
device groups, meshes, and module wrappers.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from torch import Tensor, nn


def _resolve_vulkan_index(device: Any, *, optional: bool = False) -> int:
    if isinstance(device, int):
        return device

    if isinstance(device, str):
        if device == "cpu":
            raise ValueError("Expected a Vulkan device, but got CPU")
        if device == "input":
            return -1
        device = torch.device(device)

    if isinstance(device, torch.device):
        if device.type != "vulkan":
            raise ValueError(f"Expected a Vulkan device, but got: {device}")
        if device.index is not None:
            return device.index
        if optional:
            return torch.vulkan.current_device()
        raise ValueError(
            "Expected a Vulkan device with an explicit index, but got "
            f"{device!s}"
        )

    if device is None and optional:
        return torch.vulkan.current_device()

    raise ValueError(
        f"Expected a Vulkan device, integer index, or None, but got: {device}"
    )


def _normalize_output_device(output_device: Any) -> str | torch.device | None:
    if output_device is None:
        return None
    if output_device == "cpu":
        return "cpu"
    if output_device == "input":
        return "input"
    if isinstance(output_device, str):
        output_device = torch.device(output_device)
    if isinstance(output_device, torch.device):
        if output_device.type != "vulkan":
            raise ValueError(
                "Expected output_device to be 'cpu', 'input', None, or a "
                f"Vulkan device, but got: {output_device}"
            )
        if output_device.index is None:
            return torch.device("vulkan", torch.vulkan.current_device())
        return output_device
    if isinstance(output_device, int):
        return torch.device("vulkan", output_device)
    raise ValueError(
        "Expected output_device to be 'cpu', 'input', None, a Vulkan device, "
        f"or an integer index, but got: {output_device}"
    )


def _materialize_output_device(
    output_device: str | torch.device | None,
    reference: Any,
) -> str | torch.device | None:
    if output_device != "input":
        return output_device

    input_device = _find_tensor_device(reference)
    if input_device is None:
        return "cpu"
    return input_device


def _find_tensor_device(obj: Any) -> torch.device | None:
    if torch.is_tensor(obj):
        return obj.device
    if isinstance(obj, tuple):
        for value in obj:
            device = _find_tensor_device(value)
            if device is not None:
                return device
    if isinstance(obj, list):
        for value in obj:
            device = _find_tensor_device(value)
            if device is not None:
                return device
    if isinstance(obj, dict):
        for value in obj.values():
            device = _find_tensor_device(value)
            if device is not None:
                return device
    return None


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _move_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.cpu()
    if isinstance(value, tuple):
        return tuple(_move_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_move_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_cpu(item) for key, item in value.items()}
    return value


def _move_to_output_device(
    value: Any,
    output_device: str | torch.device | None,
    reference: Any,
) -> Any:
    resolved_output_device = _materialize_output_device(output_device, reference)
    if resolved_output_device in (None, "cpu"):
        return value
    return _move_to_device(value, resolved_output_device)


def _split_sizes(total_size: int, num_chunks: int) -> tuple[int, ...]:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    base_size, remainder = divmod(total_size, num_chunks)
    return tuple(base_size + (1 if chunk_index < remainder else 0) for chunk_index in range(num_chunks))


def _infer_chunk_count(value: Any, dim: int) -> int | None:
    if torch.is_tensor(value):
        if value.dim() <= dim:
            return None
        return value.size(dim)
    if isinstance(value, tuple):
        for item in value:
            chunk_count = _infer_chunk_count(item, dim)
            if chunk_count is not None:
                return chunk_count
    if isinstance(value, list):
        for item in value:
            chunk_count = _infer_chunk_count(item, dim)
            if chunk_count is not None:
                return chunk_count
    if isinstance(value, dict):
        for item in value.values():
            chunk_count = _infer_chunk_count(item, dim)
            if chunk_count is not None:
                return chunk_count
    return None


def _scatter_value(value: Any, num_chunks: int, dim: int) -> list[Any]:
    if torch.is_tensor(value):
        if value.dim() <= dim or value.size(dim) == 0:
            return [value for _ in range(num_chunks)]
        return list(torch.tensor_split(value, num_chunks, dim=dim))
    if isinstance(value, tuple):
        if not value:
            return [tuple() for _ in range(num_chunks)]
        scattered_items = [_scatter_value(item, num_chunks, dim) for item in value]
        return [tuple(items) for items in zip(*scattered_items)]
    if isinstance(value, list):
        if not value:
            return [[] for _ in range(num_chunks)]
        scattered_items = [_scatter_value(item, num_chunks, dim) for item in value]
        return [list(items) for items in zip(*scattered_items)]
    if isinstance(value, dict):
        scattered_items = {
            key: _scatter_value(item, num_chunks, dim) for key, item in value.items()
        }
        return [
            {key: scattered_items[key][chunk_index] for key in scattered_items}
            for chunk_index in range(num_chunks)
        ]
    return [value for _ in range(num_chunks)]


def _gather_outputs(outputs: list[Any], dim: int) -> Any:
    if not outputs:
        raise ValueError("Expected at least one output to gather")

    first = outputs[0]
    if torch.is_tensor(first):
        if len(outputs) == 1:
            return first
        if first.dim() == 0:
            return torch.stack(outputs, dim=0)
        return torch.cat(outputs, dim=dim)
    if isinstance(first, tuple):
        return tuple(
            _gather_outputs([output[index] for output in outputs], dim)
            for index in range(len(first))
        )
    if isinstance(first, list):
        return [
            _gather_outputs([output[index] for output in outputs], dim)
            for index in range(len(first))
        ]
    if isinstance(first, dict):
        return {
            key: _gather_outputs([output[key] for output in outputs], dim)
            for key in first
        }
    if len(outputs) == 1:
        return first
    return outputs


def _iter_module_tensors(module: nn.Module):
    yield from module.parameters(recurse=True)
    yield from module.buffers(recurse=True)


def _ensure_cpu_hosted_module(module: nn.Module, api_name: str) -> None:
    for tensor in _iter_module_tensors(module):
        if tensor.device.type == "meta":
            raise ValueError(
                f"{api_name} does not support meta-device modules. Materialize "
                "the module on CPU first."
            )
        if tensor.device.type != "cpu":
            raise ValueError(
                f"{api_name} expects a CPU-hosted module, but found a tensor on "
                f"{tensor.device}."
            )


def _build_vulkan_device_group(devices: Any) -> DeviceGroup:
    if isinstance(devices, DeviceGroup):
        return devices
    if isinstance(devices, DeviceMesh):
        return devices.group
    return DeviceGroup(devices)


def _validate_vulkan_devices(indices: tuple[int, ...]) -> None:
    if not torch.vulkan.is_available():
        raise RuntimeError("No Vulkan devices are available")

    device_count = torch.vulkan.device_count()
    if not indices:
        raise ValueError("Expected at least one Vulkan device")
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate Vulkan devices are not allowed: {indices}")

    for device_index in indices:
        if device_index < 0 or device_index >= device_count:
            raise ValueError(
                f"Invalid Vulkan device index {device_index}; this machine has "
                f"{device_count} Vulkan device(s)."
            )
        properties = torch.vulkan.get_device_properties(device_index)
        if properties.num_compute_queues < 1:
            raise ValueError(
                f"Vulkan device {device_index} ({properties.name}) does not "
                "expose a compute queue."
            )


class DeviceGroup:
    r"""Ordered collection of Vulkan devices used for composite execution."""

    def __init__(self, devices: Any) -> None:
        if isinstance(devices, DeviceGroup):
            indices = devices.indices
        else:
            if isinstance(devices, (int, str, torch.device)):
                devices = [devices]
            indices = tuple(_resolve_vulkan_index(device) for device in devices)
        _validate_vulkan_devices(indices)
        self._indices = indices

    @property
    def indices(self) -> tuple[int, ...]:
        return self._indices

    @property
    def devices(self) -> tuple[torch.device, ...]:
        return tuple(torch.device("vulkan", device_index) for device_index in self._indices)

    @property
    def properties(self):
        return tuple(torch.vulkan.get_device_properties(device_index) for device_index in self._indices)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(properties.name for properties in self.properties)

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self):
        return iter(self.devices)

    def __getitem__(self, index: int) -> torch.device:
        return self.devices[index]

    def __repr__(self) -> str:
        pairs = ", ".join(
            f"{device_index}:{torch.vulkan.get_device_name(device_index)!r}"
            for device_index in self._indices
        )
        return f"DeviceGroup([{pairs}])"


def device_group(devices: Any) -> DeviceGroup:
    return DeviceGroup(devices)


class DeviceMesh:
    r"""A 1-D Vulkan device mesh for tensor/model parallel orchestration."""

    def __init__(self, devices: Any) -> None:
        self._group = _build_vulkan_device_group(devices)

    @property
    def group(self) -> DeviceGroup:
        return self._group

    @property
    def device_type(self) -> str:
        return "vulkan"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def shape(self) -> tuple[int]:
        return (len(self._group),)

    @property
    def indices(self) -> tuple[int, ...]:
        return self._group.indices

    @property
    def devices(self) -> tuple[torch.device, ...]:
        return self._group.devices

    def __len__(self) -> int:
        return len(self._group)

    def __iter__(self):
        return iter(self._group)

    def __repr__(self) -> str:
        return f"DeviceMesh(device_type='vulkan', shape={self.shape}, indices={self.indices})"


def device_mesh(devices: Any) -> DeviceMesh:
    return DeviceMesh(devices)


class ParallelStyle:
    r"""Base class for Vulkan composite parallel styles."""

    pass


class Replicate(ParallelStyle):
    r"""Replicate a module on each device and split the batch dimension."""

    def __init__(
        self,
        *,
        dim: int = 0,
        output_device: Any = "cpu",
        parallel: bool = True,
    ) -> None:
        self.dim = dim
        self.output_device = _normalize_output_device(output_device)
        self.parallel = parallel

    def __repr__(self) -> str:
        return (
            f"Replicate(dim={self.dim}, output_device={self.output_device}, "
            f"parallel={self.parallel})"
        )


class ColwiseParallel(ParallelStyle):
    r"""Shard a Linear module over its output features."""

    def __init__(
        self,
        *,
        output_device: Any = "cpu",
        use_local_output: bool = True,
        parallel: bool = True,
    ) -> None:
        self.output_device = _normalize_output_device(output_device)
        self.use_local_output = use_local_output
        self.parallel = parallel

    def __repr__(self) -> str:
        return (
            "ColwiseParallel("
            f"output_device={self.output_device}, "
            f"use_local_output={self.use_local_output}, "
            f"parallel={self.parallel})"
        )


class RowwiseParallel(ParallelStyle):
    r"""Shard a Linear module over its input features."""

    def __init__(
        self,
        *,
        output_device: Any = "cpu",
        use_local_output: bool = True,
        parallel: bool = True,
    ) -> None:
        self.output_device = _normalize_output_device(output_device)
        self.use_local_output = use_local_output
        self.parallel = parallel

    def __repr__(self) -> str:
        return (
            "RowwiseParallel("
            f"output_device={self.output_device}, "
            f"use_local_output={self.use_local_output}, "
            f"parallel={self.parallel})"
        )


class _ParallelExecutionModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        super().train(False)

    def train(self, mode: bool = True):
        super().train(mode)
        for replica in getattr(self, "_replicas", []):
            replica.train(mode)
        return self


class _ReplicatedModule(_ParallelExecutionModule):
    def __init__(
        self,
        module: nn.Module,
        devices: Any,
        *,
        dim: int = 0,
        output_device: Any = "cpu",
        parallel: bool = True,
    ) -> None:
        super().__init__()
        _ensure_cpu_hosted_module(module, "torch.vulkan.replicate_module")
        self._device_group = _build_vulkan_device_group(devices)
        self._dim = dim
        self._output_device = _normalize_output_device(output_device)
        self._parallel = parallel
        self._replicas = []
        for device in self._device_group.devices:
            replica = copy.deepcopy(module).eval().to(device)
            self._replicas.append(replica)

    def _active_replica_count(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
        batch_size = _infer_chunk_count(args, self._dim)
        if batch_size is None:
            batch_size = _infer_chunk_count(kwargs, self._dim)
        if batch_size is None or batch_size <= 1:
            return 1
        return min(len(self._replicas), batch_size)

    def _run_replica(
        self,
        replica: nn.Module,
        device: torch.device,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        with torch.inference_mode(), torch.vulkan.device(device):
            local_args = _move_to_device(args, device)
            local_kwargs = _move_to_device(kwargs, device)
            output = replica(*local_args, **local_kwargs)
        return _move_to_cpu(output)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        active_count = self._active_replica_count(args, kwargs)
        active_replicas = self._replicas[:active_count]
        active_devices = self._device_group.devices[:active_count]

        scattered_args = _scatter_value(args, active_count, self._dim)
        scattered_kwargs = _scatter_value(kwargs, active_count, self._dim)

        if self._parallel and active_count > 1:
            with ThreadPoolExecutor(max_workers=active_count) as executor:
                futures = [
                    executor.submit(
                        self._run_replica,
                        replica,
                        device,
                        shard_args,
                        shard_kwargs,
                    )
                    for replica, device, shard_args, shard_kwargs in zip(
                        active_replicas,
                        active_devices,
                        scattered_args,
                        scattered_kwargs,
                    )
                ]
                outputs = [future.result() for future in futures]
        else:
            outputs = [
                self._run_replica(replica, device, shard_args, shard_kwargs)
                for replica, device, shard_args, shard_kwargs in zip(
                    active_replicas,
                    active_devices,
                    scattered_args,
                    scattered_kwargs,
                )
            ]

        gathered = _gather_outputs(outputs, self._dim)
        return _move_to_output_device(gathered, self._output_device, args)


class _ColwiseLinear(_ParallelExecutionModule):
    def __init__(
        self,
        linear: nn.Linear,
        devices: Any,
        *,
        output_device: Any = "cpu",
        parallel: bool = True,
        use_local_output: bool = True,
    ) -> None:
        super().__init__()
        if not use_local_output:
            raise NotImplementedError(
                "Vulkan ColwiseParallel currently only supports "
                "use_local_output=True"
            )
        _ensure_cpu_hosted_module(linear, "torch.vulkan.parallelize_module")
        self._device_group = _build_vulkan_device_group(devices)
        self._output_device = _normalize_output_device(output_device)
        self._parallel = parallel
        self._split_sizes = _split_sizes(linear.out_features, len(self._device_group))
        self._replicas = []

        start = 0
        for device, shard_size in zip(self._device_group.devices, self._split_sizes):
            replica = nn.Linear(
                linear.in_features,
                shard_size,
                bias=linear.bias is not None,
            )
            with torch.no_grad():
                replica.weight.copy_(linear.weight[start : start + shard_size])
                if linear.bias is not None:
                    replica.bias.copy_(linear.bias[start : start + shard_size])
            self._replicas.append(replica.eval().to(device))
            start += shard_size

    def _run_replica(self, replica: nn.Linear, device: torch.device, input_cpu: Tensor) -> Tensor:
        with torch.inference_mode(), torch.vulkan.device(device):
            output = replica(input_cpu.to(device))
        return output.cpu()

    def forward(self, input: Tensor) -> Tensor:
        input_cpu = input.cpu() if input.device.type == "vulkan" else input
        if self._parallel and len(self._replicas) > 1:
            with ThreadPoolExecutor(max_workers=len(self._replicas)) as executor:
                futures = [
                    executor.submit(self._run_replica, replica, device, input_cpu)
                    for replica, device in zip(self._replicas, self._device_group.devices)
                ]
                outputs = [future.result() for future in futures]
        else:
            outputs = [
                self._run_replica(replica, device, input_cpu)
                for replica, device in zip(self._replicas, self._device_group.devices)
            ]

        gathered = torch.cat(outputs, dim=-1)
        return _move_to_output_device(gathered, self._output_device, input)


class _RowwiseLinear(_ParallelExecutionModule):
    def __init__(
        self,
        linear: nn.Linear,
        devices: Any,
        *,
        output_device: Any = "cpu",
        parallel: bool = True,
        use_local_output: bool = True,
    ) -> None:
        super().__init__()
        if not use_local_output:
            raise NotImplementedError(
                "Vulkan RowwiseParallel currently only supports "
                "use_local_output=True"
            )
        _ensure_cpu_hosted_module(linear, "torch.vulkan.parallelize_module")
        self._device_group = _build_vulkan_device_group(devices)
        self._output_device = _normalize_output_device(output_device)
        self._parallel = parallel
        self._split_sizes = _split_sizes(linear.in_features, len(self._device_group))
        self._replicas = []
        self._bias = linear.bias.detach().clone() if linear.bias is not None else None

        start = 0
        for device, shard_size in zip(self._device_group.devices, self._split_sizes):
            replica = nn.Linear(shard_size, linear.out_features, bias=False)
            with torch.no_grad():
                replica.weight.copy_(linear.weight[:, start : start + shard_size])
            self._replicas.append(replica.eval().to(device))
            start += shard_size

    def _run_replica(self, replica: nn.Linear, device: torch.device, input_cpu: Tensor) -> Tensor:
        with torch.inference_mode(), torch.vulkan.device(device):
            output = replica(input_cpu.to(device))
        return output.cpu()

    def forward(self, input: Tensor) -> Tensor:
        input_cpu = input.cpu() if input.device.type == "vulkan" else input
        input_shards = torch.split(input_cpu, self._split_sizes, dim=-1)

        if self._parallel and len(self._replicas) > 1:
            with ThreadPoolExecutor(max_workers=len(self._replicas)) as executor:
                futures = [
                    executor.submit(self._run_replica, replica, device, shard_input)
                    for replica, device, shard_input in zip(
                        self._replicas,
                        self._device_group.devices,
                        input_shards,
                    )
                ]
                partial_outputs = [future.result() for future in futures]
        else:
            partial_outputs = [
                self._run_replica(replica, device, shard_input)
                for replica, device, shard_input in zip(
                    self._replicas,
                    self._device_group.devices,
                    input_shards,
                )
            ]

        output = partial_outputs[0]
        for partial_output in partial_outputs[1:]:
            output = output + partial_output
        if self._bias is not None:
            output = output + self._bias
        return _move_to_output_device(output, self._output_device, input)


def replicate_module(
    module: nn.Module,
    devices: Any,
    *,
    dim: int = 0,
    output_device: Any = "cpu",
    parallel: bool = True,
) -> nn.Module:
    r"""Replicate a CPU-hosted module over a Vulkan device group for inference."""

    return _ReplicatedModule(
        module,
        devices,
        dim=dim,
        output_device=output_device,
        parallel=parallel,
    )


def _get_submodule(module: nn.Module, fqn: str) -> nn.Module:
    current = module
    if not fqn:
        return current
    for atom in fqn.split("."):
        current = getattr(current, atom)
    return current


def _set_submodule(module: nn.Module, fqn: str, replacement: nn.Module) -> nn.Module:
    if not fqn:
        return replacement
    parent_path, _, child_name = fqn.rpartition(".")
    parent = _get_submodule(module, parent_path) if parent_path else module
    parent._modules[child_name] = replacement
    return module


def _apply_parallel_style(
    module: nn.Module,
    devices: Any,
    style: ParallelStyle,
) -> nn.Module:
    if isinstance(style, Replicate):
        return replicate_module(
            module,
            devices,
            dim=style.dim,
            output_device=style.output_device,
            parallel=style.parallel,
        )

    if isinstance(style, ColwiseParallel):
        if not isinstance(module, nn.Linear):
            raise TypeError(
                "Vulkan ColwiseParallel currently only supports nn.Linear "
                f"modules, but got: {type(module).__name__}"
            )
        return _ColwiseLinear(
            module,
            devices,
            output_device=style.output_device,
            parallel=style.parallel,
            use_local_output=style.use_local_output,
        )

    if isinstance(style, RowwiseParallel):
        if not isinstance(module, nn.Linear):
            raise TypeError(
                "Vulkan RowwiseParallel currently only supports nn.Linear "
                f"modules, but got: {type(module).__name__}"
            )
        return _RowwiseLinear(
            module,
            devices,
            output_device=style.output_device,
            parallel=style.parallel,
            use_local_output=style.use_local_output,
        )

    raise TypeError(f"Unsupported Vulkan parallel style: {style!r}")


def parallelize_module(
    module: nn.Module,
    device_mesh: Any = None,
    parallelize_plan: ParallelStyle | dict[str, ParallelStyle] | None = None,
    *,
    src_data_rank: int | None = 0,
) -> nn.Module:
    r"""Parallelize CPU-hosted modules over a 1-D Vulkan device mesh."""

    if parallelize_plan is None:
        return module

    if src_data_rank not in (0, None):
        raise ValueError(
            "Vulkan parallelize_module currently only supports src_data_rank "
            "of 0 or None."
        )

    mesh = device_mesh if isinstance(device_mesh, DeviceMesh) else DeviceMesh(device_mesh)

    if isinstance(parallelize_plan, ParallelStyle):
        module_copy = copy.deepcopy(module)
        return _apply_parallel_style(module_copy, mesh.group, parallelize_plan)

    if not isinstance(parallelize_plan, dict):
        raise TypeError(
            "parallelize_plan must be a ParallelStyle or a dict mapping module "
            f"FQNs to ParallelStyle objects, but got: {type(parallelize_plan).__name__}"
        )

    module_copy = copy.deepcopy(module)
    for module_fqn, style in parallelize_plan.items():
        if not isinstance(style, ParallelStyle):
            raise TypeError(
                f"Expected a ParallelStyle for '{module_fqn}', but got: {style!r}"
            )
        target_module = _get_submodule(module_copy, module_fqn)
        replacement = _apply_parallel_style(target_module, mesh.group, style)
        module_copy = _set_submodule(module_copy, module_fqn, replacement)
    return module_copy
