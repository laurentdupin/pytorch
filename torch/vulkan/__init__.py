# mypy: allow-untyped-defs
r"""
This package adds support for Vulkan tensor types.

It exposes simple device selection and device-property queries for the Vulkan
backend.
"""

from __future__ import annotations

from typing import Any

import torch
import torch._C
from torch._utils import _dummy_type


_HAS_VULKAN_BINDINGS = all(
    hasattr(torch._C, attr)
    for attr in (
        "_VulkanDeviceProperties",
        "_vulkan_exchangeDevice",
        "_vulkan_getDeviceCount",
        "_vulkan_getDevice",
        "_vulkan_setDevice",
        "_vulkan_getDeviceProperties",
    )
)
_HAS_VULKAN = bool(getattr(torch._C, "_has_vulkan", False) or _HAS_VULKAN_BINDINGS)


if _HAS_VULKAN:
    _VulkanDeviceProperties = torch._C._VulkanDeviceProperties
    _exchange_device = torch._C._vulkan_exchangeDevice
else:
    _VulkanDeviceProperties = _dummy_type("_VulkanDeviceProperties")  # type: ignore[assignment, misc]

    def _exchange_device(device: int) -> int:
        raise NotImplementedError("PyTorch was compiled without Vulkan support")


def _is_compiled() -> bool:
    return _HAS_VULKAN


def _get_device_index(device: Any, optional: bool = False) -> int:
    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(device, torch.device):
        if device.type != "vulkan":
            raise ValueError(f"Expected a Vulkan device, but got: {device}")
        if device.index is not None:
            return device.index
        if optional:
            return current_device()
        raise ValueError(
            "Expected a Vulkan device with an explicit index, but got "
            f"{device!s}"
        )

    if isinstance(device, int):
        return device

    if device is None and optional:
        return current_device()

    raise ValueError(
        f"Expected a Vulkan device, integer index, or None, but got: {device}"
    )


def device_count() -> int:
    if not _is_compiled():
        return 0
    return torch._C._vulkan_getDeviceCount()


def is_available() -> bool:
    return device_count() > 0


def current_device() -> int:
    if not is_available():
        return -1
    return torch._C._vulkan_getDevice()


class device:
    r"""Context-manager that changes the selected Vulkan device."""

    def __init__(self, device: Any) -> None:
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        if self.idx >= 0:
            self.prev_idx = _exchange_device(self.idx)
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any):
        if self.prev_idx >= 0:
            torch._C._vulkan_setDevice(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of a Vulkan tensor."""

    def __init__(self, obj) -> None:
        idx = obj.get_device() if getattr(obj, "is_vulkan", False) else -1
        super().__init__(idx)


def set_device(device: Any) -> None:
    idx = _get_device_index(device)
    if idx >= 0:
        torch._C._vulkan_setDevice(idx)


def get_device_properties(device: Any = None) -> _VulkanDeviceProperties:
    if not is_available():
        raise RuntimeError("No Vulkan devices are available")
    device_index = _get_device_index(device, optional=True)
    return torch._C._vulkan_getDeviceProperties(device_index)


def get_device_name(device: Any = None) -> str:
    return get_device_properties(device).name


from ._composite import (  # noqa: E402
    ColwiseParallel,
    DeviceGroup,
    DeviceMesh,
    ParallelStyle,
    Replicate,
    RowwiseParallel,
    device_group,
    device_mesh,
    parallelize_module,
    replicate_module,
)
from ._graph import (  # noqa: E402
    VulkanGraphCensus,
    VulkanGraphExecutionError,
    VulkanGraphNodeRecord,
    VulkanGraphPlanningContext,
    VulkanGraphPlanReport,
    VulkanGraphProgram,
    VulkanGraphProgramKey,
    VulkanLinearLoweringReport,
    export_and_lower,
)


__all__ = [
    "ColwiseParallel",
    "DeviceGroup",
    "DeviceMesh",
    "ParallelStyle",
    "Replicate",
    "RowwiseParallel",
    "VulkanGraphCensus",
    "VulkanGraphExecutionError",
    "VulkanGraphNodeRecord",
    "VulkanGraphPlanningContext",
    "VulkanGraphPlanReport",
    "VulkanGraphProgram",
    "VulkanGraphProgramKey",
    "VulkanLinearLoweringReport",
    "current_device",
    "device",
    "device_count",
    "device_group",
    "device_mesh",
    "device_of",
    "export_and_lower",
    "get_device_name",
    "get_device_properties",
    "is_available",
    "parallelize_module",
    "replicate_module",
    "set_device",
]
