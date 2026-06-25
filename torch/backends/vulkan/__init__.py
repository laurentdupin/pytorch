# mypy: allow-untyped-defs

import torch


__all__ = [
    "is_available",
]


def is_available() -> bool:
    r"""Return whether Vulkan is currently available."""
    return torch.vulkan.is_available()
