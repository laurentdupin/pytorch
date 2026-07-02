from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from bench_common import (
    REPO_ROOT,
    WORKSPACE_ROOT,
    add_python_path,
    suppress_windows_error_dialogs,
    windows_subprocess_kwargs,
)

suppress_windows_error_dialogs()


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def _workspace_candidate_paths(*relative_parts: str) -> list[Path]:
    return [
        REPO_ROOT.joinpath(*relative_parts),
        WORKSPACE_ROOT.joinpath(*relative_parts),
    ]


def resolve_depth_anything_repo(repo: str | None) -> Path:
    if repo:
        repo_path = Path(repo).resolve()
        add_python_path(repo_path)
        return repo_path

    candidates = _workspace_candidate_paths("Depth-Anything-V2")
    repo_path = next((path.resolve() for path in candidates if path.exists()), None)
    if repo_path is None:
        candidate_text = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Depth-Anything-V2 repo not found. Checked: "
            + candidate_text
        )

    add_python_path(repo_path)
    return repo_path


def resolve_depth_anything_checkpoint(
    repo_path: Path,
    encoder: str,
    checkpoint: str | None,
) -> Path:
    if checkpoint:
        return Path(checkpoint).resolve()
    return repo_path / "checkpoints" / f"depth_anything_v2_{encoder}.pth"


def resolve_default_vulkan_device_info() -> dict[str, Any]:
    sdk_candidates = []
    if os.environ.get("VULKAN_SDK"):
        sdk_candidates.append(Path(os.environ["VULKAN_SDK"]) / "Bin" / "vulkaninfoSDK.exe")
    sdk_candidates.extend(
        _workspace_candidate_paths("VulkanSDK", "1.4.341.1", "Bin", "vulkaninfoSDK.exe")
    )

    sdk_path = next((path for path in sdk_candidates if path.exists()), None)
    if sdk_path is None:
        return {
            "vulkan_device_index": 0,
            "vulkan_device_name": None,
            "vulkan_info_source": None,
        }

    try:
        proc = subprocess.run(
            [str(sdk_path), "--summary"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            errors="replace",
            **windows_subprocess_kwargs(),
        )
    except Exception:
        return {
            "vulkan_device_index": 0,
            "vulkan_device_name": None,
            "vulkan_info_source": str(sdk_path),
        }

    gpu_index: int | None = None
    device_name: str | None = None
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("GPU") and line.endswith(":"):
            try:
                gpu_index = int(line[3:-1])
            except ValueError:
                gpu_index = None
            continue
        if gpu_index == 0 and line.startswith("deviceName"):
            _, value = line.split("=", 1)
            device_name = value.strip()
            break

    return {
        "vulkan_device_index": 0,
        "vulkan_device_name": device_name,
        "vulkan_info_source": str(sdk_path),
    }


def resolve_runtime_device(
    torch_module: Any,
    requested: str,
    vulkan_device_index: int | None = None,
    directml_device_index: int | None = None,
    cuda_device_index: int | None = None,
) -> tuple[Any, str, dict[str, Any]]:
    if requested == "directml":
        import torch_directml

        device_count = int(torch_directml.device_count())
        selected_index = directml_device_index
        if selected_index is None:
            configured_index = os.environ.get("PYTORCHVULKAN_DIRECTML_DEVICE_INDEX")
            if configured_index:
                selected_index = int(configured_index)
        if selected_index is None:
            selected_index = int(torch_directml.default_device())
        if selected_index < 0 or selected_index >= device_count:
            raise ValueError(
                f"DirectML device index {selected_index} is out of range for "
                f"{device_count} adapter(s)."
            )
        device_name = torch_directml.device_name(selected_index).rstrip("\0")
        return (
            torch_directml.device(selected_index),
            "directml",
            {
                "requested": requested,
                "resolved": "directml",
                "directml_available": bool(torch_directml.is_available()),
                "directml_device_count": device_count,
                "directml_device_index": selected_index,
                "directml_device_name": device_name,
            },
        )

    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this environment.")

        device_count = int(torch_module.cuda.device_count())
        selected_index = cuda_device_index
        if selected_index is None:
            configured_index = os.environ.get("PYTORCHVULKAN_CUDA_DEVICE_INDEX")
            if configured_index:
                selected_index = int(configured_index)
        if selected_index is None:
            selected_index = 0
        if selected_index < 0 or selected_index >= device_count:
            raise ValueError(
                f"CUDA device index {selected_index} is out of range for "
                f"{device_count} device(s)."
            )
        device = torch_module.device(f"cuda:{selected_index}")
        return (
            device,
            "cuda",
            {
                "requested": requested,
                "resolved": str(device),
                "cuda_available": True,
                "cuda_device_count": device_count,
                "cuda_device_index": selected_index,
                "cuda_device_name": torch_module.cuda.get_device_name(selected_index),
                "directml_available": False,
                "directml_device_count": 0,
            },
        )

    if requested == "vulkan":
        selected_index = vulkan_device_index
        if selected_index is None:
            configured_index = os.environ.get("PYTORCHVULKAN_VULKAN_DEVICE_INDEX")
            if configured_index:
                selected_index = int(configured_index)
        if selected_index is None:
            selected_index = int(torch_module.vulkan.current_device())
        device_count = int(torch_module.vulkan.device_count())
        if selected_index < 0 or selected_index >= device_count:
            raise ValueError(
                f"Vulkan device index {selected_index} is out of range for "
                f"{device_count} adapter(s)."
            )
        torch_module.vulkan.set_device(selected_index)
        device_name = torch_module.vulkan.get_device_name(selected_index)
        return (
            requested,
            requested,
            {
                "requested": requested,
                "resolved": requested,
                "cuda_available": bool(
                    hasattr(torch_module, "cuda") and torch_module.cuda.is_available()
                ),
                "cuda_device_count": int(
                    torch_module.cuda.device_count()
                    if hasattr(torch_module, "cuda") and torch_module.cuda.is_available()
                    else 0
                ),
                "directml_available": False,
                "directml_device_count": 0,
                "vulkan_device_count": device_count,
                "vulkan_device_index": selected_index,
                "vulkan_device_name": device_name,
                "vulkan_info_source": None,
            },
        )

    return (
        requested,
        requested,
        {
            "requested": requested,
            "resolved": requested,
            "cuda_available": bool(
                hasattr(torch_module, "cuda") and torch_module.cuda.is_available()
            ),
            "cuda_device_count": int(
                torch_module.cuda.device_count()
                if hasattr(torch_module, "cuda") and torch_module.cuda.is_available()
                else 0
            ),
            "directml_available": False,
            "directml_device_count": 0,
        },
    )


def inference_context(torch_module: Any, device_kind: str) -> Any:
    if device_kind == "directml":
        return torch_module.no_grad()
    return torch_module.inference_mode()
