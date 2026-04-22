from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
LOCAL_BUILD_BIN_DIR = REPO_ROOT / "build" / "bin" / "Release"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32" and LOCAL_BUILD_BIN_DIR.is_dir():
    existing_path = os.environ.get("PATH", "")
    path_entries = existing_path.split(os.pathsep) if existing_path else []
    if str(LOCAL_BUILD_BIN_DIR) not in path_entries:
        os.environ["PATH"] = (
            os.pathsep.join([str(LOCAL_BUILD_BIN_DIR), existing_path])
            if existing_path
            else str(LOCAL_BUILD_BIN_DIR)
        )


def add_python_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * pct
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize_durations(name: str, durations: list[float]) -> dict[str, Any]:
    ordered = sorted(durations)
    total = sum(durations)
    count = len(durations)
    mean = total / count if count else 0.0
    median = statistics.median(ordered) if ordered else 0.0
    stdev = statistics.pstdev(ordered) if count > 1 else 0.0
    return {
        "name": name,
        "count": count,
        "total_s": total,
        "mean_s": mean,
        "median_s": median,
        "min_s": ordered[0] if ordered else 0.0,
        "max_s": ordered[-1] if ordered else 0.0,
        "stdev_s": stdev,
        "p90_s": percentile(ordered, 0.90),
        "p95_s": percentile(ordered, 0.95),
        "throughput_items_per_s": (count / total) if total > 0 else 0.0,
        "durations_s": durations,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def synchronize_device(
    torch_module: Any,
    device_kind: str,
    device: Any | None = None,
) -> None:
    if device_kind == "cuda":
        if device is None:
            torch_module.cuda.synchronize()
        else:
            torch_module.cuda.synchronize(device)
        return

    if device_kind == "vulkan":
        synchronize = getattr(
            getattr(torch_module.ops, "vulkan_prepack", None),
            "synchronize",
            None,
        )
        if synchronize is not None:
            synchronize()


def synchronize_result(
    torch_module: Any,
    device_kind: str,
    result: Any,
    device: Any | None = None,
) -> float | None:
    if device_kind == "directml":
        # DirectML does not expose an explicit synchronize API here.
        # Read back one scalar so the pending work completes without
        # paying the cost of copying the full output tensor.
        return float(result.reshape(-1)[0].cpu().item())

    synchronize_device(torch_module, device_kind, device)
    return None


def first_tensor_in_payload(torch_module: Any, payload: Any) -> Any | None:
    is_tensor = getattr(torch_module, "is_tensor", None)
    if callable(is_tensor) and is_tensor(payload):
        return payload

    if isinstance(payload, dict):
        for value in payload.values():
            tensor = first_tensor_in_payload(torch_module, value)
            if tensor is not None:
                return tensor
        return None

    if isinstance(payload, (list, tuple)):
        for value in payload:
            tensor = first_tensor_in_payload(torch_module, value)
            if tensor is not None:
                return tensor
        return None

    return None


def synchronize_payload(
    torch_module: Any,
    device_kind: str,
    payload: Any,
    device: Any | None = None,
) -> float | None:
    tensor = first_tensor_in_payload(torch_module, payload)
    if tensor is not None:
        return synchronize_result(torch_module, device_kind, tensor, device)

    synchronize_device(torch_module, device_kind, device)
    return None
