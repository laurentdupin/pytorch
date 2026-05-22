from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from bench_common import (
    REPO_ROOT,
    summarize_durations,
    suppress_windows_error_dialogs,
    synchronize_payload,
    windows_subprocess_kwargs,
    write_json,
)


suppress_windows_error_dialogs()


SCHEMA_VERSION = 1

os.environ.setdefault("HF_HOME", str(REPO_ROOT / "agent_space" / "hf_home"))
os.environ.setdefault(
    "HF_HUB_CACHE",
    str(REPO_ROOT / "agent_space" / "hf_home" / "hub"),
)


VULKAN_COUNTER_NAMES = (
    "cpu_fallback_count",
    "sync_readback_count",
    "fallback_phase_counters",
    "timed_fallback_phase_counters",
    "sync_counters",
    "submit_origin_counters",
    "submit_origin_phase_counters",
    "retire_drain_counters",
    "retire_call_site_counters",
    "retired_resource_aggregate_snapshot",
    "stack_temp_lifetime_safety_snapshot",
    "stack_internal_temp_retire_batch_counters",
    "stack_internal_temp_retire_batch_snapshot",
    "stack_retire_drain_blocker_counters",
    "stack_retire_drain_blocker_snapshot",
    "stack_scratch_arena_lifetime_snapshot",
    "stack_allocation_aggregate_snapshot",
    "stack_dispatch_aggregate_snapshot",
    "stack_attention_counters",
    "stack_execution_manifest",
    "stack_capture_readiness",
    "stack_shape_plan_keys",
    "stack_shape_plan_readiness",
    "stack_shape_plan_counters",
    "stack_resource_binding_manifest",
    "stack_descriptor_binding_table",
    "stack_descriptor_binding_validation",
    "stack_planned_recording_readiness",
    "stack_planned_recording_counters",
    "stack_replay_readiness",
    "stack_replay_binding_mode",
    "stack_replay_counters",
    "attention_plan_counters",
    "linear_plan_counters",
)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def import_torch() -> Any:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if sys.platform == "win32":
        build_bin = REPO_ROOT / "build" / "bin" / "Release"
        if build_bin.is_dir():
            current = os.environ.get("PATH", "")
            parts = current.split(os.pathsep) if current else []
            if str(build_bin) not in parts:
                os.environ["PATH"] = (
                    os.pathsep.join([str(build_bin), current])
                    if current
                    else str(build_bin)
                )
    import torch

    return torch


def snapshot_vulkan_debug_counters(torch_module: Any, backend: str) -> dict[str, Any]:
    if backend != "vulkan" or not hasattr(torch_module.ops, "vulkan_prepack"):
        return {}
    ops = torch_module.ops.vulkan_prepack
    counters: dict[str, Any] = {}
    for name in VULKAN_COUNTER_NAMES:
        fn = getattr(ops, name, None)
        if fn is None:
            continue
        try:
            counters[name] = fn()
        except Exception as exc:
            counters[f"{name}_error"] = repr(exc)
    return counters


def reset_vulkan_debug_counters(torch_module: Any, backend: str) -> None:
    if backend != "vulkan" or not hasattr(torch_module.ops, "vulkan_prepack"):
        return
    reset = getattr(torch_module.ops.vulkan_prepack, "reset_fallback_counters", None)
    if reset is not None:
        reset()


def torch_device_for_backend(
    torch_module: Any,
    backend: str,
    device_index: int | None,
) -> tuple[Any, dict[str, Any]]:
    if backend == "cpu":
        return torch_module.device("cpu"), {"type": "cpu", "index": None, "name": "CPU"}
    if backend == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        index = 0 if device_index is None else int(device_index)
        return (
            torch_module.device(f"cuda:{index}"),
            {
                "type": "cuda",
                "index": index,
                "name": torch_module.cuda.get_device_name(index),
                "capability": list(torch_module.cuda.get_device_capability(index)),
            },
        )
    if backend == "directml":
        import torch_directml

        index = (
            int(torch_directml.default_device())
            if device_index is None
            else int(device_index)
        )
        return (
            torch_directml.device(index),
            {
                "type": "directml",
                "index": index,
                "name": torch_directml.device_name(index).rstrip("\0"),
            },
        )
    if backend == "vulkan":
        info = {"type": "vulkan", "index": 0 if device_index is None else device_index}
        vulkan = getattr(torch_module, "vulkan", None)
        if vulkan is not None:
            try:
                info["count"] = int(vulkan.device_count())
                info["name"] = vulkan.get_device_name(int(info["index"]))
            except Exception as exc:
                info["probe_error"] = repr(exc)
        return "vulkan", info
    raise ValueError(f"Unsupported backend: {backend}")


@contextlib.contextmanager
def timed_region(torch_module: Any, backend: str) -> Any:
    ops = getattr(getattr(torch_module, "ops", None), "vulkan_prepack", None)
    setter = getattr(ops, "set_benchmark_timed_region", None) if ops else None
    if backend == "vulkan" and setter is not None:
        setter(True)
    try:
        yield
    finally:
        if backend == "vulkan" and setter is not None:
            setter(False)


def measure_repeated(
    name: str,
    repeats: int,
    fn: Callable[[], Any],
    *,
    torch_module: Any,
    backend: str,
    device: Any,
) -> tuple[dict[str, Any], Any]:
    durations: list[float] = []
    last_output: Any = None
    for _ in range(repeats):
        start = time.perf_counter()
        with timed_region(torch_module, backend):
            last_output = fn()
            synchronize_payload(torch_module, backend, last_output, device)
        durations.append(time.perf_counter() - start)
    return summarize_durations(name, durations), last_output


@dataclass
class BenchmarkRecord:
    task: str
    model_name: str
    model_id: str
    backend: str
    device_index: int | None
    dtype: str
    warmup: int
    repeats: int
    status: str = "ok"
    failure: dict[str, Any] | None = None
    device: dict[str, Any] = field(default_factory=dict)
    input: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, Any] = field(default_factory=dict)
    output_sanity: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "task": self.task,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "backend": self.backend,
            "device_index": self.device_index,
            "device": self.device,
            "dtype": self.dtype,
            "input": self.input,
            "warmup": self.warmup,
            "repeats": self.repeats,
            "status": self.status,
            "failure": self.failure,
            "timings": self.timings,
            "counters": self.counters,
            "output_sanity": self.output_sanity,
            "environment": self.environment,
        }


def make_failure(
    *,
    task: str,
    model_name: str,
    model_id: str,
    backend: str,
    device_index: int | None,
    dtype: str,
    warmup: int,
    repeats: int,
    reason: str,
    exc: BaseException | None = None,
) -> BenchmarkRecord:
    failure: dict[str, Any] = {"reason": reason}
    if exc is not None:
        failure["exception_type"] = type(exc).__name__
        failure["exception"] = repr(exc)
        failure["traceback"] = traceback.format_exc(limit=12)
    record = BenchmarkRecord(
        task=task,
        model_name=model_name,
        model_id=model_id,
        backend=backend,
        device_index=device_index,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
        status="skip" if exc is None else "failure",
        failure=failure,
    )
    record.environment = environment_summary()
    return record


def environment_summary() -> dict[str, Any]:
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "repo_root": str(REPO_ROOT),
    }


def probe_accelerators() -> dict[str, Any]:
    torch = import_torch()
    payload: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", None),
        "vulkan": {
            "available": bool(getattr(torch, "is_vulkan_available", lambda: False)()),
        },
        "directml": {"available": False},
        "cuda": {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "devices": [],
        },
    }
    if hasattr(torch, "vulkan"):
        try:
            count = int(torch.vulkan.device_count())
            payload["vulkan"]["device_count"] = count
            payload["vulkan"]["devices"] = [
                {"index": i, "name": torch.vulkan.get_device_name(i)}
                for i in range(count)
            ]
        except Exception as exc:
            payload["vulkan"]["error"] = repr(exc)
    if torch.cuda.is_available():
        payload["cuda"]["runtime_version"] = getattr(torch.version, "cuda", None)
        for i in range(torch.cuda.device_count()):
            payload["cuda"]["devices"].append(
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": list(torch.cuda.get_device_capability(i)),
                }
            )
    if module_available("torch_directml"):
        try:
            import torch_directml

            count = int(torch_directml.device_count())
            payload["directml"] = {
                "available": bool(torch_directml.is_available()),
                "device_count": count,
                "default_device": int(torch_directml.default_device()),
                "devices": [
                    {
                        "index": i,
                        "name": torch_directml.device_name(i).rstrip("\0"),
                    }
                    for i in range(count)
                ],
            }
        except Exception as exc:
            payload["directml"]["error"] = repr(exc)
    payload["vulkaninfo"] = probe_vulkaninfo()
    return payload


def probe_vulkaninfo() -> dict[str, Any]:
    candidates: list[Path] = []
    if os.environ.get("VULKAN_SDK"):
        candidates.append(Path(os.environ["VULKAN_SDK"]) / "Bin" / "vulkaninfoSDK.exe")
    candidates.extend(
        [
            REPO_ROOT.parent / "VulkanSDK" / "1.4.341.1" / "Bin" / "vulkaninfoSDK.exe",
            Path("vulkaninfo.exe"),
            Path("vulkaninfoSDK.exe"),
        ]
    )
    exe = next((path for path in candidates if path.exists()), None)
    if exe is None:
        return {"available": False, "checked": [str(path) for path in candidates]}
    try:
        proc = subprocess.run(
            [str(exe), "--summary"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            errors="replace",
            **windows_subprocess_kwargs(),
        )
    except Exception as exc:
        return {"available": False, "path": str(exe), "error": repr(exc)}
    devices: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if line.startswith("GPU") and line.endswith(":"):
            if current is not None:
                devices.append(current)
            current = {"label": line[:-1]}
            continue
        if current is not None and "=" in line:
            key, value = [part.strip() for part in line.split("=", 1)]
            if key in {"deviceName", "deviceType", "apiVersion", "driverVersion"}:
                current[key] = value
    if current is not None:
        devices.append(current)
    return {
        "available": True,
        "path": str(exe),
        "returncode": proc.returncode,
        "devices": devices,
    }


def write_records(path: Path, records: list[BenchmarkRecord], probe: dict[str, Any]) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "benchmark_model_suite",
        "environment": environment_summary(),
        "accelerator_probe": probe,
        "records": [record.to_json() for record in records],
    }
    write_json(path, payload)
