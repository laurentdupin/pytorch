from __future__ import annotations

import contextlib
import importlib.util
import importlib.metadata
import json
import os
import platform
import re
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
os.environ.setdefault(
    "PADDLE_PDX_CACHE_HOME",
    str(REPO_ROOT / "agent_space" / "paddlex_cache_fresh"),
)
HF_HOME = Path(os.environ["HF_HOME"])
HF_HUB_CACHE = Path(os.environ["HF_HUB_CACHE"])
PADDLE_PDX_CACHE_HOME = Path(os.environ["PADDLE_PDX_CACHE_HOME"])
TORCH_IMPORT_MODE = "source"


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
    "stack_subresource_lifetime_dry_run_counters",
    "stack_subresource_lifetime_dry_run_snapshot",
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
    "linear_aggregate_snapshot",
    "linear_pack_residency_snapshot",
    "vulkan_memory_residency_snapshot",
    "last_allocation_failure_snapshot",
    "packed_weight_residency_snapshot",
    "conv_plan_counters",
    "pointwise_conv_route_counters",
    "conv_aggregate_snapshot",
    "buffer_copy_counters",
    "buffer_copy_aggregate_snapshot",
    "clone_requirement_snapshot",
    "vision_owner_counters",
    "vision_owner_context_counters",
    "vision_owner_mlp_counters",
    "vision_stack_owner_counters",
    "zero_counters",
)

AGGREGATE_METRIC_RE = re.compile(r"\b(count|bytes|queue_submit|blocking_wait|poll_only)=(\d+)")


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def module_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def configure_hf_cache(cache_dir: Path | str | None) -> None:
    if cache_dir is None:
        return
    global HF_HOME, HF_HUB_CACHE
    HF_HOME = Path(cache_dir).resolve()
    HF_HUB_CACHE = HF_HOME / "hub"
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)


def configure_torch_import_mode(mode: str) -> None:
    if mode not in {"source", "installed"}:
        raise ValueError(f"Unsupported torch import mode: {mode}")
    global TORCH_IMPORT_MODE
    TORCH_IMPORT_MODE = mode


def import_torch() -> Any:
    repo_root = str(REPO_ROOT)
    if TORCH_IMPORT_MODE == "source":
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
    else:
        sys.path[:] = [
            path for path in sys.path if Path(path or ".").resolve() != REPO_ROOT
        ]
    if sys.platform == "win32":
        build_bin = REPO_ROOT / "build" / "bin" / "Release"
        if TORCH_IMPORT_MODE == "source" and build_bin.is_dir():
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


def _counter_delta(before: Any, after: Any) -> Any:
    if isinstance(before, bool) or isinstance(after, bool):
        return None
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after - before
    if (
        isinstance(before, list)
        and isinstance(after, list)
        and len(before) == len(after)
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in before)
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in after)
    ):
        return [after_item - before_item for before_item, after_item in zip(before, after)]
    if isinstance(before, dict) and isinstance(after, dict):
        result: dict[str, Any] = {}
        for key in sorted(before.keys() & after.keys()):
            value = _counter_delta(before[key], after[key])
            if value is not None:
                result[key] = value
        return result if result else None
    return None


def _aggregate_metrics_by_key(rows: Any) -> dict[str, dict[str, int]]:
    if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
        return {}
    metrics_by_key: dict[str, dict[str, int]] = {}
    for row in rows:
        metrics = {name: int(value) for name, value in AGGREGATE_METRIC_RE.findall(row)}
        if not metrics:
            continue
        key = AGGREGATE_METRIC_RE.sub("", row)
        key = " ".join(key.split())
        metrics_by_key[key] = metrics
    return metrics_by_key


def _aggregate_delta(before: Any, after: Any) -> list[str]:
    before_rows = _aggregate_metrics_by_key(before)
    after_rows = _aggregate_metrics_by_key(after)
    if not after_rows:
        return []
    out: list[str] = []
    for key in sorted(after_rows):
        metrics = after_rows[key]
        previous = before_rows.get(key, {})
        deltas = {
            name: value - int(previous.get(name, 0))
            for name, value in metrics.items()
        }
        if not any(value != 0 for value in deltas.values()):
            continue
        fields = " ".join(f"{name}={value}" for name, value in sorted(deltas.items()))
        out.append(f"{key} {fields}".strip())
    return out


def diff_vulkan_debug_counters(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for key in sorted(before.keys() & after.keys()):
        value = _counter_delta(before[key], after[key])
        if value is not None:
            diff[key] = value
            continue
        aggregate = _aggregate_delta(before[key], after[key])
        if aggregate:
            diff[f"{key}_delta"] = aggregate
    return diff


class VulkanCounterPhaseTracker:
    def __init__(self, torch_module: Any, backend: str) -> None:
        self.torch_module = torch_module
        self.backend = backend
        self._initial = snapshot_vulkan_debug_counters(torch_module, backend)
        self._previous = self._initial
        self._phases: list[dict[str, Any]] = []

    def mark(self, name: str) -> dict[str, Any]:
        current = snapshot_vulkan_debug_counters(self.torch_module, self.backend)
        phase = {
            "name": name,
            "start": self._previous,
            "end": current,
            "delta": diff_vulkan_debug_counters(self._previous, current),
        }
        self._phases.append(phase)
        self._previous = current
        return phase

    def summary(self) -> dict[str, Any]:
        current = snapshot_vulkan_debug_counters(self.torch_module, self.backend)
        return {
            "schema_version": 1,
            "backend": self.backend,
            "phases": self._phases,
            "total": {
                "name": "total_since_tracker_start",
                "start": self._initial,
                "end": current,
                "delta": diff_vulkan_debug_counters(self._initial, current),
            },
        }


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
    debug_traceback: bool = False,
    status: str | None = None,
) -> BenchmarkRecord:
    failure: dict[str, Any] = {"reason": reason}
    if exc is not None:
        failure["exception_type"] = type(exc).__name__
        failure["exception"] = concise_exception(exc)
        if debug_traceback:
            failure["traceback"] = traceback.format_exc(limit=12)
    resolved_status = status
    if resolved_status is None:
        resolved_status = "skip" if exc is None or is_environment_skip(exc) else "failure"
    record = BenchmarkRecord(
        task=task,
        model_name=model_name,
        model_id=model_id,
        backend=backend,
        device_index=device_index,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
        status=resolved_status,
        failure=failure,
    )
    record.environment = environment_summary()
    return record


def concise_exception(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    if not text:
        return type(exc).__name__
    return text[0][:500]


def is_environment_skip(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    skip_markers = (
        "localentrynotfound",
        "local entry not found",
        "couldn't connect to 'https://huggingface.co'",
        "cannot find the requested files in the disk cache",
        "outgoing traffic has been disabled",
        "gated repo",
        "requires you to be authenticated",
        "401 client error",
        "403 client error",
        "permissionerror",
        "access denied",
        "accès refusé",
        "no module named",
        "not found in your environment",
    )
    return any(marker in text for marker in skip_markers)


def environment_summary() -> dict[str, Any]:
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "repo_root": str(REPO_ROOT),
        "torch_import_mode": TORCH_IMPORT_MODE,
        "hf_home": str(HF_HOME),
        "hf_hub_cache": str(HF_HUB_CACHE),
        "paddle_pdx_cache_home": str(PADDLE_PDX_CACHE_HOME),
    }


def probe_accelerators() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dependency_versions": probe_dependency_versions(),
        "cache": {
            "hf_home": str(HF_HOME),
            "hf_hub_cache": str(HF_HUB_CACHE),
            "hf_home_exists": HF_HOME.exists(),
            "hf_hub_cache_exists": HF_HUB_CACHE.exists(),
            "paddle_pdx_cache_home": str(PADDLE_PDX_CACHE_HOME),
            "paddle_pdx_cache_exists": PADDLE_PDX_CACHE_HOME.exists(),
        },
    }
    try:
        torch = import_torch()
    except Exception as exc:
        payload["torch_import"] = {
            "available": False,
            "exception_type": type(exc).__name__,
            "exception": concise_exception(exc),
        }
        payload["vulkan"] = {"available": False, "skip_reason": "torch_import_failed"}
        payload["directml"] = {"available": False, "skip_reason": "torch_import_failed"}
        payload["cuda"] = {"available": False, "skip_reason": "torch_import_failed"}
        payload["vulkaninfo"] = probe_vulkaninfo()
        return payload

    payload.update(
        {
            "torch_version": getattr(torch, "__version__", None),
            "torch_import": {"available": True},
            "distributed": probe_torch_distributed(torch),
        }
    )
    payload.update(
        {
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
    )
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


def probe_torch_distributed(torch_module: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "has_distributed_c10d_extension": hasattr(
            getattr(torch_module, "_C", None),
            "_distributed_c10d",
        )
    }
    try:
        import torch.distributed as dist

        payload["python_import_available"] = True
        payload["is_available"] = bool(getattr(dist, "is_available", lambda: False)())
        payload["has_store"] = hasattr(dist, "Store")
        payload["has_backend"] = hasattr(dist, "Backend")
        payload["has_init_process_group"] = hasattr(dist, "init_process_group")
    except Exception as exc:
        payload["python_import_available"] = False
        payload["exception_type"] = type(exc).__name__
        payload["exception"] = concise_exception(exc)
    try:
        config = torch_module.__config__.show()
        payload["config"] = {
            "use_gloo": "USE_GLOO=ON" in config,
            "use_mpi": "USE_MPI=ON" in config,
            "use_nccl": "USE_NCCL=ON" in config,
        }
    except Exception as exc:
        payload["config_error"] = concise_exception(exc)
    return payload


def probe_dependency_versions() -> dict[str, Any]:
    names = {
        "transformers": "transformers",
        "diffusers": "diffusers",
        "huggingface_hub": "huggingface-hub",
        "paddleocr": "paddleocr",
        "torchvision": "torchvision",
        "torch_directml": "torch-directml",
    }
    versions: dict[str, Any] = {}
    for key, package_name in names.items():
        versions[key] = {
            "available": module_available(key),
            "version": module_version(package_name),
        }
    return versions


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
