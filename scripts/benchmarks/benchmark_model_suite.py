from __future__ import annotations

import argparse
import collections
import contextlib
import importlib.metadata
import importlib.machinery
import json
import os
import re
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any

from benchmark_suite_common import (
    BenchmarkRecord,
    PADDLE_PDX_CACHE_HOME,
    configure_hf_cache,
    configure_torch_import_mode,
    environment_summary,
    import_torch,
    is_environment_skip,
    make_failure,
    measure_repeated,
    module_available,
    probe_accelerators,
    reset_vulkan_debug_counters,
    snapshot_vulkan_debug_counters,
    torch_device_for_backend,
    write_records,
)
from bench_common import summarize_durations


DEFAULT_MODELS = {
    "torch_ops": "local_torch_conv_relu_smoke",
    "lotus": "jingheya/lotus-depth-d-v1-1",
    "hy_mt": "tencent/HY-MT1.5-1.8B",
    "paddleocr": "PaddleOCR 3.5 Transformers backend",
    "gemma": "google/gemma-4-E2B-it",
}

HY_MT_DECODE_REUSE_PROMPT = "Translate to French: Hello world."


def is_vulkan_tensor(torch: Any, value: Any) -> bool:
    return torch.is_tensor(value) and str(value.device).startswith("vulkan")


def flatten_tensors(torch: Any, payload: Any) -> list[Any]:
    if torch.is_tensor(payload):
        return [payload]
    if isinstance(payload, (list, tuple)):
        tensors: list[Any] = []
        for item in payload:
            tensors.extend(flatten_tensors(torch, item))
        return tensors
    if isinstance(payload, dict):
        tensors = []
        for item in payload.values():
            tensors.extend(flatten_tensors(torch, item))
        return tensors
    return []


def benchmark_distributed_import_status(torch: Any) -> str:
    if hasattr(getattr(torch, "_C", None), "_distributed_c10d"):
        return "real_distributed_c10d"
    if "torch._C._distributed_c10d" in sys.modules:
        module = sys.modules["torch._C._distributed_c10d"]
        if getattr(module, "_benchmark_distributed_import_shim", False):
            return "distributed_import_shim"
    if "transformers.generation.continuous_batching" in sys.modules:
        module = sys.modules["transformers.generation.continuous_batching"]
        if getattr(module, "_benchmark_distributed_import_shim", False):
            return "distributed_import_shim"
    return "missing_distributed_c10d"


def install_benchmark_distributed_import_shim(torch: Any) -> dict[str, Any]:
    if hasattr(getattr(torch, "_C", None), "_distributed_c10d"):
        return {"status": "real_distributed_c10d", "installed": False}
    c10d_module_name = "torch._C._distributed_c10d"
    c10d_existing = sys.modules.get(c10d_module_name)
    c10d_installed = False
    if c10d_existing is None:
        c10d_module = types.ModuleType(c10d_module_name)

        class _UnavailableDistributedObject:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    "Benchmark distributed import shim does not implement "
                    "torch.distributed collectives."
                )

        class _UnavailableBackend:
            FAKE = "fake"
            GLOO = "gloo"
            MPI = "mpi"
            NCCL = "nccl"
            UCC = "ucc"
            UNDEFINED = "undefined"

            @classmethod
            def register_backend(cls, *args: Any, **kwargs: Any) -> None:
                return None

        def _unavailable_distributed_function(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(
                "Benchmark distributed import shim does not implement "
                "torch.distributed collectives."
            )

        def _c10d_getattr(name: str) -> Any:
            if name.startswith("__"):
                raise AttributeError(name)
            if name.startswith("_") and name not in {
                "_DistributedBackendOptions",
                "_DEFAULT_PG_TIMEOUT",
                "_DEFAULT_PG_NCCL_TIMEOUT",
            }:
                return _unavailable_distributed_function
            return _UnavailableDistributedObject

        c10d_module.__getattr__ = _c10d_getattr
        c10d_module.__file__ = "<benchmark_distributed_import_shim>"
        c10d_module.__spec__ = importlib.machinery.ModuleSpec(
            c10d_module_name,
            loader=None,
        )
        c10d_module._DEFAULT_PG_TIMEOUT = 1800
        c10d_module._DEFAULT_PG_NCCL_TIMEOUT = 1800
        c10d_module.Backend = _UnavailableBackend
        c10d_module._DistributedBackendOptions = _UnavailableDistributedObject
        c10d_module._benchmark_distributed_import_shim = True
        sys.modules[c10d_module_name] = c10d_module
        c10d_installed = True
        try:
            import torch.distributed as torch_distributed

            for attr_name in (
                "FileStore",
                "HashStore",
                "PrefixStore",
                "ProcessGroup",
                "Store",
                "TCPStore",
                "Work",
            ):
                if not hasattr(torch_distributed, attr_name):
                    setattr(
                        torch_distributed,
                        attr_name,
                        _UnavailableDistributedObject,
                    )
            if not hasattr(torch_distributed, "Backend"):
                torch_distributed.Backend = _UnavailableBackend
        except Exception:
            pass
    module_name = "transformers.generation.continuous_batching"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return {
            "status": (
                "distributed_import_shim"
                if getattr(existing, "_benchmark_distributed_import_shim", False)
                else "missing_distributed_c10d"
            ),
            "installed": c10d_installed,
        }

    module = types.ModuleType(module_name)

    class ContinuousMixin:
        def generate_batch(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(
                "Benchmark distributed import shim does not implement "
                "Transformers continuous batching."
            )

    class _UnavailableContinuousBatching:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "Benchmark distributed import shim does not implement "
                "Transformers continuous batching."
            )

    class RequestStatus:
        pass

    class RequestState:
        pass

    module.__path__ = []
    module.ContinuousMixin = ContinuousMixin
    module.ContinuousBatchingManager = _UnavailableContinuousBatching
    module.FIFOScheduler = _UnavailableContinuousBatching
    module.PagedAttentionCache = _UnavailableContinuousBatching
    module.PrefillFirstScheduler = _UnavailableContinuousBatching
    module.RequestState = RequestState
    module.RequestStatus = RequestStatus
    module.Scheduler = _UnavailableContinuousBatching
    module.__all__ = [
        "ContinuousBatchingManager",
        "ContinuousMixin",
        "FIFOScheduler",
        "PagedAttentionCache",
        "PrefillFirstScheduler",
        "RequestState",
        "RequestStatus",
        "Scheduler",
    ]
    module._benchmark_distributed_import_shim = True
    sys.modules[module_name] = module
    submodules = {
        "cache": {"PagedAttentionCache": _UnavailableContinuousBatching},
        "continuous_api": {
            "ContinuousBatchingManager": _UnavailableContinuousBatching,
            "ContinuousMixin": ContinuousMixin,
        },
        "requests": {
            "RequestState": RequestState,
            "RequestStatus": RequestStatus,
        },
        "scheduler": {
            "FIFOScheduler": _UnavailableContinuousBatching,
            "PrefillFirstScheduler": _UnavailableContinuousBatching,
            "Scheduler": _UnavailableContinuousBatching,
        },
    }
    for suffix, attrs in submodules.items():
        submodule = types.ModuleType(f"{module_name}.{suffix}")
        submodule._benchmark_distributed_import_shim = True
        for attr_name, value in attrs.items():
            setattr(submodule, attr_name, value)
        sys.modules[submodule.__name__] = submodule
    return {
        "status": "distributed_import_shim",
        "installed": True,
        "scope": "torch_c_distributed_c10d_and_transformers_continuous_batching_import_only",
    }


def prepare_benchmark_distributed_import(torch: Any) -> dict[str, Any]:
    if hasattr(getattr(torch, "_C", None), "_distributed_c10d"):
        return {"status": "real_distributed_c10d", "installed": False}
    return install_benchmark_distributed_import_shim(torch)


def install_grid_sample_call_recorder(
    torch: Any,
    backend: str,
) -> tuple[list[dict[str, Any]], Any]:
    import torch.nn.functional as torch_functional

    calls: list[dict[str, Any]] = []
    original_grid_sample = torch_functional.grid_sample
    diag_mode = os.environ.get("PYTORCH_BENCHMARK_PADDLEOCR_GRID_SAMPLE_DIAG", "")
    capture_snapshots = diag_mode in {"trace", "synchronize_before"}
    synchronize_before = diag_mode == "synchronize_before"

    def tensor_desc(tensor: Any, include_storage: bool = False) -> dict[str, Any]:
        desc = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "stride": list(tensor.stride()),
            "is_contiguous": bool(tensor.is_contiguous()),
        }
        if not include_storage:
            return desc
        desc.update(
            {
                "numel": int(tensor.numel()),
                "element_size": int(tensor.element_size()),
                "bytes": int(tensor.numel() * tensor.element_size()),
                "storage_offset": int(tensor.storage_offset()),
            }
        )
        try:
            desc["layout"] = str(tensor.layout)
        except Exception as exc:
            desc["layout_error"] = repr(exc)
        try:
            storage = tensor.untyped_storage()
            desc["storage_nbytes"] = int(storage.nbytes())
            desc["storage_data_ptr"] = int(storage.data_ptr())
        except Exception as exc:
            desc["storage_error"] = repr(exc)
        return desc

    def grid_sampler_dispatch_desc(input: Any, grid: Any) -> dict[str, Any]:
        output_numel = int(input.shape[0] * input.shape[1] * grid.shape[1] * grid.shape[2])
        return {
            "output_shape": [
                int(input.shape[0]),
                int(input.shape[1]),
                int(grid.shape[1]),
                int(grid.shape[2]),
            ],
            "output_numel": output_numel,
            "output_bytes": output_numel * int(input.element_size()),
            "dispatch_global": [output_numel, 1, 1],
            "dispatch_local": [16, 4, 1],
            "dispatch_groups": [(output_numel + 15) // 16, 1, 1],
        }

    def wrapped_grid_sample(
        input: Any,
        grid: Any,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool | None = None,
    ) -> Any:
        call: dict[str, Any] = {
            "index": len(calls),
            "input": tensor_desc(input, include_storage=capture_snapshots),
            "grid": tensor_desc(grid, include_storage=capture_snapshots),
            "mode": mode,
            "padding_mode": padding_mode,
            "align_corners": align_corners,
            "dispatch": grid_sampler_dispatch_desc(input, grid),
            "diagnostic_mode": diag_mode or "off",
        }
        if capture_snapshots:
            call["pre_snapshot"] = snapshot_vulkan_debug_counters(torch, backend)
        if synchronize_before:
            try:
                torch.ops.vulkan_prepack.synchronize()
                call["pre_call_synchronize"] = "ok"
            except Exception as exc:
                call["pre_call_synchronize"] = f"{type(exc).__name__}: {exc}"
            call["after_pre_call_synchronize_snapshot"] = snapshot_vulkan_debug_counters(
                torch, backend
            )
        calls.append(call)
        try:
            output = original_grid_sample(
                input,
                grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )
            call["output"] = tensor_desc(output, include_storage=capture_snapshots)
            if capture_snapshots:
                call["post_snapshot"] = snapshot_vulkan_debug_counters(torch, backend)
            return output
        except Exception as exc:
            call["exception_type"] = type(exc).__name__
            call["exception"] = str(exc)
            if capture_snapshots:
                call["post_exception_snapshot"] = snapshot_vulkan_debug_counters(
                    torch, backend
                )
            raise

    torch_functional.grid_sample = wrapped_grid_sample
    return calls, (torch_functional, original_grid_sample)


def restore_grid_sample_call_recorder(patch: Any) -> None:
    if patch is None:
        return
    torch_functional, original_grid_sample = patch
    torch_functional.grid_sample = original_grid_sample


def paddleocr_cache_has_models() -> bool:
    official_models = PADDLE_PDX_CACHE_HOME / "official_models"
    if not official_models.is_dir():
        return False
    required_model_dirs = (
        "PP-LCNet_x1_0_doc_ori_safetensors",
        "UVDoc_safetensors",
        "PP-LCNet_x1_0_textline_ori_safetensors",
        "PP-OCRv5_server_det_safetensors",
        "PP-OCRv5_server_rec_safetensors",
    )
    return all(
        (official_models / model_dir).is_dir()
        and any((official_models / model_dir).rglob("*"))
        for model_dir in required_model_dirs
    )


def make_test_image(size: int) -> Any:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (size, size), color=(235, 238, 242))
    draw = ImageDraw.Draw(image)
    draw.rectangle((size // 8, size // 8, size // 2, size // 2), fill=(90, 140, 210))
    draw.ellipse((size // 2, size // 3, size - 16, size - 16), fill=(220, 120, 70))
    draw.line((0, size - 1, size - 1, 0), fill=(40, 40, 40), width=3)
    return image


def make_document_image(path: Path, size: int) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (size, size), color="white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, size // 18))
    except Exception:
        font = ImageFont.load_default()
    lines = [
        "Vulkan benchmark document",
        "Invoice: RX9070-001",
        "Total: 42.50 EUR",
        "Backend smoke OCR",
    ]
    y = size // 8
    for line in lines:
        draw.text((size // 10, y), line, fill="black", font=font)
        y += size // 8
    image.save(path)
    return path


def image_file_metadata(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        return {
            "path": str(path),
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode,
        }


def load_rgb_image(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def tensor_sanity(torch: Any, payload: Any) -> dict[str, Any]:
    tensor = payload
    if isinstance(payload, dict):
        tensor = next((v for v in payload.values() if torch.is_tensor(v)), None)
    if isinstance(payload, (list, tuple)):
        tensor = next((v for v in payload if torch.is_tensor(v)), None)
    if not torch.is_tensor(tensor):
        return {"tensor_present": False, "repr": str(type(payload))}
    cpu = tensor.detach().float().cpu()
    return {
        "tensor_present": True,
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(cpu).all().item()),
        "min": float(cpu.min().item()),
        "max": float(cpu.max().item()),
        "mean": float(cpu.mean().item()),
    }


def classify_model_move_failure(
    torch: Any,
    model: Any,
    exc: BaseException,
    backend: str,
) -> dict[str, Any]:
    text = "".join(traceback.format_exception(exc))
    if backend != "vulkan" or (
        "VK_ERROR_OUT_OF_DEVICE_MEMORY" not in text
        and "Failed to move tensor" not in text
    ):
        return {}
    info: dict[str, Any] = {
        "kind": "model_weight_vulkan_oom",
        "backend": backend,
        "exception": text[:500],
    }
    match = re.search(r"Failed to move tensor '([^']+)'", text)
    if match:
        tensor_name = match.group(1)
        info["tensor_name"] = tensor_name
        try:
            for name, parameter in model.named_parameters():
                if tensor_name == name or tensor_name.endswith(f".{name}"):
                    info.update(
                        {
                            "parameter_name": name,
                            "dtype": str(parameter.dtype),
                            "shape": [int(dim) for dim in parameter.shape],
                            "numel": int(parameter.numel()),
                            "bytes": int(parameter.numel() * parameter.element_size()),
                        }
                    )
                    break
        except Exception as metadata_exc:
            info["parameter_metadata_error"] = (
                f"{type(metadata_exc).__name__}: {str(metadata_exc)[:200]}"
            )
    vulkan_module = getattr(torch, "vulkan", None)
    if vulkan_module is not None:
        try:
            if hasattr(vulkan_module, "memory_stats"):
                info["device_memory_stats"] = vulkan_module.memory_stats()
            elif hasattr(vulkan_module, "memory_allocated"):
                info["device_memory_allocated"] = int(vulkan_module.memory_allocated())
        except Exception as memory_exc:
            info["device_memory_info_error"] = (
                f"{type(memory_exc).__name__}: {str(memory_exc)[:200]}"
            )
    return info


def tensor_metadata(torch: Any, tensor: Any) -> dict[str, Any]:
    if not torch.is_tensor(tensor):
        return {"tensor_present": False, "type": type(tensor).__name__}
    meta: dict[str, Any] = {
        "tensor_present": True,
        "shape": [int(dim) for dim in tensor.shape],
        "stride": [int(dim) for dim in tensor.stride()],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
        "bytes": int(tensor.numel() * tensor.element_size()),
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
    }
    try:
        meta["storage_offset"] = int(tensor.storage_offset())
    except Exception as exc:
        meta["storage_offset_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    try:
        meta["is_contiguous"] = bool(tensor.is_contiguous())
    except Exception as exc:
        meta["is_contiguous_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return meta


def compact_tensor_metadata(torch: Any, tensor: Any) -> dict[str, Any]:
    meta = tensor_metadata(torch, tensor)
    return {
        key: meta[key]
        for key in (
            "shape",
            "stride",
            "dtype",
            "device",
            "numel",
            "element_size",
            "bytes",
        )
        if key in meta
    }


class VulkanFallbackAttribution:
    def __init__(self, torch: Any, *, max_samples_per_key: int = 4) -> None:
        self.torch = torch
        self.max_samples_per_key = max_samples_per_key
        self.enabled = False
        self.rows_by_op: dict[str, dict[str, Any]] = {}
        self.rows_by_callsite: dict[str, dict[str, Any]] = {}
        self.rows_by_op_callsite: dict[tuple[str, str], dict[str, Any]] = {}
        self.samples: list[dict[str, Any]] = []
        self.total_dispatches = 0
        self.vulkan_dispatches = 0
        self.dispatch_error: str | None = None
        self._mode: Any = None
        self._cpu_count = None
        self._readback_count = None
        self._submit_counters = None

    def __enter__(self) -> "VulkanFallbackAttribution":
        try:
            from torch.utils._python_dispatch import TorchDispatchMode
        except Exception as exc:
            self.dispatch_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            return self

        ops = getattr(getattr(self.torch, "ops", None), "vulkan_prepack", None)
        self._cpu_count = getattr(ops, "cpu_fallback_count", None) if ops else None
        self._readback_count = (
            getattr(ops, "sync_readback_count", None) if ops else None
        )
        self._submit_counters = (
            getattr(ops, "submit_origin_counters", None) if ops else None
        )
        owner = self

        class AttributionMode(TorchDispatchMode):
            def __torch_dispatch__(
                self,
                func: Any,
                types: Any,
                args: tuple[Any, ...] = (),
                kwargs: dict[str, Any] | None = None,
            ) -> Any:
                return owner._dispatch(func, args, kwargs or {})

        self._mode = AttributionMode()
        self._mode.__enter__()
        self.enabled = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._mode is not None:
            self._mode.__exit__(exc_type, exc, tb)
        self.enabled = False

    def _counter_value(self, fn: Any) -> int:
        if fn is None:
            return 0
        try:
            return int(fn())
        except Exception:
            return 0

    def _submit_value(self) -> list[int]:
        if self._submit_counters is None:
            return []
        try:
            return [int(value) for value in self._submit_counters()]
        except Exception:
            return []

    def _dispatch(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self.total_dispatches += 1
        all_inputs = flatten_tensors(self.torch, args) + flatten_tensors(
            self.torch, kwargs
        )
        has_vulkan = any(is_vulkan_tensor(self.torch, tensor) for tensor in all_inputs)
        if has_vulkan:
            self.vulkan_dispatches += 1
        before_cpu = self._counter_value(self._cpu_count) if has_vulkan else 0
        before_readback = (
            self._counter_value(self._readback_count) if has_vulkan else 0
        )
        before_submit = self._submit_value() if has_vulkan else []
        result = func(*args, **kwargs)
        if not has_vulkan:
            return result
        after_cpu = self._counter_value(self._cpu_count)
        after_readback = self._counter_value(self._readback_count)
        after_submit = self._submit_value()
        cpu_delta = max(0, after_cpu - before_cpu)
        readback_delta = max(0, after_readback - before_readback)
        submit_delta = self._submit_delta(before_submit, after_submit)
        if cpu_delta == 0 and readback_delta == 0 and submit_delta["total"] == 0:
            return result
        self._record(
            func,
            args,
            kwargs,
            result,
            cpu_delta=cpu_delta,
            readback_delta=readback_delta,
            submit_delta=submit_delta,
        )
        return result

    def _submit_delta(self, before: list[int], after: list[int]) -> dict[str, int]:
        if not before or not after:
            return {"total": 0}
        width = min(len(before), len(after))
        deltas = [max(0, after[index] - before[index]) for index in range(width)]
        names = {
            "total": 0,
            "normal_cmd_submit_frequency": 1,
            "stack_planned_recording_submit": 5,
            "tensor_cpu_readback": 6,
            "retire_queue_drain": 8,
        }
        return {
            name: deltas[index] if index < len(deltas) else 0
            for name, index in names.items()
        }

    def _op_name(self, func: Any) -> str:
        schema = getattr(func, "_schema", None)
        if schema is not None:
            return str(schema).split("(", 1)[0]
        return str(func)

    def _callsite(self) -> dict[str, Any]:
        stack = traceback.extract_stack(limit=48)
        preferred = None
        fallback = None
        for frame in reversed(stack[:-2]):
            path = frame.filename.replace("\\", "/")
            lower_path = path.lower()
            if "torch/utils/_python_dispatch.py" in lower_path:
                continue
            if "scripts/benchmarks/benchmark_model_suite.py" in lower_path:
                continue
            if "/torch/" in lower_path or "\\torch\\" in lower_path:
                continue
            item = {
                "file": path,
                "line": int(frame.lineno),
                "function": frame.name,
                "code": (frame.line or "").strip()[:240],
            }
            if "transformers" in lower_path:
                preferred = item
                break
            if fallback is None:
                fallback = item
        selected = preferred or fallback or {}
        if not selected:
            return {"site": "unknown"}
        selected["site"] = (
            f"{selected.get('file')}:{selected.get('line')}:"
            f"{selected.get('function')}"
        )
        return selected

    def _classify(self, op_name: str, callsite: dict[str, Any], tensors: list[Any]) -> str:
        site = str(callsite.get("site", "")).lower()
        max_numel = max((int(tensor.numel()) for tensor in tensors), default=0)
        dtypes = {str(tensor.dtype) for tensor in tensors}
        if "transformers/integrations/sdpa_attention.py" in site:
            return "model_core_tensor_ops"
        if "transformers/cache_utils.py" in site:
            return "model_core_tensor_ops"
        if (
            "transformers/modeling_rope_utils.py" in site
            or "transformers/masking_utils.py" in site
        ):
            return "model_core_metadata" if max_numel <= 1024 else "model_core_tensor_ops"
        if "generation/logits_process.py" in site:
            return "logits_processors"
        if "generation/stopping_criteria.py" in site or "generation/utils.py" in site:
            return "tiny_generation_metadata" if max_numel <= 1024 else "generation_metadata"
        if max_numel <= 1024 and any("bool" in dtype or "int" in dtype for dtype in dtypes):
            return "tiny_generation_metadata"
        if "modeling" in site or "transformers/models" in site:
            return "model_core_tensor_ops"
        if "benchmark" in site:
            return "benchmark_harness_overhead"
        return "uncategorized"

    def _record(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
        *,
        cpu_delta: int,
        readback_delta: int,
        submit_delta: dict[str, int],
    ) -> None:
        op_name = self._op_name(func)
        callsite = self._callsite()
        input_tensors = flatten_tensors(self.torch, args) + flatten_tensors(
            self.torch, kwargs
        )
        output_tensors = flatten_tensors(self.torch, result)
        input_bytes = sum(
            int(tensor.numel() * tensor.element_size()) for tensor in input_tensors
        )
        output_bytes = sum(
            int(tensor.numel() * tensor.element_size()) for tensor in output_tensors
        )
        category = self._classify(op_name, callsite, input_tensors + output_tensors)
        payload = {
            "op": op_name,
            "callsite": callsite,
            "category": category,
            "count": 1,
            "cpu_fallback_count": int(cpu_delta),
            "sync_readback_count": int(readback_delta),
            "submit_origin_delta": submit_delta,
            "input_bytes": int(input_bytes),
            "output_bytes": int(output_bytes),
            "estimated_bytes": int(input_bytes + output_bytes),
            "inputs": [
                compact_tensor_metadata(self.torch, tensor) for tensor in input_tensors[:4]
            ],
            "outputs": [
                compact_tensor_metadata(self.torch, tensor) for tensor in output_tensors[:4]
            ],
        }
        self._merge(self.rows_by_op, op_name, payload)
        self._merge(self.rows_by_callsite, str(callsite.get("site", "unknown")), payload)
        self._merge(self.rows_by_op_callsite, (op_name, str(callsite.get("site", "unknown"))), payload)
        if len(self.samples) < 128:
            self.samples.append(payload)

    def _merge(self, rows: dict[Any, dict[str, Any]], key: Any, payload: dict[str, Any]) -> None:
        row = rows.get(key)
        if row is None:
            row = {
                "op": payload["op"],
                "callsite": payload["callsite"],
                "category": payload["category"],
                "count": 0,
                "cpu_fallback_count": 0,
                "sync_readback_count": 0,
                "submit_origin_delta": collections.Counter(),
                "input_bytes": 0,
                "output_bytes": 0,
                "estimated_bytes": 0,
                "samples": [],
            }
            rows[key] = row
        row["count"] += payload["count"]
        row["cpu_fallback_count"] += payload["cpu_fallback_count"]
        row["sync_readback_count"] += payload["sync_readback_count"]
        row["input_bytes"] += payload["input_bytes"]
        row["output_bytes"] += payload["output_bytes"]
        row["estimated_bytes"] += payload["estimated_bytes"]
        row["submit_origin_delta"].update(payload["submit_origin_delta"])
        if len(row["samples"]) < self.max_samples_per_key:
            row["samples"].append(
                {
                    "inputs": payload["inputs"],
                    "outputs": payload["outputs"],
                    "callsite": payload["callsite"],
                }
            )

    def _finalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        finalized = dict(row)
        finalized["submit_origin_delta"] = dict(row.get("submit_origin_delta", {}))
        return finalized

    def summary(self, top_n: int = 20) -> dict[str, Any]:
        by_op = [self._finalize_row(row) for row in self.rows_by_op.values()]
        by_callsite = [
            self._finalize_row(row) for row in self.rows_by_callsite.values()
        ]
        by_op_callsite = [
            self._finalize_row(row) for row in self.rows_by_op_callsite.values()
        ]
        categories: dict[str, dict[str, int]] = {}
        for row in by_op_callsite:
            bucket = categories.setdefault(
                str(row.get("category", "uncategorized")),
                {
                    "count": 0,
                    "cpu_fallback_count": 0,
                    "sync_readback_count": 0,
                    "estimated_bytes": 0,
                },
            )
            bucket["count"] += int(row.get("count", 0))
            bucket["cpu_fallback_count"] += int(row.get("cpu_fallback_count", 0))
            bucket["sync_readback_count"] += int(row.get("sync_readback_count", 0))
            bucket["estimated_bytes"] += int(row.get("estimated_bytes", 0))
        return {
            "enabled": self.enabled,
            "dispatch_error": self.dispatch_error,
            "total_dispatches": self.total_dispatches,
            "vulkan_dispatches": self.vulkan_dispatches,
            "categories": categories,
            "top_ops_by_count": sorted(
                by_op,
                key=lambda row: (
                    int(row.get("cpu_fallback_count", 0))
                    + int(row.get("sync_readback_count", 0)),
                    int(row.get("count", 0)),
                ),
                reverse=True,
            )[:top_n],
            "top_ops_by_estimated_bytes": sorted(
                by_op,
                key=lambda row: int(row.get("estimated_bytes", 0)),
                reverse=True,
            )[:top_n],
            "top_callsites_by_count": sorted(
                by_callsite,
                key=lambda row: (
                    int(row.get("cpu_fallback_count", 0))
                    + int(row.get("sync_readback_count", 0)),
                    int(row.get("count", 0)),
                ),
                reverse=True,
            )[:top_n],
            "top_callsites_by_readback": sorted(
                by_callsite,
                key=lambda row: (
                    int(row.get("sync_readback_count", 0)),
                    int(row.get("count", 0)),
                ),
                reverse=True,
            )[:top_n],
            "top_ops_by_tensor_cpu_readback_submit": sorted(
                by_op,
                key=lambda row: (
                    int(
                        row.get("submit_origin_delta", {}).get(
                            "tensor_cpu_readback", 0
                        )
                    ),
                    int(row.get("count", 0)),
                ),
                reverse=True,
            )[:top_n],
            "top_callsites_by_tensor_cpu_readback_submit": sorted(
                by_callsite,
                key=lambda row: (
                    int(
                        row.get("submit_origin_delta", {}).get(
                            "tensor_cpu_readback", 0
                        )
                    ),
                    int(row.get("count", 0)),
                ),
                reverse=True,
            )[:top_n],
            "top_op_callsites": sorted(
                by_op_callsite,
                key=lambda row: (
                    int(row.get("cpu_fallback_count", 0))
                    + int(row.get("sync_readback_count", 0)),
                    int(row.get("count", 0)),
                ),
                reverse=True,
            )[:top_n],
            "samples": self.samples,
        }


def summarize_model_parameters(torch: Any, model: Any) -> dict[str, Any]:
    total_numel = 0
    total_bytes = 0
    by_dtype: dict[str, dict[str, int]] = {}
    largest: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        bytes_ = int(parameter.numel() * parameter.element_size())
        total_numel += int(parameter.numel())
        total_bytes += bytes_
        dtype = str(parameter.dtype)
        bucket = by_dtype.setdefault(dtype, {"numel": 0, "bytes": 0, "count": 0})
        bucket["numel"] += int(parameter.numel())
        bucket["bytes"] += bytes_
        bucket["count"] += 1
        largest.append(
            {
                "name": name,
                "shape": [int(dim) for dim in parameter.shape],
                "dtype": dtype,
                "device": str(parameter.device),
                "numel": int(parameter.numel()),
                "bytes": bytes_,
            }
        )
    largest.sort(key=lambda item: int(item["bytes"]), reverse=True)
    return {
        "parameter_count": sum(bucket["count"] for bucket in by_dtype.values()),
        "total_numel": total_numel,
        "total_bytes": total_bytes,
        "by_dtype": by_dtype,
        "largest_parameters": largest[:16],
    }


def parse_key_value_log_line(line: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in line.strip().split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key] = value
    return parsed


def summarize_linear_plan_log(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"path": str(path) if path is not None else None, "exists": False}
    lines = path.read_text(errors="replace").splitlines()
    parsed = [parse_key_value_log_line(line) for line in lines if line.strip()]
    selected = [item for item in parsed if item.get("selected") == "1"]
    shape_counts: dict[str, int] = {}
    for item in selected:
        shape = f"m={item.get('m')} k={item.get('k')} n={item.get('n')}"
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    return {
        "path": str(path.resolve()),
        "exists": True,
        "line_count": len(lines),
        "selected_count": len(selected),
        "shape_counts": shape_counts,
        "tail": lines[-32:],
    }


def sum_retired_buffer_bytes(counters: dict[str, Any]) -> int:
    total = 0
    for row in counters.get("retired_resource_aggregate_snapshot", []):
        parsed = parse_key_value_log_line(str(row))
        if parsed.get("kind") == "buffer":
            try:
                total += int(parsed.get("bytes", "0"))
            except ValueError:
                pass
    return total


def write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_linear_forward_diagnostics(state: dict[str, Any]) -> dict[str, Any]:
    seen_modules = state.get("seen_modules", {})
    return {
        "installed": state.get("installed", False),
        "module_count": state.get("module_count", 0),
        "enter_count": state.get("enter_count", 0),
        "exit_count": state.get("exit_count", 0),
        "unique_module_count_seen": len(seen_modules),
        "unique_weight_bytes_seen": state.get("unique_weight_bytes_seen", 0),
        "last_successful": state.get("last_successful"),
        "failed_candidate": state.get("failed_candidate"),
        "last_entered": state.get("last_entered"),
        "per_module_call_counts": {
            name: int(payload.get("call_count", 0))
            for name, payload in sorted(seen_modules.items())
        },
    }


def reset_linear_forward_diagnostics_state(state: dict[str, Any]) -> None:
    handles = state.get("_handles", [])
    installed = state.get("installed", False)
    module_count = state.get("module_count", 0)
    state.clear()
    state.update(
        {
            "installed": installed,
            "events": [],
            "timeline": [],
            "last_entered": None,
            "last_successful": None,
            "failed_candidate": None,
            "module_count": module_count,
            "enter_count": 0,
            "exit_count": 0,
            "unique_weight_bytes_seen": 0,
            "seen_modules": {},
            "_handles": handles,
        }
    )


def summarize_linear_pack_cache(counters: dict[str, Any]) -> dict[str, Any]:
    totals = {
        "count": 0,
        "created": 0,
        "reused": 0,
        "packed_bytes": 0,
        "raw_weight_bytes": 0,
        "raw_bias_bytes": 0,
    }
    for row in counters.get("linear_pack_residency_snapshot", []):
        parsed = parse_key_value_log_line(str(row))
        for key in totals:
            try:
                totals[key] += int(parsed.get(key, "0"))
            except ValueError:
                pass

    summary: dict[str, Any] = {}
    for row in counters.get("packed_weight_residency_snapshot", []):
        if "packed_weight_residency_summary" not in str(row):
            continue
        for key, value in parse_key_value_log_line(str(row)).items():
            try:
                summary[key] = int(value)
            except ValueError:
                summary[key] = value
    return {
        "linear_pack_totals": totals,
        "packed_weight_cache_summary": summary,
    }


def install_linear_forward_diagnostics(torch: Any, model: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "installed": False,
        "events": [],
        "timeline": [],
        "last_entered": None,
        "last_successful": None,
        "failed_candidate": None,
        "module_count": 0,
        "enter_count": 0,
        "exit_count": 0,
        "unique_weight_bytes_seen": 0,
        "seen_modules": {},
    }
    handles = []

    def first_tensor(payload: Any) -> Any:
        if torch.is_tensor(payload):
            return payload
        if isinstance(payload, (list, tuple)):
            for item in payload:
                found = first_tensor(item)
                if found is not None:
                    return found
        if isinstance(payload, dict):
            for item in payload.values():
                found = first_tensor(item)
                if found is not None:
                    return found
        return None

    def make_event(name: str, module: Any, args: Any) -> dict[str, Any]:
        input_tensor = first_tensor(args)
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        seen_modules = state["seen_modules"]
        seen_before = name in seen_modules
        event = {
            "module": name,
            "type": type(module).__name__,
            "input": tensor_metadata(torch, input_tensor),
            "weight": tensor_metadata(torch, weight),
            "bias": tensor_metadata(torch, bias) if bias is not None else None,
            "seen_before": seen_before,
        }
        weight_meta = event["weight"]
        bias_meta = event["bias"] or {}
        event["estimated_raw_weight_bytes"] = int(weight_meta.get("bytes", 0) or 0)
        event["estimated_raw_bias_bytes"] = int(bias_meta.get("bytes", 0) or 0)
        event["estimated_min_pack_bytes"] = (
            event["estimated_raw_weight_bytes"] + event["estimated_raw_bias_bytes"]
        )
        if not seen_before:
            state["unique_weight_bytes_seen"] += event["estimated_raw_weight_bytes"]
            seen_modules[name] = {
                "weight": weight_meta,
                "estimated_raw_weight_bytes": event["estimated_raw_weight_bytes"],
            }
        event["cumulative_unique_weight_bytes_seen"] = state[
            "unique_weight_bytes_seen"
        ]
        return event

    def pre_hook(name: str) -> Any:
        def hook(module: Any, args: Any) -> None:
            event = make_event(name, module, args)
            state["enter_count"] += 1
            event["order"] = state["enter_count"]
            state["last_entered"] = event
            state["failed_candidate"] = event
            state["events"].append({"stage": "enter", **event})
            state["events"] = state["events"][-64:]
            state["timeline"].append({"stage": "enter", **event})
            state["timeline"] = state["timeline"][-256:]

        return hook

    def post_hook(name: str) -> Any:
        def hook(module: Any, args: Any, output: Any) -> None:
            event = make_event(name, module, args)
            state["exit_count"] += 1
            event["order"] = state["exit_count"]
            event["output"] = tensor_metadata(torch, first_tensor(output))
            state["last_successful"] = event
            state["failed_candidate"] = None
            state["events"].append({"stage": "exit", **event})
            state["events"] = state["events"][-64:]
            state["timeline"].append({"stage": "exit", **event})
            state["timeline"] = state["timeline"][-256:]

        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_pre_hook(pre_hook(name)))
            handles.append(module.register_forward_hook(post_hook(name)))
            state["module_count"] += 1
    state["installed"] = bool(handles)
    state["_handles"] = handles
    return state


def remove_linear_forward_diagnostics(state: dict[str, Any]) -> None:
    for handle in state.get("_handles", []):
        try:
            handle.remove()
        except Exception:
            pass
    state.pop("_handles", None)


def read_text_tail(path: Path, max_lines: int = 128) -> list[str]:
    try:
        if not path.is_file():
            return []
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    except Exception as exc:
        return [f"read_error={type(exc).__name__}: {str(exc)[:200]}"]


def run_torch_ops(args: argparse.Namespace, backend: str) -> BenchmarkRecord:
    task = "torch_backend_smoke"
    model_name = "torch_ops"
    model_id = DEFAULT_MODELS["torch_ops"]
    torch = import_torch()
    try:
        device, device_info = torch_device_for_backend(torch, backend, args.device_index)
    except Exception as exc:
        return make_failure(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            reason=f"{backend}_backend_unavailable",
            exc=exc,
            debug_traceback=args.debug_traceback,
        )

    if args.dtype != "float32":
        return make_failure(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            reason="torch_ops_smoke_only_supports_float32",
        )

    try:
        torch.manual_seed(20260522)
        x_cpu = torch.randn(1, 3, 16, 16, dtype=torch.float32)
        weight_cpu = torch.randn(4, 3, 3, 3, dtype=torch.float32)
        bias_cpu = torch.randn(4, dtype=torch.float32)

        def run_cpu() -> Any:
            return torch.nn.functional.conv2d(
                x_cpu,
                weight_cpu,
                bias_cpu,
                padding=1,
            ).relu()

        reference = run_cpu()
        if backend == "cpu":
            for _ in range(args.warmup):
                run_cpu()
            timing, output = measure_repeated(
                "device_resident_forward",
                args.repeats,
                run_cpu,
                torch_module=torch,
                backend=backend,
                device=device,
            )
        else:
            x = x_cpu.to(device)
            weight = weight_cpu.to(device)
            bias = bias_cpu.to(device)

            def run_device() -> Any:
                return torch.nn.functional.conv2d(
                    x,
                    weight,
                    bias,
                    padding=1,
                ).relu()

            for _ in range(args.warmup):
                run_device()
            timing, output = measure_repeated(
                "device_resident_forward",
                args.repeats,
                run_device,
                torch_module=torch,
                backend=backend,
                device=device,
            )

        output_cpu = output.cpu() if hasattr(output, "cpu") else output
        diff = (output_cpu - reference).abs()
        record = BenchmarkRecord(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device_info,
        )
        record.input = {
            "op": "conv2d_relu",
            "input_shape": list(x_cpu.shape),
            "weight_shape": list(weight_cpu.shape),
            "reference_backend": "cpu",
        }
        record.timings = {"device_resident_forward": timing}
        record.counters = {"vulkan_debug": snapshot_vulkan_debug_counters(torch, backend)}
        record.output_sanity = {
            "output_shape": list(output_cpu.shape),
            "finite": bool(torch.isfinite(output_cpu).all().item()),
            "max_abs_error_vs_cpu": float(diff.max().item()),
            "mean_abs_error_vs_cpu": float(diff.mean().item()),
        }
        record.environment = environment_summary()
        return record
    except Exception as exc:
        return make_failure(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            reason="torch_ops_smoke_failed",
            exc=exc,
            debug_traceback=args.debug_traceback,
        )


def load_diffusion_pipeline_with_source_tree_torch(torch: Any) -> tuple[Any, bool]:
    if "diffusers" in sys.modules:
        from diffusers import DiffusionPipeline

        return DiffusionPipeline, False

    original_version = importlib.metadata.version

    def patched_version(name: str) -> str:
        if name == "torch":
            return getattr(torch, "__version__", "0")
        return original_version(name)

    importlib.metadata.version = patched_version
    try:
        from diffusers import DiffusionPipeline
    finally:
        importlib.metadata.version = original_version
    return DiffusionPipeline, True


def load_lotus_depth_pipeline_class() -> tuple[Any, str | None]:
    lotus_repo = Path.cwd() / "agent_space" / "Lotus"
    pipeline_py = lotus_repo / "pipeline.py"
    if not pipeline_py.is_file():
        return None, None
    if str(lotus_repo) not in sys.path:
        sys.path.insert(0, str(lotus_repo))
    from pipeline import LotusDPipeline

    return LotusDPipeline, str(lotus_repo)


def run_lotus(args: argparse.Namespace, backend: str) -> BenchmarkRecord:
    task = "depth_estimation"
    model_name = "lotus"
    model_id = args.lotus_model_id
    if not module_available("diffusers"):
        return make_failure(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            reason="missing_diffusers",
        )
    torch = import_torch()
    distributed_import = prepare_benchmark_distributed_import(torch)
    try:
        lotus_pipeline_class, lotus_repo = load_lotus_depth_pipeline_class()
        diffusers_patched = False
        if lotus_pipeline_class is None:
            DiffusionPipeline, diffusers_patched = (
                load_diffusion_pipeline_with_source_tree_torch(torch)
            )
            lotus_pipeline_class = DiffusionPipeline

        device, device_info = torch_device_for_backend(torch, backend, args.device_index)
        setup_start = time.perf_counter()
        pipe = lotus_pipeline_class.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=not args.allow_downloads,
            torch_dtype=torch.float32,
        )
        pipe = pipe.to(device)
        setup_s = time.perf_counter() - setup_start
        image_metadata: dict[str, Any]
        if args.image_path:
            image_path = Path(args.image_path)
            image = load_rgb_image(image_path)
            image_metadata = image_file_metadata(image_path)
        else:
            image = make_test_image(args.image_size)
            image_metadata = {"image_size": args.image_size, "generated": True}

        def forward() -> Any:
            if lotus_repo is None:
                return pipe(image, num_inference_steps=args.num_inference_steps)
            import numpy as np

            rgb = np.array(image).astype(np.float32)
            rgb_tensor = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0)
            rgb_tensor = rgb_tensor / 127.5 - 1.0
            rgb_tensor = rgb_tensor.to(device)
            task_emb = torch.tensor([1, 0]).float().unsqueeze(0).to(device)
            task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
            return pipe(
                rgb_in=rgb_tensor,
                prompt="",
                output_type="np",
                timesteps=[1],
                task_emb=task_emb,
                processing_res=args.image_size,
                match_input_res=True,
            )

        for _ in range(args.warmup):
            forward()
        timing, output = measure_repeated(
            "device_resident_forward",
            args.repeats,
            forward,
            torch_module=torch,
            backend=backend,
            device=device,
        )
        record = BenchmarkRecord(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        record.device = device_info
        record.input = {
            "image": image_metadata,
            "num_inference_steps": args.num_inference_steps,
        }
        record.timings = {"setup_s": setup_s, "device_resident_forward": timing}
        record.counters = {"vulkan_debug": snapshot_vulkan_debug_counters(torch, backend)}
        lotus_depth_sanity: dict[str, Any] = {}
        if (
            lotus_repo is not None
            and hasattr(output, "images")
            and len(output.images) > 0
        ):
            import numpy as np

            depth = np.asarray(output.images[0])
            if depth.ndim == 3:
                depth = depth.mean(axis=-1)
            lotus_depth_sanity = {
                "depth_shape": list(depth.shape),
                "depth_min": float(np.nanmin(depth)),
                "depth_max": float(np.nanmax(depth)),
                "depth_mean": float(np.nanmean(depth)),
                "nan_count": int(np.isnan(depth).sum()),
                "inf_count": int(np.isinf(depth).sum()),
            }
        record.output_sanity = {
            "output_type": type(output).__name__,
            **lotus_depth_sanity,
            "diffusers_source_tree_torch_patch": diffusers_patched,
            "lotus_pipeline_source": lotus_repo,
            "distributed_c10d_status": distributed_import["status"],
            "distributed_import_shim": distributed_import,
        }
        record.environment = environment_summary()
        return record
    except Exception as exc:
        diffusers_reason = (
            "diffusers_source_tree_torch_incompatible"
            if isinstance(exc, AttributeError)
            and "'NoneType' object has no attribute 'to'" in str(exc)
            else (
                "lotus_pipeline_class_unavailable_in_diffusers"
                if "LotusDPipeline" in str(exc)
                else (
                    "diffusers_requires_installed_torch_distributed_metadata"
                    if "torch._C._distributed_c10d" in str(exc)
                    else (
                        "model_cache_or_dependency_unavailable"
                        if is_environment_skip(exc)
                        else "lotus_run_failed"
                    )
                )
            )
        )
        record = make_failure(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            reason=diffusers_reason,
            exc=exc,
            debug_traceback=args.debug_traceback,
            status="skip" if diffusers_reason != "lotus_run_failed" else None,
        )
        record.failure["distributed_c10d_status"] = benchmark_distributed_import_status(torch)
        record.failure["distributed_import_shim"] = distributed_import
        return record


def run_text_generation(
    args: argparse.Namespace,
    backend: str,
    *,
    task: str,
    model_name: str,
    model_id: str,
    prompt: str,
) -> BenchmarkRecord:
    if not module_available("transformers"):
        return make_failure(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            reason="missing_transformers",
        )
    torch = import_torch()
    distributed_import = prepare_benchmark_distributed_import(torch)
    linear_diag: dict[str, Any] = {}
    parameter_summary: dict[str, Any] = {}
    vulkan_parameter_summary: dict[str, Any] = {}
    fallback_attribution: dict[str, Any] | None = None
    linear_plan_log_path: Path | None = None
    if backend == "vulkan":
        linear_plan_log_path = (
            Path("agent_space")
            / f"{model_name}_vulkan_linear_plan_{int(time.time() * 1000)}.log"
        )
        linear_plan_log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            linear_plan_log_path.unlink()
        except FileNotFoundError:
            pass
        os.environ["PYTORCH_VULKAN_LINEAR_PLAN_LOG"] = str(
            linear_plan_log_path.resolve()
        )
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device, device_info = torch_device_for_backend(torch, backend, args.device_index)
        torch_dtype = torch.float32
        if args.dtype == "float16":
            torch_dtype = torch.float16
        elif args.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        setup_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=not args.allow_downloads,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=not args.allow_downloads,
        )
        parameter_summary = summarize_model_parameters(torch, model)
        try:
            model = model.to(device)
        except Exception as move_exc:
            move_info = classify_model_move_failure(
                torch,
                model,
                move_exc,
                backend,
            )
            if move_info:
                setattr(move_exc, "_benchmark_model_move_failure", move_info)
            raise
        if backend == "vulkan":
            vulkan_parameter_summary = summarize_model_parameters(torch, model)
        model.eval()
        linear_diag = install_linear_forward_diagnostics(torch, model)
        setup_s = time.perf_counter() - setup_start

        prompt_candidates = [prompt]
        if (
            args.min_generated_tokens > 0
            and model_name == "hy_mt"
            and args.translation_decode_reuse_prompt
            and args.translation_decode_reuse_prompt not in prompt_candidates
        ):
            prompt_candidates.append(args.translation_decode_reuse_prompt)

        prompt_attempts = []
        selected: dict[str, Any] | None = None
        last_output = None
        for prompt_index, prompt_candidate in enumerate(prompt_candidates):
            if backend == "vulkan":
                reset_vulkan_debug_counters(torch, backend)
            if linear_diag:
                reset_linear_forward_diagnostics_state(linear_diag)
            inputs = tokenizer(prompt_candidate, return_tensors="pt").to(device)

            def generate() -> Any:
                return model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            with torch.inference_mode():
                attribution = None
                if (
                    backend == "vulkan"
                    and model_name == "hy_mt"
                    and args.vulkan_fallback_attribution
                ):
                    attribution = VulkanFallbackAttribution(torch)
                with attribution if attribution is not None else contextlib.nullcontext():
                    for _ in range(args.warmup):
                        generate()
                    timing, output = measure_repeated(
                        "device_resident_generate",
                        args.repeats,
                        generate,
                        torch_module=torch,
                        backend=backend,
                        device=device,
                    )
                attempt_attribution = (
                    attribution.summary(top_n=32) if attribution is not None else None
                )
            generated_tokens = int(output.shape[-1] - inputs["input_ids"].shape[-1])
            attempt = {
                "prompt_index": prompt_index,
                "prompt": prompt_candidate,
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "max_new_tokens": args.max_new_tokens,
                "generated_tokens": generated_tokens,
                "decode_step_2_reached": generated_tokens >= 2,
            }
            if attempt_attribution is not None:
                attempt["fallback_attribution_summary"] = {
                    "total_dispatches": attempt_attribution.get("total_dispatches", 0),
                    "vulkan_dispatches": attempt_attribution.get("vulkan_dispatches", 0),
                    "categories": attempt_attribution.get("categories", {}),
                }
            prompt_attempts.append(attempt)
            selected = {
                "prompt": prompt_candidate,
                "inputs": inputs,
                "timing": timing,
                "generated_tokens": generated_tokens,
                "attempt": attempt,
                "fallback_attribution": attempt_attribution,
            }
            last_output = output
            if generated_tokens >= args.min_generated_tokens:
                break

        if selected is None or last_output is None:
            raise RuntimeError("HY-MT generation produced no benchmark attempt")

        inputs = selected["inputs"]
        timing = selected["timing"]
        output = last_output
        generated_tokens = selected["generated_tokens"]
        selected_prompt = selected["prompt"]
        selected_attempt = selected["attempt"]
        fallback_attribution = selected.get("fallback_attribution")
        decode_step_2_reached = generated_tokens >= 2
        if args.min_generated_tokens > 0 and generated_tokens < args.min_generated_tokens:
            raise RuntimeError(
                "HY-MT decode reuse probe did not reach the requested "
                f"{args.min_generated_tokens} generated tokens; "
                f"last_generated_tokens={generated_tokens}"
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_tokens = int(output.shape[-1] - inputs["input_ids"].shape[-1])
        record = BenchmarkRecord(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        record.device = device_info
        record.input = {
            "prompt": selected_prompt,
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            "max_new_tokens": args.max_new_tokens,
            "min_generated_tokens": args.min_generated_tokens,
            "decode_step_2_reached": decode_step_2_reached,
            "prompt_attempts": prompt_attempts,
        }
        record.timings = {"setup_s": setup_s, "device_resident_generate": timing}
        record.counters = {"vulkan_debug": snapshot_vulkan_debug_counters(torch, backend)}
        record.output_sanity = {
            "generated_tokens": generated_tokens,
            "decode_step_2_reached": decode_step_2_reached,
            "selected_prompt_attempt": selected_attempt,
            "tokens_per_s": (
                generated_tokens / timing["mean_s"] if timing["mean_s"] > 0 else 0.0
            ),
            "text": text,
            "distributed_c10d_status": distributed_import["status"],
            "distributed_import_shim": distributed_import,
        }
        if backend == "vulkan":
            record.output_sanity["linear_forward_diagnostics"] = (
                summarize_linear_forward_diagnostics(linear_diag)
            )
            record.output_sanity["linear_pack_cache"] = summarize_linear_pack_cache(
                record.counters["vulkan_debug"]
            )
            if fallback_attribution is not None:
                attribution_path = (
                    Path("agent_space")
                    / "hymt_fallback_readback_attribution_16tok.json"
                )
                callsite_path = (
                    Path("agent_space")
                    / "hymt_fallback_readback_top_callsites_16tok.json"
                )
                write_json_artifact(
                    attribution_path,
                    {
                        "model_name": model_name,
                        "model_id": model_id,
                        "backend": backend,
                        "prompt": selected_prompt,
                        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                        "max_new_tokens": args.max_new_tokens,
                        "generated_tokens": generated_tokens,
                        "vulkan_debug": record.counters["vulkan_debug"],
                        "attribution": fallback_attribution,
                    },
                )
                write_json_artifact(
                    callsite_path,
                    {
                        "model_name": model_name,
                        "model_id": model_id,
                        "backend": backend,
                        "top_callsites_by_count": fallback_attribution.get(
                            "top_callsites_by_count", []
                        ),
                        "top_callsites_by_readback": fallback_attribution.get(
                            "top_callsites_by_readback", []
                        ),
                        "top_callsites_by_tensor_cpu_readback_submit": (
                            fallback_attribution.get(
                                "top_callsites_by_tensor_cpu_readback_submit", []
                            )
                        ),
                        "top_op_callsites": fallback_attribution.get(
                            "top_op_callsites", []
                        ),
                    },
                )
                record.output_sanity["fallback_readback_attribution"] = {
                    "path": str(attribution_path.resolve()),
                    "top_callsites_path": str(callsite_path.resolve()),
                    "categories": fallback_attribution.get("categories", {}),
                    "total_dispatches": fallback_attribution.get(
                        "total_dispatches", 0
                    ),
                    "vulkan_dispatches": fallback_attribution.get(
                        "vulkan_dispatches", 0
                    ),
                }
        if linear_diag:
            remove_linear_forward_diagnostics(linear_diag)
        record.environment = environment_summary()
        return record
    except Exception as exc:
        reason = classify_transformers_failure(exc, backend, model_name)
        record = make_failure(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            reason=reason,
            exc=exc,
            debug_traceback=args.debug_traceback,
            status="skip" if reason != f"{model_name}_run_failed" else None,
        )
        move_failure = getattr(exc, "_benchmark_model_move_failure", None)
        if move_failure:
            record.failure["model_move_failure"] = move_failure
        if backend == "vulkan":
            failure_counters = snapshot_vulkan_debug_counters(torch, backend)
            if linear_diag:
                record.failure["linear_forward_diagnostics"] = {
                    "installed": linear_diag.get("installed", False),
                    "module_count": linear_diag.get("module_count", 0),
                    "enter_count": linear_diag.get("enter_count", 0),
                    "exit_count": linear_diag.get("exit_count", 0),
                    "unique_module_count_seen": len(
                        linear_diag.get("seen_modules", {})
                    ),
                    "unique_weight_bytes_seen": linear_diag.get(
                        "unique_weight_bytes_seen", 0
                    ),
                    "last_successful": linear_diag.get("last_successful"),
                    "failed_candidate": linear_diag.get("failed_candidate"),
                    "last_entered": linear_diag.get("last_entered"),
                    "recent_events": linear_diag.get("events", [])[-16:],
                    "recent_timeline": linear_diag.get("timeline", [])[-64:],
                }
            if parameter_summary:
                record.failure["model_parameter_summary"] = parameter_summary
            if vulkan_parameter_summary:
                record.failure["vulkan_model_parameter_summary"] = (
                    vulkan_parameter_summary
                )
            if linear_plan_log_path is not None:
                record.failure["linear_plan_log"] = summarize_linear_plan_log(
                    linear_plan_log_path
                )
            record.failure["vulkan_failure_counters"] = failure_counters
            if reason == "model_weight_vulkan_oom":
                raw_vs_packed = {
                    "model_name": model_name,
                    "model_id": model_id,
                    "backend": backend,
                    "raw_model_parameters": vulkan_parameter_summary,
                    "linear_pack_residency_snapshot": failure_counters.get(
                        "linear_pack_residency_snapshot", []
                    ),
                    "packed_weight_residency_snapshot": failure_counters.get(
                        "packed_weight_residency_snapshot", []
                    ),
                    "vulkan_memory_residency_snapshot": failure_counters.get(
                        "vulkan_memory_residency_snapshot", []
                    ),
                    "linear_forward_diagnostics": record.failure.get(
                        "linear_forward_diagnostics", {}
                    ),
                }
                raw_vs_packed_path = (
                    Path("agent_space")
                    / "hymt_raw_vs_packed_weight_residency.json"
                )
                write_json_artifact(raw_vs_packed_path, raw_vs_packed)
                memory_summary = {
                    "model_name": model_name,
                    "model_id": model_id,
                    "backend": backend,
                    "failure_reason": reason,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "linear_forward_diagnostics": record.failure.get(
                        "linear_forward_diagnostics", {}
                    ),
                    "model_parameter_summary": parameter_summary,
                    "vulkan_model_parameter_summary": vulkan_parameter_summary,
                    "linear_plan_log": record.failure.get("linear_plan_log", {}),
                    "vulkan_failure_counters": failure_counters,
                    "retired_buffer_bytes_observed": sum_retired_buffer_bytes(
                        failure_counters
                    ),
                    "notes": {
                        "live_vulkan_buffer_bytes": "reported by vulkan_memory_residency_snapshot",
                        "retire_poll_before_failed_pack": "not attempted; diagnostics only",
                        "packed_weight_bytes_are_reported_by_linear_pack_residency_snapshot": True,
                    },
                }
                memory_summary_path = (
                    Path("agent_space") / "hymt_linear_pack_memory_summary.json"
                )
                write_json_artifact(memory_summary_path, memory_summary)
                record.failure["linear_pack_memory_summary_path"] = str(
                    memory_summary_path.resolve()
                )
                record.failure["raw_vs_packed_weight_residency_path"] = str(
                    raw_vs_packed_path.resolve()
                )
            if linear_diag:
                remove_linear_forward_diagnostics(linear_diag)
        record.failure["distributed_c10d_status"] = benchmark_distributed_import_status(torch)
        record.failure["distributed_import_shim"] = distributed_import
        return record


def classify_transformers_failure(
    exc: BaseException,
    backend: str,
    model_name: str,
) -> str:
    text = "".join(traceback.format_exception(exc))
    if "torch._C._distributed_c10d" in text:
        return "transformers_source_tree_torch_distributed_missing"
    if backend == "vulkan" and "AutoModelForCausalLM" in text:
        return "transformers_source_tree_torch_distributed_missing"
    if backend == "vulkan" and "VK_ERROR_OUT_OF_DEVICE_MEMORY" in text:
        return "model_weight_vulkan_oom"
    if backend == "vulkan":
        return "transformers_vulkan_mapping_failed"
    if is_environment_skip(exc):
        return "model_cache_or_access_unavailable"
    return f"{model_name}_run_failed"


def try_patch_paddleocr_transformers_device(torch: Any, backend: str) -> dict[str, Any]:
    if backend != "vulkan":
        return {"attempted": False}
    patch_info: dict[str, Any] = {
        "attempted": True,
        "patched": False,
        "paddlex_predictor_patched": False,
    }
    try:
        import transformers.modeling_utils as modeling_utils

        original_to = modeling_utils.PreTrainedModel.to
        patch_info["_original_to"] = original_to

        def to_vulkan_by_default(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not args and "device" not in kwargs:
                return original_to(self, torch.device("vulkan"))
            return original_to(self, *args, **kwargs)

        modeling_utils.PreTrainedModel.to = to_vulkan_by_default
        patch_info["patched"] = True
    except Exception as exc:
        patch_info["patch_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    try:
        from paddlex.inference.models.predictors import transformers_predictor

        predictor_cls = transformers_predictor.TransformersPredictor
        original_get_device = predictor_cls._get_manual_torch_device
        patch_info["_original_paddlex_get_manual_torch_device"] = original_get_device

        def get_vulkan_device(self: Any) -> str:
            return "vulkan"

        predictor_cls._get_manual_torch_device = get_vulkan_device
        patch_info["paddlex_predictor_patched"] = True
    except Exception as exc:
        patch_info["paddlex_patch_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return patch_info


def restore_paddleocr_transformers_device_patch(patch_info: dict[str, Any]) -> None:
    original = patch_info.get("_original_to")
    if original is None:
        return
    try:
        import transformers.modeling_utils as modeling_utils

        modeling_utils.PreTrainedModel.to = original
    except Exception:
        pass
    original_get_device = patch_info.get("_original_paddlex_get_manual_torch_device")
    if original_get_device is not None:
        try:
            from paddlex.inference.models.predictors import transformers_predictor

            transformers_predictor.TransformersPredictor._get_manual_torch_device = (
                original_get_device
            )
        except Exception:
            pass


def install_paddleocr_postprocess_cpu_metadata_patch(
    torch: Any,
    backend: str,
) -> tuple[list[dict[str, Any]], Any]:
    if backend != "vulkan":
        return [], None
    try:
        from transformers.models.pp_ocrv5_server_det import (
            image_processing_pp_ocrv5_server_det,
        )
    except Exception:
        return [], None

    cls = image_processing_pp_ocrv5_server_det.PPOCRV5ServerDetImageProcessor
    original = cls.post_process_object_detection
    calls: list[dict[str, Any]] = []

    def post_process_object_detection_cpu_metadata(
        self: Any,
        predictions: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        last_hidden_state = predictions.last_hidden_state
        target_device = str(last_hidden_state.device)
        cpu_predictions = types.SimpleNamespace(last_hidden_state=last_hidden_state.cpu())
        original_boxes_from_bitmap = self._boxes_from_bitmap

        def boxes_from_bitmap_recorder(*box_args: Any, **box_kwargs: Any) -> Any:
            boxes, scores = original_boxes_from_bitmap(*box_args, **box_kwargs)
            calls.append(
                {
                    "reason": "paddleocr_postprocess_cpu_metadata_tensor",
                    "module": cls.__module__,
                    "target_device_without_patch": target_device,
                    "boxes_numpy_dtype": str(getattr(boxes, "dtype", None)),
                    "boxes_shape": list(getattr(boxes, "shape", ())),
                    "boxes_numel": int(getattr(boxes, "size", 0)),
                    "scores_count": len(scores),
                    "participates_in_model_compute": False,
                }
            )
            return boxes, scores

        self._boxes_from_bitmap = boxes_from_bitmap_recorder
        try:
            results = original(self, cpu_predictions, *args, **kwargs)
        finally:
            self._boxes_from_bitmap = original_boxes_from_bitmap

        for call, result in zip(calls[-len(results) :], results):
            boxes = result.get("boxes")
            scores = result.get("scores")
            labels = result.get("labels")
            call.update(
                {
                    "result_boxes_dtype": str(getattr(boxes, "dtype", None)),
                    "result_boxes_device": str(getattr(boxes, "device", None)),
                    "result_scores_dtype": str(getattr(scores, "dtype", None)),
                    "result_scores_device": str(getattr(scores, "device", None)),
                    "result_labels_dtype": str(getattr(labels, "dtype", None)),
                    "result_labels_device": str(getattr(labels, "device", None)),
                }
            )
        return results

    cls.post_process_object_detection = post_process_object_detection_cpu_metadata
    return calls, (cls, original)


def restore_paddleocr_postprocess_cpu_metadata_patch(patch: Any) -> None:
    if patch is None:
        return
    cls, original = patch
    cls.post_process_object_detection = original


def public_patch_info(patch_info: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in patch_info.items() if not key.startswith("_")}


def move_paddleocr_transformers_models_to_backend(
    torch: Any,
    root: Any,
    backend: str,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "attempted": backend == "vulkan",
        "visited_objects": 0,
        "model_count": 0,
        "moved_count": 0,
        "model_types": [],
    }
    if backend != "vulkan":
        return info
    import transformers.modeling_utils as modeling_utils

    device = torch.device("vulkan")
    pretrained_model = modeling_utils.PreTrainedModel
    visited: set[int] = set()
    stack: list[Any] = [root]
    max_visits = 20000

    while stack and info["visited_objects"] < max_visits:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        info["visited_objects"] += 1
        if isinstance(obj, pretrained_model):
            info["model_count"] += 1
            model_type = type(obj).__name__
            info["model_types"].append(model_type)
            obj.to(device)
            info["moved_count"] += 1
            continue
        if isinstance(obj, dict):
            stack.extend(obj.values())
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
            continue
        obj_dict = getattr(obj, "__dict__", None)
        if obj_dict:
            stack.extend(obj_dict.values())

    if stack:
        info["truncated"] = True
    return info


def classify_paddleocr_backend_failure(exc: BaseException, backend: str) -> str:
    text = "".join(traceback.format_exception(exc))
    if "torch._C._distributed_c10d" in text:
        return "paddleocr_transformers_source_tree_torch_distributed_missing"
    if backend == "vulkan":
        return "paddleocr_vulkan_mapping_failed"
    return classify_paddleocr_failure(exc)


def device_info_for_backend(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    torch = import_torch()
    try:
        _, device_info = torch_device_for_backend(torch, backend, args.device_index)
        return device_info
    except Exception as exc:
        return {"type": backend, "probe_error": repr(exc)}


def classify_paddleocr_failure(exc: BaseException) -> str:
    text = str(exc)
    if "Torchvision library" in text or "No module named 'torchvision'" in text:
        return "paddleocr_transformers_missing_torchvision"
    if "AutoImageProcessor" in text or "torch._C._distributed_c10d" in text:
        return "paddleocr_transformers_requires_installed_torch_metadata"
    if "No module named 'sympy'" in text:
        return "paddleocr_transformers_source_tree_torch_dependency_missing"
    if "AutoModelForImageClassification" in text:
        return "paddleocr_transformers_source_tree_torch_distributed_missing"
    if "dependency 'transformers' is not installed" in text:
        return "paddleocr_transformers_dependency_missing"
    if "Unrecognized image processor" in text:
        return "paddleocr_transformers_model_processor_unregistered"
    if is_environment_skip(exc):
        return "paddleocr_dependency_or_model_unavailable"
    return "paddleocr_run_failed"


def run_paddleocr(args: argparse.Namespace, backend: str) -> BenchmarkRecord:
    task = "ocr_document_pipeline"
    model_name = "paddleocr_transformers"
    model_id = args.paddleocr_model_id
    if not module_available("paddleocr"):
        return make_failure(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            reason="missing_paddleocr",
        )
    if not args.allow_downloads and not paddleocr_cache_has_models():
        return make_failure(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            reason="paddleocr_model_cache_unavailable_downloads_disabled",
        )
    torch = import_torch()
    device_info = device_info_for_backend(args, backend)
    device_kwargs: dict[str, Any] = {}
    if backend == "cpu":
        device_kwargs["device"] = "cpu"
    patch_info: dict[str, Any] = {"attempted": False}
    grid_sample_calls: list[dict[str, Any]] = []
    grid_sample_patch = None
    postprocess_metadata_calls: list[dict[str, Any]] = []
    postprocess_metadata_patch = None
    distributed_import = prepare_benchmark_distributed_import(torch)
    try:
        from paddleocr import PaddleOCR

        image_path = (
            Path(args.document_image_path)
            if args.document_image_path
            else make_document_image(Path(args.out).with_suffix(".doc.png"), args.image_size)
        )
        patch_info = try_patch_paddleocr_transformers_device(torch, backend)
        if backend == "vulkan":
            grid_sample_calls, grid_sample_patch = install_grid_sample_call_recorder(
                torch,
                backend,
            )
            postprocess_metadata_calls, postprocess_metadata_patch = (
                install_paddleocr_postprocess_cpu_metadata_patch(torch, backend)
            )
        setup_start = time.perf_counter()
        try:
            ocr = PaddleOCR(engine="transformers", **device_kwargs)
        except TypeError:
            ocr = PaddleOCR(**device_kwargs)
        model_mapping_info = move_paddleocr_transformers_models_to_backend(
            torch,
            ocr,
            backend,
        )
        setup_s = time.perf_counter() - setup_start

        def run_ocr() -> Any:
            return ocr.predict(str(image_path))

        for _ in range(args.warmup):
            run_ocr()
        durations: list[float] = []
        output: Any = None
        for _ in range(args.repeats):
            start = time.perf_counter()
            output = run_ocr()
            durations.append(time.perf_counter() - start)
        timing = summarize_durations("end_to_end_pipeline", durations)
        record = BenchmarkRecord(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        record.device = device_info
        record.input = {
            "document_image": image_file_metadata(image_path),
            "image_size": args.image_size,
        }
        record.timings = {"setup_s": setup_s, "end_to_end": timing}
        record.counters = {"vulkan_debug": snapshot_vulkan_debug_counters(torch, backend)}
        record.output_sanity = {
            "output_type": type(output).__name__,
            "output_items": len(output) if hasattr(output, "__len__") else None,
            "paddleocr_device_patch": public_patch_info(patch_info),
            "paddleocr_transformers_model_mapping": model_mapping_info,
            "grid_sample_calls": grid_sample_calls,
            "paddleocr_postprocess_cpu_metadata_tensors": postprocess_metadata_calls,
            "distributed_c10d_status": distributed_import["status"],
            "distributed_import_shim": distributed_import,
        }
        record.environment = environment_summary()
        restore_paddleocr_postprocess_cpu_metadata_patch(postprocess_metadata_patch)
        restore_grid_sample_call_recorder(grid_sample_patch)
        restore_paddleocr_transformers_device_patch(patch_info)
        return record
    except Exception as exc:
        restore_paddleocr_postprocess_cpu_metadata_patch(postprocess_metadata_patch)
        restore_grid_sample_call_recorder(grid_sample_patch)
        restore_paddleocr_transformers_device_patch(patch_info)
        reason = classify_paddleocr_backend_failure(exc, backend)
        record = make_failure(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            reason=reason,
            exc=exc,
            debug_traceback=args.debug_traceback,
            status="skip" if reason != "paddleocr_run_failed" else None,
        )
        record.failure["grid_sample_calls"] = grid_sample_calls
        record.failure["paddleocr_postprocess_cpu_metadata_tensors"] = (
            postprocess_metadata_calls
        )
        record.failure["distributed_c10d_status"] = benchmark_distributed_import_status(torch)
        record.failure["distributed_import_shim"] = distributed_import
        return record


def run_task(args: argparse.Namespace, task: str, backend: str) -> BenchmarkRecord:
    try:
        torch = import_torch()
    except Exception as exc:
        model_name = task
        model_id = DEFAULT_MODELS.get(task, task)
        return make_failure(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            reason="torch_import_failed",
            exc=exc,
            debug_traceback=args.debug_traceback,
            status="skip",
        )
    reset_vulkan_debug_counters(torch, backend)
    device_info: dict[str, Any] = {}
    try:
        _, device_info = torch_device_for_backend(torch, backend, args.device_index)
    except Exception as exc:
        device_info = {"type": backend, "probe_error": repr(exc)}
    if task == "torch_ops":
        record = run_torch_ops(args, backend)
    elif task == "lotus":
        record = run_lotus(args, backend)
    elif task == "hy_mt":
        record = run_text_generation(
            args,
            backend,
            task="translation",
            model_name="hy_mt",
            model_id=args.hy_mt_model_id,
            prompt=args.translation_prompt,
        )
    elif task == "gemma":
        record = run_text_generation(
            args,
            backend,
            task="llm_generation",
            model_name="gemma",
            model_id=args.gemma_model_id,
            prompt=args.gemma_prompt,
        )
    elif task == "paddleocr":
        record = run_paddleocr(args, backend)
    else:
        raise ValueError(f"Unknown task: {task}")
    if not record.device:
        record.device = device_info
    if backend == "vulkan" and "vulkan_debug" not in record.counters:
        record.counters["vulkan_debug"] = snapshot_vulkan_debug_counters(torch, backend)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-model benchmark smoke/profiling entries."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["torch_ops", "lotus", "hy_mt", "paddleocr", "gemma"],
        choices=["torch_ops", "lotus", "hy_mt", "paddleocr", "gemma"],
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["cpu"],
        choices=["cpu", "vulkan", "directml", "cuda"],
    )
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-path")
    parser.add_argument("--document-image-path")
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--min-generated-tokens",
        type=int,
        default=0,
        help="Require at least this many generated tokens for generation probes.",
    )
    parser.add_argument(
        "--cache-dir",
        default="agent_space/hf_home",
        help="Hugging Face cache root. Defaults inside the repo scratch space.",
    )
    parser.add_argument(
        "--dependency-path",
        action="append",
        default=[],
        help=(
            "Extra dependency path appended after importing local PyTorch. "
            "Prefer task-specific virtual environments from prepare_model_suite_envs.py."
        ),
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow model downloads. Default is local-cache-only smoke behavior.",
    )
    parser.add_argument(
        "--debug-traceback",
        action="store_true",
        help="Include short Python tracebacks in failure rows.",
    )
    parser.add_argument(
        "--vulkan-fallback-attribution",
        action="store_true",
        help=(
            "Attribute HY-MT Vulkan CPU fallback/readback deltas by dispatched op "
            "and Python call site."
        ),
    )
    parser.add_argument(
        "--torch-import-mode",
        choices=["source", "installed"],
        default="source",
        help=(
            "Use the repo source-tree torch for Vulkan backend testing, or an "
            "installed wheel torch for CPU model framework coverage."
        ),
    )
    parser.add_argument("--lotus-model-id", default=DEFAULT_MODELS["lotus"])
    parser.add_argument("--hy-mt-model-id", default=DEFAULT_MODELS["hy_mt"])
    parser.add_argument("--paddleocr-model-id", default=DEFAULT_MODELS["paddleocr"])
    parser.add_argument("--gemma-model-id", default=DEFAULT_MODELS["gemma"])
    parser.add_argument(
        "--translation-prompt",
        default="Translate to French: The Vulkan backend should report clean skips.",
    )
    parser.add_argument(
        "--translation-decode-reuse-prompt",
        default=HY_MT_DECODE_REUSE_PROMPT,
        help=(
            "Fallback HY-MT prompt used by the decode-reuse probe when the "
            "default prompt stops before the requested token count."
        ),
    )
    parser.add_argument(
        "--gemma-prompt",
        default="Write one sentence about GPU benchmark coverage.",
    )
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--out", default="agent_space/model_suite_benchmark.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_cache(args.cache_dir)
    configure_torch_import_mode(args.torch_import_mode)
    if args.dependency_path:
        import_torch()
        for raw_path in args.dependency_path:
            path = str(Path(raw_path).resolve())
            if path not in sys.path:
                sys.path.append(path)
    probe = probe_accelerators()
    records: list[BenchmarkRecord] = []
    if not args.probe_only:
        for backend in args.backends:
            for task in args.tasks:
                records.append(run_task(args, task, backend))
    out_path = Path(args.out).resolve()
    write_records(out_path, records, probe)
    print(out_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
