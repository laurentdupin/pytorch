from __future__ import annotations

import argparse
import importlib.metadata
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


def benchmark_distributed_import_status(torch: Any) -> str:
    if hasattr(getattr(torch, "_C", None), "_distributed_c10d"):
        return "real_distributed_c10d"
    if "transformers.generation.continuous_batching" in sys.modules:
        module = sys.modules["transformers.generation.continuous_batching"]
        if getattr(module, "_benchmark_distributed_import_shim", False):
            return "distributed_import_shim"
    return "missing_distributed_c10d"


def install_benchmark_distributed_import_shim(torch: Any) -> dict[str, Any]:
    if hasattr(getattr(torch, "_C", None), "_distributed_c10d"):
        return {"status": "real_distributed_c10d", "installed": False}
    module_name = "transformers.generation.continuous_batching"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return {
            "status": (
                "distributed_import_shim"
                if getattr(existing, "_benchmark_distributed_import_shim", False)
                else "missing_distributed_c10d"
            ),
            "installed": False,
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
        "scope": "transformers_continuous_batching_import_only",
    }


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
    if backend == "vulkan":
        return make_failure(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            reason="diffusers_pipeline_does_not_support_torch_vulkan_device",
        )
    try:
        DiffusionPipeline, diffusers_patched = (
            load_diffusion_pipeline_with_source_tree_torch(torch)
        )

        device, device_info = torch_device_for_backend(torch, backend, args.device_index)
        setup_start = time.perf_counter()
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=not args.allow_downloads,
        )
        pipe = pipe.to(device)
        setup_s = time.perf_counter() - setup_start
        image = make_test_image(args.image_size)

        def forward() -> Any:
            return pipe(image, num_inference_steps=args.num_inference_steps)

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
        record.input = {"image_size": args.image_size, "num_inference_steps": args.num_inference_steps}
        record.timings = {"setup_s": setup_s, "device_resident_forward": timing}
        record.counters = {"vulkan_debug": snapshot_vulkan_debug_counters(torch, backend)}
        record.output_sanity = {
            "output_type": type(output).__name__,
            "diffusers_source_tree_torch_patch": diffusers_patched,
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
        return make_failure(
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
    distributed_import = install_benchmark_distributed_import_shim(torch)
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
        model.eval()
        setup_s = time.perf_counter() - setup_start
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        def generate() -> Any:
            return model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

        with torch.inference_mode():
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
            "prompt": prompt,
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            "max_new_tokens": args.max_new_tokens,
        }
        record.timings = {"setup_s": setup_s, "device_resident_generate": timing}
        record.counters = {"vulkan_debug": snapshot_vulkan_debug_counters(torch, backend)}
        record.output_sanity = {
            "generated_tokens": generated_tokens,
            "tokens_per_s": (
                generated_tokens / timing["mean_s"] if timing["mean_s"] > 0 else 0.0
            ),
            "text": text,
            "distributed_c10d_status": distributed_import["status"],
            "distributed_import_shim": distributed_import,
        }
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
    distributed_import = install_benchmark_distributed_import_shim(torch)
    try:
        from paddleocr import PaddleOCR

        image_path = make_document_image(Path(args.out).with_suffix(".doc.png"), args.image_size)
        patch_info = try_patch_paddleocr_transformers_device(torch, backend)
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
        record.input = {"document_image": str(image_path), "image_size": args.image_size}
        record.timings = {"setup_s": setup_s, "end_to_end": timing}
        record.counters = {"vulkan_debug": snapshot_vulkan_debug_counters(torch, backend)}
        record.output_sanity = {
            "output_type": type(output).__name__,
            "output_items": len(output) if hasattr(output, "__len__") else None,
            "paddleocr_device_patch": public_patch_info(patch_info),
            "paddleocr_transformers_model_mapping": model_mapping_info,
            "distributed_c10d_status": distributed_import["status"],
            "distributed_import_shim": distributed_import,
        }
        record.environment = environment_summary()
        restore_paddleocr_transformers_device_patch(patch_info)
        return record
    except Exception as exc:
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
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
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
