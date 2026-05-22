from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from benchmark_suite_common import (
    BenchmarkRecord,
    environment_summary,
    import_torch,
    make_failure,
    measure_repeated,
    module_available,
    probe_accelerators,
    reset_vulkan_debug_counters,
    snapshot_vulkan_debug_counters,
    torch_device_for_backend,
    write_records,
)
from bench_common import summarize_durations, write_json


DEFAULT_MODELS = {
    "lotus": "jingheya/lotus-depth-d-v1-1",
    "hy_mt": "tencent/HY-MT1.5-1.8B",
    "paddleocr": "PaddleOCR 3.5 Transformers backend",
    "gemma": "google/gemma-4-E2B-it",
}


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
    try:
        from diffusers.utils import is_torch_available

        if not is_torch_available():
            return make_failure(
                task=task,
                model_name=model_name,
                model_id=model_id,
                backend=backend,
                device_index=args.device_index,
                dtype=args.dtype,
                warmup=args.warmup,
                repeats=args.repeats,
                reason="diffusers_does_not_detect_local_torch_install",
            )
    except Exception:
        pass
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
    torch = import_torch()
    try:
        from diffusers import DiffusionPipeline

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
        record.output_sanity = {"output_type": type(output).__name__}
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
            reason="lotus_run_failed",
            exc=exc,
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
            reason="transformers_generation_does_not_support_torch_vulkan_device",
        )
    torch = import_torch()
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
        ).to(device)
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
            reason=f"{model_name}_run_failed",
            exc=exc,
        )


def run_paddleocr(args: argparse.Namespace, backend: str) -> BenchmarkRecord:
    task = "ocr_document_pipeline"
    model_name = "paddleocr_transformers"
    model_id = args.paddleocr_model_id
    if backend != "cpu":
        return make_failure(
            task=task,
            model_name=model_name,
            model_id=model_id,
            backend=backend,
            device_index=args.device_index,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            reason="paddleocr_transformers_backend_is_not_mapped_to_requested_torch_backend",
        )
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
    try:
        from paddleocr import PaddleOCR

        image_path = make_document_image(Path(args.out).with_suffix(".doc.png"), args.image_size)
        setup_start = time.perf_counter()
        try:
            ocr = PaddleOCR(engine="transformers")
        except TypeError:
            ocr = PaddleOCR()
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
        record.device = {"type": "cpu", "index": None, "name": "CPU"}
        record.input = {"document_image": str(image_path), "image_size": args.image_size}
        record.timings = {"setup_s": setup_s, "end_to_end": timing}
        record.output_sanity = {
            "output_type": type(output).__name__,
            "output_items": len(output) if hasattr(output, "__len__") else None,
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
            reason="paddleocr_run_failed",
            exc=exc,
        )


def run_task(args: argparse.Namespace, task: str, backend: str) -> BenchmarkRecord:
    torch = import_torch()
    reset_vulkan_debug_counters(torch, backend)
    device_info: dict[str, Any] = {}
    try:
        _, device_info = torch_device_for_backend(torch, backend, args.device_index)
    except Exception as exc:
        device_info = {"type": backend, "probe_error": repr(exc)}
    if task == "lotus":
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
        default=["lotus", "hy_mt", "paddleocr", "gemma"],
        choices=["lotus", "hy_mt", "paddleocr", "gemma"],
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
        "--allow-downloads",
        action="store_true",
        help="Allow model downloads. Default is local-cache-only smoke behavior.",
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
