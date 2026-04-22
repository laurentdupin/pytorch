from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import cv2
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTALL_ROOT = Path(
    os.environ.get(
        "DEPTH_EXTRACTOR_DEPTH_ANYTHING_V25",
        r"C:\Users\REDACTED\AppData\Local\DepthExtractor\Depth-Anything-V25",
    )
)


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


@contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def resolve_depth_anything_root(install_root: Path) -> Path:
    root = install_root / "Depth-Anything"
    if not root.exists():
        raise FileNotFoundError(f"Depth-Anything root does not exist: {root}")
    return root


def load_installed_load_model(depth_anything_root: Path) -> ModuleType:
    os.environ.setdefault("TORCH_HOME", str(depth_anything_root / "torchcache"))
    os.environ.setdefault("HF_HOME", str(depth_anything_root / "hfcache"))
    if str(depth_anything_root) not in sys.path:
        sys.path.insert(0, str(depth_anything_root))

    load_model_path = depth_anything_root / "LoadModel.py"
    if not load_model_path.exists():
        raise FileNotFoundError(f"LoadModel.py does not exist: {load_model_path}")

    spec = importlib.util.spec_from_file_location(
        "desktop_depth_anything_load_model",
        load_model_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load spec for {load_model_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with pushd(depth_anything_root):
        spec.loader.exec_module(module)
    return module


def set_parameters(
    load_model: ModuleType,
    *,
    encoder: str,
    size: int,
    half_precision: str,
) -> None:
    load_model.SetParameter("Encoder", encoder)
    load_model.SetParameter("Size", str(size))
    load_model.SetParameter("HalfPrecision", half_precision)


def load_model_on_device(
    load_model: ModuleType,
    depth_anything_root: Path,
    *,
    device: str,
) -> float:
    start = time.perf_counter()
    with pushd(depth_anything_root):
        load_model.LoadModel(device)
    return time.perf_counter() - start


def make_desktop_packet(image_path: Path) -> tuple[list[tuple[bytes, Any]], dict[str, Any]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif image.shape[2] != 4:
        raise RuntimeError(
            f"Unsupported image shape for desktop packet: {image.shape!r}"
        )

    header = SimpleNamespace(
        Type=1,
        Index=0,
        Size=int(image.shape[0] * image.shape[1] * image.shape[2]),
        Width=int(image.shape[1]),
        Height=int(image.shape[0]),
        Depth=4,
        Flags=0,
    )
    metadata = {
        "input_shape": [int(dim) for dim in image.shape],
        "input_width": int(image.shape[1]),
        "input_height": int(image.shape[0]),
        "input_depth": int(image.shape[2]),
    }
    return [(image.tobytes(), header)], metadata


def make_output_headers() -> list[Any]:
    return [
        SimpleNamespace(
            Type=0,
            Index=0,
            Size=0,
            Width=0,
            Height=0,
            Depth=0,
            Flags=0,
        )
    ]


def run_desktop_forward(
    load_model: ModuleType,
    packet: list[tuple[bytes, Any]],
    *,
    device: str,
) -> tuple[Any, list[Any]]:
    transformed = load_model.TransformInput(packet)
    headers = make_output_headers()
    with torch.inference_mode():
        outputs = load_model.RunModel(transformed, headers, device)
    return outputs, headers


def run_model_only(
    load_model: ModuleType,
    transformed_image: Any,
    *,
    device: str,
) -> tuple[Any, list[Any]]:
    headers = make_output_headers()
    with torch.inference_mode():
        outputs = load_model.RunModel(transformed_image, headers, device)
    return outputs, headers


def summarize_outputs(outputs: Any, headers: list[Any]) -> dict[str, Any]:
    if not outputs:
        return {"output_present": False}

    first_output = outputs[0][0]
    first_header = headers[0]
    summary = {
        "output_present": True,
        "output_shape": [int(dim) for dim in first_output.shape],
        "output_dtype": str(first_output.dtype),
        "output_min": float(first_output.min()),
        "output_max": float(first_output.max()),
        "output_mean": float(first_output.mean()),
        "header_type": int(first_header.Type),
        "header_size": int(first_header.Size),
        "header_width": int(first_header.Width),
        "header_height": int(first_header.Height),
        "header_depth": int(first_header.Depth),
    }
    return summary


def benchmark_callable(
    repeats: int,
    fn,
) -> tuple[list[float], Any]:
    durations: list[float] = []
    last_result = None
    for _ in range(repeats):
        start = time.perf_counter()
        last_result = fn()
        durations.append(time.perf_counter() - start)
    return durations, last_result


def default_output_path(device: str, size: int) -> Path:
    return REPO_ROOT / "comparison" / (
        f"benchmark_depth_anything_desktop_forward_{device}_{size}_20260422.json"
    )


def resolve_default_image(depth_anything_root: Path) -> Path:
    return depth_anything_root / "assets" / "examples" / "demo01.jpg"


def resolve_default_image_dir(depth_anything_root: Path) -> Path:
    return depth_anything_root / "assets" / "examples"


def list_corpus_images(image_dir: Path, limit: int | None) -> list[Path]:
    images = sorted(path for path in image_dir.glob("*.jpg") if path.is_file())
    if limit is not None:
        images = images[:limit]
    return images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the exact Deep Desktop Depth-Anything forward path."
    )
    parser.add_argument(
        "--install-root",
        default=str(DEFAULT_INSTALL_ROOT),
        help="DepthExtractor installation root containing Depth-Anything and Python310.",
    )
    parser.add_argument("--device", default="vulkan")
    parser.add_argument("--encoder", default="vits")
    parser.add_argument("--size", type=int, default=280)
    parser.add_argument(
        "--half-precision",
        choices=["YES", "NO"],
        default="NO",
        help="Use the installed LoadModel half-precision toggle exactly as implemented.",
    )
    parser.add_argument("--image", help="Single image for repeated timing.")
    parser.add_argument(
        "--image-dir",
        help="Image directory for corpus timing. Defaults to the installed examples dir.",
    )
    parser.add_argument("--corpus-limit", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", help="Optional JSON output path.")
    args = parser.parse_args()

    install_root = Path(args.install_root).resolve()
    depth_anything_root = resolve_depth_anything_root(install_root)
    image_path = (
        Path(args.image).resolve()
        if args.image
        else resolve_default_image(depth_anything_root).resolve()
    )
    image_dir = (
        Path(args.image_dir).resolve()
        if args.image_dir
        else resolve_default_image_dir(depth_anything_root).resolve()
    )
    corpus_image_paths = list_corpus_images(image_dir, args.corpus_limit)
    if not image_path.exists():
        raise FileNotFoundError(f"Single image does not exist: {image_path}")
    if not corpus_image_paths:
        raise FileNotFoundError(f"No JPG images found in {image_dir}")

    load_model = load_installed_load_model(depth_anything_root)
    set_parameters(
        load_model,
        encoder=args.encoder,
        size=args.size,
        half_precision=args.half_precision,
    )

    single_packet, input_metadata = make_desktop_packet(image_path)
    corpus_packets = [make_desktop_packet(path)[0] for path in corpus_image_paths]

    if getattr(load_model, "LoadedModel", None) is not None:
        load_model.CleanModel()

    cold_load_s = load_model_on_device(
        load_model,
        depth_anything_root,
        device=args.device,
    )
    cold_forward_start = time.perf_counter()
    cold_outputs, cold_headers = run_desktop_forward(
        load_model,
        single_packet,
        device=args.device,
    )
    cold_first_forward_s = time.perf_counter() - cold_forward_start
    cold_total_s = cold_load_s + cold_first_forward_s
    cold_output_summary = summarize_outputs(cold_outputs, cold_headers)
    load_model.CleanModel()

    warm_load_s = load_model_on_device(
        load_model,
        depth_anything_root,
        device=args.device,
    )

    for _ in range(args.warmup):
        _ = run_desktop_forward(load_model, single_packet, device=args.device)

    transformed_once = load_model.TransformInput(single_packet)

    transform_durations, transformed_last = benchmark_callable(
        args.repeats,
        lambda: load_model.TransformInput(single_packet),
    )
    run_model_durations, run_model_last = benchmark_callable(
        args.repeats,
        lambda: run_model_only(load_model, transformed_once, device=args.device),
    )
    desktop_forward_durations, desktop_forward_last = benchmark_callable(
        args.repeats,
        lambda: run_desktop_forward(load_model, single_packet, device=args.device),
    )

    corpus_forward_durations: list[float] = []
    corpus_output_summary = None
    for packet in corpus_packets:
        start = time.perf_counter()
        outputs, headers = run_desktop_forward(load_model, packet, device=args.device)
        corpus_forward_durations.append(time.perf_counter() - start)
        if corpus_output_summary is None:
            corpus_output_summary = summarize_outputs(outputs, headers)

    if corpus_output_summary is None:
        corpus_output_summary = {}

    current_parameters = load_model.GetCurrentParameters()
    output_path = (
        Path(args.out).resolve()
        if args.out
        else default_output_path(args.device, args.size).resolve()
    )
    result = {
        "benchmark_name": "benchmark_depth_anything_desktop_forward",
        "benchmark_contract": "exact_loadmodel_transforminput_plus_runmodel",
        "notes": [
            "Uses the installed Deep Desktop LoadModel.py path directly.",
            "Measures TransformInput plus RunModel, which includes the current forward path, normalization, and CPU readback.",
            "Does not use infer_image, compiled-session bridges, fixed-shape replay helpers, or output-copy skipping.",
            "Runs the exact forward path under torch.inference_mode() so Vulkan inference is benchmarkable on the current packaged build.",
        ],
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_file": getattr(torch, "__file__", None),
        "torch_vulkan_available": bool(
            getattr(torch, "is_vulkan_available", lambda: False)()
        ),
        "inference_mode_enabled": True,
        "install_root": str(install_root),
        "depth_anything_root": str(depth_anything_root),
        "load_model_path": str(depth_anything_root / "LoadModel.py"),
        "launcher_python": str(install_root / "Python310" / "python.exe"),
        "device": args.device,
        "encoder": args.encoder,
        "size": args.size,
        "half_precision": args.half_precision,
        "current_parameters": current_parameters,
        "single_image": str(image_path),
        "image_dir": str(image_dir),
        "image_count": len(corpus_image_paths),
        "input_metadata": input_metadata,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cold_model_load_s": cold_load_s,
        "cold_first_forward_s": cold_first_forward_s,
        "cold_load_plus_first_forward_s": cold_total_s,
        "warm_model_load_s": warm_load_s,
        "single_image_transform_input": summarize_durations(
            "single_image_transform_input",
            transform_durations,
        ),
        "single_image_run_model_only": summarize_durations(
            "single_image_run_model_only",
            run_model_durations,
        ),
        "single_image_desktop_forward_path": summarize_durations(
            "single_image_desktop_forward_path",
            desktop_forward_durations,
        ),
        "full_corpus_desktop_forward_path": summarize_durations(
            "full_corpus_desktop_forward_path",
            corpus_forward_durations,
        ),
        "single_image_output": summarize_outputs(
            desktop_forward_last[0],
            desktop_forward_last[1],
        ),
        "run_model_only_output": summarize_outputs(
            run_model_last[0],
            run_model_last[1],
        ),
        "transform_input_last_shape": (
            [int(dim) for dim in transformed_last.shape]
            if transformed_last is not None
            else None
        ),
        "cold_output": cold_output_summary,
        "corpus_output": corpus_output_summary,
    }
    write_json(output_path, result)
    print(output_path.read_text(encoding="utf-8"))

    load_model.CleanModel()


if __name__ == "__main__":
    main()
