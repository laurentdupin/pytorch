from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bench_common import suppress_windows_error_dialogs, windows_subprocess_kwargs

suppress_windows_error_dialogs()


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "comparison" / "depth_anything_desktop_output_compare_20260422"


@dataclass(frozen=True)
class InstallSpec:
    name: str
    root: Path
    python_path: Path
    device_key: str


def default_installs(depth_extractor_root: Path) -> tuple[InstallSpec, ...]:
    return (
        InstallSpec(
            name="vulkan",
            root=depth_extractor_root / "Depth-Anything-V2",
            python_path=depth_extractor_root
            / "Depth-Anything-V2"
            / "Python310"
            / "python.exe",
            device_key="VULKAN",
        ),
        InstallSpec(
            name="directml",
            root=depth_extractor_root / "Depth-Anything-V22",
            python_path=depth_extractor_root
            / "Depth-Anything-V22"
            / "Python310"
            / "python.exe",
            device_key="DIRECT_ML0",
        ),
        InstallSpec(
            name="rocm",
            root=depth_extractor_root / "Depth-Anything-V23",
            python_path=depth_extractor_root
            / "Depth-Anything-V23"
            / "Python312"
            / "python.exe",
            device_key="CUDA0",
        ),
        InstallSpec(
            name="cuda",
            root=depth_extractor_root / "Depth-Anything-V24",
            python_path=depth_extractor_root
            / "Depth-Anything-V24"
            / "Python310"
            / "python.exe",
            device_key="CUDA0",
        ),
    )


def default_image_dir(depth_extractor_root: Path) -> Path:
    return (
        depth_extractor_root
        / "Depth-Anything-V2"
        / "Depth-Anything"
        / "assets"
        / "examples"
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _emit_output(
    *,
    install_root: Path,
    device_key: str,
    image_path: Path,
    out_npy: Path,
    out_json: Path,
    encoder: str,
    size: int,
    half_precision: str,
) -> None:
    import cv2

    depth_root = install_root / "Depth-Anything"
    os.chdir(depth_root)
    sys.path.insert(0, str(depth_root))

    import Launch  # type: ignore[import-not-found]
    import LoadModel  # type: ignore[import-not-found]

    devices = Launch.RegisterTorchDevices()
    if device_key not in devices:
        raise RuntimeError(
            f"Requested device key {device_key!r} is not available in {install_root}. "
            f"Available keys: {sorted(devices.keys())}"
        )
    device = devices[device_key]

    LoadModel.SetParameter("Encoder", encoder)
    LoadModel.SetParameter("Size", str(size))
    LoadModel.SetParameter("HalfPrecision", half_precision)
    LoadModel.LoadModel(device)

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    if image.ndim != 3:
        raise RuntimeError(f"Expected HWC image, got shape {tuple(image.shape)}")
    if image.shape[2] == 3:
        alpha = np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)
        image = np.concatenate((image, alpha), axis=2)
    if image.shape[2] != 4:
        raise RuntimeError(f"Expected 4-channel image packet, got shape {tuple(image.shape)}")

    packet = [
        (
            image.tobytes(),
            SimpleNamespace(
                Type=1,
                Index=0,
                Size=int(image.size),
                Width=int(image.shape[1]),
                Height=int(image.shape[0]),
                Depth=4,
                Flags=0,
            ),
        )
    ]

    transformed = LoadModel.TransformInput(packet)
    headers = [
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
    outputs = LoadModel.RunModel(transformed, headers, device)
    depth = np.asarray(outputs[0][0], dtype=np.float32)

    _ensure_parent(out_npy)
    _ensure_parent(out_json)
    np.save(out_npy, depth)

    payload = {
        "install_root": str(install_root),
        "depth_root": str(depth_root),
        "image_path": str(image_path),
        "device_key": device_key,
        "device_repr": str(device),
        "encoder": encoder,
        "size": size,
        "half_precision": half_precision,
        "output_shape": list(depth.shape),
        "output_dtype": str(depth.dtype),
        "output_min": float(np.min(depth)),
        "output_max": float(np.max(depth)),
        "output_mean": float(np.mean(depth)),
        "output_std": float(np.std(depth)),
        "finite": bool(np.isfinite(depth).all()),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_install_output(
    spec: InstallSpec,
    image_path: Path,
    output_dir: Path,
    *,
    encoder: str,
    size: int,
    half_precision: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    out_npy = output_dir / f"{stem}.{spec.name}.npy"
    out_json = output_dir / f"{stem}.{spec.name}.json"

    command = [
        str(spec.python_path),
        str(Path(__file__).resolve()),
        "--emit-output",
        "--install-root",
        str(spec.root),
        "--device-key",
        spec.device_key,
        "--image",
        str(image_path),
        "--out-npy",
        str(out_npy),
        "--out-json",
        str(out_json),
        "--encoder",
        encoder,
        "--size",
        str(size),
        "--half-precision",
        half_precision,
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        **windows_subprocess_kwargs(),
    )
    metadata = json.loads(out_json.read_text(encoding="utf-8"))
    metadata["stdout"] = completed.stdout
    metadata["stderr"] = completed.stderr
    metadata["output_npy"] = str(out_npy)
    metadata["output_json"] = str(out_json)
    return metadata


def _compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    if ref.shape != cand.shape:
        raise RuntimeError(f"Shape mismatch: {ref.shape} vs {cand.shape}")

    diff = cand - ref
    ref_flat = ref.reshape(-1)
    cand_flat = cand.reshape(-1)
    finite = np.isfinite(ref_flat) & np.isfinite(cand_flat)
    if finite.any():
        corr = float(np.corrcoef(ref_flat[finite], cand_flat[finite])[0, 1])
    else:
        corr = math.nan

    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "max_abs": float(np.max(np.abs(diff))),
        "pearson": corr,
        "candidate_min": float(np.min(cand)),
        "candidate_max": float(np.max(cand)),
        "candidate_mean": float(np.mean(cand)),
        "candidate_std": float(np.std(cand)),
    }


def _run_compare(args: argparse.Namespace) -> None:
    if not args.depth_extractor_root:
        raise SystemExit("--depth-extractor-root is required unless --emit-output is set")
    depth_extractor_root = Path(args.depth_extractor_root).resolve()
    images: list[Path]
    if args.image:
        images = [Path(args.image).resolve()]
    else:
        image_dir = default_image_dir(depth_extractor_root)
        images = sorted(image_dir.glob("*.jpg"))[: args.image_count]

    if not images:
        raise RuntimeError("No images selected for comparison.")

    specs = default_installs(depth_extractor_root)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    installs_payload: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}

    for image_path in images:
        stem = image_path.stem
        installs_payload[stem] = {}
        for spec in specs:
            installs_payload[stem][spec.name] = _run_install_output(
                spec,
                image_path.resolve(),
                output_dir,
                encoder=args.encoder,
                size=args.size,
                half_precision=args.half_precision,
            )

        arrays = {
            spec.name: np.load(output_dir / f"{stem}.{spec.name}.npy")
            for spec in specs
        }

        image_compare: dict[str, Any] = {}
        for ref_name in arrays:
            image_compare[ref_name] = {}
            for cand_name in arrays:
                if ref_name == cand_name:
                    continue
                image_compare[ref_name][cand_name] = _compare_arrays(
                    arrays[ref_name], arrays[cand_name]
                )
        comparisons[stem] = image_compare

    summary = {
        "images": [str(path) for path in images],
        "encoder": args.encoder,
        "size": args.size,
        "half_precision": args.half_precision,
        "installs": installs_payload,
        "comparisons": comparisons,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary_path={summary_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-output", action="store_true")
    parser.add_argument(
        "--depth-extractor-root",
        help="DepthExtractor install root. Required unless --emit-output is set.",
    )
    parser.add_argument("--install-root")
    parser.add_argument("--device-key")
    parser.add_argument("--image")
    parser.add_argument("--out-npy")
    parser.add_argument("--out-json")
    parser.add_argument("--encoder", default="vits")
    parser.add_argument("--size", type=int, default=280)
    parser.add_argument("--half-precision", default="NO")
    parser.add_argument("--image-count", type=int, default=3)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    suppress_windows_error_dialogs()
    args = _parse_args()
    if args.emit_output:
        _emit_output(
            install_root=Path(args.install_root).resolve(),
            device_key=args.device_key,
            image_path=Path(args.image).resolve(),
            out_npy=Path(args.out_npy).resolve(),
            out_json=Path(args.out_json).resolve(),
            encoder=args.encoder,
            size=args.size,
            half_precision=args.half_precision,
        )
        return
    _run_compare(args)


if __name__ == "__main__":
    main()
