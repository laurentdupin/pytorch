from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bench_common import suppress_windows_error_dialogs, windows_subprocess_kwargs

suppress_windows_error_dialogs()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def default_image_for_root(depth_extractor_root: Path) -> Path:
    return (
        depth_extractor_root
        / "Depth-Anything-V2"
        / "Depth-Anything"
        / "assets"
        / "examples"
        / "demo01.jpg"
    )


def find_install_python(install_root: Path) -> Path | None:
    for python_dir in sorted(install_root.glob("Python*")):
        python = python_dir / "python.exe"
        if python.exists():
            return python
    return None


def find_load_model(install_root: Path) -> Path | None:
    matches = sorted(install_root.rglob("LoadModel.py"))
    return matches[0] if matches else None


def probe_install(install_root: Path) -> dict[str, Any]:
    python = find_install_python(install_root)
    load_model = find_load_model(install_root)
    payload: dict[str, Any] = {
        "name": install_root.name,
        "install_root": str(install_root),
        "python": str(python) if python else None,
        "load_model": str(load_model) if load_model else None,
        "model_root": str(load_model.parent) if load_model else None,
    }
    if python is None:
        payload["skip_reason"] = "missing_python"
        return payload
    code = """
import json, sys
out = {"python": sys.executable}
try:
    import torch
    out["torch_version"] = torch.__version__
    out["torch_file"] = torch.__file__
    out["vulkan_available"] = bool(getattr(torch, "is_vulkan_available", lambda: False)())
    if hasattr(torch, "vulkan"):
        try:
            out["vulkan_device_count"] = torch.vulkan.device_count()
            out["vulkan_devices"] = [
                torch.vulkan.get_device_name(i)
                for i in range(torch.vulkan.device_count())
            ]
        except Exception as exc:
            out["vulkan_probe_error"] = repr(exc)
except Exception as exc:
    out["torch_error"] = repr(exc)
print(json.dumps(out))
"""
    completed = subprocess.run(
        [str(python), "-c", code],
        text=True,
        capture_output=True,
        timeout=60,
        **windows_subprocess_kwargs(),
    )
    payload["probe_returncode"] = completed.returncode
    payload["probe_stderr"] = completed.stderr.strip()
    try:
        payload.update(json.loads(completed.stdout))
    except json.JSONDecodeError:
        payload["probe_stdout"] = completed.stdout.strip()
    return payload


def make_packet(image_path: Path) -> tuple[list[tuple[bytes, Any]], dict[str, Any]]:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif image.shape[2] != 4:
        raise RuntimeError(f"Unsupported image shape: {image.shape!r}")

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
        "image": str(image_path),
        "shape": [int(v) for v in image.shape],
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "depth": int(image.shape[2]),
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


def load_python_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_app_device(model_root: Path, device_index: int) -> Any:
    launch_path = model_root / "Launch.py"
    if launch_path.exists():
        launch = load_python_module(launch_path, "depth_extractor_launch")
        devices = launch.RegisterTorchDevices()
        key = f"VULKAN{device_index}"
        if key in devices:
            return devices[key]
    return f"vulkan:{device_index}"


def set_safe_parameters(load_model: Any) -> dict[str, Any]:
    current: dict[str, Any] = {}
    params: dict[str, Any] = {}
    try:
        params = load_model.GetModelParameters()
    except Exception:
        params = {}
    try:
        current = load_model.GetCurrentParameters()
    except Exception:
        current = {}

    for key, spec in params.items():
        options = spec.get("options", []) if isinstance(spec, dict) else []
        if not options:
            continue
        value = current.get(key)
        if value not in options:
            value = options[0]
        if key == "HalfPrecision" and "NO" in options:
            value = "NO"
        if key == "Size":
            if "280" in options:
                value = "280"
            elif "384" in options:
                value = "384"
        try:
            load_model.SetParameter(key, value)
        except Exception:
            pass

    try:
        return load_model.GetCurrentParameters()
    except Exception:
        return current


def summarize_outputs(outputs: Any, headers: list[Any]) -> dict[str, Any]:
    import numpy as np

    if not outputs:
        return {"output_present": False}
    first_output = np.asarray(outputs[0][0])
    first_header = headers[0]
    return {
        "output_present": True,
        "shape": [int(v) for v in first_output.shape],
        "dtype": str(first_output.dtype),
        "finite": bool(np.isfinite(first_output).all()),
        "min": float(np.min(first_output)),
        "max": float(np.max(first_output)),
        "mean": float(np.mean(first_output)),
        "header_type": int(first_header.Type),
        "header_width": int(first_header.Width),
        "header_height": int(first_header.Height),
        "header_depth": int(first_header.Depth),
    }


def worker(args: argparse.Namespace) -> None:
    suppress_windows_error_dialogs()
    if not args.install_root or not args.load_model or not args.image or not args.out:
        raise SystemExit(
            "--worker requires --install-root, --load-model, --image, and --out"
        )
    install_root = Path(args.install_root).resolve()
    load_model_path = Path(args.load_model).resolve()
    model_root = load_model_path.parent
    os.chdir(model_root)
    sys.path.insert(0, str(model_root))
    os.environ.setdefault("TORCH_HOME", str(model_root / "torchcache"))
    os.environ.setdefault("HF_HOME", str(model_root / "hfcache"))

    result: dict[str, Any] = {
        "name": install_root.name,
        "install_root": str(install_root),
        "model_root": str(model_root),
        "device_index": args.device_index,
        "status": "unknown",
    }
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["torch_file"] = torch.__file__
        result["vulkan_available"] = bool(torch.is_vulkan_available())
        if hasattr(torch, "vulkan"):
            result["vulkan_device_count"] = torch.vulkan.device_count()
            result["vulkan_device_name"] = torch.vulkan.get_device_name(
                args.device_index
            )

        load_model = load_python_module(load_model_path, "depth_extractor_load_model")
        result["parameters"] = set_safe_parameters(load_model)
        device = resolve_app_device(model_root, args.device_index)
        result["device"] = str(device)

        packet, input_metadata = make_packet(Path(args.image).resolve())
        result["input"] = input_metadata

        start = time.perf_counter()
        load_model.LoadModel(device)
        result["load_s"] = time.perf_counter() - start

        start = time.perf_counter()
        transformed = load_model.TransformInput(packet)
        result["transform_s"] = time.perf_counter() - start

        headers = make_output_headers()
        start = time.perf_counter()
        outputs = load_model.RunModel(transformed, headers, device)
        result["run_model_s"] = time.perf_counter() - start
        result["output"] = summarize_outputs(outputs, headers)
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        write_json(Path(args.out), result)
        print(json.dumps(result, indent=2, default=str))


def run_parent(args: argparse.Namespace) -> None:
    suppress_windows_error_dialogs()
    if not args.depth_extractor_root:
        raise SystemExit("--depth-extractor-root is required unless --worker is set")
    root = Path(args.depth_extractor_root).resolve()
    image = (
        Path(args.image).resolve()
        if args.image
        else default_image_for_root(root).resolve()
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    devices = [int(item) for item in args.devices.split(",") if item.strip()]
    installs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if args.models:
        selected = {item.strip() for item in args.models.split(",") if item.strip()}
        installs = [p for p in installs if p.name in selected]

    inventory = [probe_install(install) for install in installs]
    write_json(output_dir / "inventory.json", inventory)

    results: list[dict[str, Any]] = []
    for item in inventory:
        if item.get("skip_reason"):
            results.append({**item, "status": "skipped"})
            continue
        if not item.get("vulkan_available"):
            results.append({**item, "status": "skipped", "skip_reason": "no_vulkan"})
            continue
        python = item.get("python")
        load_model = item.get("load_model")
        if not python or not load_model:
            results.append({**item, "status": "skipped", "skip_reason": "missing_contract"})
            continue
        for device_index in devices:
            out = output_dir / f"{item['name']}.vulkan{device_index}.json"
            command = [
                python,
                str(Path(__file__).resolve()),
                "--worker",
                "--install-root",
                item["install_root"],
                "--load-model",
                load_model,
                "--device-index",
                str(device_index),
                "--image",
                str(image),
                "--out",
                str(out),
            ]
            try:
                completed = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    timeout=args.timeout_s,
                    **windows_subprocess_kwargs(),
                )
                if out.exists():
                    result = json.loads(out.read_text(encoding="utf-8"))
                else:
                    result = {
                        "name": item["name"],
                        "device_index": device_index,
                        "status": "error",
                        "error_type": "MissingOutput",
                    }
                result["worker_returncode"] = completed.returncode
                result["worker_stderr_tail"] = completed.stderr[-4000:]
            except subprocess.TimeoutExpired as exc:
                result = {
                    "name": item["name"],
                    "device_index": device_index,
                    "status": "timeout",
                    "timeout_s": args.timeout_s,
                    "stdout_tail": (exc.stdout or "")[-4000:],
                    "stderr_tail": (exc.stderr or "")[-4000:],
                }
            results.append(result)
            write_json(output_dir / "summary.json", results)

    write_json(output_dir / "summary.json", results)
    print(json.dumps(results, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--depth-extractor-root",
        help="DepthExtractor install root. Required unless --worker is set.",
    )
    parser.add_argument("--models", help="Comma-separated install directory names.")
    parser.add_argument("--devices", default="0,1,2")
    parser.add_argument(
        "--image",
        help=(
            "Input image. Defaults to the Depth-Anything-V2 demo image under "
            "--depth-extractor-root."
        ),
    )
    parser.add_argument("--output-dir", default="comparison/depth_extractor_model_smoke_20260424")
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--install-root")
    parser.add_argument("--load-model")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
