from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bench_common import summarize_durations, synchronize_payload, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "comparison" / "depth_anything_desktop_profile_compare_20260422"


@dataclass(frozen=True)
class InstallSpec:
    name: str
    root: Path
    python_path: Path
    device_key: str
    sync_kind: str


DEFAULT_INSTALLS = (
    InstallSpec(
        name="vulkan",
        root=Path(r"C:\Users\REDACTED\AppData\Local\DepthExtractor\Depth-Anything-V2"),
        python_path=Path(r"C:\Users\REDACTED\AppData\Local\DepthExtractor\Depth-Anything-V2\Python310\python.exe"),
        device_key="VULKAN",
        sync_kind="vulkan",
    ),
    InstallSpec(
        name="directml",
        root=Path(r"C:\Users\REDACTED\AppData\Local\DepthExtractor\Depth-Anything-V22"),
        python_path=Path(r"C:\Users\REDACTED\AppData\Local\DepthExtractor\Depth-Anything-V22\Python310\python.exe"),
        device_key="DIRECT_ML0",
        sync_kind="directml",
    ),
    InstallSpec(
        name="rocm",
        root=Path(r"C:\Users\REDACTED\AppData\Local\DepthExtractor\Depth-Anything-V23"),
        python_path=Path(r"C:\Users\REDACTED\AppData\Local\DepthExtractor\Depth-Anything-V23\Python312\python.exe"),
        device_key="CUDA0",
        sync_kind="cuda",
    ),
    InstallSpec(
        name="cuda",
        root=Path(r"C:\Users\REDACTED\AppData\Local\DepthExtractor\Depth-Anything-V24"),
        python_path=Path(r"C:\Users\REDACTED\AppData\Local\DepthExtractor\Depth-Anything-V24\Python310\python.exe"),
        device_key="CUDA0",
        sync_kind="cuda",
    ),
)


class Recorder:
    def __init__(self) -> None:
        self._durations: dict[str, list[float]] = {}

    def add(self, name: str, duration_s: float) -> None:
        self._durations.setdefault(name, []).append(float(duration_s))

    def summary(self) -> dict[str, Any]:
        return {
            name: summarize_durations(name, durations)
            for name, durations in sorted(self._durations.items())
        }


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


def strip_repo_torch_shadow_paths() -> None:
    repo_root = REPO_ROOT.resolve()
    filtered = []
    for entry in sys.path:
        try:
            resolved = Path(entry or ".").resolve()
        except OSError:
            filtered.append(entry)
            continue
        if resolved == repo_root:
            continue
        filtered.append(entry)
    sys.path[:] = filtered


def load_installed_module(depth_anything_root: Path, relative_path: str, module_name: str) -> ModuleType:
    if str(depth_anything_root) not in sys.path:
        sys.path.insert(0, str(depth_anything_root))

    module_path = depth_anything_root / relative_path
    if not module_path.exists():
        raise FileNotFoundError(f"Module path does not exist: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with pushd(depth_anything_root):
        spec.loader.exec_module(module)
    return module


def load_depth_anything_modules(depth_anything_root: Path) -> tuple[ModuleType, ModuleType]:
    os.environ.setdefault("TORCH_HOME", str(depth_anything_root / "torchcache"))
    os.environ.setdefault("HF_HOME", str(depth_anything_root / "hfcache"))

    launch = load_installed_module(
        depth_anything_root,
        "Launch.py",
        "desktop_depth_anything_launch",
    )
    load_model = load_installed_module(
        depth_anything_root,
        "LoadModel.py",
        "desktop_depth_anything_load_model",
    )
    return launch, load_model


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


def make_desktop_packet(image_path: Path) -> tuple[list[tuple[bytes, Any]], Any]:
    import cv2
    from types import SimpleNamespace

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif image.shape[2] != 4:
        raise RuntimeError(f"Unsupported image shape for desktop packet: {image.shape!r}")

    header = SimpleNamespace(
        Type=1,
        Index=0,
        Size=int(image.shape[0] * image.shape[1] * image.shape[2]),
        Width=int(image.shape[1]),
        Height=int(image.shape[0]),
        Depth=4,
        Flags=0,
    )
    return [(image.tobytes(), header)], image


def summarize_output(depth: Any) -> dict[str, Any]:
    return {
        "output_shape": [int(dim) for dim in depth.shape],
        "output_dtype": str(depth.dtype),
        "output_min": float(depth.min()),
        "output_max": float(depth.max()),
        "output_mean": float(depth.mean()),
        "output_std": float(depth.std()),
    }


def resolve_device(launch: ModuleType, device_key: str) -> Any:
    devices = launch.RegisterTorchDevices()
    if device_key not in devices:
        raise RuntimeError(
            f"Requested device key {device_key!r} is not available. "
            f"Available keys: {sorted(devices.keys())}"
        )
    return devices[device_key]


@contextmanager
def patch_call(
    target: Any,
    attr_name: str,
    label: str,
    recorder: Recorder,
    torch_module: Any,
    sync_kind: str,
    device: Any,
):
    original = getattr(target, attr_name)

    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        synchronize_payload(torch_module, sync_kind, result, device)
        recorder.add(label, time.perf_counter() - start)
        return result

    setattr(target, attr_name, wrapped)
    try:
        yield
    finally:
        setattr(target, attr_name, original)


@contextmanager
def instrument_model(
    model: Any,
    recorder: Recorder,
    torch_module: Any,
    sync_kind: str,
    device: Any,
):
    with ExitStack() as stack:
        backbone = getattr(model, "pretrained", None)
        if backbone is not None:
            for module_name in ("patch_embed", "norm", "head"):
                if hasattr(backbone, module_name):
                    module = getattr(backbone, module_name)
                    if hasattr(module, "forward"):
                        stack.enter_context(
                            patch_call(
                                module,
                                "forward",
                                f"backbone.{module_name}",
                                recorder,
                                torch_module,
                                sync_kind,
                                device,
                            )
                        )
            if hasattr(backbone, "get_intermediate_layers"):
                stack.enter_context(
                    patch_call(
                        backbone,
                        "get_intermediate_layers",
                        "backbone.total",
                        recorder,
                        torch_module,
                        sync_kind,
                        device,
                    )
                )
            if hasattr(backbone, "prepare_tokens_with_masks"):
                stack.enter_context(
                    patch_call(
                        backbone,
                        "prepare_tokens_with_masks",
                        "backbone.prepare_tokens",
                        recorder,
                        torch_module,
                        sync_kind,
                        device,
                    )
                )
            blocks = getattr(backbone, "blocks", [])
            for index, block in enumerate(blocks):
                if hasattr(block, "forward"):
                    stack.enter_context(
                        patch_call(
                            block,
                            "forward",
                            f"backbone.block.{index}",
                            recorder,
                            torch_module,
                            sync_kind,
                            device,
                        )
                    )
                for module_name in ("norm1", "attn", "norm2", "mlp"):
                    if hasattr(block, module_name):
                        module = getattr(block, module_name)
                        if hasattr(module, "forward"):
                            stack.enter_context(
                                patch_call(
                                    module,
                                    "forward",
                                    f"backbone.block.{index}.{module_name}",
                                    recorder,
                                    torch_module,
                                    sync_kind,
                                    device,
                                )
                            )
                if hasattr(block, "attn"):
                    attn_module = getattr(block, "attn")
                    for module_name in ("qkv", "proj"):
                        if hasattr(attn_module, module_name):
                            module = getattr(attn_module, module_name)
                            if hasattr(module, "forward"):
                                stack.enter_context(
                                    patch_call(
                                        module,
                                        "forward",
                                        f"backbone.block.{index}.attn.{module_name}",
                                        recorder,
                                        torch_module,
                                        sync_kind,
                                        device,
                                    )
                                )

        depth_head = getattr(model, "depth_head", None)
        if depth_head is not None:
            if hasattr(depth_head, "forward"):
                stack.enter_context(
                    patch_call(
                        depth_head,
                        "forward",
                        "decoder.total",
                        recorder,
                        torch_module,
                        sync_kind,
                        device,
                    )
                )
            for index, module in enumerate(getattr(depth_head, "projects", [])):
                if hasattr(module, "forward"):
                    stack.enter_context(
                        patch_call(
                            module,
                            "forward",
                            f"decoder.project.{index}",
                            recorder,
                            torch_module,
                            sync_kind,
                            device,
                        )
                    )
            for index, module in enumerate(getattr(depth_head, "resize_layers", [])):
                if hasattr(module, "forward"):
                    stack.enter_context(
                        patch_call(
                            module,
                            "forward",
                            f"decoder.resize.{index}",
                            recorder,
                            torch_module,
                            sync_kind,
                            device,
                        )
                    )
            scratch = getattr(depth_head, "scratch", None)
            if scratch is not None:
                for index in range(1, 5):
                    layer_name = f"layer{index}_rn"
                    if hasattr(scratch, layer_name):
                        stack.enter_context(
                            patch_call(
                                getattr(scratch, layer_name),
                                "forward",
                                f"decoder.{layer_name}",
                                recorder,
                                torch_module,
                                sync_kind,
                                device,
                            )
                        )
                for index in range(1, 5):
                    layer_name = f"refinenet{index}"
                    if hasattr(scratch, layer_name):
                        stack.enter_context(
                            patch_call(
                                getattr(scratch, layer_name),
                                "forward",
                                f"decoder.{layer_name}",
                                recorder,
                                torch_module,
                                sync_kind,
                                device,
                            )
                        )
                for layer_name in ("output_conv1", "output_conv2"):
                    if hasattr(scratch, layer_name):
                        stack.enter_context(
                            patch_call(
                                getattr(scratch, layer_name),
                                "forward",
                                f"decoder.{layer_name}",
                                recorder,
                                torch_module,
                                sync_kind,
                                device,
                            )
                        )
        yield


def run_profile_once(
    *,
    load_model: ModuleType,
    packet: list[tuple[bytes, Any]],
    device: Any,
    torch_module: Any,
    sync_kind: str,
    top_level: Recorder | None,
    modules: Recorder | None,
) -> Any:
    import torch.nn.functional as F

    record_top = top_level is not None
    record_modules = modules is not None

    overall_start = time.perf_counter()

    start = time.perf_counter()
    transformed = load_model.TransformInput(packet)
    if record_top:
        top_level.add("desktop.transform_input", time.perf_counter() - start)

    width, height = load_model.transform.transforms[0].get_size(
        transformed.shape[1], transformed.shape[0]
    )

    start = time.perf_counter()
    mean = torch_module.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch_module.tensor([0.229, 0.224, 0.225]).to(device)
    image = torch_module.from_numpy(transformed).to(device)
    synchronize_payload(torch_module, sync_kind, image, device)
    if record_top:
        top_level.add("run_model.to_device", time.perf_counter() - start)

    start = time.perf_counter()
    image = image.unsqueeze(0)
    image = image.permute((0, 3, 1, 2))
    synchronize_payload(torch_module, sync_kind, image, device)
    if record_top:
        top_level.add("run_model.layout_nchw", time.perf_counter() - start)

    start = time.perf_counter()
    image = F.interpolate(image, (width, height))
    synchronize_payload(torch_module, sync_kind, image, device)
    if record_top:
        top_level.add("run_model.resize", time.perf_counter() - start)

    start = time.perf_counter()
    image = image.permute((0, 2, 3, 1))
    image = image[0]
    image = image.float()
    image = image / 255.0
    image = (image - mean) / std
    image = image.permute((2, 0, 1))
    image = image.unsqueeze(0)
    if load_model.parameters["HalfPrecision"] == "YES":
        image = image.half()
    synchronize_payload(torch_module, sync_kind, image, device)
    if record_top:
        top_level.add("run_model.normalize", time.perf_counter() - start)

    model = load_model.LoadedModel
    start = time.perf_counter()
    if record_modules:
        with instrument_model(model, modules, torch_module, sync_kind, device):
            depth = model.forward(image)
    else:
        depth = model.forward(image)
    synchronize_payload(torch_module, sync_kind, depth, device)
    if record_top:
        top_level.add("run_model.model_forward", time.perf_counter() - start)

    start = time.perf_counter()
    depth = depth[0]
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    synchronize_payload(torch_module, sync_kind, depth, device)
    if record_top:
        top_level.add("run_model.depth_normalize", time.perf_counter() - start)

    start = time.perf_counter()
    depth = depth.cpu()
    depth = depth.detach().numpy()
    if record_top:
        top_level.add("run_model.readback", time.perf_counter() - start)
        top_level.add("desktop.total", time.perf_counter() - overall_start)

    return depth


def profile_install(
    *,
    install_root: Path,
    device_key: str,
    backend_name: str,
    sync_kind: str,
    image_path: Path,
    out_path: Path,
    encoder: str,
    size: int,
    half_precision: str,
    warmup: int,
    repeats: int,
) -> None:
    strip_repo_torch_shadow_paths()
    import torch

    depth_anything_root = resolve_depth_anything_root(install_root)
    launch, load_model = load_depth_anything_modules(depth_anything_root)
    device = resolve_device(launch, device_key)

    set_parameters(
        load_model,
        encoder=encoder,
        size=size,
        half_precision=half_precision,
    )

    with pushd(depth_anything_root):
        load_model.LoadModel(device)

    packet, image = make_desktop_packet(image_path)

    for _ in range(warmup):
        _ = run_profile_once(
            load_model=load_model,
            packet=packet,
            device=device,
            torch_module=torch,
            sync_kind=sync_kind,
            top_level=None,
            modules=None,
        )

    top_level = Recorder()
    modules = Recorder()
    last_output = None
    for _ in range(repeats):
        last_output = run_profile_once(
            load_model=load_model,
            packet=packet,
            device=device,
            torch_module=torch,
            sync_kind=sync_kind,
            top_level=top_level,
            modules=modules,
        )

    payload = {
        "backend": backend_name,
        "sync_kind": sync_kind,
        "sync_note": (
            "DirectML uses scalar readback for stage synchronization."
            if sync_kind == "directml"
            else "Explicit device synchronization is used at stage boundaries."
        ),
        "install_root": str(install_root),
        "depth_anything_root": str(depth_anything_root),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_file": getattr(torch, "__file__", None),
        "device_key": device_key,
        "device_repr": str(device),
        "encoder": encoder,
        "size": size,
        "half_precision": half_precision,
        "image_path": str(image_path),
        "input_shape": [int(dim) for dim in image.shape],
        "warmup": warmup,
        "repeats": repeats,
        "top_level": top_level.summary(),
        "modules": modules.summary(),
        "output": summarize_output(last_output),
    }
    write_json(out_path, payload)


def run_install_profile(
    spec: InstallSpec,
    image_path: Path,
    out_path: Path,
    *,
    encoder: str,
    size: int,
    half_precision: str,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    if spec.name == "rocm":
        miopen_root = out_path.parent / "miopen"
        user_db = miopen_root / "user-db"
        cache_dir = miopen_root / "cache"
        user_db.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        env["MIOPEN_USER_DB_PATH"] = str(user_db)
        env["MIOPEN_CUSTOM_CACHE_DIR"] = str(cache_dir)

    command = [
        str(spec.python_path),
        str(Path(__file__).resolve()),
        "--profile-install",
        "--install-root",
        str(spec.root),
        "--device-key",
        spec.device_key,
        "--backend-name",
        spec.name,
        "--sync-kind",
        spec.sync_kind,
        "--image",
        str(image_path),
        "--out",
        str(out_path),
        "--encoder",
        encoder,
        "--size",
        str(size),
        "--half-precision",
        half_precision,
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
    ]
    subprocess.run(command, check=True, cwd=spec.root, env=env)
    return json.loads(out_path.read_text(encoding="utf-8"))


def build_summary(profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    top_level_names: set[str] = set()
    module_names: set[str] = set()
    for profile in profiles.values():
        top_level_names.update(profile.get("top_level", {}).keys())
        module_names.update(profile.get("modules", {}).keys())

    top_level_table = []
    for name in sorted(top_level_names):
        row = {"name": name}
        for backend, profile in profiles.items():
            row[backend] = float(profile.get("top_level", {}).get(name, {}).get("mean_s", 0.0))
        top_level_table.append(row)

    module_table = []
    for name in sorted(module_names):
        row = {"name": name}
        for backend, profile in profiles.items():
            row[backend] = float(profile.get("modules", {}).get(name, {}).get("mean_s", 0.0))
        module_table.append(row)

    top_level_table.sort(
        key=lambda row: max(float(value) for key, value in row.items() if key != "name"),
        reverse=True,
    )
    module_table.sort(
        key=lambda row: max(float(value) for key, value in row.items() if key != "name"),
        reverse=True,
    )

    return {
        "top_level_table": top_level_table,
        "module_table": module_table,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the exact installed Depth-Anything desktop path across backends."
    )
    parser.add_argument("--profile-install", action="store_true")
    parser.add_argument("--install-root")
    parser.add_argument("--device-key")
    parser.add_argument("--backend-name")
    parser.add_argument("--sync-kind")
    parser.add_argument("--image")
    parser.add_argument("--out")
    parser.add_argument("--encoder", default="vits")
    parser.add_argument("--size", type=int, default=280)
    parser.add_argument("--half-precision", choices=["YES", "NO"], default="NO")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile_install:
        profile_install(
            install_root=Path(args.install_root).resolve(),
            device_key=args.device_key,
            backend_name=args.backend_name,
            sync_kind=args.sync_kind,
            image_path=Path(args.image).resolve(),
            out_path=Path(args.out).resolve(),
            encoder=args.encoder,
            size=args.size,
            half_precision=args.half_precision,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        return

    image_path = (
        Path(r"C:\Users\REDACTED\Downloads\AIProspection\PytorchVulkan\Depth-Anything-V2\assets\examples\demo01.jpg")
        .resolve()
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles: dict[str, dict[str, Any]] = {}
    for spec in DEFAULT_INSTALLS:
        out_path = output_dir / f"profile_{spec.name}.json"
        profiles[spec.name] = run_install_profile(
            spec,
            image_path,
            out_path,
            encoder=args.encoder,
            size=args.size,
            half_precision=args.half_precision,
            warmup=args.warmup,
            repeats=args.repeats,
        )

    summary = {
        "image_path": str(image_path),
        "encoder": args.encoder,
        "size": args.size,
        "half_precision": args.half_precision,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "profiles": profiles,
        "comparison": build_summary(profiles),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
