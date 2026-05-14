from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import cv2

from bench_common import (
    REPO_ROOT,
    WORKSPACE_ROOT,
    enable_local_pytorch_repo_imports,
    summarize_durations,
    synchronize_result,
    write_json,
)
from depth_anything_common import (
    MODEL_CONFIGS,
    inference_context,
    resolve_depth_anything_checkpoint,
    resolve_depth_anything_repo,
    resolve_runtime_device,
)


OUTPUT_MODE_DEVICE_RESIDENT = "device_resident"
OUTPUT_MODE_READBACK = "readback"


def env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value not in {"0", "false", "FALSE"}


def parse_owner_limit() -> int | None:
    value = os.environ.get("PYTORCH_VULKAN_DAV2_BLOCK_OWNER_LIMIT", "1")
    if value == "all":
        return None
    try:
        return max(int(value), 0)
    except ValueError as exc:
        raise ValueError(
            "PYTORCH_VULKAN_DAV2_BLOCK_OWNER_LIMIT must be an integer or 'all'"
        ) from exc


def optional_bias(module: Any) -> Any:
    return getattr(module, "bias", None)


def optional_layerscale_gamma(module: Any) -> Any:
    gamma = getattr(module, "gamma", None)
    return gamma if gamma is not None else None


class VulkanDAv2BlockOwner:
    def __init__(self, torch_module: Any, block: Any, block_index: int) -> None:
        self.torch = torch_module
        self.block = block
        self.original_forward = block.forward
        self.block_index = block_index
        self.training = bool(getattr(block, "training", False))
        self.context = torch_module.ops.vulkan_prepack.create_vision_backbone_block_context(
            block.norm1.weight,
            block.norm1.bias,
            float(block.norm1.eps),
            block.attn.qkv.weight,
            optional_bias(block.attn.qkv),
            int(block.attn.num_heads),
            block.attn.proj.weight,
            optional_bias(block.attn.proj),
            optional_layerscale_gamma(block.ls1),
            block.norm2.weight,
            block.norm2.bias,
            float(block.norm2.eps),
            block.mlp.fc1.weight,
            optional_bias(block.mlp.fc1),
            block.mlp.fc2.weight,
            optional_bias(block.mlp.fc2),
            optional_layerscale_gamma(block.ls2),
            f"depth.dino.real.block{block_index}",
        )

    def __call__(self, x_or_x_list: Any) -> Any:
        if self.training or not isinstance(x_or_x_list, self.torch.Tensor):
            return self.original_forward(x_or_x_list)
        if getattr(x_or_x_list.device, "type", None) != "vulkan":
            return self.original_forward(x_or_x_list)
        return self.torch.ops.vulkan_prepack.run_vision_backbone_block_context(
            x_or_x_list,
            self.context,
        )


def vulkan_dav2_block_owner_forward(self: Any, x_or_x_list: Any) -> Any:
    return self._vulkan_dav2_block_owner(x_or_x_list)


def iter_dav2_block_slots(model: Any) -> Any:
    blocks = getattr(getattr(model, "pretrained", None), "blocks", None)
    if blocks is None:
        return
    for outer_index, entry in enumerate(blocks):
        if hasattr(entry, "attn") and hasattr(entry, "mlp"):
            yield blocks, outer_index, entry
            continue
        for inner_index, block in enumerate(entry):
            if hasattr(block, "attn") and hasattr(block, "mlp"):
                yield entry, inner_index, block


def install_vulkan_dav2_block_owner(torch_module: Any, model: Any) -> dict[str, Any]:
    enabled = env_flag_enabled("PYTORCH_VULKAN_DAV2_BLOCK_OWNER")
    limit = parse_owner_limit() if enabled else 0
    installed = 0
    if enabled and limit != 0:
        for block_index, (_container, _slot, block) in enumerate(
            iter_dav2_block_slots(model)
        ):
            if limit is not None and installed >= limit:
                break
            block._vulkan_dav2_block_owner = VulkanDAv2BlockOwner(
                torch_module,
                block,
                block_index,
            )
            block.forward = types.MethodType(vulkan_dav2_block_owner_forward, block)
            installed += 1

    return {
        "enabled": enabled,
        "limit": "all" if limit is None else limit,
        "installed": installed,
    }


def snapshot_vulkan_debug_counters(torch_module: Any, device_kind: str) -> dict[str, Any]:
    if device_kind != "vulkan" or not hasattr(torch_module.ops, "vulkan_prepack"):
        return {}

    ops = torch_module.ops.vulkan_prepack
    counters: dict[str, Any] = {}
    for name in (
        "cpu_fallback_count",
        "sync_readback_count",
        "sync_counters",
        "attention_plan_counters",
        "linear_plan_counters",
        "conv_plan_counters",
        "buffer_copy_counters",
        "vision_owner_counters",
        "zero_counters",
    ):
        fn = getattr(ops, name, None)
        if fn is None:
            continue
        try:
            counters[name] = fn()
        except RuntimeError as exc:
            counters[name] = f"unavailable: {exc}"
    return counters


def forward_sync_mode(device_kind: str) -> str:
    if device_kind == "directml":
        return "directml_single_scalar_readback"
    if device_kind in {"vulkan", "cuda"}:
        return "explicit_backend_sync"
    return "full_output_copy"


def default_output_dir() -> Path:
    candidates = [
        WORKSPACE_ROOT / "comparison",
        REPO_ROOT / "comparison",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def default_output_path(device: str, encoder: str) -> Path:
    return default_output_dir() / (
        f"benchmark_depth_anything_{device}_{encoder}_reconstructed_20260422.json"
    )


def prepare_image_on_device(
    model: Any,
    raw_image: Any,
    input_size: int,
    device: Any,
) -> tuple[Any, tuple[int, int]]:
    image, image_size = model.image2tensor(raw_image, input_size)
    if str(device) != "cpu":
        image = image.to(device)
    return image, image_size


def compute_depth_on_device(
    model: Any,
    image: Any,
    output_size: tuple[int, int],
    functional: Any,
) -> Any:
    depth = model.forward(image)
    return functional.interpolate(
        depth[:, None],
        output_size,
        mode="bilinear",
        align_corners=True,
    )[0, 0]


def consume_depth_output(
    depth: Any,
    torch_module: Any,
    device_kind: str,
    device: Any,
    output_mode: str,
) -> Any:
    if output_mode == OUTPUT_MODE_DEVICE_RESIDENT:
        _ = synchronize_result(torch_module, device_kind, depth, device)
        return depth
    if output_mode == OUTPUT_MODE_READBACK:
        return depth.cpu().numpy()
    raise ValueError(f"Unsupported output_mode: {output_mode}")


def infer_image_on_device(
    model: Any,
    raw_image: Any,
    input_size: int,
    device: Any,
    torch_module: Any,
    functional: Any,
    device_kind: str,
    output_mode: str,
) -> Any:
    image, (height, width) = prepare_image_on_device(model, raw_image, input_size, device)
    depth = compute_depth_on_device(model, image, (height, width), functional)
    return consume_depth_output(
        depth,
        torch_module,
        device_kind,
        device,
        output_mode,
    )


def run() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Depth Anything V2 inference.")
    parser.add_argument("--repo", help="Optional Depth Anything V2 repo override.")
    parser.add_argument("--checkpoint", help="Optional checkpoint override.")
    parser.add_argument("--encoder", default="vits", choices=sorted(MODEL_CONFIGS))
    parser.add_argument(
        "--device",
        required=True,
        choices=["cpu", "vulkan", "directml", "cuda"],
    )
    parser.add_argument(
        "--directml-device-index",
        type=int,
        help="Optional explicit DirectML adapter index override.",
    )
    parser.add_argument(
        "--cuda-device-index",
        type=int,
        help="Optional explicit CUDA device index override.",
    )
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument(
        "--image",
        help="Single image path for repeated timing. Defaults to demo01.jpg in the resolved Depth-Anything repo.",
    )
    parser.add_argument(
        "--image-dir",
        help="Directory of JPG demo images for one-pass timing. Defaults to assets/examples in the resolved Depth-Anything repo.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--skip-output-copy",
        action="store_true",
        help="Avoid full output readback in timed iterations. Vulkan uses explicit sync; DirectML uses single-scalar readback.",
    )
    parser.add_argument(
        "--out",
        help="Path to write JSON results. Defaults to the first available comparison directory.",
    )
    args = parser.parse_args()

    repo_path = resolve_depth_anything_repo(args.repo)
    default_image_path = repo_path / "assets" / "examples" / "demo01.jpg"
    default_image_dir = repo_path / "assets" / "examples"

    enable_local_pytorch_repo_imports()
    import torch
    import torch.nn.functional as F
    from depth_anything_v2.dpt import DepthAnythingV2

    device, device_kind, device_info = resolve_runtime_device(
        torch,
        args.device,
        directml_device_index=args.directml_device_index,
        cuda_device_index=args.cuda_device_index,
    )
    checkpoint = resolve_depth_anything_checkpoint(repo_path, args.encoder, args.checkpoint)
    image_path = Path(args.image).resolve() if args.image else default_image_path.resolve()
    image_dir = Path(args.image_dir).resolve() if args.image_dir else default_image_dir.resolve()
    image_paths = sorted(path for path in image_dir.glob("*.jpg") if path.is_file())

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if not image_paths:
        raise FileNotFoundError(f"No JPG files found in {image_dir}")

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.eval()
    if str(device) != "cpu":
        model = model.to(device)
    vulkan_block_owner = (
        install_vulkan_dav2_block_owner(torch, model)
        if device_kind == "vulkan"
        else {"enabled": False, "limit": 0, "installed": 0}
    )
    if device_kind == "vulkan" and hasattr(torch.ops, "vulkan_prepack"):
        reset_fallback = getattr(
            torch.ops.vulkan_prepack,
            "reset_fallback_counters",
            None,
        )
        if reset_fallback is not None:
            reset_fallback()

    raw_image = cv2.imread(str(image_path))
    if raw_image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    image_tensor, (height, width) = prepare_image_on_device(
        model,
        raw_image,
        args.input_size,
        device,
    )
    legacy_forward_output_mode = (
        OUTPUT_MODE_DEVICE_RESIDENT if args.skip_output_copy else OUTPUT_MODE_READBACK
    )
    legacy_sync_mode = (
        forward_sync_mode(device_kind)
        if legacy_forward_output_mode == OUTPUT_MODE_DEVICE_RESIDENT
        else "full_output_copy"
    )

    with inference_context(torch, device_kind):
        for _ in range(args.warmup):
            _ = infer_image_on_device(
                model,
                raw_image,
                args.input_size,
                device,
                torch,
                F,
                device_kind,
                OUTPUT_MODE_READBACK,
            )
            depth = compute_depth_on_device(model, image_tensor, (height, width), F)
            _ = consume_depth_output(
                depth,
                torch,
                device_kind,
                device,
                OUTPUT_MODE_DEVICE_RESIDENT,
            )
            depth = compute_depth_on_device(model, image_tensor, (height, width), F)
            _ = consume_depth_output(
                depth,
                torch,
                device_kind,
                device,
                OUTPUT_MODE_READBACK,
            )
            if legacy_forward_output_mode != OUTPUT_MODE_READBACK:
                _ = infer_image_on_device(
                    model,
                    raw_image,
                    args.input_size,
                    device,
                    torch,
                    F,
                    device_kind,
                    legacy_forward_output_mode,
                )

    end_to_end_with_readback_durations: list[float] = []
    legacy_end_to_end_durations: list[float] = []
    forward_device_resident_durations: list[float] = []
    forward_with_readback_durations: list[float] = []

    with inference_context(torch, device_kind):
        for _ in range(args.repeats):
            start = time.perf_counter()
            _ = infer_image_on_device(
                model,
                raw_image,
                args.input_size,
                device,
                torch,
                F,
                device_kind,
                OUTPUT_MODE_READBACK,
            )
            end_to_end_with_readback_durations.append(time.perf_counter() - start)

        if legacy_forward_output_mode != OUTPUT_MODE_READBACK:
            for _ in range(args.repeats):
                start = time.perf_counter()
                _ = infer_image_on_device(
                    model,
                    raw_image,
                    args.input_size,
                    device,
                    torch,
                    F,
                    device_kind,
                    legacy_forward_output_mode,
                )
                legacy_end_to_end_durations.append(time.perf_counter() - start)

        for _ in range(args.repeats):
            start = time.perf_counter()
            depth = compute_depth_on_device(model, image_tensor, (height, width), F)
            _ = consume_depth_output(
                depth,
                torch,
                device_kind,
                device,
                OUTPUT_MODE_DEVICE_RESIDENT,
            )
            forward_device_resident_durations.append(time.perf_counter() - start)

        for _ in range(args.repeats):
            start = time.perf_counter()
            depth = compute_depth_on_device(model, image_tensor, (height, width), F)
            _ = consume_depth_output(
                depth,
                torch,
                device_kind,
                device,
                OUTPUT_MODE_READBACK,
            )
            forward_with_readback_durations.append(time.perf_counter() - start)

    corpus_with_readback_durations: list[float] = []
    legacy_corpus_durations: list[float] = []
    with inference_context(torch, device_kind):
        for corpus_image_path in image_paths:
            corpus_image = cv2.imread(str(corpus_image_path))
            if corpus_image is None:
                raise RuntimeError(f"Failed to load corpus image: {corpus_image_path}")
            start = time.perf_counter()
            _ = infer_image_on_device(
                model,
                corpus_image,
                args.input_size,
                device,
                torch,
                F,
                device_kind,
                OUTPUT_MODE_READBACK,
            )
            corpus_with_readback_durations.append(time.perf_counter() - start)

        if legacy_forward_output_mode != OUTPUT_MODE_READBACK:
            for corpus_image_path in image_paths:
                corpus_image = cv2.imread(str(corpus_image_path))
                if corpus_image is None:
                    raise RuntimeError(f"Failed to load corpus image: {corpus_image_path}")
                start = time.perf_counter()
                _ = infer_image_on_device(
                    model,
                    corpus_image,
                    args.input_size,
                    device,
                    torch,
                    F,
                    device_kind,
                    legacy_forward_output_mode,
                )
                legacy_corpus_durations.append(time.perf_counter() - start)

    if legacy_forward_output_mode == OUTPUT_MODE_READBACK:
        legacy_end_to_end_durations = list(end_to_end_with_readback_durations)
        legacy_corpus_durations = list(corpus_with_readback_durations)

    legacy_forward_durations = (
        forward_device_resident_durations
        if legacy_forward_output_mode == OUTPUT_MODE_DEVICE_RESIDENT
        else forward_with_readback_durations
    )

    result = {
        "benchmark_name": "benchmark_depth_anything",
        "benchmark_contract": "legacy_depth_anything_v2_repo_forward",
        "script_origin": "ported_from_workspace_scripts_benchmarks_2026_04_22",
        "python_executable": sys.executable,
        "python_version": sys.version,
        "repo_root": str(REPO_ROOT),
        "workspace_root": str(WORKSPACE_ROOT),
        "depth_anything_repo": str(repo_path),
        "checkpoint": str(checkpoint),
        "device": args.device,
        "encoder": args.encoder,
        "input_size": args.input_size,
        "image": str(image_path),
        "image_dir": str(image_dir),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "image_count": len(image_paths),
        "torch_version": torch.__version__,
        "torch_vulkan_available": bool(
            getattr(torch, "is_vulkan_available", lambda: False)()
        ),
        "skip_output_copy": bool(args.skip_output_copy),
        "vulkan_dav2_block_owner": vulkan_block_owner,
        "vulkan_debug_counters": snapshot_vulkan_debug_counters(
            torch,
            device_kind,
        ),
        "timing_mode": legacy_forward_output_mode,
        "timing_sync_mode": legacy_sync_mode,
        "forward_measurement_modes": {
            "single_image_forward_device_resident": {
                "completion": "backend_complete_without_full_output_copy",
                "sync_mode": forward_sync_mode(device_kind),
            },
            "single_image_forward_with_readback": {
                "completion": "full_output_copy",
                "sync_mode": "full_output_copy",
            },
            "single_image_forward_only_legacy_alias": {
                "output_mode": legacy_forward_output_mode,
                "sync_mode": legacy_sync_mode,
            },
        },
        "device_info": device_info,
        "single_image_end_to_end": summarize_durations(
            "single_image_end_to_end",
            legacy_end_to_end_durations,
        ),
        "single_image_end_to_end_with_readback": summarize_durations(
            "single_image_end_to_end_with_readback",
            end_to_end_with_readback_durations,
        ),
        "single_image_forward_only": summarize_durations(
            "single_image_forward_only",
            legacy_forward_durations,
        ),
        "single_image_forward_device_resident": summarize_durations(
            "single_image_forward_device_resident",
            forward_device_resident_durations,
        ),
        "single_image_forward_with_readback": summarize_durations(
            "single_image_forward_with_readback",
            forward_with_readback_durations,
        ),
        "full_corpus_end_to_end": summarize_durations(
            "full_corpus_end_to_end",
            legacy_corpus_durations,
        ),
        "full_corpus_end_to_end_with_readback": summarize_durations(
            "full_corpus_end_to_end_with_readback",
            corpus_with_readback_durations,
        ),
    }

    out_path = Path(args.out).resolve() if args.out else default_output_path(
        args.device,
        args.encoder,
    )
    write_json(out_path, result)
    print(out_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    run()
