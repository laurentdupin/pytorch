from __future__ import annotations

import argparse
import contextlib
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
from benchmark_suite_common import VulkanCounterPhaseTracker
from vulkan_model_probe import create_vulkan_model_probe


OUTPUT_MODE_DEVICE_RESIDENT = "device_resident"
OUTPUT_MODE_READBACK = "readback"

FALLBACK_PHASE_UNKNOWN = 0
FALLBACK_PHASE_MODEL_SETUP = 1
FALLBACK_PHASE_OWNER_CONTEXT_CREATE = 2
FALLBACK_PHASE_OWNER_FORWARD = 3
FALLBACK_PHASE_POSITIONAL_EMBEDDING_SETUP = 5
FALLBACK_PHASE_READBACK = 6

SUBMIT_PHASE_UNKNOWN = 0
SUBMIT_PHASE_MODEL_SETUP = 1
SUBMIT_PHASE_PATCH_EMBED = 2
SUBMIT_PHASE_POSITIONAL_EMBEDDING_SETUP = 3
SUBMIT_PHASE_STACK_OWNER = 4
SUBMIT_PHASE_DECODER = 9
SUBMIT_PHASE_READBACK = 13
SUBMIT_PHASE_EXPLICIT_SYNCHRONIZE = 14

_TORCHVISION_COMPAT_LIBS: list[Any] = []


def ensure_torchvision_runtime_compat(torch_module: Any) -> None:
    try:
        torch_module._C._dispatch_find_schema_or_throw("torchvision::nms", "")
        return
    except RuntimeError:
        pass

    lib = torch_module.library.Library("torchvision", "DEF")
    lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    _TORCHVISION_COMPAT_LIBS.append(lib)


def optional_bias(module: Any) -> Any:
    return getattr(module, "bias", None)


def optional_layerscale_gamma(module: Any) -> Any:
    gamma = getattr(module, "gamma", None)
    return gamma if gamma is not None else None


def set_vulkan_fallback_phase(torch_module: Any, phase: int) -> None:
    ops = getattr(getattr(torch_module, "ops", None), "vulkan_prepack", None)
    setter = getattr(ops, "set_fallback_phase", None) if ops is not None else None
    if setter is not None:
        setter(int(phase))


def set_vulkan_submit_phase(torch_module: Any, phase: int) -> None:
    ops = getattr(getattr(torch_module, "ops", None), "vulkan_prepack", None)
    setter = getattr(ops, "set_submit_phase", None) if ops is not None else None
    if setter is not None:
        setter(int(phase))


def set_vulkan_timed_region(torch_module: Any, enabled: bool) -> None:
    ops = getattr(getattr(torch_module, "ops", None), "vulkan_prepack", None)
    setter = (
        getattr(ops, "set_benchmark_timed_region", None)
        if ops is not None
        else None
    )
    if setter is not None:
        setter(bool(enabled))


@contextlib.contextmanager
def vulkan_fallback_phase(torch_module: Any, phase: int) -> Any:
    set_vulkan_fallback_phase(torch_module, phase)
    try:
        yield
    finally:
        set_vulkan_fallback_phase(torch_module, FALLBACK_PHASE_UNKNOWN)


@contextlib.contextmanager
def vulkan_submit_phase(torch_module: Any, phase: int) -> Any:
    set_vulkan_submit_phase(torch_module, phase)
    try:
        yield
    finally:
        set_vulkan_submit_phase(torch_module, SUBMIT_PHASE_UNKNOWN)


@contextlib.contextmanager
def vulkan_timed_region(torch_module: Any) -> Any:
    set_vulkan_timed_region(torch_module, True)
    try:
        yield
    finally:
        set_vulkan_timed_region(torch_module, False)


class VulkanDAv2OwnerContextCache:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self._cache: dict[tuple[int, str, str, int, int], Any] = {}

    def key(self, block: Any) -> tuple[int, str, str, int, int]:
        weight = block.attn.qkv.weight
        return (
            id(block),
            str(weight.device),
            str(weight.dtype),
            int(block.attn.num_heads),
            int(weight.shape[1] // int(block.attn.num_heads)),
        )

    def get_or_create(self, block: Any, block_index: int) -> Any:
        key = self.key(block)
        cached = self._cache.get(key)
        if cached is not None:
            recorder = getattr(
                self.torch.ops.vulkan_prepack,
                "record_vision_owner_context_cache_hit",
                None,
            )
            if recorder is not None:
                recorder()
            return cached

        with vulkan_fallback_phase(self.torch, FALLBACK_PHASE_OWNER_CONTEXT_CREATE):
            context = self.torch.ops.vulkan_prepack.create_vision_backbone_block_context(
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
        self._cache[key] = context
        return context


class VulkanDAv2BlockOwner:
    def __init__(
        self,
        torch_module: Any,
        block: Any,
        block_index: int,
        context_cache: VulkanDAv2OwnerContextCache,
    ) -> None:
        self.torch = torch_module
        self.block = block
        self.original_forward = block.forward
        self.block_index = block_index
        self.training = bool(getattr(block, "training", False))
        self.context_cache = context_cache
        self.context_cache.get_or_create(block, block_index)

    def __call__(self, x_or_x_list: Any) -> Any:
        if self.training or not isinstance(x_or_x_list, self.torch.Tensor):
            return self.original_forward(x_or_x_list)
        if getattr(x_or_x_list.device, "type", None) != "vulkan":
            return self.original_forward(x_or_x_list)
        context = self.context_cache.get_or_create(self.block, self.block_index)
        with vulkan_fallback_phase(self.torch, FALLBACK_PHASE_OWNER_FORWARD):
            return self.torch.ops.vulkan_prepack.run_vision_backbone_block_context(
                x_or_x_list,
                context,
            )


def vulkan_dav2_block_owner_forward(self: Any, x_or_x_list: Any) -> Any:
    return self._vulkan_dav2_block_owner(x_or_x_list)


class VulkanDAv2BackboneStackOwner:
    def __init__(
        self,
        torch_module: Any,
        pretrained: Any,
        blocks: list[Any],
        context_cache: VulkanDAv2OwnerContextCache,
    ) -> None:
        self.torch = torch_module
        self.pretrained = pretrained
        self.blocks = blocks
        self.context_cache = context_cache
        self.original_not_chunked = pretrained._get_intermediate_layers_not_chunked
        self.training = bool(getattr(pretrained, "training", False))
        self.block_contexts = [
            context_cache.get_or_create(block, block_index)
            for block_index, block in enumerate(blocks)
        ]
        first = blocks[0]
        hidden = int(first.attn.qkv.weight.shape[1])
        mlp_hidden = int(first.mlp.fc1.weight.shape[0])
        num_heads = int(first.attn.num_heads)
        head_dim = hidden // num_heads
        self.stack_context = (
            self.torch.ops.vulkan_prepack.create_vision_backbone_stack_context(
                self.block_contexts,
                num_heads,
                head_dim,
                hidden,
                mlp_hidden,
            )
        )

    def __call__(self, x: Any, n: Any = 1) -> Any:
        if self.training or not isinstance(x, self.torch.Tensor):
            return self.original_not_chunked(x, n)
        if getattr(x.device, "type", None) != "vulkan":
            return self.original_not_chunked(x, n)

        x = self.pretrained.prepare_tokens_with_masks(x)
        total_block_len = len(self.blocks)
        if isinstance(n, int):
            capture_indices = list(range(total_block_len - n, total_block_len))
        else:
            capture_indices = [int(index) for index in n]

        with vulkan_submit_phase(
            self.torch,
            SUBMIT_PHASE_STACK_OWNER,
        ), vulkan_fallback_phase(self.torch, FALLBACK_PHASE_OWNER_FORWARD):
            outputs = self.torch.ops.vulkan_prepack.run_vision_backbone_stack_context(
                x,
                self.stack_context,
                capture_indices,
            )
        return list(outputs)


def vulkan_dav2_stack_not_chunked(self: Any, x: Any, n: Any = 1) -> Any:
    return self._vulkan_dav2_stack_owner(x, n)


def install_vulkan_fallback_phase_wrappers(torch_module: Any, model: Any) -> None:
    pretrained = getattr(model, "pretrained", None)
    if pretrained is None or getattr(pretrained, "_vulkan_phase_wrapped", False):
        return

    interpolate = getattr(pretrained, "interpolate_pos_encoding", None)
    if interpolate is not None:
        pos_cache: dict[tuple[str, str, tuple[int, ...], int, int], Any] = {}

        def phase_interpolate_pos_encoding(self: Any, *args: Any, **kwargs: Any) -> Any:
            x = args[0] if args else kwargs.get("x")
            w = args[1] if len(args) > 1 else kwargs.get("w")
            h = args[2] if len(args) > 2 else kwargs.get("h")
            if x is not None and w is not None and h is not None:
                key = (
                    str(x.device),
                    str(x.dtype),
                    tuple(int(s) for s in x.shape),
                    int(w),
                    int(h),
                )
                cached = pos_cache.get(key)
                if cached is not None:
                    return cached

            with vulkan_submit_phase(
                torch_module,
                SUBMIT_PHASE_POSITIONAL_EMBEDDING_SETUP,
            ), vulkan_fallback_phase(
                torch_module,
                FALLBACK_PHASE_POSITIONAL_EMBEDDING_SETUP,
            ):
                result = interpolate(*args, **kwargs)

            if x is not None and w is not None and h is not None:
                pos_cache[key] = result
            return result

        pretrained.interpolate_pos_encoding = types.MethodType(
            phase_interpolate_pos_encoding,
            pretrained,
        )

    pretrained._vulkan_phase_wrapped = True


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
    enabled = True
    limit = None
    installed = 0
    context_cache = VulkanDAv2OwnerContextCache(torch_module)
    flat_blocks: list[Any] = []
    for block_index, (_container, _slot, block) in enumerate(
        iter_dav2_block_slots(model)
    ):
        flat_blocks.append(block)
        context_cache.get_or_create(block, block_index)
        installed += 1

    pretrained = getattr(model, "pretrained", None)
    if (
        pretrained is not None
        and flat_blocks
        and hasattr(pretrained, "_get_intermediate_layers_not_chunked")
    ):
        pretrained._vulkan_dav2_stack_owner = VulkanDAv2BackboneStackOwner(
            torch_module,
            pretrained,
            flat_blocks,
            context_cache,
        )
        pretrained._get_intermediate_layers_not_chunked = types.MethodType(
            vulkan_dav2_stack_not_chunked,
            pretrained,
        )

    return {
        "enabled": enabled,
        "limit": "all",
        "installed": installed,
    }


def prewarm_vulkan_dav2_patch_and_positional_setup(
    torch_module: Any,
    model: Any,
    image_tensor: Any,
) -> None:
    pretrained = getattr(model, "pretrained", None)
    patch_embed = getattr(pretrained, "patch_embed", None)
    if pretrained is None or patch_embed is None:
        return
    if getattr(image_tensor.device, "type", None) != "vulkan":
        return

    with vulkan_fallback_phase(torch_module, FALLBACK_PHASE_MODEL_SETUP):
        with vulkan_submit_phase(torch_module, SUBMIT_PHASE_PATCH_EMBED):
            patch_tokens = patch_embed(image_tensor)
        cls_token = getattr(pretrained, "cls_token", None)
        if cls_token is None:
            return
        tokens = torch_module.cat(
            (cls_token.expand(patch_tokens.shape[0], -1, -1), patch_tokens),
            dim=1,
        )
        with vulkan_submit_phase(
            torch_module,
            SUBMIT_PHASE_POSITIONAL_EMBEDDING_SETUP,
        ):
            pretrained.interpolate_pos_encoding(
                tokens,
                int(image_tensor.shape[2]),
                int(image_tensor.shape[3]),
            )


def snapshot_vulkan_debug_counters(torch_module: Any, device_kind: str) -> dict[str, Any]:
    if device_kind != "vulkan" or not hasattr(torch_module.ops, "vulkan_prepack"):
        return {}

    ops = torch_module.ops.vulkan_prepack
    counters: dict[str, Any] = {}
    for name in (
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
        "linear_aggregate_snapshot",
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
    torch_module: Any | None = None,
) -> Any:
    if torch_module is None:
        depth = model.forward(image)
        return functional.interpolate(
            depth[:, None],
            output_size,
            mode="bilinear",
            align_corners=True,
        )[0, 0]
    else:
        with vulkan_submit_phase(torch_module, SUBMIT_PHASE_DECODER):
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
        with vulkan_submit_phase(torch_module, SUBMIT_PHASE_EXPLICIT_SYNCHRONIZE):
            _ = synchronize_result(torch_module, device_kind, depth, device)
        return depth
    if output_mode == OUTPUT_MODE_READBACK:
        with vulkan_submit_phase(
            torch_module,
            SUBMIT_PHASE_READBACK,
        ), vulkan_fallback_phase(torch_module, FALLBACK_PHASE_READBACK):
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
    with vulkan_submit_phase(torch_module, SUBMIT_PHASE_MODEL_SETUP):
        image, (height, width) = prepare_image_on_device(
            model,
            raw_image,
            input_size,
            device,
        )
    depth = compute_depth_on_device(
        model,
        image,
        (height, width),
        functional,
        torch_module,
    )
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
    parser.add_argument(
        "--vulkan-model-probe",
        choices=["off", "record", "continue_cpu_to_vulkan_safe"],
        default="off",
        help=(
            "Opt-in generic Vulkan model probe mode. Probe runs are diagnostics "
            "only and do not produce valid performance timings."
        ),
    )
    parser.add_argument("--vulkan-model-probe-policy")
    parser.add_argument("--vulkan-model-probe-out")
    parser.add_argument("--vulkan-model-probe-max-records", type=int)
    parser.add_argument(
        "--vulkan-model-probe-disable-owner-programs",
        action="store_true",
        help=(
            "Generic probe option: skip benchmark owner/region programs so "
            "TorchDispatch can observe underlying ATen ops. Normal runtime and "
            "probe runs without this flag keep owner programs enabled."
        ),
    )
    args = parser.parse_args()

    repo_path = resolve_depth_anything_repo(args.repo)
    default_image_path = repo_path / "assets" / "examples" / "demo01.jpg"
    default_image_dir = repo_path / "assets" / "examples"

    enable_local_pytorch_repo_imports()
    import torch
    import torch.nn.functional as F

    ensure_torchvision_runtime_compat(torch)
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

    if device_kind == "vulkan" and hasattr(torch.ops, "vulkan_prepack"):
        reset_fallback = getattr(
            torch.ops.vulkan_prepack,
            "reset_fallback_counters",
            None,
        )
        if reset_fallback is not None:
            reset_fallback()
    vulkan_phase_tracker = (
        VulkanCounterPhaseTracker(torch, device_kind)
        if device_kind == "vulkan"
        else None
    )

    with vulkan_submit_phase(torch, SUBMIT_PHASE_MODEL_SETUP), vulkan_fallback_phase(
        torch,
        FALLBACK_PHASE_MODEL_SETUP,
    ):
        model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.eval()
        if str(device) != "cpu":
            model = model.to(device)
    probe_enabled = device_kind == "vulkan" and args.vulkan_model_probe != "off"
    disable_owner_programs = (
        probe_enabled and args.vulkan_model_probe_disable_owner_programs
    )
    if device_kind == "vulkan" and not disable_owner_programs:
        with vulkan_submit_phase(torch, SUBMIT_PHASE_MODEL_SETUP):
            vulkan_block_owner = install_vulkan_dav2_block_owner(torch, model)
    elif device_kind == "vulkan":
        vulkan_block_owner = {
            "enabled": False,
            "limit": 0,
            "installed": 0,
            "disabled_by_vulkan_model_probe_owner_program_option": True,
        }
    else:
        vulkan_block_owner = {"enabled": False, "limit": 0, "installed": 0}
    if device_kind == "vulkan":
        with vulkan_submit_phase(torch, SUBMIT_PHASE_MODEL_SETUP):
            install_vulkan_fallback_phase_wrappers(torch, model)

    raw_image = cv2.imread(str(image_path))
    if raw_image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    with vulkan_submit_phase(torch, SUBMIT_PHASE_MODEL_SETUP):
        image_tensor, (height, width) = prepare_image_on_device(
            model,
            raw_image,
            args.input_size,
            device,
        )
    probe = None
    probe_summary: dict[str, Any] | None = None
    if probe_enabled:
        probe_out = (
            Path(args.vulkan_model_probe_out).resolve()
            if args.vulkan_model_probe_out
            else Path(args.out or "agent_space/depth_anything_probe.json")
            .resolve()
            .with_suffix(".probe.jsonl")
        )
        probe = create_vulkan_model_probe(
            torch,
            mode=args.vulkan_model_probe,
            out_path=probe_out,
            policy_path=args.vulkan_model_probe_policy,
            max_records=args.vulkan_model_probe_max_records,
            model={
                "task": "depth_anything_v2",
                "model_name": args.encoder,
                "model_id": args.encoder,
                "backend": args.device,
                "device": str(device),
            },
        )
    if device_kind == "vulkan":
        prewarm_vulkan_dav2_patch_and_positional_setup(torch, model, image_tensor)
        for corpus_image_path in image_paths:
            corpus_image = cv2.imread(str(corpus_image_path))
            if corpus_image is None:
                raise RuntimeError(f"Failed to load corpus image: {corpus_image_path}")
            with vulkan_submit_phase(torch, SUBMIT_PHASE_MODEL_SETUP):
                corpus_tensor, _ = prepare_image_on_device(
                    model,
                    corpus_image,
                    args.input_size,
                    device,
                )
            prewarm_vulkan_dav2_patch_and_positional_setup(
                torch,
                model,
                corpus_tensor,
            )
    if vulkan_phase_tracker is not None:
        vulkan_phase_tracker.mark("setup")
    legacy_forward_output_mode = (
        OUTPUT_MODE_DEVICE_RESIDENT if args.skip_output_copy else OUTPUT_MODE_READBACK
    )
    legacy_sync_mode = (
        forward_sync_mode(device_kind)
        if legacy_forward_output_mode == OUTPUT_MODE_DEVICE_RESIDENT
        else "full_output_copy"
    )

    if probe_enabled and probe is not None:
        probe.__enter__()
    try:
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
                depth = compute_depth_on_device(
                    model,
                    image_tensor,
                    (height, width),
                    F,
                    torch,
                )
                _ = consume_depth_output(
                    depth,
                    torch,
                    device_kind,
                    device,
                    OUTPUT_MODE_DEVICE_RESIDENT,
                )
                depth = compute_depth_on_device(
                    model,
                    image_tensor,
                    (height, width),
                    F,
                    torch,
                )
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
        if vulkan_phase_tracker is not None:
            vulkan_phase_tracker.mark("warmup")

        end_to_end_with_readback_durations: list[float] = []
        legacy_end_to_end_durations: list[float] = []
        forward_device_resident_durations: list[float] = []
        forward_with_readback_durations: list[float] = []

        with inference_context(torch, device_kind):
            for _ in range(args.repeats):
                start = time.perf_counter()
                with vulkan_timed_region(torch):
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
                    with vulkan_timed_region(torch):
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
                with vulkan_timed_region(torch):
                    depth = compute_depth_on_device(
                        model,
                        image_tensor,
                        (height, width),
                        F,
                        torch,
                    )
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
                with vulkan_timed_region(torch):
                    depth = compute_depth_on_device(
                        model,
                        image_tensor,
                        (height, width),
                        F,
                        torch,
                    )
                    _ = consume_depth_output(
                        depth,
                        torch,
                        device_kind,
                        device,
                        OUTPUT_MODE_READBACK,
                    )
                forward_with_readback_durations.append(time.perf_counter() - start)
        if vulkan_phase_tracker is not None:
            vulkan_phase_tracker.mark("timed_forward")

        corpus_with_readback_durations: list[float] = []
        legacy_corpus_durations: list[float] = []
        with inference_context(torch, device_kind):
            for corpus_image_path in image_paths:
                corpus_image = cv2.imread(str(corpus_image_path))
                if corpus_image is None:
                    raise RuntimeError(f"Failed to load corpus image: {corpus_image_path}")
                start = time.perf_counter()
                with vulkan_timed_region(torch):
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
                    with vulkan_timed_region(torch):
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
        if vulkan_phase_tracker is not None:
            vulkan_phase_tracker.mark("timed_corpus")
    finally:
        if probe_enabled and probe is not None:
            probe.__exit__(*sys.exc_info())
            probe_summary = probe.summary()

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
        "vulkan_model_probe_disable_owner_programs": bool(disable_owner_programs),
        "vulkan_dav2_block_owner": vulkan_block_owner,
        "vulkan_debug_counters": snapshot_vulkan_debug_counters(
            torch,
            device_kind,
        ),
        "vulkan_phase_counters": (
            vulkan_phase_tracker.summary()
            if vulkan_phase_tracker is not None
            else None
        ),
        "vulkan_model_probe": probe_summary,
        "performance_valid": not bool(probe_summary),
        "probe_timing_note": (
            "probe run uses CPU substitution/taint diagnostics; timing fields are not valid"
            if probe_summary
            else None
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
