from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import traceback
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
TOKEN_PREFIX_CAT_ADD_TOKEN_COUNTS = frozenset(
    (150, 260, 600, 620, 1350, 1380, 2400, 2440, 3750, 3850)
)
TOKEN_PREFIX_CAT_ADD_FEATURE_DIMS = frozenset((384, 768, 1024))
PATCH_EMBED_FEATURE_MAP_TO_TOKENS_SHAPES = frozenset(
    (
        (384, 10, 15),
        (768, 10, 15),
        (1024, 10, 15),
        (384, 13, 20),
        (768, 13, 20),
        (1024, 13, 20),
        (384, 20, 30),
        (768, 20, 30),
        (1024, 20, 30),
        (384, 20, 31),
        (768, 20, 31),
        (1024, 20, 31),
        (384, 30, 45),
        (768, 30, 45),
        (1024, 30, 45),
        (384, 30, 46),
        (768, 30, 46),
        (1024, 30, 46),
        (384, 40, 60),
        (768, 40, 60),
        (1024, 40, 60),
        (384, 40, 61),
        (768, 40, 61),
        (1024, 40, 61),
    )
)


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


def optional_module_bias(module: Any) -> Any:
    bias = getattr(module, "bias", None)
    return None if bias is None else bias.detach().to(dtype=module.weight.dtype).contiguous()


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


def create_vulkan_conv2d_context_from_module(torch_module: Any, module: Any) -> Any:
    return torch_module.ops.vulkan_prepack.create_conv2d_context(
        module.weight.detach().to(torch_module.float32).contiguous(),
        optional_module_bias(module),
        list(module.stride),
        list(module.padding),
        list(module.dilation),
        int(module.groups),
    )


def create_vulkan_tconv2d_context_from_module(torch_module: Any, module: Any) -> Any:
    return torch_module.ops.vulkan_prepack.create_tconv2d_context(
        module.weight.detach().to(torch_module.float32).contiguous(),
        optional_module_bias(module),
        list(module.stride),
        list(module.padding),
        list(module.output_padding),
        list(module.dilation),
        int(module.groups),
    )


def create_vulkan_decoder_fusion_context_from_module(
    torch_module: Any,
    module: Any,
    label: str,
) -> Any:
    residual_1 = getattr(module, "residual_1", None)
    residual_2 = getattr(module, "residual_2", None)
    if residual_1 is None:
        residual_1 = getattr(module, "resConfUnit1")
    if residual_2 is None:
        residual_2 = getattr(module, "resConfUnit2")
    return torch_module.ops.vulkan_prepack.create_vision_decoder_fusion_block_context(
        residual_1.conv1.weight.detach().to(torch_module.float32).contiguous(),
        optional_module_bias(residual_1.conv1),
        residual_1.conv2.weight.detach().to(torch_module.float32).contiguous(),
        optional_module_bias(residual_1.conv2),
        residual_2.conv1.weight.detach().to(torch_module.float32).contiguous(),
        optional_module_bias(residual_2.conv1),
        residual_2.conv2.weight.detach().to(torch_module.float32).contiguous(),
        optional_module_bias(residual_2.conv2),
        module.out_conv.weight.detach().to(torch_module.float32).contiguous(),
        optional_module_bias(module.out_conv),
        bool(getattr(module, "align_corners", True)),
        label,
    )


def create_vulkan_decoder_preprocess_head_context_from_model(
    torch_module: Any,
    model: Any,
    label: str,
) -> Any:
    depth_head = getattr(model, "depth_head")
    scratch = getattr(depth_head, "scratch")
    head_context = torch_module.ops.vulkan_prepack.create_vision_decoder_head_context(
        torch_module.zeros(1, dtype=torch_module.float32),
        create_vulkan_decoder_fusion_context_from_module(
            torch_module,
            scratch.refinenet4,
            f"{label}.refinenet4",
        ),
        create_vulkan_decoder_fusion_context_from_module(
            torch_module,
            scratch.refinenet3,
            f"{label}.refinenet3",
        ),
        create_vulkan_decoder_fusion_context_from_module(
            torch_module,
            scratch.refinenet2,
            f"{label}.refinenet2",
        ),
        create_vulkan_decoder_fusion_context_from_module(
            torch_module,
            scratch.refinenet1,
            f"{label}.refinenet1",
        ),
        create_vulkan_conv2d_context_from_module(torch_module, scratch.output_conv1),
        create_vulkan_conv2d_context_from_module(torch_module, scratch.output_conv2[0]),
        create_vulkan_conv2d_context_from_module(torch_module, scratch.output_conv2[2]),
        True,
        f"{label}.head",
    )
    return torch_module.ops.vulkan_prepack.create_vision_decoder_preprocess_head_context(
        torch_module.zeros(1, dtype=torch_module.float32),
        create_vulkan_conv2d_context_from_module(torch_module, depth_head.projects[0]),
        create_vulkan_conv2d_context_from_module(torch_module, depth_head.projects[1]),
        create_vulkan_conv2d_context_from_module(torch_module, depth_head.projects[2]),
        create_vulkan_conv2d_context_from_module(torch_module, depth_head.projects[3]),
        create_vulkan_tconv2d_context_from_module(
            torch_module,
            depth_head.resize_layers[0],
        ),
        create_vulkan_tconv2d_context_from_module(
            torch_module,
            depth_head.resize_layers[1],
        ),
        create_vulkan_conv2d_context_from_module(
            torch_module,
            depth_head.resize_layers[3],
        ),
        create_vulkan_conv2d_context_from_module(torch_module, scratch.layer1_rn),
        create_vulkan_conv2d_context_from_module(torch_module, scratch.layer2_rn),
        create_vulkan_conv2d_context_from_module(torch_module, scratch.layer3_rn),
        create_vulkan_conv2d_context_from_module(torch_module, scratch.layer4_rn),
        head_context,
        label,
    )


def create_vulkan_stack_output_device_bridge_contexts(
    torch_module: Any,
    model: Any,
) -> dict[str, Any]:
    pretrained = getattr(model, "pretrained")
    norm = getattr(pretrained, "norm")
    return {
        "norm_context": torch_module.ops.vulkan_prepack.create_layernorm_context(
            norm.weight,
            norm.bias,
            float(norm.eps),
        ),
        "decoder_context": create_vulkan_decoder_preprocess_head_context_from_model(
            torch_module,
            model,
            "vision.stack_output_device_bridge",
        ),
    }


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


def try_prepare_tokens_with_fused_prefix_cat_add(
    torch_module: Any,
    pretrained: Any,
    x: Any,
) -> Any | None:
    ops = getattr(getattr(torch_module, "ops", None), "vulkan_prepack", None)
    fused = getattr(ops, "token_prefix_cat_add", None) if ops is not None else None
    feature_map_to_tokens = (
        getattr(ops, "patch_embed_feature_map_to_tokens", None)
        if ops is not None
        else None
    )
    if fused is None or feature_map_to_tokens is None:
        return None

    patch_embed = getattr(pretrained, "patch_embed", None)
    patch_proj = getattr(patch_embed, "proj", None) if patch_embed is not None else None
    patch_norm = getattr(patch_embed, "norm", None) if patch_embed is not None else None
    cls_token = getattr(pretrained, "cls_token", None)
    interpolate_pos_encoding = getattr(
        pretrained,
        "interpolate_pos_encoding",
        None,
    )
    register_tokens = getattr(pretrained, "register_tokens", None)
    if (
        patch_embed is None
        or patch_proj is None
        or cls_token is None
        or interpolate_pos_encoding is None
        or register_tokens is not None
    ):
        return None
    if not isinstance(x, torch_module.Tensor):
        return None
    if getattr(x.device, "type", None) != "vulkan" or x.dtype != torch_module.float32:
        return None
    if x.dim() != 4 or int(x.shape[0]) != 1:
        return None
    if patch_norm is not None and not isinstance(patch_norm, torch_module.nn.Identity):
        return None

    patch_size = getattr(patch_embed, "patch_size", None)
    if isinstance(patch_size, int):
        patch_h = patch_w = int(patch_size)
    elif isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
        patch_h = int(patch_size[0])
        patch_w = int(patch_size[1])
    else:
        return None
    if patch_h <= 0 or patch_w <= 0:
        return None

    height = int(x.shape[-2])
    width = int(x.shape[-1])
    if height % patch_h != 0 or width % patch_w != 0:
        return None
    expected_tokens = (height // patch_h) * (width // patch_w)
    if expected_tokens not in TOKEN_PREFIX_CAT_ADD_TOKEN_COUNTS:
        return None
    expected_feature_h = height // patch_h
    expected_feature_w = width // patch_w
    expected_feature_dim = int(getattr(patch_proj, "out_channels", 0))
    if (
        expected_feature_dim,
        expected_feature_h,
        expected_feature_w,
    ) not in PATCH_EMBED_FEATURE_MAP_TO_TOKENS_SHAPES:
        return None

    with vulkan_submit_phase(torch_module, SUBMIT_PHASE_PATCH_EMBED):
        feature_map = patch_proj(x)
    if (
        not isinstance(feature_map, torch_module.Tensor)
        or getattr(feature_map.device, "type", None) != "vulkan"
        or feature_map.dtype != torch_module.float32
        or feature_map.dim() != 4
        or int(feature_map.shape[0]) != 1
        or (
            int(feature_map.shape[1]),
            int(feature_map.shape[2]),
            int(feature_map.shape[3]),
        )
        not in PATCH_EMBED_FEATURE_MAP_TO_TOKENS_SHAPES
    ):
        return None
    tokens = feature_map_to_tokens(feature_map)
    if (
        not isinstance(tokens, torch_module.Tensor)
        or getattr(tokens.device, "type", None) != "vulkan"
        or tokens.dtype != torch_module.float32
        or tokens.dim() != 3
        or int(tokens.shape[0]) != 1
    ):
        return None
    token_count = int(tokens.shape[1])
    feature_dim = int(tokens.shape[2])
    if (
        token_count != expected_tokens
        or token_count not in TOKEN_PREFIX_CAT_ADD_TOKEN_COUNTS
        or feature_dim not in TOKEN_PREFIX_CAT_ADD_FEATURE_DIMS
    ):
        return None
    if (
        not isinstance(cls_token, torch_module.Tensor)
        or getattr(cls_token.device, "type", None) != "vulkan"
        or cls_token.dtype != torch_module.float32
        or tuple(int(s) for s in cls_token.shape) != (1, 1, feature_dim)
    ):
        return None

    pos_probe = torch_module.empty(
        (1, token_count + 1, feature_dim),
        dtype=tokens.dtype,
        device=tokens.device,
    )
    pos = interpolate_pos_encoding(pos_probe, width, height)
    if (
        not isinstance(pos, torch_module.Tensor)
        or getattr(pos.device, "type", None) != "vulkan"
        or pos.dtype != torch_module.float32
        or tuple(int(s) for s in pos.shape)
        != (1, token_count + 1, feature_dim)
    ):
        return None

    return fused(cls_token, tokens, pos)


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

        fused_tokens = try_prepare_tokens_with_fused_prefix_cat_add(
            self.torch,
            self.pretrained,
            x,
        )
        x = (
            fused_tokens
            if fused_tokens is not None
            else self.pretrained.prepare_tokens_with_masks(x)
        )
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


class VulkanStackOutputDeviceBridge:
    def __init__(
        self,
        torch_module: Any,
        model: Any,
        stack_owner: VulkanDAv2BackboneStackOwner,
        bridge_contexts: dict[str, Any],
        label: str,
    ) -> None:
        self.torch = torch_module
        self.model = model
        self.pretrained = getattr(model, "pretrained")
        self.stack_owner = stack_owner
        self.original_forward = model.forward
        self.capture_indices = [
            int(index)
            for index in getattr(model, "intermediate_layer_idx")[
                getattr(model, "encoder")
            ]
        ]
        hidden = int(getattr(self.pretrained, "embed_dim"))
        self.normalized_shape = [hidden]
        self.strip_prefix_tokens = 1 + int(
            getattr(self.pretrained, "num_register_tokens", 0)
        )
        self.norm_context = bridge_contexts["norm_context"]
        self.decoder_context = bridge_contexts["decoder_context"]
        self.bridge_consumer_id = f"{label}.decoder_preprocess_head"
        self.bridge_consumer_context = "VisionDecoderPreprocessHeadContext"

    def registrations_for_input(self, x: Any) -> list[dict[str, Any]]:
        if not isinstance(x, self.torch.Tensor) or x.dim() != 4:
            return []
        patch_h = int(x.shape[-2]) // 14
        patch_w = int(x.shape[-1]) // 14
        token_count = patch_h * patch_w + self.strip_prefix_tokens
        hidden = self.normalized_shape[0]
        return [
            stack_output_device_consumer_registration(
                captured_block=block,
                captured_substep="residual2",
                output_role="stack_residual2_output",
                output_shape=f"[{token_count},{hidden}]",
                downstream_device_consumer_id=self.bridge_consumer_id,
                downstream_device_consumer_context=self.bridge_consumer_context,
                expected_consumer_input_index=index,
                expected_consumer_shape=f"[1,{patch_h * patch_w},{hidden}]",
                expected_consumer_layout="vulkan_buffer_token_sequence",
                stack_context_id="VisionBackboneStackContext",
                stack_session_id="benchmark_forward_bridge_region",
                stack_plan_id=None,
                producer_layout="vulkan_buffer_token_sequence_with_prefix",
                strip_token_or_view_relation=(
                    f"strip_prefix_tokens={self.strip_prefix_tokens}"
                ),
                consumer_in_same_planned_region=True,
                python_public_boundary_before_consumption=False,
                host_visible_boundary_before_consumption=False,
                host_visible_access_before_consumption=False,
                host_readback_before_consumption=False,
            )
            for index, block in enumerate(self.capture_indices)
        ]

    def __call__(self, x: Any) -> Any:
        if getattr(self.model, "training", False):
            return self.original_forward(x)
        if not isinstance(x, self.torch.Tensor):
            return self.original_forward(x)
        if getattr(x.device, "type", None) != "vulkan" or x.dim() != 4:
            return self.original_forward(x)

        patch_h = int(x.shape[-2]) // 14
        patch_w = int(x.shape[-1]) // 14
        if patch_h <= 0 or patch_w <= 0:
            return self.original_forward(x)

        tokens = try_prepare_tokens_with_fused_prefix_cat_add(
            self.torch,
            self.pretrained,
            x,
        )
        if tokens is None:
            tokens = self.pretrained.prepare_tokens_with_masks(x)

        output_size = [patch_h * 14, patch_w * 14]
        with vulkan_submit_phase(
            self.torch,
            SUBMIT_PHASE_STACK_OWNER,
        ), vulkan_fallback_phase(self.torch, FALLBACK_PHASE_OWNER_FORWARD):
            depth = (
                self.torch.ops.vulkan_prepack
                .run_vision_stack_captures_decoder_preprocess_bridge(
                    tokens,
                    self.stack_owner.stack_context,
                    self.capture_indices,
                    self.normalized_shape,
                    self.norm_context,
                    self.strip_prefix_tokens,
                    patch_h,
                    patch_w,
                    output_size,
                    self.decoder_context,
                )
            )
        return self.torch.relu(depth).squeeze(1)


def vulkan_stack_output_device_bridge_forward(self: Any, x: Any) -> Any:
    return self._vulkan_stack_output_device_bridge(x)


def install_vulkan_stack_output_device_bridge(
    torch_module: Any,
    model: Any,
    bridge_contexts: dict[str, Any] | None,
) -> dict[str, Any]:
    pretrained = getattr(model, "pretrained", None)
    stack_owner = (
        getattr(pretrained, "_vulkan_dav2_stack_owner", None)
        if pretrained is not None
        else None
    )
    if pretrained is None or stack_owner is None:
        return {"enabled": False, "reason": "missing_stack_owner"}
    if bridge_contexts is None:
        return {"enabled": False, "reason": "missing_bridge_contexts"}
    if getattr(model, "_vulkan_stack_output_device_bridge_enabled", False):
        return {"enabled": True, "already_installed": True}

    bridge = VulkanStackOutputDeviceBridge(
        torch_module,
        model,
        stack_owner,
        bridge_contexts,
        "vision.stack_output_device_bridge",
    )
    model._vulkan_stack_output_device_bridge = bridge
    model._vulkan_original_forward_for_stack_output_device_bridge = model.forward
    model.forward = types.MethodType(vulkan_stack_output_device_bridge_forward, model)
    model._vulkan_stack_output_device_bridge_enabled = True
    return {
        "enabled": True,
        "contract_name": STACK_OUTPUT_DEVICE_CONSUMER_BRIDGE_CONTRACT,
        "backend_op": "vulkan_prepack::run_vision_stack_captures_decoder_preprocess_bridge",
        "capture_indices": bridge.capture_indices,
        "consumer_id": bridge.bridge_consumer_id,
        "consumer_context": bridge.bridge_consumer_context,
    }


def validate_vulkan_stack_output_device_bridge_sanity(
    model: Any,
    image_tensor: Any,
) -> dict[str, Any]:
    bridge = getattr(model, "_vulkan_stack_output_device_bridge", None)
    if bridge is None:
        return {"enabled": False, "reason": "bridge_not_installed"}
    with bridge.torch.inference_mode():
        bridge_output = bridge(image_tensor)
        reference_output = bridge.original_forward(image_tensor)
    bridge_cpu = bridge_output.detach().cpu()
    reference_cpu = reference_output.detach().cpu()
    diff = (bridge_cpu - reference_cpu).abs()
    return {
        "enabled": True,
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "finite": bool(bridge.torch.isfinite(bridge_cpu).all().item()),
        "bridge_shape": list(bridge_output.shape),
        "reference_shape": list(reference_output.shape),
        "passed": bool(
            bridge.torch.allclose(
                bridge_cpu,
                reference_cpu,
                atol=5e-3,
                rtol=5e-3,
            )
        ),
    }


def vulkan_dav2_stack_not_chunked(self: Any, x: Any, n: Any = 1) -> Any:
    return self._vulkan_dav2_stack_owner(x, n)


def install_vulkan_prepare_tokens_wrapper(torch_module: Any, pretrained: Any) -> None:
    if pretrained is None or getattr(
        pretrained, "_vulkan_prepare_tokens_wrapped", False
    ):
        return
    prepare_tokens = getattr(pretrained, "prepare_tokens_with_masks", None)
    if prepare_tokens is None:
        return

    original_prepare_tokens = prepare_tokens

    def phase_prepare_tokens_with_masks(
        self: Any, x: Any, *args: Any, **kwargs: Any
    ) -> Any:
        masks = kwargs.get("masks")
        if args:
            masks = args[0]
        if masks is None:
            fused_tokens = try_prepare_tokens_with_fused_prefix_cat_add(
                torch_module,
                self,
                x,
            )
            if fused_tokens is not None:
                return fused_tokens
        return original_prepare_tokens(x, *args, **kwargs)

    pretrained._vulkan_original_prepare_tokens_with_masks = original_prepare_tokens
    pretrained.prepare_tokens_with_masks = types.MethodType(
        phase_prepare_tokens_with_masks,
        pretrained,
    )
    pretrained._vulkan_prepare_tokens_wrapped = True


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

    install_vulkan_prepare_tokens_wrapper(torch_module, pretrained)

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
    prepare_tokens = getattr(pretrained, "prepare_tokens_with_masks", None)
    if prepare_tokens is not None:
        with vulkan_fallback_phase(torch_module, FALLBACK_PHASE_MODEL_SETUP):
            prepare_tokens(image_tensor)
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
        "region_lifetime_submit_attribution_snapshot",
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


def output_path_from_args(args: argparse.Namespace) -> Path:
    return (
        Path(args.out).resolve()
        if args.out
        else default_output_path(args.device, args.encoder).resolve()
    )


def _safe_phase_summary(vulkan_phase_tracker: Any) -> Any:
    if vulkan_phase_tracker is None:
        return None
    try:
        return vulkan_phase_tracker.summary()
    except Exception as exc:
        return {"error": repr(exc)}


def _parse_vulkan_snapshot_fields(row: Any) -> dict[str, str]:
    if not isinstance(row, str):
        return {}
    fields: dict[str, str] = {}
    for token in row.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value.rstrip(",")
    return fields


def _counter_as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _field_bool(fields: dict[str, str], key: str) -> bool | str:
    if key not in fields:
        return "unknown"
    return fields[key] == "1"


def _phase_delta(phase_summary: Any, phase_name: str) -> dict[str, Any]:
    if not isinstance(phase_summary, dict):
        return {}
    for phase in phase_summary.get("phases", []):
        if isinstance(phase, dict) and phase.get("name") == phase_name:
            delta = phase.get("delta")
            return delta if isinstance(delta, dict) else {}
    return {}


def _bridge_snapshot_rows(
    debug_counters: dict[str, Any],
    phase_summary: Any,
) -> tuple[str, list[Any]]:
    timed_delta = _phase_delta(phase_summary, "timed_forward")
    timed_rows = timed_delta.get("region_lifetime_submit_attribution_snapshot_delta")
    if isinstance(timed_rows, list):
        return "timed_forward", timed_rows
    rows = debug_counters.get("region_lifetime_submit_attribution_snapshot")
    if isinstance(rows, list):
        return "total", rows
    return "unavailable", []


def _bridge_stack_capture_storage_rows(
    debug_counters: dict[str, Any],
    phase_summary: Any,
) -> dict[str, dict[str, str]]:
    timed_delta = _phase_delta(phase_summary, "timed_forward")
    rows = timed_delta.get("stack_allocation_aggregate_snapshot_delta")
    if not isinstance(rows, list):
        rows = debug_counters.get("stack_allocation_aggregate_snapshot")
    storage_rows: dict[str, dict[str, str]] = {}
    if not isinstance(rows, list):
        return storage_rows
    for row in rows:
        fields = _parse_vulkan_snapshot_fields(row)
        if (
            fields.get("phase") != "intermediate_capture"
            or fields.get("role") != "vision_stack_capture"
        ):
            continue
        block = fields.get("block")
        if block is not None:
            storage_rows[block] = fields
    return storage_rows


def _field_or_storage_bool(
    fields: dict[str, str],
    storage_fields: dict[str, str],
    key: str,
) -> bool | str:
    value = _field_bool(fields, key)
    return value if value != "unknown" else _field_bool(storage_fields, key)


STACK_OUTPUT_DEVICE_CONSUMER_BRIDGE_CONTRACT = "StackOutputToDeviceConsumerBridgeContract"
_BRIDGE_NOT_REGISTERED = "not_registered"
_BRIDGE_PUBLIC_BOUNDARY_REJECT = (
    "public_tensor_array_boundary_before_downstream_consumer"
)


def stack_output_device_consumer_registration(
    *,
    captured_block: int | str,
    captured_substep: str,
    output_role: str,
    output_shape: str,
    downstream_device_consumer_id: str,
    downstream_device_consumer_context: str,
    expected_consumer_input_index: int | str,
    expected_consumer_shape: str,
    expected_consumer_layout: str,
    stack_context_id: str = _BRIDGE_NOT_REGISTERED,
    stack_session_id: str = _BRIDGE_NOT_REGISTERED,
    stack_plan_id: str | None = None,
    producer_allocation_id: str | None = None,
    producer_allocation_generation: str | None = None,
    producer_byte_offset: str | None = None,
    producer_byte_range: str | None = None,
    producer_layout: str = "unknown",
    strip_token_or_view_relation: str = "unknown",
    consumer_in_same_planned_region: bool = False,
    python_public_boundary_before_consumption: bool = True,
    host_visible_boundary_before_consumption: bool | str = "unknown",
    host_visible_access_before_consumption: bool | str = "unknown",
    host_readback_before_consumption: bool | str = "unknown",
) -> dict[str, Any]:
    """Build a dry-run stack-output to device-consumer registration record."""
    return {
        "contract_name": STACK_OUTPUT_DEVICE_CONSUMER_BRIDGE_CONTRACT,
        "stack_context_id": stack_context_id,
        "stack_session_id": stack_session_id,
        "stack_plan_id": stack_plan_id,
        "captured_block": str(captured_block),
        "captured_substep": captured_substep,
        "output_role": output_role,
        "output_shape": output_shape,
        "producer_allocation_id": producer_allocation_id,
        "producer_allocation_generation": producer_allocation_generation,
        "producer_byte_offset": producer_byte_offset,
        "producer_byte_range": producer_byte_range,
        "producer_layout": producer_layout,
        "strip_token_or_view_relation": strip_token_or_view_relation,
        "downstream_device_consumer_id": downstream_device_consumer_id,
        "downstream_device_consumer_context": downstream_device_consumer_context,
        "expected_consumer_input_index": expected_consumer_input_index,
        "expected_consumer_shape": expected_consumer_shape,
        "expected_consumer_layout": expected_consumer_layout,
        "consumer_in_same_planned_region": bool(consumer_in_same_planned_region),
        "python_public_boundary_before_consumption": bool(
            python_public_boundary_before_consumption
        ),
        "host_visible_boundary_before_consumption": (
            host_visible_boundary_before_consumption
        ),
        "host_visible_access_before_consumption": host_visible_access_before_consumption,
        "host_readback_before_consumption": host_readback_before_consumption,
    }


def _bridge_registration_key(fields: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(fields.get("captured_block", "-1")),
        str(fields.get("output_role", "unknown")),
        str(fields.get("output_shape", "unknown")),
    )


def _normalize_stack_output_device_consumer_registrations(
    registrations: Any,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    normalized: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not isinstance(registrations, list):
        return normalized
    for registration in registrations:
        if not isinstance(registration, dict):
            continue
        normalized[_bridge_registration_key(registration)] = dict(registration)
    return normalized


def _bridge_reject_reason(capture: dict[str, Any]) -> str:
    if (
        capture["downstream_device_consumer_id"] == _BRIDGE_NOT_REGISTERED
        or capture["downstream_device_consumer_context"] == _BRIDGE_NOT_REGISTERED
    ):
        return _BRIDGE_PUBLIC_BOUNDARY_REJECT
    if capture["python_public_boundary_before_consumption"]:
        return _BRIDGE_PUBLIC_BOUNDARY_REJECT
    if not capture["consumer_in_same_planned_region"]:
        return "downstream_device_consumer_not_in_same_planned_region"
    if capture["host_visible_boundary_before_consumption"] is not False:
        return "host_visible_boundary_before_downstream_consumer"
    if capture["host_visible_access_before_consumption"] is not False:
        return "host_visible_access_before_downstream_consumer"
    if capture["host_readback_before_consumption"] is not False:
        return "host_readback_before_downstream_consumer"
    return "none"


def build_stack_output_to_device_consumer_bridge_dry_run(
    debug_counters: dict[str, Any],
    phase_summary: Any,
    downstream_device_consumer_registrations: Any = None,
) -> dict[str, Any]:
    """Summarize whether escaping stack captures have a proven device consumer."""
    phase_name, rows = _bridge_snapshot_rows(debug_counters, phase_summary)
    capture_storage_rows = _bridge_stack_capture_storage_rows(
        debug_counters,
        phase_summary,
    )
    plan_keys = debug_counters.get("stack_shape_plan_keys")
    stack_plan_id = plan_keys[0] if isinstance(plan_keys, list) and plan_keys else None
    registrations = _normalize_stack_output_device_consumer_registrations(
        downstream_device_consumer_registrations
    )
    captures: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        fields = _parse_vulkan_snapshot_fields(row)
        if fields.get("resource_class") != "host_visible_or_requested_output":
            continue
        if fields.get("role") not in {"stack_residual2_output", "stack_requested_output"}:
            continue
        storage_fields = capture_storage_rows.get(fields.get("block", ""), {})
        key = (
            fields.get("block", "-1"),
            fields.get("role", "unknown"),
            fields.get("shape", "unknown"),
        )
        registration = registrations.get(key, {})
        capture = captures.setdefault(
            key,
            {
                "producer_stack_context_id": registration.get(
                    "stack_context_id",
                    "not_exposed",
                ),
                "producer_stack_session_id": registration.get(
                    "stack_session_id",
                    "not_exposed",
                ),
                "stack_plan_id": stack_plan_id,
                "captured_block": fields.get("block", "-1"),
                "captured_substep": fields.get("producer_substep", "unknown"),
                "output_role": fields.get("role", "unknown"),
                "output_shape": fields.get("shape", "unknown"),
                "captured_tensor_shape": storage_fields.get("shape", "unknown"),
                "captured_tensor_strides": storage_fields.get("strides", "unknown"),
                "output_dtype": fields.get("dtype", "unknown"),
                "output_lifetime": fields.get("lifetime", "unknown"),
                "direct_buffer": _field_or_storage_bool(
                    fields,
                    storage_fields,
                    "direct_buffer",
                ),
                "buffer_storage": _field_or_storage_bool(
                    fields,
                    storage_fields,
                    "buffer_storage",
                ),
                "image_storage": _field_or_storage_bool(
                    fields,
                    storage_fields,
                    "image_storage",
                ),
                "allocation_label": fields.get("allocation_label", "unknown"),
                "allocation_has_generation": _field_bool(
                    fields,
                    "allocation_has_generation",
                ),
                "allocation_has_byte_range": _field_bool(
                    fields,
                    "allocation_has_byte_range",
                ),
                "allocation_byte_offset": fields.get("allocation_byte_offset"),
                "allocation_byte_range": fields.get("allocation_byte_range"),
                "allocation_allocated_bytes": fields.get("allocation_allocated_bytes"),
                "last_use_candidate": fields.get("last_use_candidate", "unknown"),
                "expected_consumer_phase": fields.get("expected_consumer_phase", "unknown"),
                "expected_consumer_block": fields.get("expected_consumer_block", "-1"),
                "requested_intermediate": _field_bool(fields, "requested_intermediate"),
                "final_output": _field_bool(fields, "final_output"),
                "escapes_stack": _field_bool(fields, "escapes_stack"),
                "alias_or_view": _field_bool(fields, "alias_or_view"),
                "aliases_runtime_input": _field_bool(fields, "aliases_runtime_input"),
                "aliases_runtime_output": _field_bool(fields, "aliases_runtime_output"),
                "strip_token_or_view_relation": registration.get(
                    "strip_token_or_view_relation",
                    "not_observed_in_stack_region",
                ),
                "downstream_device_consumer_id": registration.get(
                    "downstream_device_consumer_id",
                    _BRIDGE_NOT_REGISTERED,
                ),
                "downstream_device_consumer_context": registration.get(
                    "downstream_device_consumer_context",
                    _BRIDGE_NOT_REGISTERED,
                ),
                "expected_consumer_input_index": registration.get(
                    "expected_consumer_input_index",
                    "unknown",
                ),
                "expected_consumer_shape": registration.get(
                    "expected_consumer_shape",
                    "unknown",
                ),
                "expected_consumer_layout": registration.get(
                    "expected_consumer_layout",
                    "unknown",
                ),
                "consumer_in_same_planned_region": bool(
                    registration.get("consumer_in_same_planned_region", False)
                ),
                "python_public_boundary_before_consumption": bool(
                    registration.get("python_public_boundary_before_consumption", True)
                ),
                "host_visible_boundary_before_consumption": registration.get(
                    "host_visible_boundary_before_consumption",
                    _field_bool(fields, "capture_or_public_output"),
                ),
                "host_visible_access_before_consumption": registration.get(
                    "host_visible_access_before_consumption",
                    "unknown",
                ),
                "host_readback_before_consumption": registration.get(
                    "host_readback_before_consumption",
                    "unknown",
                ),
                "capture_storage_observed": bool(storage_fields),
                "registration_observed": bool(registration),
                "accepted": False,
                "reject_reason": _BRIDGE_PUBLIC_BOUNDARY_REJECT,
                "resource_records": 0,
                "queue_submit_records": 0,
                "bytes": 0,
            },
        )
        capture["resource_records"] += _counter_as_int(fields.get("count", 1)) or 1
        capture["queue_submit_records"] += _counter_as_int(fields.get("queue_submit"))
        capture["bytes"] += _counter_as_int(fields.get("bytes"))
    rejected: dict[str, int] = {}
    accepted = 0
    would_remove = 0
    for capture in captures.values():
        reject_reason = _bridge_reject_reason(capture)
        if reject_reason == "none":
            capture["accepted"] = True
            capture["reject_reason"] = "none"
            accepted += 1
            would_remove += capture["queue_submit_records"]
        else:
            capture["reject_reason"] = reject_reason
            rejected[reject_reason] = rejected.get(reject_reason, 0) + 1
    return {
        "contract_name": STACK_OUTPUT_DEVICE_CONSUMER_BRIDGE_CONTRACT,
        "mode": "dry_run",
        "phase_source": phase_name,
        "behavior_changed": False,
        "stack_plan_id": stack_plan_id,
        "registered_downstream_device_consumer_count": len(registrations),
        "bridge_candidate_count": len(captures),
        "proven_device_consumer_count": accepted,
        "rejected_reasons": rejected,
        "would_remove_phase_boundary_syncs": would_remove,
        "captures": list(captures.values()),
        "architecture_gap": (
            "captured stack outputs are public Tensor[] results before a "
            "downstream device consumer is registered in the same planned region"
        ),
    }


def install_failure_artifact_hook(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> Any:
    original_hook = sys.excepthook

    def failure_hook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        try:
            torch_module = context.get("torch")
            device_kind = str(context.get("device_kind", ""))
            debug_counters = (
                snapshot_vulkan_debug_counters(torch_module, device_kind)
                if torch_module is not None
                else {}
            )
            phase_counters = _safe_phase_summary(context.get("vulkan_phase_tracker"))
            out_path = output_path_from_args(args)
            result = {
                "benchmark_name": "benchmark_depth_anything",
                "benchmark_contract": "legacy_depth_anything_v2_repo_forward",
                "status": "fail",
                "timing_valid": False,
                "performance_valid": False,
                "failure": {
                    "type": exc_type.__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exception(exc_type, exc, tb),
                },
                "python_executable": sys.executable,
                "python_version": sys.version,
                "repo_root": str(REPO_ROOT),
                "workspace_root": str(WORKSPACE_ROOT),
                "depth_anything_repo": str(context.get("repo_path", "")),
                "checkpoint": str(context.get("checkpoint", "")),
                "device": args.device,
                "encoder": args.encoder,
                "input_size": args.input_size,
                "image": str(context.get("image_path", "")),
                "image_dir": str(context.get("image_dir", "")),
                "warmup": args.warmup,
                "repeats": args.repeats,
                "torch_version": getattr(torch_module, "__version__", None),
                "torch_vulkan_available": (
                    bool(getattr(torch_module, "is_vulkan_available", lambda: False)())
                    if torch_module is not None
                    else None
                ),
                "skip_output_copy": bool(args.skip_output_copy),
                "vulkan_model_probe": context.get("probe_summary"),
                "vulkan_model_probe_disable_owner_programs": bool(
                    context.get("disable_owner_programs", False)
                ),
                "vulkan_dav2_block_owner": context.get("vulkan_block_owner"),
                "vulkan_stack_output_device_bridge": context.get(
                    "vulkan_stack_output_device_bridge"
                ),
                "vulkan_stack_output_device_bridge_sanity": context.get(
                    "vulkan_stack_output_device_bridge_sanity"
                ),
                "stack_output_device_consumer_registrations": context.get(
                    "stack_output_device_consumer_registrations"
                ),
                "device_info": context.get("device_info"),
                "vulkan_debug_counters": debug_counters,
                "vulkan_phase_counters": phase_counters,
                "vulkan_stack_output_device_bridge_dry_run": (
                    build_stack_output_to_device_consumer_bridge_dry_run(
                        debug_counters,
                        phase_counters,
                        context.get("stack_output_device_consumer_registrations"),
                    )
                    if device_kind == "vulkan"
                    else None
                ),
                "allocation_failure_snapshot": debug_counters.get(
                    "last_allocation_failure_snapshot",
                    [],
                ),
            }
            write_json(out_path, result)
        except Exception as hook_exc:
            print(
                f"Failed to write benchmark failure artifact: {hook_exc!r}",
                file=sys.stderr,
            )
        finally:
            original_hook(exc_type, exc, tb)

    sys.excepthook = failure_hook
    return original_hook


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
    parser.add_argument(
        "--vulkan-stack-output-device-bridge",
        action="store_true",
        help=(
            "Opt in to the generic Vulkan stack-capture to decoder/head bridge "
            "path. The bridge keeps captures private to a same-region device "
            "consumer and records StackOutputToDeviceConsumerBridgeContract "
            "diagnostics."
        ),
    )
    args = parser.parse_args()
    failure_context: dict[str, Any] = {}
    original_excepthook = install_failure_artifact_hook(args, failure_context)

    repo_path = resolve_depth_anything_repo(args.repo)
    default_image_path = repo_path / "assets" / "examples" / "demo01.jpg"
    default_image_dir = repo_path / "assets" / "examples"
    failure_context["repo_path"] = repo_path

    enable_local_pytorch_repo_imports()
    import torch
    import torch.nn.functional as F
    failure_context["torch"] = torch

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
    failure_context.update(
        {
            "checkpoint": checkpoint,
            "device_kind": device_kind,
            "device_info": device_info,
            "image_path": image_path,
            "image_dir": image_dir,
        }
    )

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
    failure_context["vulkan_phase_tracker"] = vulkan_phase_tracker

    with vulkan_submit_phase(torch, SUBMIT_PHASE_MODEL_SETUP), vulkan_fallback_phase(
        torch,
        FALLBACK_PHASE_MODEL_SETUP,
    ):
        model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.eval()
        vulkan_stack_output_device_bridge_contexts = (
            create_vulkan_stack_output_device_bridge_contexts(torch, model)
            if device_kind == "vulkan" and args.vulkan_stack_output_device_bridge
            else None
        )
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
    failure_context.update(
        {
            "disable_owner_programs": disable_owner_programs,
            "vulkan_block_owner": vulkan_block_owner,
        }
    )
    if device_kind == "vulkan":
        with vulkan_submit_phase(torch, SUBMIT_PHASE_MODEL_SETUP):
            install_vulkan_fallback_phase_wrappers(torch, model)

    vulkan_stack_output_device_bridge: dict[str, Any] = {
        "enabled": False,
        "requested": bool(args.vulkan_stack_output_device_bridge),
    }
    if (
        device_kind == "vulkan"
        and args.vulkan_stack_output_device_bridge
        and not disable_owner_programs
    ):
        with vulkan_submit_phase(torch, SUBMIT_PHASE_MODEL_SETUP):
            vulkan_stack_output_device_bridge = install_vulkan_stack_output_device_bridge(
                torch,
                model,
                vulkan_stack_output_device_bridge_contexts,
            )
    elif args.vulkan_stack_output_device_bridge and disable_owner_programs:
        vulkan_stack_output_device_bridge = {
            "enabled": False,
            "requested": True,
            "reason": "disabled_by_owner_program_probe_option",
        }
    failure_context["vulkan_stack_output_device_bridge"] = (
        vulkan_stack_output_device_bridge
    )

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
    bridge = getattr(model, "_vulkan_stack_output_device_bridge", None)
    if bridge is not None:
        failure_context["stack_output_device_consumer_registrations"] = (
            bridge.registrations_for_input(image_tensor)
        )
        with vulkan_submit_phase(torch, SUBMIT_PHASE_MODEL_SETUP):
            failure_context["vulkan_stack_output_device_bridge_sanity"] = (
                validate_vulkan_stack_output_device_bridge_sanity(
                    model,
                    image_tensor,
                )
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
            failure_context["probe_summary"] = probe_summary

    if legacy_forward_output_mode == OUTPUT_MODE_READBACK:
        legacy_end_to_end_durations = list(end_to_end_with_readback_durations)
        legacy_corpus_durations = list(corpus_with_readback_durations)

    legacy_forward_durations = (
        forward_device_resident_durations
        if legacy_forward_output_mode == OUTPUT_MODE_DEVICE_RESIDENT
        else forward_with_readback_durations
    )

    debug_counters = snapshot_vulkan_debug_counters(
        torch,
        device_kind,
    )
    phase_counters = (
        vulkan_phase_tracker.summary()
        if vulkan_phase_tracker is not None
        else None
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
        "vulkan_stack_output_device_bridge": failure_context.get(
            "vulkan_stack_output_device_bridge",
        ),
        "vulkan_stack_output_device_bridge_sanity": failure_context.get(
            "vulkan_stack_output_device_bridge_sanity",
        ),
        "stack_output_device_consumer_registrations": failure_context.get(
            "stack_output_device_consumer_registrations",
        ),
        "vulkan_debug_counters": debug_counters,
        "vulkan_phase_counters": phase_counters,
        "vulkan_stack_output_device_bridge_dry_run": (
            build_stack_output_to_device_consumer_bridge_dry_run(
                debug_counters,
                phase_counters,
                failure_context.get("stack_output_device_consumer_registrations"),
            )
            if device_kind == "vulkan"
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

    out_path = output_path_from_args(args)
    write_json(out_path, result)
    print(out_path.read_text(encoding="utf-8"))
    sys.excepthook = original_excepthook


if __name__ == "__main__":
    run()
