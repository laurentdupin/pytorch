# Vulkan Vision Owner Paths

## DAv2 One-Block Owner

The DAv2 benchmark can route one transformer block through
`vulkan_prepack::run_vision_backbone_block_context`. This backend-owned block
sees the qkv, proj, fc1, and fc2 packed contexts together while preserving the
current eager Vulkan math.

The first blocker for making this owner path clean was not the MLP handoff. It
was `aten::zero_` on byte buffers created during packed context setup. Those
zeros previously fell back to CPU because the Vulkan zero path only supported
float buffers. The byte-buffer shader zero path removes that fallback for the
owner setup tensors.

The owner path remains gated by the benchmark/runtime flags and should stay
separate from MLP scratch work. The next safe step after zero fallback is gone is
to implement the owned `fc1 -> GELU(out scratch) -> fc2` handoff inside this
owner, without aliasing clone outputs.

## Fallback Phase Accounting

The benchmark now reports fallback phase counters:

- `unknown`
- `model_setup`
- `owner_context_create`
- `owner_forward`
- `decoder_setup`
- `positional_embedding_setup`
- `readback`
- `test_harness`

For one-block DAv2 owner runs, context creation is cached outside the block
forward. A short `warmup=1`, `repeats=3` run produced one owner context create
and repeated cache hits during forward execution. The owner block forward itself
had zero fallback. Remaining fallback in that run was setup or non-owner work:

- owner context unpack readbacks during `owner_context_create`
- positional embedding `view` materialization during `positional_embedding_setup`
- one patch-embed convolution weight materialization sync in `unknown`

The owner MLP body already uses `run_linear_gelu_context` before `fc2`, so it has
an owner-local fused `fc1 + GELU` handoff available once the block owner is
entered. A separate GELU-out scratch path should only be added after measuring
copy traffic with the existing owner fused MLP path enabled.
