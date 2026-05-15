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

## Eager Versus Owner Copy Profile

The 2026-05 owner comparison measured eager Vulkan against the owner path using
copy aggregate and clone-requirement snapshots. The owner path records
`vision_owner_block` rows with the block index, token count, hidden size, head
configuration, and whether `run_linear_gelu_context` and the following fc2
context were used.

With owner disabled, the stable DAv2 profile produced 1104 `[1,T,1536]`
`fc2_input_preparation` clone requirements, totaling 14.13 GB. Routing only
block 0 through `run_vision_backbone_block_context` reduced that to 1012 clones
and 12.95 GB. The difference, 92 clones and 1.18 GB, matches one block's
measured forward activity. The owner MLP counters also showed 92
`linear_gelu_hit` and 92 `fc2_after_linear_gelu_hit` events.

Scaling the owner limit showed the same proportional reduction:

```
owner_limit=1    [1,T,1536] clone bytes=12.95 GB
owner_limit=2    [1,T,1536] clone bytes=7.19 GB
owner_limit=4    [1,T,1536] clone bytes=5.75 GB
owner_limit=all  [1,T,1536] clone bytes=0
```

The all-block owner path produced 1104 `linear_gelu_hit` events and removed the
measured MLP GELU clone traffic. A separate GELU-out scratch path is therefore
not needed for this stage; the existing owner-local `run_linear_gelu_context`
route already removes the intended copy class.

The measured all-block owner output matched eager Vulkan on the DAv2 demo image:

```
MAE=4.07e-7
max_abs=2.01e-5
NaN/Inf=0
shape=(1362, 2048)
```

Runs with GPU timestamp logging enabled increment `queue_wait_idle_count` because
the timestamp profiler resets query pools. The same all-block owner run without
`PYTORCH_VULKAN_GPU_TIMESTAMP_LOG` had `queue_wait_idle_count=0`.
