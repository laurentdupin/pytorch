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

## All-Block Owner Benchmark Default

The DAv2 Vulkan benchmark now enables the block owner by default and routes all
12 transformer blocks through `run_vision_backbone_block_context`. Disable it
with:

```
PYTORCH_VULKAN_DAV2_BLOCK_OWNER=0
```

The owner limit remains available for stepwise testing:

```
PYTORCH_VULKAN_DAV2_BLOCK_OWNER_LIMIT=<N|all>
```

Before promoting the owner default, the remaining timed fallbacks were outside
the owner block:

- `vulkan_prepack::conv2d_context` patch-embed weight materialization from the
  first patch embedding convolution
- `aten::view` materialization in positional embedding setup for
  `[1, 37, 57, 384] -> [1, tokens, 384]`

Both are setup/cache work, not owner forward work. The benchmark now prewarms the
patch embedding and positional embedding setup before measured forward loops.
The positional embedding result is cached by device, dtype, input token shape,
and image size. This does not eliminate those setup fallbacks globally; it moves
their cache misses outside the timed region and reports them in phase counters.

A short all-owner run before this cache/prewarm step reported:

```
total CPU fallback=26
timed positional_embedding_setup fallback=26
timed unknown readback event=1
owner_forward fallback=0
queue_wait_idle_count=0
```

After the cache/prewarm step, the same all-owner configuration reported:

```
total CPU fallback=1
timed fallback=0
model_setup readback event=1
positional_embedding_setup fallback=1
owner_context_create readbacks=168
owner_forward fallback=0
queue_wait_idle_count=0
```

The all-owner stable benchmark on the DAv2 vits demo image used
`warmup=3`, `repeats=10`, input size 518:

```
eager device-resident mean=0.5899s median=0.5909s p90=0.6018s
owner device-resident mean=0.5405s median=0.5180s p90=0.5664s
owner readback-inclusive mean=0.5112s median=0.5118s p90=0.5163s
```

The owner run had 1104 block hits, 1104 `linear_gelu_hit` events, zero
`[1,T,1536]` MLP GELU clone requirements, zero timed fallback, and zero
queue-idle waits without timestamp logging. A separate timestamp-profiling run
recorded queue-idle waits from the profiler itself and should not be compared to
no-timestamp latency runs.

The follow-up attention variant sweep kept the all-block owner default and
compared qtile 2, 4, and 8. Query tile 4 remained the canonical production path
because it was fastest in both no-timestamp device-resident timing and
timestamped attention totals on the measured DAv2 vits run. Query tile 2 and 8
were removed from production routing after that measurement. The owner path and
fallback properties were unchanged across the experiment: 1104 owner hits, zero
timed fallback, and zero queue-idle waits without timestamp logging.

The q4 path was then tested with subgroup reduction as a replacement candidate,
not as an env-selected alternative. On adapters that support full compute
subgroups and required subgroup size 64 for compute, the canonical q4 path uses
the subgroup shader. Other adapters keep the shared-memory q4 shader for the
same supported shape class. The measured subgroup replacement reduced attention
GPU time from 5617144.3 us to 4229888.7 us while keeping the owner and fallback
properties unchanged.

One-image accuracy stayed in the same Vulkan band:

```
CPU vs eager raw MAE=0.00229934 normalized MAE=0.00139754 max_abs=0.0276299
CPU vs owner raw MAE=0.00229951 normalized MAE=0.00139764 max_abs=0.0276322
eager vs owner MAE=4.07e-7 max_abs=2.00e-5
NaN/Inf=0
shape=(1362, 2048)
```
