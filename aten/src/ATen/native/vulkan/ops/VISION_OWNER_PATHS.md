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

The owner path is now canonical for the supported DAv2 Vulkan benchmark route.
It stays separate from local clone aliasing and MLP scratch experiments; the
existing owner-local `run_linear_gelu_context` handoff removes the measured MLP
clone traffic without a separate GELU-out scratch path.

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

The DAv2 Vulkan benchmark routes all 12 transformer blocks through
`run_vision_backbone_block_context`. The all-block owner is canonical for the
supported DAv2 Vulkan benchmark path; there is no long-lived performance env
selector for disabling it or limiting owner coverage.

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

After removing the remaining performance env selectors, a fresh canonical run
with `warmup=3`, `repeats=10`, input size 518 reported:

```
device-resident mean=0.3712s median=0.3673s p90=0.3742s
readback-inclusive mean=0.3453s median=0.3461s p90=0.3473s
owner hits=1104
linear_gelu hits=1104
q4 hits=1104
q4 subgroup hits=1104
timed fallback=0
queue_wait_idle_count=0
```

A short timestamp profile for the same canonical route reported:

```
attention=4184122.3 us (31.72%)
conv=3779737.6 us (28.65%)
linear/mm=2672533.3 us (20.26%)
copy=103498.9 us (0.78%)
```

Attention remains the largest bucket, but convolution is close enough that the
next optimization should start with conv classification and profiling rather
than another qtile variant.

The first conv classification pass added a diagnostic-only aggregate snapshot
and kept routing unchanged. The timestamped all-owner DAv2 profile split conv
GPU time across several classes:

```
pointwise_1x1   1392.572 ms (37.04% of conv)
conv_3x3_s1p1   1117.084 ms (29.71% of conv)
decoder_or_head  980.358 ms (26.07% of conv)
patch_embed      269.988 ms (7.18% of conv)
```

Because no single conv class clearly dominated, no canonical conv kernel change
was made in that pass. The result is documented in `CONV_FAST_PATHS.md`; the
next conv implementation should target the decoder/head and large pointwise
overlap only after that class is isolated more narrowly.

The refined decoder/head split selected the 3x3 stride2 padding1 class as the
first canonical conv route change. The existing
`conv2d_buffer_float_3x3_s2p1` shader is now selected for the validated
384-channel FP32 buffer decoder/head shape class. In the timestamped DAv2 run,
the target class dropped from 1051.482 ms to 159.905 ms while timed fallback
stayed at zero. The remaining conv work is now mostly 3x3 stride1 padding1,
pointwise 1x1, and patch embedding.

The next post-s2p1 conv profile found no equally safe routing fix. The largest
remaining conv class, `other_3x3_s1p1`, is already on the canonical specialized
3x3 stride1 padding1 shader. The 52 pointwise generic fallthroughs are
`KnownBadLargePointwiseConv` cases with non-direct input and output layouts and
`input_offset=384`, so they are not safe to route into the direct-buffer 1x1
shader. That pass only refined diagnostics and kept routing unchanged.

One-image accuracy stayed in the same Vulkan band:

```
CPU vs eager raw MAE=0.00229934 normalized MAE=0.00139754 max_abs=0.0276299
CPU vs owner raw MAE=0.00229951 normalized MAE=0.00139764 max_abs=0.0276322
eager vs owner MAE=4.07e-7 max_abs=2.00e-5
NaN/Inf=0
shape=(1362, 2048)
```
