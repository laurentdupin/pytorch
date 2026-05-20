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

The later q4 subgroup broadcast experiment did not replace the canonical
attention shader. The candidate moved duplicated online-softmax scalar updates
to an elected subgroup lane and broadcast the scales, but it regressed
device-resident DAv2 median latency on the measured adapter. The owner path
therefore continues to use the existing q4 subgroup shader on subgroup64-capable
adapters and the shared-memory q4 shader otherwise.

A source-blocked q4 attention experiment was also evaluated after conv routing
cleanup. It split the source-token loop into block-local pass 1 summaries and a
pass 2 exact softmax merge. For source_block=256, the DAv2 vits temporary storage
requirement was about 28.2 MiB to 28.7 MiB per attention call shape. The
candidate reduced the attention timestamp kernel total in isolation, but
regressed no-timestamp device-resident median from 0.3107s to 0.3210s because
the extra dispatches and temporary buffer traffic outweighed the source
parallelism. The source-blocked route and shaders were removed, so the owner path
continues to use canonical q4 subgroup attention where capability-safe.

After the local attention experiments and conv routing cleanup, FP32 linear/mm
was profiled by owner role and M/K/N shape. The profile did not identify a
single dominant canonical target: fc1_gelu was the largest role at 710.212 ms
and 29.05% of linear/mm time, while fc2, projection, and qkv were each about
23-24%. The largest individual role/shape was `fc1_gelu M=2073 K=384 N=1536`
at 380.470 ms, only 15.56% of linear/mm time. No FP32 tiled replacement was
merged from that profile; the useful next linear work would need to cover a
broader owner class, not just one isolated shape.

## Diagnostic Stack Owner

The DAv2 Vulkan benchmark now creates one
`VisionBackboneStackContext` containing the 12 canonical block contexts and
enters `vulkan_prepack::run_vision_backbone_stack_context` from the Python
benchmark bridge. The first stack owner is intentionally a safe diagnostic
boundary: it sequences the already validated block owner in C++ and captures the
requested intermediate block outputs for DAv2, but it does not use replay or
compiled-session capture.

Existing stack replay and compiled-session bridges remain separate experimental
program paths. The canonical benchmark stack owner rejects execution under an
active runtime capture label so it cannot accidentally create unsafe nested
replay. Stack owner log rows report:

```
vision_stack_owner selected=1 blocks=12 tokens=2073 hidden=384 heads=6 head_dim=64 mlp_hidden=1536 owner_forward_fallback=0 stack_contexts=12 uses_program=0 uses_replay=0 unsafe_nested_replay=0
```

Additional counters are exposed through
`vision_stack_owner_counters()`:

```
total_attempts
stack_owner_hit
block_context_count
block_execute_count
reject_missing_context
reject_shape
reject_dtype
reject_layout
reject_unsafe_replay
```

The benchmark also snapshots these counters. A `warmup=3`, `repeats=10`,
device-resident DAv2 vits run produced:

```
stack_owner_hit=92
block_context_count=1104
block_execute_count=1104
owner_forward fallback=0
timed fallback=0
queue_wait_idle_count=0
```

The same run reported device-resident forward median 0.3423s. Because this
first stack owner records the same kernels as the per-block owner, the useful
result is the ownership structure: Python enters one stack-level owner call per
backbone pass while all 12 block contexts are visible together at the backend
boundary.

`sync_counters()` keeps its existing fields and appends diagnostic
`compute_dispatch_count` and `submit_compute_job_count` at the end. These
counters are diagnostic-only and do not affect route selection.

## Stack Lifetime And Dispatch Diagnostics

The stack owner now labels internal phases while it sequences the existing block
owner:

```
stack_entry
block_entry
norm1
qkv_linear
qkv_transform
attention
proj_linear
residual1
norm2
fc1_gelu
fc2
residual2
intermediate_capture
stack_exit
```

Two diagnostic-only snapshots are exposed:

```
vulkan_prepack::stack_allocation_aggregate_snapshot()
vulkan_prepack::reset_stack_allocation_aggregate()
vulkan_prepack::stack_dispatch_aggregate_snapshot()
vulkan_prepack::reset_stack_dispatch_aggregate()
vulkan_prepack::stack_attention_counters()
vulkan_prepack::reset_stack_attention_counters()
```

`stack_allocation_aggregate_snapshot()` records stack-phase allocations created
through Vulkan buffer tensor helpers while a stack phase is active. Rows include
phase, block index, block allocation label, shape, strides, dtype, storage
layout flags, and lifetime classification. The lifetime classes currently used
by the stack owner are:

```
internal_temp
requested_intermediate_output
final_stack_output
```

`stack_dispatch_aggregate_snapshot()` records compute dispatches by stack phase,
block index, and shader name. This is also diagnostic-only and does not choose a
route.

A canonical DAv2 vits stack run with `warmup=3`, `repeats=10`, and
`--skip-output-copy` reported:

```
device-resident mean=0.3220s median=0.3212s p90=0.3245s min=0.3199s max=0.3267s
stream_submit_count=4681
compute_dispatch_count=23237
submit_compute_job_count=23237
timed_fallback=0
queue_wait_idle_count=0
CPU fallback count=1 setup-only
```

The allocation profile was dominated by internal attention-phase buffers:

```
shape=[6,2073,2073] lifetime=internal_temp bytes_per_stack=1326.0 MB
shape=[6,2110,2110] lifetime=internal_temp bytes_per_stack=1190.6 MB
shape=[6,2073,64]   lifetime=internal_temp bytes_per_stack=20.5 MB
shape=[6,2110,64]   lifetime=internal_temp bytes_per_stack=18.1 MB
shape=[1,2073,384]  lifetime=requested/final output bytes_per_stack=6.8 MB
shape=[1,2110,384]  lifetime=requested/final output bytes_per_stack=6.0 MB
```

The phase split was:

```
attention internal_temp bytes_per_stack=2555.2 MB
intermediate_capture escaping bytes_per_stack=12.8 MB
```

The dispatch profile was:

```
attention      2064 dispatches, 36.9 per stack
residual1      1344 dispatches, 24.0 per stack
residual2      1344 dispatches, 24.0 per stack
fc1_gelu        816 dispatches, 14.6 per stack
fc2             816 dispatches, 14.6 per stack
proj_linear     816 dispatches, 14.6 per stack
qkv_linear      816 dispatches, 14.6 per stack
norm1           672 dispatches, 12.0 per stack
norm2           672 dispatches, 12.0 per stack
qkv_transform   672 dispatches, 12.0 per stack
```

Top shaders by dispatch count were:

```
buffer_add                                               1400
mm_buffer_float_bias                                     1344
native_layer_norm_width_buffer_float                     1344
buffer_mul                                               1344
buffer_to_buffer                                          672
merge_attention_heads_buffer                              672
scaled_dot_product_scores_value_buffer_float_head64_q4_subgroup 672
mm_buffer_float_bias_gelu                                 672
mm_buffer_float                                           672
transform_bias_rescale_qkv_buffer                         672
```

The scratch decision from this profile is deliberately conservative. Internal
temporary allocation pressure is material, but the largest reusable class is
attention scratch (`[6,T,T]` scores/probability buffers) owned inside the
attention implementation. Reusing it safely from the stack requires a first-class
attention scratch/out interface or stack-level program scratch arena ownership.
No stack scratch slot was merged in this pass because redirecting those buffers
from the stack boundary would either require an attention-path API change or a
copy into scratch, which would defeat the purpose and risk changing visible
lifetimes. Requested intermediate and final stack outputs are small by
comparison and must continue to escape the stack.

The next program-level step should therefore make attention a first-class stack
phase, not add a generic stack scratch allocator.

## Stack-Owned Direct Attention

The stack owner now bypasses the decomposed attention carrier tensors for the
supported DAv2 q4 attention shape. The previous stack path entered the generic
runtime bridge through:

```
qkv -> q/k/v -> bmm scores -> softmax probs -> bmm value
```

Inside Vulkan, the decomposed bridge recognized this pattern and consumed the
candidate with the canonical fused q4 attention shader, but the stack allocation
profile proved that the `scores` and `probs` carrier tensors were still created:

```
aten::decomposed_attention_bridge.scores  count=672
aten::decomposed_attention_bridge.softmax count=672
shape=[6,2073,2073] lifetime=internal_temp
shape=[6,2110,2110] lifetime=internal_temp
```

Those tensors are not visible stack outputs and are not required by the q4 fused
attention shader. For stack-owner attention phases, the block owner now calls the
same canonical runtime attention program directly from q/k/v:

```
qkv -> q/k/v -> vulkan_prepack::vision_stack_attention_direct -> proj
```

The generic decomposed attention bridge remains available outside the stack owner
for eager Vulkan `matmul -> softmax -> matmul` patterns. This is not a runtime
performance selector; it is the canonical stack-owner path for the already owned
DAv2 q4 shape.

On the post-change DAv2 vits diagnostic run, the deterministic counters changed
as follows:

```
stack_attention total/direct/bypass: 1104 / 1104 / 1104
[6,T,T] stack allocation rows:       24 -> 0
attention internal temp bytes/stack: 2538.9 MB -> 38.4 MB
decomposed scores op hits:           672 -> 0
decomposed softmax op hits:          672 -> 0
vision_stack_attention_direct hits:  0 -> 672
compute dispatches:                  23237 -> 22133
submit_compute_job:                  23237 -> 22133
buffer copy bytes:                   4071.5 MB -> 538.5 MB
timed fallback:                      0 -> 0
queue_wait_idle without timestamps:  0 -> 0
```

The no-timestamp timing sample collected during this pass was intentionally not
used as a selection signal because the adapter was running another heavy
workload. The allocation and dispatch counters are deterministic and show the
intended effect: the stack owner no longer allocates `[6,T,T]` attention carrier
buffers for the supported direct q4 path.

## Stack Execution Manifest

The stack owner now exposes a diagnostic execution manifest:

```
vulkan_prepack::stack_execution_manifest()
vulkan_prepack::reset_stack_execution_manifest()
vulkan_prepack::stack_capture_readiness()
```

Rows are ordered by stack execution and include the block index, phase, op label,
kernel label, input/output shapes, dtype, allocation/write mode, escape flags,
fallback/sync/replay flags, and row-level capture safety. The manifest is
diagnostic-only and does not select an alternate route.

The DAv2 diagnostic run recorded 6528 rows with coverage for all stack phases:

```
norm1
qkv_linear
qkv_transform
attention
proj_linear
residual1
norm2
fc1_gelu
fc2
residual2
intermediate_capture
```

Capture readiness for that run was:

```
fixed_shapes=0
no_cpu_fallback=1
no_host_sync=1
no_nested_replay=1
no_active_capture=1
requested_intermediates_marked=1
internal_outputs_owned=1
known_lifetimes=1
safe_to_capture=0
```

No programmed sequence or command-buffer replay was merged. The exact blocker is
fixed-shape ownership: the current canonical DAv2 benchmark stack context sees
both `T=2073` and `T=2110`, so manifest rows are marked
`uses_dynamic_shape=1`. A safe stack execution program needs a shape-keyed plan
and resource rebinding/invalidation rules before replay can be canonical.

## Shape-Keyed Stack Plans

The stack owner now builds fixed-shape execution plans inside the same canonical
stack context. The parent context may still see multiple token lengths, but each
observed token length has its own plan:

```
plan(T=2073)
plan(T=2110)
```

The two token counts come from the DAv2 image preprocessing path. At input size
518 the benchmark images produce patch grids of `37x56` and `37x57`; with the
class token those become `2073` and `2110`. The requested intermediate set is
included in the key so plans cannot be reused across different escaping output
behavior.

The shape key includes token count, hidden size, head count, head dimension,
MLP hidden size, number of blocks, dtype, device/capability key, layout policy
version, attention policy version, owner program version, requested intermediate
mask, direct-attention flag, and q4-subgroup availability.

The diagnostic APIs are:

```
vulkan_prepack::stack_shape_plan_keys()
vulkan_prepack::stack_shape_plan_readiness()
vulkan_prepack::stack_shape_plan_counters()
vulkan_prepack::reset_stack_shape_plan_counters()
vulkan_prepack::validate_stack_shape_plan_binding(...)
```

`stack_execution_manifest()` also includes `stack_shape_plan_manifest` rows for
created plans. These rows are fixed-shape (`uses_dynamic_shape=0`) while the
global dynamic manifest rows remain useful for showing why the whole context is
not a single fixed-shape capture target.

The first plan layer does not replay command buffers and does not change kernel
execution. It validates runtime resource binding against the plan and falls back
to the existing safe stack owner if a binding is invalid. Current invalidation
reasons include token, hidden, dtype, requested-intermediate, capability, and
context-identity changes.

The next program-level step is to use the fixed-shape plan as the canonical
scheduler for a programmed sequence, then consider command-buffer capture only
after runtime resource rebinding is proven for each shape key.

## Stack Replay Readiness

The stack owner now exposes a resource binding manifest for each shape plan:

```
vulkan_prepack::stack_resource_binding_manifest()
vulkan_prepack::stack_replay_readiness()
vulkan_prepack::stack_replay_binding_mode()
vulkan_prepack::stack_replay_counters()
```

The manifest classifies runtime inputs, requested intermediate outputs,
internal activations, q/k/v attention buffers, attention outputs, linear and
residual outputs, packed linear weights, biases, and norm parameters. Persistent
weights and norm parameters are stable. Runtime inputs, escaping outputs, and
internal temporaries are runtime-varying resources.

Replay is not enabled. The current command recording path binds concrete
descriptor sets during each compute submission. The shape plan now carries a
descriptor table for planned re-recording, but it still does not own
replay-stable descriptor sets. The resulting readiness is:

```
fixed_shape_plan=1
resources_classified=1
runtime_bindings_validated=1
descriptor_table_complete=1
descriptor_indices_known=1
descriptors_rebindable=1
persistent_resources_stable=1
internal_temps_owned=1
escaping_outputs_marked=1
no_cpu_fallback=1
no_host_sync=1
no_nested_replay=1
no_queue_idle=1
command_capture_safe=0
```

The binding mode is `re_record_command_buffer_per_forward`, not command replay.
This avoids stale descriptor references. The next program-level task should use
the descriptor binding table in a planned command recording layer before
attempting canonical command-buffer replay.

## Stack Descriptor Binding Table

Shape-keyed stack plans now emit descriptor binding tables:

```
vulkan_prepack::stack_descriptor_binding_table()
vulkan_prepack::stack_descriptor_binding_validation()
```

Rows are derived from fixed plan steps and shader argument order. They mark
runtime inputs, requested intermediate outputs, runtime tensor metadata, and
internal temps as descriptor-update resources. Packed weights, biases, and norm
parameters are persistent. Descriptor set `0` and binding indices are known for
the planned stack phases, so validation can now report
`ready_for_re_record_per_forward=1`.

Command replay is still not enabled. The validation keeps
`ready_for_command_replay=0` and `command_capture_safe=0` because the existing
backend still records concrete descriptor sets per compute job and
program-owned temporaries are not stable replay resources. The next safe step is
a planned command-recording layer that uses this table to re-record each
forward, not replay of old command buffers.

The stack owner now uses planned per-forward recording when the fixed shape
plan and descriptor table validate. The earlier deadlock came from holding the
command mutex across stack execution while each compute op re-entered
`submit_compute_job`. The accepted Context API instead keeps only a stack
recording state: `begin_stack_planned_recording()` takes the command mutex
briefly, each compute job locks once and appends descriptors, barriers, and
dispatches to the current command buffer, and
`end_stack_planned_recording_and_submit()` submits once at stack exit.

This is not command replay. Commands are recorded with current runtime
input/output/internal-temp descriptors every forward, and no command buffer is
persisted across forwards. `stack_planned_recording_readiness()` now reports
`command_recording_scope_available=1`, `barriers_recordable=1`, and
`safe_to_record_stack_per_forward=1` for ready shape plans. Replay remains
blocked until program-owned temporaries become replay-stable resources or can
be safely rebound without re-recording.

Submit-origin diagnostics now classify actual queue submissions separately from
logical compute jobs. `submit_origin_counters()` reports stack planned
recording submits, normal `cmdSubmitFrequency` submits, pre/post stack flushes,
explicit synchronization, CPU readback, retire drains, profiling submits, and
unknown origins. `stack_planned_recording_counters()` also reports premature
stack submits and frequency flushes suppressed by stack recording. A healthy
stack recording has no premature stack submits, one stack submit per stack
scope, and zero unknown submit origins in the benchmark path.

The DAv2 no-timestamp submit-origin profile validates the local stack win:

```
recorded_stack_compute_jobs=15068
cmdSubmitFrequency=16
estimated_old_stack_submits=942
actual_stack_planned_submits=92
local_stack_submit_reduction=850
premature_stack_submits=0
suppressed_frequency_flushes=868
```

Total queue submits still measured 4773, matching `stream_submit_count`. The
remaining non-stack origins were dominated by `retire_queue_drain=2698`,
`explicit_synchronize=1150`, `tensor_cpu_readback=560`, and
`normal_cmd_submit_frequency=273`; `unknown=0`. The next submit-side work should
therefore classify and reduce non-stack retire/sync/readback origins before
changing stack planned recording again.

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
