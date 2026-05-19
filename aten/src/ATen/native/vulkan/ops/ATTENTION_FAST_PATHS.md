# Vulkan Attention Fast Paths

## DAv2 FP32 query-tiled attention

The DAv2 Vulkan profile is dominated by FP32 attention. The old default path for
the real DAv2 backbone shapes is:

```
scaled_dot_product_scores_value_buffer_float
```

That shader computes one query row per workgroup and uses an online softmax loop
over source tokens while directly accumulating the value output.

The query-tiled path is canonical for the strict safe shape class. Unsupported
shapes and capabilities use the existing generic safe paths.

It routes safe head64 FP32 direct-buffer attention to the canonical query-4
path:

```
scaled_dot_product_scores_value_buffer_float_head64_q4_subgroup
scaled_dot_product_scores_value_buffer_float_head64_q4
```

The subgroup shader is used only when the adapter supports full compute
subgroups and a required subgroup size of 64 for compute pipelines. Devices
without that capability use the shared-memory q4 shader for the same logical
path.

Query tile 2 and query tile 8 were benchmarked and rejected. They are not
reachable from production routing, and there is no runtime env selector for
competing qtile variants.

Eligibility:

- Vulkan query/key/value tensors
- rank-3 buffer-direct tensors
- FP32 dtype
- head_dim == 64 and value_dim == 64
- target_len >= 128 and source_len >= 128
- no mask, no dropout, non-causal path

The qtile shader preserves the existing online-softmax source-token order. It
groups four query rows per workgroup so K/V loads are reused across a small query
tile. Logical output shape is unchanged. Tail query rows are guarded, so the
target length does not need to be divisible by four.

## Validation

Focused tests:

```
python test/test_vulkan.py -k test_dinov2_attention_qtile
```

The test covers the DINOv2/DAv2-style shape:

```
batch=1, heads=6, target_len=601, source_len=601, head_dim=64, value_dim=64
```

The full DAv2 run also routes the real shapes:

```
batch_heads=6 target_len=2073 source_len=2073 head_dim=64 value_dim=64
batch_heads=6 target_len=2110 source_len=2110 head_dim=64 value_dim=64
```

In the instrumented run, attention GPU timestamp time changed from:

```
scaled_dot_product_scores_value_buffer_float: 27846.035 ms
```

to:

```
scaled_dot_product_scores_value_buffer_float_head64_q4: 6038.139 ms
```

One-image CPU vs Vulkan accuracy stayed in the current Vulkan band:

```
qtile off raw MAE: 0.0022994170
qtile on  raw MAE: 0.0022993390
```

No CPU fallback or queue idle was observed.

The 2026-05 all-block-owner DAv2 comparison used `warmup=3`, `repeats=10`,
input size 518, and `--skip-output-copy` with no timestamp logging:

```
variant  device_resident_mean  median    p90      qtile_hits
q4       0.3560s               0.3555s   0.3587s  1104
q8       0.3614s               0.3606s   0.3655s  1104
q2       0.3867s               0.3860s   0.3902s  1104
```

A separate short timestamp run (`warmup=1`, `repeats=3`) reported:

```
variant  attention_us  attention_share  conv_share  linear_mm_share  copy_share
q4       5703311.8     38.78%           25.83%      13.38%          0.72%
q8       5939673.6     39.74%           25.35%      13.21%          0.71%
q2       7398812.7     44.78%           23.11%      12.14%          0.64%
```

Timestamp logging increments queue-idle counters because it resets query pools,
so those runs are not used for the no-timestamp queue-idle acceptance check.

One-image CPU vs all-owner Vulkan accuracy stayed in the current Vulkan band:

```
variant  raw_mae     normalized_mae  correlation  max_abs    NaN/Inf
q4       0.00229951  0.00022704     0.99999916   0.0276322  0/0
q8       0.00229951  0.00022704     0.99999916   0.0276322  0/0
q2       0.00229948  0.00022704     0.99999916   0.0276318  0/0
```

Query tile 8 matched query tile 4 exactly on the demo image. Query tile 2
differed from query tile 4 by MAE 4.53e-7 and max abs 2.03e-5.

Because q4 was fastest, q2/q8 shader files and runtime routing were removed
after the benchmark. Future attention experiments should be evaluated as
replacement candidates and merged only if they become the canonical path for
this shape/capability class.

## Subgroup q4 replacement

The q4 shared-memory reduction was compared with a subgroup q4 replacement that
keeps the same online-softmax order, score scaling, query tile, head64/value64
shape, and output indexing. On the measured adapter, capability reporting was:

```
has_compute_full_subgroups=1
min_subgroup_size=32
max_subgroup_size=64
required_subgroup_size_stages=32
supports required subgroup size 64 for compute=yes
```

The subgroup q4 candidate is selected only under that capability check and sets
`required_subgroup_size=64` plus full-subgroup pipeline creation. There is no
runtime selector between shared and subgroup q4.

All-owner DAv2 comparison:

```
path              device_resident_mean  median    p90       attention_us  attention_share
q4 shared         0.3665s               0.3612s   0.3871s   5617144.3     37.76%
q4 subgroup       0.3444s               0.3370s   0.3674s   4229888.7     31.49%
```

One-image CPU vs all-owner Vulkan subgroup accuracy:

```
raw_mae     normalized_mae  correlation  max_abs     NaN/Inf
0.00229943  0.00022703     0.99999916   0.0276284   0/0
```

The q4 subgroup shader was also tested with an elected-lane broadcast update for
the online-softmax scalar state. That candidate computed row max, denominator,
and exp scales in one subgroup lane and broadcast the scales to the other lanes.
It preserved the same source-token order and passed the focused qtile attention
test, but it regressed no-timestamp DAv2 device-resident median from 0.3315s to
0.7552s on the measured adapter. The broadcast candidate was rejected, and the
canonical q4 subgroup shader remains the per-lane scalar update version.

## Source-blocked q4 candidate

A source-blocked q4 candidate was evaluated as a replacement for the canonical
large-sequence DAv2 FP32 head64 path. The candidate split the source-token
dimension into blocks, computed block-local max/denominator/unnormalized value
partials in pass 1, and merged those summaries in pass 2 with exact softmax
merge math. It did not use approximate softmax and preserved the same non-causal,
no-mask, no-dropout semantics.

Estimated temporary storage for the DAv2 vits attention shapes was:

```
target/source  source_block  source_blocks  temp_mib
2073           64            33             103.49
2073           128           17             53.31
2073           256           9              28.22
2073           512           5              15.68
2110           64            33             105.28
2110           128           17             54.24
2110           256           9              28.71
2110           512           5              15.95
```

The implemented candidate used source_block=256. It was functionally viable but
slower on the measured adapter: no-timestamp DAv2 device-resident median changed
from 0.3107s to 0.3210s. The timestamp profile showed pass 1 plus pass 2 at
roughly 3.91s total GPU time, compared with 4.37s for the canonical q4 subgroup
kernel, but the extra dispatches, temporary storage traffic, and full-model
timing did not justify replacing the canonical path. The candidate route and
shaders were removed.

## Stack-owned direct q4 attention

The DAv2 stack owner uses the same canonical q4 subgroup/shared attention
implementation, but it now calls the runtime attention program directly from
q/k/v while the stack phase is known. This bypasses the generic decomposed
attention carrier tensors that existed only to bridge eager
`matmul -> softmax -> matmul` into fused attention:

```
aten::decomposed_attention_bridge.scores
aten::decomposed_attention_bridge.softmax
```

The generic bridge remains available outside the stack owner. Inside the stack
owner, supported FP32 head64/value64 direct-buffer attention is a first-class
phase and records:

```
vulkan_prepack::vision_stack_attention_direct
```

The post-change stack diagnostic run showed 1104 direct stack attention hits and
1104 decomposed placeholder bypasses. Stack allocation rows for `[6,2073,2073]`
and `[6,2110,2110]` attention temps dropped to zero, while q4 attention plan
selection remained unchanged. Timed fallback stayed zero and queue idle stayed
zero in the no-timestamp run.

## Remaining limitations

- Query tile 4 is the only production qtile variant for the validated FP32
  head64 path. On devices that support required subgroup size 64 for compute,
  q4 uses subgroup reduction; otherwise it uses the shared-memory q4 shader.
- Masked, causal, dropout, BF16, and KV-cache attention remain on existing paths.
