# Vulkan Attention Fast Paths

## DAv2 FP32 query-tiled attention

The DAv2 Vulkan profile is dominated by FP32 attention. The old default path for
the real DAv2 backbone shapes is:

```
scaled_dot_product_scores_value_buffer_float
```

That shader computes one query row per workgroup and uses an online softmax loop
over source tokens while directly accumulating the value output.

The query-tiled path is default-on for the strict safe shape class. It can be
disabled with:

```
PYTORCH_VULKAN_ATTENTION_QTILE=0
```

It routes safe head64 FP32 direct-buffer attention to the query-tiled shaders:

```
scaled_dot_product_scores_value_buffer_float_head64_q2
scaled_dot_product_scores_value_buffer_float_head64_q4
scaled_dot_product_scores_value_buffer_float_head64_q8
```

The selected tile can be forced with:

```
PYTORCH_VULKAN_ATTENTION_QTILE_VARIANT=<2|4|8|auto>
```

The default is `auto`. After the 2026-05 DAv2 all-owner benchmark, `auto`
remains conservative and selects query tile 4 for the current FP32 head64
shape. Query tile 8 and query tile 2 are available for explicit benchmarking.

Eligibility:

- Vulkan query/key/value tensors
- rank-3 buffer-direct tensors
- FP32 dtype
- head_dim == 64 and value_dim == 64
- target_len >= 128 and source_len >= 128
- no mask, no dropout, non-causal path

The qtile shaders preserve the existing online-softmax source-token order. They
group 2, 4, or 8 query rows per workgroup so K/V loads are reused across a small
query tile. Logical output shape is unchanged. Tail query rows are guarded, so
the target length does not need to be divisible by the tile size.

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

## Remaining limitations

- Query tile 4 remains the default selected variant because q8 was slightly
  slower on the measured all-owner DAv2 profile and q2 was clearly slower.
- Masked, causal, dropout, BF16, and KV-cache attention remain on existing paths.
