# Vulkan Attention Fast Paths

## DAv2 FP32 query-tiled attention

The DAv2 Vulkan profile is dominated by FP32 attention. The old default path for
the real DAv2 backbone shapes is:

```
scaled_dot_product_scores_value_buffer_float
```

That shader computes one query row per workgroup and uses an online softmax loop
over source tokens while directly accumulating the value output.

The query-tiled path is enabled only with:

```
PYTORCH_VULKAN_ATTENTION_QTILE=1
```

It routes safe head64 FP32 direct-buffer attention to the existing query-4 shader:

```
scaled_dot_product_scores_value_buffer_float_head64_q4
```

Eligibility:

- Vulkan query/key/value tensors
- rank-3 buffer-direct tensors
- FP32 dtype
- head_dim == 64 and value_dim == 64
- target_len >= 128 and source_len >= 128
- no mask, no dropout, non-causal path

The qtile shader preserves the existing online-softmax source-token order. It
groups four query rows per workgroup so K/V loads are reused across a small query
tile. Logical output shape is unchanged.

## Validation

Focused test:

```
python test/test_vulkan.py -k test_dinov2_attention_qtile_matches_numpy_reference
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

## Remaining limitations

- The qtile path is disabled by default.
- Query tile sizes 2 and 8 are not added yet; the current validated variant is
  query tile 4.
- Masked, causal, dropout, BF16, and KV-cache attention remain on existing paths.
