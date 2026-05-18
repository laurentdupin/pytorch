# Vulkan Convolution Fast Paths

## DAv2 Canonical Conv Profile

The DAv2 Vulkan benchmark now records convolution aggregate rows through
`vulkan_prepack::conv_aggregate_snapshot()`. The aggregate is diagnostic-only:
it does not select routes and does not add performance env gates. Rows include
the selected method, kernel name, role, input and weight shape, stride, padding,
dilation, direct-buffer flags, count, and estimated input/output/weight bytes.

The first canonical all-owner conv profile after q4 subgroup attention used
DAv2 vits, input size 518, `warmup=1`, `repeats=3`, and timestamp logging. The
top conv GPU kernels were:

```
conv2d_buffer_float          1629.445 ms
conv2d_buffer_float_3x3_s1p1 1117.084 ms
conv2d_buffer_float_1x1      1013.472 ms
```

Joining `aten::convolution.submit` rows with timestamp rows classified the conv
bucket by role:

```
pointwise_1x1   1392.572 ms (37.04% of conv)
conv_3x3_s1p1   1117.084 ms (29.71% of conv)
decoder_or_head  980.358 ms (26.07% of conv)
patch_embed      269.988 ms (7.18% of conv)
```

The top individual conv shapes by GPU time were:

```
[1,384,37,56] -> [1,384,19,28], weight [384,384,3,3], stride 2, padding 1
[1,384,37,57] -> [1,384,19,29], weight [384,384,3,3], stride 2, padding 1
[1,384,37,56] -> [1,384,37,56], weight [384,384,1,1], stride 1, padding 0
[1,3,518,784] -> [1,384,37,56], weight [384,3,14,14], stride 14, padding 0
[1,384,37,57] -> [1,384,37,57], weight [384,384,1,1], stride 1, padding 0
```

The conv work is split across pointwise, 3x3 stride1 padding1, decoder/head
stride2, and patch embedding. The follow-up refined profile split the
decoder/head overlap into explicit roles:

```
other_3x3_s1p1              1165.987 ms (29.03% of conv)
decoder_head_3x3_s2p1       1051.482 ms (26.18% of conv)
other_pointwise_1x1          776.602 ms (19.33% of conv)
decoder_head_pointwise_1x1   674.670 ms (16.80% of conv)
patch_embed                  348.118 ms (8.67% of conv)
```

The largest isolated actionable class was decoder/head 3x3 stride2 padding1,
not pointwise 1x1. Pointwise route counters in the same run reported:

```
total_1x1=504
specialized_1x1_hit=452
generic_1x1_hit=52
reject_not_direct_buffer=52
```

The 1x1 generic fallthrough is smaller and tied to non-direct-buffer decoder
layouts, so the canonical change selected here is the existing 3x3 stride2
padding1 buffer shader for the validated 384-channel decoder/head shape class.

## Current Canonical Routes

- 1x1 pointwise FP32 buffer conv uses `conv2d_buffer_float_1x1` when the
  existing route policy accepts the shape and layout.
- 3x3 stride1 padding1 FP32 buffer conv uses `conv2d_buffer_float_3x3_s1p1`.
- 3x3 stride2 padding1 FP32 buffer conv with 384 input and output channels uses
  `conv2d_buffer_float_3x3_s2p1`.
- Patch embedding and remaining decoder/head pointwise fallthrough cases still
  use the generic `conv2d_buffer_float` route when their layout is not accepted
  by an existing specialized route.
- There is no runtime env selector for choosing competing conv implementations.

## Decoder 3x3 Stride2 Padding1

The decoder/head 3x3 stride2 padding1 path did not need a new shader. The
`conv2d_buffer_float_3x3_s2p1` shader, dispatch, and op-hit label already
existed, but the route selector never chose it. The selector now sends the
validated FP32 buffer class to that shader when:

```
groups == 1
kernel == 3x3
stride == 2
padding == 1
dilation == 1
input channels == output channels == 384
```

The exact DAv2 shapes tested are:

```
[1,384,37,56] -> [1,384,19,28], weight [384,384,3,3]
[1,384,37,57] -> [1,384,19,29], weight [384,384,3,3]
```

The focused test `test_vulkan_conv2d_decoder_3x3_s2p1_uses_specialized_path`
checks CPU equivalence for both shapes and asserts that the conv aggregate sees
`kernel=conv2d_buffer_float_3x3_s2p1`.

The timestamped DAv2 profile moved the target class from the generic kernel to
the specialized kernel:

```
before decoder_head_3x3_s2p1 GPU time=1051.482 ms
after  decoder_head_3x3_s2p1 GPU time=159.905 ms
```

Total conv GPU time in that short timestamp profile changed from about
4016.9 ms to 3183.5 ms. The same run still reported zero timed fallback. The
no-timestamp run reported queue_wait_idle_count=0.

One-image CPU vs Vulkan accuracy remained in the existing Vulkan band:

```
raw MAE=0.00229957
normalized MAE=0.00030728
correlation=1.0
max_abs_error=0.0276370
NaN/Inf=0
shape=(1362, 2048)
CPU fallback=0
queue_wait_idle_count=0
```

## Next Target

After this route change, the remaining conv time is led by 3x3 stride1
padding1, pointwise 1x1, and patch embedding. A fresh post-s2p1 canonical
profile with DAv2 vits, input size 518, `warmup=1`, `repeats=3`, and timestamp
logging reported:

```
attention                         4355.759 ms (33.55%)
conv2d_buffer_float_3x3_s1p1      1164.632 ms (8.97%)
conv2d_buffer_float_1x1           1056.069 ms (8.14%)
conv2d_buffer_float                665.596 ms (5.13%)
conv2d_buffer_float_3x3_s2p1       154.328 ms (1.19%)
```

Within the conv bucket, the role split was:

```
other_3x3_s1p1              1164.632 ms (38.30% of conv)
other_pointwise_1x1          775.835 ms (25.52% of conv)
decoder_head_pointwise_1x1   672.057 ms (22.10% of conv)
patch_embed                  273.773 ms (9.00% of conv)
decoder_head_3x3_s2p1        154.328 ms (5.08% of conv)
```

The largest remaining conv class is already using the canonical specialized
`conv2d_buffer_float_3x3_s1p1` path. No safe one-line routing fix exists for
that class; improving it would require shader work and should be benchmarked as
a replacement candidate, not added as a second runtime route.

The 1x1 fallthroughs were refined with pointwise route counters and log rows.
The short timestamp profile reported:

```
total_1x1=504
specialized_1x1_hit=452
generic_1x1_hit=52
reject_not_direct_buffer=52
reject_input_not_buffer=0
reject_input_not_direct_buffer=52
reject_output_not_direct_buffer=0
reject_storage_offset=0
```

The generic 1x1 rows are the known bad decoder layouts:

```
input=[1,384,37,57]
output=[1,192,37,57] or [1,384,37,57]
input_direct=0
output_direct=0
input_offset=384
reject=KnownBadLargePointwiseConv
```

Because both input and output are non-direct layouts, routing these tensors into
the direct-buffer 1x1 shader would be unsafe without a separate proven layout
normalization or a non-direct 1x1 shader. This pass therefore keeps production
routing unchanged after the s2p1 fix and records the classification result.
