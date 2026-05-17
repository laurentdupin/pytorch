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
stride2, and patch embedding. No single class is dominant enough to justify a
new canonical kernel in this pass. The safe result is classification only.

## Current Canonical Routes

- 1x1 pointwise FP32 buffer conv uses `conv2d_buffer_float_1x1` when the
  existing route policy accepts the shape and layout.
- 3x3 stride1 padding1 FP32 buffer conv uses `conv2d_buffer_float_3x3_s1p1`.
- Patch embedding and decoder/head cases still use the generic
  `conv2d_buffer_float` route.
- There is no runtime env selector for choosing competing conv implementations.

## Next Target

The next conv optimization should start with the decoder/head and large
pointwise overlap rather than adding a broad conv alternative. In the measured
profile, the largest individual shapes are small-spatial, high-channel decoder
convs around `[1,384,37,56/57]`. A useful follow-up should decide whether those
are best handled by a canonical specialized decoder conv or by making the
validated 1x1 pointwise path accept the currently rejected direct-buffer cases.
