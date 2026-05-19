# Vulkan Linear Fast Paths

DAv2 vision backbones commonly run linear layers over 2073 token rows. That row
count is not a multiple of the 16-row cooperative matrix tile, so requiring M
alignment prevents qkv, projection, fc1, and fc2 linears from using the BF16
cooperative path even when K and N are otherwise compatible.

Per-op materialized row padding is still forbidden. Padding token rows locally in
linear would change the visible sequence unless all later attention, residual,
normalization, and softmax paths also mask or trim the padded rows correctly.

The BF16 buffer path therefore keeps the public output shape logical. For M-tail
shapes it dispatches the aligned M prefix through the existing cooperative matrix
shader, then dispatches a scalar BF16 tail shader for rows `row >= aligned_m`.
The tail shader writes only rows below the logical M and columns below logical N.

This first version still requires K and N to be 16-aligned for the cooperative
prefix. Shapes that fail K or N alignment use the existing scalar BF16 buffer
path. The plan counters exposed through `vulkan_prepack::linear_plan_counters`
record cooperative hits, tail-M cooperative hits, and K/N/capability rejections
for benchmark diagnostics.

Focused C++ coverage includes M=17 and M=2073 BF16 prepacked linear cases. Both
compare against CPU using BF16-rounded inputs, weights, and bias while asserting
the logical output shape and checking that queue-idle counters do not increase.

## FP32 DAv2 role profile

The DAv2 all-owner Vulkan benchmark now exposes
`vulkan_prepack::linear_aggregate_snapshot()` for diagnostic role and shape
classification. It records the selected kernel, owner role, M/K/N, dtype,
direct-buffer flags, packed-weight state, and estimated input/weight/output
bytes. The snapshot is diagnostic-only and does not affect route selection.

The canonical profile after attention and conv cleanup showed that FP32 linear/mm
time is spread across the four owner block roles rather than concentrated in a
single shape:

```
role       gpu_ms   share_of_linear
fc1_gelu   710.212  29.05%
fc2        582.000  23.81%
proj       582.000  23.81%
qkv        570.210  23.33%
```

The largest role/shape entry was `fc1_gelu` with `M=2073 K=384 N=1536`, but it
accounted for only 15.56% of linear/mm GPU time. The matching `M=2110` fc1_gelu
shape accounted for another 13.49%. All real DAv2 owner linears in this profile
were FP32, direct-buffer input/output, and packed-weight buffer cases.

Because no single role or shape clearly dominated, no FP32 tiled replacement was
merged in this pass. A future linear kernel should either cover a broader
validated owner class, such as both `fc1_gelu` shapes together, or be evaluated
only after a new profile shows a single role-specific target large enough to
justify a canonical replacement.
