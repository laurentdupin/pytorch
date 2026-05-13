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
