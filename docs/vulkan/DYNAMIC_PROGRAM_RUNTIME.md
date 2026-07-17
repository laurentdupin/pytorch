# Vulkan Dynamic Program Runtime

## Purpose

The dynamic program runtime is the path away from exact-shape admission as the
default execution model.

The old model was:

```text
exact row or finite rowset admits shape -> optimized Vulkan path
otherwise -> hard fail or fallback
```

The target model is:

```text
semantic family validates operation legality
  -> generic dynamic Vulkan program runs the actual runtime descriptor
  -> hot rows may select optimized plans
  -> unsupported semantics reject with a reason
```

Contracts remain the safety and policy layer. They decide whether an operation
is legal, whether its layout/storage semantics are supported, and whether
shader compilation or command-list generation is allowed. They should not
require every legal H/W or sequence length to have been observed before.

## Runtime Layers

### Semantic Family Validator

A validator checks operation semantics and backend limits, not exact prior
observation.

Examples:

```text
PointwiseConv1x1DirectBuffer
ElementwiseBroadcastDirectBuffer
ReductionDirectBuffer
LinearOrMatmulDirectBuffer
Conv2DDirectBuffer
PackedBufferConv2D
PatchEmbedFloatBufferConvRoute
RegionCommandList
```

Each validator must return one of:

```text
accepted
unsupported dtype/rank/layout/storage
unsupported operation semantics
requires generated shader
requires generated command list
missing validator
```

Unsupported cases fail closed with a reason. They must not silently move to CPU
fallback unless the route explicitly owns a fallback budget.

### Program Key

`DynamicProgramKey` identifies the reusable compiled/runtime program. For a
generic runtime-shape program, dynamic dimensions are not part of the key unless
they change shader code or descriptor layout.

For the first pointwise example:

```text
family = PointwiseConv1x1DirectBuffer
specialization = GenericRuntimeShape
dtype = float32
rank = input rank 4 / weight rank 4
layout = direct buffer required
bias = yes/no
```

Actual H/W is carried by tensor metadata and dispatch geometry.

### Program Admission

`DynamicProgramAdmission` is the decision record between contracts and route
code. It records:

```text
accepted
reject reason
program key
runtime-shape metadata use
whether shader compilation is allowed
whether command-list generation is allowed
```

The initial implementation authorizes:

```text
PointwiseConv1x1DirectBuffer / GenericRuntimeShape
SequenceCatDirectBuffer / GenericRuntimeShape
InitialSequenceCatDirectBuffer / GenericRuntimeShape
ElementwiseBroadcastDirectBuffer / GenericRuntimeShape
LinearOrMatmulDirectBuffer / GenericRuntimeShape
LinearGeluBridgeContract / GenericRuntimeShape
EmbeddingLookupDirectBuffer / ValidCpuIndices
EmbeddingLookupDirectBuffer / ValidVulkanIndices
FeatureMapToTokensDirectBuffer / GenericRuntimeShape
CatAxisDirectBuffer / GenericRuntimeShape
BatchNormInferenceDirectBuffer / GenericRuntimeShape
DirectDecodeGQASDPADirectBuffer / GenericRuntimeShape
DirectCausalPrefillGQASDPADirectBuffer / GenericRuntimeShape
SmallNonCausalGQASDPADirectBuffer / GenericRuntimeShape
DirectNonCausalMHASDPADirectBuffer / GenericRuntimeShape
MaskedTinySDPAContract / AdditiveFloatMaskRuntimeShape
DiffusionSDPAContract / CrossAttentionRuntimeShape
TokenPrefixCatAddDirectBuffer / GenericRuntimeShape
PatchEmbedFloatBufferConvRoute / GenericRuntimeShape
NoOverlapConvTranspose2DContract / DynamicKernelStrideFloatBuffer
```

`Conv2DDirectBuffer` is represented as a semantic target family for a future
direct-layout generic conv shader. A focused random-shape probe showed the
current `conv2d_buffer_float` path has value parity for generic groups-one,
dilation-one cases, but it records metadata-packed buffer inputs and often
metadata-packed outputs (`input_direct=0`, `output_direct=0`). Do not stamp it
as `Conv2DDirectBuffer`; the next correct migration is a packed-buffer conv
semantic family or a layout-transition contract that proves direct output
ownership.

`PackedBufferConv2D` is the semantic family for the existing metadata-packed
float-buffer conv path. The v0 production stamp is intentionally limited to
the generic `conv2d_buffer_float` branch for batch-one fp32 rank-4 buffer-backed
convs with groups `1`, dilation `[1,1]`, positive kernel/stride/spatial/channel
dimensions, non-negative padding, matching input/weight channels, and positive
computed output spatial dimensions. It uses existing runtime tensor metadata.
It does not claim batched-conv coverage, direct-buffer ownership, specialized
3x3 shader ownership, or generated shader coverage.

`LinearGeluBridgeContract` `GenericRuntimeShape` defers a Vulkan linear output
only by semantic bridge constraints: rank-2/rank-3 input flattening must match
the packed weight input dimension, output features must be positive, bias must
be present, `out=` must be absent, alpha/beta must be `1`, and the downstream
GELU approximation must be `none` or `tanh`. The old
`BackboneMlpHidden384To1536` row remains evidence, not a runtime shape gate.

`PatchEmbedFloatBufferConvRoute` is route-authorized separately from generic
conv2d. It admits fp32 Vulkan buffer patch projections with input `[1,3,H,W]`,
weight `[C,3,14,14]`, stride `[14,14]`, zero padding, dilation `[1,1]`, and
groups `1` through the existing `conv2d_buffer_float` shader. Observed DAv2 rows
remain regression evidence, but legal runtime H/W and output-channel values no
longer need exact row admission.

### Generic Runtime Program

A generic runtime program uses:

```text
descriptor bindings
tensor metadata UBOs
push/uniform params
runtime dispatch geometry
```

For pointwise 1x1, the existing `conv2d_buffer_float_1x1` shader already reads
runtime sizes and strides. Unknown legal H/W only changes metadata and
dispatch dimensions, not the shader module.

### Generated Shader and Command List Ownership

Some semantic families may eventually need generated shaders or command lists.
Those are graph-owned plan modes, not eager-route escape hatches. Graph lowering
must validate the family first, and the resulting Vulkan graph program must own
value lifetime, descriptor bindings, barriers, output storage, and command
partition execution.

The retired eager runtime-elementwise experiment proved that generated fp32
buffer add/mul math could match standalone eager results. It also proved that
retaining eager tensor handles or returning uninitialized placeholders could
not establish stack value lifetime, output ownership, or consumer order:
DAv2 bridge sanity failed with runtime-generated and static shaders, copied
input leases, and explicit materialization boundaries. The runtime glslc
compiler, owned-SPIR-V descriptors, live sidecar recorder, deferred placeholder
registry, and VisionBlocks generated-chain routes were therefore deleted.
Git is the implementation archive; this rejection result is the durable design
constraint.

## First Implemented Slice

`SmallSpatialPointwiseConvContract` `GenericDynamicHW` admits fp32 direct-buffer
1x1 convolution under semantic guards:

```text
input rank = 4
weight rank = 4
batch > 0
groups = 1
kernel = 1x1
stride = 1
padding = 0
dilation = 1
input channels = weight input channels
input channels > 0
output channels > 0
```

This slice routes to the existing dynamic-shape 1x1 buffer shader. When the
same semantic admission also proves batch-one width-packed input, weight, bias,
and output metadata, the route may select the existing as-linear plan without
requiring a sparse projection row. Sparse `SmallSpatialPointwiseConvContract`
rows remain evidence and regression fixtures rather than runtime legality gates
for unseen H/W.

Exact sparse pointwise rows remain useful as:

```text
performance evidence
optimized-plan seeds
regression fixtures
negative guard examples
```

They are no longer the required mechanism for every unseen legal H/W in this
family.

`ElementwiseBroadcastDirectBuffer` admits fp32 rank-1 through rank-4 Vulkan
buffer add/mul/sub when the operands are mathematically broadcast-compatible,
`alpha == 1`, the op is not inplace/out, and the existing buffer elementwise
route is selected. It has no exact shape row requirement; unsupported cases are
semantic issues such as dtype, rank, layout, broadcast incompatibility, or an
unsupported op.

`SequenceCatDirectBuffer` admits fp32 rank-4 direct-buffer `cat` on dim 2 when
batch, head count, and head dimension match and both sequence lengths are
positive. It routes through the existing runtime-sized dim-2 buffer cat shader
without requiring a finite KV-cache row for every sequence length.
`InitialSequenceCatDirectBuffer` covers the matching empty-cache bootstrap case:
`torch.cat([empty, cache], dim=2)` with a Vulkan empty left operand, a fp32
rank-4 Vulkan buffer cache tensor, and positive runtime batch/head/sequence/head
dim values. It uses the existing buffer copy path, so initial prompt sequence
length and head geometry no longer need exact `InitialCache` row admission.

`LinearOrMatmulDirectBuffer` admits fp32 rank-2 and rank-3 direct-buffer linear
or matmul-style execution when the RHS is rank 2, dimensions are positive,
`K` matches, and the existing `mm_buffer_float` or `mm_buffer_float_bias`
program can execute the runtime descriptor. Exact tiled QKV/FC2 rows remain
optimization evidence, not shape admission.

`EmbeddingLookupDirectBuffer` admits fp32 rank-2 Vulkan weights with rank-1 or
rank-2 CPU Long indices after the host index values are checked against the
runtime `num_embeddings` bound and copied into the existing int32 index buffer.
It also admits Vulkan-resident Long indices when their exact tensor descriptor
carries CPU-uploaded integer min/max provenance proving every value is in
`[0, num_embeddings)`. The old `num_embeddings <= 4096`,
`embedding_dim <= 256`, and `num_indices <= 128` rows are evidence limits for
these safe paths, not shader limits. Truly device-produced Long indices remain
blocked unless they have a no-readback value proof or a device-side error path
that preserves PyTorch's out-of-range exception.

`FeatureMapToTokensDirectBuffer` admits fp32 direct-buffer rank-4 feature maps
when the storage is buffer-backed and supports buffer-compute metadata. The
family is semantic for batch, channel, height, and width, but the current
implementation still requires width-packed buffer layout and zero storage
offset. The output shape is `[N, H * W, C]`. Corpus-shape behavioral tests
exercise the former exact-row envelope, but production and test admission use
only these semantic guards.

`TokenPrefixCatAddDirectBuffer` admits fp32 rank-3 direct-buffer prefix-token
concat plus positional add when `dim == 1`, prefix length is `1`, batch and
feature dimensions match, token count is positive, and the output sequence is
`1 + token_count`. Batch, token count, and feature dimension are runtime
descriptor values. The finite `TokenPrefixCatAddContract` rows remain evidence
for DAv2 token preparation, not production token-count or feature-dim bounds.

`CatAxisDirectBuffer` admits fp32 buffer-backed rank-4 channel-axis cats when
all inputs share batch/height/width, the cat dimension is 1, every channel
extent is positive, and the current buffer-view implementation's channel
multiple-of-4 layout constraint is satisfied. Input count, spatial size, batch,
and total channel count are runtime descriptor values. The finite
`ChannelCatContract` rows remain focused evidence and regression fixtures, not
default H/W or input-count admission.

`BatchNormInferenceDirectBuffer` admits fp32 buffer-backed rank-4 eval-mode
batch norm when running statistics and optional affine tensors are rank-1,
feature counts match the runtime channel count, and the existing
`batchnorm_4d_buffer_float` shader can consume the metadata-backed descriptor.
The finite batch-norm fixtures remain evidence, not production shape bounds.

`DirectDecodeGQASDPADirectBuffer` admits fp32 rank-4 non-causal decode GQA
with no mask, `dropout_p == 0`, batch 1, query length 1, query heads divisible
by key/value heads, runtime positive source length, head dim within the
existing direct-GQA shader budget, and the default/head-dim-equivalent scale.
It routes through the existing `scaled_dot_product_scores_value_gqa_buffer_float`
single-dispatch shader, so source length, head count, head dim, and value dim
are runtime descriptor values rather than finite Transformer row bounds. The
finite Transformer GQA SDPA rows remain evidence and optimized-policy fixtures.

`DirectCausalPrefillGQASDPADirectBuffer` admits fp32 rank-4 causal prefill GQA
and equal-head causal MHA with batch 1, equal query/source sequence length, no
explicit mask, no dropout, default/head-dim-equivalent scale, and the same
direct-GQA shader head/value-dimension budgets. GQA requires query heads to be
divisible by key/value heads; MHA is admitted only when query heads equal
key/value heads, which makes the existing direct-GQA shader's repeat factor `1`.
The causal mask is applied inside
`scaled_dot_product_scores_value_gqa_buffer_float`, so legal causal prefill
sequence length, head count, head dim, and value dim are runtime descriptor
values.

`SmallNonCausalGQASDPADirectBuffer` admits bounded fp32 rank-4 non-causal GQA
with no mask, `dropout_p == 0`, batch 1, query heads divisible by key/value
heads, runtime target/source lengths up to 64, direct-GQA shader-budget head
and value dims, and default/head-dim-equivalent scale. The exact
`SmallNonCausalGQA` rows remain evidence and adjacent-negative fixtures, but
bounded q>1 non-causal GQA no longer needs exact query/source rows.

`DirectNonCausalMHASDPADirectBuffer` admits fp32 rank-4 equal-head non-causal
MHA with batch 1, no mask/dropout/GQA, default/head-dim-equivalent scale, direct
Q/K/V buffer layout, head/value dims aligned to the direct-buffer shader lane
width, and the existing direct-GQA shader head/value-dimension budgets. It uses
the same direct shader with repeat factor `1`, so diffusion-style square MHA
and other legal equal-head rank-4 MHA shapes do not need finite diffusion row
admission when their layout is already direct-buffer compatible.

`VisionSelfAttentionSDPAContract` `Rank3Head64Scale1RuntimeShape` admits fp32
rank-3 self-attention when Q/K/V shapes match, head dim is `64`, explicit scale
is `1.0`, and mask/dropout/causal/GQA are disabled. Batch-head count and
sequence length are runtime descriptor values. `SDPAScoreSoftmaxContract`
`VisionSelfAttentionScoresRuntimeShape` similarly admits the resulting rank-3
square score tensor for the buffer last-dim softmax path. The old sparse rows
remain evidence, optimized-policy fixtures, and regression cases.

`SDPAExecutionPolicyContract` now has a semantic runtime policy for small-head
non-causal MHA. The `RecognizerNonCausalMHACloneOnly` finite row remains
evidence for the PaddleOCR recognizer case, but fp32 rank-4 no-mask,
no-dropout, non-causal, non-GQA MHA with matching Q/K/V heads, positive runtime
target/source lengths, and head/value dims within the tiled-buffer shader budget
can select the same runtime-fused direct-buffer policy without a finite row.
`TransformerDecodeGQACloneOnlyRuntimeShape` similarly removes the old
execution-policy source-length row as a production admission gate for fp32
rank-4 decode GQA when batch is `1`, query length is `1`, query heads are
divisible by key/value heads, no mask/dropout/causal mode is active, scale is
default or head-dim equivalent, and the score tensor stays within budget. The
old `TransformerDecodeGQACloneOnly` row remains evidence for the post-softmax
clone policy.
Known-bad diffusion-like head-dim and materialization policies remain row-gated
or fail-closed.

`MaskedTinySDPAContract` `AdditiveFloatMaskRuntimeShape` admits fp32 rank-3 or
rank-4 Q/K/V tensors with fp32 additive masks that broadcast to the attention
score shape under the existing math path. The fixed `[1,16,2,64]` plus
`[1,1,2,2]` row remains regression evidence, but runtime target/source lengths,
batch/head counts, rank-3/rank-4 form, and explicit finite scale no longer need
exact row admission. Bool masks, dropout, causal mode, GQA, non-finite scales,
and over-budget score tensors remain fail-closed.

`DiffusionSDPAContract` `CrossAttentionRuntimeShape` admits the mask-free fp32
rank-4 diffusion cross-attention slice where batch is `1`, Q/K/V heads and
head dim match, head dim is `64`, key/value sequence length is a small runtime
descriptor, and the score tensor stays within budget. The old `kv=2` cross
rows remain fixtures, while unseen legal `kv` lengths such as `3` now run
through the existing SDPA math path. `SquareSelfAttentionRuntimeShape` now
admits the mask-free fp32 rank-4 square self-attention slice when batch is `1`,
Q/K/V heads and head dim match, head dim is `64` or single-head `512`,
sequence length and score elements stay within budget, scale is absent or
head-dim equivalent, and the `512`-wide materialized-math path can prove a
width-pack-compatible key transpose (`sequence % 4 == 0`). The existing
`head_dim=512` rows remain evidence; non-width-pack-compatible `512` square
sequences still fail closed until the materialized key-transpose command plan
can produce a direct-buffer layout.
The paired `DiffusionMaterializedSquareRuntimeShape` execution policy keeps the
existing conservative score pre-materialization plus post-softmax clone behavior,
and `DiffusionSquareScoresRuntimeShape` removes the old exact head/sequence
score-softmax allowlist for runtime square score tensors.

`GQARepeatDirectBuffer` now authorizes the existing runtime-sized
`gqa_repeat_buffer_float` shader for fp32 rank-4 Vulkan buffer K/V tensors when
the repeat factor is positive and the repeated head count is derived from
runtime metadata. The SDPA route may use this materialization only for
non-causal, mask-free GQA when both K and V match the repeat contract and the
resulting rectangular rank-3 score tensor matches
`RectangularScoresRuntimeShape`. Random unseen query/head/source/head-dim cases
cover this path, and the shader now indexes repeated heads by output head rather
than sequence coordinate.

`SDPAScoreSoftmaxContract` now admits rectangular fp32 rank-3 score tensors
with runtime batch-head/target/source dimensions and a bounded score-element
budget. This keeps materialized-GQA probabilities on the buffer last-dim path
instead of the old known-bad texture fallback. Square diffusion and vision
score families remain separate because their probability materialization policy
is tied to attention transition evidence.

`NoOverlapConvTranspose2DContract` `DynamicKernelStrideFloatBuffer` admits fp32 rank-4 direct-buffer
transposed convolutions in the no-overlap family when `kernel == stride`,
padding/output-padding are zero, dilation is one, groups are one, and the
existing no-overlap buffer shader can stay on the clean packed-buffer path.
Batch, output channels, kernel/stride, and spatial dimensions are runtime
descriptors. Low input-channel cases currently remain blocked by the older
small-metadata/exact-rearrange materialization path and are tracked as a
layout/materialization gap, not as observed-shape evidence.

`Conv2DDirectBuffer` currently remains a direct-layout scaffold. The existing
generic conv shader is descriptor-driven and accepts runtime batch/channels,
kernel, stride, padding, and spatial dimensions, but the production path is not
direct-buffer-owned for many generic cases. That makes it a layout/transition
contract problem rather than an exact-shape row problem. Grouped, dilated,
transposed, fused-add, `out=`, and direct-output ownership all remain outside
this family until each has random-shape parity and explicit command/layout
ownership.
`PackedBufferConv2D` covers the existing metadata-packed float-buffer conv path
for batch-one fp32 groups-one, dilation-one runtime shapes on the generic
`conv2d_buffer_float` branch. Non-1x1 random-shape tests exercise this family
without requiring exact observed H/W or kernel rows.

## Validation Policy

Dynamic families need randomized legality tests, not just fixed known-good
rows.

Rules:

```text
generate random legal shapes every run
log the seed in stdout
allow seed override for reproduction
compare against CPU
assert zero CPU fallback when the family owns native execution
assert zero sync readback except final test readback
keep generated shapes practical for test runtime while not treating practical
test limits as production admission limits
capture the semantic reject reason for unsupported shapes
```

The first test uses `PYTORCH_VULKAN_DYNAMIC_SHAPE_FUZZ_SEED` as an optional
reproduction seed and otherwise samples a fresh seed with `os.urandom()`.

## Promotion Rule

New exact pointwise-conv rows should not be the default answer to an unseen
shape. A new exact row needs one of:

```text
the dynamic family rejected with a named unsupported reason
the row is optimization evidence for a faster plan
the row captures a negative guard or regression fixture
```

The next families should follow the same structure before broad promotion:

```text
PackedBufferConv2D semantic family or Conv2DDirectBuffer layout transition
ReductionDirectBuffer
RegionCommandList
```
