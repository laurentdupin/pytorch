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
ElementwiseBroadcastDirectBuffer / GenericRuntimeShape
LinearOrMatmulDirectBuffer / GenericRuntimeShape
DynamicNoOverlapConvTranspose2DContract / KernelStrideFloatBuffer
```

`Conv2DDirectBuffer` is represented as a semantic target family for the existing
descriptor-driven generic conv shader, but it is not route-authorized yet.
Random legal-shape parity exposed that the current packed-weight route cannot
be treated as a complete dynamic conv2d implementation without additional
value/readback proof.

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

### Generated Shader / Command List

Some families will need generated shaders or generated command lists. The
runtime must treat those as explicit plan modes:

```text
GeneratedShader
GeneratedCommandList
```

Generation is not a route loophole. It is allowed only when the semantic
family validator accepts the operation and the contract grants compilation or
command-list generation for that family.

## First Implemented Slice

`DynamicPointwiseConv1x1DirectBufferContract` admits fp32 direct-buffer
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

This slice is correctness-first. It deliberately routes to the existing
dynamic-shape 1x1 buffer shader and does not select the as-linear optimized plan.

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

`LinearOrMatmulDirectBuffer` admits fp32 rank-2 and rank-3 direct-buffer linear
or matmul-style execution when the RHS is rank 2, dimensions are positive,
`K` matches, and the existing `mm_buffer_float` or `mm_buffer_float_bias`
program can execute the runtime descriptor. Exact tiled QKV/FC2 rows remain
optimization evidence, not shape admission.

`DynamicNoOverlapConvTranspose2DContract` admits fp32 rank-4 direct-buffer
transposed convolutions in the no-overlap family when `kernel == stride`,
padding/output-padding are zero, dilation is one, groups are one, and the
existing no-overlap buffer shader can stay on the clean packed-buffer path.
Batch, output channels, kernel/stride, and spatial dimensions are runtime
descriptors. Low input-channel cases currently remain blocked by the older
small-metadata/exact-rearrange materialization path and are tracked as a
layout/materialization gap, not as observed-shape evidence.

`Conv2DDirectBuffer` currently models the intended descriptor-driven generic
fp32 conv2d family:
rank-4 input and weight, positive batch/channels/spatial/kernel/stride/dilation,
non-negative padding, positive groups, input channels equal to
`weight_input_channels * groups`, and output channels divisible by groups.
It remains scaffold-only until randomized legal-shape parity proves the route
has correct packed-weight/layout semantics and no hidden readback for its full
declared envelope.

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
Conv2DDirectBuffer generated/direct route implementation
ReductionDirectBuffer
RegionCommandList
```
