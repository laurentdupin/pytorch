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

The first executable generated-shader slice is the explicit
`vulkan_prepack::runtime_elementwise_chain` POC op. It is not an eager-route
replacement. It compiles a metadata-aware fp32 buffer shader at runtime with
`PYTORCH_VULKAN_RUNTIME_SHADER_GLSLC`, loads the SPIR-V into an owned
`ShaderInfo`, binds the usual tensor metadata UBOs, and dispatches a runtime
tensor-RHS op list over `add`/`mul`/`sub`/`div`. The compatibility helper
`vulkan_prepack::runtime_elementwise_chain_add_mul_sub_div` calls the same
executor for:

```text
(((input + add_rhs) * mul_rhs) - sub_rhs) / div_rhs
```

The program key is semantic rather than shape-row based: the op sequence, dtype,
rank/layout class, and descriptor pattern define the compiled program, while
same-shaped rank-1 through rank-4 Vulkan fp32 buffer tensor extents use runtime
tensor metadata for sizes, strides, storage offsets, width-pack padding, and
dispatch geometry. RHS operands may also broadcast to the root/output shape,
which covers Python scalar literals as the rank-0 Vulkan buffer operands that
the existing eager route already emits. H/W and other extents are not baked into
the shader. The slice intentionally remains explicit and narrow for direct
invocation. The current executor supports 1 to 4 tensor-RHS ops because
`submit_compute_job` is variadic; a fully unbounded generated command list needs
a lower-level dynamic descriptor binding path.

`PYTORCH_VULKAN_RUNTIME_ELEMENTWISE_CHAIN_LIVE_LOG=<path>` is the first
live-handle bridge from eager execution into that generated executor. The
normal `aten::binary_op.buffer_float` and `unary_op_buffer` paths share a
behavior-neutral mixed live-chain recorder for fp32 Vulkan buffer elementwise
chains. It records binary tensor/RHS-broadcast steps over
`add`/`mul`/`sub`/`div` and unary steps over
`exp`/`sqrt`/`log`/`sin`/`cos`/`neg`/`reciprocal`/`rsqrt`/`silu` by the logical
eager tensor handles when the root/output shape is stable and every tensor RHS
is broadcast-compatible with that shape. With
`PYTORCH_VULKAN_RUNTIME_ELEMENTWISE_CHAIN_EXECUTE=1`, the recorder submits the
matching generated runtime shader as a sidecar proof once the chain reaches at
least two ops. It does not replace the eager result, does not defer the eager
ops, and logs `behavior_change=0` /
`normal_eager_output_preserved=1` in
`VulkanRuntimeElementwiseLiveChainTrace.v0`. The generated program key is the
op sequence and operand-kind sequence, not exact tensor extents; runtime tensor
metadata supplies sizes, strides, storage offsets, and dispatch geometry. The
current sidecar supports up to eight steps and up to four tensor-RHS operands
because the executor still uses the existing variadic `submit_compute_job`
entry point. Focused random-shape coverage includes a pure unary
`neg -> exp -> sqrt` chain and a mixed
`add -> neg -> exp -> mul -> sqrt` chain. Mixed scalar UBO steps, aliases,
output ownership, and flush-time replacement remain future runtime command-list
work rather than hidden behavior. The live sidecar is intentionally opt-in
because it retains eager tensor handles long enough to submit the proof shader;
production replacement must move to metadata snapshots plus weak storage
validation before it can own output or retire behavior.
`PYTORCH_VULKAN_RUNTIME_ELEMENTWISE_CHAIN_CHECK_OUTPUT=1` is an additional
diagnostic-only mode that reads back the sidecar output and normal eager output
and logs `output_check_max_abs`; it is for parity tests and must not be used as
a performance path.

`PYTORCH_VULKAN_RUNTIME_ELEMENTWISE_CHAIN_DEFER=1` is the first
behavior-changing deferred-output slice. For the same narrow fp32 Vulkan buffer
elementwise-chain family, supported binary tensor-RHS and unary operations may
return a Vulkan placeholder instead of running the eager kernel immediately.
The placeholder owns the future output buffer, the runtime chain is retained by
logical tensor handle, and mandatory access boundaries materialize the chain by
submitting the generated runtime shader into that placeholder. CPU readback via
`copy_` is the first materialization boundary. The trace row is
`VulkanRuntimeElementwiseDeferredChainTrace.v0` and records
`behavior_change=1`.

This canary is deliberately smaller than the live-chain sidecar:

```text
supported:
  fp32 Vulkan BUFFER tensors with zero storage offset
  stable root/output shape
  same-shape tensor RHS operands
  unary steps supported by the mixed runtime generator
  <= 8 steps and <= 4 tensor RHS operands

not yet supported:
  scalar UBO steps as retained deferred operands
  storage offsets/out/in-place mutation
  non-elementwise consumers without forced materialization
  cross-region output ownership
  unbounded descriptor lists
```

If a deferred placeholder reaches an unsupported eager consumer, the consumer
must materialize it first and then use the existing eager path. Legal semantics
should fail closed with a reason rather than reading an uninitialized placeholder.

The canary also fails closed during stack planned recording. DAv2 `vits_140`
showed that the generated runtime `add`/`mul` shaders can be numerically exact
on the same standalone shapes while still producing bad model output if their
inputs are read later across stack-owned lifetime boundaries. A stack single-op
`mul` execute-at-op-site experiment also failed bridge sanity, and the same
candidate still failed after `add_buffer_out_vulkan` materialized placeholders
before its residual add. A follow-up A/B materialized the same stack deferred
single-op `mul` through the existing static registered `VK_KERNEL(buffer_mul)`
path instead of the runtime-owned SPIR-V path; bridge sanity still failed. This
makes stack deferred output replacement a value-preservation/input-lifetime
problem, not a shader-generation-only problem. With
`PYTORCH_VULKAN_RUNTIME_ELEMENTWISE_CHAIN_DEFER=1`, stack planned recording logs
`stack_plan_reject` rows with
`reason=stack_deferred_value_preservation_unproven`, including candidate
input/RHS tensor state and provenance, then uses the existing eager path. The
same trace rows classify attempted stack replacement as
`pipeline_path=stack_generated_command_list_candidate` with
`pipeline_path_status=rejected`,
`value_preservation_status=stack_value_preservation_unproven`, and a
shape-independent `program_key`; ordinary non-stack deferred materializations report
`pipeline_path=runtime_owned_spirv_deferred_materialize` and
`value_preservation_status=consumer_materialize_boundary_required`. Normal
`register_output` and `materialize_output` rows also record whether stack
planned recording was active. `add_buffer_out_vulkan` materializes
runtime-deferred placeholder inputs before checking and recording its buffer add
route, and passes a materialization callsite into the deferred trace when that
boundary fires, so generic out consumers do not read placeholder buffers
directly. Layernorm context consumers also materialize runtime-deferred
placeholders before reading context inputs.
Stack candidate rows now also carry structured value-lease and command-order
proof fields: input/RHS/output tensor keys, storage ids, view ids, generations,
logical descriptor hashes, byte ranges, provenance writer and route, stack
phase/block, command-buffer recording id, stack recorded
compute-job/descriptor/barrier counters, topology status, and
`authorizes_stack_dynamic_path=0`. The current `vits_140` evidence observes a
valid stack recording domain for residual1/residual2 candidates, but reports
`stack_command_order_proof_status=recording_domain_dispatch_count_observed_consumer_order_unproven`
and `value_lease_status=captured_tensor_handle_without_stack_value_lease`.
Additional local POCs narrowed the blocker: the generated runtime
`[tokens,384] * [384]` chain is exact outside the stack, including when both
inputs are Vulkan clone leases. Inside stack planned recording, broad deferred
layerscale still failed bridge sanity after explicit layernorm materialization,
after private device-copy input/RHS leases, and after materializing the stack
`mul` through the existing static binary shader before copying into the deferred
placeholder. That points at current-topology placeholder/output ownership and
consumer ordering, so stack runtime elementwise deferral remains rejected until
a generated command-list region owns the output and consumer sequence directly.
The first behavior-bearing stack uses of the runtime generator therefore avoid
placeholder replacement: non-program residual1 and residual2 now try
materialized runtime-generated `mul -> add` chains for
`attention_output * ls1_gamma + input_2d` and
`mlp_output * ls2_gamma + hidden_states`, returning generated outputs as normal
eager tensors. The optional helper materializes existing deferred bridge
operands before compiling/submitting the generated chains, matching the ordinary
binary-op materialization discipline. A focused DAv2 `vits_140`
`stack_capture_decoder_preprocess` run recorded 144
`vulkan_prepack::runtime_elementwise_chain` submissions, reduced
`aten::binary_op.buffer_float` hits to 24, kept zero CPU fallback and zero sync
readback, and passed bridge sanity `max_abs=1.1846423149108887e-06` /
`mean_abs=6.115393347272402e-08`. A direct output-slot variant was rejected:
both the existing static `add_scaled_buffer_float` and the generated `mul ->
add` shader corrupted bridge sanity when they wrote a fresh stack block output
slot, confirming that slot/output ownership still belongs to future region work
rather than this eager generated-chain slice.
The next ownership step is generated out-buffer execution, not placeholder
deferral. `try_runtime_elementwise_chain_out_vulkan` submits the same mixed
runtime elementwise shader into caller-owned output storage after materializing
deferred operands and marking the output for Vulkan execution. It is currently
used only by `token_prefix_cat_add` for the prefix and token position-add views
inside one concatenated output tensor; unsupported layout/storage cases fall
back to `add_buffer_out_vulkan`. A focused `vits_140` bridge run recorded 14
`vulkan_prepack::runtime_mixed_elementwise_chain_out` hits and reduced
`aten::binary_op.buffer_float` hits to 10 while preserving the same bridge
sanity and zero fallback/readback counters. The remaining 10 binary adds are
not the repeated bridge-private decoder context path: a follow-up attribution
run showed one `[1,64,5,8]` add and three adds at each of `[1,64,10,15]`,
`[1,64,20,30]`, and `[1,64,40,60]`, all before the first
`run_vision_decoder_fusion_block_context.*` bridge record. Binary buffer hits
now log `kind`, operand/output sizes, optional `callsite`, and parent caller in
`PYTORCH_VULKAN_OP_HIT_LOG` so future generated-chain work can classify those
setup/preflight residual adds separately from timed bridge execution.
Consumers that are not part of the generated chain, including convolution,
activation/clamp and upsample, materialize a placeholder before reading it; the
central `ensure_buffer_storage` and execution-planner preparation helpers do
the same for generic eager consumers.

`PYTORCH_VULKAN_DEFERRED_REGION_PLAN_LOG=<path>` is the first broad
behavior-neutral `DeferredRegionPlan.v0` surface. Instead of observing only the
old label-only lazy chain, it hooks the central tensor provenance writer and
records a logical deferred tensor handle for every Vulkan output that reports a
write, plus value-lease counts for its Vulkan inputs. Mandatory boundaries emit
`VulkanDeferredRegionPlanTrace.v0` rows with the op nodes, routes, handle count,
value-lease count, alias/view count, boundary kind, and the exact fail-closed
prerequisites still blocking execution. Vulkan-to-CPU `copy_` now also attaches
the concrete value-access boundary that forced the flush, including source and
destination tensor-state descriptions. Eager execution remains authoritative:
the row records `execution_enabled=0` and `behavior_change=0`. This is the
foundation needed before a future flush-boundary executor can replace eager
dispatches with a generated shader or generated command list.

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
offset. The output shape is `[N, H * W, C]`. The finite
`PatchEmbedFeatureMapToTokensContract` rows remain evidence and regression
fixtures, not default H/W admission.

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

## Lazy Chain Observation

`PYTORCH_VULKAN_LAZY_CHAIN_LOG=<path>` records the eager Vulkan operation chain
that could be collected by a future lazy region compiler. The observer is
behavior-neutral: ops still execute eagerly, but every `log_vulkan_op_hit`
record is appended to a per-thread chain until a mandatory access boundary
flushes it. Current flush boundaries are:

```text
device_copy
host_upload
host_readback
cpu_fallback
sync_readback
synchronize_stream
synchronize_device
stream_flush
stream_exchange
flush_pending_cmds
event_record
event_block
event_synchronize
```

Each JSONL row uses schema `VulkanLazyEagerChain.v0` and contains the boundary,
submit phase, caller/runtime labels where available, and the ordered op list.
This is the first diagnostic surface for answering, "what chain could we have
captured before Python or the backend required a real value?"

`flush_pending_cmds` submits are counted under submit origin
`pending_command_flush` instead of `tensor_cpu_readback`, and the lazy/deferred
trace rows carry a finer `reason` field. Current reasons separate eager
linear/addmm submits, linear+GELU submits, replay input-upload visibility,
first-record warmup, compiled replay submit guards, vision replay output
materialization, stack replay step submit guards, and temporary-clone lifetime
protection. This keeps the trace useful for deferred execution planning: a
future region planner can rank natural-chain cuts by semantic cause instead of
treating every internal pending-command flush as the same blocker or polluting
readback counters.

`PYTORCH_VULKAN_LINEAR_PENDING_FLUSH_DEFERRAL=linear` is the first opt-in
behavior canary built on that classification. It only targets plain
`aten::linear` pending-command flushes after the packed-context path. It does
not affect `addmm`, raw-direct linear, linear+GELU, repeat temp-clone lifetime,
replay input uploads, replay warmup, replay submit guards, or output
materialization. The canary is accepted only in inference mode, with Vulkan
input/weight/output tensors, a rank-2 Vulkan weight, and a persistent packed
linear context retained by a bounded canary owner until blocking readback/fence
cleanup releases it. Inference mode is the autograd guard; module parameter
tensors may still carry `requires_grad=True` metadata. If those guards fail, or
if the retention budget is exhausted, the existing `linear_eager_submit` flush
still runs.

`PYTORCH_VULKAN_DEFERRED_EXECUTION_LOG=<path>` records the lifecycle of the
existing deferred bridge paths that already return placeholder Vulkan tensors
and materialize or fuse later. This is still behavior-neutral; it does not make
new ops lazy and it does not skip eager execution outside those existing
bridges. The log uses schema `VulkanDeferredExecutionTrace.v0` and emits:

```text
deferred_bridge_event
deferred_region_flush
```

The first event records bridge families such as image normalization,
linear+GELU, add+layer-norm, layer-scale, attention query-scale, and decomposed
attention bridges when they defer, alias, hit, fuse, materialize, go stale, or
clear a registry entry. The flush event is emitted at the same mandatory
boundaries as `PYTORCH_VULKAN_LAZY_CHAIN_LOG`, but only when at least one
deferred bridge event is pending. It records the pending deferred event count
plus the pending lazy-chain op count and, for pending-command flushes, the same
reason code used by the lazy-chain observer. This provides the initial central
deferred-region trace surface before a broader runtime command-list executor
exists.

## Runtime Shader Compilation POC

`PYTORCH_VULKAN_RUNTIME_SHADER_COMPILE_LOG=<path>` plus
`PYTORCH_VULKAN_RUNTIME_SHADER_CACHE_DIR=<dir>` enables the first
behavior-neutral runtime shader compilation proof of concept. When a mandatory
flush boundary is reached, the lazy-chain observer inspects the pending op
backlog. If it sees a supported resident tensor-buffer elementwise chain, it
generates a compute GLSL shader for that chain and writes it to the cache
directory.

If `PYTORCH_VULKAN_RUNTIME_SHADER_GLSLC=<path-to-glslc>` is also set, the POC
invokes `glslc` and validates the generated SPIR-V header. The generated row
uses schema `VulkanRuntimeShaderCompileTrace.v0` and records:

```text
family
group_key
operand_kind
ops
boundary_kind
reason
glsl_path
spv_path
compile status
cache hit
behavior_change=0
```

The first supported family is `ElementwiseChain` for fp32 tensor-buffer
elementwise chains whose operands are already Vulkan-resident before the chain
starts. It now recognizes binary `add`, `sub`, `mul`, `div`, `floor_divide`,
and `pow` plus unary `exp`, `sqrt`, `log`, `sin`, `cos`, `neg`, `rsqrt`, and
`silu`. Runtime shape is represented by `numel`; the generated shader does not
require exact H/W or rank rows. A focused test pre-uploads all operands,
executes `add -> mul -> sub -> div -> pow`, reaches a host-readback boundary,
and asserts that the POC generated GLSL, compiled SPIR-V when `glslc` is
available, and left normal eager execution numerically unchanged. Unary ops are
already supported by the generator, but current eager unary kernels may insert
device-copy boundaries between unary dispatches; those boundaries correctly
prevent the POC from pretending they formed one natural no-sync chain.

This POC does not redirect execution, bind the generated pipeline, defer tensor
SSA values, or replace existing eager dispatches. Its purpose is narrower:
prove that a backend-visible op backlog at a flush point can be recognized,
converted to runtime-generated shader source, compiled, cached by a group key,
and logged without changing behavior.

`PYTORCH_VULKAN_RUNTIME_COMMAND_LIST_LOG=<path>` records the next
behavior-neutral layer for the same recognized backlog. It emits
`VulkanRuntimeCommandListPlanTrace.v0` rows that describe the command buffer
the generated-region runtime would record:

```text
program key
shader family
runtime shape policy
dispatch geometry
descriptor slots
params uniform buffer
push-constant support status
barriers
commands
deferred-handle/output/alias proof requirements
missing execution prerequisites
execution_enabled=0
behavior_change=0
```

For the initial `ElementwiseChain` tensor-buffer family, the plan has one
output storage buffer, one base input buffer, one right-hand-side buffer per
binary op, a required params uniform buffer carrying `numel` and any scalar
constants, pre-dispatch read/write barriers, and the command sequence
`bind pipeline -> bind descriptors -> bind params -> dispatch`. The plan records
`push_constants_supported=0` because current Vulkan compute pipeline layouts do
not expose push-constant ranges. This is the intended replacement shape for old
replay: generate a fresh command plan from semantic descriptors at runtime. It
still does not execute because the current eager op-hit backlog does not yet
carry deferred tensor SSA handles, output allocation ownership, alias/escape
proof, a generated-shader executor hook, or params-UBO executor plumbing.

`api::ShaderInfo` now has an owned-SPIR-V construction path so future generated
programs can give the shader module cache a stable byte owner instead of a
temporary vector. Static registered shaders keep their existing pointer/size
cache identity; owned runtime shaders use content-based hashing. This is an
execution prerequisite only, not generated-region execution by itself.

The same command-list log also recognizes `DeviceCopyChain` candidates for
Vulkan buffer-to-buffer `copy_` work. These rows intentionally use
`shader_family=none_copy_command_list`: they are not fused compute shaders.
They describe a future generated command list with copy barriers and
`copy_buffer_to_buffer` commands, blocked today by missing deferred tensor
handle capture, source/destination identity capture, copy-command executor
plumbing, and alias/escape proof.

The command-list POC also logs full multi-dispatch regions when a mandatory
flush backlog is not a pure elementwise or pure copy chain. These rows are not
single generated shaders. They are examples of a future generated command list
that would record existing Vulkan kernels in order, with explicit region
ownership and barriers:

```text
ConvPrepackUploadCommandListRegion
DecoderConvUpsampleCatCommandListRegion
LinearGeluMlpCommandListRegion
PatchEmbedFeatureMapToTokensCommandListRegion
PointwiseUpsampleCommandListRegion
ResidualNormCommandListRegion
TokenPrefixBackboneCommandListRegion
TransformerBlockCommandListRegion
UpsampleCommandListRegion
VisionPatchTokenPrepCommandListRegion
ObservedMultiOpCommandListRegion
```

The row contains the raw op tokens, subfamily tags such as
`contains_elementwise`, `contains_convolution`, `contains_patch_or_token_layout`,
`contains_upsample`, `contains_attention_or_bmm`, `contains_linear`,
`contains_norm`, `contains_cat`, and `contains_copy_or_transfer`, the proposed
command sequence, the runtime descriptor shape policy, and fail-closed
prerequisites such as producer/consumer edge capture, descriptor binding
capture, barrier-plan execution, region output ownership, deferred tensor
handles, and alias/escape proof. A focused `elementwise add -> bilinear
upsample` test now proves that a complete non-elementwise chain is logged as a
full multi-dispatch region candidate instead of being mistaken for an inner
elementwise subsequence.

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
