# Vulkan Roadmap

## Operating Principle

Each task should convert observed model failures into reusable contracts,
generated tests, or structural diagnostics. Stop after a new blocker unless the
task explicitly authorizes fixing it.

The approved performance path is defined in `docs/vulkan/GRAPH_RUNTIME.md`.
Existing eager deferred, replay, compiled-session, and stack-region work is
migration evidence, not an expansion point.

## Phase 1: Route Specialization Audit

Goal: make model-name and exact-shape production routing visible.

Inputs:

- `agent_space/model_named_routes.txt`
- `agent_space/exact_shape_routes.txt`
- `agent_space/five_model_blockers.json`
- current `RoutePolicy`, ops, and tests

Deliverables:

- Updated classification of allowed, temporary, migrate, and delete routes.
- Temporary exceptions with expiry and migration target.
- No production behavior changes unless a follow-up task explicitly requests
  one.

## Phase 2: Contract Table And Codegen MVP

Goal: create a small, reviewable path from contract table to route predicate
and generated tests.

Start with finite contracts already backed by artifacts. A minimal MVP should
support:

- contract name
- family label
- tuple fields
- dtype/layout/rank/options
- materialization policy
- route label and op-hit label
- positive parity test metadata
- adjacent negative guard metadata
- device coverage notes
- fallback behavior

Generated output should be deterministic and narrow. Do not introduce a broad
kernel generator or shader generator before the table/test path is proven.

Current MVP status:

- JSON contract spec fixtures exist for `EmbeddingLookupContract`,
  `ChannelCatContract`, and `KVCacheAppendContract` `SequenceAppend` and
  `InitialCache`, plus `NoOverlapConvTranspose2DContract`
  `Kernel2Stride2FloatBuffer` and `SDPAScoreSoftmaxContract`
  `DiffusionSquareScores`, plus `GQARepeatContract`
  `Batch1Heads4Factor4Sequence100To116Dim128`, plus
  `MaskedTinySDPAContract` `AdditiveFloatMask`, plus
  `DiffusionSDPAContract` `SparseAttentionRows`, plus
  `TransformerGQASDPAContract` `SparseAttentionRows`, plus
  `VisionSelfAttentionSDPAContract` `SparseAttentionRows`, plus
  `SafeViewReshapeContract` `ViewMaterializedDirectBuffer` and
  `ReshapeAliasDenseBufferDirect`, plus `BatchNormInferenceContract`
  `BufferFloat4D` and `MaterializedBufferFloat4D`.
  `ElementwiseBroadcastContract` `FloatTensorTensorBufferBroadcast` records
  the first float tensor/tensor buffer-broadcast envelope and has a production
  metadata/provenance canary that runs after the existing buffer route is
  selected.
  `TokenPrefixCatAddContract` records rank-3 prefix-token concat plus
  position-add evidence; `TokenPrefixCatAddDirectBuffer` now owns runtime-shape
  admission for the legal fp32 direct-buffer family.
- `test/vulkan_contract_specs/contract_spec_utils.py` owns shared spec loading,
  case iteration, log naming, and expected-negative helpers.
- `TestVulkanGovernance` discovers all spec fixtures, validates shared schema,
  and checks fixture metadata against live contract sources.
- ShapeEnvelope role handling now goes through a small
  `SHAPE_ENVELOPE_ROLE_REGISTRY` with generic case-key fields and temporary
  runtime-case adapters, preserving current generated case counts while
  avoiding another open-coded role/key dispatch table.
- Generic deterministic boundary/fuzz assignment generation now interprets
  common ShapeEnvelope v1 value sets, min/max bounds, multiples, optional dims,
  scalar attributes, and adjacent-negative axes. It is governance-only today:
  a coverage bridge validates which abstract assignment paths and
  adjacent-negative axes are represented by the current generated/checked-in
  runtime cases, and runtime case counts remain unchanged until a later task
  explicitly opts a family into extra checked-in cases.
- `ShapeEnvelope` v1 now backs the generated C++ source-of-truth fixtures,
  `ChannelCatContract` `Rank4Dim1BufferView`, `EmbeddingLookupContract`
  `SmallBoundedLookup`, and `ElementwiseBroadcastContract`
  `FloatTensorTensorBufferBroadcast`, plus the first layout/materialization
  runtime fixture,
  `SafeViewReshapeContract` `ViewMaterializedDirectBuffer`, and the matching
  `_reshape_alias` direct-buffer fixture,
  `SafeViewReshapeContract` `ReshapeAliasDenseBufferDirect`, plus the first
  checked-in-case runtime fixtures for `BatchNormInferenceContract`
  `BufferFloat4D` and `MaterializedBufferFloat4D`, plus
  `ElementwiseBroadcastContract` `FloatTensorTensorBufferBroadcast`, plus
  `GQARepeatContract` `Batch1Heads4Factor4Sequence100To116Dim128`, plus
  `MaskedTinySDPAContract` `AdditiveFloatMask`, plus
  `DiffusionSDPAContract` `SparseAttentionRows`, plus
  `SDPAScoreSoftmaxContract` `DiffusionSquareScores`, plus
  `NoOverlapConvTranspose2DContract` `Kernel2Stride2FloatBuffer`, plus
  `KVCacheAppendContract` `SequenceAppend` and `InitialCache`, plus
  `SmallMetadataPaddedConv2DContract` `MaterializedBufferInput2x2`, plus
  `SmallSpatialPointwiseConvContract` `SparseProjectionRows`, plus
  `LinearGeluBridgeContract` `BackboneMlpHidden384To1536`, plus
  `TransformerGQASDPAContract` `SparseAttentionRows`, plus
  `VisionSelfAttentionSDPAContract` `SparseAttentionRows`, plus the proof-only
  `AttentionProbabilityMaterializationContract`
  `DecomposedAttentionProbabilityToValueBmm` layout-transition edge.
  The schema captures symbolic dims, min/max, values, multiples, optional
  dims, generic `broadcast_compatible` relationships, generic
  `sparse_rowsets` for correlated finite rows, aggregate bounds, layout and
  capability requirements, policies, positive cases, adjacent negatives, and
  fuzz hints for validation and codegen.
- Deterministic legal-case and adjacent-negative generation is active for
  ShapeEnvelope-backed ChannelCat, EmbeddingLookup, and both SafeViewReshape
  direct-buffer fixtures. The MVP compares generated legal positives and
  negatives against checked-in cases by semantic key, violated axis, adjacent
  value, and expected fallback/readback policy, then the runtime spec tests
  execute the generated cases through shared iterator plumbing. Checked-in
  cases remain review/parity fixtures.
- `ChannelCatContract` has the first generated C++ typed-row/helper artifact:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `ExecutionContractsChannelCatSpec.h` from the `ShapeEnvelope` v1 data in
  `channel_cat_contract.json` through the generic variadic tensor-list
  generator path, with cross-input iteration and match result construction
  still handwritten.
- `EmbeddingLookupContract` `SmallBoundedLookup` now uses the generic
  ShapeEnvelope C++ metadata/helper path: the generator emits
  `ExecutionContractsEmbeddingLookupSpec.h` from `embedding_lookup_contract.json`
  for metadata, route label, dtype/rank-list, range, boolean option bounds, the
  derived indices product helper, and matcher helper predicates. The
  token-batch row remains handwritten as evidence. Production admission is now
  dynamic for CPU-resident valid Long indices and for CPU-uploaded
  Vulkan-resident Long indices carrying integer min/max provenance; remaining
  device-produced index tensors need a value-proof/error contract rather than
  another shape row.
- `ElementwiseBroadcastContract` `FloatTensorTensorBufferBroadcast` has the
  first generic ShapeEnvelope C++ metadata/helper artifact:
  `ExecutionContractsElementwiseBroadcastSpec.h` is emitted without adding a
  family-specific generator branch. It provides contract identity, metadata,
  scalar/rank/layout/attribute bounds, the bounded `add`/`mul`/`sub` op-axis,
  and the generated right-aligned `broadcast_compatible` helper for the
  existing provenance canary only.
- `NoOverlapConvTranspose2DContract` `Kernel2Stride2FloatBuffer` now uses the
  generic ShapeEnvelope C++ simple-bounds generator path:
  `ExecutionContractsNoOverlapConvTranspose2DSpec.h` provides contract
  identity, metadata, dtype/rank/options/layout bounds, and helper predicates
  for the existing matcher, including input/weight channel equality.
  Output-shape arithmetic, prepack resource behavior, and match-result
  construction remain handwritten.
- `SmallMetadataPaddedConv2DContract` now has a semantic
  `RuntimeMaterializedBufferInput2x2` matcher for batch-one fp32 width-packed
  non-direct small-channel padded inputs. The checked-in
  `MaterializedBufferInput2x2` JSON row and generated helper remain evidence
  for the original PaddleOCR tuple, but production admission no longer depends
  on exact input height, width, output channels, or the original channel count.
  Tensor-info extraction, materialization dispatch, op-hit logging, fallback
  visibility, and match-result construction remain handwritten.
- `LinearGeluBridgeContract` `BackboneMlpHidden384To1536` now uses the generic
  ShapeEnvelope C++ simple-bounds generator path:
  `ExecutionContractsLinearGeluBridgeSpec.h` provides contract identity,
  metadata, rank/shape/packed-weight/options bounds, minimum flattened rows,
  and result-policy constants for the evidence fixture. Runtime bridge
  admission now uses `LinearGeluBridgeContract` `GenericRuntimeShape`
  semantic M/K/N, bias, alpha/beta, `out=`, and GELU-approximation guards
  instead of exact hidden dimensions. Deferred registry ownership, alias
  retargeting, materialization on non-GELU consumers, fused-GELU execution,
  op-hit labels, rank-3 equality, and match-result construction remain
  handwritten.
- `SmallSpatialPointwiseConvContract` `SparseProjectionRows` now uses the
  generic ShapeEnvelope C++ sparse-rowset generator path:
  `ExecutionContractsSmallSpatialPointwiseConvSpec.h` provides contract
  identity, per-row metadata, input/weight channel equality, the 39 correlated
  projection rows, exact lookup helpers, and the generated factorized
  depth-vision projection helper for the 144 cross-adapter proven shapes. The
  factorized helper admits only the approved channel-pair group crossed with
  the approved spatial-pair group; min/max envelopes and wider independent
  channel/spatial cross-products remain out of scope. Route-policy hard-fail
  rescue, shader-family decisions, family op-hit labels, and match-result
  construction remain handwritten.
- `SmallSpatialPointwiseConvContract` `GenericDynamicHW` is the first runtime-adaptive
  pointwise example. It validates semantic legality for
  fp32 direct-buffer 1x1 conv, then reuses the existing dynamic-shape
  1x1 buffer shader for unseen H/W instead of requiring an exact sparse row.
  Batch-one width-packed cases may select the existing as-linear plan from the
  same dynamic admission; sparse rows remain evidence and regression fixtures.
- `ElementwiseBroadcastDirectBuffer` records the same dynamic-program pattern
  for fp32 rank-1 through rank-4 Vulkan buffer add/mul/sub with mathematically
  legal broadcasting. Exact shapes are not admission gates for this family.
- `KVCacheAppendContract` `SequenceAppend` and `InitialCache` now use the
  generic ShapeEnvelope C++ simple-bounds generator path:
  `ExecutionContractsKVCacheAppendSpec.h` and
  `ExecutionContractsKVCacheAppendInitialSpec.h` provide contract identity,
  metadata, route labels, dtype/rank/scalar/range bounds, helper predicates,
  and SequenceAppend batch/heads/head-dim equality for the existing matcher.
  Initial-empty handling, sequence lower bounds, InitialCache cross-input
  handling, and match-result construction remain handwritten.
- `GQARepeatContract` `Batch1Heads4Factor4Sequence100To116Dim128` now uses the
  generic ShapeEnvelope C++ simple-bounds generator path:
  `ExecutionContractsGQARepeatSpec.h` provides contract identity, metadata,
  dtype/rank/source tensor bounds, repeat-factor constants, and
  target-head/target-sequence metadata for the existing matcher. SDPA
  admission, materialization allocation and dispatch, op-hit labels, sequence
  lower-bound preservation, and match-result construction remain handwritten.
- `SDPAScoreSoftmaxContract` `DiffusionSquareScores` now uses the generic
  ShapeEnvelope C++ simple-bounds generator path:
  `ExecutionContractsSDPAScoreSoftmaxSpec.h` provides contract identity,
  metadata, dtype/rank/last-dim, heads value-set, sequence value-set,
  square-score, and fallback/materialization policy constants for the existing
  matcher. Softmax route ordering, guard fallback labeling, and broader SDPA
  policy remain handwritten.
- `MaskedTinySDPAContract` `AdditiveFloatMask` now uses the generic
  ShapeEnvelope C++ simple-bounds generator path:
  `ExecutionContractsMaskedTinySDPASpec.h` provides contract identity,
  metadata, exact query/key/value/mask dtype, rank, shape, scalar options, and
  fallback/materialization policy constants for the existing matcher. Route
  hard-fail ordering, scale tolerance, SDPA execution, and broader SDPA policy
  remain handwritten.
- `DiffusionSDPAContract` `SparseAttentionRows` now uses the generic
  ShapeEnvelope C++ sparse-rowset generator path:
  `ExecutionContractsDiffusionSDPASpec.h` provides contract identity, per-row
  metadata, the 11 correlated square/cross-attention rows, and exact lookup and
  row-match equality by heads, query sequence, key/value sequence, and head dim
  for the existing matcher. Route-policy hard-fail ordering, scale tolerance,
  SDPA execution, materialization policy, and broader SDPA policy remain
  handwritten.
- `SDPAExecutionPolicyContract` `SparsePolicyRows` now uses the generic
  ShapeEnvelope C++ sparse-rowset generator path:
  `ExecutionContractsSDPAExecutionPolicySpec.h` provides contract identity,
  per-row metadata, the six correlated execution-policy rows, exact lookup and
  row-match bounds by family, heads, sequence bounds, head dim, and GQA flag,
  and per-row materialization policy strings for the existing matcher.
  Diffusion contract admission, tuple-id cross-checks, scale tolerance, score
  pre-materialization, materialized math path, post-softmax clone behavior, and
  broader SDPA policy remain handwritten.
- `TransformerGQASDPAContract` `SparseAttentionRows` now uses the generic
  ShapeEnvelope C++ sparse-rowset generator path:
  `ExecutionContractsTransformerGQASDPASpec.h` provides contract identity,
  per-row metadata, the four correlated causal/prefill/decode GQA rows, exact
  lookup by contract family plus causal/GQA flags, and row-match
  bounds/conditional equal-sequence checks for the existing matcher. Scale
  tolerance, route-policy hard-fail ordering, tensor extraction/early dtype-rank
  guards, SDPA execution, materialization policy, and broader SDPA policy remain
  handwritten.
- `VisionSelfAttentionSDPAContract` `SparseAttentionRows` now uses the generic
  ShapeEnvelope C++ sparse-rowset generator path:
  `ExecutionContractsVisionSelfAttentionSDPASpec.h` provides contract
  identity, per-row metadata, and exact row-match bounds for six proven rank-3
  float vision self-attention rows with head dim 64, `BH in {6,12,16}`,
  `T in {151,261}`, no mask, non-causal, dropout 0, GQA off, and explicit
  scale 1.0. Proof requires the materialized math path with post-softmax clone;
  route-policy hard-fail ordering, scale tolerance, tensor extraction, SDPA
  execution, materialization policy, and broader SDPA policy remain
  handwritten.
- `AttentionProbabilityMaterializationContract`
  `DecomposedAttentionProbabilityToValueBmm` now has ShapeEnvelope
  sparse-rowset coverage, generated C++ metadata/row helpers in
  `ExecutionContractsAttentionProbabilityMaterializationSpec.h`, and
  transition-log attribution for required probability materialization events.
  It records nine direct-safe Lotus decomposed-attention probability/value-BMM
  rows, the six proven low-resolution VisionSelfAttention direct-safe rows,
  and the remaining Lotus `[10,126,126]`
  `vulkan_clone_probability_before_value_bmm` row. The vision rows now skip the
  clone only under the existing VisionSelfAttention SDPA row and direct-safe
  transition-row guard; all non-matching rows keep the clone/materialization
  path.
- `HostUploadTransitionContract`, `MetadataViewTransitionContract`,
  `FinalReadbackContract`, `IntermediateReadbackTransitionContract`,
  `SafeContiguousMaterializationContract`,
  `FallbackMaterializationContract`, and `LayoutRepackTransitionContract` are
  schema-only transition reason-bucket specs for existing transition-log
  evidence. The five-model validation collector uses them as source-of-truth
  mappings for matching reason logs only; they do not admit backend routes or
  change upload, metadata-view, copy, materialization, fallback, layout-repack,
  or readback behavior.
- `ConvWeightLayoutRepackTransitionContract` refines the conv packed-context
  weight CPU materialization edge as a specific value-bearing transition bucket
  without removing the readback or changing conv route behavior.
- Tensor provenance/value traces can carry optional contract-admission
  metadata for producers that pass an existing match. BatchNorm canaries
  distinguish direct `BufferFloat4D` admission and materialized
  `MaterializedBufferFloat4D` admission without changing the executed
  `buffer_inference_4d_float` route label. ElementwiseBroadcast uses the same
  provenance path after the existing `aten::binary_op.buffer_float` route has
  already been selected.
- `PYTORCH_VULKAN_CONTRACT_ADMISSION_LOG` is the first env-gated admission
  diagnostic surface for candidate accept/reject decisions. It emits JSONL
  `vulkan_contract_admission` records with contract metadata, outcome, phase,
  predicate, reason code, and source. The current MVP is wired to
  `ElementwiseBroadcastContract` and `BatchNormInferenceContract`; it is
  intentionally separate from op-hit logs and tensor provenance/value traces.

## Phase 2.5: Admission Diagnostics

Goal: make contract admission decisions debuggable without changing route
behavior or expanding accepted shapes.

Rules for expansion:

- Keep diagnostics opt-in through `PYTORCH_VULKAN_CONTRACT_ADMISSION_LOG`.
- Emit stable, low-volume JSONL records for candidate accept/reject decisions.
- Keep payloads free of raw shapes, tensor ids, storage ids, and tensor values
  unless a later diagnostic task explicitly scopes that expansion.
- Preserve route labels, fallback/readback behavior, materialization policy,
  match results, and accepted shape envelopes.
- Add focused accept/reject coverage when wiring a new contract family.

BatchNormInference was the first completed expansion after the
ElementwiseBroadcast MVP, and SafeViewReshape direct
`ViewMaterializedDirectBuffer` plus `_reshape_alias`
`ReshapeAliasDenseBufferDirect` are now the first layout/view slices wired to
the same surface. The governance helper
`contract_spec_utils.py --admission-diagnostics-census` now records the current
diagnostic coverage as three wired contracts, five wired spec rows, and three
source files, and validates JSONL payload fields plus accept/reject hook
presence. Suggested next expansion order, if the simple slices hold:
SDPA/attention contracts. Do not treat this order as a mandate to add
diagnostics to every contract immediately; it is guidance for future diagnostic
tasks.

## Phase 3: Migrate Next Contract Family

Use `agent_space/vulkan_contract_migration_plan.md` as the decision record.

Migration shape:

- rename production predicates and labels to contract terms
- keep model names in tests, docs, and artifacts
- keep existing legality unless the task explicitly asks for new coverage
- add or update negative tests for adjacent unsupported shapes
- record any remaining temporary exception

Choose the next task by trigger, not by a stale fixed family name:

- Add the next spec fixture when an exact tuple has stable positive and
  adjacent-negative evidence but lacks generated fixture coverage.
- Add or tighten a capability-profile canary when route admission depends on
  optional device features.
- Split an `ExecutionContracts` family when the shared table becomes harder to
  review without changing behavior.
- Refresh the real-model matrix after a default backend behavior change or
  before claiming or raising a model gate.
- Treat model rows that fail before useful Vulkan execution as environment
  blockers, not backend regression budgets. Current Lotus telemetry requires a
  real distributed/DTensor-capable source-tree build or compatible runtime; do
  not fake compiled `torch._C` DTensor APIs in the benchmark harness.

## Phase 3.5: Dynamic Program Runtime

Goal: make unknown legal shapes run through semantic dynamic families instead
of requiring exact sparse rows.

The source-of-truth design is `docs/vulkan/DYNAMIC_PROGRAM_RUNTIME.md`.

The migration order is:

- Add a semantic validator for the family.
- Add a generic runtime-shape program or generated-program authorization.
- Add randomized legal-shape parity tests with logged seeds.
- Keep exact rows as optimization evidence, regression fixtures, or named
  unsupported-reason exceptions.

The first implemented slice is
`SmallSpatialPointwiseConvContract` `GenericDynamicHW`, which admits legal fp32
direct-buffer 1x1 convolutions with unseen batch/H/W under semantic
1x1/direct-buffer guards and routes them through the existing dynamic-shape
1x1 buffer shader or, for batch-one width-packed cases, the existing as-linear
plan.
`ElementwiseBroadcastDirectBuffer` covers fp32 Vulkan buffer add/mul/sub under
rank/layout/broadcast semantics.
`SequenceCatDirectBuffer` covers fp32 rank-4 direct-buffer dim-2 sequence append
under batch/head/head-dim equality semantics.
`LinearOrMatmulDirectBuffer` covers fp32 rank-2/rank-3 direct-buffer linear
execution under semantic M/K/N compatibility. `FeatureMapToTokensDirectBuffer`
covers fp32 direct-buffer `[N,C,H,W] -> [N,H*W,C]` layout conversion under
runtime batch/channel/H/W metadata while still requiring the current
width-packed zero-offset buffer layout.
`CatAxisDirectBuffer` covers fp32 buffer-backed rank-4 dim-1 cat by semantic
batch/height/width equality and runtime input-count/spatial/channel metadata;
the old channel-cat rowset remains evidence while the current implementation
keeps the channel multiple-of-4 layout constraint.
`BatchNormInferenceDirectBuffer` covers fp32 rank-4 eval-mode buffer batch norm
through runtime N/C/H/W and feature-count semantics.
`TokenPrefixCatAddDirectBuffer` covers fp32 rank-3 prefix-token concat plus
position-add through runtime batch/token-count/feature metadata when prefix
length is `1` and dim is `1`.
`GQARepeatDirectBuffer` now behavior-authorizes runtime K/V head
materialization for fp32 rank-4 Vulkan buffer tensors, and SDPA may use it for
non-causal, mask-free GQA when both K and V match and the resulting rectangular
rank-3 score tensor matches `RectangularScoresRuntimeShape`. The repeat shader
is no longer bounded by exact source-length rows; downstream score-softmax
admission is owned by runtime rectangular-score semantics and score-element
budget. `DirectDecodeGQASDPADirectBuffer`
now covers fp32 rank-4 non-causal decode GQA by runtime head/source/head-dim
semantics through the existing direct-GQA buffer shader, so finite
Transformer decode source rows are evidence rather than default runtime
admission.
`DirectCausalPrefillGQASDPADirectBuffer` similarly covers fp32 rank-4 causal
prefill GQA and equal-head MHA when query/source sequence lengths are equal and
the direct-GQA shader budgets hold. MHA uses the same direct shader with repeat
factor `1`, while unequal-head MHA without `enable_gqa` remains rejected by
semantic contract.
`SmallNonCausalGQASDPADirectBuffer` covers bounded fp32 rank-4 q>1 non-causal
GQA with target/source lengths up to 64 through the existing direct-GQA shader.
The old small non-causal rows are now evidence and guard fixtures, not runtime
admission bounds.
`DirectNonCausalMHASDPADirectBuffer` covers fp32 rank-4 equal-head non-causal
MHA when the direct-buffer layout and lane-aligned head/value dims hold, using
the same direct shader with repeat factor `1`. Diffusion square rows that meet
those semantic/layout constraints should remain evidence, not production shape
allowlists.
`DiffusionSDPAContract` `SquareSelfAttentionRuntimeShape` and
`CrossAttentionRuntimeShape` now cover mask-free fp32 rank-4 diffusion
attention by runtime head/sequence/head-dim/score-budget semantics, with square
runtime cases preserving the conservative score pre-materialization and
post-softmax clone policy through `DiffusionMaterializedSquareRuntimeShape`.
The square runtime family now covers single-head `head_dim=512` when the
materialized key transpose can remain width-pack compatible (`sequence % 4 == 0`);
non-compatible `512` sequences remain blocked on a direct-buffer materialization
command plan rather than exact-row admission.
`VisionSelfAttentionSDPAContract` `Rank3Head64Scale1RuntimeShape` covers fp32
rank-3 vision self-attention by semantic Q/K/V equality, head dim `64`, explicit
scale `1.0`, and disabled mask/dropout/causal/GQA. Its paired
`SDPAScoreSoftmaxContract` `VisionSelfAttentionScoresRuntimeShape` keeps square
rank-3 vision score softmax on the buffer path for runtime `BH/T`.
`SDPAExecutionPolicyContract` now also has
`TransformerDecodeGQACloneOnlyRuntimeShape`, which covers the decode-GQA
post-softmax clone policy by runtime batch/head/source/head-dim semantics and
score-element budget. The old transformer decode execution-policy row is
evidence for clone behavior, not a source-length production gate.
`DynamicNoOverlapConvTranspose2D`
covers the clean packed-buffer no-overlap transposed-conv family by
kernel/stride/layout semantics. `BatchNormInferenceContract` already follows
the semantic-family model for fp32 4D eval mode and now has randomized legal
shape parity coverage. A generic conv probe showed `conv2d_buffer_float` is
already descriptor-driven for groups-one, dilation-one runtime shapes, but it
uses metadata-packed buffer layouts rather than direct-buffer ownership.
`PackedBufferConv2D` now owns the batch-one metadata-packed runtime family on
the generic `conv2d_buffer_float` branch. The remaining generic conv migration
is explicit direct-output ownership, batched-conv ownership, or a
layout-transition contract, not another exact conv row.

## Phase 4: Export And Lowering

Goal: turn an unmodified inference model into a model-independent Vulkan graph
program without capturing from Vulkan opaque tensors.

- capture on CPU with `torch.export(strict=False)`;
- rewrite captured factory-device arguments;
- upload lifted parameters and buffers to the selected Vulkan device;
- use semantic dynamic families as node-admission predicates;
- represent unsupported work as explicit partitions or fail-loud nodes;
- lower transition contracts onto graph edges;
- emit a stable graph-coverage and parity census;
- execute the first lowered program through existing eager Vulkan kernels.

Current status: the Python correctness executor has model-independent passes
for inference-wrapper normalization, packed-context ordering, static factory
and lifted-literal constants, explicit bool tensor placement, proven-identity
advanced indexing, and static GQA head repetition. The generic bounded
boolean-mask SDPA runtime family converts PyTorch keep masks to additive buffers
on device. A four-token HY-MT prefill now executes the complete Python program
with zero lower-time unsupported nodes, CPU fallback, sync readback, or
deferred-value creation. Numerical parity, repeated-output lifetime, dynamic
guards, submit, memory, and latency evidence remain open before this result can
contribute to a Migration deletion gate. The first generic C++ plan slice now
executes eligible tensor-only SSA graphs without Python per-node callbacks; the
remaining work is to extend that executor and its ownership model, not add a
corpus-specific route.

Exit criteria:

- DAv2 runs through the product graph API with Vulkan/eager parity;
- no diagnostic clone barriers are required for correctness;
- unsupported nodes are named and counted rather than hidden by fallback;
- graph capture/lowering has model-independent tests and random-shape fixtures.

## Phase 5: C++ Graph Plan Executor

Goal: remove Python per-node execution and make output/lifetime ownership
structural.

- define an immutable generic graph-plan schema;
- preallocate input, constant, temporary, output, workspace, and parameter
  slots from SSA lifetimes;
- build descriptor and barrier plans once;
- attach execution to a stream/timeline completion token;
- keep public outputs generation-safe across repeated invocations;
- give stateful inputs an explicit update or invalidation protocol.

Current status: `VulkanGraphPlan.v1` is an immutable C++ Tensor-SSA plan for
fully bound, non-mutating Vulkan/composite operators with single-Tensor returns.
It owns operator handles and constants, validates use-count/last-use metadata,
releases non-escaping values after last use, rejects concurrent invocation, and
checks each instruction for implicit host boundaries. A multi-instruction
linear/GELU/residual graph executes repeatedly without Python node callbacks and
preserves an earlier live output. Unsupported plan structure reports a reason
and retains the Python correctness executor. HY-MT currently stops plan
compilation at the non-Tensor return from `aten::_assert_tensor_metadata` and
therefore remains Python-executed. Effect-only/control values, nested dynamic
arguments, list/tuple and multi-output values, program memory slots,
descriptors, submission/completion ownership, and corpus parity remain open.

Exit criteria:

- the C++ executor matches the Python correctness executor;
- repeated no-readback execution cannot overwrite live outputs or overflow the
  host stack;
- fallback, transition, copy, submit, and memory counters remain explicit.

## Phase 6: Recorded Command Partitions And Fusion

Goal: approach kernel-bound latency through stable program ownership.

- record bounded Vulkan-only partitions against program-owned slots;
- cache by graph, guard, device/driver, capability, layout, and weight version;
- reuse descriptors and pipelines;
- move linear/GELU, residual/layer-scale, token-prefix, normalization, SDPA,
  and elementwise fusion into graph rewrites;
- move generated shader compilation behind a real compiler and pipeline cache;
- add capability-keyed heuristic and bounded autotuning plans.

Exit criteria:

- recorded partitions pass repeated-process correctness and lifetime tests;
- program execution materially reduces control-plane submits and descriptor
  rebuilds;
- a rejected plan cannot become a global default on another adapter.

## Phase 7: Replacement Cleanup

This phase is a background track governed by
`docs/vulkan/CLEANUP_POLICY.md` and `docs/vulkan/cleanup_ledger.json`. Inventory,
review gates, and already-eligible deletion waves can interleave with earlier
phases. Deletions that depend on graph evidence are consequences of Phase 5/6
progress, not prerequisites for the C++ executor.

Delete only after replacement parity against supported plain eager and graph
defaults, with the baseline artifact recorded at deletion time:

- speculative eager deferred bridges and per-consumer materialization hooks;
- replay and compiled-session bridge APIs;
- stack-region proof maps and rejected canary implementations;
- model-specific stack/decoder orchestration;
- obsolete environment toggles and monolithic tests/docs.

Retain the eager kernel library, semantic admission contracts, transition
rules, packed-weight caches, streams/events, capability profiles, and compact
runtime scoreboards.

## Phase 8: Corpus And Milestone Comparisons

Use DAv2, Lotus, HY-MT, PaddleOCR, and Gemma to measure graph coverage,
explicit partitions, parity, memory, transitions, submits, and latency. Use
CUDA and DirectML only after a graph-runtime milestone or before claiming a
model/device gate.

Do not let comparison deltas choose daily one-off Vulkan route additions.
