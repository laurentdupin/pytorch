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

- JSON contract spec fixtures exist for `ChannelCatContract` and
  `KVCacheAppendContract` `SequenceAppend` and
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
  `TokenPrefixCatAddDirectBuffer` owns runtime-shape admission for legal fp32
  rank-3 prefix-token concat plus position-add; corpus tests preserve the old
  observed envelope without an exact-row contract.
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
  `ChannelCatContract` `Rank4Dim1BufferView` and `ElementwiseBroadcastContract`
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
  ShapeEnvelope-backed ChannelCat and both SafeViewReshape
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
- Embedding lookup now has one production admission model:
  `EmbeddingLookupDirectBuffer`. The broader semantic route superseded the
  unreachable finite token-batch and small-bounded matcher, so its JSON spec,
  generated helper, duplicate dispatch, and exact contract source were
  deleted. Remaining device-produced index tensors need a value-proof/error
  contract rather than another shape row.
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
its sequence extent is governed by the shared overflow-safe score-element budget
rather than the former 640 ceiling. Exact-SHA `207730deaa2` admits and proves
the Lotus `[1,1,784,512]` row while preserving the over-budget hard fail.
Non-compatible `512` sequences remain blocked on a direct-buffer materialization
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
on device. A four-token HY-MT prefill now executes the complete program as an
immutable C++ plan with zero lower-time unsupported nodes, CPU fallback, sync
readback, or deferred-value creation. All 64 mutable detach nodes satisfy a
generic single-user fresh-chain proof and are rewritten functionally; aliased
or branched cases remain fail-closed. Numerical parity, repeated-output
lifetime, dynamic guards, submit, memory, and latency evidence remain open
before this result can contribute to a Migration deletion gate. The remaining
work is to extend the generic executor and its ownership model, not add a
corpus-specific route.

The exact-SHA DAv2 evidence also proves both remaining `aten::relu_`
nodes consume single-use, non-aliasing results from functional `aten::conv2d`.
The schema/alias proof rewrites only those fresh mutations; input aliases,
views, and branched values remain fail-closed. Normal and alternate DAv2 shapes
then execute a 404-instruction C++ plan with exact graph-versus-eager parity
and zero fallback, readback, or deferred values. Exact-SHA DAv2 and PaddleOCR
artifacts now also cover repeated live outputs, top-level submission ownership,
same-process peak memory, and supported-default latency for their recorded
shapes. They clear those recorded-shape bars, not the wider corpus and resource
ownership required by a subsystem deletion gate.

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

Current status: `VulkanGraphPlan.v9` is an immutable C++ IValue-SSA plan for
fully bound, non-mutating Vulkan/composite operators with any schema-declared
return count. It owns operator handles and constants, preserves ordered
effects, represents multi-schema returns and typed flat lists, validates
use-count/last-use metadata, releases non-escaping values after last use,
rejects concurrent invocation, and checks every instruction for implicit host
boundaries. Checked integer `add`, `sub`, `mul`, and `floordiv` instructions
consume immutable symbolic-size reads without Python callbacks. Unsupported
plan structure or scalar type reports a reason and retains the Python
correctness executor.

The first bounded storage-reuse rule executes functional `aten::relu` in-place
only for a non-escaping value at exact last use with unique Vulkan storage and
no graph-input, constant, duplicate-live-value, or metadata-view alias. The
plan reports candidate and accepted reuse counts. This is a direct Stage 2
lifetime result, not a model route or a claim that the program arena exists.

Compatible plans own the outer normal Context recording transaction for the
whole invocation, including direct-buffer `aten::lift_fresh_copy` and bounded
`VulkanGraphRegionPlan` instructions. Frequency and large-linear maintenance
boundaries become graph-owner checkpoints, bounded conv regions inherit the
next outer submission token for their scratch, and the plan exposes the final
token by invocation generation. Command-free plans complete without a synthetic
token. Repeated executions preserve an earlier live output.

`VulkanGraphPlanningContext` now supplies the model domain, execution phase,
packed-layout preference, and optional fixed graph-input shape explicitly.
Those fields participate in the program key, are applied while graph lowering
creates packed contexts, and are stored and reapplied by the immutable C++ plan
at invocation. This completes the plumbing prerequisite for removing label and
tensor-shape inference. The removal gate remains open until supported HY-MT and
PaddleOCR eager/graph runs record lane, residency, fallback/readback, and
peak-memory parity on 8 GB adapters.

Exact-SHA `4b688faac33` DAv2 and PaddleOCR evidence records complete executor
outcomes on both shapes. Liveness-owned ReLU reuse with a 32-job cadence records
10 DAv2 and 11 PaddleOCR submissions per inference, down from 13 and 14. Their
30-sample graph medians remain below supported eager, and all
first/repeat-with-live-output peaks remain inside the 5% memory gate. These are
recorded-shape results, not corpus-specific production routes or whole-corpus
deletion evidence.

The four- and five-token HY-MT probes execute the complete 2,668-instruction
plan with top-level submission/completion ownership, recorded numerical parity,
zero graph fallback/readback, repeated live outputs, and peak memory inside the
5% gate. The exact-SHA 32-job cadence reduces submissions from 114 to 88 per
inference. Exact-SHA checked evidence now records 30 alternating latency samples
for both guards. The four-token graph is 25.9% slower than plain eager while the
five-token graph is 21.4% faster, so latency no-regression remains open. Plain
eager also retains its legacy `DepthDiffusion` attention lane and host
boundaries; lane parity remains open.

Exact-SHA checked HY-MT decode evidence now compiles separate first-step and
second-step 2,732-instruction plans with 66 inputs and 65 outputs. All 64
key/value outputs from the first guard variant feed the second guard variant
directly on Vulkan. The chained replay uploads only the next token and mask,
performs no state readback, preserves the first generation's outputs, and stays
inside correctness and memory gates. Its 30-sample graph medians remain
41%-65% slower than eager and plain eager retains the legacy lane, so this
closes the explicit state-protocol item without clearing latency or lane gates.
Exact-SHA `8b60bf3ba4a` preallocates the host boxed-SSA invocation workspace,
liveness bytes, dispatcher stack, and alias-safe typed list recipes. Descriptor
and barrier plans, recorded command partitions, general dtype/rank/dynamic
resource slots, deeper operator-schema containers, and mutable/in-place state
protocols remain Phase 5/6 work. The next performance work should reduce these
per-inference graph costs while preserving the proven 32-job boundaries;
exact-op kernel tuning is secondary until attribution shows GPU work is
dominant.

Exact-SHA `46ece5d7dc9` removes 48 statically proven inference-dropout identities
from DAv2 before plan construction, reducing the plan from 404 to 356
instructions without changing submissions or memory. Continue this graph-level
fixed-cost direction only for semantic identities proven from exported
arguments; the Phase 6 target remains recorded partitions over program-owned
Vulkan resources rather than an accumulating list of operator exceptions.

Exact-SHA `e536f16cf36` also consumes the existing fresh single-user detach
functionalization proof under inference. HY-MT removes 64/64 proven detach
dispatches and reduces its plan from 2,732 to 2,668 instructions without changing
its 88-checkpoint cadence, parity, fallback/readback, or memory gate. Arbitrary
detach aliases remain untouched. Single-sample timing is mixed, so this is a
bounded control-plane reduction rather than a latency claim; integration and
recorded resource ownership remain higher priority than further identity rules.
The checked 30-sample follow-up confirms a mixed guard-dependent latency result
and records the eager lane mismatch rather than promoting it as parity.

Exact-SHA `c8332a964bb` precomputes and validates per-instruction SSA release
lists during immutable plan construction. Invocation no longer validates the
whole plan or rescans argument/output recipes to rediscover last uses. DAv2 keeps
its exact parity, 10-submission cadence, and memory envelope. This is a direct
step toward program-owned resource slots, but it does not yet preallocate Vulkan
tensor storage, descriptors, barriers, or recorded partitions and therefore
does not make a migration subsystem delete-ready.

Exact-SHA `e00b4f0aa8b` upgrades the current plan to v9 and completes the first
bounded Vulkan tensor-resource arena. Nonescaping exact-shape fp32 linear and
add-layernorm results receive stable descriptors, with schema-alias components
extending last use or rejecting an escaping writer. Two arena generations are
reused only after submission completion and TensorImpl/storage exclusivity;
otherwise execution spills. Partial failure poisons the generation, and plan
destruction releases or timeline-retires every safe slot independently. Both
DAv2 guards report four slots over 80 planned resource values and 13
alias-extended lifetimes with exact graph/eager parity. The qualifying RX 9070
run completed 8,372 checked invocations and 33 recaptures in 600.540 seconds,
with 33 immediate releases, zero unsafe-slot leaks or retirement failures, and
all registered memory/fallback/readback gates passing.

The preceding exact candidate `5d9001ebcc7` also treated convolution as a
stable-output writer. Its ten-minute soak failed the final-live and replacement
peak limits, and writer-family isolation found convolution physical-view
aliases unsafe at plan destruction. That writer was removed rather than kept
behind a flag. It may be reconsidered only after physical-view ownership is
explicit and repeated plan destruction records zero unsafe slots. This first
arena does not claim descriptor reuse, explicit barrier planning, recorded
commands, arbitrary dtype/rank, or concurrent/multi-invocation flight, and it
does not by itself close a legacy-subsystem deletion gate.

Exact-SHA `520e4ae8ee6` replaces the list-wrapped V1 graph-region ABI with its
enforced Tensor-to-Tensor contract, and exact-SHA `f0d1d1766df` adds an fp32
ReLU instruction. Their arena-writer extensions are rejected after the
`4b45cb8121a` qualified soak records 84 unsafe slots across 28 recaptures.
Exact-SHA `da216f221f5` removes those writer cases while preserving their
semantic lowering and adds zero unsafe-slot and retirement-failure conditions
to the soak gate.

The same corrected source lowers the strict scaled-QK, softmax, probability-V
attention chain as one semantic instruction and owns its final output. Both
DAv2 guards lower all 12 blocks, shrink from 336 to 288 instructions, and own
70 writers over 92 values in five slots with zero bypass and bit-exact
graph/eager parity. The corrected ten-minute RX 9070 soak checks 8,600
invocations and 34 recaptures with zero unsafe slots, zero retirement failures,
and bounded memory. This closes the contiguous transformer semantic-partition
gate, not recorded execution: the next Phase 6 candidate must record useful
work inside that partition without restoring fine-grained primary buffers.

The resource ABI now transports storage type, GPU memory layout, and execution
layout per slot instead of assuming that shape and dtype fully describe a
target. Direct width-packed buffers retain the proven allocator path, and
standalone buffer-view targets fail closed until base-storage and physical-view
metadata are represented. Operator admission is unchanged; the next recording
candidate must use this descriptor substrate to own a contiguous transformer
span rather than restoring isolated rejected writers.
At that historical source, `eec01e49a15` keeps both DAv2 guards exact and
zero-bypass with the same 86 writers and memory deltas under 2.01%. This closes
descriptor transport, not contiguous ownership or recorded execution; the
later lifetime result above supersedes those writer-ownership counts.

Exit criteria:

- the C++ executor matches the Python correctness executor;
- repeated no-readback execution cannot overwrite live outputs or overflow the
  host stack;
- the RX 9070 long-session gate completes at least ten minutes and 3,000
  per-frame-readback invocations with periodic guard recapture, numerical
  parity, and bounded live/high-water memory;
- fallback, transition, copy, submit, and memory counters remain explicit.

## Phase 6: Recorded Command Partitions And Fusion

Goal: approach kernel-bound latency through stable program ownership.

Current attribution status: exact-SHA `28d8f7b3133` measured the existing
ten-submit graph topology and exact unrecorded candidates at five, two, and one
submission. GPU work is stable at 23-25 ms, but five is latency-neutral and
fails the 5% repeat-live memory gate, while two and one are materially slower
and fail memory. Ten remains the supported unrecorded baseline. The first
recorded candidate at `f80ad5960893` proved stable rank-N linear output storage
and replayed nine primed writers, but one primary command buffer per writer
raised GPU work by more than 5% and increased inter-submit gaps by 9-12 ms. It
was rejected and its recording machinery deleted by `e13bdc8d517`; only the
generic rank-N storage correction and bug-class regression remain. The next
candidate must pre-record useful contiguous multi-instruction work against
stable resources and descriptors without increasing primary command-buffer
count per submission. Merely widening checkpoint cadence or replaying isolated
writers is explicitly rejected.

- record bounded Vulkan-only partitions against program-owned slots;
- cache by graph, guard, device/driver, capability, layout, and weight version;
- reuse descriptors and pipelines;
- move linear/GELU, residual/layer-scale, token-prefix, normalization, SDPA,
  and elementwise fusion into graph rewrites;
- move generated shader compilation behind a real compiler and pipeline cache;
- add capability-keyed heuristic and bounded autotuning plans.

Corpus progress at exact-SHA `3d666cbacd7`: Lotus now passes the compiled
DTensor preflight and runs end to end on RX 9070 after bounded retirement for
uncached inference sliding-window convolution weights above 8 MB prevents VAE
decode stack overflow. The one-repeat 224x224 run is finite and completes in
2.790 seconds, but 11 CPU fallbacks and two sync readbacks keep its clean-route
gate open. This is a generic convolution lifetime correction, not a Lotus route.
Fresh companion rows keep the remaining boundary explicit: PaddleOCR completes
a 224x224 generated-document pass with zero fallback/readback, while HY-MT's
one-token pass completes all 225 linears but retains 28 fallbacks and nine
readbacks in caller-owned generation control. Those scalar loop decisions are
not a reason to add model orchestration to eager Vulkan; the existing prefill
and two-step decode graph evidence remains the model-core replacement gate.

Exit criteria:

- recorded partitions pass repeated-process correctness and lifetime tests;
- program execution materially reduces control-plane submits and descriptor
  rebuilds;
- recorded execution does not increase primary command-buffer count enough to
  erase the host-recording savings;
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

Gemma E2B is split into explicit prerequisites rather than treated as routine
graph coverage. The first landed slice recognizes an oversized direct-input
embedding and executes a bounded host gather/upload prelude while preserving a
C++ Vulkan body. The bounded pure derived-index subgraph used by the
conditional-generation wrapper is now admitted with explicit escaping-output
uploads and integer-bounds provenance. BF16 embedding constants now use the
same explicit host partition until a generic native gather is proven; the real
checkpoint lowers both its tied token table and per-layer table this way.
BF16-preserving tensor-scalar/tensor-tensor arithmetic, aligned native linear
output, aligned BF16 upcast, GELU/tanh, rotary concatenation, rank-5 metadata
expansion, and bounded masked attention through width 512 are now established.
The checkpoint-backed one-token graph executes end to end with zero
fallback/readback and explicit host partitions. It preserves the CPU argmax and
10/10 top-10 entries after adopting a generic eight-lane FP32 partial-sum
reduction for BF16 buffer linear, but 1.6% of logits still miss the strict
elementwise parity gate due to accumulated error before the final hidden/logit
projection. The next Gemma gates are full CPU parity, multi-token generation
and decode coverage, memory, and latency distributions. Repeated one-token
execution with prior outputs live is now bit-stable. Padded multi-row odd-K
linear remains generic dtype breadth work; it fails loudly while single-row
odd-K remains supported. Static BF16 graph-linear bias and dynamic
FP32-to-BF16 graph casts are bit-exact and are no longer pending items. The
remaining odd-K breadth item does not block the current one-token graph.
Whole-model float32 upload and monolithic buffers above the storage-buffer
binding range are not candidate routes.

The producer-side device contract now exposes physical UUID, valid Windows
LUID, optional normalized PCI address, and pipeline-cache UUID. Process-local
selection consumes `PYTORCH_VULKAN_VISIBLE_DEVICE_UUID` before adapter/context
construction and presents the selected physical adapter as `vulkan:0`; current
Windows evidence covers all three installed adapters by UUID and LUID. Release
readiness still requires a clean exact-commit Torch/torchvision wheel pipeline,
manifested hashes and toolchain/backend identities, signatures, and an offline
installation test. Local source-build evidence does not satisfy that release
gate.
