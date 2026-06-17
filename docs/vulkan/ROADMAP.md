# Vulkan Roadmap

## Operating Principle

Each task should convert observed model failures into reusable contracts,
generated tests, or structural diagnostics. Stop after a new blocker unless the
task explicitly authorizes fixing it.

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
  `TokenPrefixCatAddContract` records the bounded rank-3 prefix-token concat
  plus position-add rowset and feeds a generic fused Vulkan helper for the
  observed token-preparation envelope.
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
  token-batch row remains handwritten until it gets its own fixture.
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
- `SmallMetadataPaddedConv2DContract` `MaterializedBufferInput2x2` now uses
  the generic ShapeEnvelope C++ simple-bounds generator path:
  `ExecutionContractsSmallMetadataPaddedConv2DSpec.h` provides contract
  identity, metadata, exact input/weight/options bounds, and helper predicates
  for the existing matcher, including input/weight channel equality.
  Tensor-info extraction, materialization dispatch, op-hit logging, fallback
  visibility, and match-result construction remain handwritten.
- `LinearGeluBridgeContract` `BackboneMlpHidden384To1536` now uses the generic
  ShapeEnvelope C++ simple-bounds generator path:
  `ExecutionContractsLinearGeluBridgeSpec.h` provides contract identity,
  metadata, rank/shape/packed-weight/options bounds, minimum flattened rows,
  and result-policy constants for the existing matcher. Deferred registry
  ownership, alias retargeting, materialization on non-GELU consumers,
  fused-GELU execution, op-hit labels, rank-3 equality, and match-result
  construction remain handwritten.
- `SmallSpatialPointwiseConvContract` `SparseProjectionRows` now uses the
  generic ShapeEnvelope C++ sparse-rowset generator path:
  `ExecutionContractsSmallSpatialPointwiseConvSpec.h` provides contract
  identity, per-row metadata, input/weight channel equality, the 39 correlated
  projection rows, exact lookup helpers, and the generated factorized
  depth-vision projection helper for the 108 cross-adapter proven shapes. The
  factorized helper admits only the approved channel-pair group crossed with
  the approved spatial-pair group; min/max envelopes and wider independent
  channel/spatial cross-products remain out of scope. Route-policy hard-fail
  rescue, shader-family decisions, family op-hit labels, and match-result
  construction remain handwritten.
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
  rows plus seven `vulkan_clone_probability_before_value_bmm` rows: the Lotus
  `[10,126,126]` proof row and the six existing low-resolution
  VisionSelfAttention rows. This deliberately does not remove the clone or
  change production softmax/BMM dispatch.
- `HostUploadTransitionContract`, `MetadataViewTransitionContract`,
  `FinalReadbackContract`, `IntermediateReadbackTransitionContract`, and
  `SafeContiguousMaterializationContract` are schema-only transition
  reason-bucket specs for existing transition-log evidence. The five-model
  validation collector uses them as source-of-truth mappings for matching
  reason logs only; they do not admit backend routes or change upload,
  metadata-view, copy, materialization, fallback, or readback behavior.
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

## Phase 4: Region And Layout Contracts

Promote DAv2 stack-owner evidence into reusable region-planning rules:

- shape/capability keys
- explicit intermediate escape policy
- binding validation
- descriptor table readiness
- planned-recording diagnostics
- lifetime/provenance proof

Promote layout/materialization evidence into `LayoutTransitionContract` rules:

- metadata view legality
- dense materialization legality
- buffer/texture transition reasons
- readback legality
- finite-value and provenance diagnostics

## Phase 5: Milestone Comparisons

Use CUDA and DirectML comparisons only as milestone checks:

- after a contract family lands
- after a region-planning milestone lands
- before claiming a model gate is ready

Do not let comparison deltas choose daily one-off Vulkan route additions.
