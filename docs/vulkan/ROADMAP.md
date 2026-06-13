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
  `SafeViewReshapeContract` `ViewMaterializedDirectBuffer` and
  `ReshapeAliasDenseBufferDirect`, plus `BatchNormInferenceContract`
  `BufferFloat4D` and `MaterializedBufferFloat4D`.
  `ElementwiseBroadcastContract` `FloatTensorTensorBufferBroadcast` records
  the first float tensor/tensor buffer-broadcast envelope and has a production
  metadata/provenance canary that runs after the existing buffer route is
  selected.
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
- `ShapeEnvelope` v1 now backs the two generated C++ source-of-truth fixtures,
  `ChannelCatContract` `Rank4Dim1BufferView` and `EmbeddingLookupContract`
  `SmallBoundedLookup`, plus the first layout/materialization runtime fixture,
  `SafeViewReshapeContract` `ViewMaterializedDirectBuffer`, and the matching
  `_reshape_alias` direct-buffer fixture,
  `SafeViewReshapeContract` `ReshapeAliasDenseBufferDirect`, plus the first
  checked-in-case runtime fixtures for `BatchNormInferenceContract`
  `BufferFloat4D` and `MaterializedBufferFloat4D`, plus
  `ElementwiseBroadcastContract` `FloatTensorTensorBufferBroadcast`.
  The schema captures symbolic dims, min/max, values, multiples, optional
  dims, generic `broadcast_compatible` relationships, aggregate bounds, layout
  and capability requirements, policies, positive cases, adjacent negatives,
  and fuzz hints for validation and codegen.
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
  `channel_cat_contract.json`, with cross-input iteration and match result
  construction still handwritten.
- `EmbeddingLookupContract` `SmallBoundedLookup` has the second generated C++
  artifact: the same generator emits `ExecutionContractsEmbeddingLookupSpec.h`
  from the `ShapeEnvelope` v1 data in `embedding_lookup_contract.json` for
  metadata, route label, simple bounds, and matcher helper predicates. The
  token-batch row remains handwritten until it gets its own fixture.
- Tensor provenance/value traces can carry optional contract-admission
  metadata for producers that pass an existing match. BatchNorm canaries
  distinguish direct `BufferFloat4D` admission and materialized
  `MaterializedBufferFloat4D` admission without changing the executed
  `buffer_inference_4d_float` route label. ElementwiseBroadcast uses the same
  provenance path after the existing `aten::binary_op.buffer_float` route has
  already been selected.

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
