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
  `Batch1Heads4Factor4Sequence100To116Dim128`.
- `test/vulkan_contract_specs/contract_spec_utils.py` owns shared spec loading,
  case iteration, log naming, and expected-negative helpers.
- `TestVulkanGovernance` discovers all spec fixtures, validates shared schema,
  and checks fixture metadata against live contract sources.
- `ChannelCatContract` has the first generated C++ typed-row/helper artifact:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `ExecutionContractsChannelCatSpec.h` from `channel_cat_contract.json`, with
  cross-input iteration and match result construction still handwritten.
- `EmbeddingLookupContract` `SmallBoundedLookup` has the second generated C++
  artifact: the same generator emits `ExecutionContractsEmbeddingLookupSpec.h`
  from `embedding_lookup_contract.json` for metadata, route label, and simple
  bounds. The token-batch row remains handwritten until it gets its own fixture.

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
