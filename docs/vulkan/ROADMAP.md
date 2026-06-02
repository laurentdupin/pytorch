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

## Phase 3: Migrate Next Contract Family

Use `agent_space/vulkan_contract_migration_plan.md` as the decision record.
The current recommended next family is `TransformerGQASDPAContract` because
HY-MT-specific naming remains in SDPA route policy and related helpers.

Migration shape:

- rename production predicates and labels to contract terms
- keep model names in tests, docs, and artifacts
- keep existing legality unless the task explicitly asks for new coverage
- add or update negative tests for adjacent unsupported shapes
- record any remaining temporary exception

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
