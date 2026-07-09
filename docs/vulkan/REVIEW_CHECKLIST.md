# Vulkan Review Checklist

Use this checklist for Vulkan backend changes.

## Route Policy

- No new production predicate is named for DAv2, Lotus, HY-MT, PaddleOCR,
  Gemma, or another model.
- Capability-profile tests route only from normalized feature bits after
  intersection with the live adapter. Profile IDs and GPU-family labels are
  test selectors, not dispatch predicates.
- Capability-profile admission changes include a canary when an optional
  feature, limit, or layout policy can affect route selection.
- Exact tuples live in a contract table with contract name, family, tuple
  fields, materialization policy, parity evidence, device coverage, and
  fallback behavior.
- New exact tuples or finite envelopes include execution-contract metadata:
  `contract_name`, `family_name`, `tuple_id`, `evidence_id`, `guard_id`,
  `fallback_policy`, and `materialization_policy`.
- Any model-name production occurrence is either removed or listed in
  `TEMPORARY_EXCEPTIONS.md` with expiry and migration target.
- Broad route enablement is backed by positive parity tests and adjacent
  negative tests.
- Exact or finite contract-row changes update an existing
  `test/vulkan_contract_specs/*.json` fixture, add a new fixture, or explicitly
  document why no fixture is suitable yet.
- Generated C++ contract fixtures that are ready for schema validation use
  `ShapeEnvelope` v1 or document why they remain pre-envelope. The envelope
  records symbolic dims, bounds, relationships, aggregate bounds,
  layout/capability requirements, policies, positives, adjacent negatives, and
  fuzz hints without changing route legality.
- Correlated finite tables use generic `ShapeEnvelope` `sparse_rowsets` instead
  of independent value sets that would admit unproven cross-products.
- ShapeEnvelope specs with generated C++ helper headers are listed in
  `test/vulkan_contract_specs/generated_cpp_manifest.json`, and manifest
  validation covers header presence, byte-for-byte regeneration, helper
  markers, and full ShapeEnvelope coverage.
- Before adding or mirroring exact rows, consult
  `test/vulkan_contract_specs/contract_spec_utils.py --contract-coverage-census`
  so the change is framed against existing ShapeEnvelope/generated-helper
  coverage, live contracts without JSON specs, and temporary-exception linkage.
- Before adding exact rows for an op family with a dynamic program contract,
  confirm that the dynamic family rejected with a named unsupported reason or
  that the row is explicitly performance evidence, a regression fixture, or a
  negative guard. Do not use exact rows as the default answer for unseen legal
  shapes.

## Fallback And Readback

- CPU fallback is explicit, counted, and visible in diagnostics.
- Synchronous readback is explicit, counted, and justified.
- No route hides fallback/readback to make a benchmark look device-resident.
- No fake storage, fake `data_ptr`, or fake allocator stream hook is introduced.

## Tests

- Numerics use `assertEqual` for tensor equality where appropriate.
- On-device numerics use device-generic tests where feasible.
- Multiple input families use `@parametrize` where feasible.
- Adjacent unsupported shapes/layouts/dtypes have negative tests.
- Dynamic program families include randomized legal-shape parity tests that
  generate fresh shapes per run and print or log a reproduction seed.
- Existing model-provenance tests keep model names only as provenance, not as
  production route criteria.

## Contracts

- New or changed route logic maps to a `KernelFamilyContract`,
  `RegionContract`, or `LayoutTransitionContract`.
- Contract labels are stable and reusable across model evidence.
- Exact-row metadata is complete and points to evidence and guard coverage; it
  is a review guardrail, not proof that broader shapes are legal.
- Materialization and layout transition behavior is part of the contract, not a
  hidden side effect.
- Producer-consumer materialization-edge specs distinguish direct-safe evidence
  from materialization-required rows and do not become blanket softmax/BMM,
  view/alias, or other broad layout-transition rules without proof.
- Temporary feature flags have expiry and migration target; permanent flags are
  not used to carry incomplete behavior.

## Admission Diagnostics

- New contract-admission diagnostics are opt-in through
  `PYTORCH_VULKAN_CONTRACT_ADMISSION_LOG`; they do not write broad logs on
  successful execution when the env var is unset.
- Admission diagnostics stay separate from `PYTORCH_VULKAN_OP_HIT_LOG` and
  tensor provenance/value traces. Op-hit logs describe executed routes; tensor
  provenance describes accepted output metadata; admission diagnostics describe
  candidate accept/reject decisions.
- Wiring a new contract family to admission diagnostics must not change route
  ordering, accepted shapes, route labels, fallback/readback behavior,
  materialization policy, match-result assembly, or shader execution.
- The MVP payload uses stable metadata and reason taxonomy only:
  `event`, `contract_name`, `family_name`, `tuple_id`, `outcome`, `phase`,
  `predicate`, `reason_code`, and `source`. Do not add raw shapes, tensor ids,
  storage ids, allocation ids, or tensor values without a scoped diagnostic
  task and review.
- Emit at most one stable first-failure event per candidate matcher attempt,
  plus one accepted event for matched candidates. Reason codes should name the
  failed contract predicate, not the model or benchmark that exposed it.
- Each newly wired contract family includes focused enabled-log coverage for
  at least one accept and one reachable reject case. Existing contracts are not
  retroactively blocked merely because they do not emit admission diagnostics
  yet.

## Scope Control

- No unrelated Vulkan source edits.
- No benchmark behavior changes unless the task explicitly requests them.
- Benchmark-local distributed/import shims stay import-only and
  single-process. Do not fake compiled `torch._C` distributed or DTensor APIs to
  make a workload reach Vulkan telemetry.
- No capability-profile row is treated as GPU emulation; RX 9070, RX 6700 XT,
  and GTX 1080 remain real-hardware validation rows.
- No `.ci/docker` edits unless rebuilding Docker images is intentional.
- Documentation updates match the code behavior and do not claim unvalidated
  model readiness.
- Vulkan performance candidates consult and update
  `test/vulkan_contract_proofs/performance_plan_evidence_manifest.json` when
  accepting a fix, adding a canary, rejecting a slower route, or blocking an
  unsafe/correctness-failed topology. Do not repeat a rejected candidate unless
  its recorded revisit conditions are met.
- Replay/compiled-session paths are quarantined under
  `docs/vulkan/REPLAY_RETIREMENT.md`. Do not add new replay benchmark modes,
  public replay bridge APIs, or replay-backed default routes. New region work
  should target runtime-generated command lists instead.
- Rerun the real-model matrix after default backend behavior changes or before
  claiming or raising model gates. Pure docs, spec-helper, or fixture-only
  changes do not require a matrix refresh unless they reveal stale gate claims.

## Graph Runtime

- New performance work follows `docs/vulkan/GRAPH_RUNTIME.md`.
- New eager deferred placeholders or per-consumer materialization protocols are
  rejected; graph fusion owns future-consumer knowledge.
- Graph capture happens from CPU until Vulkan tensors have a supported
  FakeTensor/storage representation.
- Node admission uses semantic contracts rather than exact observed shapes.
- CPU partitions are explicit, counted, and visible in the program plan.
- Layout transitions are graph-edge plan steps with reason and byte budgets.
- Temporary lifetime and slot reuse come from SSA first/last use, not
  retire-time inference.
- Recorded command buffers bind program-owned stable slots and descriptors.
- Stateful values define update, version-invalidation, or partition-boundary
  behavior before recording is enabled.
- Runtime-generated shaders are graph codegen; production execution does not
  shell out to a manually configured compiler executable.
- Legacy replay, stack-region, or deferred code may be changed only for a
  correctness fix, migration hook, or deletion enabled by replacement parity.
