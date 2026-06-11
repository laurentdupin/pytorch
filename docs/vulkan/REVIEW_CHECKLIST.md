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
- Temporary feature flags have expiry and migration target; permanent flags are
  not used to carry incomplete behavior.

## Scope Control

- No unrelated Vulkan source edits.
- No benchmark behavior changes unless the task explicitly requests them.
- No capability-profile row is treated as GPU emulation; RX 9070, RX 6700 XT,
  and GTX 1080 remain real-hardware validation rows.
- No `.ci/docker` edits unless rebuilding Docker images is intentional.
- Documentation updates match the code behavior and do not claim unvalidated
  model readiness.
- Rerun the real-model matrix after default backend behavior changes or before
  claiming or raising model gates. Pure docs, spec-helper, or fixture-only
  changes do not require a matrix refresh unless they reveal stale gate claims.
