# Vulkan Review Checklist

Use this checklist for Vulkan backend changes.

## Route Policy

- No new production predicate is named for DAv2, Lotus, HY-MT, PaddleOCR,
  Gemma, or another model.
- Exact tuples live in a contract table with contract name, family, tuple
  fields, materialization policy, parity evidence, device coverage, and
  fallback behavior.
- Any model-name production occurrence is either removed or listed in
  `TEMPORARY_EXCEPTIONS.md` with expiry and migration target.
- Broad route enablement is backed by positive parity tests and adjacent
  negative tests.

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
- Materialization and layout transition behavior is part of the contract, not a
  hidden side effect.
- Temporary feature flags have expiry and migration target; permanent flags are
  not used to carry incomplete behavior.

## Scope Control

- No unrelated Vulkan source edits.
- No benchmark behavior changes unless the task explicitly requests them.
- No `.ci/docker` edits unless rebuilding Docker images is intentional.
- Documentation updates match the code behavior and do not claim unvalidated
  model readiness.
