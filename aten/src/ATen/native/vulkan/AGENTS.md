# Vulkan Backend Instructions

This directory follows the repo-level Vulkan planning memory in
`docs/vulkan/`. Read those files before changing Vulkan production code.

## Direction

The backend direction is contract-driven hybrid Vulkan execution planning:

- `KernelFamilyContract` for reusable kernel legality, route labels, fallback
  behavior, and generated parity/negative tests.
- `RegionContract` for planned multi-op regions, shape/capability keys,
  binding validation, lifetimes, and replay readiness.
- `LayoutTransitionContract` for explicit representation changes,
  materialization reasons, readback legality, and provenance.

DAv2, Lotus, HY-MT, PaddleOCR, and Gemma E2B are coverage evidence. Model names
are allowed in tests, benchmarks, docs, diagnostics, and explicit opt-in
model-family APIs. They are not allowed as implicit generic op dispatch
criteria.

## Production Route Rules

- New route predicates must be contract-named, not model-named.
- Exact tuples may exist only in a contract table with contract name, family,
  dtype/layout/rank/options, tuple fields, materialization policy, parity test
  or artifact, device coverage, negative tests, and fallback behavior.
- Unsupported or unproven shapes must reject or take an explicit safe fallback.
- Do not broaden a route by formula until adjacent shapes and layouts have
  parity and negative coverage.
- Do not add hidden CPU fallback/readback, fake storage or `data_ptr`, or
  permanent feature flags.

## Review Expectations

Before finishing a Vulkan change, check `docs/vulkan/REVIEW_CHECKLIST.md`.
If a temporary exception is needed, update
`docs/vulkan/TEMPORARY_EXCEPTIONS.md` in the same change with expiry and
migration target.
