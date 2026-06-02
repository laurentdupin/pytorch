# Vulkan Project Charter

## Mission

Build a contract-driven hybrid Vulkan backend that can execute real model
workloads without accumulating model-name production routes or one-off exact
shape exceptions.

The immediate coverage corpus is DAv2, Lotus, HY-MT, PaddleOCR, and Gemma E2B.
These models are evidence sources. They should produce reusable contracts,
diagnostics, generated tests, and milestone comparisons, not bespoke production
paths.

## Direction

Prefer three contract classes:

- `KernelFamilyContract`: reusable kernel-family legality, route labels,
  fallback policy, device coverage, and parity/negative tests.
- `RegionContract`: multi-op planned execution with shape keys, capability
  keys, lifetime/provenance, binding validation, and replay-readiness
  diagnostics.
- `LayoutTransitionContract`: explicit representation transition legality,
  materialization reasons, readback safety, alias/view rules, and provenance.

Contract work may start with finite proven tuples, but the tuple table must be
named for the backend behavior, not for the model that exposed it.

## Non-Goals

- Do not chase the next exact model shape unless the task explicitly requests a
  diagnostic census.
- Do not add production routes selected by model name.
- Do not hide CPU fallback or synchronous readback.
- Do not fake storage or `data_ptr` behavior to satisfy an API shortcut.
- Do not add permanent feature flags for unproven routes.
- Do not let CUDA or DirectML comparisons drive daily kernel selection. Use
  them as milestone checks.

## Accepted Model-Name Uses

Model names are allowed in:

- benchmark harnesses
- tests and synthetic fixtures
- docs and task cards
- diagnostic labels and artifacts
- explicit opt-in model-family APIs

Model names are not allowed as implicit generic op dispatch criteria unless
the route is recorded as a temporary exception with expiry and migration
target.

## Evidence Sources

Use these artifacts when they exist:

- `agent_space/vulkan_contract_migration_plan.md`
- `agent_space/model_suite_real_life_status_current.md`
- `agent_space/exact_shape_routes.txt`
- `agent_space/model_named_routes.txt`
- `agent_space/five_model_blockers.json`
- contract drafts under `agent_space/*contract*_draft.md`

Do not recursively scan large caches, venvs, model downloads, or PaddleX model
caches.
