# Vulkan Current State

Last bootstrapped: 2026-06-02.

## Repo State Summary

`docs/vulkan` was absent before this bootstrap. Existing direction lived in
ignored `agent_space` artifacts and Vulkan ops markdown files.

The worktree already has a first contract-table migration:

- `aten/src/ATen/native/vulkan/planning/ExecutionContracts.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContracts.cpp`

That table currently represents `SmallSpatialPointwiseConvContract` families
for depth vision projection, OCR projection, and diffusion projection without
adding or removing tuples.

## Coverage Corpus

The five-model corpus is:

- DAv2: primary vision stack-owner and region-planning signal.
- Lotus: diffusion depth pipeline signal for SDPA, cross-attention, pointwise
  projection, UNet concat, resize, and layout/materialization behavior.
- HY-MT: Transformer decode signal for GQA SDPA, GQA repeat, KV-cache append,
  embedding gather, and fallback/readback attribution.
- PaddleOCR: OCR pipeline signal for batch norm, small-spatial pointwise conv,
  grid sample diagnostics, and remaining conv-transpose/fallback pressure.
- Gemma E2B: memory/dtype roadmap signal; current evidence says it is blocked
  before useful Vulkan route coverage by float32 model-weight OOM.

Do not infer production route names from this corpus.

## Existing Audit Artifacts

- `agent_space/vulkan_contract_migration_plan.md`: policy lock and initial
  contract groups.
- `agent_space/model_named_routes.txt`: route-specialization audit with A/B/C/D
  classification.
- `agent_space/exact_shape_routes.txt`: finite tuple audit for conv, SDPA,
  embedding, cat, GQA repeat, batch norm, and safe view/reshape routes.
- `agent_space/five_model_blockers.json`: five-model blocker summary and next
  discovery focus.
- `agent_space/lotus_diffusion_sdpa_contract_draft.md`: draft finite
  `DiffusionSDPAContract` and `DiffusionCrossAttentionContract` evidence.
- `agent_space/lotus_pointwise_projection_contract_draft.md`: finite diffusion
  projection evidence for `SmallSpatialPointwiseConvContract`.

These files are diagnostic inputs. Production code must not depend on
`agent_space`.

## Current Contract Groups

- `SmallSpatialPointwiseConvContract`: first migrated table; keep exact rows
  until broader legality is proven.
- `EmbeddingLookupContract`: finite token-batch and small-bounded embedding
  lookup contract; keep exact rows until broader legality is proven.
- `CatAxisContract`: umbrella for bounded last-dim, channel-dim, and rank-3
  cat patterns.
- `KVCacheAppendContract`: proposed migration target for bounded Transformer
  sequence append.
- `UNetChannelConcatContract`: mostly generic already; keep model provenance in
  tests/docs.
- `GQARepeatContract`: proposed migration target for bounded K/V head repeat.
- `BatchNormInferenceContract`: already contract-shaped for float32 4D
  inference batch norm.
- `DiffusionSDPAContract` and `DiffusionCrossAttentionContract`: finite
  explicit tuple contracts until broader materialization behavior is proven.
- `SafeViewReshapeMaterializationContract`: mostly generic; document alias,
  dense materialization, storage-offset, and provenance rules.
- DAv2 region/stack contracts: best current example of shape keys, capability
  keys, planned regions, binding validation, and replay-readiness diagnostics.

## Validation Caveats

- Model status artifacts can be stale relative to each other. Before changing a
  production route, confirm the relevant current blocker with a bounded smoke,
  focused test, or fresh diagnostic artifact.
- DAv2 stack owner is intentionally safe and does not merge command-buffer
  replay until descriptor ownership and binding validation are ready.
- Some compatibility evidence is device-specific. RX 9070 remains the primary
  optimization signal; RX 6700 XT and GTX 1080 are compatibility checks.
- Gemma E2B is a memory/dtype milestone, not a reason to add exact route
  exceptions.
- PaddleOCR completes smoke coverage in current artifacts but retains high
  fallback/readback budget. Treat it as a contract discovery source until tight
  budgets are established.

## Build Context

On this Windows machine, use the existing Visual Studio CMake build tree from
`build/CMakeCache.txt`. The local cache records Visual Studio 17 2022, x64,
Release, `USE_VULKAN=ON`, `USE_VULKAN_API=ON`, strict SPIR-V, Vulkan 1.3, and
SPIR-V 1.6 targets.
