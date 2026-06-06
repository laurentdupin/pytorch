# Vulkan Current State

Last refreshed: 2026-06-06 at local HEAD
`64e36838f76806db0155ea9889a6e424318d406f`.

## Repo State Summary

The Vulkan backend planning direction is now repo-local in `docs/vulkan`.
Ignored `agent_space` artifacts remain evidence inputs, not production
dependencies.

`ExecutionContracts.*` is the shared contract table for the current bounded
operator-family envelopes. `ExecutionContracts.h` remains the public umbrella
API; implementation is now split across:

- `aten/src/ATen/native/vulkan/planning/ExecutionContracts.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContracts.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsBatchNormInference.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsChannelCat.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsDiffusionSDPA.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsEmbeddingLookup.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsGQARepeat.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsKVCacheAppend.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsLinearGeluBridge.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsMaskedTinySDPA.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsNoOverlapConvTranspose2D.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsSafeViewReshape.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsSDPAExecutionPolicy.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsSDPAScoreSoftmax.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsSmallMetadataPaddedConv2D.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsSmallSpatialPointwiseConv.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsTransformerGQASDPA.cpp`

The table owns finite tuples/envelopes with `ExecutionContractMetadata` for
contract name, family, tuple id, evidence id, guard id, fallback policy, and
materialization policy. Some rows are still exact and temporary; they are
allowed only as guarded contract rows while generated parity/negative coverage
is built. `BatchNormInferenceContract`, `ChannelCatContract`,
`EmbeddingLookupContract`, `GQARepeatContract`, `KVCacheAppendContract`,
`LinearGeluBridgeContract`, `NoOverlapConvTranspose2DContract`, and
`SafeViewReshapeContract`, `SmallMetadataPaddedConv2DContract`,
`SmallSpatialPointwiseConvContract`, `MaskedTinySDPAContract`, and
`TransformerGQASDPAContract`, `DiffusionSDPAContract`,
`DiffusionCrossAttentionContract`, `SDPAExecutionPolicyContract`, and
`SDPAScoreSoftmaxContract` are split into family-specific sources. The former
score-softmax allowlist is now a named, metadata-backed finite contract for
float rank-3 square score tensors with heads `{1, 5}` and sequence
`{504, 640}`. `ExecutionContracts.cpp` now owns the shared metadata
completeness helper rather than an SDPA-specific route-policy bucket.

The current local tree also has a submit-origin diagnostic split for
CPU-to-Vulkan float-buffer conv prepack uploads. That split keeps true tensor
CPU readbacks classified separately and applies the tiny-old-path pending
handling only to the fenced conv prepack upload path. Recent stability work
keeps the prepack-retire drain policy scoped to float-buffer conv prepack
uploads and preserves real tensor CPU readback behavior and diagnostics.

`docs/vulkan/CAPABILITY_PROFILES.md` and
`docs/vulkan/capability_profiles.json` define the first capability-profile
harness. Profiles are reduced feature masks intersected with the live adapter;
they are not GPU emulation and must not route by profile or GPU-family name.
Focused canaries cover manifest shape and C++ ID parity, non-emulation docs,
minimum-profile runtime-policy feature masking, minimum-profile compiled-session
layout clamping, and minimum-profile SDPA qtile admission to the shared path
instead of the subgroup path.

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

- `SmallSpatialPointwiseConvContract`: finite projection rows, now split into
  a family-specific source; keep exact rows until broader legality is proven.
- `NoOverlapConvTranspose2DContract`: bounded float-buffer 2x2 stride-2
  no-overlap transposed-conv envelope. The `Kernel2Stride2FloatBuffer` slice
  has a JSON contract spec with generated positive and adjacent negative
  runtime coverage; preserve unsupported-case fallback outside that envelope.
- `SmallMetadataPaddedConv2DContract`: one proven padded low-channel
  buffer-input materialization tuple, now split into a family-specific source;
  keep adjacent guards.
- `TransformerGQASDPAContract`: bounded Transformer causal/prefill and decode
  GQA SDPA legality with model-neutral naming, now split into a
  family-specific source.
- `MaskedTinySDPAContract`: tiny additive-mask SDPA tuple, now split into a
  family-specific source; keep the exact tuple until broader mask-family
  behavior is proven.
- `DiffusionSDPAContract` and `DiffusionCrossAttentionContract`: finite
  explicit tuple contracts, now split into a family-specific source; keep exact
  rows until broader materialization behavior is proven.
- `SDPAExecutionPolicyContract`: finite execution materialization, softmax
  score, post-softmax clone, and repeat policy contract, now split into a
  family-specific source; keep exact rows until broader layout-transition
  behavior is proven.
- `SDPAScoreSoftmaxContract`: finite float rank-3 square score-softmax
  contract for heads `{1, 5}` and sequence `{504, 640}`, now named and
  metadata-backed in a family-specific source. Keep the temporary exception
  until broader score-softmax/layout behavior is proven or generated spec
  coverage lands.
- `EmbeddingLookupContract`: finite token-batch and small-bounded embedding
  lookup contract; the small-bounded lookup slice has a JSON contract spec with
  generated positive and adjacent negative runtime coverage. Keep remaining
  exact rows until broader legality is proven.
- `CatAxisContract`: umbrella for bounded last-dim, channel-dim, and rank-3
  cat patterns. The `ChannelCatContract` rank-4 dim-1 buffer slice has a JSON
  contract spec with generated positive and adjacent negative runtime coverage.
- `KVCacheAppendContract`: bounded Transformer sequence append and initial
  empty-cache cat rows. Both `SequenceAppend` and `InitialCache` slices have
  JSON contract specs with generated positive and adjacent negative runtime
  coverage. InitialCache positives now log the contract-owned
  `aten::cat.kv_cache_initial_dim2_buffer` op-hit label while unrelated
  direct-buffer cat paths keep their generic labels.
- `UNetChannelConcatContract`: mostly generic already; keep model provenance in
  tests/docs.
- `GQARepeatContract`: finite bounded K/V head repeat contract, now split into
  a family-specific source; keep exact rows until broader legality is proven.
- `BatchNormInferenceContract`: float32 4D inference batch norm, including the
  materialized-buffer layout transition used by current OCR evidence.
- `SafeViewReshapeContract`: finite dense direct-buffer view and reshape-alias
  contract, now split into a family-specific source; document alias, dense
  materialization, storage-offset, and provenance rules.
- `LinearGeluBridgeContract`: pure legality for the deferred linear/GELU
  bridge; registry, alias, and materialization side effects stay outside the
  contract.
- DAv2 region/stack contracts: best current example of shape keys, capability
  keys, planned regions, binding validation, and replay-readiness diagnostics.

## Governance Guardrails

- `test/test_vulkan.py::TestVulkanGovernance` statically checks that tuple
  matches in `ExecutionContracts*.cpp` set metadata, active temporary
  exceptions include expiry and migration target, active exception locations
  still resolve where practical, and selected generic routing files do not
  introduce model-name strings.
- Contract spec governance discovers all `test/vulkan_contract_specs/*.json`,
  validates a shared schema, checks `contract_name`/`family`/`tuple_id` against
  live `ExecutionContracts*.cpp` metadata, and keeps family-specific shape
  checks for EmbeddingLookup, ChannelCat, KVCacheAppend, and
  NoOverlapConvTranspose2D. Shared helpers in
  `test/vulkan_contract_specs/contract_spec_utils.py` keep generated runtime
  tests from copying spec loading, case iteration, log naming, and expected
  negative handling.
- `SDPAScoreSoftmaxContract` has stable metadata but no JSON contract spec
  fixture yet. Treat that fixture as follow-up governance work, not as a reason
  to refresh the real-model matrix by itself.
- Submit-origin counter tests use a named Python helper instead of raw numeric
  indices. The helper is intentionally test-local; no C++ diagnostic API change
  was made for this guardrail refresh.
- Capability-profile governance checks ensure the required profile IDs are in
  the manifest, the normalized feature/limit keys are present, docs state the
  non-emulation semantics, and runtime-policy tests verify optional ML features
  are clamped under `vk_min_1_1_compute`.

## Validation Caveats

- Model status artifacts can be stale relative to each other. Before changing a
  production route, confirm the relevant current blocker with a bounded smoke,
  focused test, or fresh diagnostic artifact.
- DAv2 stack owner is intentionally safe and does not merge command-buffer
  replay until descriptor ownership and binding validation are ready.
- Some compatibility evidence is device-specific. RX 9070 remains the primary
  optimization signal; RX 6700 XT and GTX 1080 are compatibility checks.
- Capability-profile tests are planner admission checks on the current device.
  They can find route over-admission under reduced feature masks, but they do
  not replace the RX 9070/RX 6700 XT/GTX 1080 real-hardware rows.
- Gemma E2B is a memory/dtype milestone, not a reason to add exact route
  exceptions.
- PaddleOCR completes current matrix-sensitive RX 9070/RX 6700 XT/GTX 1080
  smoke coverage in Task033/Task034-era artifacts with zero sync readback, but
  that matrix is stale by commit relative to the profile/spec governance stack.
  The intervening spec/profile work, InitialCache observability-label update,
  NoOverlapConvTranspose2D fixture coverage, ChannelCat source split, and
  NoOverlapConvTranspose2D, BatchNormInference, KVCacheAppend,
  EmbeddingLookup, GQARepeat, SafeViewReshape, DiffusionSDPA, and
  SDPAExecutionPolicy, SDPAScoreSoftmax, SmallMetadataPaddedConv2D,
  SmallSpatialPointwiseConv, MaskedTinySDPA, and TransformerGQASDPA source
  splits should not change accepted shapes or default no-profile model
  routing, but rerun the real-model matrix after the next backend behavior
  change or before claiming or raising a model gate.

## Build Context

On this Windows machine, use the existing Visual Studio CMake build tree from
`build/CMakeCache.txt`. The local cache records Visual Studio 17 2022, x64,
Release, `USE_VULKAN=ON`, `USE_VULKAN_API=ON`, strict SPIR-V, Vulkan 1.3, and
SPIR-V 1.6 targets.
