# Vulkan Current State

Last refreshed: 2026-06-14 at local HEAD
`7d10cec4fde84856d19eb0d63544900db940b4bc`.

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
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsBatchNormInferenceMaterializedSpec.h`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsBatchNormInferenceSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsChannelCat.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsChannelCatSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsDiffusionSDPA.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsElementwiseBroadcast.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsElementwiseBroadcastSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsEmbeddingLookup.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsEmbeddingLookupSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsGQARepeat.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsKVCacheAppend.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsKVCacheAppendInitialSpec.h`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsKVCacheAppendSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsLinearGeluBridge.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsMaskedTinySDPA.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsNoOverlapConvTranspose2D.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsNoOverlapConvTranspose2DSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsSafeViewReshape.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsSafeViewReshapeAliasSpec.h`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsSafeViewReshapeSpec.h`
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
`SmallSpatialPointwiseConvContract`, `MaskedTinySDPAContract`,
`ElementwiseBroadcastContract`, and `TransformerGQASDPAContract`,
`DiffusionSDPAContract`,
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

## Current Telemetry Checkpoint

Task179 and Task181 artifacts are planner telemetry only; they do not raise a
model gate and they do not imply model-specific production routes.

- DAv2 RX 9070: stable. Task179 completed with `cpu_fallback=0`,
  `sync_readback=169`, `tensor_cpu_readback=430`, `retire_drains=102`, and
  `conv_prepack_upload=4`.
- HY-MT RX 9070 99-token prompt with 16 generated tokens: stable but still
  high in fallback/readback attribution. Task179 reported `cpu_fallback=423`,
  `sync_readback=83`, `tensor_cpu_readback=5827`, and model-core tensor-op
  fallback/readback `0/0`.
- PaddleOCR RX 9070 screenshot: stable in the Task179 single row. It reported
  `cpu_fallback=1`, `sync_readback=1`, `tensor_cpu_readback=1824`, and
  `conv_prepack_upload=140`; the earlier first-attempt DeviceLost did not
  reproduce in that run.
- Gemma E2B: still blocked before useful route coverage by model-weight Vulkan
  OOM while moving
  `gemma4forconditionalgeneration.model.language_model.embed_tokens_per_layer.weight`.
- Lotus: Task181 cleared the benchmark-local `_c10d_functional.wait_tensor`
  import blocker, but Lotus still fails before useful Vulkan execution because
  the source-tree environment lacks the compiled DTensor C API
  `_DTensor_OpSchema_post_init` in `torch._C`. The Lotus counters remain zero
  and the row must not contribute backend regression budgets.

Benchmark-local distributed shims must stay import-only and single-process.
`_c10d_functional.wait_tensor` may be an identity shim for telemetry imports;
collective and DTensor op schema stubs must raise if executed. Do not add
benchmark-local fakes for compiled `torch._C` DTensor APIs. Restoring Lotus
telemetry now requires a real source-tree distributed/DTensor-capable build or
a compatible runtime environment, not a Vulkan backend change.

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
- `agent_space/task179_real_workload_status_telemetry.md`: telemetry checkpoint
  for DAv2, Lotus, HY-MT, PaddleOCR, and Gemma on the current local corpus.
- `agent_space/task181_lotus_shim_validation.md`: benchmark-local Lotus shim
  validation and current `missing_compiled_dtensor_c_api` blocker.

These files are diagnostic inputs. Production code must not depend on
`agent_space`.

## Current Contract Groups

- `SmallSpatialPointwiseConvContract`: finite projection rows, now split into
  a family-specific source; keep exact rows until broader legality is proven.
- `NoOverlapConvTranspose2DContract`: bounded float-buffer 2x2 stride-2
  no-overlap transposed-conv envelope. The `Kernel2Stride2FloatBuffer` slice
  has a JSON contract spec backed by `ShapeEnvelope` v1 with checked-in
  positive/adjacent-negative runtime cases and generic ShapeEnvelope C++
  metadata/simple-bound helper output. Packed input-channel equality, output
  shape arithmetic, prepack resource behavior, and match-result assembly remain
  handwritten; preserve unsupported-case fallback outside that envelope.
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
  contract for heads `{1, 5}` and sequence `{504, 640}`. The
  `DiffusionSquareScores` slice has a JSON contract spec with generated
  positive and adjacent negative runtime coverage; keep the temporary exception
  until broader score-softmax/layout behavior is proven.
- `EmbeddingLookupContract`: finite token-batch and small-bounded embedding
  lookup contract; the small-bounded lookup slice has a JSON contract spec with
  generated positive and adjacent negative runtime coverage. The
  `SmallBoundedLookup` slice now uses the generic ShapeEnvelope C++ generator
  path for generated metadata, bounds, and matcher helper predicates while the
  token-batch row remains handwritten. Keep remaining exact rows until broader
  legality is proven.
- `CatAxisContract`: umbrella for bounded last-dim, channel-dim, and rank-3
  cat patterns. The `ChannelCatContract` rank-4 dim-1 buffer slice has a JSON
  contract spec with generated positive and adjacent negative runtime coverage
  and a `ShapeEnvelope` v1 source for symbolic dims, relationships, aggregate
  bounds, layout/capability requirements, and policies. Its contract identity,
  route label, metadata, simple bounds, typed spec row, and scalar/per-input
  helper predicates are emitted by the generic ShapeEnvelope C++ generator into
  a generated C++ header while the cross-input loop and match result
  construction remain handwritten.
- `KVCacheAppendContract`: bounded Transformer sequence append and initial
  empty-cache cat rows. Both `SequenceAppend` and `InitialCache` slices have
  JSON contract specs backed by `ShapeEnvelope` v1 with checked-in positive
  and adjacent negative runtime cases plus generic ShapeEnvelope C++
  metadata/simple-bound helper output. The generated helpers provide contract
  identity, route labels, metadata, dtype/rank/scalar/range bounds, and helper
  predicates while initial-empty handling, sequence lower bounds, cross-input
  equality, and match-result assembly remain handwritten. InitialCache
  positives log the contract-owned `aten::cat.kv_cache_initial_dim2_buffer`
  op-hit label while unrelated direct-buffer cat paths keep their generic
  labels.
- `UNetChannelConcatContract`: mostly generic already; keep model provenance in
  tests/docs.
- `GQARepeatContract`: finite bounded K/V head repeat contract, now split into
  a family-specific source. The
  `Batch1Heads4Factor4Sequence100To116Dim128` slice has a JSON contract spec
  with generated positive and adjacent negative runtime coverage; keep exact
  rows until broader legality is proven.
- `BatchNormInferenceContract`: float32 4D inference batch norm. The
  `BufferFloat4D` and `MaterializedBufferFloat4D` slices both have JSON
  contract specs backed by `ShapeEnvelope` v1 with checked-in
  positive/adjacent-negative runtime cases. Both slices now use the generic
  ShapeEnvelope C++ generator path for generated metadata, simple bounds, and
  helper predicates while feature-count equality, parameter checks, and match
  result assembly remain handwritten. Tensor provenance and value traces report
  the admitted contract name, family, tuple id, and materialization policy for
  BatchNorm canaries without changing the visible execution route.
- `SafeViewReshapeContract`: finite dense direct-buffer view and reshape-alias
  contract, now split into a family-specific source. Both direct-buffer slices
  now have JSON contract specs with ShapeEnvelope-generated legal and
  adjacent-negative runtime coverage: `ViewMaterializedDirectBuffer` for the
  materialized `aten::view` path and `ReshapeAliasDenseBufferDirect` for the
  materialized `aten::_reshape_alias` path. Both slices now consume generated
  ShapeEnvelope C++ shape/layout simple-bounds headers for contract identity,
  metadata, rank bounds, storage-offset, output last-dim multiple, and policy
  constants while product equality, dense/contiguous-stride checking, and
  match result assembly remain handwritten. Keep broader view/layout,
  storage-offset, and provenance rules documented separately.
- `LinearGeluBridgeContract`: pure legality for the deferred linear/GELU
  bridge; registry, alias, and materialization side effects stay outside the
  contract.
- `ElementwiseBroadcastContract`: production metadata/provenance canary for the
  existing float32 tensor/tensor buffer-broadcast route. The
  `FloatTensorTensorBufferBroadcast` slice records the route shape in JSON and
  runtime tests, backed by a generic `ShapeEnvelope` `broadcast_compatible`
  relationship. Its contract identity, metadata, simple bounds, layout
  requirements, and attribute helpers are emitted by the generic
  ShapeEnvelope C++ generator v0 while right-aligned broadcast compatibility
  remains handwritten. The matcher is queried only after the existing
  `aten::binary_op.buffer_float` route is selected, so it records contract
  admission metadata without changing route behavior.
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
  live contract sources, validates any `ShapeEnvelope` v1 blocks present, and
  keeps family-specific shape checks for BatchNormInference, EmbeddingLookup,
  ChannelCat, KVCacheAppend, GQARepeat, SDPAScoreSoftmax,
  NoOverlapConvTranspose2D, and SafeViewReshape.
  `test/vulkan_contract_specs/generated_cpp_manifest.json` declares which
  ShapeEnvelope specs have checked-in generated C++ helper headers; governance
  validates that the manifest covers every current ShapeEnvelope spec, each
  header exists, each header regenerates byte-for-byte from its spec, and each
  header contains the expected helper markers.
  Shared helpers in `test/vulkan_contract_specs/contract_spec_utils.py` keep
  generated runtime tests from copying spec loading, case iteration, log
  naming, expected negative handling, and shape-envelope validation. A
  `SHAPE_ENVELOPE_ROLE_REGISTRY` now centralizes role validation, temporary
  runtime-case adapters, and data-driven semantic key fields so new roles do
  not add another open-coded key dispatch table. The same utility layer also
  has deterministic boundary/fuzz assignment generation for common
  ShapeEnvelope v1 concepts: value sets, min/max bounds, multiples, optional
  dims, scalar attributes, `broadcast_compatible` relationships, and
  adjacent-negative axes, plus a generic coverage bridge that maps abstract
  assignment paths and adjacent-negative axes onto the current
  generated/checked-in runtime cases without executing additional fuzz
  assignments. BatchNormInference `BufferFloat4D`,
  `MaterializedBufferFloat4D`, ElementwiseBroadcast
  `FloatTensorTensorBufferBroadcast`, KVCacheAppend `SequenceAppend` and
  `InitialCache`, and NoOverlapConvTranspose2D `Kernel2Stride2FloatBuffer` use
  generic checked-in case plumbing under the ShapeEnvelope registry.
  ChannelCat, EmbeddingLookup, and both
  SafeViewReshape direct-buffer slices have
  deterministic `ShapeEnvelope` legal-case and adjacent-negative generators
  that must match the checked-in positive and negative cases by semantic key,
  violated axis, adjacent value, and fallback/readback policy. Their runtime
  spec tests now execute generated legal positives and adjacent negatives
  through shared iterator plumbing while checked-in cases remain review/parity
  fixtures.
- ChannelCat has the first source-of-truth C++ table/matcher proof:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` regenerates
  `generated/ExecutionContractsChannelCatSpec.h` from
  `channel_cat_contract.json`, including a typed row and helper predicates.
  Generation now consumes the fixture's `ShapeEnvelope` v1 metadata, variadic
  tensor-list input, aggregate channel bounds, and matcher hints through the
  generic ShapeEnvelope generator path; governance compares the output
  byte-for-byte with the checked-in header.
- EmbeddingLookup `SmallBoundedLookup` now consumes the generic ShapeEnvelope
  C++ metadata/helper generator path. `tools/vulkan_contracts/gen_contract_spec_cpp.py`
  emits `generated/ExecutionContractsEmbeddingLookupSpec.h` from
  `embedding_lookup_contract.json` for metadata, route label, dtype/rank-list,
  range, boolean option bounds, and helper predicates; the matcher loop, result
  construction, and token-batch family remain handwritten.
- ElementwiseBroadcast `FloatTensorTensorBufferBroadcast` is the first
  consumer of generic ShapeEnvelope C++ metadata/helper generation v0:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsElementwiseBroadcastSpec.h` from
  `elementwise_broadcast_contract.json` for contract identity, metadata,
  scalar/rank/layout/attribute bounds, and simple helper predicates. The
  broadcast relationship and match result construction remain handwritten, and
  the generated helpers are used only by the metadata/provenance canary after
  the existing route is selected.
- BatchNormInference `BufferFloat4D` and `MaterializedBufferFloat4D` now
  consume the same generic ShapeEnvelope simple-bounds generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsBatchNormInferenceSpec.h` and
  `generated/ExecutionContractsBatchNormInferenceMaterializedSpec.h` from the
  direct and materialized BatchNorm JSON specs for contract identity, metadata,
  dtype/rank/layout/training bounds, materialization policy, and simple helper
  predicates. The simple-bounds generator emits row-qualified contract-name
  constants so sibling generated rows can be included in the same translation
  unit without duplicate symbols.
- KVCacheAppend `SequenceAppend` and `InitialCache` consume the generic
  ShapeEnvelope simple-bounds generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsKVCacheAppendSpec.h` and
  `generated/ExecutionContractsKVCacheAppendInitialSpec.h` from the sequence
  and initial-cache JSON specs for contract identity, metadata, route labels,
  dtype/rank/scalar/range bounds, and helper predicates. Initial-empty
  handling, sequence lower bounds, cross-input equality, and match-result
  construction remain handwritten so route behavior is unchanged.
- NoOverlapConvTranspose2D `Kernel2Stride2FloatBuffer` consumes the generic
  ShapeEnvelope simple-bounds generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsNoOverlapConvTranspose2DSpec.h` from
  `no_overlap_conv_transpose2d_contract.json` for contract identity, metadata,
  dtype/rank/options/layout bounds, and helper predicates. Packed-channel
  equality, output-shape arithmetic, prepack resource behavior, and match
  result construction remain handwritten so route behavior is unchanged.
- SafeViewReshape `ViewMaterializedDirectBuffer` and
  `ReshapeAliasDenseBufferDirect` consume the generic ShapeEnvelope
  shape/layout simple-bounds generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsSafeViewReshapeSpec.h` and
  `generated/ExecutionContractsSafeViewReshapeAliasSpec.h` from the regular
  view and reshape-alias JSON specs for contract identity, metadata, rank
  ranges, storage offset, stride/storage policy constants, Vulkan
  requirement, product equality policy, and output last-dim multiple helpers.
  Contiguous/dense-stride and product-of-sizes checks remain handwritten so
  route behavior is unchanged.
- Submit-origin counter tests use a named Python helper instead of raw numeric
  indices. The helper is intentionally test-local; no C++ diagnostic API change
  was made for this guardrail refresh.
- Tensor provenance/value-trace diagnostics can carry optional admitted
  contract metadata (`contract_name`, `contract_family`, `contract_tuple_id`,
  and `contract_materialization_policy`) for producers that pass an existing
  contract match. BatchNorm canaries distinguish direct buffer and
  materialized-buffer admission while the executed buffer kernel route label
  remains stable. ElementwiseBroadcast uses the same provenance path after the
  existing `aten::binary_op.buffer_float` route has already been selected.
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
- Lotus is telemetry-unavailable in the current source-tree environment. Do
  not fake compiled `torch._C` DTensor APIs in the benchmark harness to make it
  run; use a compatible distributed/DTensor-capable build or runtime before
  treating Lotus as backend evidence.
- PaddleOCR completed the Task179 RX 9070 screenshot row with one known CPU
  fallback and one sync readback, but that is still telemetry-only and not
  cross-adapter gate-ready. Rerun the real-model matrix after the next backend
  behavior change or before claiming or raising a model gate.

## Build Context

On this Windows machine, use the existing Visual Studio CMake build tree from
`build/CMakeCache.txt`. The local cache records Visual Studio 17 2022, x64,
Release, `USE_VULKAN=ON`, `USE_VULKAN_API=ON`, strict SPIR-V, Vulkan 1.3, and
SPIR-V 1.6 targets.
