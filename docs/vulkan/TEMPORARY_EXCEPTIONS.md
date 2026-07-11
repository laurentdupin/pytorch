# Vulkan Temporary Exceptions

Temporary exceptions are allowed only when they preserve correctness while a
route migrates toward a reusable contract. Each exception needs an expiry
condition and migration target.

## Active Exceptions

### Runtime Label Lane Inference

- Location: `aten/src/ATen/native/vulkan/planning/Request.cpp`
- Status: temporary
- Reason: runtime/allocation labels still infer broad planning lanes from text
  such as LLM, decoder, depth, DINO, patch embed, and refinenet.
- Expiry: benchmark/model wrappers pass explicit `VulkanPlanningRequest`
  scopes.
- Migration target: explicit planning-scope API and lane selection independent
  of model-name string matching.

### Runtime Elementwise Deferred Chain

- Location: `aten/src/ATen/native/vulkan/ops/BinaryOp.cpp`
- Status: explicit experimental opt-in, default off
- Reason: `PYTORCH_VULKAN_RUNTIME_ELEMENTWISE_CHAIN_DEFER=1` is the sole
  remaining Tensor placeholder canary. It is not a default eager route.
  `guard_vulkan_deferred_value_registration` rejects registration during graph
  execution before placeholder allocation or registry mutation. No recursive
  materialization or submission may be added to `convert()`, resource lookup,
  descriptor binding, or another locked low-level accessor.
- Expiry: graph-owned elementwise fusion/codegen has corpus parity, dynamic
  shape coverage, repeated-run lifetime safety, and zero CPU fallback/readback.
- Migration target: Vulkan graph-region elementwise instructions with
  program-owned value lifetime and execution.

### Linear Pending-Flush Deferral

- Location: `aten/src/ATen/native/vulkan/ops/Mm.cpp`
- Status: explicit experimental opt-in, default off
- Reason: `PYTORCH_VULKAN_LINEAR_PENDING_FLUSH_DEFERRAL=1` retains a
  `LinearPackedContext` through the normal pending-command flush. The linear
  output is already concrete; this policy exposes no public Tensor placeholder.
  Graph execution is rejected by the common deferred-registration guard.
- Expiry: graph programs own packed contexts and their completion retirement,
  with coverage-corpus graph parity and repeated-run lifetime safety.
- Migration target: graph-program packed-context ownership and timeline-gated
  retirement.

### Exact Tuple Rows In Contract Tables

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*`
- Status: temporary but allowed
- Reason: finite rows are proven by existing artifacts and keep unsupported
  adjacent shapes guarded.
- Metadata: each new exact row must carry contract/family/tuple/evidence/guard,
  fallback, and materialization metadata. Metadata is a migration guardrail, not
  an expiry condition by itself.
- Proof ledger: high-risk exact rows are also tracked in
  `test/vulkan_contract_proofs/contract_proof_manifest.json`. Changes to exact
  rows in covered contracts must update the accepted-row manifest and proof
  ledger through `tools/vulkan_contract_codegen/compare_contract_admission.py`
  so row debt, dependency drift, and cardinality changes are reviewed rather
  than silently accepted.
- Expiry: generated positive and adjacent negative tests cover the family well
  enough to review a parameterized policy.
- Migration target: generated `KernelFamilyContract` tables with positive and
  negative tests.

### KV Cache Append Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Concat.cpp`
- Status: temporary, contract-named
- Reason: finite Transformer KV-cache append rows are proven by existing cat
  tests and now act as evidence/regression fixtures for the dynamic sequence
  append and initial-cache families. `SequenceCatDirectBuffer` covers fp32
  rank-4 direct-buffer dim-2 appends by semantic batch/head/head-dim equality
  and positive sequence lengths instead of exact observed sequence rows.
  `InitialSequenceCatDirectBuffer` covers the empty-cache bootstrap by requiring
  a Vulkan empty left operand, a fp32 rank-4 Vulkan buffer cache tensor,
  dim-2 semantics, and positive runtime batch/head/sequence/head-dim values.
- Generated spec coverage: `test/vulkan_contract_specs/kv_cache_append_contract.json`
  covers the `SequenceAppend` slice for source sequence `S=1..115`, and
  `test/vulkan_contract_specs/kv_cache_append_initial_contract.json` covers the
  `InitialCache` slice for initial prompt sequence `S=14..116`, with
  ShapeEnvelope-backed checked-in positive and adjacent negative runtime cases
  plus generic ShapeEnvelope C++ metadata/simple-bound helper output in
  `generated/ExecutionContractsKVCacheAppendSpec.h` and
  `generated/ExecutionContractsKVCacheAppendInitialSpec.h`. SequenceAppend
  batch/heads/head-dim equality is generated while initial-empty handling,
  sequence lower bounds, and match-result assembly remain handwritten. The
  InitialCache rows are now regression fixtures around
  `InitialSequenceCatDirectBuffer`; positives and random dynamic cases log
  `aten::cat.kv_cache_initial_dim2_buffer`.
- Expiry: broader cat-axis behavior has dynamic semantic parity plus adjacent
  negative coverage.
- Migration target: `SequenceCatDirectBuffer` plus generated
  `KVCacheAppendContract` and `CatAxisContract` tables with positive and
  negative tests.

### Large Linear Execution Checkpoint

- Location: `aten/src/ATen/native/vulkan/ops/Mm.cpp`
- Status: temporary, generic lifetime guard
- Reason: large inference-only packed-weight linear sequences can accumulate
  enough pending resources to trigger Windows stack overflow/device instability
  before the broader decoder-region ownership path can retire them safely. The
  checkpoint is keyed by packed linear weight size plus submission/byte budgets,
  not by model name.
- Evidence: HY-MT layer-depth probing showed full-model depth/resource
  accumulation fails around 8 layers without synchronization and succeeds when
  explicit synchronization is inserted at a smaller interval. Cross-adapter
  one-token HY-MT diagnostics under
  `agent_space/paddle_hymt_perf_goal_c5dee8d/diagnostic_post_large_linear_checkpoint/`
  complete on RX 9070, GTX 1080, and RX 6700 XT without stack overflow.
- Expiry: region-owned decoder/linear command ownership can transfer and retire
  large linear packed-weight resources without periodic stream synchronization,
  and HY-MT one-token decode remains stable across all three adapters with the
  checkpoint disabled.
- Migration target: a planned Transformer decoder region or linear execution
  ownership contract with explicit packed-weight/scratch lifetime handoff,
  timeline-gated retire, and no generation-control CPU fallback hiding.

### Embedding Lookup Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Indexing.cpp`
- Status: partially migrated, contract-named
- Reason: finite token-batch and small-bounded embedding lookup rows remain
  evidence/regression fixtures. The dynamic program runtime now owns fp32
  rank-2 Vulkan weights with CPU-resident Long rank-1/rank-2 indices
  after host index-bounds checking, so vocab size, embedding dim, and index
  count are no longer production admission limits for that safe path. It also
  owns CPU-uploaded Vulkan Long index tensors whose exact descriptor carries
  integer min/max provenance proving values are within the runtime vocab bound.
  Truly device-produced Vulkan-resident indices still need a no-readback
  value-bounds proof or device-side error path before they can be broadly
  admitted without weakening PyTorch's out-of-range semantics.
- Generated spec coverage: `test/vulkan_contract_specs/embedding_lookup_contract.json`
  covers the small-bounded lookup slice with generated positive and adjacent
  negative runtime tests and generic ShapeEnvelope C++ metadata/helper output,
  including the derived indices product helper.
  Other embedding rows still need broader generated coverage before the
  exception can expire.
- Expiry: device-produced Vulkan-resident index tensors have a value-bounds or
  error contract that preserves PyTorch out-of-range behavior without
  per-shape allowlists.
- Migration target: `EmbeddingLookupDirectBuffer` semantic admission plus
  generated value-bounds and layout tests; finite rows remain only as
  performance evidence or regression fixtures.

### Cat Axis And Channel Cat Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Concat.cpp`
- Status: partially migrated, contract-named
- Reason: bounded cat rows are proven for current last-dim, channel-dim, and
  rank-3 patterns, but broader axis/layout behavior is not proven yet.
  Rank-4 dim-1 fp32 buffer-backed cat now has semantic admission through
  `CatAxisDirectBuffer`; input count, spatial size, batch, and total channels
  are runtime descriptor values rather than production row bounds. The current
  multi-input buffer-view path still requires channel extents and total channel
  count to be multiples of 4 because that is an implementation/layout
  constraint in the existing buffer-view copy path.
- Generated spec coverage: `test/vulkan_contract_specs/channel_cat_contract.json`
  covers the rank-4 dim-1 channel-cat buffer slice with generated positive and
  adjacent negative runtime tests and generic ShapeEnvelope C++ metadata/helper
  output for the multi-input route. The two-input rank-4 dim-1 buffer route
  `aten::cat.buffer_channel_pair` is covered by focused parity/op-hit tests,
  including odd-channel pairs under bounded `N=1`, total `C <= 4096`, and
  spatial `224x224` guards, and remains handwritten until the generated
  `ChannelCatContract` admits the pair case directly. The generated finite
  multi-input rows remain evidence/regression fixtures for
  `CatAxisDirectBuffer`, not default H/W or input-count admission. Other
  cat-axis rows still need fixtures or documented follow-up before this
  exception can expire.
- Expiry: rank-3 and remaining axis variants either have semantic dynamic
  families or are documented as unsupported with specific layout/semantic
  reasons, and the rank-4 dim-1 channel-alignment constraint has either been
  lifted by implementation work or recorded as a permanent layout contract.
- Migration target: `CatAxisDirectBuffer` plus generated `CatAxisContract` and
  `ChannelCatContract` evidence tables with positive and negative tests.

### Token Prefix Cat/Add Exact Rowset

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*`,
  `aten/src/ATen/native/vulkan/ops/VisionBlocks.cpp`, and
  `scripts/benchmarks/benchmark_depth_anything.py`
- Status: migrated to semantic dynamic admission, contract-named evidence kept
- Reason: the fused prefix-token concat plus position-add route is now semantic
  for fp32 rank-3 direct-buffer inputs with prefix length `1`, dim `1`, matching
  batch/feature dimensions, positive token count, and output sequence
  `1 + token_count`. Arbitrary cat+add fusion and split-token consumer regions
  are not proven.
- Generated spec coverage:
  `test/vulkan_contract_specs/token_prefix_cat_add_contract.json` covers
  `prefix=[1,1,C]`, `tokens=[1,N,C]`, `pos/out=[1,N+1,C]`,
  `C in {384,768,1024}`, and
  `N in {150,260,600,620,1350,1380,2400,2440,3750,3850}` with generated C++
  sparse-rowset helper output in
  `generated/ExecutionContractsTokenPrefixCatAddSpec.h`.
- Expiry: the finite rowset can be removed from production admission once
  `TokenPrefixCatAddDirectBuffer` random-shape parity and adjacent negatives
  remain stable as the only token-prefix admission path.
- Migration target: keep generated rows as regression fixtures while
  `TokenPrefixCatAddDirectBuffer` owns runtime-shape admission; broader
  token-preparation region fusion needs a separate `RegionContract`.

### GQA Repeat Rectangular Score Budget

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Softmax.cpp`
- Status: migrated to semantic dynamic admission inside the rectangular score
  budget, contract-named evidence kept
- Reason: `GQARepeatDirectBuffer` now behavior-authorizes the runtime-sized
  repeat shader for fp32 rank-4 Vulkan buffer K/V tensors, and SDPA can use it
  for non-causal, mask-free materialized GQA when both K and V match and the
  resulting rectangular rank-3 score tensor matches
  `RectangularScoresRuntimeShape`. Random unseen-shape coverage found and fixed
  a repeat shader indexing bug where the sequence coordinate was divided by the
  repeat factor instead of the head coordinate. Materialized-GQA source lengths
  are no longer production-gated by exact repeat rows or by a fixed
  `source_len < 64` cap; downstream score-softmax admission is owned by runtime
  rectangular-score semantics and score-element budget.
- Generated spec coverage: `test/vulkan_contract_specs/gqa_repeat_contract.json`
  covers the `Batch1Heads4Factor4Sequence100To116Dim128` slice with
  ShapeEnvelope-backed checked-in positive and adjacent negative runtime cases
  plus generic ShapeEnvelope C++ simple-bound helper output in
  `generated/ExecutionContractsGQARepeatSpec.h`. Vulkan tensor/storage
  extraction, SDPA admission, materialization allocation and dispatch, op-hit
  labels, sequence lower-bound preservation, and match-result assembly remain
  handwritten.
- Expiry: the generated spec and runtime tests cover the rectangular score
  budget as first-class dynamic evidence, and over-budget or unsupported
  rectangular score shapes produce named semantic rejects rather than exact-row
  misses.
- Migration target: `SDPAScoreSoftmaxContract` runtime-family coverage for
  rectangular materialized-GQA scores, with `GQARepeatContract` rows retained as
  regression fixtures rather than production source-length admission.

### BatchNorm Inference Exact Envelopes

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Batchnorm.cpp`
- Status: partially migrated, contract-named
- Reason: float32 4D inference batch norm is proven for current buffer and
  materialized-buffer paths. The buffer path now admits through
  `BatchNormInferenceDirectBuffer` dynamic semantic validation, so runtime
  N/C/H/W are not finite production bounds. Broader dtype/rank/layout/training
  behavior and the materialized-buffer layout-transition path are not proven
  broadly yet.
- Generated spec coverage: `test/vulkan_contract_specs/batch_norm_inference_contract.json`
  covers the `BufferFloat4D` slice with ShapeEnvelope-backed checked-in
  positive and adjacent negative runtime cases plus generic ShapeEnvelope C++
  metadata/simple-bound helper output in
  `generated/ExecutionContractsBatchNormInferenceSpec.h`.
  `test/vulkan_contract_specs/batch_norm_inference_materialized_contract.json`
  covers the `MaterializedBufferFloat4D` layout-transition slice with
  checked-in positive and adjacent negative runtime cases plus generic
  ShapeEnvelope C++ metadata/simple-bound helper output in
  `generated/ExecutionContractsBatchNormInferenceMaterializedSpec.h`.
  Optional-aware feature-count equality is generated while parameter checks,
  provenance, storage/materialization policy, and match-result assembly remain
  handwritten.
- Expiry: materialized-buffer layout-transition behavior also has dynamic
  semantic parity, and unsupported dtype/rank/training cases are documented as
  permanent semantic rejects or implemented.
- Migration target: `BatchNormInferenceDirectBuffer` plus generated
  `BatchNormInferenceContract` evidence tables with positive and negative
  tests.

### Elementwise Broadcast Exact Envelope

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/BinaryOp.cpp`
- Status: temporary, contract-named
- Reason: the float32 tensor/tensor buffer-broadcast route is proven for
  `add`/`mul`/`sub` through `ElementwiseBroadcastDirectBuffer`, which admits
  fresh rank-1 through rank-4 broadcast-compatible runtime shapes by semantic
  dtype/layout/op rules. Exact rows remain evidence/regression fixtures around
  the dynamic family; dtype-promotion, `out=`, inplace, and scalar behavior are
  still separate unsupported semantics.
- Generated spec coverage: `test/vulkan_contract_specs/elementwise_broadcast_contract.json`
  covers the `FloatTensorTensorBufferBroadcast` slice with ShapeEnvelope-backed
  checked-in positive and adjacent negative runtime cases, including the
  bounded `sub` op-axis slice. Positives keep the existing
  `aten::binary_op.buffer_float` route label and record contract admission
  metadata in tensor provenance. The slice also has generated C++ metadata,
  simple-bound, op-attribute, and `broadcast_compatible` helper coverage in
  `generated/ExecutionContractsElementwiseBroadcastSpec.h`.
- Expiry: scalar, `out=`, inplace, and promotion families have dynamic semantic
  parity plus adjacent negative coverage.
- Migration target: `ElementwiseBroadcastDirectBuffer` plus broader generated
  `ElementwiseBroadcastContract` tables with positive and negative tests.

### Transformer GQA SDPA Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*`,
  `aten/src/ATen/native/vulkan/planning/RoutePolicy.cpp`, and
  `aten/src/ATen/native/vulkan/ops/Softmax.cpp`
- Status: temporary, contract-named
- Reason: bounded Transformer causal/prefill and decode GQA SDPA rows are
  proven for the current envelope, but broader Transformer attention shapes,
  masks, causal MHA dynamic behavior, and materialized execution policy are not
  proven yet. Non-causal decode GQA now has a semantic
  `DirectDecodeGQASDPADirectBuffer` path for fp32 4D direct-buffer tensors with
  batch 1, query length 1, no mask/dropout, divisible query/key-value heads,
  default/head-dim-equivalent scale, and direct-GQA shader-budget dimensions.
  Causal prefill GQA and equal-head MHA now use the semantic
  `DirectCausalPrefillGQASDPADirectBuffer` path for fp32 4D direct-buffer
  tensors with batch 1, equal query/source sequence length, no explicit
  mask/dropout, default/head-dim-equivalent scale, and direct-GQA
  shader-budget dimensions. GQA requires divisible query/key-value heads; MHA
  requires equal heads so the direct-GQA shader repeat factor is `1`. The finite
  rows remain evidence for exact decode policy and adjacent negative guard
  coverage. Bounded q>1 non-causal GQA now uses
  `SmallNonCausalGQASDPADirectBuffer` when target/source lengths are at most
  64 and direct-GQA shader budgets hold, so the exact small non-causal rows are
  evidence and guard fixtures rather than runtime admission bounds.
  Equal-head non-causal MHA now uses
  `DirectNonCausalMHASDPADirectBuffer` when direct-buffer layout,
  lane-aligned head/value dims, and direct-GQA shader budgets hold, so
  diffusion-style square rows matching those constraints are semantic runtime
  shapes rather than finite diffusion admission rows.
  `SDPAExecutionPolicyContract` now has
  `TransformerDecodeGQACloneOnlyRuntimeShape` for the decode-GQA
  post-softmax clone policy, using runtime batch/head/source/head-dim
  semantics and score-element budget instead of the old finite source-length
  execution-policy row.
  Unequal-head causal MHA without `enable_gqa` remains semantically rejected
  rather than materialized or repeated implicitly.
- Generated spec coverage: `test/vulkan_contract_specs/transformer_gqa_sdpa_contract.json`
  covers the `SparseAttentionRows` slice with ShapeEnvelope-backed checked-in
  positive and adjacent negative runtime cases plus generic ShapeEnvelope C++
  sparse-rowset helper output in
  `generated/ExecutionContractsTransformerGQASDPASpec.h`. The generated helper
  owns the four correlated causal MHA, causal GQA, decode GQA, and small
  non-causal GQA rows, per-row metadata, exact lookup by contract family plus
  causal/GQA flags, and row-match bounds/conditional equal-sequence checks.
  Optional scale tolerance, route-policy hard-fail ordering, tensor
  extraction/early dtype-rank guards, SDPA execution, materialization policy,
  and match-result assembly remain handwritten.
- Expiry: broader Transformer SDPA/GQA parity plus adjacent negative coverage
  are available for causal MHA, masked, materialized, and repeat-materialized
  execution policies without relying on exact row bounds.
- Migration target: broader generated `TransformerGQASDPAContract` tables with
  positive, adjacent negative, and materialization-policy coverage.

### Vision Self-Attention SDPA Runtime Family

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*`,
  `aten/src/ATen/native/vulkan/planning/RoutePolicy.cpp`, and
  `aten/src/ATen/native/vulkan/ops/Softmax.cpp`
- Status: migrated to semantic dynamic admission, contract-named evidence kept
- Reason: rank-3 float vision self-attention is now semantic for `[BH,T,64]`
  when Q/K/V shapes match, no mask is present, attention is non-causal,
  dropout is `0`, GQA is off, and explicit scale is `1.0`. Runtime `BH` and
  `T` are not production row bounds. Broader head dims, implicit-scale policy,
  masks, causal attention, and direct softmax-to-value-BMM behavior are not
  proven yet.
- Generated spec coverage:
  `test/vulkan_contract_specs/vision_self_attention_sdpa_contract.json` covers
  the `SparseAttentionRows` slice with ShapeEnvelope sparse-rowset rows,
  checked-in positive and adjacent negative runtime cases, and generic
  ShapeEnvelope C++ metadata/row helpers in
  `generated/ExecutionContractsVisionSelfAttentionSDPASpec.h`. The generated
  helper owns row metadata and row-match bounds for evidence rows. Scale tolerance,
  route-policy hard-fail ordering, tensor extraction/early dtype-rank guards,
  materialized math-path selection, post-softmax clone behavior, and
  match-result assembly remain handwritten. The score-softmax route now mirrors
  the same runtime `BH/T` policy via `SDPAScoreSoftmaxContract`
  `VisionSelfAttentionScoresRuntimeShape`; it writes probabilities into a fresh
  direct buffer and does not enable the previously failing direct
  softmax-probability-to-value-BMM path.
- Expiry: broader vision self-attention parity and adjacent negative coverage
  are available across head dim, scale, mask/causal, and probability
  materialization behavior without regressing existing SDPA rows.
- Migration target: keep exact rows as regression fixtures while semantic
  `VisionSelfAttentionSDPAContract` admission owns runtime `BH/T`; a reviewed
  attention probability materialization policy must separately replace the
  no-clone/materialization rowset.

### SDPA Execution Policy Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Softmax.cpp`
- Status: temporary, contract-named
- Reason: materialization, repeat, and post-softmax clone decisions are proven
  for finite SDPA execution envelopes, but broader layout-transition behavior is
  not proven yet. The recognizer-style small-head non-causal MHA direct-buffer
  policy has migrated to semantic runtime admission; finite recognizer rows now
  act as evidence/regression fixtures rather than production shape gates.
- Generated spec coverage: `test/vulkan_contract_specs/sdpa_execution_policy_contract.json`
  covers the `SparsePolicyRows` slice with ShapeEnvelope-backed checked-in
  positive and adjacent negative runtime cases plus generic ShapeEnvelope C++
  sparse-rowset helper output in
  `generated/ExecutionContractsSDPAExecutionPolicySpec.h`. The generated
  helper owns row-match bounds for the correlated policy rows while diffusion
  contract admission, tuple-id cross-checks, scale tolerance, score
  pre-materialization, materialized math path, post-softmax clone behavior, and
  match-result assembly remain handwritten. Randomized
  `RecognizerNonCausalMHA` runtime-shape tests cover legal small-head
  direct-buffer MHA outside the finite rowset.
- Expiry: broader SDPA execution/layout parity plus adjacent negative coverage
  are available for diffusion materialization, transformer clone-only, and
  masked policy decisions.
- Migration target: broader generated `SDPAExecutionPolicyContract` tables
  with positive and negative tests.

### Attention Probability Materialization Proof Rows

- Location: `test/vulkan_contract_specs/attention_probability_materialization_contract.json`
  and
  `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsAttentionProbabilityMaterializationSpec.h`
- Status: bounded transition-policy slice active for the six vision direct-safe
  rows; Lotus `[10,126,126]` remains clone-required
- Reason: Lotus decomposed-attention proof found ten rank-3 float
  softmax-probability/value-BMM rows. Nine are direct-safe evidence, while the
  `[10,126,126]` probability row requires explicit Vulkan clone/materialization
  before value BMM in proof. The rowset also records the six already-admitted
  low-resolution VisionSelfAttention probability rows; owner-path diagnostic
  proof promoted those six rows to direct-safe no-clone under the existing
  VisionSelfAttention SDPA and transition-row guards. The rowset prevents this
  evidence from becoming a blanket softmax-to-BMM rule or a model-named route.
- Generated spec coverage:
  `test/vulkan_contract_specs/attention_probability_materialization_contract.json`
  covers the `DecomposedAttentionProbabilityToValueBmm` layout-transition edge
  with ShapeEnvelope sparse-rowset rows, positive proof cases, adjacent
  negatives, and generated C++ metadata/row helpers. Runtime proof coverage
  verifies direct probabilities pass for direct-safe rows, cloned/materialized
  probabilities remain valid for all observed rows, and the current
  direct-consumer failure is preserved for materialization-required rows.
- Expiry: a reviewed production `LayoutTransitionContract` materialization
  policy consumes the generated row metadata for required rows, or broader
  proof invalidates/removes the rowset.
- Migration target: production `AttentionProbabilityMaterializationContract`
  or region-level attention probability materialization policy with zero CPU
  fallback/readback and explicit direct-safe versus materialization-required
  row semantics.

### SDPA Score Softmax Exact Envelope

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*`,
  `aten/src/ATen/native/vulkan/planning/RoutePolicy.cpp`, and
  `aten/src/ATen/native/vulkan/ops/Softmax.cpp`
- Status: partially migrated to semantic dynamic admission
- Reason: diffusion score-softmax exact rows are retained as evidence, but
  runtime square fp32 rank-3 score tensors now admit through
  `DiffusionSquareScoresRuntimeShape` when head/sequence budgets hold. The
  paired diffusion SDPA square runtime family covers `head_dim=64`; existing
  `head_dim=512` rows remain exact evidence until unseen 512-dim sequences have
  direct-buffer materialization proof. Vision
  self-attention rank-3 square scores also admit runtime positive `BH/T` values
  through `VisionSelfAttentionScoresRuntimeShape`. Broader score-softmax/layout
  behavior outside the rank-3 square fp32 buffer family is not proven yet.
- Generated spec coverage: `test/vulkan_contract_specs/sdpa_score_softmax_contract.json`
  covers the `DiffusionSquareScores` slice with ShapeEnvelope-backed
  checked-in positive and adjacent negative runtime cases plus generic
  ShapeEnvelope C++ simple-bound helper output in
  `generated/ExecutionContractsSDPAScoreSoftmaxSpec.h`. The
  `DiffusionSquareScoresRuntimeShape` owns runtime square diffusion-like score
  admission; the `VisionSelfAttentionScores` evidence slice reuses
  `generated/ExecutionContractsVisionSelfAttentionSDPASpec.h`; production
  runtime-shape admission is handled by
  `VisionSelfAttentionScoresRuntimeShape`. Softmax route ordering,
  `can_run_buffer_softmax` policy, guard
  op-hit logging for `aten::_softmax.buffer_lastdim_known_bad_texture_fallback`,
  fallback visibility, and match-result assembly remain handwritten.
- Expiry: broader SDPA score-softmax/layout parity plus adjacent negative
  coverage is available.
- Migration target: broader generated `SDPAScoreSoftmaxContract` tables with
  positive and negative tests.

### Safe View Reshape Exact Envelope

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Shape.cpp`
- Status: temporary, contract-named
- Reason: dense direct-buffer `view` and `_reshape_alias` materialization is
  proven for the current guarded envelope, but broader view/layout behavior is
  not proven yet.
- Generated spec coverage: `test/vulkan_contract_specs/safe_view_reshape_contract.json`
  covers the `ViewMaterializedDirectBuffer` slice with generated legal and
  adjacent-negative runtime tests, plus
  `generated/ExecutionContractsSafeViewReshapeSpec.h` for generated contract
  identity, metadata, shape/layout bounds, and helper predicates.
  `test/vulkan_contract_specs/safe_view_reshape_alias_contract.json` covers the
  `ReshapeAliasDenseBufferDirect` slice with generated legal and
  adjacent-negative runtime tests, plus
  `generated/ExecutionContractsSafeViewReshapeAliasSpec.h` for generated
  contract identity, metadata, shape/layout bounds, and helper predicates.
  Product-equality helpers are generated; dense-stride checks and match-result
  assembly remain handwritten for both slices.
- Expiry: broader view/reshape parity plus adjacent negative coverage are
  available.
- Migration target: generated `SafeViewReshapeContract` tables with positive
  and negative tests.

### Small Metadata Padded Conv2D Exact Tuple

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Convolution.cpp`
- Status: temporary, contract-named
- Reason: the padded low-channel buffer materialization is now admitted by
  semantic layout guards for batch-one fp32 width-packed non-direct
  small-channel 2x2 convs, but the implementation is still handwritten and does
  not cover batched inputs, other kernels, grouped convs, or direct-output
  ownership.
- Generated spec coverage: `test/vulkan_contract_specs/small_metadata_padded_conv2d_contract.json`
  covers the original `MaterializedBufferInput2x2` tuple as review evidence.
  Runtime random-shape coverage now exercises the semantic
  `RuntimeMaterializedBufferInput2x2` family. Tensor-info extraction, input
  materialization, op-hit logging, fallback to
  `aten::convolution.buffer_float_skip.small_metadata_input`, and match-result
  assembly remain handwritten.
- Expiry: the semantic family is represented in generated contract metadata or
  migrated into a `LayoutTransitionContract` with positive and negative tests.
- Migration target: generated `SmallMetadataPaddedConv2DContract` or
  `LayoutTransitionContract` tables for the runtime materialization policy.

### Small Spatial Pointwise Conv Bounded Rows

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Convolution.cpp`
- Status: temporary, contract-named
- Reason: finite 1x1 pointwise projection rows and one bounded factorized
  depth-vision projection group are proven for current projection envelopes,
  but broader optimized-plan row evidence is not proven yet.
- Generated spec coverage: `test/vulkan_contract_specs/small_spatial_pointwise_conv_contract.json`
  covers the `SparseProjectionRows` slice with ShapeEnvelope-backed checked-in
  positive and adjacent negative runtime cases plus generic ShapeEnvelope
  sparse-rowset helper output in
  `generated/ExecutionContractsSmallSpatialPointwiseConvSpec.h`. The generated
  helper owns the 67 correlated projection rows, per-row metadata,
  input/weight channel equality, and exact `(input_c, input_h, input_w,
  output_c)` lookup. It also owns the cross-adapter proven 144-shape
  factorized depth-vision projection group: 18 approved channel pairs crossed
  with eight approved spatial pairs, including 84 validated corpus/proof
  shapes and 60 proven extrapolations. The two `18x10` depth-vision decoder
  rows are exact sparse rows only; their spatial pair is not part of the
  factorized group. The sixteen newly admitted mid-resolution
  depth-vision rows are exact sparse rows only; their `(30,45)` and `(40,62)`
  spatial pairs are not part of the factorized group. The PaddleOCR
  `3x80` and `6x80` OCR recognizer rows are exact OCR sparse rows admitted to
  unblock existing OCR projection routes:
  `ocr_projection_384_3x80_384`, `ocr_projection_512_3x80_512`,
  `ocr_projection_512_6x80_192`, `ocr_projection_512_6x80_1024`,
  `ocr_projection_1024_3x80_384`, `ocr_projection_1024_3x80_2048`,
  `ocr_projection_1024_6x80_192`, `ocr_projection_1664_6x80_512`,
  `ocr_projection_2176_6x80_512`, and `ocr_projection_3328_3x80_1024`.
  OCR projection rows additionally allow a bounded dynamic crop batch
  `N=1..8`, with checked-in positive coverage for the observed batch-6 and
  batch-3 rows; depth-vision and diffusion projection rows remain batch-1.
  This does not widen the OCR spatial/channel envelope.
  `SmallSpatialPointwiseConvContract` now has a `GenericDynamicHW`
  adaptive family for legal fp32 direct-buffer 1x1 pointwise
  convolutions with unseen batch/H/W under semantic 1x1/direct-buffer guards.
  It uses the existing dynamic-shape 1x1 buffer shader
  and does not require exact H/W observation for correctness. Batch-one
  width-packed cases may select the existing as-linear plan from dynamic
  admission. New exact pointwise rows should be added only as evidence fixtures
  or when this dynamic family rejects with a named unsupported semantic or
  layout reason.
  Naive min/max envelopes and independent H/W cross-products for optimized
  evidence rows remain rejected by
  `KnownBadLargePointwiseConv`.
  Route-policy hard-fail rescue, shader-family decisions, family op-hit labels,
  and match-result assembly remain handwritten.
- Expiry: exact-row optimized plan evidence is either promoted to a bounded
  plan policy or retired as redundant evidence after dynamic pointwise parity
  covers layout, storage, channel-pair, spatial-pair, and output-channel
  families.
- Migration target: `SmallSpatialPointwiseConvContract` dynamic families plus
  broader generated pointwise `KernelFamilyContract` tables with positive and
  negative tests.
  For OCR, the next promotion should be an OCR recognizer finite-rowset or
  bounded correlation-group proof, not independent channel/spatial
  cross-products.

### Patch Embed Float-Buffer Conv Route

- Location: `aten/src/ATen/native/vulkan/ops/Convolution.cpp`
- Status: temporary, contract-named
- Reason: a bounded patch-embed conv family can avoid the value-bearing legacy
  conv weight CPU repack/readback by using the existing float-buffer conv route,
  but broader kernel-14/stride-14 conv layout behavior is not proven yet.
- Generated spec coverage:
  `test/vulkan_contract_specs/patch_embed_float_buffer_conv_route_contract.json`
  and generated sparse-rowset helper output in
  `generated/ExecutionContractsPatchEmbedFloatBufferConvRouteSpec.h`. The
  observed-row predicate records float Vulkan tensors with input `[1,3,H,W]`,
  DAv2 evidence-row `(H,W)` pairs, weight `[C,3,14,14]`,
  `C in {384,768,1024}`, stride `[14,14]`, zero padding, dilation `[1,1]`,
  and groups `1`. Production admission now falls through to the semantic
  `GenericKernel14Stride14FloatBuffer` family under the same
  `PatchEmbedFloatBufferConvRoute` contract when the op is still a legal fp32
  RGB patch projection but H/W or output channels were not previously observed.
  The proven descriptor-view input
  leg requires zero storage offset, width-packed buffer storage, and metadata
  strides compatible with `conv2d_buffer_float`. Semantic negatives remain on
  the legacy path. The downstream `PatchEmbedFeatureMapToTokensContract`
  layout-transition slice has generated spec coverage in
  `test/vulkan_contract_specs/patch_embed_feature_map_to_tokens_contract.json`
  and generated sparse-rowset helper output in
  `generated/ExecutionContractsPatchEmbedFeatureMapToTokensSpec.h`; those
  finite rows now serve as evidence/regression fixtures. Production admission
  for fp32 direct-buffer rank-4 feature maps is handled by
  `FeatureMapToTokensDirectBuffer`, which validates `[N,C,H,W] -> [N,H*W,C]`
  semantics from runtime batch/channel/H/W metadata without exact H/W row
  matching. The current implementation still requires width-packed buffer
  storage and zero storage offset.
- Expiry: the patch-embed conv leg also has a semantic dynamic execution-plan
  family, and finite feature-map-to-token rows are no longer needed as
  migration guardrails.
- Migration target: generated `PatchEmbedFloatBufferConvRoute` execution-plan
  metadata or a `ConvWeightDeviceRepackTransitionContract` if the legacy packed
  layout is migrated to device-side repack; feature-map-to-token conversion is
  covered by `FeatureMapToTokensDirectBuffer`.

### No-Overlap ConvTranspose2D Exact Envelope

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Convolution.cpp`
- Status: temporary, contract-named
- Reason: the float-buffer 2x2 stride-2 no-overlap transposed-conv rows are
  proven evidence fixtures, and `NoOverlapConvTranspose2DContract` now has a
  `DynamicKernelStrideFloatBuffer` family covering the clean packed-buffer
  no-overlap family by semantic
  `kernel == stride`, zero padding/output-padding, dilation-one, groups-one
  rules. Batch, output channels, kernel/stride, and spatial sizes are runtime
  descriptors in the clean envelope. Low input-channel cases still hit the
  small-metadata/exact-rearrange materialization path and remain a named
  layout/materialization gap.
- Generated spec coverage: `test/vulkan_contract_specs/no_overlap_conv_transpose2d_contract.json`
  covers the `Kernel2Stride2FloatBuffer` slice with ShapeEnvelope-backed
  checked-in positive and adjacent negative runtime cases plus generic
  ShapeEnvelope C++ metadata/simple-bound helper output in
  `generated/ExecutionContractsNoOverlapConvTranspose2DSpec.h`, including
  input/weight channel equality. Output-shape arithmetic, prepack resource
  behavior, and match-result assembly remain handwritten.
- Expiry: low-channel materialization has a direct no-readback path, and
  broader conv-transpose options have dynamic semantic parity plus adjacent
  negative coverage.
- Migration target: `NoOverlapConvTranspose2DContract` dynamic families plus
  generated positive and negative tests.

### Diffusion SDPA Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  Lotus diffusion contract drafts
- Status: partially migrated, contract-named
- Reason: materialization requirements differ by square self-attention tuple,
  so enabling that slice by formula is not justified yet. Cross-attention now
  has `CrossAttentionRuntimeShape` admission for mask-free fp32 rank-4 batch-1
  tensors with matching heads/head dim, head dim `64`, small runtime key/value
  sequence length, and a bounded score tensor. Square self-attention now has
  `SquareSelfAttentionRuntimeShape` admission for `head_dim=64` and single-head
  `head_dim=512` when the score budget holds and the `512` materialized math
  path can prove a width-pack-compatible key transpose (`sequence % 4 == 0`).
  Non-compatible `512` sequences remain a layout/materialization exception, not
  an exact-row admission target.
- Generated spec coverage: `test/vulkan_contract_specs/diffusion_sdpa_contract.json`
  covers the `SparseAttentionRows` slice with ShapeEnvelope-backed checked-in
  positive and adjacent negative runtime cases plus generic ShapeEnvelope
  sparse-rowset helper output in
  `generated/ExecutionContractsDiffusionSDPASpec.h`. The generated helper owns
  the 11 correlated square and cross-attention rows, per-row metadata, exact
  `(heads, query_sequence, key_value_sequence, head_dim)` lookup, and row-match
  equality for those fields. Route-policy hard-fail ordering, scale tolerance,
  SDPA execution, materialization policy, and match-result assembly remain
  handwritten. Cross-attention random-shape tests now cover unseen legal
  key/value lengths around the old `kv=2` fixtures.
- Expiry: broader parity and materialization census covers non-width-pack
  compatible `head_dim=512` diffusion self-attention shapes, and cross-attention
  runtime admission has generated semantic positive/negative coverage rather
  than handwritten-only guards.
- Migration target: broader generated `DiffusionSDPAContract` and
  `DiffusionCrossAttentionContract` tables with positive, adjacent negative,
  materialization-policy coverage, and a square q/k/v direct-buffer layout
  command plan for non-width-pack-compatible `head_dim=512` key transposes.

### Tiny Mask SDPA Tuple

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*`
- Status: partially migrated, contract-named
- Reason: the original tiny additive-mask shape is proven and remains a
  regression fixture. Runtime additive-float-mask admission now uses
  `AdditiveFloatMaskRuntimeShape`, which checks fp32 rank-3/rank-4 Q/K/V
  compatibility, PyTorch-style mask broadcast compatibility, finite scale, and
  a bounded score-tensor budget instead of exact target/source length rows.
  Bool masks, causal mode, GQA, dropout, non-finite scale, and over-budget
  score tensors remain unsupported.
- Generated spec coverage: `test/vulkan_contract_specs/masked_tiny_sdpa_contract.json`
  covers the `AdditiveFloatMask` slice with ShapeEnvelope-backed checked-in
  positive and adjacent negative runtime cases plus generic ShapeEnvelope C++
  simple-bound helper output in
  `generated/ExecutionContractsMaskedTinySDPASpec.h`. Runtime random-shape
  tests cover unseen legal mask broadcasts through the handwritten semantic
  matcher. Route hard-fail ordering, SDPA execution, and match-result assembly
  remain handwritten.
- Expiry: bool-mask, causal-mask, and GQA mask semantics either have Vulkan
  device-resident implementations or are recorded as permanent unsupported
  semantics with transition/fallback budgets.
- Migration target: `MaskedTinySDPAContract` runtime mask families or a
  reviewed mask field in a broader SDPA contract.

## Retired Historical Records

### Linear GELU Deferred Bridge

- Status: retired; not an active exception.
- Default eager execution now produces concrete linear output followed by
  concrete GELU. The speculative candidate registry, alias propagation, and
  consumer materialization bridge were removed.
- `test/vulkan_contract_specs/linear_gelu_bridge_contract.json` remains
  graph-lowering evidence only. It must never authorize an eager deferred
  route.
- Migration target: graph-owned linear/GELU lowering and region execution.

## Rules For New Exceptions

- Add the exception before or with the route change.
- Include location, reason, expiry, and migration target.
- Keep the route narrow until tests prove broader legality.
- Remove the exception when the migration lands.
