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

### Exact Tuple Rows In Contract Tables

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*`
- Status: temporary but allowed
- Reason: finite rows are proven by existing artifacts and keep unsupported
  adjacent shapes guarded.
- Metadata: each new exact row must carry contract/family/tuple/evidence/guard,
  fallback, and materialization metadata. Metadata is a migration guardrail, not
  an expiry condition by itself.
- Expiry: generated positive and adjacent negative tests cover the family well
  enough to review a parameterized policy.
- Migration target: generated `KernelFamilyContract` tables with positive and
  negative tests.

### KV Cache Append Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Concat.cpp`
- Status: temporary, contract-named
- Reason: finite Transformer KV-cache append rows are proven by existing cat
  tests, but broader sequence/head/layout behavior is not proven yet.
- Generated spec coverage: `test/vulkan_contract_specs/kv_cache_append_contract.json`
  covers the `SequenceAppend` slice and
  `test/vulkan_contract_specs/kv_cache_append_initial_contract.json` covers the
  `InitialCache` slice with ShapeEnvelope-backed checked-in positive and
  adjacent negative runtime cases plus generic ShapeEnvelope C++
  metadata/simple-bound helper output in
  `generated/ExecutionContractsKVCacheAppendSpec.h` and
  `generated/ExecutionContractsKVCacheAppendInitialSpec.h`. SequenceAppend
  batch/heads/head-dim equality is generated while initial-empty handling,
  sequence lower bounds, InitialCache cross-input handling, and match-result
  assembly remain handwritten. InitialCache positives log
  `aten::cat.kv_cache_initial_dim2_buffer`.
- Expiry: broader KV-cache append and cat-axis parity plus adjacent negative
  coverage are available.
- Migration target: generated `KVCacheAppendContract` and `CatAxisContract`
  tables with positive and negative tests.

### Embedding Lookup Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Indexing.cpp`
- Status: temporary, contract-named
- Reason: finite token-batch and small-bounded embedding lookup rows are proven
  by existing embedding tests, but broader vocab, index-rank, and layout
  behavior is not proven yet.
- Generated spec coverage: `test/vulkan_contract_specs/embedding_lookup_contract.json`
  covers the small-bounded lookup slice with generated positive and adjacent
  negative runtime tests and generic ShapeEnvelope C++ metadata/helper output,
  including the derived indices product helper.
  Other embedding rows still need broader generated coverage before the
  exception can expire.
- Expiry: broader embedding lookup parity plus adjacent negative coverage are
  available.
- Migration target: generated `EmbeddingLookupContract` tables with positive
  and negative tests.

### Cat Axis And Channel Cat Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Concat.cpp`
- Status: temporary, contract-named
- Reason: bounded cat rows are proven for current last-dim, channel-dim, and
  rank-3 patterns, but broader axis/layout behavior is not proven yet.
- Generated spec coverage: `test/vulkan_contract_specs/channel_cat_contract.json`
  covers the rank-4 dim-1 channel-cat buffer slice with generated positive and
  adjacent negative runtime tests and generic ShapeEnvelope C++ metadata/helper
  output. Other cat-axis rows still need fixtures or documented follow-up
  before this exception can expire.
- Expiry: broader cat-axis parity plus adjacent negative coverage are
  available.
- Migration target: generated `CatAxisContract` and `ChannelCatContract`
  tables with positive and negative tests.

### Token Prefix Cat/Add Exact Rowset

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*`,
  `aten/src/ATen/native/vulkan/ops/VisionBlocks.cpp`, and
  `scripts/benchmarks/benchmark_depth_anything.py`
- Status: temporary, contract-named
- Reason: the fused prefix-token concat plus position-add route is proven for
  the observed rank-3 float token-preparation envelope, but arbitrary cat+add
  fusion and split-token consumer regions are not proven.
- Generated spec coverage:
  `test/vulkan_contract_specs/token_prefix_cat_add_contract.json` covers
  `prefix=[1,1,C]`, `tokens=[1,N,C]`, `pos/out=[1,N+1,C]`,
  `C in {384,768,1024}`, and
  `N in {150,260,600,620,1350,1380,2400,2440,3750,3850}` with generated C++
  sparse-rowset helper output in
  `generated/ExecutionContractsTokenPrefixCatAddSpec.h`.
- Expiry: broader token-prefix concat/add and downstream region-fusion parity
  plus adjacent negative coverage are available.
- Migration target: generated `TokenPrefixCatAddContract` or token-preparation
  `RegionContract` tables with positive and negative tests.

### GQA Repeat Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Softmax.cpp`
- Status: temporary, contract-named
- Reason: bounded repeat `[1,4,S,128]` to `[1,16,S,128]` is proven for the
  current Transformer decode envelope, but broader GQA materialization behavior
  is not proven yet.
- Generated spec coverage: `test/vulkan_contract_specs/gqa_repeat_contract.json`
  covers the `Batch1Heads4Factor4Sequence100To116Dim128` slice with
  ShapeEnvelope-backed checked-in positive and adjacent negative runtime cases
  plus generic ShapeEnvelope C++ simple-bound helper output in
  `generated/ExecutionContractsGQARepeatSpec.h`. Vulkan tensor/storage
  extraction, SDPA admission, materialization allocation and dispatch, op-hit
  labels, sequence lower-bound preservation, and match-result assembly remain
  handwritten.
- Expiry: broader GQA repeat parity plus adjacent negative coverage are
  available.
- Migration target: broader generated `GQARepeatContract` tables with positive
  and negative tests.

### BatchNorm Inference Exact Envelopes

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Batchnorm.cpp`
- Status: temporary, contract-named
- Reason: float32 4D inference batch norm is proven for current buffer and
  materialized-buffer paths, but broader dtype/rank/layout/training behavior is
  not proven yet.
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
- Expiry: broader batch-norm inference parity plus adjacent negative coverage
  are available for buffer and materialized-buffer families.
- Migration target: broader generated `BatchNormInferenceContract` tables with
  positive and negative tests.

### Elementwise Broadcast Exact Envelope

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/BinaryOp.cpp`
- Status: temporary, contract-named
- Reason: the float32 tensor/tensor buffer-broadcast route is proven for a
  narrow `add`/`mul`/`sub` canary envelope, but broader binary-op,
  dtype-promotion, `out=`, inplace, and scalar behavior is not proven yet.
- Generated spec coverage: `test/vulkan_contract_specs/elementwise_broadcast_contract.json`
  covers the `FloatTensorTensorBufferBroadcast` slice with ShapeEnvelope-backed
  checked-in positive and adjacent negative runtime cases, including the
  bounded `sub` op-axis slice. Positives keep the existing
  `aten::binary_op.buffer_float` route label and record contract admission
  metadata in tensor provenance. The slice also has generated C++ metadata,
  simple-bound, op-attribute, and `broadcast_compatible` helper coverage in
  `generated/ExecutionContractsElementwiseBroadcastSpec.h`.
- Expiry: broader elementwise broadcast parity plus adjacent negative coverage
  are available for tensor/tensor, scalar, `out=`, inplace, and promotion
  families.
- Migration target: broader generated `ElementwiseBroadcastContract` tables
  with positive and negative tests.

### Transformer GQA SDPA Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*`,
  `aten/src/ATen/native/vulkan/planning/RoutePolicy.cpp`, and
  `aten/src/ATen/native/vulkan/ops/Softmax.cpp`
- Status: temporary, contract-named
- Reason: bounded Transformer causal/prefill and decode GQA SDPA rows are
  proven for the current envelope, but broader Transformer attention shapes,
  masks, scale policy, and direct decode GQA behavior are not proven yet.
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
  are available without direct decode GQA broadening.
- Migration target: broader generated `TransformerGQASDPAContract` tables with
  positive, adjacent negative, and materialization-policy coverage.

### Vision Self-Attention SDPA Exact Rows

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*`,
  `aten/src/ATen/native/vulkan/planning/RoutePolicy.cpp`, and
  `aten/src/ATen/native/vulkan/ops/Softmax.cpp`
- Status: temporary, contract-named
- Reason: six low-resolution rank-3 float vision self-attention rows
  `[BH,T,64]` are proven with `BH in {6,12,16}`, `T in {151,261}`, q/k/v equal
  shape, no mask, non-causal, dropout 0, GQA off, and explicit scale 1.0.
  Broader vision SDPA sequence/head layouts, implicit-scale policy, masks,
  causal attention, and direct softmax-to-value-BMM behavior are not proven
  yet.
- Generated spec coverage:
  `test/vulkan_contract_specs/vision_self_attention_sdpa_contract.json` covers
  the `SparseAttentionRows` slice with ShapeEnvelope sparse-rowset rows,
  checked-in positive and adjacent negative runtime cases, and generic
  ShapeEnvelope C++ metadata/row helpers in
  `generated/ExecutionContractsVisionSelfAttentionSDPASpec.h`. The generated
  helper owns row metadata and row-match bounds. Scale tolerance,
  route-policy hard-fail ordering, tensor extraction/early dtype-rank guards,
  materialized math-path selection, post-softmax clone behavior, and
  match-result assembly remain handwritten. The score-softmax probability
  materialization edge is bounded to the same generated rowset via
  `SDPAScoreSoftmaxContract` `VisionSelfAttentionScores`; it writes
  probabilities into a fresh direct buffer and does not enable the previously
  failing direct softmax-probability-to-value-BMM path.
- Expiry: broader vision self-attention SDPA parity and adjacent negative
  coverage are available across head-batch, sequence, scale, mask/causal, and
  probability materialization behavior without regressing existing SDPA rows.
- Migration target: broader generated `VisionSelfAttentionSDPAContract` tables
  and/or a reviewed attention probability materialization policy that can
  replace the exact rowset with a parameterized, zero-fallback policy.

### SDPA Execution Policy Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Softmax.cpp`
- Status: temporary, contract-named
- Reason: materialization, repeat, and post-softmax clone decisions are proven
  for finite SDPA execution envelopes, but broader layout-transition behavior is
  not proven yet.
- Generated spec coverage: `test/vulkan_contract_specs/sdpa_execution_policy_contract.json`
  covers the `SparsePolicyRows` slice with ShapeEnvelope-backed checked-in
  positive and adjacent negative runtime cases plus generic ShapeEnvelope C++
  sparse-rowset helper output in
  `generated/ExecutionContractsSDPAExecutionPolicySpec.h`. The generated
  helper owns row-match bounds for the correlated policy rows while diffusion
  contract admission, tuple-id cross-checks, scale tolerance, score
  pre-materialization, materialized math path, post-softmax clone behavior, and
  match-result assembly remain handwritten.
- Expiry: broader SDPA execution/layout parity plus adjacent negative coverage
  are available.
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
- Status: temporary, contract-named
- Reason: finite SDPA score-softmax shapes guard route hard-fail and native
  buffer softmax eligibility, but broader score-softmax/layout behavior is not
  proven yet.
- Generated spec coverage: `test/vulkan_contract_specs/sdpa_score_softmax_contract.json`
  covers the `DiffusionSquareScores` slice with ShapeEnvelope-backed
  checked-in positive and adjacent negative runtime cases plus generic
  ShapeEnvelope C++ simple-bound helper output in
  `generated/ExecutionContractsSDPAScoreSoftmaxSpec.h`. The
  `VisionSelfAttentionScores` slice is production-wired but reuses
  `generated/ExecutionContractsVisionSelfAttentionSDPASpec.h` as its row
  source of truth until a dedicated multi-family score-softmax fixture is
  justified. Softmax route ordering, `can_run_buffer_softmax` policy, guard
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

### Linear GELU Bridge Exact Envelope

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Mm.cpp`
- Status: temporary, contract-named
- Reason: the deferred linear/GELU bridge is proven for the current
  `BackboneMlpHidden384To1536` legality envelope, but broader hidden sizes,
  output sizes, rank behavior, `out=`, alpha/beta, post-op, inference-mode,
  and GELU-consumption behavior are not proven yet.
- Generated spec coverage: `test/vulkan_contract_specs/linear_gelu_bridge_contract.json`
  covers the `BackboneMlpHidden384To1536` slice with ShapeEnvelope-backed
  checked-in positive and adjacent negative runtime cases plus generic
  ShapeEnvelope C++ simple-bound helper output in
  `generated/ExecutionContractsLinearGeluBridgeSpec.h`. Tensor-info
  extraction, rank-3 equality, deferred candidate registry ownership, alias
  retargeting, materialization on non-GELU consumers, fused-GELU execution,
  op-hit labels, and match-result assembly remain handwritten.
- Expiry: broader linear/GELU bridge parity plus adjacent negative coverage
  are available across hidden/output sizes, rank behavior, option handling, and
  GELU approximation consumption.
- Migration target: generated `LinearGeluBridgeContract` tables with positive
  and negative tests plus a reviewed side-effect boundary for deferred
  registry and materialization behavior.

### Small Metadata Padded Conv2D Exact Tuple

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Convolution.cpp`
- Status: temporary, contract-named
- Reason: the padded low-channel buffer materialization is proven for one
  finite 2x2 conv2d tuple, but broader small-metadata padded conv behavior is
  not proven yet.
- Generated spec coverage: `test/vulkan_contract_specs/small_metadata_padded_conv2d_contract.json`
  covers the `MaterializedBufferInput2x2` slice with ShapeEnvelope-backed
  checked-in positive and adjacent negative runtime cases plus generic
  ShapeEnvelope C++ exact simple-bound helper output in
  `generated/ExecutionContractsSmallMetadataPaddedConv2DSpec.h`, including
  input/weight channel equality. Tensor-info extraction, input materialization,
  op-hit logging, fallback to
  `aten::convolution.buffer_float_skip.small_metadata_input`, and match-result
  assembly remain handwritten.
- Expiry: broader padded-conv layout parity plus adjacent negative coverage are
  available.
- Migration target: generated `SmallMetadataPaddedConv2DContract` or
  `LayoutTransitionContract` tables with positive and negative tests.

### Small Spatial Pointwise Conv Bounded Rows

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Convolution.cpp`
- Status: temporary, contract-named
- Reason: finite 1x1 pointwise projection rows and one bounded factorized
  depth-vision projection group are proven for current projection envelopes,
  but broader pointwise shape/layout behavior is not proven yet.
- Generated spec coverage: `test/vulkan_contract_specs/small_spatial_pointwise_conv_contract.json`
  covers the `SparseProjectionRows` slice with ShapeEnvelope-backed checked-in
  positive and adjacent negative runtime cases plus generic ShapeEnvelope
  sparse-rowset helper output in
  `generated/ExecutionContractsSmallSpatialPointwiseConvSpec.h`. The generated
  helper owns the 51 correlated projection rows, per-row metadata,
  input/weight channel equality, and exact `(input_c, input_h, input_w,
  output_c)` lookup. It also owns the cross-adapter proven 144-shape
  factorized depth-vision projection group: 18 approved channel pairs crossed
  with eight approved spatial pairs, including 84 validated corpus/proof
  shapes and 60 proven extrapolations. The twelve newly admitted mid-resolution
  depth-vision rows are exact sparse rows only; their `(30,45)` and `(40,62)`
  spatial pairs are not part of the factorized group. Naive min/max envelopes,
  independent H/W cross-products, and wider channel/spatial cross-products
  remain rejected by `KnownBadLargePointwiseConv`. Route-policy hard-fail rescue,
  shader-family decisions, family op-hit labels, and match-result assembly
  remain handwritten.
- Expiry: broader pointwise conv parity plus adjacent negative coverage are
  available across layout, storage, channel-pair, spatial-pair, and
  output-channel families.
- Migration target: generated `SmallSpatialPointwiseConvContract` or broader
  pointwise `KernelFamilyContract` tables with positive and negative tests.

### Patch Embed Float-Buffer Conv Route

- Location: `aten/src/ATen/native/vulkan/ops/Convolution.cpp`
- Status: temporary, contract-named
- Reason: a bounded patch-embed conv family can avoid the value-bearing legacy
  conv weight CPU repack/readback by using the existing float-buffer conv route,
  but broader kernel-14/stride-14 conv layout behavior is not proven yet.
- Generated spec coverage: none yet. The current route predicate is handwritten
  and limited to float Vulkan tensors with input `[1,3,H,W]`, `(H,W)` in
  `{(140,210),(280,434)}`, weight `[C,3,14,14]`, `C in {384,768,1024}`,
  stride `[14,14]`, zero padding, dilation `[1,1]`, and groups `1`. The proven
  descriptor-view input leg requires zero storage offset, width-packed buffer
  storage, and metadata strides compatible with `conv2d_buffer_float`.
  Adjacent negatives remain on the legacy path. The downstream
  `PatchEmbedFeatureMapToTokensContract` layout-transition slice now has
  generated spec coverage in
  `test/vulkan_contract_specs/patch_embed_feature_map_to_tokens_contract.json`
  and generated sparse-rowset helper output in
  `generated/ExecutionContractsPatchEmbedFeatureMapToTokensSpec.h`; it covers
  only `[1,C,H,W] -> [1,H*W,C]` for `C in {384,768,1024}` and feature spatial
  pairs `(10,15)` and `(20,31)`.
- Expiry: generated patch-embed conv execution-plan or transition-contract
  coverage exists with positive and adjacent negative tests, including input
  layout/materialization accounting.
- Migration target: generated `PatchEmbedFloatBufferConvRoute` execution-plan
  metadata or a `ConvWeightDeviceRepackTransitionContract` if the legacy packed
  layout is migrated to device-side repack.

### No-Overlap ConvTranspose2D Exact Envelope

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Convolution.cpp`
- Status: temporary, contract-named
- Reason: the float-buffer 2x2 stride-2 no-overlap transposed-conv envelope is
  proven for bounded current topologies, but broader transposed-conv
  shape/options behavior is not proven yet.
- Generated spec coverage: `test/vulkan_contract_specs/no_overlap_conv_transpose2d_contract.json`
  covers the `Kernel2Stride2FloatBuffer` slice with ShapeEnvelope-backed
  checked-in positive and adjacent negative runtime cases plus generic
  ShapeEnvelope C++ metadata/simple-bound helper output in
  `generated/ExecutionContractsNoOverlapConvTranspose2DSpec.h`, including
  input/weight channel equality. Output-shape arithmetic, prepack resource
  behavior, and match-result assembly remain handwritten.
- Expiry: broader conv-transpose parity plus adjacent negative coverage are
  available.
- Migration target: generated `NoOverlapConvTranspose2DContract` tables with
  positive and negative tests.

### Diffusion SDPA Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  Lotus diffusion contract drafts
- Status: temporary, contract-named
- Reason: materialization requirements differ by tuple, so enabling by formula
  is not justified yet.
- Generated spec coverage: `test/vulkan_contract_specs/diffusion_sdpa_contract.json`
  covers the `SparseAttentionRows` slice with ShapeEnvelope-backed checked-in
  positive and adjacent negative runtime cases plus generic ShapeEnvelope
  sparse-rowset helper output in
  `generated/ExecutionContractsDiffusionSDPASpec.h`. The generated helper owns
  the 11 correlated square and cross-attention rows, per-row metadata, exact
  `(heads, query_sequence, key_value_sequence, head_dim)` lookup, and row-match
  equality for those fields. Route-policy hard-fail ordering, scale tolerance,
  SDPA execution, materialization policy, and match-result assembly remain
  handwritten.
- Expiry: broader parity and materialization census covers adjacent diffusion
  self-attention and cross-attention shapes.
- Migration target: broader generated `DiffusionSDPAContract` and
  `DiffusionCrossAttentionContract` tables with positive, adjacent negative,
  and materialization-policy coverage.

### Tiny Mask SDPA Tuple

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*`
- Status: temporary, contract-named
- Reason: a tiny additive-mask shape is proven, but mask-family behavior is not
  broad enough to merge into diffusion SDPA.
- Generated spec coverage: `test/vulkan_contract_specs/masked_tiny_sdpa_contract.json`
  covers the `AdditiveFloatMask` slice with ShapeEnvelope-backed checked-in
  positive and adjacent negative runtime cases plus generic ShapeEnvelope C++
  simple-bound helper output in
  `generated/ExecutionContractsMaskedTinySDPASpec.h`. Route hard-fail
  ordering, scale tolerance, SDPA execution, and match-result assembly remain
  handwritten.
- Expiry: mask-family parity and negative tests are available.
- Migration target: `MaskedTinySDPAContract` or a reviewed mask field in an
  SDPA contract.

## Rules For New Exceptions

- Add the exception before or with the route change.
- Include location, reason, expiry, and migration target.
- Keep the route narrow until tests prove broader legality.
- Remove the exception when the migration lands.
