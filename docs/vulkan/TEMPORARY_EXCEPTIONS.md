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

### Transformer GQA SDPA HY-MT-Derived Naming

- Location: SDPA route policy and related helpers
- Status: migrate
- Reason: HY-MT exposed reusable causal/GQA SDPA prefill and decode patterns.
- Expiry: next SDPA contract migration task.
- Migration target: `TransformerGQASDPAContract` with split prefill/decode
  legality, explicit scale/dropout/mask/GQA fields, and adjacent guard tests.

### KV Cache Append Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Concat.cpp`
- Status: temporary, contract-named
- Reason: finite Transformer KV-cache append rows are proven by existing cat
  tests, but broader sequence/head/layout behavior is not proven yet.
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
- Expiry: broader embedding lookup parity plus adjacent negative coverage are
  available.
- Migration target: generated `EmbeddingLookupContract` tables with positive
  and negative tests.

### GQA Repeat Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Softmax.cpp`
- Status: temporary, contract-named
- Reason: bounded repeat `[1,4,S,128]` to `[1,16,S,128]` is proven for the
  current Transformer decode envelope, but broader GQA materialization behavior
  is not proven yet.
- Expiry: broader GQA repeat parity plus adjacent negative coverage are
  available.
- Migration target: generated `GQARepeatContract` tables with positive and
  negative tests.

### SDPA Execution Policy Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Softmax.cpp`
- Status: temporary, contract-named
- Reason: materialization, repeat, and post-softmax clone decisions are proven
  for finite SDPA execution envelopes, but broader layout-transition behavior is
  not proven yet.
- Expiry: broader SDPA execution/layout parity plus adjacent negative coverage
  are available.
- Migration target: generated `SDPAExecutionPolicyContract` tables with
  positive and negative tests.

### Safe View Reshape Exact Envelope

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Shape.cpp`
- Status: temporary, contract-named
- Reason: dense direct-buffer `view` and `_reshape_alias` materialization is
  proven for the current guarded envelope, but broader view/layout behavior is
  not proven yet.
- Expiry: broader view/reshape parity plus adjacent negative coverage are
  available.
- Migration target: generated `SafeViewReshapeContract` tables with positive
  and negative tests.

### Small Metadata Padded Conv2D Exact Tuple

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  `aten/src/ATen/native/vulkan/ops/Convolution.cpp`
- Status: temporary, contract-named
- Reason: the padded low-channel buffer materialization is proven for one
  finite 2x2 conv2d tuple, but broader small-metadata padded conv behavior is
  not proven yet.
- Expiry: broader padded-conv layout parity plus adjacent negative coverage are
  available.
- Migration target: generated `SmallMetadataPaddedConv2DContract` or
  `LayoutTransitionContract` tables with positive and negative tests.

### Diffusion SDPA Exact Tuples

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*` and
  Lotus diffusion contract drafts
- Status: temporary, contract-named
- Reason: materialization requirements differ by tuple, so enabling by formula
  is not justified yet.
- Expiry: broader parity and materialization census covers adjacent diffusion
  self-attention and cross-attention shapes.
- Migration target: `DiffusionSDPAContract` and
  `DiffusionCrossAttentionContract`.

### Tiny Mask SDPA Tuple

- Location: `aten/src/ATen/native/vulkan/planning/ExecutionContracts.*`
- Status: temporary, contract-named
- Reason: a tiny additive-mask shape is proven, but mask-family behavior is not
  broad enough to merge into diffusion SDPA.
- Expiry: mask-family parity and negative tests are available.
- Migration target: `MaskedTinySDPAContract` or a reviewed mask field in an
  SDPA contract.

## Rules For New Exceptions

- Add the exception before or with the route change.
- Include location, reason, expiry, and migration target.
- Keep the route narrow until tests prove broader legality.
- Remove the exception when the migration lands.
