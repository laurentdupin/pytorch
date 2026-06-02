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
- Expiry: contract table metadata and generated tests cover the family well
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

### HY-MT-Derived KV Cache Append Names

- Location: `aten/src/ATen/native/vulkan/ops/Concat.cpp`
- Status: migrate
- Reason: helpers implement a reusable Transformer KV-cache append pattern but
  retain HY-MT names.
- Expiry: next concat/KV contract migration task.
- Migration target: `KVCacheAppendContract` and `CatAxisContract`.

### HY-MT-Derived GQA Repeat Envelope

- Location: SDPA/GQA repeat materialization helpers
- Status: migrate
- Reason: bounded repeat `[1,4,S,128]` to `[1,16,S,128]` is a reusable
  Transformer contract, while direct decode GQA remains disabled.
- Expiry: next SDPA materialization contract task.
- Migration target: `GQARepeatContract`.

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
