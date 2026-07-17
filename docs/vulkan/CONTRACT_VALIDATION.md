# Vulkan Contract Validation

Vulkan contracts are allowed to admit bounded rows only when the proof and the
runtime matcher stay in sync. This page describes the proof-carrying validation
layer that sits beside the existing JSON specs and generated C++ helpers. It is
governance only: it does not change route policy, fallback behavior, copy
behavior, submit policy, or accepted shapes.

## Accepted Row Manifest

`test/vulkan_contract_proofs/accepted_contract_rows_manifest.json` is the
checked-in manifest of currently admitted contract rows and groups. It is
generated from `test/vulkan_contract_specs/*.json` by:

```
python tools/vulkan_contract_codegen/compare_contract_admission.py \
  --repo-root . \
  --write-accepted-manifest test/vulkan_contract_proofs/accepted_contract_rows_manifest.json
```

Each manifest entry records the practical admission surface:

- contract name and family
- tuple id
- admission kind: `exact_sparse_row`, `factorized_group`,
  `bounded_envelope`, or transition reason bucket
- row or group fields when available
- fallback and materialization policy
- extrapolation class and cardinality estimate
- generated helper source
- route labels when present

The manifest is intentionally generated from the source-of-truth specs. It is
not a second route table and must not be consulted by production code.

## Dependency Digests

Each accepted row or group carries deterministic dependency digests:

- JSON spec hash
- generated C++ helper hash when a generated helper exists
- matcher source hash for known high-risk contracts
- route-policy source hash for known high-risk contracts
- transition-contract source hash for transition rows
- expected counter policy digest when declared

The first version focuses on practical drift detection. If a matcher,
generated helper, route policy, or transition classifier changes under a row,
the entry hash changes and the checked-in manifest validation fails until the
proof ledger is reviewed and regenerated.

## Proof Manifest

`test/vulkan_contract_proofs/contract_proof_manifest.json` maps admitted rows
and groups to proof status. The initial high-risk coverage includes:

- `AttentionProbabilityMaterializationContract`
- `PatchEmbedFloatBufferConvRoute`
- `SmallSpatialPointwiseConvContract`

For covered contracts, every admitted entry must have a proof-manifest entry.
The entry records positive proof/runtime cases when known, adjacent negative
coverage, fallback/readback/copy budget, proof dependency digest, expiry or
migration target for exact-row debt, and proof status. Broader coverage is
tracked as explicit manifest debt instead of silently omitted coverage.

Generate the proof manifest with:

```
python tools/vulkan_contract_codegen/compare_contract_admission.py \
  --repo-root . \
  --write-proof-manifest test/vulkan_contract_proofs/contract_proof_manifest.json
```

## Comparing Admissions

`tools/vulkan_contract_codegen/compare_contract_admission.py` compares two
accepted-row manifests and reports:

- newly admitted rows
- removed rows
- metadata changes
- dependency hash changes and stale-proof candidates
- cardinality increases
- exact-row debt changes

Example:

```
python tools/vulkan_contract_codegen/compare_contract_admission.py \
  --repo-root . \
  --baseline-manifest agent_space/baseline_contract_rows.json \
  --current-manifest agent_space/current_contract_rows.json \
  --report-json agent_space/contract_admission_delta.json \
  --report-md agent_space/contract_admission_delta.md
```

The tool can also generate a manifest directly from a historical commit with
`--git-ref <rev>` for review-time comparisons.

## Governance Checks

The focused governance tests fail when:

- the accepted-row manifest is stale relative to the JSON specs, generated
  helpers, or known matcher/route/transition dependencies
- a covered contract admits a row without a proof-manifest entry
- proof entries point at a stale dependency digest
- the comparison tool stops reporting cardinality, metadata, dependency, or
  exact-row-debt deltas

Run the checks directly with:

```
python tools/vulkan_contract_codegen/compare_contract_admission.py --repo-root . --self-test
python tools/vulkan_contract_codegen/compare_contract_admission.py --repo-root . --validate-accepted-manifest --accepted-manifest test/vulkan_contract_proofs/accepted_contract_rows_manifest.json
python tools/vulkan_contract_codegen/compare_contract_admission.py --repo-root . --validate-proof-manifest --proof-manifest test/vulkan_contract_proofs/contract_proof_manifest.json
```

The same checks are also wired into `TestVulkanGovernance` in
`test/test_vulkan.py`.

## Update Workflow

When changing a Vulkan contract:

1. Update the JSON spec and generated helper through the existing contract
   generation flow.
2. Regenerate the accepted-row manifest.
3. Regenerate or manually review the proof manifest for covered contracts.
4. Compare against a baseline manifest and include the admission delta in the
   review artifacts.
5. Keep exact sparse rows tied to expiry and migration targets in
   `TEMPORARY_EXCEPTIONS.md` until broader generated proof coverage exists.

Do not treat a clean manifest update as proof that a behavior change is safe.
Behavior changes still need the contract-specific runtime parity, fallback,
readback, copy, and adjacent-negative validation required by the Vulkan
planning docs.

## Performance Plan Evidence

Shape admission manifests intentionally do not record every performance
candidate. Correct-but-slower routes, unsafe recording topologies, and opt-in
canaries live in
`test/vulkan_contract_proofs/performance_plan_evidence_manifest.json` and are
described in `docs/vulkan/PERFORMANCE_EVIDENCE.md`.
Finite `StackRegionSegmentPlan.v0` canary rowsets are summarized separately in
`test/vulkan_contract_proofs/stack_region_segment_plan_manifest.json` so a
group of proven benchmark rows has a durable boundary without becoming a
production dispatch table.

Use that manifest before retrying a shader workgroup, region-recording topology,
fusion path, replay shortcut, or other execution-plan candidate. A
`rejected_slower` entry does not make the route illegal: it only says the
recorded device/input/topology did not justify promoting that plan. A future
autotune or device-keyed plan cache may revisit it only under the entry's
`revisit_conditions`.

For Depth Anything V2 segmented stack-owned recording runs, first inspect the
benchmark artifact's `vulkan_stack_region_segment_plan` field. It is the
per-run catalog for `StackRegionSegmentPlan.v0` rows and should be promoted to
the performance evidence manifest whenever it changes a durable decision.
When the same segment plan is proven across a finite rowset, add or update a
rowset in `stack_region_segment_plan_manifest.json`. Keep variants separate:
`vits` evidence does not admit `vitb` or `vitl`, and non-DAv2 corpus evidence
needs its own rowset.
