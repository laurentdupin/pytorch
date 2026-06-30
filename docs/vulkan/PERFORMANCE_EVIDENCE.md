# Vulkan Performance Evidence

This page catalogs performance-plan evidence that is not shape admission. It
exists to stop repeated diagnostics on the same route, shape family, or
recording topology after the project has already measured or rejected it.

Execution contracts still answer whether a shape/layout/dtype is legal.
Transition contracts still answer which materialization, copy, readback, or
layout edge is allowed. This evidence catalog answers a different question:

```
For a legal path, which execution plan or topology has already been tried, and
what was the decision?
```

The machine-readable ledger is
`test/vulkan_contract_proofs/performance_plan_evidence_manifest.json`.
Bounded segmented stack-region rowsets are additionally summarized in
`test/vulkan_contract_proofs/stack_region_segment_plan_manifest.json`. That
manifest records which finite model/input rowsets have proven a
`StackRegionSegmentPlan.v0` canary and links back to the per-row evidence
entries. It is review memory only, not a production route table.

Depth Anything V2 benchmark artifacts also include
`vulkan_stack_region_segment_plan` when `StackRegionSegmentPlan.v0` rows are
present in the Vulkan debug snapshot. Treat that field as the per-run catalog
for segmented stack-owned recording: it records observed modes, status/fail
reason counts, dispatch-budget counts, and sampled segment rows for the
measured model/input. Promote durable decisions from that artifact into the
manifest instead of leaving them only in ignored `agent_space` files.

## Status Values

- `accepted_default_fix`: behavior has been accepted as the normal safe path.
- `accepted_canary`: opt-in path is correct and useful, but not a default.
- `rejected_slower`: candidate was correct but slower for the recorded target.
- `correctness_blocked`: candidate failed or lacks required parity proof.
- `unsafe_blocked`: candidate exposed corruption, stack overflow, device loss,
  early destroy, double retire, or another safety blocker.
- `proposed_unstarted`: idea recorded only to reserve a future proof task.

Rejected performance evidence does not invalidate the underlying execution
contract. A slower-but-correct path may remain a valid fallback, canary, or
future autotune candidate. It must not be promoted to the default plan unless a
new entry records the changed device, driver, shader, topology, or benchmark
condition that justifies revisiting it.

## Update Rules

Before trying a Vulkan performance candidate, search the manifest by:

- contract or topology scope
- model/input provenance
- candidate id
- shader/kernel/route label
- device and driver notes

For segmented stack-owned recording, also inspect the benchmark artifact's
`vulkan_stack_region_segment_plan` field when it is available. If a mode is
already cataloged as accepted, slower, rejected, or unsafe for the same
model/input/device/topology, update the existing evidence entry instead of
rerunning the same diagnostic sweep.
If a whole finite rowset has been proven, update
`stack_region_segment_plan_manifest.json` rather than relying on the individual
artifact list. Do not broaden a rowset across model variants: `vits`, `vitb`,
and `vitl` need separate rowset entries and separate evidence ids.

Use the query helper before launching long-running diagnostics:

```
python tools/vulkan_contract_codegen/query_performance_evidence.py \
  --query wide4 --query 140x210
```

The helper searches the checked-in manifest recursively and can also summarize
the segment-plan catalog from a benchmark artifact:

```
python tools/vulkan_contract_codegen/query_performance_evidence.py \
  --artifact agent_space/dav2_vits140_wide4.json
```

Add or update an entry when a task:

- accepts a default performance fix;
- creates or changes an opt-in canary;
- proves a candidate correct but slower;
- blocks a candidate for correctness or safety;
- changes a revisit condition for a previous decision.

Each entry must include:

- stable `id`
- `status`
- `head` or commit/range that produced the evidence
- model/input provenance used for measurement
- device notes
- contract or topology scope
- candidate description
- correctness result
- counter result, especially fallback/readback/copy/sync state
- timing result when timing is meaningful
- artifacts, even if they are ignored `agent_space` paths
- decision
- revisit conditions

Do not use this manifest as a production route table. It is review memory and
planning input only.

## Current DAv2 Entries

The initial catalog records the current `vits_140` performance lane:

- stream-sync persistent external-recording pool reset: accepted default fix;
- segmented stack-owned wide4 bridge canary: accepted canary;
- context-owned stack-output bridge repeated timing: unsafe blocked because
  three-repeat bridge runs stack-overflow while one-repeat sanity passes;
- benchmark preflight guard for that unsafe context-owned repeat topology:
  accepted default control-plane fix;
- wide3 and prefix-tail segment modes: valid in the latest `vits_140`
  three-repeat sweep, but slower than wide4 on the recorded RX 9070 lane;
- the first `vits_182` wide4 graph-catalog and timing runs: bridge sanity and
  `StackRegionSegmentPlan.v0` evidence passed, and a separate no-graph
  three-repeat timing pass measured about 74.8 ms mean device-resident forward;
- the first `vits_280` wide4 graph-catalog and timing runs: bridge sanity and
  `StackRegionSegmentPlan.v0` evidence passed, and a separate no-graph
  three-repeat timing pass measured about 91.9 ms mean device-resident forward;
- the first `vits_420` wide4 graph-catalog and timing runs: bridge sanity and
  `StackRegionSegmentPlan.v0` evidence passed, and a separate no-graph
  three-repeat timing pass measured about 127.7 ms mean device-resident forward;
- the first `vits_560` wide4 graph-catalog and timing runs: bridge sanity and
  `StackRegionSegmentPlan.v0` evidence passed, and a separate no-graph
  three-repeat timing pass measured about 316.7 ms mean device-resident forward;
- the first `vits_700` wide4 graph-catalog and timing runs: bridge sanity and
  `StackRegionSegmentPlan.v0` evidence passed, and a separate no-graph
  three-repeat timing pass measured about 446.8 ms mean device-resident forward;
- a finite DAv2 `vits` wide4 rowset summary now groups the six measured input
  sizes as one opt-in canary evidence family; it must not be broadened to
  `vitb`, `vitl`, other models, or default behavior without separate rowset
  evidence. The rowset is also recorded in
  `stack_region_segment_plan_manifest.json` so future agents can find the
  accepted canary boundary without repeating the graph cataloging work;
- the first DAv2 `vitb_140` wide4 graph-catalog and timing runs: bridge sanity
  passed with zero CPU fallback/readback, `StackRegionSegmentPlan.v0` showed
  the same 20 accepted / 36 rejected row pattern as the `vits` rowset, and the
  separate no-graph timing pass measured about 110.2 ms mean device-resident
  forward. This is a separate one-row `vitb` canary rowset, not an expansion of
  the `vits` rowset;
- DAv2 `vitl_140` through the same wide4 canary remains unsafe blocked: the
  graph-summary probe emitted segment-plan rows, but the process exited with
  Windows stack overflow `-1073741571` before writing a benchmark result
  artifact. Do not infer `vitl` support from `vits` or `vitb` evidence;
- `conv2d_buffer_float_3x3_s1p1` 16x8 workgroup: correct but slower;
- decoder-tail ReLU via conv clamp: correct but slower;
- fused Depth Anything V2 head shader path: correctness blocked;
- compiled-session bridge/replay shortcut: unsafe blocked;
- current-topology single-boundary submit elision: unsafe blocked.

These entries are intentionally generic where possible: the model is evidence
provenance, while the contract/topology scopes name reusable backend behavior.
