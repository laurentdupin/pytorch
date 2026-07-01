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

Conv workgroup tuning evidence uses a separate offline ladder. GPU timestamp
labels first become `VulkanConvPlanExactLabelEvidence.v0` rows by joining
per-kernel timing to `VulkanConvPlanKey.v0` snapshots. The optional
`VulkanConvPlanTuningTable.v0` artifact is then built only from exact-label
rows with a unique plan-key match, a matched default label, and a
`locally_improved` decision. The table is intentionally offline review memory:
`runtime_loader_enabled` must remain false until a separate runtime loader
contract and repeat-stability evidence exist.
When more than one candidate improves the same match key and capability
profile, the table keeps only the best delta as the active row and preserves the
other candidates in that row's evidence.

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
- wide6 segment mode: valid and evidence-visible for `vits_140`, but not
  promoted. After the Context canary-admission fix, a focused RX 9070
  five-repeat run selected two full segments covering blocks 0-5 and 6-11 at
  62/72 planned dispatches each, kept bridge sanity at max_abs
  `1.6391277313232422e-06`, and kept CPU fallback, sync readback, and buffer
  copies at zero. It reduced stack-planned submits from 20 to 15 over five
  repeats versus wide4, but measured about 79.9 ms mean device-resident forward
  versus wide4 at about 64.3 ms, with both modes still reporting 15 stack-owner
  retire-drain submits. The next performance path is not wider fixed segments;
  it is retire/capture ownership or shader-plan tuning with exact evidence;
- stack-exit pending-retire handoff variants are cataloged as rejected for the
  current `vits_140` lane. `private_bridge_capture_handoff` preserved bridge
  sanity and kept CPU fallback, sync readback, and copies at zero, but measured
  about 69.0 ms mean and still reported 15 stack-owner retire-drain submits.
  More importantly, private bridge captures must release after decoder bridge
  consumer completion, not at backbone stack exit, so this is the wrong release
  boundary for promotion. `residual2_norm1_carry_handoff` also preserved
  correctness and zero fallback/readback/copy counters, but measured about
  73.2 ms mean and did not reduce the timed retire-drain submits. The next
  bridge-release probe used the correct post-decoder-consumer release boundary,
  but it observed no transferable pending-retire source, kept
  `transfers_pending_retires=0`, and measured about 70.3 ms mean. The next
  segment-completion cleanup handoff probe moved exact external-recording
  cleanup pending-retire entries into the stack-exit handoff batch under
  `StackRegionSegmentCompletionRetireHandoffContract.v0`, preserved bridge
  sanity and zero fallback/readback/copy counters, but still reported 15
  stack-owner retire-drain submits and regressed to about 75.3 ms mean. Do not
  repeat stack-exit handoff as a latency path unless a later submit-plan change
  makes the moved entries reduce actual queue submits. The next ownership task
  needs either segment-local completion ownership or a distinct bridge-scoped
  handoff batch with release and restore ownership before moving entries, not
  reuse of the same stack-exit batch as a standalone optimization;
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
- DAv2 `vitl_140` through the same wide4 canary remains a separate evidence
  family: the older context-owned bridge and the Python-mediated deep split are
  unsafe blocked by Windows stack overflow, but the opt-in native
  `StackOutputBridgeDeepSplitPlanRuntime.v0` canary now runs the 24-block bridge
  as two 12-block native chunks with a device-private baton. The focused smoke
  and 10-repeat runs passed bridge sanity, kept `cpu_fallback=0` and
  `sync_readback=0`, and wrote valid artifacts. This is not a default and not
  evidence to widen the `vits` or `vitb` segment-plan rowsets;
- a benchmark-local `python_private_baton` proof canary for that deep split
  topology is also unsafe blocked. The attempted run still hit native Windows
  stack overflow inside `run_vision_backbone_stack_private_capture_debug`
  before a result artifact could be written. The benchmark now records
  `PYTORCH_VULKAN_STACK_OUTPUT_BRIDGE_DEEP_SPLIT=python_private_baton` as
  unsafe-blocked metadata in the deep split plan and fails closed before
  native bridge execution;
- stack-output bridge depth guard: accepted benchmark control-plane fix. It
  keeps bridge requests with `block_count > 12` fail-closed unless the native
  private-baton deep-split canary is explicitly requested, while preserving the
  `vitb_140` positive row. The runtime contract is
  `StackOutputBridgeDeepSplitPlanRuntime.v0`, not a larger hardcoded stack-depth
  limit;
- `conv2d_buffer_float_3x3_s1p1` 16x8 workgroup: correct but slower;
- `conv2d_buffer_float_3x3_s1p1` 16x4 workgroup: rejected as a default after
  segmented multi-GPU DAv2 checks showed RX 9070 `vits_280` gains but RX 9070
  `vits_140` and GTX 1080 `vits_280` regressions/noise;
- DAv2 `vitb_140` GPU timestamp attribution: RX 9070 and GTX 1080 performance
  gaps are dominated by buffer conv families, so the next useful work is a
  `VulkanConvPlanKey`/candidate-plan tuning path rather than another graph
  diagnostic;
- `VulkanConvPlanKey.v0` snapshot rows: accepted reporting infrastructure for
  float-buffer conv submissions, including the fused `3x3_s1p1_add` path. The
  rows include selected kernel, contract provenance, layout/storage/dtype
  classes, offsets, global/local workgroup, candidate count, cacheability,
  tunability, and device/capability profile fields needed for future
  device/driver-keyed plan tuning. This is not a behavior change and does not
  promote a new conv plan. Offline tuning result artifacts should use
  `scripts/benchmarks/vulkan_conv_plan_tuning.py` and remain keyed by
  `VulkanConvPlanKey.v0` fields plus capability-profile fields, not GPU names;
- `PYTORCH_VULKAN_CONV_PLAN_WORKGROUP_CANARY`: accepted opt-in canary for
  `Kernel3x3Stride1Pad1` workgroup candidates `3x3_s1p1_16x4` and
  `3x3_s1p1_16x8`. Defaults remain `8x8x1`; any default promotion still needs
  bounded multi-GPU evidence because static 16x4/16x8 defaults were rejected.
  A focused DAv2 `vits_140`/`vitb_140` sweep on RX 9070, GTX 1080, and RX 6700
  XT also rejected promoting either canary globally: `16x4` helped `vitb_140`
  on RX 9070/GTX 1080 but regressed `vits_140` and RX 6700 XT rows, while
  `16x8` had a different mixed win/regression pattern;
- `VulkanConvPlanTuningResult.v0`: accepted offline evidence format produced by
  `scripts/benchmarks/vulkan_conv_plan_tuning.py`. It records candidate
  decisions by plan-key and capability-profile evidence, including an optional
  exact-plan-key granularity mode. It is not consumed by runtime route selection
  yet, and exact-plan-key accepted rows still require stable per-kernel timing
  before they can become promotion evidence;
- conv-plan GPU timestamp labels: accepted reporting infrastructure that tags
  float-buffer conv timestamp rows with selected kernel, shape, attrs, and
  workgroup fields. This makes per-kernel timing joinable to
  `VulkanConvPlanKey.v0` rows. The offline tuning tool can emit
  `VulkanConvPlanTimestampSummary.v0` from a timestamp log, but this does not
  change routing or promote a plan;
- `VulkanRuntimeAttributionReport.v0`: accepted measurement infrastructure for
  phase-isolated benchmark attribution. Depth Anything V2 can isolate
  `single_image_forward_device_resident` timestamp rows after warmup and the
  report splits GPU time by kernel class, runtime label, submit phase, stack
  phase, and recent op while reporting the same phase's fallback/readback/copy,
  submit-origin, and retire-drain counters. This is not timing evidence by
  itself and must not be used as a production route table;
- conv-plan timestamp sweep over DAv2 `vits_140`/`vitb_140` on RX 9070, GTX
  1080, and RX 6700 XT: rejected broad promotion of both `3x3_s1p1_16x4` and
  `3x3_s1p1_16x8`. Both candidates were correctness-clean and hit expected
  local workgroups, but each had three improved and three regressed model/device
  rows. `16x8` is promising for `vitb_140`, yet exact-label deltas are still
  device-mixed, so promotion needs a tuning cache or narrower exact-plan policy;
- bounded exact 16x4 conv policy attempt: rejected before commit. The finite
  seven-label policy routed only exact labels that had improved on all three
  timestamp-sweep devices, but default post-policy validation still regressed
  whole-row timing on GTX 1080 and RX 6700 XT `vitb_140`; keep it as evidence
  for a future full-row tuning cache rather than a static exact rowset;
- forced float-buffer tiled linear canary for label-inferred vision-backbone
  linears: correctness blocked and backed out. The first activation attempt did
  not hit tiled kernels; the corrected probe did hit
  `aten::linear.buffer_float_tiled_bias[_gelu]` rows, but `vits_140` bridge
  sanity failed at `max_abs=0.21523737907409668` and device-resident timing was
  about 71.1 ms mean. Do not reintroduce a broad force-tiled linear gate; a
  future linear plan needs a parity-proven kernel or narrower contract before
  timing;
- decoder-tail ReLU via conv clamp: correct but slower;
- fused Depth Anything V2 head shader path: correctness blocked;
- compiled-session bridge/replay shortcut: unsafe blocked;
- current-topology single-boundary submit elision: unsafe blocked.

These entries are intentionally generic where possible: the model is evidence
provenance, while the contract/topology scopes name reusable backend behavior.
