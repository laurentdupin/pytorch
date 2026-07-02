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
The compact current decision table is
`test/vulkan_contract_proofs/stack_performance_canary_decision_table.json`;
check it before adding another DAv2-driven canary so already viable, slower,
unsafe, evidence-only, and blocked paths are not rediscovered.

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
  makes the moved entries reduce actual queue submits. The runtime env gate for
  this rejected canary is retired; external-recording cleanup rows remain
  metadata-only. The next ownership task needs either segment-local completion
  ownership or a distinct bridge-scoped handoff batch with release and restore
  ownership before moving entries, not reuse of the same stack-exit batch as a
  standalone optimization;
- stack-owned segment retire-drain deferral:
  the previous opt-in
  `PYTORCH_VULKAN_STACK_REGION_RETIRE_DRAIN_DEFER=stack_planned_region_exit`
  evidence remains rejected for the pre-cleanup-batching path, where removing
  the timed `retire_queue_drain` submits moved cost to later lifetime work.
  After stack-planned cleanup callback batching, the same narrow
  active-stack-owned-recording/current-thread/resource-budget gate is default.
  A warmup-3/repeat-30 RX 9070 `vits_140` bridge run stayed correctness-clean,
  kept CPU fallback, sync readback, and timed buffer copies at zero, removed the
  three timed retire-drain queue submits per request, and measured about
  65.9 ms mean / 66.1 ms median / 67.1 ms p95 device-resident forward.
  The fast path records the deferred pending resources through the existing
  retire-drain counters; `disabled` restores the old submit behavior for
  diagnostics;
- `StackProgramOwnedTempStabilityContract.v0` is accepted as reporting
  infrastructure for that next control-plane task. It records program-owned
  internal-temp descriptor counts and distinguishes `stable_for_re_record=1`
  from `stable_for_command_replay=0` with
  `fail_closed_reason=program_owned_temp_slot_identity_unproven`. This is not a
  replay promotion and does not change execution; it prevents future agents from
  treating the replay blocker as an unknown gap. The companion
  `StackProgramOwnedTempLiveIdentityJoin.v0` row shows that broad
  `(phase, block, binding)` live descriptor matches are observable but
  overbroad/unstable. `StackProgramOwnedTempSlotIdentity.v0` now proves the
  planner can name stable program-owned temp plan slots with descriptor indices
  and shapes, but replay remains blocked until those slots have allocator-backed
  identity or exact live descriptor-slot joins;
- the first `vits_182` wide4 graph-catalog and timing runs: bridge sanity and
  `StackRegionSegmentPlan.v0` evidence passed, and a separate no-graph
  three-repeat timing pass measured about 74.8 ms mean device-resident forward;
- a benchmark-only `vits_140` compiled-session bridge canary was exposed through
  `--vulkan-stack-output-device-bridge-mode=compiled_session_bridge`. The default
  `stack_capture_decoder_preprocess` mode remains unchanged and measured about
  68.9 ms in a one-repeat sanity run; the compiled-session canary exited with Windows stack
  overflow `-1073741571` before writing a result JSON, so it is not a viable
  performance path until the replay/compiled-session stack-overflow failure is
  fixed;
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
  unsafe blocked by Windows stack overflow, but the native
  `StackOutputBridgeDeepSplitPlanRuntime.v0` runtime runs the 24-block bridge
  as two 12-block native chunks with a device-private baton. The DAv2 benchmark
  safe path now auto-selects that runtime for deeper-than-12-block
  stack-output bridge rows when no explicit deep-split env is set, the
  stack-capture bridge mode is used, and a bounded segmented stack-owned mode
  is active. No-env RX 9070 `vitl_140` and `vitl_182` smokes passed bridge
  sanity, kept `cpu_fallback=0` and `sync_readback=0`, and wrote valid
  artifacts. A post-commit `vitl_140` guardrail also passed on RX 9070,
  GTX 1080, and RX 6700 XT with `runtime_auto_selected=true` and zero timed
  fallback/readback/copies. This is benchmark control-plane default policy,
  not direct native op default admission and not evidence to widen the `vits`
  or `vitb` segment-plan rowsets. A post-`a343` recheck with deferred stack-exit
  diagnostic publication passed `vitl_140` repeats 1 and 2 under
  `native_private_baton` with no Windows stack overflow, `cpu_fallback=0`,
  `sync_readback=0`, and
  `diagnostic_payload_publish_mode=deferred_after_context_unlock`;
- a benchmark-local `python_private_baton` proof canary for that deep split
  topology is also unsafe blocked. The attempted run still hit native Windows
  stack overflow inside `run_vision_backbone_stack_private_capture_debug`
  before a result artifact could be written. The benchmark now records
  `PYTORCH_VULKAN_STACK_OUTPUT_BRIDGE_DEEP_SPLIT=python_private_baton` as
  unsafe-blocked metadata in the deep split plan and fails closed before
  native bridge execution;
- stack-output bridge depth guard: accepted benchmark control-plane fix. It
  keeps direct native bridge requests with `block_count > 12` fail-closed
  unless the native private-baton deep-split runtime is explicitly requested,
  while the DAv2 benchmark safe path can auto-select that runtime under the
  segmented stack-owned bridge predicates described above. The runtime contract is
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
- benchmark CPU timeline phase summaries: accepted measurement infrastructure.
  When `PYTORCH_VULKAN_CPU_TIMELINE_SUMMARY_LOG` is set, the Depth Anything V2
  benchmark emits a summary dump at the begin and end of each selected
  measurement phase. The begin dump clears setup/warmup rows; the end dump is
  the useful timed-phase CPU submission/copy attribution. Stack-owned external
  recording rows are included with `external_recording=1`, so segmented
  stack-recording CPU cost is no longer hidden from the summary. A focused RX
  9070 `vits_140` wide4 five-repeat run with this instrumentation measured
  about 68.4 ms mean / 68.1 ms median / 72.2 ms p95 device-resident forward,
  with zero timed CPU fallback, sync readback, or buffer copies. The top
  external-recording rows account for about 20 ms/request of CPU
  recording/binding work, making descriptor/recording reuse the next measured
  control-plane bottleneck. This does not change model execution,
  synchronization, copy, fallback, or route selection;
- descriptor-update allocation flattening: accepted default infrastructure fix.
  `DescriptorSet` reserves its per-set binding list to the shader layout size,
  and `get_bind_handle()` uses an inline-capacity descriptor-write list for the
  common update case instead of allocating a fresh heap vector per dispatch.
  A focused RX 9070 `vits_140` wide4 repeat-30 run measured about 64.2 ms mean /
  64.1 ms median / 65.7 ms p95 device-resident forward versus the prior
  65.9 ms / 66.1 ms / 67.1 ms retire-deferral baseline, with bridge sanity
  max_abs `1.1846423149108887e-06`, zero timed sync readback, and unchanged
  submit/retire/stack-planned counters. This is generic descriptor-path cleanup,
  not a DAv2 route, shader, submit, copy, fallback, or readback change;
- stack descriptor dependency diagnostic gating: accepted default diagnostic
  overhead fix. The default timing path no longer builds live-descriptor,
  pre-dispatch proof-table, or barrier-canary descriptor rows unless graph rows,
  diagnostic rows, or a selected barrier canary request them. Focused stage
  attribution showed that the external-recording descriptor proof rows accounted
  for about 18 ms/request of CPU work after descriptor-update allocation
  flattening. A focused RX 9070 `vits_140` wide4 warmup-3/repeat-10 run measured
  about 57.6 ms mean / 57.8 ms median / 58.9 ms p95 device-resident forward,
  with bridge sanity max_abs `1.1846423149108887e-06`, zero timed CPU fallback,
  sync readback, and buffer copies. A separate graph smoke kept the
  pre-dispatch proof and submit-epoch rows visible when
  `PYTORCH_VULKAN_STACK_DEP_GRAPH` was set. This is not a stable repeat-30
  baseline by itself; the earlier repeat-20/30 Windows stack-overflow failure
  around the thirteenth repeated forward is now cataloged separately as a
  stream-sync pool-lifetime issue, not as a descriptor-diagnostic gating issue;
- broad stack-owner retire-drain deferral with a wider old-path pending-resource
  budget: rejected before commit. An explicit
  `PYTORCH_VULKAN_STACK_REGION_RETIRE_DRAIN_DEFER=stack_owner_wide_budget`
  experiment on RX 9070 `vits_140` reduced timed `retire_queue_drain` submits
  from 15/forward to 4/forward, but regressed device-resident latency to about
  82.3 ms mean / 79.5 ms median / 95.5 ms p95 over 10 repeats and grew pending
  retire pressure to about 4.35 GB observed over the measurement. Retesting
  after normal-submit cleanup callback batching still regressed to about
  83.6 ms mean / 87.2 ms median / 95.5 ms p95, so the issue is not just callback
  count. Keep the existing narrow default deferral budget; the next cleanup path
  needs segment-local completion ownership that avoids growing stack-close work,
  not broader old-path submit suppression;
- stack-region external recording pool lease:
  `PYTORCH_VULKAN_STACK_REGION_EXTERNAL_RECORDING_POOL_LEASE=per_stack` is an
  opt-in ownership experiment for stack-owned external recording. A focused
  `vits_140` wide4 RX 9070 probe remained correctness-clean, but the lease was
  slower than the retained persistent-pool default and did not fix the
  `compiled_session_bridge` Windows stack-overflow exit `-1073741571`. Keep it
  as ownership/lifetime evidence only; do not enable it on the hot path without
  new proof that it fixes a concrete lifetime bug or improves timing;
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
- exact `vits_140` fc2 float-buffer tiled linear canary:
  `PYTORCH_VULKAN_LINEAR_TILED_CANARY=vision_fc2_exact_151x1536x384`
  correctly routed only the `[151,1536] -> [151,384]` bias/no-post-op rows to
  `aten::linear.buffer_float_tiled_bias` after adding tiled shader output-tile
  metadata. Bridge sanity passed with `max_abs=1.6391277313232422e-06`, CPU
  fallback and sync readback stayed zero, but device-resident timing regressed
  to about 93.5 ms mean versus the 64.3 ms wide4 baseline. Keep the tile
  metadata/correctness guard as canary evidence; do not promote this tiled fc2
  route without a faster kernel or tuning policy;
- exact `vits_140` fc2 vec2 tiled linear canary:
  `PYTORCH_VULKAN_LINEAR_TILED_CANARY=vision_fc2_exact_151x1536x384_vec2`
  routes only the exact `[151,1536] x [384,1536] + [384]` bias/no-post-op FC2
  row to `aten::linear.buffer_float_tiled_bias_vec2`. The focused route test
  matches CPU within `atol=1e-3, rtol=1e-3`, and a warmup-3/repeat-30 RX 9070
  `vits_140` wide4 bridge run measured about 45.8 ms mean / 45.3 ms median /
  48.4 ms p95 device-resident forward with bridge sanity
  `max_abs=1.1846423149108887e-06`, CPU fallback zero, and sync readback zero.
  This remains historical accepted opt-in canary evidence. The runtime env gate
  is retired now that the exact row is covered by
  `VisionFc2ExactTiledVec2LinearPlanContract`; future FC2 changes should extend
  linear plan contracts rather than restoring the env canary;
- latest `vits_140` RX 9070 attribution after recovery-flush gating,
  retained-pool wide4 recording, and default descriptor-diagnostic gating:
  the repeated fixed-feature decoder/bridge stack overflow was traced to the
  public stream-sync cleanup boundary. `synchronize_stream()` waited the
  current stream but did not recycle the normal context command and descriptor
  pools, so repeated device-resident DPT decoder forwards over stable Vulkan
  features could exhaust/grow the default 1024-set descriptor pool around the
  fourteenth repeat. Current-stream synchronization now flushes the normal
  command and descriptor pools after a successful current-stream wait, matching
  `synchronize_device()` and fence-wait cleanup semantics without broad
  recovery flushes. A synthetic fixed-feature DPTHead loop now completes 32
  repeats, and the original warmup-0/repeat-30 `vits_140` device-resident
  benchmark completes at about 62.3 ms mean / 60.9 ms median / 72.1 ms p95
  with zero timed CPU fallback and sync readback. The phase GPU timestamp
  profile still shows about 43-47 ms of kernel work per forward. The top GPU
  rows were `fc2` (`mm_buffer_float_bias`, about
  13.9 ms/forward), decoder/other convs, `fc1_gelu`, `qkv_linear`,
  `proj_linear`, attention BMM, and LayerNorm. Sub-50 work therefore needs
  further control-plane reduction and a parity-proven FP32 linear plan; the
  existing tiled fc2 canary remains non-promoted evidence;
- normal-submit cleanup callback batching: accepted default infrastructure fix.
  `retire_deferred_cleanup()` now batches ordinary-submit pending buffers and
  images into one timeline-gated `RetiredResource` callback, while preserving
  per-resource retire accounting and the same submit/timeline ownership. A
  focused warmup-0/repeat-30 RX 9070 `vits_140` device-resident run measured
  about 58.4 ms mean / 57.6 ms median / 61.4 ms p95, with timed CPU fallback
  zero, sync readback zero, and the same submit-origin and retire-drain counts
  as the pre-batch run. Cleanup callbacks dropped from 38,745 to 2,491 over 30
  repeats. A no-skip-output three-repeat sanity completed with
  `performance_valid=true`, but the no-bridge path does not emit a
  model-vs-reference `max_abs` field;
- stack/control-plane flatness guard: accepted diagnostic contract. Stack and
  external recording begin/end paths now log rejected nested/underflow scopes,
  central submit/retire cleanup paths log only on reentry, and replay
  record/warmup callbacks fail closed on nested callbacks. A bounded wide4
  `vits_140` bridge smoke with sync logging emitted no depth-guard rows,
  passed bridge sanity at `max_abs=1.1846423149108887e-06`, and measured about
  57.1 ms mean / 57.7 ms median / 58.9 ms p95 over three device-resident
  repeats. A bounded segmented `compiled_session_bridge` three-repeat probe
  also avoided process stack overflow and passed sanity at
  `max_abs=8.787959814071655e-06`, but remained slower at about 101.3 ms mean,
  so it remains diagnostic evidence rather than a promoted performance path;
- graph-dump reentry deferral: accepted diagnostic flatness guard. Requested
  `PYTORCH_VULKAN_STACK_DEP_GRAPH` dumps now skip opportunistic full graph
  serialization while central submit, pending-retire drain, retire-cleanup, or
  external-recording cleanup control-plane scopes are active. The skipped write
  is visible as `StackRegionGraphDumpSkip.v0` with
  `stack_region_graph_dump_skipped_reentrant_submit_or_cleanup`, and recursive
  graph serialization is counted separately. Focused graph-dump tests still
  pass, and a bounded RX 9070 `vits_140` wide4 smoke after the guard measured
  about 55.0 ms mean / 55.0 ms median / 55.6 ms p95 with bridge sanity
  `max_abs=1.1846423149108887e-06`, CPU fallback zero, and sync readback zero.
  This is not a route, submit, retire, shader, copy, readback, or fallback
  change;
- stack-region exit control-plane work batch: accepted reporting
  infrastructure. `end_stack_planned_recording_and_submit()` now prepares a
  heap-owned `StackRegionControlPlaneWorkBatch.v0` after closing/submitting the
  stack region and drains its typed ordered action list immediately under the
  same lock and in the same order as the previous inline code. Snapshot rows
  record `stage=prepared` and `stage=drained_inline` with
  `drain_mode=prepared_not_drained` before the drain,
  `drain_mode=iterative_inline` after the drain, `drain_action_count=6`,
  matching `drained_action_count=6` only on the drained row,
  `executor_mode=not_started` on the prepared row,
  `executor_mode=context_control_plane_inline` on the drained row,
  `executor_depth_before=0`, `executor_depth=1`, `executor_depth_after=0`,
  `executor_reentry_status=not_reentrant`, `executor_reentry_rejected=0`,
  `executor_depth_guard=raii`,
  `diagnostic_payload_publish_mode=deferred_after_context_unlock`,
  `before_handoff_retained_state_payload_captured`, and
  `after_finalize_retained_state_payload_captured`,
  `retained_state_live_log_reread_count=0`,
  `retained_state_deferred_payload_count`,
  `submit_topology_preserved=1`,
  `phase_boundary_submits_preserved=1`, `submit_elision_enabled=0`, and
  `deferred_submit_enabled=0`. `StackRegionDependencyGraph.v0` exposes those
  rows in `stack_region_control_plane_work_batch_rows` for both full and
  summary-only graph dumps. The rows also carry `source_snapshot_state`,
  stack-internal-temp batch count/bytes, and stack-region handoff batch
  count/bytes for the next cleanup-pressure decision. Focused graph tests
  assert both rows are visible; a short RX 9070 `vits_140` wide4 smoke after
  the scaffold measured about
  56.9 ms mean / 56.9 ms median / 57.2 ms p95, with bridge sanity
  `max_abs=1.1846423149108887e-06`, CPU fallback zero, and sync readback zero.
  This is a stack-overflow flatness scaffold only: cleanup is not deferred,
  submit topology is unchanged, and no performance claim is attached;
- post-guard RX 9070 `vits_140` timestamp attribution:
  `agent_space/dav2_vits140_post_guard_gpu_timestamp_summary.md` records about
  44.2 ms of timestamped GPU work per forward under deliberately intrusive
  timestamp logging. The instrumented wall time is not a baseline, but the
  kernel split is useful: `fc2 | mm_buffer_float_bias` is about 15.2 ms/forward,
  total `mm_buffer_float_bias` is about 19.6 ms/forward, decoder/other convs
  are about 8.9 ms/forward, and qkv/proj/fc1/attention/LayerNorm make up most
  of the rest. The existing exact tiled FC2 canary remains rejected as slower,
  so the next GPU-side plan must be a new parity-proven FP32 linear candidate
  rather than promotion of the old tiled route;
- decoder-tail ReLU via conv clamp: correct but slower;
- fused Depth Anything V2 head shader path: correctness blocked;
- compiled-session bridge/replay shortcut: unsafe blocked;
- current-topology single-boundary submit elision: unsafe blocked.

These entries are intentionally generic where possible: the model is evidence
provenance, while the contract/topology scopes name reusable backend behavior.
