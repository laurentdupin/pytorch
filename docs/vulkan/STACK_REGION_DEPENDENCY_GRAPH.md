# Stack Region Dependency Graph

## Purpose

`StackRegionDependencyGraph` is the planned RegionContract surface for Vulkan
stack-owner execution. Its job is to describe dispatch-level producer-consumer
ordering before command recording, so future submit reduction can replace
phase-boundary host/queue synchronization with explicit device-side
dependencies when that is correct.

The graph is not a kernel scheduler, not a shader-tuning mechanism, and not a
model-name route. It must not broaden shape admission, hide CPU fallback or
readback, weaken public-output safety, or turn requested Python-visible
captures into private tensors without an explicit same-region consumer
contract.

## Why Retire-Time Proof Was Insufficient

Prior DAv2 `vits_140` experiments showed that lifetime and retire provenance
can identify many stack-internal resources, but removing
`maybe_synchronize_after_norm()` submits from retire-time evidence alone is
unsafe for capture-sensitive stack outputs. The failing paths proved that:

- A downstream device consumer registration proves that a consumer exists, but
  not that the producer value is preserved when a phase-boundary submit is
  removed.
- Capture-owned snapshots attempted after the wrong stack-plan point do not
  replace the ordering semantics that the phase-boundary submit currently
  provides.
- Retire-time barriers are too late. The dependency must be known before the
  consumer dispatch is recorded, with the live descriptor binding and buffer
  range available.

Therefore the next optimization cannot be another broad
`stack_owner_phase_boundary` coalescing rule. It needs a graph of dispatch
edges and boundary requirements built while recording the stack region.

## Graph Model

A stack-region graph has four node classes:

- **Dispatch nodes**: one node per recorded stack dispatch, with dispatch id,
  command-buffer sequence, op label, stack block, stack substep, shader label,
  and descriptor bindings.
- **Resource nodes**: Vulkan allocation id, generation, byte range, storage
  kind, layout, dtype, logical shape, provenance role, capture/final-output
  flags, and host-visible/readback flags.
- **Boundary nodes**: phase-boundary events such as native-layernorm
  boundaries, stack planned-recording boundaries, capture points, and final
  stack exits.
- **Capture/consumer nodes**: requested stack outputs, private bridge captures,
  decoder/head consumers, and any Python/public boundary.

Edges are explicit:

- **Producer edge**: dispatch writes a resource range.
- **Consumer edge**: dispatch reads or writes a resource range through a known
  descriptor binding.
- **Boundary requirement edge**: a boundary requires an ordered write-to-read
  or write-to-write transition before a later consumer.
- **Capture edge**: a stack output becomes public, private, or bridge-owned.
- **Escape edge**: host-visible, public Python, final-output, or explicit
  readback access.

## Planned Dump Fields

The graph dump should be deterministic and compact enough for benchmark
artifacts:

- region id, stack context id, stack plan id, bridge/session id if present
- dispatch id, command-buffer id or sequence, op label, stack block/substep
- producer and consumer descriptor binding index
- dependency kind: write-to-read or write-to-write
- resource allocation id, generation, byte offset, byte range, bytes
- logical shape, dtype, layout/direct-buffer/storage flags
- provenance role, stack lifetime class, last-use status
- capture flags: requested intermediate, private bridge capture, final output
- escape flags: public Python boundary, host-visible access, readback request
- boundary id, phase, block, required edge count, covered edge count
- budget fields: per-boundary bytes and per-region live-byte estimate
- accept/reject reason for each boundary and each graph edge

The initial behavior-neutral implementation is enabled with
`PYTORCH_VULKAN_STACK_DEP_GRAPH=<path>`. It writes one JSON object to the path
when a stack-owner recording scope ends. The v0 schema is
`StackRegionDependencyGraph.v0` and contains:

- `summary` counters for dispatch nodes, dependency edges, resource nodes,
  allocation nodes, boundary nodes, capture edges, fully proven edge records,
  and queue-submit boundary records
- `dispatch_nodes`, `dependency_edges`, `capture_edges`, `resource_nodes`,
  `pre_dispatch_insertion_point_nodes`, `allocation_nodes`,
  `phase_boundary_nodes`, and `region_lifetime_rows`
- per-row `fields` maps parsed from the existing stack dispatch, lifetime,
  and region attribution diagnostics
- explicit `missing_metadata_fields` on dependency edges
- `unproven_or_missing_metadata_fields` for graph-level gaps such as region id,
  stack context id, bridge/session id, complete boundary dependency sets, and
  capture consumer dispatch positions
- a nested `barrier_plan` object with schema
  `StackRegionBarrierPlan.v0`. This is a dry-run plan derived from dependency
  edges. Each record names producer and consumer dispatch node ids, allocation
  id/generation/range, dependency kind, source and destination stage/access,
  descriptor binding, planned barrier location, whether the edge could cover a
  phase-boundary ordering requirement, and the precise rejection reason when
  the edge is not plannable. Planned non-capture next-block Norm1 consumers
  are recorded separately from observed consumer dispatches; they reduce the
  consumer-identity proof gap but still reject as missing consumer dispatch
  position until command recording can provide an exact insertion point. The
  dry-run now also records the stack-plan logical position for planned
  consumers before command recording starts, and records the completed-run
  command dispatch position when the graph later observes it.
  `pre_dispatch_insertion_point_nodes` are recorded by the command-recording
  path immediately before a stack dispatch is registered. They provide a
  stable dry-run token of the form `stack_scope:*:before_phase:*:block:*` that
  a future barrier hook can match before recording the consumer dispatch. The
  token does not insert a barrier or remove a submit by itself; it only proves
  that an insertion location is observable before the dispatch is recorded.
  `live_vulkan_buffer_binding_nodes` are also recorded by the
  command-recording path for Vulkan buffer descriptor arguments. They carry the
  stack scope, phase, block, shader label, descriptor binding, allocation
  id/generation/range, allocation label, and opaque live `VkBuffer` and
  wrapper-object tokens. BarrierPlan v0 joins these rows back to dependency
  records by exact scope/consumer/binding/allocation/range. Exact matches are
  reported as `live_buffer_bound`; allocation-only matches with a different
  range stay rejected as `binding_range_mismatch`; missing rows remain
  `missing_live_vulkan_buffer_binding`.
  The plan also records the boundary-level metadata still missing before any
  submit can be removed. `behavior_change_allowed=false` is a hard veto for
  every v0 record: env opt-in must not override it. A future barrier or submit
  canary also needs proof-to-live-`VulkanBuffer` binding and validated
  visibility dependency fields, or an explicit no-visibility-dependency proof.
  Dry-run stage/access and insertion-point strings alone are not executable
  barrier proof.
- a nested `capture_output_boundary_contract` object with schema
  `CaptureOutputBoundaryContract.v0`. This is a capture-specific dry-run proof
  surface for requested intermediate capture edges such as
  `residual2 -> intermediate_capture`. It records producer block/substep/role,
  capture block/index/output role, allocation id/generation/range, requested
  intermediate/public/private capture observations, same-region downstream
  device-consumer registration when a generic bridge publishes it, public or
  host-visible boundary status, and why the phase-boundary submit remains
  required. Public `Tensor[]` captures stay unsafe. The contract emits a
  combined view plus public-only and bridge-private-only scope records so a
  bridge run that also evaluates a public reference path can keep public
  rejection visible without contaminating bridge-private proof diagnostics.
  `mixed_scope_rejected_records` counts combined records that observed both
  scopes. `bridge_private_proof_complete_records` means the capture-specific
  bridge scope has same-region registration and allocation/range proof; it does
  not make a boundary complete until the full boundary dependency set is
  complete.
- `stack_output_device_consumer_registrations` rows record generic bridge
  diagnostics for captured stack outputs. The v0 key is stack context/session,
  capture block/substep/output role, output layout, strip/view relation,
  downstream consumer context/id/input slot, expected consumer shape/layout,
  and booleans for same planned region, Python public boundary, host-visible
  boundary, host-visible access, and host readback before consumption. These
  rows only feed graph proof diagnostics; they do not authorize sync removal.
- a nested `boundary_complete_dependency_proof` object with schema
  `BoundaryCompleteDependencyProof.v0`. The v0 proof is intentionally narrow:
  it only groups non-capture `residual2 -> norm1` stack boundaries and records
  required edge records, BarrierPlan-covered edge records, retire-only resource
  classes, ordering-required resource classes, public/host/final/requested
  blockers, phase-boundary lifetime rows, missing fields, and complete/not
  complete status. It also records planned non-capture next-block Norm1
  consumer metadata separately from recorded consumer dispatch metadata, so the
  proof can show when the `consumer_dispatch` gap shrinks without making the
  edge eligible for a BarrierPlan. The same boundary proof records planned
  non-capture `residual2 -> norm1` formal last-use metadata separately from
  runtime stack-lifetime proof metadata. It is proof-only and does not
  authorize a submit skip.
- a nested `capture_boundary_dependency_set` object with schema
  `CaptureBoundaryDependencySet.v0`. This joins capture-specific
  `CaptureOutputBoundaryContract` records with matching phase-boundary rows.
  It reports combined/public/bridge-private complete-boundary counts
  separately. Public `Tensor[]` captures remain incomplete. Bridge-private
  capture boundaries can become complete at capture-scope when every capture
  edge has same-region consumer and allocation/range proof, but
  `full_boundary_complete_boundaries` stays separate and remains zero until
  all non-capture ordering resources and boundary blockers are proven.
- a nested `stack_activation_capture_proof` object with schema
  `StackActivationCaptureProof.v0`. This is a bridge-private-only proof surface
  for capture-sensitive residual2 activations at requested intermediate capture
  boundaries. It records stack owner scope, producer block/substep/role,
  allocation id/generation/range, same-region bridge registration,
  capture-boundary dependency-set membership, direct-buffer storage, and
  alias/runtime-alias flags. Public `Tensor[]` capture scope stays rejected.
  A proof-complete activation record only moves the root blocker from
  `missing_stack_activation_proof` or `capture_sensitive_stack_activation` to
  the next explicit boundary blocker; it does not insert a barrier or remove a
  submit.
- a nested `phase_boundary_budget_recompute` object with schema
  `PhaseBoundaryBudgetRecompute.v0`. This recomputes bridge-private capture
  boundary classes after `CaptureBoundaryDependencySet` and
  `StackActivationCaptureProof` are applied. It reports pending bytes before
  and after proof classification, ordering-required bytes, retire-only bytes,
  proof-classified capture-activation bytes, public/host/final/requested
  blockers, block and scope budget status, and complete or incomplete reason
  for each bridge-private boundary. Public combined-scope capture remains
  rejected. A recomputed complete boundary is dry-run completeness evidence
  only; it is not canary-ready while `behavior_change_allowed=false` or while
  live visibility proof is missing. The object is dry-run and does not insert
  barriers or remove submits.
- a nested `stack_region_boundary_submit_plan` object with schema
  `StackRegionBoundarySubmitPlan.v0`. This is the online submit-site hook for
  future one-boundary canaries. It records each live stack-owner phase-boundary
  submit with the live boundary id, the selected bridge-private boundary id,
  the selected proof id/version, same-region consumer registration status,
  public-scope rejection status, queue-submit status, and fail-closed online
  plan status such as `not_planned`,
  `planned_live_boundary_match_proof_pending`, `rejected_boundary_mismatch`,
  or `rejected_public_scope_or_no_same_region_consumer`. It is diagnostic
  only: `barriers_inserted` and `submits_removed` remain zero until a later
  behavior CL consumes a current-run proof and explicitly enables one selected
  boundary. A current-run proof match is still rejected when the proof producer
  reports `behavior_change_allowed=false`; the submit-plan records expose this
  as `rejected_behavior_change_not_allowed`.
- a nested `stack_region_barrier_only_canary` object with schema
  `StackRegionBarrierOnlyCanary.v0`. This is an opt-in command-recording
  barrier canary for exactly one non-capture `residual2 -> norm1` boundary,
  selected with
  `PYTORCH_VULKAN_STACK_REGION_BARRIER_CANARY=non_capture_residual2_norm1_block1`
  or `producer_block_0_consumer_block_1`. The hook runs at the consumer
  descriptor-recording site, where the live `VulkanBuffer`, descriptor binding,
  stage/access labels, and pre-dispatch insertion point are visible. When the
  current-run `StackRegionPreDispatchProofTable.v0` row is complete, it records
  one compute-shader write-to-read buffer barrier before the consumer dispatch
  and reports `barriers_inserted > 0`. It keeps the existing phase-boundary
  submit intact: `submits_removed` stays zero. If the current-run proof is
  missing, it fails closed with
  `missing_current_run_proof_match_at_consumer_recording`. This flag cannot
  override `behavior_change_allowed=false` for submit elision.
- a nested `stack_region_pre_dispatch_proof_table` object with schema
  `StackRegionPreDispatchProofTable.v0`. This is the narrow online proof table
  consumed by the barrier-only canary at the consumer descriptor-recording
  point. The first row class covers only the generic non-capture
  `residual2@0 -> norm1@1` boundary: it requires the live buffer binding,
  producer dispatch observation, planned next-block Norm1 consumer position,
  pre-dispatch insertion token, exact allocation id/generation/range, and no
  capture between producer and consumer. A complete table row lets the canary
  insert the barrier-only canary before command recording reaches the consumer
  dispatch. It still does not remove submits; submit elision remains a separate
  canary.
- a nested `stack_region_boundary_optimization_plan` object with schema
  `StackRegionBoundaryOptimizationPlan.v0`. This is a generic eligibility table
  derived from pre-dispatch proof records and any current-run barrier canary
  records. It classifies each boundary record by boundary class, scope,
  live-buffer binding, allocation/generation/range match, stage/access
  availability, insertion point availability, barrier-only validation status,
  `behavior_change_allowed`, and submit-elision eligibility. Public, capture,
  final, host-visible, and readback boundaries must remain fail-closed. The
  table is data-driven over proof predicates rather than model names or a
  single benchmark route. A future submit-elision canary must use a separate
  `PYTORCH_VULKAN_STACK_REGION_SUBMIT_ELISION_CANARY` opt-in and only after the
  same current run has inserted the real barrier records for the selected
  non-capture boundary. Without that future opt-in, eligible records report
  `eligible_requires_submit_elision_opt_in` and `submits_removed=0`.
- a nested `stack_boundary_proof_records` object with schema
  `StackBoundaryProofRecord.v0`. This is the consolidated per-boundary proof
  row surface for stack carry and submit-equivalence readiness. Each row records
  the boundary id/class, producer and consumer roles/blocks/substeps, produced
  range, actual consumer input range, old carry range, descriptor observation,
  formal last-use and non-escape proof status, public/host/final/readback/alias
  blocker status, actual input and old-carry barrier status,
  `behavior_change_allowed`, submit-equivalence candidate status, and reject
  reason. Legacy `stack_carry_visibility_*` histograms may still be emitted for
  continuity, but readiness summaries such as barrier-ready records,
  submit-elision-ready records, top reject reason, and highest-leverage missing
  proof field are generated from the typed rows.
- a nested `submit_level_equivalence_proof` object inside
  `stack_boundary_proof_records`, with schema
  `StackBoundarySubmitLevelEquivalenceProof.v0`. This is the submit-site
  aggregate that must stay complete before any typed row can be considered
  submit-elision-ready. It records the current-run topology signature,
  pending dispatch/resource/write-set counts and bytes, descriptor/update and
  retire side-effect counts, real-barrier-to-pending-allocation matches, and
  the booleans `removed_submit_pending_dispatch_set_complete`,
  `removed_submit_has_no_unmodeled_execution_side_effects`, and
  `all_pending_writes_covered_by_barrier_or_nonescaping`. Any false boolean,
  incomplete candidate status, or topology/cardinality mismatch forces the
  typed boundary rows to fail closed with a non-`none` reject reason. Submit
  rows are keyed by a generic live submit key, not only by boundary id: boundary
  id/class, producer and consumer blocks, stack phase, descriptor binding,
  callsite, submit phase, command-buffer ids, and submit epochs. A boundary
  rollup remains fail-closed unless exactly one live submit key maps to that
  boundary in the current run.
- a nested `boundary_submit_equivalence_proof` object inside
  `stack_boundary_proof_records`, with schema
  `StackBoundarySubmitEquivalenceProof.v0`. This rolls the typed rows up to the
  selected boundary scope. It reports the selected boundary id/class, total
  rows, submit-ready rows, non-ready rows, outside-boundary rows, barrier-covered
  bytes, old-carry retire-only/nonescaping bytes, public/host/final/alias
  blockers, same-command-buffer or submit-epoch status, producer/barrier/
  consumer command-buffer ids, producer/consumer submit epochs, and a
  fail-closed reject reason. The command-buffer/epoch aggregate is joined from
  generic `StackRegionBoundaryOptimizationPlan.v0` rows for the same selected
  boundary id and must cover every selected submit-ready typed row before it is
  marked complete. Row-level submit readiness is not enough to skip a submit:
  boundary-level submit equivalence remains false unless this selected-boundary
  command-buffer/submit-epoch proof is complete for the submit being considered.

This dump is diagnostic by default. It does not skip submits, change routes, or
change accepted shapes. A real barrier is recorded only under the explicit
barrier-only canary flag above.

## Barrier Plan Stage

The first behavior stage is barrier planning only. It may insert a device-side
barrier immediately before a consumer dispatch only when the graph proves:

- exact producer dispatch and consumer dispatch identity
- exact allocation id, generation, and byte range
- required access transition and descriptor binding
- no host-visible, public-output, final-output, or alias escape blocker
- all capture consumers for the edge are either public and synchronized, or
  private in the same planned region
- the barrier is emitted before command recording reaches the consumer

Barrier insertion without a matching submit-reduction decision is not a
performance win and should remain diagnostic-only.

Before any barrier canary, the plan must bind the proof allocation id,
generation, and byte range to the live `VulkanBuffer` or equivalent resource
object used by the command recorder. If the graph can only report allocation
ids from completed dry-run rows, the correct status is
`missing_live_vulkan_buffer_binding`; do not treat such records as executable
barrier records.

## Submit Plan Stage

A phase-boundary submit may be skipped only when the boundary node is complete:

- every required producer-consumer ordering edge is covered by a planned
  device-side dependency
- unrelated pending retire resources are proven retire-only and do not require
  the boundary ordering
- public, host-visible, final-output, debug, and explicit readback blockers are
  absent or handled by a separate contract
- byte budgets are satisfied for the boundary and for the enclosing region
- the proof producer sets `behavior_change_allowed=true`
- either real barrier records were inserted and validated at the consumer
  dispatch, or the boundary has an explicit no-visibility-dependency proof
- capture parity and bridge output sanity pass for the guarded path

The decision must emit counters for complete boundaries, skipped submits,
inserted barriers, rejected boundaries, rejected edges, and budget failures.

The first submit-elision canary is intentionally narrower than the planning
table: it may remove at most one non-capture boundary submit selected from
`StackRegionBoundaryOptimizationPlan.v0`, and only after the live submit site
matches the same command-buffer scope and proof id that recorded the real
barrier. Capture, public, final, host-visible, and readback scopes stay
ineligible regardless of environment flags.

## Validation Gates

Any future behavior-changing CL using this graph must include:

- focused graph construction tests with positive and negative boundary cases
- private/public capture parity when captures are involved
- bridge output sanity for bridge-active paths
- targeted `vits_140` validation only for the changed path
- CPU fallback, sync readback, unexpected readback, and buffer-copy counters
- submit-origin and retire-drain deltas showing actual submit reduction
- `git diff --check`

If correctness fails, if no submit is skipped, or if fallback/readback/copy
counters regress, the behavior change must be reverted. The graph dump may
remain only when it is behavior-neutral and useful for the next proof step.
