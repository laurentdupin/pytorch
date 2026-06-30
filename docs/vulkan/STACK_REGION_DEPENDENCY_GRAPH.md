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
  override `behavior_change_allowed=false` for submit elision. The emitted
  barrier row is typed as
  `barrier_target_role=actual_norm1_input_visibility` and carries the
  `actual_consumer_visibility_transition_*` fields when the canary records the
  real barrier for the live Norm1 activation-input descriptor range.
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
  proof field are generated from the typed rows. Rows also carry the source
  `stack_region_instance_id` from the live submit record that emitted their raw
  provenance. Submit-level proof joins prefer the matching
  `boundary_id + stack_region_instance_id` aggregate and only fall back to the
  boundary-wide aggregate as an explicit fail-closed diagnostic.
  The CUDA and DirectML backend probes tightened the row contract: each typed
  boundary row now also exposes descriptor or binding-table identity, descriptor
  update-generation status, command-buffer/submit visibility identity,
  allocator pool or region identity status, transition node provenance, and an
  alias/public/final/host/readback escape class. Missing values in these fields
  are explicit fail-closed proof gaps. Logical descriptor identity is still
  derived from live descriptor argument order, while actual descriptor update
  tokens are emitted separately as `StackDescriptorSetUpdateGeneration.v0` rows
  from `DescriptorSet::get_bind_handle()` and joined into submit-level proof
  accounting. These fields do not authorize submit elision.
  Actual Norm1 input visibility is reported as a typed transition-provenance
  join rather than a bare boolean. Rows include
  `actual_consumer_visibility_transition_status`,
  `actual_consumer_visibility_transition_source`,
  `actual_consumer_visibility_transition_contract`,
  `actual_consumer_visibility_producer_role`,
  `actual_consumer_visibility_consumer_role`, and
  `actual_consumer_visibility_resource_digest`. If an existing real
  barrier-only canary record covers the exact actual-consumer input range, the
  row reports the joined source with
  `actual_consumer_visibility_transition_joined_from_real_barrier`. Otherwise
  the row remains fail-closed with a concrete missing source field.
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
  id/class, producer and consumer blocks, stack phase, stack-region instance id,
  descriptor binding, callsite, submit phase, command-buffer ids, and submit
  epochs. This separates repeated stack/forward instances in the same run. A
  boundary-wide rollup remains fail-closed unless exactly one live submit key
  maps to that boundary. Instance-specific joins are available to typed rows,
  but they remain behavior-neutral and still require the submit-level side
  effect booleans to be complete.
  Submit-level rows also classify per-instance side effects without changing
  execution: retire entries are split into proven
  retire-only/nonescaping entries and unknown-or-ordering-required entries, and
  the remaining capture-sensitive activation resource reports whether it has a
  typed boundary relation. Unknown or unjoined side effects remain hard blockers
  for submit equivalence.
  Descriptor bookkeeping is now split the same way: submit-level rows distinguish
  missing actual descriptor update evidence, observed actual descriptor updates
  whose submit equivalence remains unproven, and pending-dispatch visibility
  states. Observed descriptor updates are not sufficient to skip a submit unless
  the pending dispatch set and command-buffer/submit-batch visibility are also
  proven complete.
  Pending dispatch bookkeeping now carries a behavior-neutral command-list
  identity: stack-region instance, command-buffer recording id, submit epochs,
  and the recorded dispatch-position range present at the phase-boundary submit
  site. `pending_dispatch_list_status`,
  `pending_dispatch_range_completeness_status`,
  `pending_dispatch_completion_visibility_status`,
  `pending_dispatch_command_buffer_identity_status`,
  `pending_dispatch_submit_epoch_transition_status`,
  `pending_dispatch_command_buffer_epoch_relation`, and
  `command_buffer_submit_epoch_visibility_proof_status` separate missing lists,
  contiguous range/set completeness, active command-buffer identity, submit
  epoch transition, command-buffer epoch visibility, and fully proven
  same-buffer/batch cases. The companion
  `command_buffer_submit_epoch_visibility_missing_source` field names the
  remaining source field or contract. Current rows are expected to stay
  fail-closed unless both the range completeness proof and command-buffer epoch
  visibility proof are complete.
  `PhaseSubmitEpochVisibilityContract` is the behavior-neutral contract
  skeleton for the phase-submit visibility gate. Submit-level rows expose
  `phase_submit_epoch_visibility_contract`,
  `phase_submit_epoch_visibility_contract_requirement_status`,
  `phase_submit_epoch_visibility_contract_status`,
  `phase_submit_epoch_visibility_contract_reason`,
  `phase_submit_epoch_visibility_contract_required_fields`,
  `phase_submit_epoch_visibility_contract_predicate_status`,
  `phase_submit_epoch_visibility_contract_failed_predicate`,
  `phase_submit_epoch_visibility_contract_predicate_details`,
  `phase_submit_epoch_visibility_contract_proof_ready`,
  `phase_submit_epoch_visibility_contract_behavior_enabled`, and
  `phase_submit_epoch_visibility_contract_submits_removed`. The status taxonomy
  distinguishes `no_phase_submit_epoch_crossing_observed`,
  `phase_submit_epoch_crossing_contract_required`,
  `phase_submit_epoch_crossing_contract_missing_unimplemented`,
  `phase_submit_epoch_visibility_contract_rejected_missing_command_buffer_identity`,
  `phase_submit_epoch_visibility_contract_rejected_predicate_failed`, and
  `phase_submit_epoch_visibility_contract_proof_only_accepted`.
  The proof-only accepted state requires the same active command-buffer
  recording scope, complete pending dispatch range and side-effect rows,
  actual descriptor update-generation evidence, a matched actual Norm1 input
  barrier-only canary record, old-carry retire-only/non-escape proof, zero
  unknown/order-required retire entries, no public/final/host/readback blocker,
  and preserved submits. This state is still diagnostics only:
  `phase_submit_epoch_visibility_contract_behavior_enabled=0`,
  `phase_submit_epoch_visibility_contract_submits_removed=0`, and submit
  elision remains disabled until a separate behavior slice explicitly consumes
  the proof.
  Submit-level rows also report the backend-planning fields that CUDA and
  DirectML make explicit: compiled-session identity, descriptor/binding-table
  identity, descriptor update generation, command visibility status, resource
  state transition status, allocator scope status, transition node provenance,
  and public/alias escape policy. Current Vulkan rows intentionally keep
  allocator-region identity visible rather than treating allocation generation
  alone as a replay-safe proof. Descriptor identity is present as a logical
  submit/binding key until a lower-level descriptor-set update token is
  available.
  Submit-level rows also join
  `barrier_target_role=actual_norm1_input_visibility` records from
  `StackRegionBarrierOnlyCanary.v0` by selected boundary id and
  `stack_region_instance_id`. The join reports
  `actual_consumer_barrier_record_status`,
  `actual_consumer_barrier_records`,
  `actual_consumer_matched_barrier_records`,
  `actual_consumer_covered_by_barrier_count`, plus the joined barrier source
  and resource digest. A matching actual-consumer barrier can make the
  submit-level barrier counters nonzero, but it does not make submit
  equivalence complete while pending dispatch/resource side effects remain
  unproven.
  The same submit-level proof now joins pending old-carry
  `capture_sensitive_stack_activation` resources back to typed
  `StackBoundaryProofRecord.v0` rows using the selected boundary id,
  `stack_region_instance_id`, and exact allocation/generation/range digest.
  Fields such as `old_carry_submit_proof_status`,
  `old_carry_typed_proof_records`, `old_carry_matched_proof_records`,
  `old_carry_retire_only_proven_records`,
  `old_carry_retire_only_proven_bytes`, `old_carry_unsafe_records`, and
  `old_carry_submit_proof_source` distinguish missing typed proof, range
  mismatch, a retire-only/nonescaping match, and matched-but-unsafe proof.
  Submit-level rows also expose the raw pending capture-sensitive edge and the
  subset joined to old-carry proof through
  `capture_sensitive_submit_pending_records`,
  `capture_sensitive_submit_pending_bytes`,
  `capture_sensitive_submit_pending_old_carry_joined_records`,
  `capture_sensitive_submit_pending_old_carry_joined_bytes`,
  `capture_sensitive_submit_pending_join_status`, and
  `capture_sensitive_submit_pending_join_reject_reason`. These fields answer
  whether the pending submit-site `capture_sensitive_stack_activation` resource
  is the same old carry proven elsewhere, but they do not relax the submit
  guard or authorize elision.
  The submit-site join reads `raw_buffer_provenance_signature` from the live
  boundary submit row and only falls back to the older raw-provenance field name
  for compatibility; missing raw provenance reports
  `missing_raw_buffer_provenance_signature`.
  A proven match reclassifies that pending resource out of
  `retire_entry_unknown_or_ordering_required`, but submit elision still requires
  the complete submit-level side-effect proof.
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

The first attempt to consume the proof-only
`PhaseSubmitEpochVisibilityContract` for the selected `residual2@0 -> norm1@1`
boundary removed one selected submit and failed bridge output sanity. That
behavior was backed out. The graph may still emit
`phase_contract_guard_proof_ready` rows under the opt-in submit-elision flag,
but the current source preserves the submit and records `submits_removed=0`.
Future behavior work must add a stronger value-preservation proof for removing
the phase submit, not just the predicate-ready visibility contract.
`StackRegionLiveSubmitEquivalenceBinding.v0` records the live-hook gap: whether
the submit-elision decision site has command-buffer ids, submit epochs, pending
dispatch range, and side-effect completion booleans before the submit executes.
Current rows can observe the live command-buffer recording-scope id and submit
epoch before/after values at the hook. They also bind the live pending-dispatch
range using the same `scope:...:command_buffer:...:positions:first-last`
identity convention as the graph proof rows. The live side-effect completion
status is now bound from the selected-boundary submit-level predicates:
descriptor update-generation evidence, matched actual Norm1 input barrier-only
record, old-carry retire-only/non-escape proof, zero unknown/order-required
retire entries, and absence of public/final/host/readback blockers. This binding
is still behavior-neutral. Rows keep
`phase_submit_epoch_visibility_contract_authorizes_submit_elision=0` and
`submits_removed=0`.
Because those rows are emitted before the final submit-level proof table is
rolled up, they are not behavior proof. The graph therefore also emits
`StackRegionLiveSubmitEquivalenceBindingExactJoin.v0` rows from the finalized
`StackBoundarySubmitLevelEquivalenceProof.v0` table. The exact join key uses
the selected boundary id/class, producer and consumer blocks, stack-region
instance id, descriptor binding, callsite/phase, producer and consumer
command-buffer ids, submit epochs, and the pending-dispatch range identity.
Join statuses distinguish an exact complete proof row, an exact incomplete
proof row with its reject reason, no proof row for the live submit key, and
ambiguous multiple proof rows. The original cumulative live counters remain
available for diagnosing emission order only and are marked as not
behavior-authoritative.
`StackRegionPendingDispatchCompletionEquivalenceProof.v0` is emitted from the
same exact submit-level row. It records whether the pending dispatch range has
an exact command-list proof, whether command recording is complete, whether
execution visibility is still submit-dependent, and whether any modeled side
effects remain beyond phase-submit execution/flush semantics. The current
status remains fail-closed: this proof surface can narrow the blocker, but it
does not authorize submit elision or change barrier behavior.
`PhaseSubmitExecutionFlushContract` is the named follow-up contract for that
remaining blocker. It records whether a phase-submit execution/flush dependency
is observed, whether the contract is required, the required primitive, the
candidate replacement primitive, and the missing source. Current selected
`residual2@0 -> norm1@1` rows can report that all modeled resource hazards are
covered while execution visibility still depends on the phase submit. In that
case the contract reports an absent replacement primitive, for example a
queue-submit-equivalent execution visibility primitive such as an event,
timeline, or other command-buffer-continuity proof. The field
`phase_submit_execution_flush_authorizes_submit_elision` remains `0`.
`PhaseSubmitCommandBufferContinuityProof.v0` is the first behavior-neutral
shape of that replacement proof. It records same-active-command-buffer scope,
whether the phase submit closes or submits the command buffer before the
consumer, whether a later real queue-submit/timeline candidate is observed for
the same command buffer or batch, whether the current phase submit supplies a
retire timeline, whether later-retire coverage is proven, and whether
intervening public, host, final, readback, stream/queue, fence, command-pool, or
descriptor-pool blockers are modeled. Current selected rows remain fail-closed
when command-buffer continuity is present but no later queue-submit/timeline
candidate is observed. The field
`phase_submit_command_buffer_continuity_authorizes_submit_elision` remains `0`.
`StackRegionSubmitPoint.v0` exposes observed submit points as graph nodes. Each
row records the submit-point id/key, stack-region instance, phase/scope/callsite,
command-buffer recording identity, submit epoch before and after, submit-point
kind such as `phase_boundary_submit`, and deferred target status. Current rows
represent the existing phase-boundary real queue submits; they are not later
region targets and report `authorizes_submit_elision=0`.
`StackRegionPlannedSubmitPoint.v0` is the matching behavior-neutral planning
surface for synthetic future targets. The current row shape names a planned
`stack_exit` / `region_exit` submit point, stack-region owner, expected
same-owner/same-stream relation, command-buffer continuity requirement,
descriptor and command-pool lifetime requirements, retire timeline migration
requirement, and the missing real implementation hook. These rows are
`synthetic_planned_only`, do not create a queue submit, and also report
`authorizes_submit_elision=0`.
`StackRegionDeferredSubmitRuntimeHookPlan.v0` spells out that missing runtime
hook. It is keyed by the planned submit point and records whether a hook is
installed, whether it can close and submit a region-owned command buffer or
batch, and the required capabilities for cross-phase recording ownership,
retire timeline migration, descriptor lifetime extension, command-pool lifetime
extension, same stream/queue proof, and host/fence/public/readback blockers.
The current rows report `hook_installed=0`,
`can_close_submit_region_owned_command_buffer=0`, and
`missing_command_pool_lifetime_extension` as the first concrete missing
capability behind the unavailable runtime API skeleton.
`StackRegionCommandBufferOwnershipPlan.v0` decomposes that capability into the
ownership shape a future runtime hook would need. It records the stack-region
instance, current command-buffer recording id and scope, current owner scope,
whether a region-owned command buffer or batch exists, whether ownership
transfer is required, whether recording could remain open across phase
boundaries, whether the phase submit could be suppressed while preserving
recording, whether a planned region-exit close/submit and timeline/retire point
could be provided, descriptor and command-pool lifetime coverage, same
stream/queue guarantee status, and the stack-owner request hook status. Current
rows report that the Vulkan context owns command buffers per submit phase, no
region-owned command buffer or batch is present, and the stack owner has no
region command-buffer request hook.
`StackRegionCommandBufferRequest.v0` and
`StackRegionCommandBufferRequestResult.v0` are the behavior-neutral request
surface for that hook. The request row models a stack owner asking for a
`command_buffer_or_batch` with stack-region lifetime, same stream/queue,
descriptor lifetime, command-pool lifetime, retire timeline ownership,
fallback, and public/final/host/readback policy requirements. The row is now
backed by a minimal runtime API skeleton,
`StackRegionCommandBufferRequestRuntimeApi.v0`, which is callable by
diagnostics and always returns unavailable without allocating, switching, or
submitting command buffers. The result row currently reports the API present
but unavailable: `request_result_runtime_api_present_unavailable`.
`StackRegionOwnedCommandBufferContract.v0` is the behavior-neutral design
surface behind that unavailable result. It records whether the contract is
required, the owner scope, same-stream/queue requirement, command-buffer or
batch ownership, command-pool lifetime, descriptor lifetime,
allocator/retire-timeline scope, planned stack-entry acquire point,
stack-exit release/submit point, and public/final/host/readback policy. Current
rows remain fail-closed with
`owned_command_buffer_contract_runtime_api_present_result_unavailable`; they
define the planned-region object and lifetime contract but do not allocate,
switch, defer, or submit a command buffer. The first concrete blocker behind
the API is command-pool lifetime extension for a region-owned command buffer or
batch.
`StackRegionCommandBufferLifetimeReservation.v0` is the behavior-neutral
request/result surface for that first concrete blocker. It is keyed by the
stack-region instance, selected boundary, and planned region-exit submit point,
and records the requested command-buffer or batch identity, requested
stack/region lifetime, command-pool lifetime scope, owner/requester scope,
whether reservation is required, runtime API source, result status, and the
specific command-pool, command-buffer, and region-exit release-point lifetime
statuses. The current runtime API skeleton always returns unavailable and
performs no allocation, command-buffer switch, submit, or defer operation.
Current selected rows therefore report
`command_buffer_lifetime_reservation_unavailable` with
`command_pool_cannot_extend_beyond_phase_submit` as the top blocker.
`StackRegionCommandPoolLifetimeContract.v0` is emitted below that reservation
to decompose the command-pool lifetime failure. It records the stack-region
instance, current command-pool owner/scope, selected phase-submit boundary id,
requested region lifetime scope, planned region-exit release point id and
status, linked command-buffer lifetime reservation key, command-pool retention
API status, command-pool reset deferral status, and runtime API source. Current
rows report `command_pool_lifetime_contract_unavailable`,
`command_pool_owner_context_phase_submit_scope_retained_until_release_point`,
`command_pool_retention_request_api_present_release_point_observed`, and
`command_pool_reset_deferral_proof_unavailable_reset_deferral_implementation_missing`.
The contract authorizes no submit elision and performs no command-pool reset
deferral or command-buffer lifetime change.
`StackRegionCommandPoolRetentionRequest.v0` and
`StackRegionCommandPoolRetentionResult.v0` are emitted below the command-pool
lifetime contract. They are keyed by the selected stack-region instance,
boundary, planned region-exit release point, lifetime reservation, and
command-pool lifetime contract. The request records the current context
phase-submit command-pool owner scope, requested stack/region retention scope,
same-stream/queue requirement, public/final/host/readback policy, and
fail-closed runtime API source. The result distinguishes API-present but
unavailable, observed context command-pool retention through the preserved
stack-exit submit, missing reset-deferral proof, planned release-point status,
and same-stream/queue proof status. It does not transfer command-pool reset
ownership, defer resets, allocate command buffers, create queue submits, or
authorize submit elision. Current selected rows report
`command_pool_retention_result_context_pool_retained_until_observed_release_point`
with `command_pool_reset_deferral_implementation_missing` as the top blocker
once the stack-exit release point is observed.
`StackRegionExitReleasePoint.v0` is emitted beside the planned submit point to
name the future stack/region exit release target. It records the stack-region
instance, owner scope, planned recording/stack-exit callsite, planned submit
point id, command-buffer/batch release target, and release responsibilities for
command-buffer close/submit, descriptor lifetime, retire timeline,
allocator/resource retirement, and command-pool cleanup. Current rows remain
behavior-neutral, but the stack planned-recording exit submit is now an
observed release anchor:
`exit_release_point_runtime_observed_context_submit_preserved`. The
command-buffer release target is still not connected to a region-owned
command-buffer abstraction. Ordinary phase submits are modeled as
closing/submitting the active command buffer, consuming recording state, and
creating a retire timeline, not as a literal command-pool reset.
`StackRegionExitReleaseOwnershipContract.v0` is emitted for the same selected
boundary and planned exit point. It is keyed by stack-region instance,
boundary, and exit-release point, and records whether the future stack/region
owner would own command-buffer close/submit, queue submit/timeline signaling,
descriptor lifetime release, retire timeline release, allocator/resource
release, and command-pool cleanup/reset. Current rows are behavior-neutral and
fail closed with `exit_release_ownership_contract_unavailable`; the top blocker
is `missing_region_exit_release_ownership_implementation`. The contract
authorizes no submit elision and does not install a release hook.
`RegionCommandBufferOwnership.v0` is the first scaffold for that ownership
direction. It emits paired `stack_entry_acquire` and `stack_exit_release`
records for the selected stack-region instance. Acquire rows expose the region
id, `StackRegionCommandBufferAcquireHook.v0` key/status,
`RegionOwnedCommandBufferLease.v0` key/status, preserved context phase-submit
command-buffer candidate identity/status, unavailable stack-region
command-buffer and command-pool lease identities, descriptor generation base,
scratch/temporary resource scope, and owner/requester scope.
Release rows expose output release status, pending-retire transfer status, and
command-pool reset deferral status. The rows are diagnostic-only: phase submits
are preserved, deferred submit is disabled, and `authorizes_submit_elision=0`.
`StackRegionCommandBufferAcquireHook.v0` is the behavior-neutral runtime hook
surface behind the acquire-side lease. It snapshots current stack
planned-recording ownership, current command-buffer recording id, the current
`vulkan_context_phase_submit_owner` scope, and context/phase-submit-owned
descriptor and command-pool scopes. When the stack planned recording is active,
it records the preserved context command-buffer batch as a non-transferable
candidate with a stack-entry lifecycle id finalized at submit or cancel. It
still reports `behavior_enabled=0`, `lease_available=0`,
`submit_elision_enabled=0`, and `new_queue_submit_created=0` in this slice.

`StackRegionSingleRecordingPlan.v0` is emitted between that acquire hook and
the lease rows. It records the planned single-region recording status,
current execution recording mode, plan lifecycle id/status, and the reason a
borrowed context command buffer cannot be used as a phase-spanning region
lease. Current rows report
`stack_region_single_recording_plan_present_behavior_disabled`,
`context_phase_submit_recording`, and
`borrowed_context_command_buffer_region_lease_rejected_phase_submit_closes_recording`;
they preserve phase-boundary submits, create no queue submit, keep command
buffer execution topology unchanged, and authorize no submit elision.

`RegionOwnedCommandBufferLease.v0` is the behavior-neutral acquire-side lease
surface. It is keyed by stack-region instance and selected boundary, joins the
acquire hook key/status, and records the planned region-exit release/submit
point, requested stack-region owner scope, current Vulkan context/phase-submit
owner scope, whether a lease was requested and emitted, region command-buffer
or batch lease availability, command-pool lease availability, descriptor
lifetime scope, retire timeline scope, same-stream/queue requirement status,
public/final/host/readback blocker status, and behavior-disabled flags. Current
rows report
`region_owned_command_buffer_lease_unavailable_single_recording_owner_lacks_close_submit_ownership`;
no command buffer is allocated, switched, replayed, deferred, closed, or
submitted.
`StackRegionSingleRecordingOwner.v0` is emitted between the single-recording
plan and acquire/lease rows. It records a real stack planned-recording
lifecycle, including owner id, lifecycle status, current command-buffer
recording id, and the fact that close/submit, command-pool, descriptor-scope,
and retire-timeline ownership all remain with the context phase-submit path.
It is behavior-neutral and does not authorize submit elision.

`StackRegionSingleRecordingCanary.v0` is the first opt-in behavior canary built
from this surface. It is enabled only by
`PYTORCH_VULKAN_STACK_REGION_SINGLE_RECORDING_CANARY=non_capture_residual2_norm1_block1`
and still requires
`PYTORCH_VULKAN_STACK_REGION_BARRIER_CANARY=non_capture_residual2_norm1_block1`.
The canary is intentionally two-stage: a proof warmup records the selected
non-capture `residual2@0 -> norm1@1` boundary plan and real consumer-side
barrier evidence, then a second pass may defer exactly one matching
stack-owner phase-boundary submit to stack exit. It does not enable the older
submit-elision canary, does not broaden the selected boundary, and records
`authorizes_submit_elision=0` because the behavior is owned by the
stack-region single-recording owner. The live guard still fails closed unless
the selected boundary is active, a stack planned-recording owner is active, the
command-buffer recording id is observed, the pending dispatch range is
complete, the actual Norm1 input barrier proof is present, validated barrier
coverage spans the pending dispatch range, and host/final/readback blockers are
absent.
The first real `vits_140` bridge run with the canary did remove one selected
phase-boundary submit, but stack-output bridge sanity failed. The benchmark
harness therefore marks those timings invalid through
`performance_invalid_reasons=["vulkan_stack_output_device_bridge_sanity_failed"]`.
The canary remains a diagnostic proof surface only; it must not be promoted as
a performance path.
When the planned stack scope is active, the canary rows now report the same
preserved phase-submit batch lease seen by the region ownership rows:
`region_owned_command_buffer_batch_lease_available_preserved_phase_submits`
with
`region_exit_close_submit_owner_preserved_phase_submit_batch_fail_closed`.
That lease is still accounting-only. The close/submit owner blocker remains
`region_exit_close_submit_owner_handoff_blocked_preserved_phase_submit_batch_context_owned`,
and the pending-dispatch barrier coverage guard still prevents submit
deferral.
`StackRegionCommandBufferTopologyPlan.v0` is the bounded post-failure scaffold
above that local submit hook. It records the selected stack-region instance,
planned stack-exit submit point, linked single-recording plan/owner, current
context phase-submit topology, and requested region-owned stack-entry-to-exit
command-buffer or batch topology. Current rows preserve phase-boundary submits
and fail closed. Bridge-scoped rows now observe
`vision_stack_decoder_bridge_region` through `VulkanStackPlannedRegionScope`
and fail closed with
`planned_region_topology_present_close_submit_still_context_owned`. Non-bridge
rows can still report
`missing_region_owned_command_buffer_topology_owner_above_stack_scope`. This
states that a planned region identity is necessary but not sufficient: close
and submit still belong to the context phase-submit topology.
The close/submit owner request, result, and owner rows carry the same planned
bridge context so the release-owner proof can report the precise blocker
`planned_region_topology_present_close_submit_still_context_owned`. This is a
fail-closed classification only; it does not create a region-owned command
buffer, defer a submit, close a command buffer at stack exit, or change
descriptor/retire ownership.
`StackRegionExitSubmitRuntimePoint.v0` records the real preserved
`StackPlannedRecordingSubmit` at stack planned-recording exit. When that row is
present for the planned bridge context, `StackRegionPlannedSubmitPoint.v0` and
the close/submit owner rows advance from a synthetic planned target to
`planned_region_exit_submit_point_runtime_observed_context_submit_preserved`.
`StackRegionExitReleasePoint.v0` now also advances to
`exit_release_point_runtime_observed_context_submit_preserved`, giving later
ownership proofs a concrete stack-exit release point.
`StackBoundarySubmitLevelEquivalenceProof.v0` also consumes that preserved
runtime exit-submit point when the planned context has already closed and the
graph contains a single unambiguous `StackPlannedRecordingSubmit` exit row. The
submit-level deferred-submit status can therefore report
`planned_region_exit_submit_point_runtime_observed_context_submit_preserved`
instead of stopping at a synthetic planned target. The next blocker becomes
`retire_timeline_migration`: the runtime submit point is observed, but pending
resources have not been migrated to a later region-owned retire timeline. This
does not authorize submit elision or change any submit behavior.
`StackRegionRetireTimelineMigration.v0` is the behavior-neutral accounting
surface for that blocker. It consumes the same runtime exit-submit observation,
the selected submit-level pending resource counts, and the close/submit owner
state, then reports whether retire transfer accounting is available. Current
rows can report
`retire_timeline_migration_accounting_available_behavior_disabled` with
`pending_retires_transfer_accounting_available_behavior_disabled`, but the
current retire timeline and resource lifetime still belong to the preserved
context submit path. The row is not a transfer, does not move retire entries,
does not defer or create submits, and does not authorize submit elision.
`StackRegionRetireTimelineOwner.v0` now consumes that migration row and adds
the corresponding Context-owned lifecycle surface. Stack planned recording
creates a `ContextStackRegionRetireTimelineOwnerState.v0` id, then finalizes
it on submit or cancel while keeping the current ownership scope context-owned:
the active/submitted/canceled states all end in
`context_owned_not_transferred`. This is an accounting owner only. Generic rows
keep `owner_available=0`; stack-exit close-submit owner mode can report
`owner_available=1` for accounting once the runtime exit-submit owner is
joined. Both forms keep `behavior_enabled=0`, `transfers_retire_timeline=0`,
`authorizes_submit_elision=0`, and fail closed on
`retire_timeline_owner_behavior_disabled` when migration accounting is
available.
`StackRegionPendingRetireTransferPlan.v0` is the next behavior-neutral handoff
surface. It snapshots the Context pending-retire queues and stack-internal
retire batch without moving entries, then compares those counts and bytes with
the submit-level graph pending set. The plan can say whether the source still
lives in context pending-retire storage, in the stack-internal batch, has
already been consumed by the preserved submit, or differs from the graph view.
It does not transfer resources, change retire queue ownership, or authorize
submit elision.
The Context now also binds pending-retire sources before the preserved submit
path consumes them. A stack-exit source can report
`pending_retire_transfer_source_bound_to_region_exit_submit` when it matches
the graph pending set. A source observed earlier at the preserved phase-boundary
submit reports
`pending_retire_transfer_source_complete_at_preserved_phase_submit` when it
matches the graph pending set, making clear that the source exists but is still
owned by the preserved phase-submit path. A preserved phase-submit source that
is a superset of the selected graph-pending set reports
`pending_retire_transfer_source_superset_at_preserved_phase_submit`; it remains
fail-closed because the extra source resources are not transferable region-exit
ownership. Partial source bindings remain explicit through
`pending_retire_transfer_source_partially_bound_to_region_exit_submit` or
`pending_retire_transfer_source_partially_bound_to_preserved_phase_submit`.
These rows make the gap between known sources and transferable region-exit
ownership explicit before any future owner can claim retire transfer behavior.
The opt-in
`PYTORCH_VULKAN_STACK_REGION_PENDING_RETIRE_TRANSFER_OWNER=stack_internal_until_stack_exit`
mode lets the stack-exit stack-internal retire batch source override the
earlier preserved phase-submit source for source-binding diagnostics only. It
does not transfer retire entries, defer a submit, or authorize submit elision.
If the stack-exit batch exactly matches the selected graph-pending set, the row
can advance to a stack-exit source-available state while behavior remains
disabled. Partial, superset, mixed, or missing stack-exit sources continue to
fail closed; the current selected synthetic boundary reports a partial
stack-exit source and therefore remains blocked by
`pending_retire_transfer_source_incomplete`.
`PYTORCH_VULKAN_STACK_REGION_BATCH_QKV_RETIRES=1` separately enables the
already-classified QKV stack-temp retire-batch candidate class. With QKV
batching plus stack-exit source binding, the current selected synthetic
boundary covers the graph-pending bytes at stack exit, but still has partial
raw resource-count coverage because metadata/uniform bookkeeping entries are
not stack-internal retire-batch targets. The transfer-plan row now exposes
those typed graph bookkeeping entries as
`graph_bookkeeping_excluded_resource_count/bytes`, derives
`graph_transfer_required_resource_count/bytes`, and emits
`source_coverage_after_bookkeeping_exclusion_status` plus filtered missing
transfer-required counts. This filtered coverage is an accounting diagnostic,
not source identity. The main `source_match_status` remains the raw source
match, and the owner remains fail-closed unless a concrete source match is
proven without relying on bookkeeping-excluded count/byte coverage.
The graph now also exports a pending allocation signature for source-identity
diagnostics. `StackRegionBoundarySubmitPlan.v0` includes
`pending_allocation_signature` entries keyed by allocation id, generation, byte
range, resource class, count, and bytes. The pending-retire transfer plan
filters the graph signature to transfer-required non-bookkeeping entries and
compares it with the source signature bound at region exit. It reports the
result as `source_identity_match_status` plus
`graph_transfer_required_allocation_signature`,
`region_exit_bound_source_allocation_signature`, and missing transfer-required
identity count/byte fields. Malformed signatures are explicit fail-closed
identity statuses, not silently ignored transfer sets. This does not replace
the raw `source_match_status`: identity coverage is observable, but the owner
requires exact or source-superset identity coverage before source availability
can be reported, and no pending-retire entries move until an explicit
ownership implementation consumes that proof.
The source signature is retained per stack-region source id and the transfer
plan reads the snapshot matching its `stack_region_instance_id` when available.
That makes repeated warm/timed stack runs distinguishable. The current
stack-exit batch source is therefore instance-correct but still not
identity-equivalent to the selected phase-submit pending graph set, so the row
continues to report source-incomplete ownership rather than granting transfer.
To make that blocker actionable, the transfer row also reports
`source_identity_mismatch_axis`,
`source_identity_exact_intersection_count/bytes`,
`source_identity_allocation_range_overlap_count/bytes`, and
`source_identity_class_only_overlap_count/bytes`. The current selected
stack-exit batch path reports
`source_identity_mismatch_same_class_different_allocation_set`: there is
resource-class overlap, but no exact identity or allocation/range overlap with
the selected phase-submit pending graph set.
The transfer row also keeps the preserved phase-submit source snapshot for the
same stack-region instance. That comparison can prove the graph pending set is
present at the preserved phase-submit source while the stack-exit batch source
is a different allocation set. The preserved source fields include source id,
state, status, resource count/bytes, allocation signature, identity status, and
missing identity counts. By default this is still a handoff candidate only:
the preserved phase-submit path remains `context_owned_not_transferred`, and no
pending-retire transfer or submit elision is authorized.
`StackRegionPendingRetireTransferOwner.v0` is the corresponding owner handoff
surface. It consumes the transfer-plan status, source match, retire-timeline
owner status, and planned release submit point, then emits a separate owner
decision. Generic rows keep `owner_available=0`; rows with a concrete source
match can expose `owner_available=1` for accounting only. Without an opt-in
handoff canary, rows keep `behavior_enabled=0`,
`transfers_pending_retires=0`, and `authorizes_submit_elision=0`. A complete
source therefore fails closed on
`pending_retire_transfer_owner_behavior_disabled`, a source available only at
the preserved phase submit fails closed on
`pending_retire_transfer_preserved_phase_submit_handoff_behavior_disabled`
with
`pending_retire_transfer_owner_preserved_phase_submit_handoff_available_behavior_disabled_fail_closed`,
and it emits explicit handoff API-present/candidate/behavior/transfer fields.
`Context` also owns an empty-by-default stack-region pending-retire handoff
batch with stack-entry clear, stack-exit retire, cancel restore, forced-clear
cleanup, and source-signature participation. The opt-in
`PYTORCH_VULKAN_STACK_REGION_PENDING_RETIRE_TRANSFER_OWNER=preserved_phase_submit_handoff`
canary moves only exact allocation id/generation/byte-range/resource-class
matches from the live phase-boundary target signature into that batch. It
preserves the phase-boundary submit, keeps submit elision disabled, restores the
batch on stack cancel, and retires the batch at stack exit under the observed
stack-exit submission timeline. The first synthetic bridge run reported the
remaining exact-identity gap as
`source_identity_missing_capture_sensitive_stack_activation_count/bytes` with
`source_identity_mismatch_axis=missing_capture_sensitive_stack_activation`.
The follow-up canary moves that activation only when exact allocation
id/generation/byte-range/resource-class identity matches and the pending retire
carries residual2 -> next-block norm1 provenance with no
public/final/requested/alias/runtime-input/output escape. The current row
reports zero missing capture-sensitive identities and
`pending_retire_transfer_owner_preserved_phase_submit_handoff_transferred_no_submit_elision`.
Submit elision remains disabled; the next fail-closed owner is the retire
timeline owner.
`PYTORCH_VULKAN_STACK_REGION_RETIRE_TIMELINE_OWNER=stack_exit_close_submit`
then transfers that owner only when retire-timeline migration accounting and
the stack-exit close-submit owner are present. It reports
`retire_timeline_owner_transferred_to_stack_exit_close_submit_no_submit_elision`
and keeps `authorizes_submit_elision=0`. With all current ownership canaries
enabled, the joined exit-ownership row becomes
`region_exit_ownership_transfer_complete_fail_closed`; the remaining blocker is
the explicit authorization gate, not a missing close-submit, pending-retire, or
retire-timeline owner.
An incomplete or bookkeeping-excluded source fails closed on
`pending_retire_transfer_source_incomplete`, and a
blocked plan propagates the transfer-plan blocker.
Exit-release ownership, region command-buffer ownership, and deferred-submit
runtime-hook plan rows now consume this owner status as a separate
pending-retire owner handoff field. This keeps the source snapshot and the
ownership handoff distinct: a report can say that the source is known while the
region owner remains behavior-disabled, or that the owner is waiting on the
transfer plan itself.
The row now carries `ContextStackRegionPendingRetireTransferOwnerState.v0`
lifecycle id/state/status/source. This mirrors the close-submit,
reset-deferral, and retire-timeline owner surfaces: the owner is observed from
stack entry through stack exit, but the active/submitted/canceled states remain
context-owned and not transferred.
The next blocker remains fail-closed:
`region_exit_close_submit_owner_handoff_blocked_preserved_phase_submit_batch_context_owned`
because the preserved phase-submit batch lease is available only as an
accounting/lifecycle lease, not as a region close/submit owner.
`RegionCommandBufferOwnership.v0` mirrors that lifecycle at stack entry and
stack exit. It can report the planned stack-region scope and the preserved
phase-submit batch lifecycle, but it keeps
`stack_entry_acquire_record_emitted=1`,
`stack_exit_release_record_emitted=1`,
`region_command_buffer_ownership_acquired=0`,
`region_command_buffer_ownership_released=0`,
`command_pool_reset_deferred_to_region_release=0`, and
`actual_elided_submit_count=0`.
The acquire/release emitted-record fields only prove the ownership rows were
populated from the graph/runtime context; command-buffer, command-pool,
descriptor, and retire ownership remain with the preserved context submit path.
The rows expose that explicitly with `region_owned_close_submit_available=0`,
`close_submit_ownership_status=close_submit_still_context_phase_submit_owned`,
`command_pool_reset_ownership_status=command_pool_reset_still_context_owned_not_deferred`,
`descriptor_lifetime_ownership_status=descriptor_lifetime_still_context_owned_not_releasable`,
and
`retire_timeline_ownership_status=retire_timeline_still_context_owned_not_transferred`.
Those rows also carry `ContextRegionCommandBufferOwnershipState.v0`, a separate
context lifecycle for acquire/release observation. The lifecycle states are
context-owned and fail-closed; they do not make the close/submit owner
available and do not authorize submit elision.
This does not remove, defer, batch, replay, or create a submit.
`StackRegionExitReleaseOwnership.v0` is emitted beside those release rows to
classify the release responsibilities that a future stack/region owner would
need to take over. It reports public, private bridge, captured,
requested-intermediate, and final output release status; pending-retire
transfer; descriptor, retire-timeline, allocator/resource, command-buffer
close/submit, queue-submit/timeline, and command-pool cleanup ownership. Current
rows preserve all runtime behavior and fail closed with
`missing_command_buffer_close_submit_release_ownership` after observing that the
phase submit still owns command-buffer close/submit and retire-timeline
creation.
`StackRegionCommandBufferCloseSubmitOwnership.v0` is the typed component row
for that first missing release owner. It is keyed by stack-region instance and
selected boundary, carries the current command-buffer recording id/scope and
planned region-exit release point, and reports whether close/submit ownership
belongs to the current phase submit or a future region-exit owner. Current rows
report `current_phase_submit_owns_command_buffer_close_submit`,
`command_buffer_not_region_owned`, and
`planned_region_exit_submit_point_runtime_observed_context_submit_preserved`
when the real preserved stack-exit submit point is observed, with
`StackRegionExitCloseSubmitOwnerRequest.v0` /
`StackRegionExitCloseSubmitOwnerResult.v0` feeding a first-class
`RegionExitCloseSubmitOwner.v0` owner surface. The owner row is emitted for the
selected stack-region instance and reports the planned release point, current
context/phase-submit close-submit owner, requested region-exit ownership,
region-owned command-buffer or batch availability, queue/timeline owner
availability, retire-timeline handoff availability, descriptor-lifetime
handoff availability, command-pool cleanup availability, and final
fail-closed reason. Current rows still treat the preserved phase-submit batch
lease as accounting evidence rather than transferable close/submit ownership.
When the command-pool reset-deferral proof has already identified the selected
missing implementation, the close-submit owner request/result and surface rows
propagate `command_pool_reset_deferral_implementation_missing` as the more
specific fail-closed blocker instead of stopping at the generic preserved-batch
classification. The surface is diagnostic-only and does not create, defer,
close, or submit command buffers.
When the reset-deferral owner accounting bridge is present, close-submit
request/result rows can report accounting availability, but
`RegionExitCloseSubmitOwner.v0` remains fail-closed with
`region_exit_close_submit_owner_accounting_available_behavior_disabled_fail_closed`
and `ownership_available=0`. This keeps accounting separate from behavior
authorization.
`PYTORCH_VULKAN_STACK_REGION_CLOSE_SUBMIT_OWNER=preserved_phase_submit_batch`
is the next canary layer. It requires the reset-deferral owner blocker to be
clear, then records the active preserved phase-boundary close/submit lifecycle
state `7` /
`region_exit_close_submit_owner_active_preserved_phase_submit_close_submit_available`
plus close-submit owner behavior availability while keeping
`region_exit_close_submit_owner_authorizes_submit_elision=0`. This is
accounting over the existing phase-boundary submit only, not transferable
region-exit ownership. The fail-closed reason becomes
`region_exit_close_submit_owner_handoff_blocked_preserved_phase_submit_batch_context_owned`,
so the next behavior-changing step still requires a real region close/submit
owner rather than preserved-batch accounting.
`PYTORCH_VULKAN_STACK_REGION_CLOSE_SUBMIT_OWNER=stack_exit_close_submit`
adds that next behavior-neutral owner surface. It reports only the stack-exit
close/submit lifecycle as region-owned for accounting, emits
`region_exit_close_submit_owner_handoff_available_stack_exit_close_submit_owner`,
and leaves submit elision authorization disabled on the stack-exit runtime-point
row. Earlier phase-boundary rows remain preserved-batch/context-owned, so the
submit-level graph joins the stack-exit runtime-point owner back into the
selected-boundary aggregate transfer. That join lets the transfer advance past
close-submit ownership, then fail closed on the next incomplete owner.
The live `StackRegionSingleRecordingCanary.v0` guard consumes the same opt-in
flag. This lets selected-boundary rows distinguish an unavailable preserved
batch from a behavior-enabled close-submit owner surface whose submit-elision
authorization is still disabled. The live guard also has a separate
close-submit authorization input, passed as `0` in the current implementation,
so owner availability cannot remove a submit without an explicit future
authorization source. It does not remove, defer, or create submits.
`StackRegionSingleRecordingCanary.v0` now consumes a live Context-owned
close/submit owner lifecycle id/state for that same decision. Active stack
planned recording creates a lifecycle record, but the current state only
represents the preserved phase-submit batch as accounting evidence, not
transferable region close/submit ownership. The availability source is recorded
as `ContextStackRegionCloseSubmitOwnerState.v0`, with a separate
behavior-enabled bit that remains `0`. Submit removal stays disabled until a
future lifecycle state is backed by a real region exit close/submit owner and
explicitly enables the behavior.
The lifecycle id, state, status, behavior bit, authorization bit, and
`ContextStackRegionCloseSubmitOwnerState.v0` source are also emitted on the
single-recording owner, acquire hook, region-owned command-buffer lease,
exit-close-submit request/result, region exit owner, command-buffer
close/submit ownership, and region command-buffer ownership rows. These fields
are graph provenance only; they do not replace value-preservation proof or
authorize submit elision.
`StackRegionCommandPoolResetDeferralProof.v0` is emitted from that retention
result. It records the current phase-submit recording-epoch consumption point,
planned region-exit release/reset point, linked retention result key/status,
descriptor lifetime status, command-buffer lifetime status, retire-timeline
status, and fail-closed top blocker. Current rows report
`command_pool_reset_deferral_proof_complete_context_pool_retained_until_release_point`
when the preserved context command pool is observed through the stack-exit
release point. This proves current context retention, not region ownership, and
refines the older `command_pool_reset_deferral_proof_blocked_retention_unavailable`
and implementation-missing proof buckets without changing command-pool reset
behavior. `StackRegionCommandPoolLifetimeContract.v0` reports the same split as
`command_pool_lifetime_context_retained_not_region_owned`.
`StackRegionCommandPoolResetDeferralOwner.v0` consumes that proof and emits the
first-class owner surface for reset deferral. It is still diagnostic-only:
proof-complete rows can report `owner_available=1` for accounting, but
`reset_deferral_behavior_enabled=0`, `defers_command_pool_reset=0`, and
`authorizes_submit_elision=0`. When the proof layer is complete, the owner row
now fails closed with `command_pool_reset_deferral_owner_behavior_disabled`
instead of treating the accounting owner as execution ownership. The owner row
also records
`ContextStackRegionCommandPoolResetDeferralOwnerState.v0`
lifecycle id/state/status/source so a later implementation can distinguish
not-started, active context-owned, submit-finalized context-owned, and
cancel-finalized context-owned states. None of those states transfers reset
ownership to the region. Close-submit ownership must consume this owner surface
rather than treating proof strings as behavior authorization.
`RegionExitOwnershipTransfer.v0` joins the close-submit owner, command-pool
reset-deferral owner, pending-retire transfer owner, retire-timeline owner, and
stack-exit release point surfaces into one behavior-neutral transfer row for
the selected stack-region instance and phase boundary. The row can report
joined accounting when all required component surfaces are present, and it now
computes ownership completion separately from accounting. Current rows still
report `submit_elision_enabled=0`, `deferred_submit_enabled=0`, and
`phase_boundary_submits_preserved=1` by default. With all explicit owner
canaries enabled, the aggregate transfer can reach
`ownership_transfer_complete=1` while still failing closed on authorization.
That completed transfer is a prerequisite for experiments, not permission to
remove or defer a submit.
The existing `StackRegionSingleRecordingCanary.v0` guard consumes the same
component lifecycle state and records the aggregate transfer status on canary
rows. The guard remains fail-closed by default at
`region_exit_close_submit_owner_authorizes_submit_elision_disabled`. Enabling
the explicit submit-elision canary can make the row report
`region_exit_ownership_transfer_complete_authorized_canary`, but the current
topology is still blocked by
`single_recording_current_topology_value_preservation_rejected`: skipping the
selected phase-boundary submit after the fact was observed to corrupt
private-capture outputs intermittently. Future performance work should build a
planned single-region recording path rather than adding more local proof fields
to this submit-deletion path.
The opt-in
`PYTORCH_VULKAN_STACK_REGION_RESET_DEFERRAL_OWNER=context_retained_release_point`
canary advances only this owner surface when the proof is complete:
`reset_deferral_behavior_enabled=1`, `defers_command_pool_reset=1`, and
`authorizes_submit_elision=0`. It intentionally leaves close-submit ownership
fail-closed so a later slice can prove that boundary separately.
`StackRegionCommandBufferRequestHookPlan.v0` joins that request/result pair
with the planned callsites. Current rows are behavior-neutral with
`hook_installed=0`, `request_hook_plan_api_present_result_unavailable`, and
`authorizes_submit_elision=0`.
`StackRegionDeferredSubmitPlan.v0` is the corresponding architecture planning
row. It models a future region-owned command-buffer or command-buffer-batch
that would keep the selected pending dispatch range open and submit it later at
a planned stack boundary with equivalent timeline and retire semantics. The row
records the candidate phase-submit key, command-buffer scope, submit epochs,
the current mandatory reason for the submit, whether a later region submit point
is observed, same-stream/queue and same-region-owner status, retire timeline
migration, descriptor lifetime and command-pool reset risk, host/fence/public
blockers, and the top migration blocker. Current rows remain fail-closed with
`stack_region_deferred_submit_authorizes_submit_elision=0`; when only the
synthetic region-exit target exists the plan status is
`stack_region_deferred_submit_plan_planned_target_unimplemented`, with
`planned_region_submit_point_exists_but_unimplemented` as the reason and
missing region-exit release ownership implementation as the top blocker.

`StackRegionRecordingDomain.v0` records the current command-buffer topology
beside the existing stack graph rows. It emits behavior-neutral rows for stack
entry, preserved phase-boundary submits, and stack exit with
`recording_domain_mode=context_phase_submit_compat`,
`command_buffer_owner_scope=vulkan_context_phase_submit_owner`,
`region_owned_command_buffer_active=0`, and
`current_topology_submit_elision_forbidden=1`. These rows are not a new owner
and do not authorize submit elision; they document that the current topology
still consumes context-owned command-buffer epochs at phase boundaries before a
planned region-exit submit. The next behavior-bearing architecture must provide
a real region-owned command-buffer recording domain before a single-recording
path can replace the preserved phase submits. With stack graph diagnostics
enabled, the same schema also records `active_cmd_context` observations from
`Context::active_cmd()` during stack planned recording. Those rows prove that
dispatch recording is still routed through the context-owned command buffer;
they are intentionally gated diagnostics and do not create a region-owned
recording scope.
The opt-in
`PYTORCH_VULKAN_STACK_REGION_OWNED_COMMAND_BUFFER=stack_entry_to_exit` canary
uses the existing prepared-command-buffer external-recording path to create the
first stack-entry-to-exit recording domain. Under that canary, stack dispatches
emit `active_cmd_external` rows and stack exit submits the prepared command
buffer. This is a focused canary only; the stack owner currently limits it to
small private-bridge stack scopes and keeps larger/full DAv2-style stacks on
the context-owned path. It does not change the default path and does not
convert the older current-topology submit-elision canary into an accepted
optimization.
`StackRegionRecordingDomain.v0` keeps the legacy
`phase_boundary_submits_preserved=1` compatibility marker, but also records
`phase_boundary_submit_calls_preserved=1` and
`phase_boundary_queue_submits_preserved=0` for the owned canary to show that
logical phase-boundary call sites remain while queue submit ownership moves to
the stack-exit prepared command buffer.
The canary uses a separate stack-owned recording dispatch counter for graph
rows. It does not reuse the context `submit_count_`, so diagnostics can report
recorded work without accidentally re-enabling context phase-boundary submits.
`PYTORCH_VULKAN_STACK_REGION_OWNED_COMMAND_BUFFER=segmented_stack_entry_to_exit`
adds a bounded segmented variant for private-bridge stacks. It splits at
capture boundaries, opens one stack-owned recording scope for each segment of
four blocks or fewer, allows at most two scopes, then submits each scope at its
local stack-exit. Over-budget or too-many-segment stacks stay on the
context-owned path because unbounded and four-scope full-stack experiments have
already shown stack-overflow/device-loss risk.
`PYTORCH_VULKAN_STACK_REGION_OWNED_COMMAND_BUFFER=segmented_stack_prefix_to_exit`
is a narrower behavior canary for too-many-segment stacks: when all candidate
segments satisfy the per-segment block budget but exceed the two-scope budget,
it records only the first two segments through stack-owned external recording
and leaves the remaining tail on the existing context-owned path. Segment rows
mark `segment_selected_for_recording`, and the summary reports
`segment_plan_coverage=prefix`; this is not a full-stack recording proof. The
selected external segments must also stay within the small planned-dispatch
budget, so higher-dispatch real-model segments reject before external recording
starts.
`PYTORCH_VULKAN_STACK_REGION_OWNED_COMMAND_BUFFER=segmented_stack_dispatch_budget_prefix_to_exit`
is the first dispatch-budget behavior canary after persistent external
recording pools gained a global-completion reset owner. It derives
block-boundary segments from planned dispatch counts, records only the first two
segments through stack-owned external recording, and leaves the remaining tail
on the existing context-owned path. It is still a prefix canary: larger
multi-scope sequences remain blocked until repeat stability is proven for every
selected scope, cleanup-retire ownership, and persistent-pool reset ownership.
For private bridge runs, the graph may still emit
`dispatch_budget_candidate_summary` and `dispatch_budget_candidate_segment`
rows. These rows are evidence only: they derive hypothetical block-boundary
segments from the same planned-dispatch budget while reporting
`owned_command_buffer_mode=dispatch_budget_candidate_only`,
`segment_selected_for_recording=0`, and
`planned_dispatch_count_admission_predicate=0`. They remain separate from the
new two-scope dispatch-budget prefix canary. If the full candidate plan needs
more than the current two-scope canary budget, it fails closed with
`dispatch_budget_candidate_scope_limit_exceeded`. Such rows also report
`candidate_sequence_scope_limit_exceeded=1`,
`candidate_sequence_requires_multi_scope_owner=1`,
`candidate_sequence_repeat_stability_required=1`, and
`candidate_sequence_blocker=multi_scope_repeat_stability_unproven` so the
blocker is not confused with a request to raise the scope-count constant. A
larger dispatch-budget sequence needs a repeat-stable multi-scope ownership
contract before it can become a behavior canary.
Repeated model-level runs may set
`PYTORCH_VULKAN_STACK_DEP_GRAPH_MODE=summary_only` with
`PYTORCH_VULKAN_STACK_DEP_GRAPH`. Summary-only dumps keep the root summary and
the low-volume ownership rows needed for segment readiness, recording-domain,
cleanup-boundary, cleanup-retire, canary, and consumer-registration checks.
They omit the full dispatch/resource/live-binding arrays and derived full-graph
proof sections. This is only a graph-output mode; it does not change
recording, submit, cleanup, retire, segment selection, or canary admission.
`StackRegionSegmentPlan.v0` records the generic plan evidence for that choice.
It emits a summary row even when segmentation rejects, and per-segment rows
when candidate segments are computed. Rows carry only generic inputs and
budgets: private bridge state, runtime capture indices, plan-capture indices,
block count, segment ends, total and per-segment planned dispatch counts from
the existing stack shape plan, the four-block segment limit, the two-scope
limit, and fail-closed reasons such as `segment_scope_limit_exceeded`. The
planned dispatch count is a small-scope canary admission predicate for selected
external segments; over-budget selected segments fail closed before external
recording starts. The row itself remains behavior-neutral: it does not open
scopes, change command-buffer topology, move pending retires, defer submits, or
authorize submit elision.
Stack-owned external cleanup logical-boundary rows now carry the same segment
identity when the segmented canary opens a segment scope. The segment index,
block range, and segment planned dispatch count make cleanup rows joinable to
`StackRegionSegmentPlan.v0` without inferring from row order. These fields are
metadata only; they do not transfer pending retires, enforce cleanup budgets, or
authorize larger segment scopes.
`StackRegionExternalRecordingCleanupRetire.v0` records the stack-exit cleanup
scheduling side of stack-owned external recording. It reports retained
buffer/image counts, retained cleanup bytes, stack-exit submit timeline
validity, the cleanup-retire action, and external recording pool-pressure
counters. The pool-pressure fields report cumulative and per-scope persistent
command-buffer acquisitions plus descriptor-set allocations observed while an
external recording command buffer was active. Persistent external recording
pools are reset only at global completion/fence-wait flush points, after no
external recording is active. Rows with observed pool pressure report
`external_pool_reset_required=1`,
`external_pool_reset_owner_available=1`,
`external_pool_reset_point=global_completion_flush`, and
`persistent_pool_reset_proven=1`; the per-row
`persistent_command_pool_reset_performed=0` and
`persistent_descriptor_pool_reset_performed=0` fields mean reset is not
performed at cleanup-retire emission itself. A valid row with
`scheduled_on_stack_exit_submission` only proves the cleanup batch was handed
to the existing retire queue under the stack-exit submission. It does not
transfer pending retires, remove phase-boundary submits, defer submits, or prove
that larger external-recording segments are repeat-stable.

`StackBoundaryValuePreservationContract.v0` is the decisive behavior gate for
any future phase-submit elision. It is documented in
`docs/vulkan/STACK_BOUNDARY_VALUE_PRESERVATION.md` and requires both
resource-side proof and submit-level value-preservation semantics: same
recording scope, producer-before-consumer order, descriptor generation
stability, actual consumer input barrier coverage, old-carry non-escape,
absence of public/final/host/readback/requested-intermediate dependencies,
pending-write coverage, lifetime/allocator side-effect safety, and a real
replacement or proof for phase-submit execution/flush semantics. Current
`vits_140` rows are barrier-ready but not behavior-ready because the stack
owner can request a region-owned command buffer or batch only through a
diagnostic API that returns unavailable; there is still no implementation that
preserves the removed submit's execution, timeline, and retire role.

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
