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
`command_pool_owner_context_phase_submit_scope`,
`command_pool_retention_request_api_present_result_unavailable`, and
`missing_command_pool_reset_deferral_proof`. The contract authorizes no submit
elision and performs no command-pool retention or command-buffer lifetime
change.
`StackRegionCommandPoolRetentionRequest.v0` and
`StackRegionCommandPoolRetentionResult.v0` are emitted below the command-pool
lifetime contract. They are keyed by the selected stack-region instance,
boundary, planned region-exit release point, lifetime reservation, and
command-pool lifetime contract. The request records the current context
phase-submit command-pool owner scope, requested stack/region retention scope,
same-stream/queue requirement, public/final/host/readback policy, and
fail-closed runtime API source. The result distinguishes API-present but
unavailable, missing region-exit release ownership, missing reset-deferral
proof, planned release-point status, and same-stream/queue proof status. It
does not retain command pools, defer resets, allocate command buffers, create
queue submits, or authorize submit elision. Current selected rows report
`command_pool_retention_result_api_present_unavailable` with
`missing_region_exit_release_ownership` as the top blocker.
`StackRegionExitReleasePoint.v0` is emitted beside the planned submit point to
name the future stack/region exit release target. It records the stack-region
instance, owner scope, planned recording/stack-exit callsite, planned submit
point id, command-buffer/batch release target, and release responsibilities for
command-buffer close/submit, descriptor lifetime, retire timeline,
allocator/resource retirement, and command-pool cleanup. Current rows remain
synthetic/planned-only and fail closed with
`exit_release_point_synthetic_planned_only`; the command-buffer release target
is not connected to a region-owned command-buffer abstraction. Ordinary phase
submits are modeled as closing/submitting the active command buffer, consuming
recording state, and creating a retire timeline, not as a literal command-pool
reset.
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
id, `RegionOwnedCommandBufferLease.v0` key/status, unavailable
command-buffer and command-pool lease identities, descriptor generation base,
scratch/temporary resource scope, and owner/requester scope.
Release rows expose output release status, pending-retire transfer status, and
command-pool reset deferral status. The rows are diagnostic-only: phase submits
are preserved, deferred submit is disabled, and `authorizes_submit_elision=0`.
`RegionOwnedCommandBufferLease.v0` is the behavior-neutral acquire-side lease
surface. It is keyed by stack-region instance and selected boundary and records
the planned region-exit release/submit point, requested stack-region owner
scope, current Vulkan context/phase-submit owner scope, whether a lease was
requested and emitted, region command-buffer or batch lease availability,
command-pool lease availability, descriptor lifetime scope, retire timeline
scope, same-stream/queue requirement status, public/final/host/readback
blocker status, and behavior-disabled flags. Current rows report
`region_owned_command_buffer_lease_unavailable_context_phase_submit_owner`; no
command buffer is allocated, switched, replayed, deferred, closed, or submitted.
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
`planned_region_exit_submit_point_synthetic_unimplemented`, with
`StackRegionExitCloseSubmitOwnerRequest.v0` /
`StackRegionExitCloseSubmitOwnerResult.v0` feeding a first-class
`RegionExitCloseSubmitOwner.v0` owner surface. The owner row is emitted for the
selected stack-region instance and reports the planned release point, current
context/phase-submit close-submit owner, requested region-exit ownership,
region-owned command-buffer or batch availability, queue/timeline owner
availability, retire-timeline handoff availability, descriptor-lifetime
handoff availability, command-pool cleanup availability, and final
fail-closed reason. Current rows fail closed with
`region_owned_command_buffer_lease_unavailable_context_phase_submit_owner`; the
surface is diagnostic-only and does not create, defer, close, or submit command
buffers.
`StackRegionCommandPoolResetDeferralProof.v0` is emitted from that retention
result. It records the current phase-submit recording-epoch consumption point,
planned region-exit release/reset point, linked retention result key/status,
descriptor lifetime status, command-buffer lifetime status, retire-timeline
status, and fail-closed top blocker. Current rows report
`command_pool_reset_deferral_proof_blocked_retention_unavailable`, because the
retention result is unavailable until region-exit release ownership exists;
this refines the older `missing_command_pool_reset_deferral_proof` bucket
without changing command-pool reset behavior.
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
