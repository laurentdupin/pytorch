# Vulkan Current State

Last refreshed: 2026-06-29 after owner-complete single-recording submit-elision
canary rejection.

## Repo State Summary

The Vulkan backend planning direction is now repo-local in `docs/vulkan`.
Ignored `agent_space` artifacts remain evidence inputs, not production
dependencies.

## DAv2 Stack Region Policy Lock

The unsafe sub-50 ms DAv2 `vits_140` path is rejected. Retire-time or
lifetime-only removal of native-layernorm phase-boundary submits corrupted
capture-sensitive stack outputs and must not be pursued as the next
optimization. The old coalescing proof showed that consumer existence and
retire lifetime proof are not sufficient: the missing contract is dispatch
ordering at the producer-consumer edge before command recording.

The current safe DAv2 bridge baseline is approximately 102-113 ms for
`vits_140`. The opt-in generic stack-captures-to-decoder bridge is correct,
copy-free, CPU-fallback-free, and sync-readback-free, but it remains limited by
stack-owner explicit synchronizes and submits. The bridge must remain guarded;
public `Tensor[]` capture behavior remains the safe default when a same-region
consumer is not registered.

The next architecture direction is a dispatch-level
`StackRegionDependencyGraph` built before command recording. Future submit
elision work must start from that graph, prove all ordering edges for a
boundary, insert device-side dependencies at the consumer dispatch point, and
only then consider skipping the matching host/queue phase-boundary submit. See
`docs/vulkan/STACK_REGION_DEPENDENCY_GRAPH.md`.

`StackRegionBoundarySubmitPlan.v0` is now the behavior-neutral online hook that
connects current-run graph proof ids to live stack-owner phase-boundary submit
sites. It records selected bridge-private boundary ids, same-region consumer
registration, public-scope rejection, and live boundary match status while
leaving `submits_removed=0` and `barriers_inserted=0`. It is the intended input
for a later one-boundary canary; it is not submit elision by itself.

The first block-2 bridge-private submit-skip canary was rejected and backed out:
`ordering_required_bytes_after_proof=0` did not prove that the phase-boundary
submit had no correctness role. Future canaries must fail closed while the
proof producer reports `behavior_change_allowed=false`. Env opt-in cannot
override that veto. A behavior-changing submit or barrier canary also needs
proof-to-live-`VulkanBuffer` binding plus validated stage/access visibility, or
an explicit no-visibility-dependency proof. Current `StackRegionBarrierPlan.v0`
records are still dry-run: they can expose planned stage/access and insertion
point metadata, but they report missing live Vulkan-buffer binding rather than
executable barrier readiness.

The owner-complete single-recording canary was also rejected for the current
execution topology. With stack-exit close-submit ownership, command-pool reset
deferral, pending-retire handoff, retire-timeline transfer, and the explicit
submit-elision env all enabled, the selected `residual2@0 -> norm1@1` phase
submit could be skipped exactly once, but stress checks produced intermittent
private-capture output corruption. The runtime now keeps that path fail-closed
with `single_recording_current_topology_value_preservation_rejected`:
ownership can be complete and explicitly authorized, but current-topology
phase-submit deletion is not value-preserving. The next performance
implementation should move to planned single-region recording instead of adding
more local submit-elision proof fields.

`StackRegionRecordingDomain.v0` now records the current command-buffer topology
without changing execution. Stack entry, preserved phase-boundary submits, and
stack exit emit rows in `context_phase_submit_compat` mode with
`region_owned_command_buffer_active=0`,
`phase_boundary_submits_preserved=1`, and
`current_topology_submit_elision_forbidden=1`. This makes the remaining blocker
explicit: the current context-owned command-buffer path consumes phase-submit
command-buffer epochs before stack exit, so future performance work needs a
real region-owned command-buffer recording domain rather than another local
submit-deletion guard. When stack graph diagnostics are enabled, `active_cmd()`
also records `active_cmd_context` rows during stack planned recording to show
that dispatch recording still uses the context command buffer rather than a
region-owned command buffer.

`PYTORCH_VULKAN_STACK_REGION_OWNED_COMMAND_BUFFER=stack_entry_to_exit` is now
the first opt-in planned-region command-buffer canary. It reuses the existing
prepared-command-buffer/external-recording substrate: stack entry acquires a
persistent command buffer, stack dispatches record through
`active_cmd_external`, and stack exit closes and submits that prepared command
buffer. The default path remains context-owned. The canary is validated only on
the focused two-block private-bridge synthetic vision stack, and the stack
owner refuses the canary for larger/full DAv2-style stacks. It still makes no
broad DAv2 performance claim.
Recording-domain rows distinguish this canary with
`phase_boundary_queue_submits_preserved=0`: logical phase-boundary calls remain,
but queue submit ownership is deferred to stack exit.

Vulkan availability checks in this tree should use
`torch.is_vulkan_available()` or `torch.vulkan.is_available()`.
`torch.backends.vulkan.is_available()` is not a valid availability signal here.

`StackRegionBarrierPlan.v0` now has a behavior-neutral live descriptor binding
surface for stack-region buffer consumers. It can join non-capture dependency
records to descriptor-bound live Vulkan buffers by exact stack scope, consumer
phase/block, descriptor binding, allocation id/generation, and byte range. This
reduces the earlier `missing_live_vulkan_buffer_binding` blocker where the live
descriptor object is observable, but canary execution remains blocked by
`behavior_change_allowed=false` and by the absence of an executed, validated
barrier path. This slice records `barriers_inserted=0` and `submits_removed=0`.
`StackRegionPreDispatchProofTable.v0` carries the first selected non-capture
`residual2@0 -> norm1@1` proof into the live consumer descriptor recording site
before dispatch recording. The table binds the proof to the live allocation
id/generation/range, producer dispatch observation, planned consumer position,
insertion token, and stage/access labels. The first explicit barrier-only
canary consumes this table under
`PYTORCH_VULKAN_STACK_REGION_BARRIER_CANARY=non_capture_residual2_norm1_block1`
and records real compute-shader write-to-read buffer barriers at the consumer
dispatch site while preserving the existing phase-boundary submit. It does not
skip submits, change public capture semantics, or broaden shapes.

`StackRegionBoundaryOptimizationPlan.v0` is the next data-driven eligibility
table over the pre-dispatch proof rows. It classifies non-capture, capture,
public, final, host-visible, and readback boundary records by live buffer
binding, allocation/generation/range match, stage/access availability,
insertion point availability, barrier-only validation, and submit-elision
eligibility. Default behavior remains unchanged. Any submit-elision experiment
must add and use a separate
`PYTORCH_VULKAN_STACK_REGION_SUBMIT_ELISION_CANARY` opt-in, must consume real
barrier insertion plus current-run proof match, and may select at most one
non-capture boundary.

`StackBoundaryProofRecord.v0` consolidates the carry, actual Norm1 input, old
carry retirement, barrier coverage, and submit-equivalence diagnostics into one
typed per-boundary row surface. The legacy histograms remain available, but
readiness should be decided from these rows: they report the candidate boundary,
producer and consumer roles, produced/actual/old-carry ranges, live descriptor
status, formal last-use and non-escape proof status, blocker status, barrier
status, `behavior_change_allowed=false`, and the fail-closed reject reason.
The current typed proof can now make the first non-capture boundary rows
submit-elision-ready after proving old residual2 carry non-escape and
retire-only status. `StackBoundarySubmitEquivalenceProof.v0` rolls those rows
up at boundary scope and remains behavior-neutral: it reports the selected
boundary row set, barrier-covered bytes, old-carry retire-only bytes,
public/host/final/alias blockers, and selected-boundary command-buffer and
submit-epoch linkage. The command/epoch proof is derived from generic
`StackRegionBoundaryOptimizationPlan.v0` records for the same boundary id. It
does not skip submits, and a later canary must still be explicit and separate.
`StackBoundarySubmitLevelEquivalenceProof.v0` is now the hard fail-closed
submit-site gate above those typed rows. It records the current-run topology
signature, pending dispatch/resource/write-set counts, descriptor/update and
retire side effects, real-barrier-to-pending-allocation matches, and the
required submit-level proof booleans. A typed row or boundary cannot report
submit-elision readiness while the submit-level proof is incomplete, while any
required boolean is false, or while topology/cardinality differs from the
current run. Submit-level rows are keyed by live submit identity rather than
boundary id alone. The key includes generic boundary fields, stack-region
instance id, command-buffer ids, submit epochs, callsite, phase, and descriptor
binding. This separates repeated stack/forward instances from the same run.
Boundary-wide rollups still reject if more than one live submit key maps to the
same selected boundary without an instance-specific join.
`StackBoundaryProofRecord.v0` rows now carry the source
`stack_region_instance_id` from the live submit row that produced their raw
provenance and prefer an instance-specific
`StackBoundarySubmitLevelEquivalenceProof.v0` join. In the current one-image
`vits_140` bridge diagnostic run, the selected non-capture boundary has one
typed barrier-ready row per stack-region instance, but no instance is
submit-equivalence complete. The remaining blocker is submit-level side-effect
completion: descriptor updates and command-buffer bookkeeping are still modeled
as pending dispatch side effects, retire entries remain pending, and a
capture-sensitive activation resource remains unmodeled for the selected
submit.
The submit-level proof rows now classify those per-instance side effects
directly: retire entries are split into
`retire_entry_proven_retire_only_or_nonescaping_*` and
`retire_entry_unknown_or_ordering_required_*`, while the remaining
capture-sensitive activation is kept fail-closed as
`capture_sensitive_activation_submit_site_relation_unproven` until it can be
joined to a typed boundary row at the live submit site. This is diagnostics
only; no submit elision or new barrier behavior is enabled.
The CUDA and DirectML backend probes have now been folded into the typed proof
schema. `StackBoundaryProofRecord.v0` and
`StackBoundarySubmitLevelEquivalenceProof.v0` explicitly report descriptor or
binding-table identity, descriptor update generation status, command-buffer and
submit visibility status, allocator pool or region identity status, transition
node provenance, and alias/public/final/host/readback escape class. These
fields are proof surfaces only. Descriptor identity remains a logical
live-binding/update-order identity derived from the stack descriptor argument
rows and submit key. Actual descriptor update-generation evidence is now emitted
as separate `StackDescriptorSetUpdateGeneration.v0` rows from
`DescriptorSet::get_bind_handle()` and joined into submit-level proof rows as
diagnostic evidence. This does not prove submit equivalence by itself: missing
pending-dispatch completion, command visibility, allocator region identity, or
transition provenance remains a hard blocker rather than an optimization
trigger.
`StackBoundaryProofRecord.v0` also records actual Norm1 input visibility
transition provenance with
`actual_consumer_visibility_transition_status`,
`actual_consumer_visibility_transition_source`,
`actual_consumer_visibility_transition_contract`,
`actual_consumer_visibility_producer_role`,
`actual_consumer_visibility_consumer_role`, and
`actual_consumer_visibility_resource_digest`. The existing opt-in
`StackRegionBarrierOnlyCanary.v0` path now labels its command-recorded barrier
record as `actual_norm1_input_visibility` when it covers the live Norm1
activation-input descriptor range, so typed rows can join the real barrier-only
record for that exact range. Default runs still report the precise missing
source, such as
`missing_barrier_only_canary_record_for_actual_consumer_input_range`. This is
still barrier-only/proof work: submits are preserved, and no default runtime
behavior changes.
`StackBoundarySubmitLevelEquivalenceProof.v0` now accounts for those
actual-consumer barrier-only records as submit-level proof inputs. Rows
distinguish no actual-consumer barrier, a barrier that exists for a different
boundary or stack instance, and a matching actual-consumer barrier that is still
blocked by another submit-level side effect. The matching join can make
`real_barrier_records`, `matched_barrier_records`, and
`actual_consumer_matched_barrier_records` nonzero for the selected boundary, but
submit elision remains disallowed while descriptor updates, retire entries, or
other submit-level side effects are incomplete.
The submit-level proof also joins pending `capture_sensitive_stack_activation`
old-carry resources back to matching `StackBoundaryProofRecord.v0` rows by
boundary, stack-region instance, and exact allocation/range digest. When the
typed row proves formal last use, no later descriptor read, no public/final/host
or alias escape, and retire-only eligibility, the submit-level accounting moves
that resource from `retire_entry_unknown_or_ordering_required` into
`old_carry_retire_only_proven_*` and reports
`typed_old_carry_proof_matches_retire_only_nonescaping`. The join uses the
live submit row's `raw_buffer_provenance_signature`; missing raw provenance is
reported explicitly instead of being treated as a successful proof miss. This is
still a
behavior-neutral accounting join; submit-equivalence remains fail-closed until
all submit-level side effects are covered. The same rows now expose explicit
submit-pending capture-sensitive join fields:
`capture_sensitive_submit_pending_records`,
`capture_sensitive_submit_pending_old_carry_joined_records`,
`capture_sensitive_submit_pending_join_status`, and
`capture_sensitive_submit_pending_join_reject_reason`. These fields make the
remaining submit-site blocker visible without authorizing submit elision.
Descriptor updates now have actual update-generation evidence, but pending
dispatch completion and command-buffer visibility are still separate
fail-closed gates. Submit-level rows report
pending dispatch list identity, recorded-position range, command-buffer
recording id, submit epochs, and explicit completion/visibility status. A
recorded position range is only diagnostic; it does not prove the pending
command list is complete or that a phase-boundary submit can be elided. The
latest proof split reports range/set completeness separately from
command-buffer submit-epoch visibility: a contiguous recorded range may match
the pending descriptor/bookkeeping side-effect rows, while
`command_buffer_submit_epoch_visibility_proof_status` still blocks submit
equivalence if the range crosses a phase-boundary submit epoch without a proven
visibility relation. That visibility gate is now split again into command-buffer
identity, submit-epoch transition, and missing-source fields. Current rows can
report `pending_dispatch_range_complete_side_effect_rows_match` while still
failing closed with
`pending_dispatches_span_completed_phase_submit_epoch_boundary_fail_closed` and
`missing_phase_submit_epoch_visibility_contract`; this is diagnostics only and
does not enable submit elision. The missing policy is now represented as the
behavior-neutral `PhaseSubmitEpochVisibilityContract` skeleton in
`StackBoundarySubmitLevelEquivalenceProof.v0`. It records whether a
phase-submit epoch crossing was observed, whether the contract is required,
the contract status, strict predicate details, and the required fields. Under
the opt-in barrier-only canary, the selected `vits_140` bridge rows may report
`phase_submit_epoch_visibility_contract_proof_only_accepted` after proving the
active command-buffer scope, complete pending dispatch range, actual descriptor
update generation, actual Norm1 input barrier, old-carry retire-only proof,
zero unknown/order-required retire entries, no public/final/host/readback
blocker, and preserved submits. This proof-ready state is not wired to submit
elision: `phase_submit_epoch_visibility_contract_behavior_enabled=0`,
`phase_submit_epoch_visibility_contract_submits_removed=0`, and
`submit_elision_ready` remains false.

The first opt-in submit-elision canary for the selected
`residual2@0 -> norm1@1` boundary removed one selected submit but failed
bridge output sanity, so the behavior-changing return path was backed out.
The current source keeps only diagnostic guard rows: selected rows can report
`phase_contract_guard_proof_ready`, but `submits_removed` remains zero. The
next proof gap is that the current `PhaseSubmitEpochVisibilityContract`
predicates are not sufficient to prove value preservation when the phase submit
itself is removed.

Post-failure hardening adds `StackRegionLiveSubmitEquivalenceBinding.v0`
diagnostic fields to the submit-elision canary rows. They explicitly distinguish
live command-buffer recording-scope identity, live submit-epoch identity,
live pending-dispatch range identity, and live side-effect completion status
at the live submit hook. The live pending range uses the same
`scope:...:command_buffer:...:positions:first-last` identity convention as the
graph proof rows. Side-effect completion is reported from the same selected
boundary predicates used by the current-run proof guard: descriptor update
generation, matched actual Norm1 input barrier, old-carry retire-only proof,
zero unknown/order-required retire entries, and no public/final/host/readback
blockers. The proof-only phase contract still reports
`phase_submit_epoch_visibility_contract_authorizes_submit_elision=0`; this is a
diagnostic binding only and does not remove submits.
`StackRegionLiveSubmitEquivalenceBindingExactJoin.v0` now supplements those
live rows at graph serialization time by joining each live submit row to the
finalized `StackBoundarySubmitLevelEquivalenceProof.v0` row with the exact
boundary id, stack-region instance, command-buffer ids, submit epochs, and
pending dispatch range identity. The earlier cumulative live side-effect
counters remain visible as emission-order diagnostics, but they are explicitly
marked non-authoritative for behavior decisions. Submit elision remains
disabled and `submits_removed=0`.
`StackRegionPendingDispatchCompletionEquivalenceProof.v0` now names the
remaining submit-level execution side-effect gap per exact submit key. It
separates exact command-list/range proof from command-buffer execution
visibility, and can report that all modeled resource hazards are covered while
the only remaining unproven dependency is phase-submit execution/flush
semantics. This proof is diagnostic only and still leaves
`phase_submit_epoch_visibility_contract_authorizes_submit_elision=0`.
`PhaseSubmitExecutionFlushContract` now makes that last blocker explicit in the
submit-level rows. The contract reports whether a phase-submit execution/flush
dependency is observed, what primitive would be required to replace it, which
candidate replacement is missing, and why barriers alone do not currently
replace the phase submit. Current selected rows remain fail-closed with
`phase_submit_execution_flush_authorizes_submit_elision=0` and
`submits_removed=0`.
`PhaseSubmitCommandBufferContinuityProof.v0` now records the first
behavior-neutral replacement-proof shape for deferring the phase submit to a
later real queue submit or timeline. The rows expose same-active-command-buffer
scope, phase-submit command-buffer close/submit observation, later
queue-submit/timeline candidate status, current retire-timeline requirement,
later-retire coverage, pending-resource escape status, and intervening blocker
status. Current selected rows may prove command-buffer continuity but still
reject because no later queue-submit/timeline candidate is observed; submit
elision stays disabled.
`StackRegionSubmitPoint.v0` now exposes the current phase-boundary real queue
submits as first-class graph nodes with submit-point id/key, stack-region
instance, phase/scope/callsite, command-buffer identity, submit epochs,
submit-point kind/status, and deferred-target status. These rows are currently
observed phase-boundary submits, not deferred-submit targets, and keep
`authorizes_submit_elision=0`.
`StackRegionPlannedSubmitPoint.v0` adds a behavior-neutral synthetic target for
a future region-exit submit. It names the planned stack/region exit point,
stack-region owner, expected same-owner/same-stream relation, command-buffer
continuity requirement, descriptor and command-pool lifetime requirements,
retire timeline migration requirement, and the missing runtime implementation
hook. It does not create a real queue submit or authorize submit elision.
`StackRegionDeferredSubmitRuntimeHookPlan.v0` now decomposes that missing
runtime hook into concrete required capabilities: a region-owned command buffer
or batch, cross-phase recording ownership, retire timeline migration,
descriptor lifetime extension, command-pool lifetime extension, same
stream/queue proof, and host/fence/public/readback blocker status. Current rows
report the hook uninstalled, unable to close or submit a region-owned command
buffer, and missing the stack-owner region command-buffer request hook API as the
first capability blocker.
`StackRegionCommandBufferOwnershipPlan.v0` now records the missing ownership
shape directly: stack-region instance, current command-buffer recording id and
scope, current owner scope, region-owned command buffer and batch presence,
ownership-transfer requirement, cross-phase recording capability, planned
region-exit close/submit capability, timeline/retire point coverage,
descriptor and command-pool lifetime coverage, same stream/queue guarantee, and
the stack-owner request hook status. Current rows report that command-buffer
ownership is still the Vulkan context's per-submit-phase responsibility and no
stack-owner hook exists to request a region-owned command buffer.
`StackRegionCommandBufferRequest.v0` and
`StackRegionCommandBufferRequestResult.v0` now model that missing request API as
a behavior-neutral surface. Rows name the stack-region instance,
requester/owner scope, requested resource type (`command_buffer_or_batch`),
stack-region lifetime, same stream/queue requirement, descriptor and command
pool lifetime requirements, retire timeline ownership, fallback behavior, and
public/final/host/readback policy. The request API surface is present, but the
result is now produced by a minimal runtime API skeleton,
`StackRegionCommandBufferRequestRuntimeApi.v0`, that is callable by diagnostics
and always returns unavailable without allocating, switching, deferring, or
submitting command buffers. The result reports
`request_result_runtime_api_present_unavailable`.
`StackRegionOwnedCommandBufferContract.v0` now records the corresponding object
and lifetime contract explicitly. It is still behavior-neutral: rows name the
stack/region owner scope, command-buffer or batch ownership requirement,
command-pool lifetime, descriptor lifetime, allocator/retire-timeline scope,
planned stack-entry acquire point, planned stack-exit release/submit point, and
public/final/host/readback policy. Current rows fail closed with
`owned_command_buffer_contract_runtime_api_present_result_unavailable`; no
region-owned command buffer is allocated, no submit is deferred, and no
submit-elision behavior is authorized. The top blocker has moved from an absent
API to the first concrete capability behind it:
`missing_command_pool_lifetime_extension`.
`StackRegionCommandBufferLifetimeReservation.v0` is now the next typed
diagnostic request/result surface behind that blocker. It models a stack-region
owner reserving a future region-owned command buffer or batch through a planned
region-exit submit point, with stack/region lifetime, command-pool lifetime,
owner/requester scope, and fail-closed public/final/host/readback policy. The
runtime API skeleton is present and callable by diagnostics, but returns
unavailable without allocating, switching, deferring, or submitting command
buffers. Current rows report
`command_buffer_lifetime_reservation_unavailable` and refine the top blocker to
`command_pool_cannot_extend_beyond_phase_submit`.
`StackRegionCommandPoolLifetimeContract.v0` now models the specific command-pool
lifetime extension required by that reservation. It records the stack-region
instance, current context phase-submit command-pool owner scope, selected
phase-boundary id, requested stack/region lifetime scope, planned region-exit
release point, reservation key, command-pool retention API status, and
command-pool reset deferral status. The contract remains fail-closed and
behavior-neutral: no command pool is retained, no command buffer crosses phase
boundaries, and no submit is deferred. Current rows report
`command_pool_lifetime_contract_unavailable`; the refined implementation
blocker is now `command_pool_retention_implementation_missing`.
`StackRegionExitReleasePoint.v0` now represents the stack/region exit point
that would release a future region-owned command buffer or batch, descriptor
lifetime, allocator/resource retire ownership, and retire timeline. It is
diagnostic-only, but when the stack planned-recording exit submit is observed
it now reports `exit_release_point_runtime_observed_context_submit_preserved`
as a real release anchor. Ordinary phase submits should be interpreted here as
closing/submitting the active command buffer, clearing the recording epoch, and
creating a retire timeline; they are not ordinary raw `vkResetCommandPool`
calls.
`StackRegionExitReleaseOwnershipContract.v0` now names the ownership contract
behind that observed exit point. It records the stack/region owner identity,
command-buffer close/submit ownership, queue submit/timeline ownership,
descriptor release ownership, retire timeline release ownership,
allocator/resource release ownership, and command-pool cleanup/reset
ownership. The contract is diagnostic-only and remains unavailable; selected
rows refine the blocker to
`missing_region_exit_release_ownership_implementation`.
The next architecture direction is `RegionCommandBufferOwnership.v0`, described
in `docs/vulkan/STACK_REGION_COMMAND_OWNERSHIP.md`. That design card defines a
stack-entry acquire and stack-exit release owner for command-buffer leases,
command-pool leases, descriptor generations, temporary resource scope, retire
transfer, and output ownership. It is not a behavior path yet: phase-boundary
submits remain preserved and submit elision remains disabled.
The first scaffold now emits behavior-neutral `RegionCommandBufferOwnership.v0`
records. `stack_entry_acquire` rows now join
`StackRegionCommandBufferAcquireHook.v0` rows and
`RegionOwnedCommandBufferLease.v0`, which record the selected stack-region
instance, boundary, planned region-exit release point, requested stack-region
owner scope, current Vulkan context/phase-submit owner scope, preserved
context phase-submit command-buffer candidate status, unavailable region
command-buffer or batch lease, unavailable command-pool lease, descriptor
lifetime scope request, retire timeline scope request, same stream/queue
requirement, and public/final/host/readback blocker status. The hook snapshots
current stack planned-recording and command-buffer owner state near `Context`.
When stack planned recording is active it can name the preserved context
command-buffer batch candidate, but that candidate is marked not transferable;
behavior remains disabled and no stack-region lease is granted. The candidate
now has a stack-entry lifecycle id that is finalized at stack planned-recording
submit or cancel, while command-buffer close/submit and command-pool lifetime
remain owned by the context phase-submit path.
`StackRegionSingleRecordingPlan.v0` now sits below that hook as the planned
single-region recording scaffold. It is emitted for the selected boundary and
records that execution still uses `context_phase_submit_recording`, phase
boundary submits are preserved, command-buffer execution topology is unchanged,
and borrowed context command-buffer ownership is rejected because
`Context::submit_cmd_to_gpu` still closes/submits the active recording at phase
boundaries.
`stack_exit_release` rows make public/private/captured/requested/final output
release, pending retire transfer, and command-pool reset deferral explicit. The
scaffold proves current behavior is preserved: no submit elision, no deferred
submit, no command-buffer replay, no new queue submit, and no command-pool
reset behavior change.
`StackRegionExitReleaseOwnership.v0` now refines the release rows into concrete
release responsibilities. It records output release classes, pending-retire
transfer, descriptor lifetime release, retire timeline release,
allocator/resource release, command-buffer close/submit ownership,
queue-submit/timeline ownership, and command-pool cleanup/reset ownership. It
is still behavior-neutral and fail-closed: phase submits are preserved, pending
retires remain on the current phase-submit timeline, and the refined blocker is
`missing_command_buffer_close_submit_release_ownership`.
`StackRegionCommandBufferCloseSubmitOwnership.v0` now splits that first
component out as its own behavior-neutral row. It records the current
command-buffer recording id/scope, planned region-exit release point, current
phase-submit close/submit owner, region-exit owner status, and region-owned
command-buffer status. Current selected rows still preserve every phase submit
and now join a real `RegionExitCloseSubmitOwner.v0` owner surface. That owner
record is emitted and proves the current phase-submit close/submit owner is
preserved, but it cannot take region-exit ownership because the command buffer
is still context/phase-submit owned and no region-owned command-buffer or batch
lease is available. The refined blocker is now
`region_owned_command_buffer_lease_unavailable_single_recording_owner_lacks_close_submit_ownership`.
`StackRegionExitCloseSubmitOwnerRequest.v0` and
`StackRegionExitCloseSubmitOwnerResult.v0` are the behavior-neutral request
surface behind that blocker. They model a future stack-exit owner asking to
close and submit a region-owned command buffer or batch. They now feed
`RegionExitCloseSubmitOwner.v0`, which reports queue/timeline, retire,
descriptor-lifetime, and command-pool handoff availability as unavailable while
still creating no queue submit, no deferred submit, no submit elision, and no
command-buffer execution-topology change.
`StackRegionSingleRecordingOwner.v0` is now emitted as the lifecycle surface
under `StackRegionSingleRecordingPlan.v0`. It begins after the pre-stack flush
and is finalized with stack planned-recording submit/cancel, but it is
behavior-neutral: close/submit ownership, command-pool ownership, descriptor
scope, and retire timeline ownership all remain context/phase-submit owned.
The owner record is joined into the acquire hook and
`RegionOwnedCommandBufferLease.v0`, which now fails closed on missing
single-recording owner close/submit ownership rather than on a missing owner
surface.
`StackRegionCommandBufferAcquireHook.v0` now also surfaces the active context
phase-submit command-buffer batch candidate when the planned recording is
owned by the current thread. This is a lease-candidate observation only:
`lease_available=0`, `behavior_enabled=0`, `new_queue_submit_created=0`, and
`authorizes_submit_elision=0` remain enforced, while downstream lease rows keep
failing closed until a real region-owned command-buffer or batch lease exists.
`StackRegionSingleRecordingCanary.v0` is now the first opt-in behavior canary
above that owner surface. It is controlled by
`PYTORCH_VULKAN_STACK_REGION_SINGLE_RECORDING_CANARY=non_capture_residual2_norm1_block1`
and requires the existing barrier canary for the same boundary. The canary uses
a proof warmup pass to populate the selected non-capture
`residual2@0 -> norm1@1` boundary plan, then the second pass may keep the
stack-region command recording open across exactly that phase-boundary submit
and close it at stack exit. Default behavior is unchanged, the older
`PYTORCH_VULKAN_STACK_REGION_SUBMIT_ELISION_CANARY` path remains disabled, and
the canary records `authorizes_submit_elision=0` because it is a
region-owned single-recording experiment rather than a retire-time submit
elision proof. After real `vits_140` evidence showed one Norm1-input barrier
does not preserve the full phase submit, the canary now also fails closed unless
validated barrier coverage spans the pending dispatch range. Focused tests cover
output parity, zero selected submit deferrals under incomplete barrier coverage,
no submit removed outside the selected boundary, live command-buffer ownership,
barrier proof, pending dispatch range proof, and host/final/readback blockers
staying absent.
The first real `vits_140` bridge measurement with this canary removed exactly
one selected submit but failed stack-output bridge sanity, so the result is not
a valid performance improvement. The benchmark harness now marks bridge runs
with failed `vulkan_stack_output_device_bridge_sanity` as
`performance_valid=false` and records
`vulkan_stack_output_device_bridge_sanity_failed` in
`performance_invalid_reasons`. Do not promote this canary or use its timings as
evidence; the next behavior path must be a planned region-owned command-buffer
topology rather than another local phase-submit deferral.
`StackRegionCommandBufferTopologyPlan.v0` now records that topology direction
explicitly. It is behavior-neutral and shows the current execution topology is
still `context_phase_submit_command_buffer_topology_preserved`, while the
requested future topology is a region-owned command buffer or batch from stack
entry to stack exit. The vision stack capture-to-decoder bridge now installs a
`VulkanStackPlannedRegionScope` so graph dumps expose
`vision_stack_decoder_bridge_region`, `VisionBackboneStackContext`,
`vision_stack_output_device_bridge`, and
`vision_stack_capture_decoder_preprocess_plan` instead of graph-level missing
region fields. Rows still fail closed, but the bridge blocker advances to
`planned_region_topology_present_close_submit_still_context_owned`; they do not
remove submits, defer submits, create a queue submit, or switch command buffers.
`StackRegionExitCloseSubmitOwnerRequest.v0`,
`StackRegionExitCloseSubmitOwnerResult.v0`, and
`RegionExitCloseSubmitOwner.v0` now carry the same planned bridge context
fields (`VisionBackboneStackContext`, `vision_stack_output_device_bridge`, and
`vision_stack_capture_decoder_preprocess_plan`) into the close/submit owner
surface. Those rows fail closed with
`planned_region_topology_present_close_submit_still_context_owned` when the
planned region scope is present but command-buffer close/submit still belongs
to the context phase-submit path. This is still behavior-neutral: phase-boundary
submits are preserved and no region-owned command buffer or batch is closed or
submitted.
`StackRegionExitSubmitRuntimePoint.v0` now records the real stack planned
recording exit submit point at `Context::end_stack_planned_recording_and_submit`
while preserving the existing `StackPlannedRecordingSubmit` path. Bridge rows
therefore distinguish the observed preserved exit submit point from region
close/submit ownership: `StackRegionPlannedSubmitPoint.v0` can report
`planned_region_exit_submit_point_runtime_observed_context_submit_preserved`,
and the close/submit owner surface advances to
`region_exit_close_submit_owner_handoff_blocked_preserved_phase_submit_batch_context_owned`
because the preserved phase-submit batch lease is available only as an
accounting/lifecycle lease, not as a region close/submit owner.
Submit-level equivalence rows now consume that preserved runtime exit-submit
point even when the planned region context has already closed and the graph has
exactly one unambiguous `StackPlannedRecordingSubmit` exit row. This moves the
deferred-submit plan status from a synthetic planned-target blocker to
`stack_region_deferred_submit_plan_available_retire_migration_unproven`, with
the top blocker reported as `retire_timeline_migration`. It remains
behavior-neutral: the runtime exit submit is preserved, phase-boundary submits
are preserved, and submit elision stays disabled.
`StackRegionRetireTimelineMigration.v0` is now the typed accounting surface
under that blocker. It records the observed runtime stack-exit submit point,
the selected boundary's pending resource and retire side-effect counts, the
current context-owned retire timeline, the requested region-owned retire
timeline owner, and the pending-retire transfer status. Current rows can report
`retire_timeline_migration_accounting_available_behavior_disabled` and
`pending_retires_transfer_accounting_available_behavior_disabled`, but no
resource lifetime, retire queue, submit, or command-pool behavior changes:
`authorizes_submit_elision=0`, phase-boundary submits remain preserved, and the
next implementation gate is still a real behavior-enabled retire-timeline
handoff under region ownership.
`StackRegionRetireTimelineOwner.v0` is now the matching behavior-neutral owner
surface. Context creates a `ContextStackRegionRetireTimelineOwnerState.v0`
lifecycle id at stack planned-recording entry and finalizes it on submit or
cancel, but the observed states remain context-owned and not transferred:
`retire_timeline_owner_candidate_active_context_owned_not_transferred`,
`retire_timeline_owner_finalized_submit_context_owned_not_transferred`, or
`retire_timeline_owner_finalized_cancel_context_owned_not_transferred`. The
owner row can report migration accounting availability. Generic rows keep
`owner_available=0`, but the stack-exit close-submit owner mode can expose
`owner_available=1` for accounting after the runtime exit-submit owner is
joined. In both cases `transfers_retire_timeline=0`,
`authorizes_submit_elision=0`, and the row fail-closes with
`retire_timeline_owner_behavior_disabled` until a real region-owned retire
timeline handoff is implemented.
`StackRegionPendingRetireTransferPlan.v0` now snapshots the concrete pending
retire source that such a handoff would need to own. The Context reports the
current pending-retire resource count/bytes plus the stack-internal retire
batch count/bytes, and the row compares that source with the submit-level graph
pending set. This is still planning only: rows can distinguish context-pending,
stack-batch, already-consumed-by-preserved-submit, or mismatched sources, but
`transfer_behavior_enabled=0`, `transfers_pending_retires=0`, and
`authorizes_submit_elision=0`.
The Context also records a stack-exit source binding before the preserved
submit path retires the stack-internal batch. When that bound count/byte tuple
matches the graph pending set, the plan reports
`pending_retire_transfer_source_bound_to_region_exit_submit` instead of only
`pending_retire_transfer_source_already_consumed_by_preserved_submit`. This is
source accounting, not a resource transfer.
The source binding now also records the preserved phase-boundary pending-retire
set before that submit consumes it. When that earlier source covers the graph
pending set, the plan reports
`pending_retire_transfer_source_complete_at_preserved_phase_submit` rather
than treating it as a region-exit source. When the preserved phase-submit
source is a superset of the selected graph-pending set, it reports
`pending_retire_transfer_source_superset_at_preserved_phase_submit`; that row
is still fail-closed because the extra source resources remain owned by the
preserved phase-submit path, not by a region-exit owner. Partial source
bindings remain explicit through
`pending_retire_transfer_source_partially_bound_to_region_exit_submit` or
`pending_retire_transfer_source_partially_bound_to_preserved_phase_submit` plus
the bound and missing count/byte tuples. This keeps the next ownership blocker
visible without transferring pending retires or enabling submit elision.
The opt-in
`PYTORCH_VULKAN_STACK_REGION_PENDING_RETIRE_TRANSFER_OWNER=stack_internal_until_stack_exit`
mode lets the stack-exit stack-internal retire batch source supersede the
earlier preserved phase-submit source for diagnostics only. It preserves all
submits and does not move resources. On the current selected synthetic
`residual2@0 -> norm1@1` boundary, that mode reports only a partial
stack-exit source, so the owner still fail-closes on
`pending_retire_transfer_source_incomplete`.
`PYTORCH_VULKAN_STACK_REGION_BATCH_QKV_RETIRES=1` expands the existing
stack-internal batch predicate to the separately proven QKV stack-temp class.
With both opt-ins enabled, the selected synthetic boundary's stack-exit source
now covers the graph-pending bytes, but raw resource-count coverage remains
partial because metadata/uniform bookkeeping entries are not stack-internal
retire-batch targets. The transfer-plan row reports those typed graph entries
separately through `graph_bookkeeping_excluded_resource_count/bytes`, derives
`graph_transfer_required_resource_count/bytes`, and records
`source_coverage_after_bookkeeping_exclusion_status`. This is accounting only:
the main `source_match_status` remains the raw source match, and the owner does
not treat a count/byte superset after bookkeeping exclusion as transferable
source identity. The owner remains fail-closed with
`transfers_pending_retires=0` and `authorizes_submit_elision=0` until per-entry
source ownership is proven.
The transfer-plan row now also carries per-entry allocation identity for this
source check. `StackRegionBoundarySubmitPlan.v0` publishes a
`pending_allocation_signature` for graph pending resources, and
`StackRegionPendingRetireTransferPlan.v0` compares the transfer-required
non-bookkeeping entries against the source bound at region exit by
allocation id, generation, byte range, resource class, count, and bytes. The
result is reported through
`graph_transfer_required_identity_resource_count/bytes`,
`graph_transfer_required_allocation_signature`,
`region_exit_bound_source_allocation_signature`,
`region_exit_bound_missing_transfer_required_identity_count/bytes`, and
`source_identity_match_status`. Malformed graph or source signatures are
reported as explicit source-identity failures rather than being treated as
empty transfer sets. This remains diagnostics only: exact or superset identity
coverage does not change `source_match_status` and does not authorize
pending-retire transfer or submit elision. Conversely, the pending-retire owner
surface now requires exact or source-superset identity coverage before it can
report source availability; count/byte coverage without per-entry identity
stays fail-closed as source-incomplete accounting.
Source identity snapshots are retained per stack-region source id, which maps
to the current stack-region instance id in the bridge diagnostics. This prevents
later warm/timed instances from overwriting an earlier instance's source
signature at report time. The current stack-exit batch source is still not
identity-equivalent to the selected phase-submit pending graph set: the source
id is instance-correct, but the allocation identities differ, so the owner stays
blocked by source-incomplete accounting.
The row also classifies that mismatch through `source_identity_mismatch_axis`
and overlap counters for exact identity, same allocation/range, and same
resource class. In the current selected stack-exit batch path, exact and
allocation/range overlap are zero while resource-class overlap is nonzero, so
the mismatch is reported as
`source_identity_mismatch_same_class_different_allocation_set`. This
distinguishes a real different-source-set blocker from malformed signatures or
resource-class taxonomy drift; the current interpretation is that the stack-exit
batch source is not the selected phase-submit pending graph set.
The transfer row now also retains the preserved phase-submit source snapshot
for the same stack-region instance and compares it against the graph pending
set. It reports the preserved source id, state, status, resource count/bytes,
allocation signature, identity status, and missing identity counts. Current
selected rows can show exact or source-superset preserved-phase identity
coverage while the stack-exit batch source still mismatches. This proves the
source exists before the preserved phase submit consumes it, but it remains
`context_owned_not_transferred`; the row does not move ownership away from that
preserved submit or enable deferred submit behavior.
`StackRegionPendingRetireTransferOwner.v0` now consumes that transfer-plan row
and records the region-owner handoff decision that would be required before a
future close/submit owner can take retire entries away from the preserved
context submit path. It is an owner surface, not a transfer implementation:
generic rows can report transfer-plan accounting and source matching while
keeping `owner_available=0`. When the stack-exit owner path has a concrete
source match, the row can expose `owner_available=1` for accounting only. It
still keeps `behavior_enabled=0`, `transfers_pending_retires=0`, and
`authorizes_submit_elision=0`. When the transfer plan and source are otherwise
complete, the row fail-closes on
`pending_retire_transfer_owner_behavior_disabled`; when the source is
available only at the preserved phase submit, it fails closed on
`pending_retire_transfer_preserved_phase_submit_handoff_behavior_disabled` and
reports
`pending_retire_transfer_owner_preserved_phase_submit_handoff_available_behavior_disabled_fail_closed`;
the owner row also emits explicit handoff API-present, candidate-available,
behavior-enabled, and transfer flags, all keeping behavior disabled and
`transfers_pending_retires=0`.
`Context` now has an empty-by-default stack-region pending-retire handoff batch
with stack-entry clear, stack-exit retire, cancel restore, forced-clear cleanup,
and source-signature participation. By default no producer moves entries into
that batch. The opt-in
`PYTORCH_VULKAN_STACK_REGION_PENDING_RETIRE_TRANSFER_OWNER=preserved_phase_submit_handoff`
canary moves only exact allocation id/generation/byte-range/resource-class
matches from the live phase-boundary target signature into that batch. The
phase-boundary submit is preserved, submit elision remains disabled, and stack
exit retires the handoff batch only under the observed stack-exit submission
timeline; cancel restores entries to normal pending-retire storage. The first
canary classified the remaining exact-identity gap as
`source_identity_missing_capture_sensitive_stack_activation_count/bytes` with
`source_identity_mismatch_axis=missing_capture_sensitive_stack_activation`.
The follow-up canary moves that activation under the same opt-in, but only when
exact
allocation id/generation/byte-range/resource-class identity matches and the
pending retire carries residual2 -> next-block norm1 provenance with no
public/final/requested/alias/runtime-input/output escape. The row then reports
`pending_retire_transfer_source_identity_required_entries_present_source_superset`,
zero missing capture-sensitive identities, and
`pending_retire_transfer_owner_preserved_phase_submit_handoff_transferred_no_submit_elision`.
Submit elision remains disabled; region-exit ownership now fails closed on the
next owner layer, currently `retire_timeline_owner_behavior_disabled`.
`PYTORCH_VULKAN_STACK_REGION_RETIRE_TIMELINE_OWNER=stack_exit_close_submit`
enables the next opt-in owner handoff: when migration accounting and the
stack-exit close-submit owner are both available, the retire timeline owner
reports `retire_timeline_owner_transferred_to_stack_exit_close_submit_no_submit_elision`.
This still does not authorize submit elision. With reset-deferral,
close-submit, pending-retire handoff, and retire-timeline canaries enabled, the
joined region-exit ownership row reaches
`region_exit_ownership_transfer_complete_fail_closed` and stops on
`region_exit_ownership_transfer_authorization_disabled`.
When the source identity is incomplete, including bookkeeping-excluded
count/byte coverage without per-entry source identity, it fails closed on
`pending_retire_transfer_source_incomplete`;
and when the transfer plan is blocked, it propagates the plan blocker instead
of hiding it behind close-submit ownership.
That owner handoff status is now threaded into
`StackRegionExitReleaseOwnership.v0`, `RegionCommandBufferOwnership.v0`, and
`StackRegionDeferredSubmitRuntimeHookPlan.v0` as a separate owner-release
status. The older transfer-source status remains available, but downstream
release/command ownership reports can now show whether the missing piece is the
transfer source, the retire timeline owner, or the region pending-retire owner
handoff.
The owner row is also anchored to
`ContextStackRegionPendingRetireTransferOwnerState.v0`; stack planned recording
creates a lifecycle id at stack entry and finalizes it on submit or cancel. All
current lifecycle states remain context-owned and not transferred, so the row is
proven to be lifecycle-backed without changing ownership or moving resources.
`RegionCommandBufferOwnership.v0` now carries this through explicit
stack-entry/stack-exit lifecycle fields: the planned stack-region scope is
observed, the preserved phase-submit batch lifecycle is recorded, but actual
region command-buffer acquire/release remains `0`, preserved phase-submit counts
are recorded, command-pool reset is not deferred to region release, and actual
submit elision remains `0`.
The rows now also expose `ContextRegionCommandBufferOwnershipState.v0`, a
Context-owned acquire/release lifecycle id and status created at stack planned
recording entry and finalized at stack exit submit or cancel. This anchors the
stack-entry acquire and stack-exit release records to runtime stack scope while
still reporting that the command buffer remains context/phase-submit owned.
The current row contract makes that fail-closed ownership explicit:
`region_owned_close_submit_available=0`,
`close_submit_ownership_status=close_submit_still_context_phase_submit_owned`,
`command_pool_reset_ownership_status=command_pool_reset_still_context_owned_not_deferred`,
`descriptor_lifetime_ownership_status=descriptor_lifetime_still_context_owned_not_releasable`,
and
`retire_timeline_ownership_status=retire_timeline_still_context_owned_not_transferred`.
This remains fail-closed and behavior-neutral: no submit is removed, deferred,
batched, replayed, or newly created.
`StackRegionSingleRecordingCanary.v0` now mirrors that ownership state in its
own selected-boundary rows: active planned-recording scopes report the preserved
phase-submit batch lease as available for accounting, then fail closed on
`region_exit_close_submit_owner_handoff_blocked_preserved_phase_submit_batch_context_owned`.
This only aligns the canary readiness report with the ownership rows; it does
not authorize submit elision, make the batch a region close/submit owner, or
turn selected-boundary barrier proof into permission to skip a submit. The
single-recording canary now treats the actual Norm1 input barrier as
selected-boundary value-preservation evidence rather than requiring one barrier
per pending dispatch/bookkeeping row, so rows with complete selected-boundary
proof advance to the `region_exit_ownership_transfer_incomplete` guard.
The behavior guard also has an explicit close/submit-owner capability check, so
even after a future barrier-coverage proof becomes complete the canary remains
fail-closed until a real region exit close/submit owner exists.
That capability check is now driven by Context-owned lifecycle state rather
than by a standalone hardcoded unavailable helper. Stack planned recording
creates a live close/submit owner lifecycle id, keeps it in the
preserved-phase-submit-batch-only state while the region is active, and records
that state in `StackRegionSingleRecordingCanary.v0` rows through
`ContextStackRegionCloseSubmitOwnerState.v0`. The canary also requires a
separate behavior-enabled bit, so a lifecycle state cannot authorize submit
elision by itself: `actual_elided_submit_count=0` and phase-boundary submits
are preserved. With
`PYTORCH_VULKAN_STACK_REGION_CLOSE_SUBMIT_OWNER=preserved_phase_submit_batch`,
the live canary can observe the preserved phase-boundary close/submit lifecycle
as state `7` /
`region_exit_close_submit_owner_active_preserved_phase_submit_close_submit_available`
and report the preserved-batch handoff blocker. This is accounting over the
existing phase-boundary submit only: it does not make that submit a transferable
region-exit owner. Without a real region-owned close/submit owner it reports
`region_exit_close_submit_owner_handoff_blocked_preserved_phase_submit_batch_context_owned`.
The live guard now has a separate close-submit authorization input, currently
passed as `0`, so close-submit owner availability cannot become submit removal
by itself. Both paths remain fail-closed until a real region-owned
close/submit owner replaces the preserved batch accounting state and explicitly
authorizes submit elision.
The same lifecycle source is now threaded through the ownership row chain:
`StackRegionSingleRecordingOwner.v0`,
`StackRegionCommandBufferAcquireHook.v0`,
`RegionOwnedCommandBufferLease.v0`,
`StackRegionExitCloseSubmitOwnerRequest.v0`,
`StackRegionExitCloseSubmitOwnerResult.v0`,
`RegionExitCloseSubmitOwner.v0`,
`StackRegionCommandBufferCloseSubmitOwnership.v0`, and
`RegionCommandBufferOwnership.v0`. This is row-schema propagation only. Those
rows still report behavior disabled, no submit authorization, preserved
phase-boundary submits, and unavailable region close/submit ownership.
`RegionCommandBufferOwnership.v0` now makes the stack-entry and stack-exit
record distinction explicit: acquire rows carry
`stack_entry_acquire_record_emitted=1` and release rows carry
`stack_exit_release_record_emitted=1`, with statuses that say whether the
planned region scope was observed. The actual ownership bits remain
`region_command_buffer_ownership_acquired=0` and
`region_command_buffer_ownership_released=0`, and the owner-status fields
continue to report that the command buffer is still context-owned. This is
behavior-neutral and does not authorize submit elision.
Context now also owns a separate
`ContextRegionCommandBufferOwnershipState.v0` lifecycle id/state for that
acquire/release observation. Its active, submitted, and canceled states are
explicitly named as context-owned fail-closed states; they do not imply a
region-owned command buffer, command pool, descriptor scope, or retire timeline.
`StackRegionCommandPoolRetentionRequest.v0` and
`StackRegionCommandPoolRetentionResult.v0` are the fail-closed request/result
surface behind the retention blocker. They model a stack-region owner asking to
retain the current command pool across phase boundaries until the planned
region-exit release point. The runtime API is present for diagnostics only and,
when the observed stack planned-recording exit submit is available, records
`command_pool_retention_result_context_pool_retained_until_observed_release_point`.
This is context-owned retention, not a region-owned command-pool lease: it does
not defer a reset, allocate or switch command buffers, create a queue submit, or
authorize submit elision. Current selected rows now refine the top blocker to
`command_pool_reset_deferral_implementation_missing`, with
`command_pool_reset_deferral_proof_unavailable_reset_deferral_implementation_missing`
reported as the reset-deferral proof status.
`StackRegionCommandPoolResetDeferralProof.v0` is the corresponding
behavior-neutral proof surface. It records the stack-region instance, current
context phase-submit owner scope, the recording epoch consumed at the selected
phase submit, planned region-exit release/reset point, linked command-pool
retention result, and descriptor, command-buffer, and retire-timeline lifetime
blockers. The proof currently returns unavailable and complete=false; it does
not defer a reset or retain a command pool. Current selected rows fail closed
because the context command pool is retained only by the preserved stack-exit
submit path and no region-owned reset-deferral implementation exists yet. The
proof can now report
`command_pool_reset_deferral_proof_complete_context_pool_retained_until_release_point`
when the context-retained command pool is observed through the stack-exit
release point. That is a proof of current context retention only; it is not a
region-owned reset-deferral owner. The command-pool lifetime contract now uses
`command_pool_lifetime_context_retained_not_region_owned` for that state instead
of blaming the reset-deferral proof layer.
That reset-deferral proof status and top blocker now flow into
`StackRegionExitCloseSubmitOwnerRequest.v0`,
`StackRegionExitCloseSubmitOwnerResult.v0`, and
`RegionExitCloseSubmitOwner.v0` rows. When a preserved phase-submit batch is
otherwise observed, close-submit owner diagnostics now fail closed on the more
specific reset-deferral owner blocker instead of only reporting the generic
preserved-batch-only blocker. This is classification only: no submit is
deferred, elided, closed, or transferred to a region owner.
When the reset-deferral owner accounting surface is present but behavior is
disabled, the close-submit request/result rows can expose accounting
availability while still reporting
`region_exit_close_submit_owner_preserved_batch_blocked_by_reset_deferral_behavior_disabled`.
The downstream `RegionExitCloseSubmitOwner.v0` surface remains unavailable for
execution and reports
`region_exit_close_submit_owner_accounting_available_behavior_disabled_fail_closed`.
An opt-in close-submit owner canary is available through
`PYTORCH_VULKAN_STACK_REGION_CLOSE_SUBMIT_OWNER=preserved_phase_submit_batch`.
It only applies after reset deferral has no blocker. It can report the active
preserved phase-submit close/submit lifecycle state, behavior availability,
and an available close-submit owner surface, but still reports
`region_exit_close_submit_owner_authorizes_submit_elision=0` and
`region_exit_close_submit_owner_handoff_blocked_preserved_phase_submit_batch_context_owned`.
`PYTORCH_VULKAN_STACK_REGION_CLOSE_SUBMIT_OWNER=stack_exit_close_submit`
is the next behavior-neutral mode: the actual stack-exit close/submit scope can
report lifecycle state `4` /
`region_exit_close_submit_owner_active_region_owned_close_submit_available`
and a close-submit handoff status of
`region_exit_close_submit_owner_handoff_available_stack_exit_close_submit_owner`.
Earlier live phase-boundary canary rows still report the preserved-batch
context-owned blocker. The submit-level graph, however, now joins the
stack-exit runtime-point owner back into selected-boundary
`RegionExitOwnershipTransfer.v0` rows, so those rows can report
`runtime_close_submit_owner_joined=1`, close-submit ownership complete, and
then fail closed on the next incomplete owner. Phase-boundary submits remain
preserved and `authorizes_submit_elision=0`.
`StackRegionCommandPoolResetDeferralOwner.v0` is now the behavior-neutral owner
surface between that proof and close-submit ownership. It records whether a
region-owned command-pool reset-deferral owner exists, whether reset deferral is
enabled, and whether command-pool reset would be deferred. Current proof-complete
rows can expose `owner_available=1` for accounting, but still report
`reset_deferral_behavior_enabled=0`, `defers_command_pool_reset=0`, and
`authorizes_submit_elision=0`; the selected blocker is
`command_pool_reset_deferral_owner_behavior_disabled` until a real reset
deferral behavior gate is implemented. The row also carries
`ContextStackRegionCommandPoolResetDeferralOwnerState.v0` lifecycle id/state
from stack entry through submit or cancel finalization, but all observed states
remain context-owned and not deferred.
An opt-in reset-deferral owner canary is available through
`PYTORCH_VULKAN_STACK_REGION_RESET_DEFERRAL_OWNER=context_retained_release_point`.
When the context-retained proof is complete, this canary sets the owner row to
`reset_deferral_behavior_enabled=1` and `defers_command_pool_reset=1` while
keeping `authorizes_submit_elision=0`. It does not remove submits, create
deferred submits, or transfer close-submit ownership.
`RegionExitOwnershipTransfer.v0` is now the aggregate handoff row above the
close-submit owner, command-pool reset-deferral owner, pending-retire transfer
owner, retire-timeline owner, and stack-exit release-point surfaces. It reports
whether those
component surfaces can be joined for the selected stack-region instance and
phase boundary, then computes a stricter ownership-completion predicate over
the close-submit owner, reset-deferral owner, pending-retire transfer owner,
retire-timeline owner, and exit release point. Preserved phase-submit batch
accounting does not count as completed close/submit ownership. Current rows
still keep `ownership_transfer_complete=0`, `submit_elision_enabled=0`,
`deferred_submit_enabled=0`, `authorizes_submit_elision=0`, and
`phase_boundary_submits_preserved=1`. Rows can distinguish joined accounting
from missing or incomplete close-submit ownership, reset-deferral ownership,
pending-retire transfer ownership, retire-timeline ownership, runtime exit
submit point, or public/final/host/readback output-boundary blockers. This is
still behavior-neutral and does not transfer command-buffer, command-pool,
descriptor, retire, or output ownership.
`StackRegionSingleRecordingCanary.v0` now consumes that aggregate transfer as a
live guard. Its rows include the transfer status, top blocker, accounting
joined bit, completion bit, and component lifecycle state for close-submit,
reset-deferral, retire-timeline, and pending-retire transfer ownership. The
guard remains fail-closed with
`region_exit_ownership_transfer_incomplete` after earlier proof/barrier gates
until a future region-exit ownership transfer implementation can set
`region_exit_ownership_transfer_complete=1` and explicitly authorize submit
elision. Current rows still keep `submits_removed=0`,
`deferred_submit_enabled=0`, and
`region_exit_ownership_transfer_complete=0`.
`StackRegionCommandBufferRequestHookPlan.v0` joins that request/result pair to
the planned stack-entry and stack-exit callsites. The hook is not installed,
authorizes no behavior, and refines the top blocker to
`missing_region_exit_release_ownership_implementation` through the exit-release
ownership contract.
`StackBoundaryValuePreservationContract.v0` is now the behavior gate that a
future submit-elision canary must satisfy before removing even one selected
phase-boundary submit. The design lives in
`docs/vulkan/STACK_BOUNDARY_VALUE_PRESERVATION.md`. The latest one-image
`vits_140` bridge graph with the existing barrier-only canary classifies all
selected `residual2@0 -> norm1@1` rows as
`barrier_ready_but_submit_proof_incomplete`, not canary-ready. The single
missing semantic proof is now command-pool retention/reset-deferral support for
a future region-owned command buffer or batch. That capability is part of the
broader owned-command-buffer contract needed to preserve the current phase
submit's execution, timeline, and retire semantics. The current context path now
proves retention through the observed stack-exit submit anchor, but behavior
still fails closed until region-owned reset deferral exists.
`StackRegionDeferredSubmitPlan.v0` now records the future architecture plan
that would be needed to use that proof: a region-owned command-buffer or batch
kept live until a later planned stack submit with equivalent execution
visibility, timeline signaling, and retire semantics. The plan records the
phase-submit key, current mandatory reason, later submit-point availability,
same stream/queue and same region-owner status, retire migration, descriptor
and command-pool lifetime risk, host/fence/public blockers, and the top
migration blocker. Current `vits_140` rows report a synthetic planned
region-exit target but fail closed because the region-owned command-buffer
implementation, retire migration, and descriptor/command-pool lifetime coverage
remain unimplemented. They keep
`stack_region_deferred_submit_authorizes_submit_elision=0`.

`ExecutionContracts.*` is the shared contract table for the current bounded
operator-family envelopes. `ExecutionContracts.h` remains the public umbrella
API; implementation is now split across:

- `aten/src/ATen/native/vulkan/planning/ExecutionContracts.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContracts.cpp`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractDiagnostics.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractDiagnostics.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsAttentionProbabilityMaterializationSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsBatchNormInference.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsBatchNormInferenceMaterializedSpec.h`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsBatchNormInferenceSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsChannelCat.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsChannelCatSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsDiffusionSDPA.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsDiffusionSDPASpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsElementwiseBroadcast.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsElementwiseBroadcastSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsEmbeddingLookup.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsEmbeddingLookupSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsGQARepeat.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsGQARepeatSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsKVCacheAppend.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsKVCacheAppendInitialSpec.h`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsKVCacheAppendSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsLinearGeluBridge.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsLinearGeluBridgeSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsMaskedTinySDPA.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsMaskedTinySDPASpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsNoOverlapConvTranspose2D.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsNoOverlapConvTranspose2DSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsSafeViewReshape.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsSafeViewReshapeAliasSpec.h`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsSafeViewReshapeSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsSDPAExecutionPolicy.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsSDPAExecutionPolicySpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsSDPAScoreSoftmax.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsSDPAScoreSoftmaxSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsSmallMetadataPaddedConv2D.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsSmallMetadataPaddedConv2DSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsSmallSpatialPointwiseConv.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsSmallSpatialPointwiseConvSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsTokenPrefixCatAdd.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsTokenPrefixCatAddSpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsTransformerGQASDPA.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsTransformerGQASDPASpec.h`
- `aten/src/ATen/native/vulkan/planning/ExecutionContractsVisionSelfAttentionSDPA.cpp`
- `aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsVisionSelfAttentionSDPASpec.h`

The table owns finite tuples/envelopes with `ExecutionContractMetadata` for
contract name, family, tuple id, evidence id, guard id, fallback policy, and
materialization policy. Some rows are still exact and temporary; they are
allowed only as guarded contract rows while generated parity/negative coverage
is built. Every current live contract name has JSON spec, ShapeEnvelope, and
generated C++ helper coverage; remaining exact-row policy debt is tracked as
temporary exceptions rather than as untracked live-contract debt.
`BatchNormInferenceContract`, `ChannelCatContract`,
`EmbeddingLookupContract`, `GQARepeatContract`, `KVCacheAppendContract`,
`LinearGeluBridgeContract`, `NoOverlapConvTranspose2DContract`, and
`SafeViewReshapeContract`, `SmallMetadataPaddedConv2DContract`,
`SmallSpatialPointwiseConvContract`, `MaskedTinySDPAContract`,
`ElementwiseBroadcastContract`, and `TransformerGQASDPAContract`,
`VisionSelfAttentionSDPAContract`, `DiffusionSDPAContract`,
`DiffusionCrossAttentionContract`, `SDPAExecutionPolicyContract`, and
`SDPAScoreSoftmaxContract`, and `TokenPrefixCatAddContract` are split into
family-specific sources. The former
score-softmax allowlist is now a named, metadata-backed finite contract for
float rank-3 square score tensors with heads `{1, 5}` and sequence
`{504, 640}`. `ExecutionContracts.cpp` now owns the shared metadata
completeness helper rather than an SDPA-specific route-policy bucket.

Contract admission now has proof-carrying governance in
`docs/vulkan/CONTRACT_VALIDATION.md`. The checked-in accepted-row manifest
`test/vulkan_contract_proofs/accepted_contract_rows_manifest.json` records the
generated admission surface and dependency digests for JSON specs, generated
C++ helpers, and known high-risk matcher/route/transition sources. The proof
ledger `test/vulkan_contract_proofs/contract_proof_manifest.json` currently
covers the highest-risk bounded contracts:
`SmallSpatialPointwiseConvContract`, `PatchEmbedFloatBufferConvRoute`,
`PatchEmbedFeatureMapToTokensContract`, `TokenPrefixCatAddContract`, and
`AttentionProbabilityMaterializationContract`. The comparison tool
`tools/vulkan_contract_codegen/compare_contract_admission.py` reports admitted
row deltas, cardinality increases, exact-row debt changes, and stale dependency
digests; it is governance-only and does not change runtime route behavior.

`TokenPrefixCatAddContract` covers the bounded rank-3 prefix-token concat plus
position-add envelope observed in DAv2 token preparation:
`prefix=[1,1,C]`, `tokens=[1,N,C]`, `pos/out=[1,N+1,C]`,
`C in {384,768,1024}`, and
`N in {150,260,600,620,1350,1380,2400,2440,3750,3850}`. The generic
`vulkan_prepack::token_prefix_cat_add` route writes a real contiguous Vulkan
output; the benchmark owner path may call it only when this exact bounded
pattern is present.

`AttentionProbabilityMaterializationContract` is now the first formal
transition-contract spec and log-attribution target, but not a production
admission path. The ShapeEnvelope sparse-rowset fixture
`test/vulkan_contract_specs/attention_probability_materialization_contract.json`
records softmax-probability to value-BMM materialization evidence for rank-3
float rows. Nine Lotus-derived rows and the six existing low-resolution
`VisionSelfAttentionSDPAContract` probability rows `[BH,T,T]` with
`BH in {6,12,16}`, `T in {151,261}`, and value dim `64` are now direct-safe
evidence. The vision rows skip the probability clone only when the existing
VisionSelfAttention SDPA policy and the direct-safe transition row both match
the live zero-offset Vulkan buffer layout. The Lotus `[10,126,126]` row remains
marked `vulkan_clone_probability_before_value_bmm`. Transition logging
classifies remaining matching `aten::_softmax -> clone.buffer_to_buffer` events
as `required_correctness_materialization` / `semantic_materialization` with
`producer_contract=AttentionProbabilityMaterializationContract` and
`consumer_contract=DecomposedAttentionProbabilityToValueBmm`.

`ExecutionContractDiagnostics.h/.cpp` define the first opt-in contract
admission diagnostic surface. `PYTORCH_VULKAN_CONTRACT_ADMISSION_LOG=<path>`
emits JSONL `vulkan_contract_admission` events with stable contract metadata,
`outcome`, `phase`, `predicate`, `reason_code`, and `source` fields. This log
is separate from `PYTORCH_VULKAN_OP_HIT_LOG` and from tensor provenance/value
traces: tensor provenance records metadata for accepted output producers,
while admission diagnostics record candidate accept/reject decisions and the
first predicate failure seen by a wired matcher. The current MVP is wired to
`ElementwiseBroadcastContract`, `BatchNormInferenceContract`, and both
`SafeViewReshapeContract` direct-buffer rows:
`ViewMaterializedDirectBuffer` and `ReshapeAliasDenseBufferDirect`; do not
infer that every contract emits admission diagnostics yet.
`contract_spec_utils.py --admission-diagnostics-census` records this as three
wired contracts, five wired spec rows, and three source files while validating
the JSONL payload fields and accept/reject hook presence. The current
ElementwiseBroadcast phases are `generated_options`, `generated_bounds`,
`generated_relationship`, and `admitted`; the current reason codes are
`layout_mismatch`, `dtype_mismatch`, `self_rank_out_of_bounds`,
`other_rank_out_of_bounds`, `attribute_mismatch`, `broadcast_incompatible`,
and `matched`. BatchNorm adds direct and materialized row diagnostics with
`generated_options`, `generated_relationship`, `handwritten_policy`,
`materialization_policy`, and `admitted` phases for options, feature-count,
optional-parameter, storage/materialization, and accept decisions.
SafeViewReshape direct-view diagnostics add generated rank/storage/product and
last-dim rejects plus the handwritten output-stride/materialized-view policy
reject. SafeViewReshape reshape-alias diagnostics add generated rank/storage
offset/product and last-dim rejects plus handwritten dtype, storage, and dense
stride policy rejects.

`TransitionContracts.h/.cpp` and `TransitionPlanner.h/.cpp` introduce a
behavior-neutral transition-contract skeleton for producer/consumer edges after
kernel admission. `PYTORCH_VULKAN_TRANSITION_LOG=<path>` now emits JSONL
`vulkan_transition` events for classified observations such as device-device
copies, host uploads, readbacks, fallback materialization, layout
materialization, and metadata-view creation. The initial taxonomy lives in
`docs/vulkan/TRANSITION_CONTRACTS.md`; unknown reasons are intentionally
visible/countable while follow-up tasks add precise producer/consumer proof.
`AttentionProbabilityMaterializationContract` is the first named transition
contract attached to real events. This skeleton does not remove copies, defer
submits, alter fallback/readback policy, or broaden accepted shapes.
`HostUploadTransitionContract`, `MetadataViewTransitionContract`,
`FinalReadbackContract`, `IntermediateReadbackTransitionContract`,
`SafeContiguousMaterializationContract`, `FallbackMaterializationContract`, and
`LayoutRepackTransitionContract` now provide schema-only source-of-truth
buckets for existing transition-log evidence. The covered reasons are
`required_host_upload`, `metadata_view_only`, `required_final_readback`,
`unexpected_intermediate_readback`, `required_contiguous_materialization`,
`fallback_materialization`, and `required_layout_repack`. The five-model
validation collector loads these checked-in specs before reporting missing
transition contract buckets, so matching upload, metadata-view, final
readback, intermediate readback, safe contiguous materialization, fallback
materialization, and layout-repack events are counted without requiring
producer/consumer contract fields in old logs. These specs are
classification-only and do not change uploads, metadata-view creation, copies,
submit policy, fallback, readback, materialization, layout repack, or route
legality. The current DAv2 transition-reason census has no observed
transition reason bucket left without a source-of-truth spec.
`ConvWeightLayoutRepackTransitionContract` is the first specific
producer/consumer refinement inside `fallback_materialization`: it classifies
`vulkan_prepack::conv2d_context -> vulkan_weight_cpu_materialization` as a
value-bearing legacy conv2d weight repack readback. The log now records source
tensor metadata and a shader-packed destination target when transition logging
is enabled, but the CPU materialization, readback counters, explicit
`Conv2dPackedContext::unpack()`, pickle semantics, and route behavior are
unchanged.

`PatchEmbedFloatBufferConvRoute` is a bounded execution-plan slice for
kernel-14/stride-14 float patch-embed conv rows with input `[1,3,H,W]`,
`(H,W)` in
`{(140,210),(182,280),(280,420),(280,434),(420,630),(420,644),(560,840),(560,868)}`,
weight `[C,3,14,14]`, and
`C in {384,768,1024}`. It uses the existing `conv2d_buffer_float` path to avoid
the legacy value-bearing conv weight CPU repack/readback for those rows while
preserving the legacy path for adjacent negatives. The route now consumes the
bounded non-direct normalized input metadata view through the generic
float-buffer conv metadata UBO instead of first materializing it to a direct
buffer. This removes the route-local patch-embed input copy without adding
host staging or a new shader.

`PatchEmbedFeatureMapToTokensContract` is the bounded layout-transition
contract for the Vulkan-resident patch-embed feature map produced by that
route. It covers rank-4 float width-packed buffer feature maps
`[1,C,H,W] -> [1,H*W,C]` for `C in {384,768,1024}` and feature spatial pairs
`(H,W) in {(10,15),(13,20),(20,30),(20,31),(30,45),(30,46),(40,60)}`. The
benchmark token-preparation path may call the generic
`vulkan_prepack::patch_embed_feature_map_to_tokens` wrapper only for that exact
contract and only when patch-embed normalization is identity. The wrapper uses
the existing buffer feature-map-to-tokens kernel and keeps unsupported ranks,
dtypes, storage offsets, layout classes, channels, and spatial pairs guarded
rather than falling back through CPU. The observed `(40,62)` feature-map case
remains guarded because the downstream `TokenPrefixCatAddContract` does not
yet cover token count `2480`.

`PointwiseConvInputLayoutTransitionContract` is a schema-only proof contract
for pointwise-conv input descriptor-view legality. It records that
storage-offset-zero width-packed rows can use the existing
`FloatBufferPointwise1x1AsLinear` descriptor-view path, while nonzero
storage-offset token-slice metadata views remain on the generic pointwise path
until descriptor-view parity or an explicit layout transition is proven. This
does not broaden `SmallSpatialPointwiseConvContract`, select as-linear for
token-slice rows, or add materialization.

The five-model validation collector now also emits
`execution_plan_evidence` v0 for existing conv, pointwise-conv, and linear
model-suite counter snapshots. This normalizes observed plan-key-like fields
such as selected route/kernel labels, shapes, convolution attrs, linear
dimensions, direct-buffer/packed-weight flags, prepack/upload submit counters,
copy/readback/submit/retire context, and current plan/route counter arrays.
The evidence is reporting-only: it is not a plan cache, not an optimizer, and
does not change route selection, shader selection, fallback/readback behavior,
or accepted shapes.
The collector can also ingest optional `stack_graph_json` sidecars and summarize
`StackRegionDependencyGraph.v0` evidence per row: dispatch/resource/dependency
counts, single-recording canary guard reasons, submit-removal counts, and
pending-retire source coverage buckets. This is still reporting-only and does
not mean the graph system is five-model validated unless fresh graph sidecars
exist for those rows. Rows now also carry a graph coverage status so reports can
distinguish an available sidecar, a missing configured sidecar, stack lifetime
evidence without graph evidence, a row blocked before graph collection, and rows
with no observed stack-region evidence.

The current local tree also has a submit-origin diagnostic split for
CPU-to-Vulkan float-buffer conv prepack uploads. That split keeps true tensor
CPU readbacks classified separately and applies the tiny-old-path pending
handling only to the fenced conv prepack upload path. Recent stability work
keeps the prepack-retire drain policy scoped to float-buffer conv prepack
uploads and preserves real tensor CPU readback behavior and diagnostics.

`region_lifetime_submit_attribution_snapshot()` adds behavior-neutral
submit-pressure attribution for `retire_queue_drain` and
`explicit_synchronize` origins. It records phase, callsite, pending retire
counts/bytes, resource-role signatures, stack lifetime/provenance fields, and
allocation generation/range proof where available. The snapshot is diagnostic
only: it does not defer submits, batch retire entries, change final readback
semantics, or alter route/shape admission.

`docs/vulkan/CAPABILITY_PROFILES.md` and
`docs/vulkan/capability_profiles.json` define the first capability-profile
harness. Profiles are reduced feature masks intersected with the live adapter;
they are not GPU emulation and must not route by profile or GPU-family name.
Focused canaries cover manifest shape and C++ ID parity, non-emulation docs,
minimum-profile runtime-policy feature masking, minimum-profile compiled-session
layout clamping, and minimum-profile SDPA qtile admission to the shared path
instead of the subgroup path.

## Coverage Corpus

The five-model corpus is:

- DAv2: primary vision stack-owner and region-planning signal.
- Lotus: diffusion depth pipeline signal for SDPA, cross-attention, pointwise
  projection, UNet concat, resize, and layout/materialization behavior.
- HY-MT: Transformer decode signal for GQA SDPA, GQA repeat, KV-cache append,
  embedding gather, and fallback/readback attribution.
- PaddleOCR: OCR pipeline signal for batch norm, small-spatial pointwise conv,
  grid sample diagnostics, and remaining conv-transpose/fallback pressure.
- Gemma E2B: memory/dtype roadmap signal; current evidence says it is blocked
  before useful Vulkan route coverage by float32 model-weight OOM.

Do not infer production route names from this corpus.

## Windows Vulkan Build Defaults

The repo-owned Windows Vulkan helpers now default source-tree and wheel builds
to real distributed/c10d/Gloo support for model-framework import paths:
`USE_DISTRIBUTED=ON`, `USE_GLOO=ON`, `USE_C10D_GLOO=ON`, and `USE_LIBUV=ON`
with `libuv_ROOT` resolved from an explicit argument, the environment, or
`agent_space\libuv_install`. MPI, NCCL, c10d MPI/NCCL, and TensorPipe remain
off for this Windows-local configuration. Existing build products still need a
reconfigure and rebuild before `torch._C._distributed_c10d` appears in the
runtime; changing helper defaults does not repair an already-built
`torch/lib`.

## Current Telemetry Checkpoint

Task179 and Task181 artifacts are planner telemetry only; they do not raise a
model gate and they do not imply model-specific production routes.

- DAv2 RX 9070: stable. Task179 completed with `cpu_fallback=0`,
  `sync_readback=169`, `tensor_cpu_readback=430`, `retire_drains=102`, and
  `conv_prepack_upload=4`.
- HY-MT RX 9070 99-token prompt with 16 generated tokens: stable but still
  high in fallback/readback attribution. Task179 reported `cpu_fallback=423`,
  `sync_readback=83`, `tensor_cpu_readback=5827`, and model-core tensor-op
  fallback/readback `0/0`.
- PaddleOCR RX 9070 screenshot: stable in the Task179 single row. It reported
  `cpu_fallback=1`, `sync_readback=1`, `tensor_cpu_readback=1824`, and
  `conv_prepack_upload=140`; the earlier first-attempt DeviceLost did not
  reproduce in that run.
- Gemma E2B: still blocked before useful route coverage by model-weight Vulkan
  OOM while moving
  `gemma4forconditionalgeneration.model.language_model.embed_tokens_per_layer.weight`.
- Lotus: Task181 cleared the benchmark-local `_c10d_functional.wait_tensor`
  import blocker, but Lotus still fails before useful Vulkan execution because
  the source-tree environment lacks the compiled DTensor C API
  `_DTensor_OpSchema_post_init` in `torch._C`. The model-suite harness now
  preflights that symbol and reports a stable `missing_compiled_dtensor_c_api`
  skip before importing Lotus/Diffusers, so agents should not rediscover this
  as a raw Diffusers `ImportError`. The Lotus counters remain zero and the row
  must not contribute backend regression budgets.

Benchmark-local distributed shims must stay import-only and single-process.
`_c10d_functional.wait_tensor` may be an identity shim for telemetry imports;
collective and DTensor op schema stubs must raise if executed. Do not add
benchmark-local fakes for compiled `torch._C` DTensor APIs. Restoring Lotus
telemetry now requires a real source-tree distributed/DTensor-capable build or
a compatible runtime environment, not a Vulkan backend change.
Use
`python scripts\benchmarks\benchmark_model_suite.py --validate-lotus-dtensor-preflight`
to check the benchmark guard without running Lotus.

## Existing Audit Artifacts

- `agent_space/vulkan_contract_migration_plan.md`: policy lock and initial
  contract groups.
- `agent_space/model_named_routes.txt`: route-specialization audit with A/B/C/D
  classification.
- `agent_space/exact_shape_routes.txt`: finite tuple audit for conv, SDPA,
  embedding, cat, GQA repeat, batch norm, and safe view/reshape routes.
- `agent_space/five_model_blockers.json`: five-model blocker summary and next
  discovery focus.
- `agent_space/lotus_diffusion_sdpa_contract_draft.md`: draft finite
  `DiffusionSDPAContract` and `DiffusionCrossAttentionContract` evidence.
- `agent_space/lotus_pointwise_projection_contract_draft.md`: finite diffusion
  projection evidence for `SmallSpatialPointwiseConvContract`.
- `agent_space/task179_real_workload_status_telemetry.md`: telemetry checkpoint
  for DAv2, Lotus, HY-MT, PaddleOCR, and Gemma on the current local corpus.
- `agent_space/task181_lotus_shim_validation.md`: benchmark-local Lotus shim
  validation and current `missing_compiled_dtensor_c_api` blocker.

These files are diagnostic inputs. Production code must not depend on
`agent_space`.

## Current Contract Groups

- `SmallSpatialPointwiseConvContract`: finite projection rows, now split into
  a family-specific source. The `SparseProjectionRows` slice has a JSON
  contract spec backed by `ShapeEnvelope` v1 `sparse_rowsets` with all 55
  current projection rows plus a generated factorized depth-vision projection
  group for the cross-adapter proven 144-shape set. That group is the product
  of 18 approved `(input_c, output_c)` channel pairs and eight approved
  `(input_h, input_w)` spatial pairs, with 84 validated corpus/proof shapes
  and 60 proven factorized extrapolations; the expansion ratio is 1.7143x and stays
  below the 3x promotion cap. The generated helper provides contract identity,
  per-row metadata, input/weight channel equality, exact sparse-row lookup, and
  factorized correlation-group matching while route-policy hard-fail rescue,
  shader-family decisions, family op-hit labels, and match-result assembly
  remain handwritten. Naive min/max H/W bounds, independent H/W cross-products,
  and the 648/1296 channel/spatial cross-products remain explicitly forbidden.
- `NoOverlapConvTranspose2DContract`: bounded float-buffer 2x2 stride-2
  no-overlap transposed-conv envelope. The `Kernel2Stride2FloatBuffer` slice
  has a JSON contract spec backed by `ShapeEnvelope` v1 with checked-in
  positive/adjacent-negative runtime cases and generic ShapeEnvelope C++
  metadata/simple-bound helper output. Input/weight channel equality is
  generated; output shape arithmetic, prepack resource behavior, and
  match-result assembly remain handwritten. Preserve unsupported-case fallback
  outside that envelope.
- `SmallMetadataPaddedConv2DContract`: one proven padded low-channel
  buffer-input materialization tuple, now split into a family-specific source.
  The `MaterializedBufferInput2x2` slice has a JSON contract spec backed by
  `ShapeEnvelope` v1 with checked-in positive/adjacent-negative runtime cases
  and generic ShapeEnvelope C++ exact simple-bound helper output. The generated
  helper provides contract identity, metadata, exact input/weight/options
  predicates, and materialization policy constants while tensor-info
  extraction, input materialization, op-hit logging, fallback to
  `aten::convolution.buffer_float_skip.small_metadata_input`, and match-result
  assembly remain handwritten. Keep adjacent guards.
- `TransformerGQASDPAContract`: bounded Transformer causal/prefill and decode
  GQA SDPA legality with model-neutral naming, now split into a
  family-specific source. The `SparseAttentionRows` slice has a JSON contract
  spec backed by `ShapeEnvelope` v1 with checked-in positive/adjacent-negative
  runtime cases plus generic ShapeEnvelope C++ sparse-rowset helper output in
  `generated/ExecutionContractsTransformerGQASDPASpec.h`. The generated
  helper provides contract identity, per-row metadata, exact lookup by contract
  family plus causal/GQA flags, and row-match bounds/conditional equal-sequence
  checks while scale tolerance, route-policy hard-fail ordering, tensor
  extraction/early dtype-rank guards, SDPA execution, and match-result assembly
  remain handwritten.
- `VisionSelfAttentionSDPAContract`: bounded rank-3 float vision
  self-attention SDPA legality for the six proven low-resolution rows
  `[BH,T,64]` where `BH in {6,12,16}`, `T in {151,261}`, q/k/v share shape,
  there is no mask, non-causal, dropout is zero, GQA is off, and explicit
  scale is `1.0`. The contract uses a family-specific source with
  `ShapeEnvelope` v1 sparse-rowset spec coverage and generated C++ metadata/
  row helpers in `generated/ExecutionContractsVisionSelfAttentionSDPASpec.h`.
  Proof showed direct Vulkan softmax probabilities into value BMM are wrong
  for this family, while explicit post-softmax clone/materialization passes;
  `SDPAExecutionPolicyContract` therefore keeps the materialized math path and
  post-softmax clone decision for matched rows. The score-softmax materialized
  probability edge now uses `SDPAScoreSoftmaxContract`
  `VisionSelfAttentionScores`, which derives its six score rows `[BH,T,T]`
  from this generated rowset and writes probabilities into a fresh direct
  buffer before value BMM. `KnownBadGenericSdpa` remains active outside this
  finite rowset.
- `MaskedTinySDPAContract`: tiny additive-mask SDPA tuple, now split into a
  family-specific source. The `AdditiveFloatMask` slice has a JSON contract
  spec backed by `ShapeEnvelope` v1 with checked-in positive/adjacent-negative
  runtime cases plus generic ShapeEnvelope C++ simple-bound helper output in
  `generated/ExecutionContractsMaskedTinySDPASpec.h`. The generated helper
  provides contract identity, metadata, exact query/key/value/mask dtype, rank,
  shape, and scalar option predicates while route-policy hard-fail ordering,
  scale-tolerance comparison, SDPA execution, and match-result assembly remain
  handwritten. Keep the exact tuple until broader mask-family behavior is
  proven.
- `DiffusionSDPAContract` and `DiffusionCrossAttentionContract`: finite
  explicit tuple contracts, now split into a family-specific source; keep exact
  rows until broader materialization behavior is proven.
- `SDPAExecutionPolicyContract`: finite execution materialization, softmax
  score, post-softmax clone, and repeat policy contract, now split into a
  family-specific source. The `SparsePolicyRows` slice has a JSON contract
  spec backed by `ShapeEnvelope` v1 with checked-in positive/adjacent-negative
  runtime cases plus generic ShapeEnvelope C++ sparse-rowset helper output in
  `generated/ExecutionContractsSDPAExecutionPolicySpec.h`. The generated
  helper provides contract identity, per-row metadata, exact policy-row lookup,
  row-match bounds, and materialization policy flags while calls to
  `DiffusionSDPAContract`, tuple-id cross-checks, route hard-fail ordering,
  score materialization, post-softmax clone behavior, and match-result assembly
  remain handwritten. Keep exact rows until broader layout-transition behavior
  is proven.
- `SDPAScoreSoftmaxContract`: finite float rank-3 square score-softmax
  contract. The `DiffusionSquareScores` slice covers heads `{1, 5}` and
  sequence `{504, 640}` with a JSON contract spec backed by `ShapeEnvelope` v1,
  checked-in positive/adjacent-negative runtime cases, and generic
  ShapeEnvelope C++ simple-bound helper output in
  `generated/ExecutionContractsSDPAScoreSoftmaxSpec.h`. The
  `VisionSelfAttentionScores` slice is a bounded production materialization
  edge for the six existing VisionSelfAttention score rows `[BH,T,T]` where
  `BH in {6,12,16}` and `T in {151,261}`; it consumes the generated
  `VisionSelfAttentionSDPAContract` rowset as source of truth and keeps direct
  softmax-probability-to-value-BMM disabled. Softmax route ordering, guard
  fallback labels, buffer softmax policy, and match-result assembly remain
  handwritten. Keep the temporary exception until broader score-softmax/layout
  behavior is proven.
- `EmbeddingLookupContract`: finite token-batch and small-bounded embedding
  lookup contract; the small-bounded lookup slice has a JSON contract spec with
  generated positive and adjacent negative runtime coverage. The
  `SmallBoundedLookup` slice now uses the generic ShapeEnvelope C++ generator
  path for generated metadata, bounds, matcher helper predicates, and the
  derived indices product helper while the token-batch row remains
  handwritten. Keep remaining exact rows until broader legality is proven.
- `CatAxisContract`: umbrella for bounded last-dim, channel-dim, and rank-3
  cat patterns. The `ChannelCatContract` rank-4 dim-1 buffer slice has a JSON
  contract spec with generated positive and adjacent negative runtime coverage
  and a `ShapeEnvelope` v1 source for symbolic dims, relationships, aggregate
  bounds, layout/capability requirements, and policies. Its contract identity,
  route label, metadata, simple bounds, typed spec row, and scalar/per-input
  helper predicates are emitted by the generic ShapeEnvelope C++ generator into
  a generated C++ header while the cross-input loop and match result
  construction remain handwritten.
- `KVCacheAppendContract`: bounded Transformer sequence append and initial
  empty-cache cat rows. Both `SequenceAppend` and `InitialCache` slices have
  JSON contract specs backed by `ShapeEnvelope` v1 with checked-in positive
  and adjacent negative runtime cases plus generic ShapeEnvelope C++
  metadata/simple-bound helper output. The generated helpers provide contract
  identity, route labels, metadata, dtype/rank/scalar/range bounds, helper
  predicates, and SequenceAppend batch/heads/head-dim equality while
  initial-empty handling, sequence lower bounds, InitialCache cross-input
  handling, and match-result assembly remain handwritten. InitialCache positives
  log the contract-owned `aten::cat.kv_cache_initial_dim2_buffer` op-hit label
  while unrelated direct-buffer cat paths keep their generic labels.
- `UNetChannelConcatContract`: mostly generic already; keep model provenance in
  tests/docs.
- `GQARepeatContract`: finite bounded K/V head repeat contract, now split into
  a family-specific source. The
  `Batch1Heads4Factor4Sequence100To116Dim128` slice has a JSON contract spec
  backed by `ShapeEnvelope` v1 with checked-in positive/adjacent-negative
  runtime cases plus generic ShapeEnvelope C++ simple-bound helper output in
  `generated/ExecutionContractsGQARepeatSpec.h`. The generated helper provides
  contract identity, metadata, dtype/rank/source tensor bounds, repeat-factor
  policy constants, and target-head/target-sequence metadata while Vulkan
  tensor/storage extraction, SDPA admission, materialization allocation/kernel
  dispatch, op-hit labels, and match-result assembly remain handwritten. Keep
  exact rows until broader legality is proven.
- `BatchNormInferenceContract`: float32 4D inference batch norm. The
  `BufferFloat4D` and `MaterializedBufferFloat4D` slices both have JSON
  contract specs backed by `ShapeEnvelope` v1 with checked-in
  positive/adjacent-negative runtime cases. Both slices now use the generic
  ShapeEnvelope C++ generator path for generated metadata, simple bounds, and
  helper predicates, including optional-aware feature-count equality.
  Parameter checks, provenance, storage/materialization policy, and match
  result assembly remain handwritten. Tensor provenance and value traces report
  the admitted contract name, family, tuple id, and materialization policy for
  BatchNorm canaries without changing the visible execution route. When
  `PYTORCH_VULKAN_CONTRACT_ADMISSION_LOG` is set, direct and materialized
  candidate rows also emit first-failure or accepted admission JSONL events.
  Materialized positives intentionally log the direct-buffer storage reject,
  the materialized accept, and the post-materialization direct-buffer
  revalidation accept.
- `SafeViewReshapeContract`: finite dense direct-buffer view and reshape-alias
  contract, now split into a family-specific source. Both direct-buffer slices
  now have JSON contract specs with ShapeEnvelope-generated legal and
  adjacent-negative runtime coverage: `ViewMaterializedDirectBuffer` for the
  materialized `aten::view` path and `ReshapeAliasDenseBufferDirect` for the
  materialized `aten::_reshape_alias` path. Both slices now consume generated
  ShapeEnvelope C++ shape/layout simple-bounds headers for contract identity,
  metadata, rank bounds, storage-offset, output last-dim multiple, and policy
  constants plus generated product-equality helpers while
  dense/contiguous-stride checking and match result assembly remain
  handwritten. Keep broader view/layout,
  storage-offset, and provenance rules documented separately.
- `LinearGeluBridgeContract`: pure legality for the deferred linear/GELU
  bridge. The `BackboneMlpHidden384To1536` slice now has a JSON contract spec
  backed by `ShapeEnvelope` v1 with checked-in positive/adjacent-negative
  runtime cases and generic ShapeEnvelope C++ simple-bound helper output. The
  generated helper provides contract identity, metadata, rank/shape/packed
  weight/options predicates, minimum flattened rows, and result-policy
  constants while tensor-info extraction, rank-3 equality, deferred candidate
  registry ownership, alias retargeting, materialization on non-GELU
  consumers, fused-GELU execution, op-hit labels, and match-result assembly
  remain handwritten.
- `ElementwiseBroadcastContract`: production metadata/provenance canary for the
  existing float32 tensor/tensor buffer-broadcast route. The
  `FloatTensorTensorBufferBroadcast` slice records the route shape in JSON and
  runtime tests for `add`, `mul`, and `sub`, backed by a generic
  `ShapeEnvelope` `broadcast_compatible` relationship. Its contract identity,
  metadata, simple bounds, layout requirements, attribute helpers, and
  right-aligned broadcast compatibility helper are emitted by the generic
  ShapeEnvelope C++ generator v0. The matcher is queried only after the
  existing `aten::binary_op.buffer_float` route is selected, so it records
  contract admission metadata without adding a new route or broadening dtype,
  rank, layout, scalar, `out=`, or inplace behavior.
- DAv2 region/stack contracts: best current example of shape keys, capability
  keys, planned regions, binding validation, and replay-readiness diagnostics.

## Governance Guardrails

- `test/test_vulkan.py::TestVulkanGovernance` statically checks that tuple
  matches in `ExecutionContracts*.cpp` set metadata, active temporary
  exceptions include expiry and migration target, active exception locations
  still resolve where practical, and selected generic routing files do not
  introduce model-name strings.
- Contract spec governance discovers all `test/vulkan_contract_specs/*.json`,
  validates a shared schema, checks `contract_name`/`family`/`tuple_id` against
  live contract sources, validates any `ShapeEnvelope` v1 blocks present, and
  keeps family-specific shape checks for BatchNormInference, EmbeddingLookup,
  ChannelCat, KVCacheAppend, LinearGeluBridge, GQARepeat, MaskedTinySDPA,
  DiffusionSDPA, TransformerGQASDPA, VisionSelfAttentionSDPA,
  SDPAScoreSoftmax,
  NoOverlapConvTranspose2D, SmallMetadataPaddedConv2D, and SafeViewReshape.
  `test/vulkan_contract_specs/generated_cpp_manifest.json` declares which
  ShapeEnvelope specs have checked-in generated C++ helper headers; governance
  validates that the manifest covers every current ShapeEnvelope spec, each
  header exists, each header regenerates byte-for-byte from its spec, and each
  header contains the expected helper markers.
  `contract_spec_utils.py --contract-coverage-census` summarizes the current
  source-of-truth coverage by JSON spec row, ShapeEnvelope coverage, generated
  helper coverage, live contract names without JSON specs, and temporary
  exception linkage so new migrations do not mirror exact rows blindly.
  Shared helpers in `test/vulkan_contract_specs/contract_spec_utils.py` keep
  generated runtime tests from copying spec loading, case iteration, log
  naming, expected negative handling, and shape-envelope validation. A
  `SHAPE_ENVELOPE_ROLE_REGISTRY` now centralizes role validation, temporary
  runtime-case adapters, and data-driven semantic key fields so new roles do
  not add another open-coded key dispatch table. The same utility layer also
  has deterministic boundary/fuzz assignment generation for common
  ShapeEnvelope v1 concepts: value sets, min/max bounds, multiples, optional
  dims, scalar attributes, `broadcast_compatible` relationships, and
  adjacent-negative axes. It also validates an optional generic
  `sparse_rowsets` ShapeEnvelope concept for correlated finite-row contracts,
  including row identity uniqueness, lookup-key uniqueness, tuple-label
  uniqueness, independent cross-product census, and forbidden-cross-product
  negative metadata. `SmallSpatialPointwiseConvContract` and
  `DiffusionSDPAContract`, `SDPAExecutionPolicyContract`, and
  `TransformerGQASDPAContract`, and `VisionSelfAttentionSDPAContract` are the
  current real sparse-rowset consumers.
  A generic coverage bridge maps abstract assignment paths and
  adjacent-negative axes onto the current generated/checked-in runtime cases
  without executing additional fuzz assignments. BatchNormInference `BufferFloat4D`,
  `MaterializedBufferFloat4D`, ElementwiseBroadcast
  `FloatTensorTensorBufferBroadcast`, GQARepeat
  `Batch1Heads4Factor4Sequence100To116Dim128`, KVCacheAppend `SequenceAppend`
  and `InitialCache`, MaskedTinySDPA `AdditiveFloatMask`, DiffusionSDPA
  `SparseAttentionRows`,
  NoOverlapConvTranspose2D `Kernel2Stride2FloatBuffer`, SDPAScoreSoftmax
  `DiffusionSquareScores`, SmallMetadataPaddedConv2D
  `MaterializedBufferInput2x2`, and LinearGeluBridge
  `BackboneMlpHidden384To1536`, and TransformerGQASDPA
  `SparseAttentionRows`, and VisionSelfAttentionSDPA `SparseAttentionRows`
  use generic checked-in case plumbing under the ShapeEnvelope registry.
  ChannelCat, EmbeddingLookup, and both
  SafeViewReshape direct-buffer slices have
  deterministic `ShapeEnvelope` legal-case and adjacent-negative generators
  that must match the checked-in positive and negative cases by semantic key,
  violated axis, adjacent value, and fallback/readback policy. Their runtime
  spec tests now execute generated legal positives and adjacent negatives
  through shared iterator plumbing while checked-in cases remain review/parity
  fixtures.
- ChannelCat has the first source-of-truth C++ table/matcher proof:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` regenerates
  `generated/ExecutionContractsChannelCatSpec.h` from
  `channel_cat_contract.json`, including a typed row and helper predicates.
  Generation now consumes the fixture's `ShapeEnvelope` v1 metadata, variadic
  tensor-list input, aggregate channel bounds, and matcher hints through the
  generic ShapeEnvelope generator path; governance compares the output
  byte-for-byte with the checked-in header.
- EmbeddingLookup `SmallBoundedLookup` now consumes the generic ShapeEnvelope
  C++ metadata/helper generator path. `tools/vulkan_contracts/gen_contract_spec_cpp.py`
  emits `generated/ExecutionContractsEmbeddingLookupSpec.h` from
  `embedding_lookup_contract.json` for metadata, route label, dtype/rank-list,
  range, boolean option bounds, the derived indices product helper, and helper
  predicates; result construction, output-shape handling, and the token-batch
  family remain handwritten.
- ElementwiseBroadcast `FloatTensorTensorBufferBroadcast` is the first
  consumer of generic ShapeEnvelope C++ metadata/helper generation v0:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsElementwiseBroadcastSpec.h` from
  `elementwise_broadcast_contract.json` for contract identity, metadata,
  `add`/`mul`/`sub` op-axis, scalar/rank/layout/attribute bounds, and simple
  helper predicates. The
  broadcast relationship and match result construction remain handwritten, and
  the generated helpers are used only by the metadata/provenance canary after
  the existing route is selected.
- ElementwiseBroadcast is also the first consumer of env-gated admission
  diagnostics. When `PYTORCH_VULKAN_CONTRACT_ADMISSION_LOG` is set, the matcher
  emits one JSONL event for an accepted candidate or the first generated
  predicate rejection. The MVP payload intentionally excludes raw shapes,
  tensor ids, storage ids, and tensor values.
- BatchNormInference is the second admission-diagnostics consumer. Direct
  `BufferFloat4D` and materialized `MaterializedBufferFloat4D` rows use the
  same JSONL surface and preserve the existing pre-admission `training=True`
  rejection in `Batchnorm.cpp`.
- BatchNormInference `BufferFloat4D` and `MaterializedBufferFloat4D` now
  consume the same generic ShapeEnvelope simple-bounds generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsBatchNormInferenceSpec.h` and
  `generated/ExecutionContractsBatchNormInferenceMaterializedSpec.h` from the
  direct and materialized BatchNorm JSON specs for contract identity, metadata,
  dtype/rank/layout/training bounds, materialization policy, and simple helper
  predicates, including optional-aware feature-count equality. The
  simple-bounds generator emits row-qualified contract-name constants so
  sibling generated rows can be included in the same translation unit without
  duplicate symbols.
- KVCacheAppend `SequenceAppend` and `InitialCache` consume the generic
  ShapeEnvelope simple-bounds generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsKVCacheAppendSpec.h` and
  `generated/ExecutionContractsKVCacheAppendInitialSpec.h` from the sequence
  and initial-cache JSON specs for contract identity, metadata, route labels,
  dtype/rank/scalar/range bounds, helper predicates, and SequenceAppend
  batch/heads/head-dim equality. Initial-empty handling, sequence lower bounds,
  InitialCache cross-input handling, and match-result construction remain
  handwritten so route behavior is unchanged.
- GQARepeat `Batch1Heads4Factor4Sequence100To116Dim128` consumes the generic
  ShapeEnvelope simple-bounds generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsGQARepeatSpec.h` from
  `gqa_repeat_contract.json` for contract identity, metadata, dtype/rank/source
  tensor bounds, repeat-factor constants, and target-head/target-sequence
  metadata. SDPA admission, materialization allocation and dispatch, op-hit
  labels, sequence lower-bound preservation, and match-result assembly remain
  handwritten so route behavior is unchanged.
- SDPAScoreSoftmax `DiffusionSquareScores` consumes the generic ShapeEnvelope
  simple-bounds generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsSDPAScoreSoftmaxSpec.h` from
  `sdpa_score_softmax_contract.json` for contract identity, metadata,
  dtype/rank/last-dim, heads value-set, sequence value-set, square-score, and
  fallback/materialization policy constants. Softmax route ordering,
  `can_run_buffer_softmax` policy, guard op-hit logging for
  `aten::_softmax.buffer_lastdim_known_bad_texture_fallback`, and
  match-result assembly remain handwritten so route behavior is unchanged.
- MaskedTinySDPA `AdditiveFloatMask` consumes the generic ShapeEnvelope
  simple-bounds generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsMaskedTinySDPASpec.h` from
  `masked_tiny_sdpa_contract.json` for contract identity, metadata, exact
  query/key/value/mask dtype, rank, shape, and scalar option predicates. Route
  hard-fail ordering, scale tolerance, SDPA execution, and match-result
  assembly remain handwritten so route behavior is unchanged.
- DiffusionSDPA `SparseAttentionRows` consumes the generic ShapeEnvelope
  sparse-rowset generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsDiffusionSDPASpec.h` from
  `diffusion_sdpa_contract.json` for contract identity, per-row metadata, the
  11 correlated square/cross-attention rows, and exact lookup and row-match
  equality by heads, query-sequence, key/value sequence, and head dim.
  Route-policy hard-fail ordering, scale tolerance, SDPA execution,
  materialization policy, and match-result assembly remain handwritten so route
  behavior is unchanged.
- SDPAExecutionPolicy `SparsePolicyRows` consumes the generic ShapeEnvelope
  sparse-rowset generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsSDPAExecutionPolicySpec.h` from
  `sdpa_execution_policy_contract.json` for contract identity, per-row
  metadata, the six correlated execution-policy rows, exact lookup and
  row-match bounds by family, heads, sequence bounds, head dim, and GQA flag,
  and per-row materialization policy strings. Diffusion contract admission,
  tuple-id cross-checks, optional scale tolerance, score pre-materialization,
  materialized math path, post-softmax clone behavior, and broader SDPA policy
  remain handwritten so route behavior is unchanged.
- TransformerGQASDPA `SparseAttentionRows` consumes the generic ShapeEnvelope
  sparse-rowset generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsTransformerGQASDPASpec.h` from
  `transformer_gqa_sdpa_contract.json` for contract identity, per-row
  metadata, the four correlated causal/prefill/decode GQA rows, exact lookup by
  contract family plus causal/GQA flags, and row-match bounds/conditional
  equal-sequence checks. Optional scale tolerance, route-policy hard-fail
  ordering, tensor extraction/early dtype-rank guards, SDPA execution, and
  match-result assembly remain handwritten so route behavior is unchanged.
- VisionSelfAttentionSDPA `SparseAttentionRows` consumes the generic
  ShapeEnvelope sparse-rowset generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsVisionSelfAttentionSDPASpec.h` from
  `vision_self_attention_sdpa_contract.json` for contract identity, per-row
  metadata, the six correlated rank-3 head-dim-64 rows, and exact row-match
  bounds. Scale tolerance, route-policy hard-fail ordering, tensor
  extraction/early dtype-rank guards, materialized math-path selection,
  post-softmax clone behavior, and match-result assembly remain handwritten.
- NoOverlapConvTranspose2D `Kernel2Stride2FloatBuffer` consumes the generic
  ShapeEnvelope simple-bounds generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsNoOverlapConvTranspose2DSpec.h` from
  `no_overlap_conv_transpose2d_contract.json` for contract identity, metadata,
  dtype/rank/options/layout bounds, input/weight channel equality, and helper
  predicates. Output-shape arithmetic, prepack resource behavior, and match
  result construction remain handwritten so route behavior is unchanged.
- SmallMetadataPaddedConv2D `MaterializedBufferInput2x2` consumes the generic
  ShapeEnvelope simple-bounds generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsSmallMetadataPaddedConv2DSpec.h` from
  `small_metadata_padded_conv2d_contract.json` for contract identity,
  metadata, exact input/weight/options bounds, input/weight channel equality,
  and helper predicates. Tensor info extraction, materialization dispatch,
  op-hit logging, fallback visibility, and match result construction remain
  handwritten so route behavior is unchanged.
- LinearGeluBridge `BackboneMlpHidden384To1536` consumes the generic
  ShapeEnvelope simple-bounds generator path without a dtype-specific
  requirement:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsLinearGeluBridgeSpec.h` from
  `linear_gelu_bridge_contract.json` for contract identity, metadata,
  rank/shape/packed-weight/options bounds, and result-policy constants.
  Deferred registry lifetime, alias retargeting, materialization on non-GELU
  consumers, fused-GELU execution, op-hit labels, rank-3 equality, and match
  result construction remain handwritten so route behavior is unchanged.
- SmallSpatialPointwiseConv `SparseProjectionRows` consumes the generic
  ShapeEnvelope sparse-rowset generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsSmallSpatialPointwiseConvSpec.h` from
  `small_spatial_pointwise_conv_contract.json` for contract identity,
  per-row metadata, input/weight channel equality, the 55 correlated
  projection rows, exact lookup by input/output channel and spatial shape, and
  the generated 144-shape factorized depth-vision projection helper. The sparse
  rows now include sixteen exact mid-resolution depth-vision projection rows for
  spatial pairs `(30,45)` and `(40,62)` with only the proven channel/output
  pairs. Those spatial pairs were not added to the 144-shape factorized helper.
  That helper remains constrained to its approved channel-pair and spatial-pair
  correlation groups; broader min/max and independent cross-products remain
  guarded.
  Route-policy hard-fail rescue, shader-family decisions, family op-hit
  labels, and match result construction remain handwritten outside the bounded
  admission extension.
- SafeViewReshape `ViewMaterializedDirectBuffer` and
  `ReshapeAliasDenseBufferDirect` consume the generic ShapeEnvelope
  shape/layout simple-bounds generator path:
  `tools/vulkan_contracts/gen_contract_spec_cpp.py` emits
  `generated/ExecutionContractsSafeViewReshapeSpec.h` and
  `generated/ExecutionContractsSafeViewReshapeAliasSpec.h` from the regular
  view and reshape-alias JSON specs for contract identity, metadata, rank
  ranges, storage offset, stride/storage policy constants, Vulkan
  requirement, product equality policy, product-equality helpers, and output
  last-dim multiple helpers. Contiguous/dense-stride checks remain handwritten
  so route behavior is unchanged.
- Submit-origin counter tests use a named Python helper instead of raw numeric
  indices. The helper is intentionally test-local; no C++ diagnostic API change
  was made for this guardrail refresh.
- Tensor provenance/value-trace diagnostics can carry optional admitted
  contract metadata (`contract_name`, `contract_family`, `contract_tuple_id`,
  and `contract_materialization_policy`) for producers that pass an existing
  contract match. BatchNorm canaries distinguish direct buffer and
  materialized-buffer admission while the executed buffer kernel route label
  remains stable. ElementwiseBroadcast uses the same provenance path after the
  existing `aten::binary_op.buffer_float` route has already been selected.
- Capability-profile governance checks ensure the required profile IDs are in
  the manifest, the normalized feature/limit keys are present, docs state the
  non-emulation semantics, and runtime-policy tests verify optional ML features
  are clamped under `vk_min_1_1_compute`.

## Validation Caveats

- Model status artifacts can be stale relative to each other. Before changing a
  production route, confirm the relevant current blocker with a bounded smoke,
  focused test, or fresh diagnostic artifact.
- DAv2 stack owner is intentionally safe and does not merge command-buffer
  replay until descriptor ownership and binding validation are ready.
- Some compatibility evidence is device-specific. RX 9070 remains the primary
  optimization signal; RX 6700 XT and GTX 1080 are compatibility checks.
- Capability-profile tests are planner admission checks on the current device.
  They can find route over-admission under reduced feature masks, but they do
  not replace the RX 9070/RX 6700 XT/GTX 1080 real-hardware rows.
- Gemma E2B is a memory/dtype milestone, not a reason to add exact route
  exceptions.
- Lotus is telemetry-unavailable in the current source-tree environment. Do
  not fake compiled `torch._C` DTensor APIs in the benchmark harness to make it
  run; use a compatible distributed/DTensor-capable build or runtime before
  treating Lotus as backend evidence.
- PaddleOCR completed the Task179 RX 9070 screenshot row with one known CPU
  fallback and one sync readback, but that is still telemetry-only and not
  cross-adapter gate-ready. Rerun the real-model matrix after the next backend
  behavior change or before claiming or raising a model gate.

## Build Context

On this Windows machine, use the existing Visual Studio CMake build tree from
`build/CMakeCache.txt`. The local cache records Visual Studio 17 2022, x64,
Release, `USE_VULKAN=ON`, `USE_VULKAN_API=ON`, strict SPIR-V, Vulkan 1.3, and
SPIR-V 1.6 targets.
