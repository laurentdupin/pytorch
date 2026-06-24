# Stack Boundary Value Preservation Contract

`StackBoundaryValuePreservationContract.v0` is the behavior gate for any future
stack-region phase-submit elision. It is a design and diagnostic contract only;
it does not remove submits, defer submits, insert barriers, broaden shape
admission, or add model-specific production routing.

The contract exists because prior proof slices showed that barrier readiness and
per-resource hazard coverage are not enough to preserve values when a
phase-boundary submit is removed. A phase submit may also provide command
execution, queue visibility, timeline signaling, retire ownership, and
bookkeeping semantics. A future canary must prove those semantics are either
preserved by another real primitive or unnecessary for the selected boundary.

## Scope

The first intended scope is a single non-capture stack boundary, currently
represented by the diagnostic id
`non_capture_boundary:producer_block=0:consumer_block=1`. The contract is
generic over stack-region boundary ids and stack-region instance ids; model
names are not part of the production key.

Capture, public Tensor output, final output, host-visible, readback, and
requested-intermediate boundaries are ineligible unless a separate contract
proves their public visibility semantics.

## Readiness Classes

Each selected stack-region instance must classify into exactly one state:

- `not_barrier_ready`: the actual consumer input range lacks a matched real
  barrier-only canary record, live binding, stage/access proof, or old-carry
  retire/non-escape proof.
- `barrier_ready_but_submit_proof_incomplete`: barrier and resource-side proof
  are present, but submit-level execution, flush, timeline, lifetime, or
  side-effect proof is incomplete.
- `submit_proof_complete_but_value_preservation_missing`: submit-level proof is
  complete, but the value-preservation predicates below are not all satisfied.
- `submit_elision_canary_ready`: every predicate is satisfied in the current run
  and a separate explicit behavior flag is still required before any canary may
  remove a submit.

## Required Predicates

`StackBoundaryValuePreservationContract.v0` requires all of the following for
the exact selected boundary instance and exact live submit key:

- Same command-buffer recording scope is observed for the producer, barrier,
  pending dispatch range, and consumer.
- Producer command recording is ordered before the consumer command recording.
- Consumer descriptor binding is recorded after the barrier insertion point.
- Actual descriptor update generation is observed and stable for the selected
  descriptor binding, not only logical descriptor identity.
- The actual consumer input allocation/range is covered by a real
  command-recorded barrier with the required stage/access transition.
- The old residual carry allocation/range is not read later and is proven
  non-escaping and retire-only.
- No public, final, host-visible, readback, alias, or requested-intermediate
  dependency is attached to the selected boundary.
- Allocator, descriptor, command-pool, and retire/lifetime side effects do not
  require the removed phase submit.
- All pending writes in the removed submit's side-effect set are covered by
  real barriers or are proven non-escaping/retire-only.
- Phase-submit execution and flush semantics are either replaced by a real
  primitive or proven unnecessary. A barrier alone is not such a replacement.

## Execution And Flush Semantics

For a selected boundary submit, the current Vulkan phase submit can close and
submit the active command buffer, provide execution visibility through
`vkQueueSubmit`, signal a stream timeline, attach pending resources to a retire
timeline, reset submit bookkeeping, and enable later retire polling. These are
value-preserving semantics, not merely memory-barrier semantics.

The contract may accept a future canary only if one of these is proven:

- the phase submit has no execution/flush dependency for the selected pending
  dispatch range; or
- a real replacement primitive provides equivalent execution visibility,
  timeline, and retire ownership; or
- a region-owned command-buffer or command-buffer batch remains valid across
  the phase boundary and is submitted later at a proven region submit point with
  equivalent lifetime and visibility semantics.

Current diagnostics do not prove any of those accepted states.

## Required Deferred-Submit Architecture Proof

A future region-owned deferred-submit path must provide:

- a stack-owner request hook for a region-owned command buffer or batch;
- same stream/queue proof;
- command-buffer lifetime scope through the planned region-exit submit;
- descriptor lifetime scope through the planned region-exit submit;
- command-pool lifetime scope through the planned region-exit submit;
- retire timeline ownership transfer from the phase submit to the planned
  region submit; and
- fallback behavior that preserves the existing phase-boundary submit when any
  field is unavailable.

Without this hook and ownership model, the only correct status is fail-closed.

`StackRegionCommandBufferRequest.v0` and
`StackRegionCommandBufferRequestResult.v0` are the current behavior-neutral
request surface for this hook. They let the stack owner model a request for a
region-owned command buffer or batch with stack-region lifetime, same
stream/queue, descriptor lifetime, command-pool lifetime, retire timeline, and
public/final/host/readback policy requirements. The current result is
fail-closed: a minimal runtime API skeleton is present and callable by
diagnostics, but it returns unavailable without allocating, switching,
deferring, or submitting command buffers. The current result status is
`request_result_runtime_api_present_unavailable`.

`StackRegionOwnedCommandBufferContract.v0` names that missing implementation
contract directly. It is a behavior-neutral design surface for the future
region-owned object: stack/region owner scope, command-buffer or batch
ownership, same-stream/queue requirement, command-pool lifetime, descriptor
lifetime, allocator/retire-timeline scope, stack-entry acquire point,
stack-exit release/submit point, and public/final/host/readback policy. The
current status is fail-closed:
`owned_command_buffer_contract_runtime_api_present_result_unavailable`. The
first concrete blocker behind the API is
`missing_command_pool_lifetime_extension`. It does not allocate, switch, defer,
or submit command buffers.

`StackRegionCommandBufferLifetimeReservation.v0` is the current fail-closed
surface for that lifetime blocker. It models reserving a future region-owned
command buffer or batch through a planned region-exit submit point, including
stack/region lifetime, command-pool lifetime, owner/requester scope, and
public/final/host/readback fallback policy. The runtime API is present for
diagnostics and returns unavailable without changing execution. Current rows
report `command_buffer_lifetime_reservation_unavailable`; the refined blocker is
`command_pool_cannot_extend_beyond_phase_submit`.

`StackRegionCommandPoolLifetimeContract.v0` decomposes that blocker into the
future command-pool ownership requirements. It records the current context
phase-submit owner scope, selected phase-boundary id, requested region lifetime
scope, planned region-exit release point, linked lifetime reservation key,
command-pool retention API status, and command-pool reset deferral status. It is
also fail-closed: the contract currently reports
`command_pool_lifetime_contract_unavailable`, with
`command_pool_retention_request_api_present_result_unavailable` and
`command_pool_reset_deferral_proof_blocked_retention_unavailable` as the
concrete implementation gaps. It does not retain a command pool or let a
command buffer cross a phase boundary.
`StackRegionCommandPoolRetentionRequest.v0` and
`StackRegionCommandPoolRetentionResult.v0` make that first gap a typed runtime
API surface. The request models retaining the context-owned command pool until a
planned region-exit release point; the result currently returns unavailable
with `missing_region_exit_release_ownership` once the planned exit-release
point is surfaced. This remains diagnostic only: no command pool is retained, no
reset is deferred, no command buffer is allocated or switched, and no submit is
removed.
`StackRegionExitReleasePoint.v0` names that planned stack/region exit release
target. It is synthetic/planned-only today and records the future
responsibilities for command-buffer close/submit, descriptor lifetime release,
retire timeline release, allocator/resource retire ownership, and command-pool
cleanup. This corrects the current proof language: an ordinary phase submit
closes/submits active recording state and creates a retire timeline, but it is
not normally a raw command-pool reset.
`StackRegionExitReleaseOwnershipContract.v0` defines who would own those
release responsibilities at the planned exit point. Current rows expose the
stack/region owner identity but return unavailable for command-buffer
close/submit ownership, queue submit/timeline ownership, descriptor release,
retire timeline release, allocator/resource release, and command-pool cleanup
ownership. The refined blocker is
`missing_region_exit_release_ownership_implementation`.
`StackRegionCommandPoolResetDeferralProof.v0` models whether the current
phase-submit recording-epoch consumption point could move to a planned
region-exit release point. It records the current boundary, planned release
point, retention
result status, descriptor lifetime, command-buffer lifetime, and retire
timeline blockers. Current rows fail closed with
`command_pool_reset_deferral_proof_blocked_retention_unavailable`; this is a
proof-status refinement, not reset deferral.

This mirrors the CUDA/DirectML research direction at a contract level only:
safe boundary removal needs one owner for command recording, descriptor and
resource lifetime, allocator/retire lifetime, and execution completion. Vulkan
does not have that owner yet.

## Current vits_140 Status

The latest one-image `vits_140` bridge graph with the opt-in barrier-only
canary classifies all selected `residual2@0 -> norm1@1` rows as
`barrier_ready_but_submit_proof_incomplete`. The exact missing semantic proof is
`missing_region_exit_release_ownership_implementation`, after the planned
exit-release point is identified but not connected to a region-owned command
buffer/batch, descriptor lifetime, allocator/retire ownership, and timeline
release contract.
That release ownership is the first concrete runtime capability needed before a
region-owned command buffer or batch could replace the current phase submit's
execution/flush/timeline role.

The current graph must therefore keep `submits_removed=0` and
`submit_elision_ready=0`.
