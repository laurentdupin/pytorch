# Stack Region Command Ownership

`RegionCommandBufferOwnership.v0` is the next architecture direction for
stack-region execution. It is a design card and implementation task card only.
It does not remove phase-boundary submits, defer submits, replay command
buffers, tune shaders, broaden shape admission, or add model-specific routing.

## Current Scaffold

`RegionCommandBufferOwnership.v0` is now emitted as behavior-neutral graph
records for the selected stack-region boundary. The records split stack-entry
acquire from stack-exit release:

- `stack_entry_acquire` records the region id, stack-region instance id,
  owner/requester scope, `StackRegionCommandBufferAcquireHook.v0` status,
  `RegionOwnedCommandBufferLease.v0` status, preserved context submit
  command-buffer candidate status, unavailable stack-region command-buffer and
  command-pool lease status, diagnostic descriptor generation base, and
  scratch/temporary resource scope. It also records whether the planned
  stack-region scope was observed, that the acquire record itself was emitted,
  whether an actual region command-buffer owner was acquired, whether the
  context command-buffer batch candidate was observed, the candidate lifecycle
  status, preserved phase-boundary submit count, and actual elided submit
  count.
- `stack_exit_release` records public/private/captured/requested/final output
  release status, pending-retire transfer status, and command-pool reset
  deferral status. It also records whether the planned stack-region scope was
  released, that the release record itself was emitted, whether an actual
  region command-buffer owner was released, candidate lifecycle status,
  preserved phase-boundary submit count, actual elided submit count, and whether
  command-pool reset was deferred to region release.

The scaffold preserves current behavior. Phase-boundary submits remain
preserved, actual submit elision remains zero, deferred submit remains disabled,
and command-pool reset after region release is fail-closed until a real region
release owner exists.

`StackRegionExitReleaseOwnership.v0` now expands the release half of that
scaffold into the components a future region owner would have to own. It
records public, private bridge, captured, requested-intermediate, and final
output release status; pending-retire transfer; descriptor lifetime release;
retire timeline release; allocator/resource release; command-buffer
close/submit ownership; queue-submit/timeline ownership; and command-pool
cleanup/reset ownership. Current rows are still fail-closed: private bridge
output policy can be observed, but pending retires are not transferred and the
descriptor, retire, allocator, command-buffer, queue-submit, and command-pool
release owners are missing. The refined blocker is
`missing_command_buffer_close_submit_release_ownership`.

`StackRegionCommandBufferCloseSubmitOwnership.v0` is the first component under
that release scaffold. It records the selected stack-region instance, boundary,
current command-buffer recording id/scope, planned region-exit release point,
and who owns close/submit today. The current phase submit still owns
command-buffer close/submit, the command buffer is not region-owned, and the
planned region-exit submit point can now be observed at the real stack planned
recording exit submit callsite. The observed submit is still the existing
context-owned `StackPlannedRecordingSubmit`; `StackRegionExitReleasePoint.v0`
can use it as the concrete stack-exit release anchor, but it is not a
region-owned command buffer or batch lease. The stack-entry acquire row now emits a behavior-neutral
`StackRegionCommandBufferAcquireHook.v0` row and a
`RegionOwnedCommandBufferLease.v0` lease row. The acquire hook is present near
`Context` and snapshots current stack planned-recording ownership, current
command-buffer recording id, and context-owned descriptor/command-pool scope.
When stack planned recording is active and the preserved stack-exit submit point
is observed, it exposes a region-owned preserved phase-submit batch lease for
accounting. That lease does not transfer command-buffer close/submit ownership,
does not transfer command-pool reset ownership, and does not change submit
behavior. The batch has a stack-entry lifecycle id that is finalized at stack
planned-recording submit or cancel. If the runtime exit point is not observed,
the hook still fails closed on the non-transferable context candidate.
`StackRegionSingleRecordingPlan.v0` is the next behavior-neutral scaffold under
that hook. It records that the current execution mode remains
`context_phase_submit_recording`, phase-boundary submits are preserved, command
buffer execution topology is unchanged, and a borrowed context command buffer is
not a valid region lease because phase submits still close and submit the active
recording. The lease is requested but unavailable with
`region_owned_command_buffer_lease_unavailable_single_recording_owner_lacks_close_submit_ownership`:
`StackRegionSingleRecordingOwner.v0` now records the lifecycle around stack
planned recording, but it does not own close/submit, command-pool, descriptor,
or retire-timeline lifetime. Those responsibilities remain with the Vulkan
context's phase-submit path, descriptor and retire lifetime scopes are
unavailable, and same-stream/queue proof is still only a requirement. The
supporting
`StackRegionExitCloseSubmitOwnerRequest.v0` /
`StackRegionExitCloseSubmitOwnerResult.v0` request surface feeds
`RegionExitCloseSubmitOwner.v0`, which now fails closed on the missing
region-owned command-buffer or batch lease once the real preserved stack-exit
submit point is observed. This preserves all phase-boundary submits and
queue-submit behavior.
If the preserved phase-submit batch and reset-deferral owner accounting surface
are both present, close-submit rows may mark accounting availability while
leaving `ownership_available=0`,
`region_exit_close_submit_owner_behavior_enabled=0`, and
`region_exit_close_submit_owner_authorizes_submit_elision=0`. The fail-closed
blocker remains `command_pool_reset_deferral_owner_behavior_disabled` until an
actual reset-deferral behavior implementation exists.
Once reset deferral has no blocker,
`PYTORCH_VULKAN_STACK_REGION_CLOSE_SUBMIT_OWNER=preserved_phase_submit_batch`
can turn on the close-submit owner canary. This marks the owner surface
available for accounting and sets close-submit behavior enabled, but keeps
submit-elision authorization disabled. It does not defer, remove, or create a
submit.
The live single-recording canary observes this flag as well, so its
selected-boundary rows can report
`region_exit_close_submit_owner_authorizes_submit_elision_disabled` instead of
the older preserved-batch-only blocker. This is still a fail-closed ownership
classification, not a region-owned close/submit implementation. The live guard
has a separate submit-elision authorization input that is currently `0`, so a
future lifecycle-state change cannot remove a submit unless ownership and
authorization are both wired deliberately.

The current proof surfaces show that a phase-boundary submit is not just a
resource visibility edge. It also closes and submits active recording state,
creates the timeline point used by retire ownership, and anchors descriptor and
allocator lifetime. A future safe optimization needs a stack/region owner that
owns those responsibilities together.

`StackRegionSingleRecordingCanary.v0` is the first opt-in check of that model.
With both the barrier canary and
`PYTORCH_VULKAN_STACK_REGION_SINGLE_RECORDING_CANARY=non_capture_residual2_norm1_block1`
enabled, a proof warmup pass may record the selected non-capture
`residual2@0 -> norm1@1` boundary. A second pass may then keep the current
stack-region command recording open across exactly that phase boundary and
close/submit at stack exit. The canary is not default behavior, does not use
the older retire-time submit-elision canary, and does not remove any boundary
outside the selected one. The real `vits_140` failure showed that one actual
Norm1 input barrier is not enough to replace the full phase submit, so the
canary now requires validated barrier coverage to span the pending dispatch
range before any selected submit can be deferred. The focused test asserts
output parity and fail-closed behavior when barrier coverage is incomplete,
while still checking the single-recording owner, live command-buffer id,
pending dispatch range, actual Norm1 input barrier proof, and
host/final/readback blocker checks.
The canary row now mirrors the preserved phase-submit batch lease when the
planned stack scope is active:
`region_owned_command_buffer_batch_lease_available_preserved_phase_submits`
with a fail-closed
`region_exit_close_submit_owner_preserved_phase_submit_batch_fail_closed`
close/submit owner status. This does not promote that batch into a region
close/submit owner; the blocker stays
`region_exit_close_submit_owner_unavailable_preserved_phase_submit_batch_only`.

The first real `vits_140` bridge run with this canary is not a valid
performance result. It removed exactly one selected submit, but bridge sanity
failed, so the benchmark marks the row invalid with
`vulkan_stack_output_device_bridge_sanity_failed`. This keeps the canary as a
diagnostic proof surface only. The next behavior path should create a planned
region-owned command-buffer topology instead of trying to defer individual
phase submits in the current topology.

`StackRegionCommandBufferTopologyPlan.v0` is the behavior-neutral row for that
next path. It preserves current phase-boundary submits while naming the
requested stack-entry to stack-exit region-owned command-buffer topology. The
vision stack capture-to-decoder bridge now installs a
`VulkanStackPlannedRegionScope` with the stack context, bridge session, stack
plan, producer role, consumer role, and capture ids. That moves the bridge
blocker to `planned_region_topology_present_close_submit_still_context_owned`
while keeping ordinary non-bridge stack dumps fail-closed if region identity is
missing.
The close/submit owner surface now carries that planned-region context through
`StackRegionExitCloseSubmitOwnerRequest.v0`,
`StackRegionExitCloseSubmitOwnerResult.v0`, and
`RegionExitCloseSubmitOwner.v0`. Bridge rows therefore no longer collapse into
the older generic missing-topology bucket: they fail closed because the planned
region exists, but close/submit is still owned by the context phase-submit
path. This does not close, defer, submit, or switch command buffers.
`StackRegionExitSubmitRuntimePoint.v0` is the behavior-neutral runtime
observation for the existing stack planned recording exit submit. It changes
diagnostic classification only: planned bridge rows can report
`planned_region_exit_submit_point_runtime_observed_context_submit_preserved`
and then fail closed on
`command_pool_reset_deferral_implementation_missing` when the preserved
phase-submit batch lease is observed but command-pool reset deferral still has
no region-owned implementation. The preserved batch remains accounting evidence,
not a region close/submit owner, and the path stays fail-closed until a real
region-owned close/submit lease and reset-deferral owner exist.
The ownership rows now preserve that distinction at both stack entry and stack
exit: the planned stack-region scope can be observed, but
`stack_entry_acquire_record_emitted=1`,
`stack_exit_release_record_emitted=1`,
`region_command_buffer_ownership_acquired=0`,
`region_command_buffer_ownership_released=0`, and
`actual_elided_submit_count=0`.
The emitted-record fields mean the ownership surface was populated; they do not
mean command-buffer, command-pool, descriptor, or retire ownership transferred
away from the current context phase-submit path.
Rows also spell out the fail-closed ownership facts:
`region_owned_close_submit_available=0`,
`close_submit_ownership_status=close_submit_still_context_phase_submit_owned`,
`command_pool_reset_ownership_status=command_pool_reset_still_context_owned_not_deferred`,
`descriptor_lifetime_ownership_status=descriptor_lifetime_still_context_owned_not_releasable`,
and
`retire_timeline_ownership_status=retire_timeline_still_context_owned_not_transferred`.
The acquire/release observation is backed by a separate
`ContextRegionCommandBufferOwnershipState.v0` lifecycle id/state. Its states
are deliberately named as context-owned fail-closed states:
`region_command_buffer_ownership_lifecycle_acquire_observed_context_owned_fail_closed`,
`region_command_buffer_ownership_lifecycle_release_observed_context_owned_fail_closed`,
and
`region_command_buffer_ownership_lifecycle_cancel_observed_context_owned_fail_closed`.
That lifecycle is proof that the stack-entry and stack-exit records were
observed, not proof that command-buffer ownership transferred.
`StackRegionSingleRecordingCanary.v0` now gets its close/submit owner
availability from Context-owned lifecycle state. Stack planned recording
creates a live close/submit owner lifecycle id and marks it active only as a
preserved-phase-submit-batch candidate. The canary rows expose that id, state,
lifecycle status, explicit behavior-enabled bit, and
`ContextStackRegionCloseSubmitOwnerState.v0` as the availability source. This
is still fail-closed: the active state is not a region-owned close/submit
implementation, the behavior-enabled bit is `0`, phase-boundary submits remain
preserved, and the canary cannot authorize behavior until a future state
represents a real region exit close/submit owner and explicitly enables
behavior.
The same lifecycle fields are now propagated through the ownership row chain:
`StackRegionSingleRecordingOwner.v0`,
`StackRegionCommandBufferAcquireHook.v0`,
`RegionOwnedCommandBufferLease.v0`,
`StackRegionExitCloseSubmitOwnerRequest.v0`,
`StackRegionExitCloseSubmitOwnerResult.v0`,
`RegionExitCloseSubmitOwner.v0`,
`StackRegionCommandBufferCloseSubmitOwnership.v0`, and
`RegionCommandBufferOwnership.v0`. This keeps the lifecycle source queryable
from the ownership records themselves, but it remains behavior-neutral and
does not change close/submit ownership availability.

## Design Card

### Stack-Entry Acquire

At stack entry, the region owner would acquire a typed region execution scope.
The acquire record must include:

- region id and stack-region instance id;
- command-buffer lease or command-buffer batch lease;
- command-pool lease;
- descriptor generation base for the region;
- scratch and temporary resource scope for stack-owned allocations.

The acquire does not imply behavior by itself. In v0 it is a contract surface
that must fail closed if the region owner cannot prove the requested command
buffer, command pool, descriptor generation, and temporary resource scope.

### Phase-Boundary Handling

In v0, current phase-boundary submits are preserved. A phase boundary remains a
logical ordering point in the stack proof, not ownership of command-buffer
lifetime. The design must not reinterpret an ordinary phase boundary as a safe
place to remove a submit unless a region owner proves the command recording,
descriptor lifetime, allocator lifetime, and retire timeline semantics that the
submit currently provides.

### Stack-Exit Release

At stack exit, the region owner would release or publish everything owned by the
region. The release contract must classify:

- public outputs;
- private bridge outputs;
- captured outputs;
- requested intermediates;
- final outputs;
- pending retires;
- command-pool reset deferral.

Public, final, host-visible, readback, and requested-intermediate outputs must
remain fail-closed unless their output ownership and visibility semantics are
explicitly proven.

### Command-Buffer Lifetime

The region owner must define:

- when command buffers or command-buffer batches are reserved;
- when they may be closed and submitted;
- when their command pool may be reset or reused;
- which planned region-exit release point owns the release decision.

The stack planned-recording exit submit can be observed as the release point in
v0. The current context command pool is also observed as retained through that
release point, but this is not a transferable region command-pool lease. Until
region ownership exists, phase-boundary submits continue to own command-buffer
recording closure, command-pool reset eligibility, descriptor lifetime, and
timeline creation. Reset deferral remains fail-closed at the owner layer. The
proof layer can now report
`command_pool_reset_deferral_proof_complete_context_pool_retained_until_release_point`
for the existing context-retained path; that proof does not authorize submit
elision or transfer reset ownership to the region. The lifetime contract reports
that distinction as `command_pool_lifetime_context_retained_not_region_owned`.
`StackRegionCommandPoolResetDeferralOwner.v0` is the behavior-neutral owner
surface for that missing step. It records the proof key/status, current
command-pool owner scope, requested stack-region owner scope, and planned
release/reset point. Proof-complete rows may set `owner_available=1` for
accounting, but keep `reset_deferral_behavior_enabled=0`,
`defers_command_pool_reset=0`, and `authorizes_submit_elision=0`; the active
blocker is `command_pool_reset_deferral_owner_behavior_disabled`. It also
reports `ContextStackRegionCommandPoolResetDeferralOwnerState.v0` lifecycle
id/state/status/source from stack entry through submit or cancel finalization.
Those states are context-owned and not deferred; they are not region ownership.
`PYTORCH_VULKAN_STACK_REGION_RESET_DEFERRAL_OWNER=context_retained_release_point`
turns on only the reset-deferral owner canary for rows whose proof is complete.
It clears the reset-deferral owner blocker and reports that command-pool reset
would be deferred to the observed release point, but it does not authorize
submit elision or make the close-submit owner available for execution.

### Release Ownership

Before stack exit, stack-owned resources remain under the region owner. At stack
exit, ownership may transfer to public outputs, bridge-private consumers,
requested captures, final output handling, or retire queues. The completion
signal that proves release is safe must be a real execution/timeline primitive,
not a diagnostic row and not a memory barrier by itself.

The release owner must cover:

- command-buffer recording close ownership;
- queue submit and timeline ownership;
- descriptor lifetime release ownership;
- retire timeline release ownership;
- allocator and resource lifetime release ownership;
- command-pool cleanup or reset ownership when applicable.

### Failure Policy

`RegionCommandBufferOwnership.v0` must fail closed if any of these are
incomplete:

- output ownership;
- descriptor generation;
- command-pool lifetime;
- command-buffer lifetime;
- retire transfer;
- allocator/resource release;
- same stream/queue ownership;
- public, final, host-visible, readback, or requested-intermediate policy.

Failing closed means preserving all current phase-boundary submits and current
runtime behavior.

## Implementation Task Card

### Implement behavior-neutral RegionCommandBufferOwnership v0

Implement a behavior-neutral region command ownership surface that records stack
acquire and stack release ownership without changing execution. The first
implementation should produce typed acquire/release records and prove that
current behavior is preserved.

Definition of done:

- `vits_140` bridge output remains correct.
- Same-region bridge still works.
- Buffer copies remain zero.
- CPU fallback/readback remains zero.
- Phase-boundary submits are preserved.
- Actual submit elision remains zero.
- Region acquire/release records are emitted.
- Command-pool reset is proven after region release or fail-closed.
- `docs/vulkan/CURRENT_STATE.md` is updated.

Validation:

- Focused stack-region dependency graph tests for acquire/release record
  presence and fail-closed behavior.
- Targeted one-image `vits_140` bridge graph refresh with existing barrier-only
  canary if needed to confirm output and counters.
- `python -m py_compile test/test_vulkan.py` if the test file is touched.
- `git diff --check`.

Non-goals:

- no submit elision;
- no deferred submit;
- no command-buffer replay;
- no shader work;
- no shape or admission broadening;
- no DAv2-specific production routing.
