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
  stack-region scope was observed, whether an actual region command-buffer
  owner was acquired, whether the context command-buffer batch candidate was
  observed, the candidate lifecycle status, preserved phase-boundary submit
  count, and actual elided submit count.
- `stack_exit_release` records public/private/captured/requested/final output
  release status, pending-retire transfer status, and command-pool reset
  deferral status. It also records whether the planned stack-region scope was
  released, whether an actual region command-buffer owner was released,
  candidate lifecycle status, preserved phase-boundary submit count, actual
  elided submit count, and whether command-pool reset was deferred to region
  release.

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
context-owned `StackPlannedRecordingSubmit`; it is not a region-owned command
buffer or batch lease. The stack-entry acquire row now emits a behavior-neutral
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
`region_exit_close_submit_owner_unavailable_preserved_phase_submit_batch_only`
because the preserved phase-submit batch lease is not a region close/submit
owner. It remains fail-closed until a real region-owned close/submit lease
exists.
The ownership rows now preserve that distinction at both stack entry and stack
exit: the planned stack-region scope can be observed, but
`region_command_buffer_ownership_acquired=0`,
`region_command_buffer_ownership_released=0`, and
`actual_elided_submit_count=0`.

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

Until this exists, phase-boundary submits continue to own command-buffer
recording closure and timeline creation.

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
