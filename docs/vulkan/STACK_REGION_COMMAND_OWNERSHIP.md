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
  owner/requester scope, unavailable command-buffer and command-pool lease
  status, diagnostic descriptor generation base, and scratch/temporary
  resource scope.
- `stack_exit_release` records public/private/captured/requested/final output
  release status, pending-retire transfer status, and command-pool reset
  deferral status.

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
planned region-exit submit point remains synthetic/unimplemented. The component
therefore fails closed through
`StackRegionExitCloseSubmitOwnerRequest.v0` /
`StackRegionExitCloseSubmitOwnerResult.v0`: the request API is present for
diagnostics, but the result is unavailable with
`region_exit_close_submit_owner_implementation_missing`. This preserves all
phase-boundary submits and queue-submit behavior.

The current proof surfaces show that a phase-boundary submit is not just a
resource visibility edge. It also closes and submits active recording state,
creates the timeline point used by retire ownership, and anchors descriptor and
allocator lifetime. A future safe optimization needs a stack/region owner that
owns those responsibilities together.

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
