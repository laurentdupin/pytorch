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
  The plan also records the boundary-level metadata still missing before any
  submit can be removed.
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

This dump is diagnostic only. It does not insert barriers, skip submits, change
routes, or change accepted shapes.

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

## Submit Plan Stage

A phase-boundary submit may be skipped only when the boundary node is complete:

- every required producer-consumer ordering edge is covered by a planned
  device-side dependency
- unrelated pending retire resources are proven retire-only and do not require
  the boundary ordering
- public, host-visible, final-output, debug, and explicit readback blockers are
  absent or handled by a separate contract
- byte budgets are satisfied for the boundary and for the enclosing region
- capture parity and bridge output sanity pass for the guarded path

The decision must emit counters for complete boundaries, skipped submits,
inserted barriers, rejected boundaries, rejected edges, and budget failures.

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
