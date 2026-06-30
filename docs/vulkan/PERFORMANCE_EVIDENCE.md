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
- `conv2d_buffer_float_3x3_s1p1` 16x8 workgroup: correct but slower;
- decoder-tail ReLU via conv clamp: correct but slower;
- fused Depth Anything V2 head shader path: correctness blocked;
- compiled-session bridge/replay shortcut: unsafe blocked;
- current-topology single-boundary submit elision: unsafe blocked.

These entries are intentionally generic where possible: the model is evidence
provenance, while the contract/topology scopes name reusable backend behavior.
