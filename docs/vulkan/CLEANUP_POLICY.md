# Vulkan Cleanup Policy

Cleanup is a background track. The C++ plan executor and remaining corpus
coverage are the critical path; cleanup work must not turn into a prerequisite
sprint for that work. Forward graph-runtime milestones create deletion
eligibility, and the cleanup track removes eligible code in coherent waves.

The governing rule is: Git is the archive. Preserve the supported result,
current evidence, and a concise rejection or replacement reason in the tree;
do not preserve an inactive implementation merely as history.

## Four-State Ledger

Every live Vulkan operator schema, custom class, `PYTORCH_VULKAN_*` environment
read, and public Python entry point is assigned to exactly one state in
`docs/vulkan/cleanup_ledger.json`:

- **Active**: part of the supported eager or graph runtime, or required by its
  evidence and diagnostics. The ledger records why it remains.
- **Migration**: still has a live caller or correctness role, but has a named
  replacement, deletion gate, and supported baseline.
- **Compatibility**: retained only for verified serialized-model or public-API
  compatibility, with evidence and a recheck/delete condition.
- **Delete-ready**: no longer has a supported caller or migration gate. Delete
  it in the named wave while preserving only the listed result or bug-class
  contract.

There is no quarantine state and no default classification. A new discovered
surface must be assigned explicitly; stale ledger entries also fail validation.

The current compatibility audit is empty. DepthExtractor constructs Python
modules and loads state dictionaries with `weights_only=True`; its loader and
template roots contain no checked-in TorchScript/ExportedProgram artifacts or
`vulkan_prepack` references. Reopen the audit if deployment starts consuming a
serialized TorchScript, ExportedProgram, or custom-class artifact.

The Qwen/gated-delta scope decision is now a retired record. The deployed Qwen
path uses Transformers without a Vulkan custom-op call site, Qwen is outside the
five-model Vulkan corpus, and repository reachability ended in the
implementation, registration, and dedicated planning cache. The context,
operators, shader, mechanism tests, `GatedDeltaSplit` cache, and residual
Qwen/gated-delta label sentinels were deleted. Generic planning-label inference
remains Migration in `LegacyPlanningInference.*` under its separate
explicit-field and allocation-lane parity gate.

## Generated Reachability Inventory

Run:

```text
python tools/vulkan_cleanup/generate_surface_inventory.py --write
python tools/vulkan_cleanup/generate_surface_inventory.py
```

The first command regenerates
`docs/vulkan/generated/cleanup_surface_inventory.json`; the second checks it.
The generator mechanically enumerates:

- `TORCH_LIBRARY` operator schemas;
- registered Vulkan custom classes;
- C++ and Python reads of `PYTORCH_VULKAN_*` variables; and
- public `torch.vulkan` and `torch.backends.vulkan` Python exports.

Discovery is factual and deterministic. Lifecycle state remains an explicit
reviewed decision in the ledger. CI-style tests reject generated-file drift,
unclassified surfaces, duplicate classifications, stale surface IDs, invalid
state metadata, restored files or dedicated symbols from deleted scope
decisions, and a compatibility count that contradicts the deployment audit.
Retired paths and dedicated symbols belong in generated scope decisions, not in
separate hand-maintained repository scans.

## Deletion Evidence

Deletion comparisons use supported defaults only:

- plain concrete eager; and
- `VulkanGraphProgram` when the replacement is graph-owned.

Opt-in canaries, benchmark-harness-only bridges, replay, and compiled-session
lanes remain historical evidence; they are not deletion bars. At deletion time,
record the commit, adapter and driver, workload and inputs, supported route,
correctness result, fallback/readback counters, submit-origin counters, peak
memory, and latency in the evidence artifact named by the ledger.

Mechanism-only tests die with their mechanism. Behavioral regression tests that
pin a bug class survive implementation changes; examples include unbind-chain
value preservation and rejecting deferred registration during graph execution.

## Sequencing

Work that can interleave with executor milestones now:

1. maintain the generated inventory, ledger, and review gates;
2. delete experiments already marked Delete-ready;
3. keep the generic `SyncCounters.*` evidence substrate separate from the
   stack proof/canary control plane remaining in `Sync.*`, preserving
   fallback/readback, submit-origin, retire, and GPU-timestamp attribution; and
4. keep retained packed-weight and linear-context residency in
   `PackedWeightCache.*`, separate from migration-only KV-cache, scratch-arena,
   and readback objects in `ExecutionObjects.*`; and
5. keep allocation-label, tensor-shape, and device-name guesses in
   `LegacyPlanningInference.*` and `LegacyDeviceNamePolicy.h`, separate from
   explicit semantic request construction and capability discovery; and
6. split mixed subsystems only when the split reduces the next deletion unit.

Later cleanup is a consequence of graph progress, not scheduled prerequisite
work. VisionBlocks, compiled-session/replay, inference-graph, and stack-era
systems remain Migration until Phase 5/6 graph programs provide the replacement
and supported-default parity required by their ledger gates.

`LegacyPlanningInference.*`, `LegacyDeviceNamePolicy.h`, and
`ModelLanePolicy.*` follow a strict order: graph lowering first supplies
explicit semantic `VulkanPlanningRequest` fields, supported eager and graph
paths demonstrate allocation-lane parity, and only then may those heuristics be
deleted. The temporary exception remains live until those conditions hold.

Delete subsystem-era documentation with the subsystem. This includes large
stack documents such as `STACK_REGION_DEPENDENCY_GRAPH.md`,
`STACK_BOUNDARY_VALUE_PRESERVATION.md`, and related ownership/program records;
do not leave them behind as apparent current architecture.
