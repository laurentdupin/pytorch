# Vulkan Graph Runtime

## Decision

The Vulkan performance path is graph-first.

Models are captured on CPU with `torch.export`, lowered through Vulkan
contracts, and executed as cached `VulkanGraphProgram` objects. The eager
Vulkan backend remains the correctness reference and capture-free fallback. It
must not grow another speculative deferred-execution system.

This decision supersedes attempts to infer whole-program ownership from eager
op side effects. Existing stack-region, deferred-bridge, replay, and compiled
session work remains migration evidence until the graph runtime replaces it.

## User Surface

The intended model-independent API is:

```python
program = torch.vulkan.export_and_lower(
    model,
    example_inputs,
    dynamic_shapes=None,
)
output = program(*runtime_inputs)
```

The first version is inference-only. Capture happens from a CPU model and CPU
example inputs because Vulkan opaque tensors cannot currently be described by
FakeTensor storage introspection. The returned program owns the Vulkan model
state and accepts inputs matching its exported guards.

Applications may call this API from a generic model-loading wrapper. Downloaded
model source must not require Vulkan-specific rewrites or model-name dispatch.
Capture failure leaves ordinary eager Vulkan available as an explicit caller
choice; it must not create a hidden CPU partition.

## Compilation Pipeline

### Capture

`torch.export.export(..., strict=False)` produces the functional ATen graph,
graph signature, lifted parameters and buffers, range constraints, and output
signature. The exported graph is the source of truth for producer-consumer
edges, mutation, aliasing, output escape, and value lifetime.

### Normalize And Place

The lowering pass:

- rewrites factory-device arguments captured from the CPU example run;
- places lifted parameters and buffers on the selected Vulkan device;
- creates packed linear and convolution contexts outside timed execution;
- normalizes graph metadata into stable input, constant, temporary, and output
  value classes;
- rejects unsupported mutation or alias semantics with a node-level reason.

### Admit And Partition

`KernelFamilyContract` and semantic `DynamicProgramRuntime` families become
node-admission predicates. Admission is based on operation semantics, dtype,
rank, layout capability, and device limits rather than prior observation of an
exact shape.

The default policy is Vulkan-only and fail-loud. An optional CPU partitioning
policy may be added only when partition boundaries, transfer bytes, readbacks,
and synchronization are explicit in the compiled plan and runtime counters.
Unsupported nodes must never silently fall through to CPU.

### Rewrite And Fuse

Fusion is a graph rewrite, not an eager placeholder protocol. Initial rewrite
families should target existing kernels and proven generated code:

- linear plus GELU;
- layer scale plus residual add;
- token prefix cat plus position add;
- normalization chains;
- QKV transform and SDPA compositions;
- supported elementwise chains.

Every fused node retains an unfused Vulkan decomposition for correctness and
plan comparison. Fusion eligibility is stateless for a given graph and plan
key; arbitrary eager consumers never observe an unmaterialized placeholder.

### Plan Layout Transitions

`LayoutTransitionContract` rules become graph-edge rules. Each edge records the
producer layout, consumer requirement, selected representation, transition
reason, physical-copy status, bytes, and host/synchronization budget.

Metadata and descriptor views remain zero-copy. Required materializations are
explicit plan steps. Unknown transition reasons fail graph validation until
classified.

### Plan Memory

SSA first-use and last-use positions determine temporary lifetimes. A graph
program owns a stable memory arena containing:

- input slots;
- immutable constants and packed weights;
- reusable temporary slots;
- output slots;
- workspace and parameter buffers.

Slot reuse requires non-overlapping lifetimes, compatible dtype/layout and
alignment, and no escaping alias. Program outputs carry a generation and
liveness state so a later invocation cannot overwrite a still-live result.

## Program Key

A `VulkanGraphProgramKey` includes only fields that can change execution:

```text
exported graph hash
input guard / dynamic-shape signature
selected Vulkan device and driver identity
capability profile
dtype and layout policy
partition and fusion policy versions
constant / packed-weight version identities
```

Runtime dimensions remain metadata or parameters when the selected kernels are
shape-dynamic. A new exact shape key is required only when dispatch structure,
allocation size, specialization, or a recorded command binding changes.

## Execution Stages

### Stage 1: Python Correctness Executor

The first product surface executes the lowered graph through existing eager
Vulkan kernels. Its job is to prove capture, placement, admission, output
structure, and parity while producing a lowering census. Python per-node
overhead is not a performance target.

### Stage 2: C++ Plan Executor

The C++ executor consumes an immutable lowered plan. It allocates program
slots, builds descriptors, emits barriers and dispatches, and owns completion
and retirement. No Python callback runs per node.

### Stage 3: Recorded Command Partitions

Eligible Vulkan-only partitions record command buffers against stable
program-owned slots and descriptor bindings. Recording is a transaction:

1. acquire the program memory and descriptor ownership domain;
2. populate stable inputs and state parameters;
3. record dispatch, copy, fill, and barrier steps;
4. close the recording under the same stream and owner;
5. submit and associate completion with the program generation;
6. release or recycle resources only after timeline completion.

The compiler starts with bounded partitions. Whole-forward recording is a
possible plan result, not a required topology.

## Stateful Programs

State outside recorded commands requires an explicit protocol. Examples
include KV-cache length, mutable cache tensors, RNG state, packed-weight
versions, and model buffers updated between calls.

Each stateful input is one of:

- a stable input slot copied before execution;
- a device-visible scalar/parameter updated by an execution prologue;
- a versioned constant that invalidates and recompiles the program;
- an unsupported stateful boundary that splits the graph.

HY-MT prefill and decode are separate program families. Python generation
control remains outside the tensor program unless a later graph representation
can express it without host synchronization.

## Runtime Shader Generation

The current generated elementwise shader is valid fusion-codegen evidence, but
its `glslc` subprocess and filesystem cache are development plumbing. A
production graph runtime compiler must own:

- compiler availability and version identity;
- in-memory source and SPIR-V ownership;
- deterministic program keys;
- device-capability validation;
- memory and persistent pipeline caches;
- concurrent compile de-duplication;
- negative-cache and error reporting.

Production execution must not depend on a manually configured executable path.
Static existing shaders remain the first graph instruction set; runtime shader
generation is introduced after the plan executor is correct.

## Correctness Gates

Every graph program reports:

- captured and lowered node counts;
- Vulkan and explicit CPU partition counts;
- unsupported nodes and reasons;
- transition histogram and bytes;
- memory-slot and peak-byte plan;
- selected fusion and kernel plans;
- submit, readback, fallback, and copy counters;
- numerical parity against eager execution for validation runs.

The first run may use an opt-in eager A/B self-check. A failed self-check
invalidates the program and reports the first divergent graph node or output;
it never silently promotes the plan.

Randomized family tests remain required for kernel and semantic admission.
Corpus models validate graph coverage and integration; they do not create
model-name production routes.

## Migration And Deletion

The replacement order is:

1. establish a clean eager correctness baseline;
2. land CPU export, lowering, and the Python correctness executor;
3. land the C++ plan executor and explicit memory ownership;
4. record bounded graph partitions;
5. move fusion to graph rewrites and generated graph codegen;
6. migrate tests and corpus paths;
7. delete replaced eager deferred bridges, replay/compiled-session APIs,
   stack-region proof/canary machinery, and model orchestration.

Do not delete a correctness fallback before its replacement has parity. Do not
expand a legacy system while waiting for its replacement.

### Legacy Sunset Gates

| Legacy surface | Disable condition | Delete condition |
| --- | --- | --- |
| Speculative eager deferred bridges | Default eager has a concrete route and the graph executor rejects deferred registration. | Equivalent graph lowering has corpus parity, dynamic-shape coverage, repeated-run lifetime safety, and zero fallback/readback escapes. |
| Runtime deferred elementwise placeholders and consumer hooks | Graph fusion owns the equivalent semantic chain. | Program-owned graph codegen and plan execution replace every live caller; no graph or eager default path depends on a materialization hook. |
| Stack proof and canary machinery | Bounded graph regions own values, transitions, and output generations. | Graph programs own memory, descriptors, and completion retirement across the affected corpus paths. |
| Quarantined replay and compiled-session bridges | Generated bounded-region execution covers the corresponding workload. | Generated command-list regions have parity, explicit replay-state handling, repeated-run lifetime safety, and no remaining callers. |
| Obsolete performance environment toggles | Their route is replaced, explicitly rejected, or permanently default-off. | The toggle has no live production caller or its replacement is covered by a graph-plan/corpus gate. |
| Model-oriented VisionBlocks orchestration | Export lowering covers its semantic operations without model-name routing. | DepthExtractor and relevant corpus paths use graph programs with parity and no public runtime callers remain. |

## Process Rules

- The unit of work is a lowering pass, graph-plan feature, executor feature, or
  deletion enabled by a replacement.
- New eager deferred placeholders and per-consumer materialization protocols
  are forbidden.
- New public replay or compiled-session bridge APIs are forbidden.
- New model-name production orchestration is forbidden.
- Performance canaries do not accumulate on main. A successful canary becomes
  a contract-backed plan; a rejected canary becomes evidence and its runtime
  toggle is removed.
- The primary integration scoreboard is graph coverage, explicit partitioning,
  parity, transitions, submits, and latency across the five-model corpus.
