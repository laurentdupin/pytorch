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
- inlines inference/gradient wrappers only when they preserve inference-mode
  semantics;
- creates packed linear and convolution contexts outside timed execution;
- materializes static factory expressions and lifted tensor literals as
  deduplicated graph-owned constants;
- proves full-rank static advanced indices are row-major identity maps before
  replacing them with views;
- rewrites static GQA head repetition to the generic rank-4 Vulkan kernel
  rather than representing the intermediate rank-5 expand;
- places lifted parameters, buffers, constants, and runtime inputs on the
  selected Vulkan device through an explicit placement report;
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
explicit planning model domain, execution phase, and layout/shape preferences
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

`VulkanGraphPlan.v9` is the current bounded implementation slice. It consumes
tensor inputs and a graph-owned immutable instruction/constant table, dispatches
non-mutating Vulkan or composite operators in C++, and tracks IValue SSA
use-count, last-use, liveness, and Tensor output escape. Instructions may have
any schema-declared boxed return count or no return for an ordered effect.
Multi-schema returns occupy adjacent SSA slots in schema order, and a
constant-index `getitem` aliases the selected slot without adding a runtime
instruction. A constant-index `getitem` over a represented list value becomes
an internal list-projection instruction with Python-compatible negative-index
normalization and runtime bounds checking. A schema-typed list recipe assembles
a flat homogeneous dynamic argument from SSA and constant leaves before boxed
dispatch; all leaves participate in normal lifetime accounting. A zero-leaf
recipe materializes an empty list with the schema's element type, preserving
legal default sentinels that cannot be inferred from an untyped Python list.
Per-instruction graph scopes reject fallback, readback, deferred-value
creation, and non-Vulkan Tensor results. Eligible
multi-instruction graphs therefore cross Python once per invocation rather than
once per node. The compiler reports why a graph remains on the Python executor;
it does not silently mix the two execution modes.

Each program also owns a `VulkanGraphPlanningContext`. Its model domain,
execution phase, packed-layout preference, and optional fixed graph-input shape
participate in the program key and are stored in the C++ plan. The same native
request scope covers packed-context creation during lowering and input
placement plus execution during invocation. Generic/none is the neutral
default; vision may select per-operator phases or an explicit backbone/decoder
phase, while LLM programs must declare prefill or decode. These are semantic
fields, not model-name routes. A fixed graph-input shape binds the first Tensor
input and is checked both at export and every invocation.

Python numeric literals bound to Tensor schema arguments are canonicalized to
CPU 0D Tensor constants before plan construction; Vulkan eager kernels already
admit this cross-device scalar form without fallback or readback.
Graph-classified integer `add`, `sub`, `mul`, and `floordiv` instructions use
checked C++ arithmetic with Python floor semantics and no dispatcher or Python
callback. Non-integer operands, overflow, and division by zero fail closed.

When every instruction uses the normal Context ownership path, v9 executes the
invocation inside one `GraphProgramInvocationScope`. The scope rejects
unowned flush, sync, fenced recording, or cross-thread use. Frequency and
large-linear maintenance boundaries request owner-serviced checkpoints, which
run only after the current instruction and its last-use releases. Each
partition retains its real stream timeline token, and the plan records the
final token with one invocation generation. Plain eager execution retains the
16-job submit cadence, while a graph invocation requests a frequency checkpoint
after 32 recorded jobs. Large-linear maintenance waits for the exact partition
token before releasing its captured cache batches; frequency-only checkpoints
remain asynchronous, and returning a graph output does not imply that the final
token has already completed. Command-free metadata plans complete with no
fabricated token. Direct-buffer `aten::lift_fresh_copy` and
`VulkanGraphRegionPlan` instructions execute inside this ownership path.
Linear regions record into the current partition. Bounded conv regions submit
their work into the outer partition and associate private scratch with its next
submission token. A scratch slot remains pending and cannot be reused until
that token is assigned. Abort submission resolves the same observer before
unwinding. A direct region invocation outside a graph plan still creates and
submits its own private transaction. Eager-only pending-retirement
checkpoints in LayerNorm, pool, and reduction-dimension softmax defer to an
active outer graph scope; plain eager execution retains those checkpoints.

The first bounded SSA storage-reuse rule applies functional `aten::relu`
in-place only when its input is a non-escaping value at exact last use. The
runtime additionally requires unique Vulkan storage and rejects graph inputs,
constants, duplicate live TensorImpl references, and shared metadata views.
The plan reports both candidate instructions and cumulative accepted reuses.
This is a used executor lifetime rule that pays for the 32-job cadence; it is
not a preallocated arena or a general in-place rewrite.

The immutable plan also owns a reusable host invocation workspace: boxed SSA
values, byte liveness state, a bounded dispatcher stack, and typed list
argument containers whose operator cannot return an aliasing list. Instructions
with list-valued returns retain transient list containers. A scope-exit reset
releases every live boxed value and clears all owned containers after either
success or failure. Concurrent invocation remains rejected, so one workspace
per immutable plan is sufficient.

The v9 plan also owns a bounded Vulkan tensor-resource arena for non-escaping
exact-shape fp32 results written by linear and add-layernorm plan instructions.
The compiler unions SSA values through operator-schema alias annotations,
extends each candidate's last use across its alias component, and rejects the
candidate if any alias escapes. Exact descriptor matches share a stable slot.
Two arena generations bound flight: reuse requires the preceding submission to
be complete plus exclusive TensorImpl and Vulkan storage ownership for every
slot. A failed exclusivity check spills to ordinary dispatch rather than
overwriting storage. Success associates the arena with the real final timeline
token; partial failure poisons it. Plan destruction releases completed safe
buffers immediately or retires them against the recorded token, and separately
counts unsafe slots and retirement failures.

The v9 plan does not yet implement deeper or non-list dynamic containers,
projection from nested, tuple, or dictionary values, general dtype/rank/dynamic
resource descriptors, descriptor/barrier construction, recorded commands, or
concurrent/multi-invocation flight. Convolution is deliberately not a stable
resource writer: its first candidate exposed internal physical-view aliases at
plan destruction and was removed. These remain Stage 2/3 requirements rather
than being inferred from boxed eager dispatch.

The corpus harness measures allocator high-water behavior separately from
arena ownership. It resets the existing residency high-water counter to current
live bytes before supported eager, first graph, and repeated graph phases. The
repeat intentionally keeps the prior output live. This can prove that the
current boxed executor does not materially exceed the supported memory
baseline; it does not claim preallocated slots, stable addresses, or reusable
descriptor bindings.

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
- a device-resident output leaf bound to a later guard variant's input leaf;
- a device-visible scalar/parameter updated by an execution prologue;
- a versioned constant that invalidates and recompiles the program;
- an unsupported stateful boundary that splits the graph.

HY-MT prefill and decode are separate program families. Python generation
control remains outside the tensor program unless a later graph representation
can express it without host synchronization.

The immutable C++ graph plan runs both four- and five-token HY-MT prefill to
completion with zero graph fallback or readback. The graph has no stateful
mutable operator after graph preparation: all 64 `aten::detach_` nodes are
proven to consume single-user chains rooted at `aten::lift_fresh_copy` and are
rewritten to functional `aten::detach`. Input aliases and branched fresh values
remain mutable and fail closed. A caller-owned exact-SHA probe transfers
top-level submission and completion ownership across lifted copies and 225 linear
contexts, including bounded large-linear maintenance checkpoints. Exact-SHA
caller-owned evidence covers numerical parity, the guard variant, repeated live
outputs, and peak memory. Resource arenas, descriptors, explicit barrier
construction, latency no-regression, and lane parity remain Phase 5 work.

The checked HY-MT decode evidence uses two explicit guard variants. Each C++
plan accepts a token, attention mask, and 64 flattened key/value Tensor leaves
and returns logits plus 64 updated cache leaves. The generic replay protocol
binds every first-step cache output directly to the second-step cache input on
Vulkan. Across the pair, 68 host uploads cover the 66 initial leaves and only
the next token and mask; the cache handoff performs no host upload or readback.
Both generations remain numerically valid and the first generation's outputs
survive the second. This proves explicit output-to-input state transfer, not a
mutable in-place cache, a persistent resource arena, or latency parity. The
30-sample decode medians remain 41%-65% slower than plain eager.

DAv2 uses the same fail-closed preparation principle for two exported
`aten::relu_` nodes. Each source must be a single-use result of a non-mutating
operator with one non-aliasing Tensor return before the node may become
functional `aten::relu`. Placeholder inputs, view returns, and branched fresh
values remain mutable. Exact-SHA normal and alternate DAv2 runs execute a
404-instruction immutable C++ plan with exact graph-versus-eager parity and zero
fallback, readback, or deferred values. Its 12 linear/GELU and eight
conv/ReLU/conv region calls use one outer owner per invocation. Exact-SHA
liveness-owned ReLU reuse and the 32-job cadence record 20 owner checkpoints
and 24 total submits per two-run shape, including zero retire-drain submits.
Thirty-sample graph medians are 40.14 ms and 41.91 ms versus eager medians of
114.08 ms and 116.76 ms, with graph p95 below eager p95 and no timed fallback
or readback. Graph allocator high-water ranges from 0.9% to 3.2% above eager
across the first and repeat-with-prior-output-live phases. This transfers
execution and top-level submission/completion ownership and clears
recorded-shape latency and peak-memory no-regression, but does not provide a
program memory arena or descriptor ownership.

Exact-SHA `8b60bf3ba4a` additionally preallocates the host invocation workspace.
The DAv2 plan reports 425 boxed value slots, 33 alias-safe reusable list slots,
and stack capacity eight. Its 30-sample graph medians are 44.21 ms and 42.09 ms
versus eager medians of 133.32 ms and 122.63 ms; the same 20 owner checkpoints
and 24 total submits remain visible across each two-run shape. This is structural
fixed-cost removal without a separate latency claim and does not change the
resource-arena or recorded-partition work that remained at that exact SHA.

The graph preparation path now also removes static inference identities before
placement and C++ plan construction. `aten::dropout` is replaced by its source
only when probability and training are static, probability is valid, and either
training is disabled or probability is zero. The report records every lowered
or skipped candidate and participates in the program key. Exact-SHA
`46ece5d7dc9` removes 48 DAv2 instructions and value slots while preserving the
same submissions, memory phases, and numerical results. This avoids fixed boxed
dispatch; it does not replace resource-slot or recorded-command ownership.

Exact-SHA `e00b4f0aa8b` validates the v9 resource arena on both DAv2 guards.
Each reports four stable slot descriptors covering 80 planned resource values,
58 eligible writer instructions, and 13 alias-extended lifetimes. Graph output
is bit-exact with eager Vulkan, and 30-sample graph medians are 46.02/54.09 ms
against eager at 145.48/146.75 ms. The ten-minute gate below completes 8,372
checked invocations and 33 guard recaptures with 33 immediate arena releases,
zero unsafe-slot leaks or retirement failures, and bounded live/high-water
memory. This is the supported linear/add-layernorm resource arena; it is not a
claim about convolution, descriptor reuse, or recorded command partitions.

PaddleOCR represents the schema-default empty `avg_pool2d` stride as a
schema-typed zero-leaf list recipe. Exact-SHA normal and alternate runs execute
a 290-instruction immutable C++ plan with exact graph-versus-eager parity and
zero fallback, readback, or deferred values. Exact-SHA liveness-owned ReLU reuse
and the 32-job cadence record 22 owner checkpoints and 26 total submits per
two-run shape rather than one unbounded command partition. Thirty-sample graph
medians are 42.61 ms and 54.94 ms versus eager medians of 135.79 ms and
145.76 ms, with graph p95 below eager p95. Graph allocator high-water ranges
from 1.4% to 4.3% above eager and stays inside the 5% gate. It transfers boxed
execution and top-level submission/completion ownership and clears
recorded-shape latency and peak-memory no-regression, but
not program memory or descriptor ownership.

Metadata-only `aten::sym_size.int` reads also execute through the C++ plan as
integer IValues using their composite registration. The bounded pure-integer
instruction set consumes those values directly, while every other Python
operator kind remains outside the plan with an explicit unsupported-node
reason.

## Runtime Shader Generation

The retired eager runtime-elementwise experiment remains historical
fusion-codegen evidence, but its `glslc` subprocess and filesystem cache are
not a production execution path. A future production graph runtime compiler
must own:

- compiler availability and version identity;
- in-memory source and SPIR-V ownership;
- deterministic program keys;
- device-capability validation;
- memory and persistent pipeline caches;
- concurrent compile de-duplication;
- negative-cache and error reporting.

Production execution must not depend on a manually configured executable path.
Static existing shaders remain the first graph instruction set. Their
configure-time generator tracks every GLSL source, included header, template
YAML file, and the generator script as CMake dependencies, so incremental
builds regenerate `spv.cpp` after source changes. Runtime shader generation is
introduced only after the plan executor is correct.

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

### Long-Session Soak Gate

Program-owned resources and recorded partitions must pass a backend-owned
application-shaped soak before they can become supported defaults. The
qualifying gate uses DAv2 `vits` on the RX 9070 for at least 600 seconds and
3,000 completed invocations. It performs a real output readback and numerical
parity check for every frame, recompiles alternating normal and guard-variant
programs every 250 invocations, and samples Vulkan allocator live/high-water
bytes every 50 invocations.

The memory reference is a preflight that already overlaps the old and new
program during replacement. The measured session must keep final live bytes
and the replacement high-water peak within 5% of their respective references,
record zero CPU fallback, and perform every scheduled periodic recapture. Runs
shorter than ten minutes or collected on another adapter are diagnostic-only
and cannot satisfy the gate. This exercises long-session drift, allocator
fragmentation, recapture lifetime, and per-frame observation without requiring
DeepDesktop packaging or changing the application repository.

The current accepted result is exact-SHA `e00b4f0aa8b` with
`VulkanGraphPlan.v9` on RX 9070. It ran 600.540 seconds, checked 8,372
invocations, performed 33 alternating recaptures, and recorded one final
readback per frame with zero fallback or unexpected readback. All 33 retired
resource arenas released immediately; unsafe-slot and retirement-failure
counters remained zero. Final live bytes were 555,343,312 against a
673,545,566-byte limit, soak high-water was 666,539,792, and the replacement
preflight high-water was 662,898,080 against a 696,042,984-byte limit. The
preceding `5d9001ebcc7` convolution-inclusive candidate failed both memory
bounds and is retained only as rejection evidence.

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

### Cleanup Ledger

`docs/vulkan/CLEANUP_POLICY.md` defines the four-state lifecycle and deletion
evidence policy. `docs/vulkan/cleanup_ledger.json` is the authoritative
classification for live schemas, custom classes, environment reads, Python
entry points, and code-only migration units. The generated reachability
inventory prevents unclassified or stale surfaces.

Cleanup is a background track. VisionBlocks, replay/compiled-session,
inference-graph, and stack-era deletion gates are consequences of the Phase 5/6
graph executor and corpus parity work; they are not a cleanup sprint that must
finish before executor development.

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
