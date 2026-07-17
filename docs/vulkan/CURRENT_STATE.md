# Vulkan Current State

Architecture decision, 2026-07-09: the Vulkan performance path is now
graph-first. CPU `torch.export` capture, contract-driven graph lowering, a
generic `VulkanGraphProgram`, program-owned memory/descriptors, and recorded
command partitions replace further expansion of eager deferred placeholders,
stack proof canaries, replay bridges, and model-specific orchestration. See
`docs/vulkan/GRAPH_RUNTIME.md` for the source-of-truth design and
`docs/vulkan/ROADMAP.md` for migration gates. Cleanup is the background track
defined by `docs/vulkan/CLEANUP_POLICY.md`; the exact live-surface states are in
`docs/vulkan/cleanup_ledger.json`. The historical evidence below is retained to
guide migration and deletion; it is not an instruction to expand the
superseded systems.

The checked-in DAv2 and PaddleOCR graph evidence was refreshed at source commit
`4b688faac338f3784a1286a327292735a3b334b0` against `torch_cpu.dll` SHA-256
`537802036062d3277a4d74ad7a27f28a76a16f7f4a4022c1ff1e132052989a9f`.
Both corpora execute complete v8 C++ plans on both recorded shapes with exact
graph-versus-eager Vulkan parity, zero unsupported nodes, and zero graph-runtime
fallback, readback, or deferred-value creation. DAv2 executes 404 instructions
and proves that all 12 exported `linear_gelu_none` candidates lower without
rejection. All 20 region calls reuse the outer graph owner. Each two-run shape
records two scopes and final tokens, 20 owner checkpoint flushes, no
retire-drain submits, and 24 total queue submits. The immediately preceding
supported artifacts recorded 26 flushes and 30 total submits before dead-input
ReLU reuse made a 32-job checkpoint cadence memory-safe, and 38 flushes and 42
total submits before generic next-submission token inheritance removed six
bounded conv-region exits per inference. The earlier graph artifact recorded 16
nested scopes, 56 retire-drain submits, and 92 total submits. PaddleOCR remains
the GELU control,
represents its
schema-default empty `avg_pool2d` stride as a typed zero-leaf list recipe, and
executes 290 instructions. Its top-level plan records two scopes and final
tokens per two-run shape, 22 owner checkpoint flushes, two input uploads, two
output readbacks, 26 total queue submits, and no normal-frequency or
retire-drain submits. A checked exact-SHA HY-MT artifact transfers
lifted-copy and large-linear checkpoint submission ownership across its
complete plan; its evidence and remaining deletion-gate blockers are described
below.
The same checked-in cases record allocator high-water phases for eager, first
graph execution, and repeated graph execution with the prior output live. DAv2
graph peaks range from 0.9% to 3.2% above eager. PaddleOCR graph peaks range
from 1.4% to 4.3% above eager. All recorded graph phases remain within
the evidence gate of 5% above their same-process supported eager peak. This
proves bounded peak-memory parity for the recorded shapes, not program-owned
arenas or stable allocation addresses.

The first qualifying RX 9070 long-session gate now establishes the standing
lifetime baseline for Phase 5 and Phase 6 changes. At source commit
`fa22f619aa7d4f5f618493f98032be99dd878743`, using the source-tree runtime
built from `c8af28dac3e73a0b0f9dc894e91ab4aecfee6f92`, DAv2 `vits_140` ran for
600.539 seconds and completed 8,501 consecutive graph invocations. Every
invocation performed CPU parity and a final output readback; the run also
compiled 34 periodic replacement guard variants while the preceding program
remained live. It recorded zero CPU fallbacks, zero unexpected sync readbacks,
and exactly 8,501 `tensor_cpu_readback` submit origins. Live bytes ended at
547,571,728 against a preregistered 669,645,816-byte limit. The soak high-water
was 663,756,560 bytes against a 697,582,141-byte replacement-overlap limit and
the measured replacement preflight high-water was 664,363,944 bytes. All ten
registered checks passed. This proves bounded long-session behavior for the
current boxed v8 executor under fixed-shape execution, output observation, and
periodic program replacement. It does not prove program-owned arenas, stable
resource addresses, recorded command partitions, or multi-invocation flight.

The raw census and parity artifacts are under
`agent_space/graph_long_session_soak_exact_fa22f619aa7/`; their SHA-256 values
are `3b3c19c9d4f12316cefe7ee8dadbae302d5b3bbd1952519e911f96c336c8898c`
and `28080e3b5a8ef4c0e9f2e2967e57aab63cc94b607798d81b2bd1b16ebc68933d`.
The artifact records `torch_cpu.dll` SHA-256
`0871584acbed7f93d3c9208d2a5305147f1990b7e61e395c13330e6906a98f73`,
`torch_python.dll` SHA-256
`5118f781e75321947647467738a6bac8a04bfcb932da9a3869c1d35bf2f29f9a`,
and the loaded CPython extension SHA-256
`757bf8097feed0efe58615be57d62b4d2fe288b9060353fab3aa905906c9d8d1`.
The adapter was `AMD Radeon RX 9070`, Vulkan API 1.4.349, driver version
8,389,003. This is the comparison baseline for the resource-ownership batch;
the gate must be rerun after that batch rather than treating this result as its
completion evidence.

The first exact resource-arena candidate at source commit
`5d9001ebcc78c84841cfbf3fc1974aee3e9c66fe` assigned exact-shape fp32
linear, convolution, and add-layernorm outputs to stable plan-owned slots. Its
600.527-second RX 9070 soak completed 6,922 checked invocations and 27 periodic
recaptures, but failed both registered memory bounds: final live bytes reached
775,041,616 against a 700,080,427-byte limit and high-water reached
885,930,896 against a 724,772,319-byte replacement limit. Follow-up per-slot
retirement counters and writer-family isolation found that convolution's
internal physical views left eight unsafe slots for the normal guard and two
for the alternate guard when a plan died. The convolution stable-output writer
was rejected and removed; it may return only after those physical-view aliases
have explicit ownership and repeated plan destruction records zero unsafe
slots. The failed exact artifacts are under
`agent_space/graph_resource_slots_exact_5d9001ebcc7/`; the census and parity
SHA-256 values are
`17ce5a424cd04fc3b62711d5f597ba25cd5bebc14e4fddf4f683c705d7495199`
and `309b754e28e3208463305b04402cde88c4ae4c0b3eaac540a1e5514ed9fbb81c`.

The accepted follow-up at source commit
`e00b4f0aa8bd7f0ae2a0885cc5eed7c4cd170353` is
`VulkanGraphPlan.v9`. It assigns only non-escaping exact-shape fp32 linear and
add-layernorm outputs to a two-generation stable resource arena. Schema alias
components extend last use and reject an otherwise eligible writer if any
alias escapes. Runtime reuse additionally requires the preceding submission
to be complete and every slot TensorImpl/storage to be exclusive; otherwise
the invocation spills instead of overwriting live storage. Success records the
real final submission, partial failure poisons the generation, and plan
destruction independently releases or retires every safe slot while counting
unsafe or failed retirement.

Both DAv2 guards report four stable slot descriptors covering 80 planned
resource values, 58 eligible writer instructions, and 13 alias-extended
lifetimes. Their graph outputs remain bit-exact with eager Vulkan. Thirty-sample
graph medians are 46.02/54.09 ms versus supported eager at 145.48/146.75 ms;
graph p95 is 62.28/66.32 ms versus eager at 169.99/176.55 ms. The qualifying
RX 9070 soak ran for 600.540 seconds, checked all 8,372 invocations, compiled
33 alternating replacement variants, and recorded 33 immediate arena releases
with zero unsafe-slot leaks, retirement failures, fallback, or unexpected
readback. Final live bytes were 555,343,312 against a 673,545,566-byte limit;
soak high-water was 666,539,792 bytes, and replacement-preflight high-water was
662,898,080 against a 696,042,984-byte limit. All registered checks passed.
The exact census and parity artifacts are under
`agent_space/graph_resource_slots_exact_e00b4f0aa8b/`; their SHA-256 values are
`8651317727e37fad51c76cc17681e2e6a899b729005021625baac6b083ca3e48`
and `8bb1c4b857c73a934f3b2d346e42481785e0f41002e955ccd4d12d1940ce7f0c`.
The loaded `torch_cpu.dll` SHA-256 was
`257a0147bdd919a1ed54a336efb279c48d210af6f4b5a04fb783c2041bc911e5`.
This promotes the bounded linear/add-layernorm arena, not convolution,
descriptors, explicit barriers, recorded commands, arbitrary dtype/rank, or
concurrent/multi-invocation flight.

The same cases measure supported-default latency from preuploaded Vulkan inputs
to completed Vulkan outputs, alternating plain eager and `VulkanGraphProgram`
for three warmups and 30 samples per surface. DAv2 graph medians are 40.1 ms
and 41.9 ms versus eager medians of 114.1 ms and 116.8 ms. PaddleOCR graph
medians are 42.6 ms and 54.9 ms versus eager medians of 135.8 ms and 145.8 ms.
Graph p95 is below eager p95 in all four cases, with zero timed fallback or
readback. This establishes latency no-regression for the recorded shapes, not
the full corpus or eligibility to delete a Migration subsystem.
An exact-SHA `ed4975687b6` RX 9070 attribution pass explains why the DAv2 graph
times are flat across those sizes. Summed GPU timestamps rise from 23.9 ms for
`vits_140` to 32.7 ms for `vits_280`, while its 30-repeat uninstrumented
medians remain 49.06 ms and 49.09 ms. Both shapes issued 24
`pending_command_flush` checkpoints per inference; the profiling pass added one
timestamp-reset submit. CPU timeline summaries attribute about 3.4 ms and
2.9 ms per inference to measured dispatch recording plus submit calls. The
remaining 140-size slack is therefore a fixed submission/driver/queue floor,
not evidence for another exact operator kernel.

The exact-SHA `25b66ba0b8b` graph cadence separates eager's frequency of 16
from a graph frequency of 24. DAv2 now issues 19 graph checkpoints per
inference, and same-binary 30-repeat medians are 40.20 ms and 36.78 ms, 18.1%
and 25.1% below the `ed4975687b6` attribution medians. The corresponding
`25b66ba0b8b` artifacts kept graph peak memory within 1.2% below to 0.4% above
eager.
Candidate graph frequencies of 64 and 32 reduced submission further but were
rejected after DAv2 or PaddleOCR repeat-with-live-output memory exceeded the 5%
gate. Exact-SHA `b157c550fc5` next-submission token inheritance for bounded
conv-region scratch reduces DAv2 from 19 to 13 pending submissions per
inference. Normal and alternate graph medians are 42.10 ms and 40.97 ms against
eager at 116.13 ms and 121.64 ms, and repeat-with-live-output peak memory
remains 0.8% and 1.8% above eager. PaddleOCR remains at 14 submissions and
HY-MT remains at 114 because neither path captures this scratch. Every recorded
memory phase stays inside the 5% gate. Timestamp-instrumented wall
time is not a production latency baseline; the checked-in 42.1/41.0 ms
distributions are the supported deletion-gate artifact.

The current exact-SHA implementation uses SSA last-use plus unique
Vulkan-storage ownership to reuse dead ReLU inputs, then widens the graph
cadence from 24 to 32 jobs. DAv2 falls from 13 to 10 submissions per inference;
30-sample graph medians are 40.14/41.91 ms against eager at 114.08/116.76 ms,
and graph peak memory is 0.9% to 3.2% above eager. PaddleOCR falls from 14 to 11
submissions; graph medians are 42.61/54.94 ms against eager at 135.79/145.76 ms,
and peak memory is 1.4% to 4.3% above eager. Caller-owned HY-MT falls from 114
to 88 submissions and remains between 4.0% below and 0.05% above eager peak
memory. All graph paths retain their recorded correctness and zero
fallback/readback properties. The checked-in DAv2 and PaddleOCR manifests now
make this the supported deletion baseline. Raw exact-SHA files are under
`agent_space/graph_dead_relu_reuse_checkpoint32_exact_4b688faac33/`.

Exact-SHA `1fb325d1d0c` removes per-instruction heap allocation for the C++
executor's fallback counters and boxed argument stack without changing the
Python diagnostic surface. A 30-sample DAv2 control retains 10 submissions per
inference and identical peak memory; graph medians are 39.10/40.07 ms against
eager at 110.64/111.37 ms. The graph/eager ratios remain within 0.4% of the
checked baseline, so this is accepted as a structural fixed-cost removal and
latency no-regression, not as a separately measurable speedup.

Exact-SHA `8b60bf3ba4a` removes the remaining per-invocation allocation of the
C++ executor's boxed SSA values, liveness state, argument stack, and alias-safe
typed list recipes. The immutable DAv2 plan owns 425 value slots, 33 reusable
list slots, and stack capacity eight; 20 list arguments remain transient
because their instructions return a list and may alias an input container.
Scope-exit cleanup releases live values after success or failure. The exact run
retains 10 submissions per inference, exact graph/eager parity, and 0.9% to
3.2% graph high-water overhead. Graph medians are 44.21/42.09 ms against eager
at 133.32/122.63 ms. Because both surfaces moved with host load across runs,
this is fixed-cost ownership and latency no-regression evidence, not an
isolated speedup claim. Raw files are under
`agent_space/graph_executor_workspace_exact_8b60bf3ba4a2/dav2/`.

Exact-SHA `46ece5d7dc9` removes statically proven identity `aten::dropout` before
tensor placement and plan construction. Valid static probability plus either
disabled training or zero probability is required; training dropout and invalid
probabilities remain untouched. DAv2 elides all 48 candidates and reduces the
immutable plan from 404 instructions/425 values to 356/377. Exact parity, 10
submissions per inference, and the 0.9% to 3.2% graph high-water envelope remain
unchanged. Canonical 30-sample graph medians are 44.73/50.12 ms against eager at
138.50/142.87 ms. The averaged normalized ratio is unchanged from the prior
exact run, so this is accepted fixed control-plane removal, not a separately
measurable speedup. The checked DAv2 manifests are refreshed from
`agent_space/graph_static_inference_identity_exact_46ece5d7dc93/dav2_checked/`.

Exact-SHA `e536f16cf36` extends the same identity pass only to `aten::detach`
nodes carrying the preceding fresh single-user detach-functionalization proof.
Unproven input and view detaches remain visible. The caller-owned GTX 1080 HY-MT
prefill artifact lowers all 64 proven candidates and reduces both plans from
2,732 instructions/2,466 values to 2,668/2,402. Four- and five-token output
errors are identical to the prior exact artifact, graph fallback/readback stays
zero, and the 88-checkpoint cadence is unchanged. First/repeat high-water ranges
from 4.0% below to 0.035% above eager. Single graph samples of 1.77/2.10 s move
in opposite directions relative to prior runs and do not establish a latency
win or distribution gate. Raw evidence is under
`agent_space/hymt_static_detach_identity_exact_e536f16cf36/`.

The checked exact-SHA `019faaebf1593fd2f2fcbd8e5cec66a8202fd62e` GTX 1080
follow-up records three warmups and 30 alternating samples per surface for both
guards. Four-token medians are 2,193.66 ms graph versus 1,742.38 ms eager;
five-token medians are 1,404.20 ms graph versus 1,786.36 ms eager. Graph p95 is
2,232.31/1,461.69 ms versus eager at 1,759.69/1,797.07 ms. Both graph programs
retain numerical parity, zero fallback/readback, 88 submissions per inference,
and first/repeat high-water from 4.0% below to 0.04% above eager. The mixed
latency result does not clear no-regression. Graph execution is explicit
`LLM`/`Prefill`, while plain eager still selects the legacy `DepthDiffusion`
attention lane and records five fallbacks plus one readback per timed
invocation. The checked artifacts close the missing HY-MT distribution-evidence
item but keep the lane and latency deletion gates open.

The checked exact-SHA `79bf8d01ef0db5c01997042071e12434eac1b443` HY-MT
decode follow-up accepts the exported runtime PyTree for a token, mask, and 64
flattened cache tensors. Separate first-step and second-step guard variants
compile complete 2,732-instruction/2,530-value C++ plans with 66 inputs and 65
outputs. Both retain eager/CPU numerical parity, zero graph fallback/readback,
and first/repeat peak memory from 4.03% below to 0.02% above eager. The explicit
state protocol maps all 64 first-step cache outputs to second-step inputs on
Vulkan. Across the chained pair, 68 uploads cover 66 initial leaves plus only
the next token and mask; there is no cache readback or reupload. Both invocation
generations capture final tokens and the first generation's outputs remain
valid after the second completes.

Decode latency remains rejected. Thirty-sample graph medians are 2,188.41 ms
and 1,897.01 ms versus eager at 1,324.03 ms and 1,341.53 ms, with graph p95
also higher. Graph planning is explicit `LLM`/`Decode`; plain eager still
selects `DepthDiffusion` and records five fallbacks plus one readback per timed
invocation. This accepts explicit device-resident KV state replay and closes
the deeper exported-input-container gap, but does not clear latency, lane,
recorded-resource, replay/compiled-session, or legacy-inference deletion gates.

Exact-SHA `c8332a964bb` moves SSA release discovery into immutable plan
construction. Every non-escaping value with a last use is assigned exactly once
to an instruction release list, and construction validates bounds, uniqueness,
escape state, and last-use equality. Invocation no longer revalidates the whole
immutable schema or rescans arguments and outputs to rediscover releases. DAv2
retains its 356-instruction/377-value plan, exact graph/eager parity, zero graph
fallback/readback, 10 pending submissions per inference, and the identical
0.9%-3.2% high-water envelope. Two exact 30-repeat passes moved with ambient
host load; their combined graph/eager median ratios are 0.348/0.344. This is
structural lifetime ownership and behavioral no-regression evidence, not an
isolated latency claim or a legacy-subsystem deletion gate. Raw evidence is
under `agent_space/graph_release_schedule_exact_c8332a964bb/`.

A 64-job cadence was re-probed after dead-ReLU reuse and rejected again. It cut
DAv2 to five submissions per inference, but graph peak memory rose to
5.6%-6.1% above eager for the normal shape and 8.5%-9.9% for the alternate
shape. Graph medians also worsened to 42.12/43.07 ms. The supported default
therefore remains 32 jobs. The intermediate 48-job cadence was also rejected:
it reduced DAv2 from 10 to 7 submissions per inference, but graph medians
worsened to 44.57/48.77 ms and alternate repeat-with-live-output peak memory
reached 6.9% above eager. Reducing the remaining submit floor requires stronger
generic lifetime/resource reuse or recorded command partitions that preserve
useful CPU/GPU overlap, not a wider unbounded checkpoint interval. Raw rejection
evidence is under `agent_space/graph_checkpoint64_post_reuse_worktree/dav2/`
and `agent_space/graph_checkpoint48_probe_worktree/dav2/`.

The GELU `none` CPU tolerance is documented in
`docs/vulkan/GRAPH_EVIDENCE.md`; it reflects the existing eager tanh-kernel
behavior rather than a graph-only approximation.

The Python correctness executor now normalizes false inference/gradient
wrappers, creates packed contexts before generic state placement, materializes
static factory expressions and lifted tensor literals as graph-owned
constants, and records explicit input/constant tensor placement. Bool graph
inputs and constants upload directly to width-packed Vulkan buffers without a
CPU fallback or readback. Full-rank static advanced indexing lowers to a view
only when its row-major offsets prove identity order; reordered indices remain
visible and unsupported. The exact static `unsqueeze -> expand -> reshape`
form used for GQA head repetition lowers to the generic `GQARepeatContract`
kernel family rather than constructing an unsupported rank-5 Vulkan value.

The exact-SHA `4b688faac33` GTX 1080 HY-MT prefill artifact captures 3,160
nodes, lowers 225, and reports zero unsupported nodes at lower time. Both the
four-token case and the guard-recompiled five-token case execute complete
2,732-instruction C++ plans, return 65 tensor outputs, and stay within the
recorded eager/CPU tolerances with zero graph fallback, readback, or deferred
values. The unaligned five-token boolean causal-mask broadcast now uses the
generic rank-bounded buffer bool path. Each two-run case records 176 owner
checkpoint flushes, four host uploads, 130 evidence output readbacks, 310 total
queue submits, no retire-drain submits, and explicit `LLM`/`Prefill` graph
planning with zero label inference. Graph peak memory ranges from 4.0% below to
0.05% above eager. Its single-sample normal/alternate graph latencies are
2,143.2 ms and 1,605.3 ms versus eager at 1,238.4 ms and 1,247.1 ms. These
single samples are regression-control evidence, not a latency distribution. The
supported eager diagnostic still resolves the legacy `DepthDiffusion` lane, so
HY-MT lane parity is not established and
`LegacyPlanningInference`/`ModelLanePolicy` remain Migration. The latency rows
have one sample per surface and do not by themselves clear the full
distribution deletion gate.
The checked 30-sample follow-up above supersedes the single-sample timing
conclusion while preserving the same open lane and latency gates.

Phase 5 now has a top-level C++ executor and bounded resource-ownership slice.
`VulkanGraphPlan.v9`
stores a fully bound immutable list of non-mutating Vulkan/composite operator
handles, graph-owned constants and contexts, IValue SSA values, ordered
zero-return effects, schema-ordered multi-return values, schema-typed
homogeneous list argument recipes, Tensor output escapes, and C++-validated
use-count/last-use metadata. Constant-index `getitem` nodes over multi-schema
returns alias the selected adjacent SSA slot without a runtime instruction.
Constant-index `getitem` nodes over a represented list value use a bounded
internal instruction with Python-compatible negative-index normalization and
runtime range checking.
The C++ runner
executes the plan without a Python callback per node, materializes dynamic
lists as typed IValues, releases every non-escaping leaf after its last
instruction, rejects concurrent invocation, and checks every instruction for
fallback, readback, deferred-value creation, or a non-Vulkan Tensor result. It
tracks liveness separately from IValue contents so a valid `None` cannot be
confused with a released slot. Repeated linear/GELU/residual,
metadata-checked, and Tensor-list concat graphs run through this path while the
Python interpreter is disabled, and earlier live outputs remain valid after
later invocations.

The v9 plan can also own a bounded sequence of normal Context recording
transactions for the whole invocation. The outer scope converts frequency and
large-linear maintenance boundaries into graph-owner checkpoints serviced only
after a complete instruction and its last-use releases. Every partition keeps
its real stream timeline value; the plan exposes the final value with one
invocation generation. Large-linear checkpoints wait for the exact partition
token before releasing their captured packed-weight and linear-context batches.
A command-free plan completes successfully with generation advancement and no
synthetic token. Direct-buffer `aten::lift_fresh_copy` and
`VulkanGraphRegionPlan` instructions now remain inside this ownership path.
Linear regions record normally in the outer partition. Bounded conv regions
associate their private scratch ring with the next outer submission token
without forcing a region-exit checkpoint. Pending slots are not reusable until
that token is assigned, and abort submission resolves them before unwinding.
Direct region calls outside a graph plan retain their private transaction.
Pool and reduction-dimension softmax retain their eager pending-retirement
checkpoints but defer those
checkpoints while this outer transaction is active.

Plan selection is fail-closed. The v9 schema accepts tensor inputs, any
schema-declared dispatcher return count, direct SSA references, flat
homogeneous dynamic list arguments, and literal or graph-owned constants. An
empty schema-list constant is represented as a typed zero-leaf list recipe
rather than an untyped `Any[]` constant. Internal returns may be non-Tensor
IValues while public outputs remain
Tensors.
Schema-typed device constants are canonicalized before boxed dispatch. Python
numeric literals bound to Tensor schema arguments are represented as CPU 0D
Tensor constants, matching the cross-device scalar form accepted by the Vulkan
eager kernels without fallback or readback. Immutable `aten::sym_size.int`
metadata reads now follow their CompositeImplicitAutograd registration into
the C++ plan as integer IValues; dynamic view shapes execute without a Python
node callback, fallback, or readback. Graph-classified integer `add`, `sub`,
`mul`, and `floordiv` instructions execute as checked C++ plan operations. They
reject non-integer operands, detect overflow and division by zero, and preserve
Python floor-division semantics for negative values. The checked-in exact-SHA
v8 DAv2 evidence crosses its former symbolic-size, floor-division,
multi-return, list-projection, and mutable-ReLU blockers. Both `relu_` inputs
are proven single-use, non-aliasing results of functional `aten::conv2d` and
are rewritten to functional `aten::relu`. The complete normal and alternate
graphs then execute as a 404-instruction C++ plan with 425 values, two ordered
effects, eight graph-scalar instructions, 20 list projections, 53 list
arguments, and one output. Both retain exact graph-versus-eager parity and zero
fallback, readback, or deferred-value creation. Mutable dispatch, mismatched
boxed argument types, deeper or non-list dynamic containers, and non-list or
nested container projections retain the Python correctness executor with an
explicit reason. A generic functionalization pass
rewrites `aten::detach_` only when every value in its single-user producer
chain is proven to lead back to
`aten::lift_fresh_copy`; input aliases, branches, and malformed chains remain
mutable and fail closed. The four-token HY-MT probe proves this condition for
all 64 exported detach mutations and now compiles the entire graph as a
2,732-instruction `VulkanGraphPlan.v8`; it contains no graph-scalar or list
projection instructions, so the probe is also a strict executor regression
check. The current v9 executor preallocates a bounded resource arena only for
non-escaping exact-shape fp32 linear and add-layernorm outputs with schema-alias
proof and two-generation reuse. It still does not own descriptors or explicit
barrier plans, support general dtype/rank/dynamic resource descriptors, support
deeper or non-list containers, or provide checked-in HY-MT resource-arena
parity/performance evidence. The submission and scoped resource transactions
are real ownership slices, but do not by themselves satisfy a Migration
deletion gate.

Fresh-ReLU functionalization is independently fail-closed: the producer must
be a non-mutating operator with exactly one non-aliasing Tensor return, and the
produced value must have only the `relu_` consumer. Placeholder inputs,
view/alias returns, mutable producers, multi-return producers, and branched
fresh values are rejected without rewriting. This is an operator/schema proof,
not a DAv2 route.

The existing GQA repeat shader also had its generic coordinate mapping fixed:
Vulkan buffer metadata orders logical coordinates as width, sequence, heads,
batch, so repetition divides the head coordinate rather than the sequence
coordinate. Generic non-corpus shapes and the randomized eager GQA suite cover
the corrected mapping. Vulkan shader sources, included headers, template YAML,
and `tools/gen_vulkan_spv.py` are now CMake configure dependencies, so an
incremental Visual Studio build automatically regenerates `spv.cpp` after a
shader or generator change.

The supported sync accounting substrate is now isolated in `SyncCounters.*`.
It owns sync and forced-sync counters, graph-program invocation accounting,
submit-origin/phase attribution, and retire-drain/call-site accounting without
changing their schemas. This preserves fallback/readback and profiling GPU
timestamp evidence independently of the stack proof/canary control plane that
remains migration-gated in `Sync.*` and `Context.cpp`.

Retained packed-weight and linear-context cache ownership is isolated in
`PackedWeightCache.*`. Migration-only KV-cache, scratch-arena, readback, and
request-storage objects remain in `ExecutionObjects.*`; deleting those objects
after the C++ graph executor owns their replacements no longer risks deleting
the supported packed-weight residency substrate with them.

Explicit planning-request construction and scope application remain in
`Request.*`, and device capability discovery remains in `DevicePolicy.*`.
`VulkanGraphPlanningContext` now carries an explicit model domain, execution
phase, packed-layout preference, and optional fixed graph-input shape in the
program key and immutable C++ plan. Graph lowering applies that request while
creating packed contexts, and every graph invocation reapplies it across input
placement, Python correctness execution, and direct C++ plan execution. A
fixed shape binds and checks the first Tensor graph input. Even the explicit
generic/none default suppresses allocation-label and tensor-shape inference;
vision and LLM contexts select their declared semantic lanes. Route decisions
derive their lane from the same resolved request stored in the runtime policy,
so an explicit graph scope cannot be discarded after policy construction.
Temporary
allocation-label reads, model-token matching, bounded LLM tensor-shape
inference, and GPU-name overrides remain isolated in
`LegacyPlanningInference.*` and `LegacyDeviceNamePolicy.h`. They cannot be
deleted until checked HY-MT and PaddleOCR lane/residency evidence proves the
explicit route on 8 GB adapters and eager callers no longer depend on inference.

The benchmark-local `python_private_baton` deep-split canary was retired on
2026-07-14. Production only ever implemented the native private-device-baton
mode, while the Python-mediated experiment overflowed the Windows stack. The
benchmark now treats that retired spelling like any other unsupported mode and
fails closed through the generic depth guard. The historical rejection reason
remains in the performance evidence manifest, and the bug-class regression test
continues to prove that the spelling cannot enable deep-stack execution.

The default-off stack-region external recording pool lease was also retired on
2026-07-14. It was slower than the supported persistent-pool path and did not
fix the compiled-session stack overflow it targeted. Stack-owned external
recording now has one pool-ownership path: persistent command and descriptor
pools reset at their existing global-completion points. The separate default
stack-planned descriptor-pool lease remains active because it fixes a proven
repeated-request lifetime bug.

The numerically blocked DAv2 fused-head shader was retired on 2026-07-14. Its
admission predicate returned false unconditionally because it did not match the
reference DPT head across supported decoder shapes. The supported eager tail
continues to use concrete conv, bilinear upsample, conv, ReLU, conv, and ReLU
operations. The broader replay and compiled-session surfaces remain
Migration-gated and fail closed; this deletion does not treat their historical
timings as a supported-default baseline.

Runtime elementwise eager experiments were retired on 2026-07-14. The removed
stack comprised the runtime glslc compiler, owned-SPIR-V shader descriptors,
live eager-chain sidecar recorder, deferred tensor placeholders, and the
materialized generated-chain uses in VisionBlocks. Standalone generated
add/mul math reached parity, but deferred stack values repeatedly failed bridge
sanity across runtime-generated and static shaders, copied input leases, and
explicit consumer materialization. That evidence rejected retrofitting output
ownership and consumer ordering into eager tensor handles.

Supported eager operators now execute concretely, and token-prefix output views
use the existing static buffer add. Future elementwise generation belongs to
graph lowering and program-owned values, descriptors, barriers, and command
partitions. The generic graph execution scope still rejects deferred-value
registration; its behavioral regression test is independent of the retired
producer. Git retains the inactive implementations and detailed experiment
history.

The old replay/compiled-session execution surface is now quarantined under
`docs/vulkan/REPLAY_RETIREMENT.md`. The DAv2 benchmark no longer exposes
`compiled_session_bridge` as a selectable stack-output bridge mode, and
governance freezes the current public `vulkan_prepack::*replay*bridge` /
`*compiled_session*bridge` API set. Replay diagnostics and stale-tensor safety
checks remain available only as migration evidence while generated command-list
regions become the replacement path.
Internal `flush_pending_cmds` cuts now use submit origin
`pending_command_flush` instead of `tensor_cpu_readback` and carry a
first-class `PendingCommandFlushReason`, separating eager linear/addmm submits,
replay input-upload visibility, replay warmup, replay submit guards, vision
output materialization, stack replay step submit guards, and temporary-clone
lifetime protection. The trace can now show why a natural lazy chain was cut
instead of collapsing all internal submits into one opaque boundary or polluting
readback counters.
`SmallSpatialPointwiseConvContract` `GenericDynamicHW` is the first adaptive pointwise
example, admitting legal fp32 direct-buffer 1x1 conv with unseen batch/H/W
under semantic 1x1/direct-buffer guards and routing it through the existing
dynamic-shape `conv2d_buffer_float_1x1` shader. Runtime pointwise legality now
uses this semantic dynamic family before sparse projection rows; the old rows
remain evidence and regression fixtures. Batch-one width-packed cases may still
select the existing as-linear plan from dynamic admission.
`ElementwiseBroadcastDirectBuffer`
also now records fp32 rank-1 through rank-4 Vulkan buffer add/mul/sub as a
dynamic semantic family when broadcasting is mathematically legal.
`SequenceCatDirectBuffer` now admits fp32 rank-4 direct-buffer dim-2 sequence
append under batch/head/head-dim equality semantics and fresh random sequence
length coverage. `InitialSequenceCatDirectBuffer` now admits the matching
empty-cache bootstrap case by requiring a Vulkan empty left operand, a fp32
rank-4 Vulkan buffer cache tensor, dim-2 semantics, and positive runtime
batch/head/sequence/head-dim values instead of the old exact `InitialCache`
sequence/head geometry row.
`LinearOrMatmulDirectBuffer` now gates generic fp32 rank-2/rank-3 direct-buffer
linear execution by semantic M/K/N compatibility instead of exact tiled-plan
evidence. `EmbeddingLookupDirectBuffer` is the sole native embedding admission
path for fp32 2D Vulkan weights when Long or Int indices are CPU-resident and host-checked
for valid range before dispatch, and for Vulkan-resident Long indices that
carry CPU-uploaded integer min/max provenance proving the runtime vocab bound.
Truly device-produced Vulkan indices remain on fallback until a no-readback
index-bounds proof or device error path exists. `PatchEmbedFloatBufferConvRoute`
now admits legal fp32 Vulkan
buffer RGB patch projections with `[1,3,H,W]` input, `[C,3,14,14]` weight,
stride `[14,14]`, zero padding, dilation `[1,1]`, and groups `1` through a
generic runtime tuple instead of exact H/W row admission. The old observed
patch-embed conv rows remain evidence and regression fixtures around the
dynamic family. `FeatureMapToTokensDirectBuffer` now admits legal fp32
direct-buffer rank-4 feature maps by `[N,C,H,W] -> [N,H*W,C]` semantics
instead of exact patch-embed H/W rows while still requiring the current
width-packed zero-offset buffer layout. The exact feature-map matcher and
generated rows are retired; corpus-shape and former-boundary behavior remains
covered directly against the semantic family.
`TokenPrefixCatAddDirectBuffer` now admits legal fp32 rank-3 prefix-token
concat plus positional add by prefix length, batch, token-count, and feature
semantics instead of the old DAv2 token-count/feature rowset. The exact matcher,
generated rows, and JSON fixture are retired; corpus shapes, promoted former
boundaries, and semantic negatives remain direct behavioral tests.
`CatAxisDirectBuffer` now admits legal fp32 buffer-backed rank-4 dim-1 cats
by batch/height/width equality, positive channel extents, and the current
channel multiple-of-4 layout constraint instead of the old input-count,
spatial, and total-channel row bounds. The old `ChannelCatContract` rows remain
evidence and regression fixtures around the dynamic family.
`NoOverlapConvTranspose2DContract` `DynamicKernelStrideFloatBuffer` now admits no-overlap
fp32 direct-buffer transposed convs by kernel/stride/layout semantics with
random batch/channel/spatial coverage in the clean packed-buffer envelope.
Low-channel no-overlap transposed conv remains a named small-metadata
materialization blocker rather than an exact-row target. `PackedBufferConv2D`
now stamps the existing metadata-packed generic `conv2d_buffer_float` path as a
semantic dynamic family for batch-one fp32 rank-4 groups-one, dilation-one
runtime shapes with positive computed output dimensions. `Conv2DDirectBuffer`
remains a direct-layout scaffold because direct output ownership is not proven
for the generic conv path; the remaining migration is explicit
layout-transition, batched-conv ownership, or direct-output ownership, not
another exact conv row.
`BatchNormInferenceDirectBuffer` now routes fp32 4D eval-mode buffer batch norm
through the dynamic program runtime under runtime N/C/H/W and feature-count
semantics; the existing random legal-shape parity coverage remains the
regression surface. `GQARepeatDirectBuffer` now behavior-authorizes the
runtime-sized repeat materialization shader for fp32 rank-4 Vulkan buffer K/V
tensors. SDPA route admission may use that materialized-GQA family only for
non-causal, mask-free GQA when both K and V match and the downstream rectangular
score tensor matches `RectangularScoresRuntimeShape`. Random unseen
materialized-GQA coverage also fixed the repeat shader to divide the output
head coordinate, not the sequence coordinate, by the repeat factor.
`SDPAScoreSoftmaxContract` now also admits rectangular fp32 rank-3 score tensors
by runtime batch-head/target/source dimensions and score-element budget, keeping
materialized-GQA probabilities on the buffer last-dim path instead of the old
known-bad texture fallback. Broader non-causal decode GQA still prefers
`DirectDecodeGQASDPADirectBuffer` when legal. That active semantic SDPA family
admits fp32 rank-4 direct-buffer
decode GQA with batch 1, query length 1, no mask/dropout, divisible
query/key-value heads, default/head-dim-equivalent scale, and direct-GQA
shader-budget dimensions without exact source-length rows.
`DirectCausalPrefillGQASDPADirectBuffer` now also admits fp32 rank-4 direct-buffer
causal prefill GQA and equal-head causal MHA with batch 1, equal query/source
sequence length, no explicit mask/dropout, default/head-dim-equivalent scale,
and direct-GQA shader-budget dimensions. GQA requires divisible
query/key-value heads; MHA is admitted only when the head counts are equal, so
the existing direct-GQA shader runs with repeat factor `1`. The causal mask is
applied in the existing direct-GQA shader, so legal causal prefill shapes no
longer need exact sequence/head rows.
`SmallNonCausalGQASDPADirectBuffer` now admits bounded fp32 rank-4 direct-buffer
non-causal GQA with batch 1, no mask/dropout, divisible query/key-value heads,
target/source lengths up to 64, default/head-dim-equivalent scale, and
direct-GQA shader-budget dimensions. The old exact small non-causal rows remain
evidence and negative-guard fixtures rather than production admission bounds.
`DirectNonCausalMHASDPADirectBuffer` now admits fp32 rank-4 direct-buffer
equal-head non-causal MHA with batch 1, no mask/dropout/GQA,
default/head-dim-equivalent scale, direct-buffer lane-aligned head/value dims,
and direct-GQA shader-budget dimensions. This converts diffusion-style square
MHA rows that satisfy the direct-buffer layout contract from finite diffusion
admission into semantic runtime-shape execution.
`VisionSelfAttentionSDPAContract` now also admits fp32 rank-3 self-attention
by semantic Q/K/V equality, head dim `64`, explicit scale `1.0`, and disabled
mask/dropout/causal/GQA; batch-head count and sequence length are runtime
values rather than the old six-row DAv2 low-resolution set.
`SDPAScoreSoftmaxContract` mirrors that runtime-shape policy for rank-3 square
vision score tensors while the probability materialization/no-clone policy
remains separately governed by transition evidence. `SDPAExecutionPolicyContract`
now admits small-head fp32 rank-4 non-causal MHA policy by runtime Q/K/V
semantics for the recognizer-style direct-buffer fused path; diffusion
materialization policy rows remain finite because they encode known-bad
materialization behavior, not just legal SDPA geometry.
The one-off attention no-clone JSON comparison hook has been retired. It had no
supported caller and duplicated the selected SDPA branch before synchronously
reading both results to CPU. Direct-safe parity, zero fallback/copy behavior,
and materialization-required rejection remain covered by transition and op-hit
behavioral tests. The unused vision-owner text log was retired in the same
wave; evidence-harness owner counters and generic route/runtime-policy
diagnostics remain.
`TransformerDecodeGQACloneOnlyRuntimeShape` also admits the decode-GQA
post-softmax clone policy by runtime batch/head/source/head-dim semantics and
score-element budget, so the old transformer decode source-length row remains
policy evidence rather than a production source-length gate.
`MaskedTinySDPAContract` now admits additive fp32 mask SDPA by semantic
rank-3/rank-4 Q/K/V compatibility, PyTorch-style mask broadcast compatibility,
finite scale, and bounded score-tensor budget. The old exact 2x2 additive-mask
tuple remains a regression fixture, while random legal target/source lengths,
batch/head counts, and mask broadcast ranks no longer need exact row admission.
`DiffusionSDPAContract` now also has a semantic runtime cross-attention slice
for mask-free fp32 rank-4 diffusion-style cross attention with batch 1,
matching heads/head dim, head dim 64, small runtime key/value length, and a
bounded score tensor. The old `kv=2` cross rows are fixtures; square diffusion
self-attention now also has runtime semantic admission for `head_dim=64` and
single-head `head_dim=512` when the score budget holds and the `512` materialized
math path can prove a width-pack-compatible key transpose (`sequence % 4 == 0`).
Non-width-pack-compatible `512` square sequences still fail closed on the
direct-buffer materialization guard, so the remaining square blocker is a layout
command-plan issue rather than an exact-row admission issue.
HY-MT
KV-cache append broadening, device-policy packed-weight transient residency,
generic
large-linear execution checkpoints, PaddleOCR OCR projection cross-adapter
fixes, generic transfer cleanup, buffer Float->Byte cast, buffer float flip,
OCR recognizer pointwise sparse-row coverage, and two-input rank-4
channel-cat buffer dispatch coverage remain as previously recorded.
Non-conservative adapters now admit bounded large sliding-window conv packed
weights into the existing identity-checked persistent cache.
Packed-weight residency snapshots now also include per-shape aggregate
hit/miss/store/skip rows for PaddleOCR/HY-MT attribution. CPU-to-Vulkan upload
provenance now threads through generic `aten::copy_` CPU upload and linear
buffer-upload helpers, so repeated Vulkan uploads of the same CPU weight source
can reuse existing persistent linear packed weights through the existing
source/version/provenance identity cache instead of missing by fresh Vulkan
storage identity. The benchmark setup linear prepack path now consults the
live Vulkan device policy before seeding those packed weights: RX 9070 keeps
the setup prepack path, while GTX 1080 and RX 6700 XT skip it through
`avoid_weight_cache=1` instead of trying to persist the full HY-MT linear
weight set. Generic Vulkan `aten::linear` now uses the same packed-linear
runtime cache path, removing the RX 9070 HY-MT inference-mode raw-weight clone
bucket while leaving no-cache adapters on the explicit raw/direct-weight plan
blocker.

## Repo State Summary

The Vulkan backend planning direction is now repo-local in `docs/vulkan`.
Ignored `agent_space` artifacts remain evidence inputs, not production
dependencies.

`scripts/benchmarks/benchmark_model_suite.py` now honors Vulkan
`--device-index` through `torch.vulkan.set_device(index)` in the shared
benchmark device resolver. Cross-adapter PaddleOCR/HY-MT measurements before
this fix recorded requested device metadata but could still run on the current
default Vulkan device; new benchmark records include `current_index` so device
selection mismatches are visible.

HY-MT now has generic compatibility fixes for Vulkan decode.
`SequenceCatDirectBuffer` and `InitialSequenceCatDirectBuffer` now own fp32
rank-4 dim-2 KV-cache append/bootstrap by runtime batch/head/head-dim equality
and positive sequence semantics. The old `KVCacheAppendContract` rows
(`S=1..115` and `S=14..116`) remain evidence/regression fixtures rather than
production source-length gates, so Transformers KV-cache `aten::cat` updates
avoid the old CPU fallback/readback path for legal dynamic shapes. Separately,
linear buffer packed weights are marked
transient when the active adapter policy requests avoiding large persistent
weight caches, and retired packed-weight handles are released after the existing
synchronize/fence wait points instead of being quarantined indefinitely. This
is adapter-policy driven and not HY-MT-specific production routing.
Large inference linear submissions now also have a generic checkpoint in
`run_float_buffer_linear`: after a bounded number of large packed-weight linear
submissions or accumulated bytes, the backend synchronizes the stream and
releases retired packed-weight/linear contexts. The checkpoint is keyed on
linear weight size and inference mode, not on a model name.

Fresh cross-adapter diagnostics under
`agent_space/paddle_hymt_perf_goal_c5dee8d/diagnostic_post_large_linear_checkpoint/`
show the current state. HY-MT one-token decode now completes on RX 9070,
GTX 1080, and RX 6700 XT without DeviceLost or Windows stack overflow. The
remaining HY-MT row is not clean: it still reports `cpu_fallback_count=32`,
`sync_readback_count=8`, 279 tensor CPU readback submits, and 251 host-upload
submits from generation-control and packed-weight/control traffic. PaddleOCR
now completes one-repeat smokes on RX 9070, GTX 1080, and RX 6700 XT with
`cpu_fallback_count=0`. The GTX path uses transient
float-buffer conv packed-weight residency for large packed weights instead of
skipping the cache and repeatedly uploading through the previous
`conv_prepack_upload` device-lost path. The `OCRProjection` contract-family
rowset also admits the proven crop/recognition batch rows under a bounded
`N=1..8` batch policy, not a PaddleOCR model-name route, so the formerly failing
`[6,512,3,80] -> 512` and `[3,512,6,80] -> 192` 1x1 projection rows stay on
Vulkan. Current smoke timings remain single-repeat
diagnostics: PaddleOCR was about 0.81s on RX 9070, 0.65s on RX 6700 XT, and
1.14s on GTX 1080 in the post-checkpoint artifact; HY-MT one-token decode was
about 2.68s on RX 9070, 3.42s on RX 6700 XT, and 8.42s on GTX 1080.
The remaining HY-MT generation-control traffic is now classified in transition
logs instead of appearing only as generic fallback materialization:
`SmallControlTensorFallbackContract` covers bounded Bool/Long/Int control
tensors with `numel <= 16`, plus Float only for tiny comparison-control rows,
and `SmallControlScalarExtractionContract` covers scalar `_local_scalar_dense`
readbacks from bounded control tensors. Both contracts are behavior-neutral:
native execution remains unauthorized, fallback/readback counters still
increment, and the rejected Bool buffer-kernel evidence stays recorded.
Transition rows now also name `SmallControlHostResidencyContract.v0` as
fail-closed: tensor fallbacks are still uploaded back to Vulkan, scalar
extraction remains a Python control boundary, and host residency is not
authorized until a consumer-chain proof exists. Direct host-visible
small-control upload was probed and rejected because tiny Long/Int Vulkan
factory buffers hit `VK_ERROR_MEMORY_MAP_FAILED`; those uploads remain staged
until allocator/map safety is proven.
The current local control-path cleanup additionally separates host-upload
submits from tensor CPU readback submits, adds bounded Bool direct-buffer
`any`/`all` reductions for the proven small-control rows, keeps legal
value-only Float/BFloat16 reductions on Vulkan direct buffers, and lets the
PaddleOCR recognition benchmark compute its logits max on Vulkan before
CPU-side string decoding. Public `max(dim)`/`argmax(dim)` index-producing
routes are fail-closed because the current Long index-output shader path
returns incorrect indices outside the blocked prototype. Focused all-GPU
guardrails under
`agent_space/paddleocr_control_dirty_all_gpus/` show PaddleOCR still completes
with `cpu_fallback_count=0` on RX 9070, GTX 1080, and RX 6700 XT. HY-MT
one-token guardrails under `agent_space/hymt_control_dirty_all_gpus/` still
report `cpu_fallback_count=33` and `sync_readback_count=8` on all three
adapters; the remaining fallbacks are generation-control operations
(`argmax`, `isin`, scalar comparisons, Long/Bool binary/control ops, and
scalar extraction), not packed-weight or attention execution. A standalone
Float/BFloat16 last-dim `argmax` promotion was probed and left fail-closed
because the current Long index-output shader path exposed incorrect index
materialization; `max(dim)` now also uses the CPU fallback path until a real
Long index-output contract exists.
A follow-up HY-MT small-control review records the next safe direction as
`SmallControlHostResidencyContract.v0`, not native Bool/Long compute. The new
focused regression
`test_transition_log_classifies_small_control_host_residency_blocker` proves
that a tiny Long scalar comparison and its Python scalar extraction are
classified through `SmallControlTensorFallbackContract` and
`SmallControlScalarExtractionContract` with host residency explicitly
unauthorized. Native `isin`, Bool binary, general Bool reduction, Long
comparison, Long `masked_fill`, public `argmax`, and public `max(dim)` Long
index routes remain blocked until separate Bool buffer representation,
Long direct-buffer materialization, and host-control consumer-chain proofs are
added.
One narrow Long generation-control cat case is now separated from that blocker:
rank-2 Vulkan Long tensors in `BUFFER`/`TENSOR_WIDTH_PACKED` storage can append
two inputs along the last dimension by row-wise device copies when row count,
width, strides, storage offsets, and dtype/layout checks pass. This covers the
HY-MT `[1,T] + [1,1]` position/id/mask append pattern without treating padded
Long buffers as raw-contiguous storage and without enabling general Long tensor
compute. The focused regression
`test_long_last_dim_cat_two_direct_buffer_inputs_no_fallback` keeps non-last-dim
Long cat on the visible CPU fallback path. The RX 9070 HY-MT one-token smoke in
`agent_space/hymt_long_cat_row_copy_smoke/` drops `cpu_fallback_count` from 33
to 30 and leaves `sync_readback_count=8`; the remaining HY-MT control blockers
are still `argmax`, `isin`, scalar comparisons, Bool/Long binary/control ops,
and scalar extraction.
A second narrow generation-control cleanup now covers tiny Long `fill_(0/1)`
buffer tensors through an explicit host-upload transition instead of the opaque
`aten::fill_.Scalar` CPU fallback. This removes the HY-MT `new_ones` fallback
class without authorizing native Long direct-buffer writes: a probed
`fill_buffer_long` shader returned zeros for value `1`, matching the existing
public Long index-output blocker, so it was removed rather than left as a
latent fast path. The focused regression
`test_small_long_buffer_fill_no_fallback` proves correct `[2,3]` Long
`ones`/`fill_(0)` behavior with zero CPU fallback and preserves fallback for
unsupported value `2`. The RX 9070 HY-MT one-token smoke in
`agent_space/hymt_long_fill_cat_smoke/` drops `cpu_fallback_count` from 30 to
28 and leaves `sync_readback_count=8`.

Two generic PaddleOCR/HY-MT cleanup fixes are now in place. First, legal 2D
Vulkan buffer linear weights can use a metadata-only transposed view for
linear context packing, so inference/labeled linear prepack no longer has to
read the weight back to CPU just to form `weight.t().contiguous()` when the
existing buffer-view guards pass. Unsupported storage/layout cases still fall
back to the old CPU transpose path and remain visible through the same fallback
reason labels. Second, raw buffer host upload/readback fence waits now flush the
normal command pool but skip the descriptor-pool flush because those transfer
paths do not allocate descriptor sets. Shader-packed/image transfer paths keep
the old descriptor-pool flush behavior. A focused PaddleOCR RX 9070 diagnostic
on the current screenshot input improved from about 4.78s to about 4.30s with
the same CPU fallback, sync readback, submit, and retire counters; the sync log
showed 1957 raw-transfer fence waits with `flush_descriptor_pool=0` and 6
descriptor/shader paths that still flushed descriptors. HY-MT one-token RX 9070
linear diagnostics removed the Vulkan-weight CPU transpose fallbacks. A later
large-linear checkpoint probe showed the recurring stack overflow is triggered
by full-model depth/resource accumulation: reduced-layer HY-MT succeeds through
6 layers without extra synchronization, fails around 8 layers, and succeeds at
full depth when explicit synchronization is inserted every 7 layers or fewer.
The production checkpoint is a generic large-linear lifetime guard, not that
layer-count probe.

The local PaddleOCR follow-up adds two more reusable frontend cleanup paths:
Float buffer-to-Byte buffer dtype casts and buffer-backed float `flip` now stay
on Vulkan when the existing buffer/storage guards pass. It also adds the
remaining observed OCR recognizer `3x80` pointwise projection rows as finite
`OCRProjection` sparse rows. A focused RX 9070 normal, non-probe PaddleOCR run
under
`agent_space/paddle_hymt_perf_goal_c5dee8d/paddleocr_rx9070_after_ocr3x80_rows/`
completed with `cpu_fallback_count=0`; the remaining explicit
`sync_readback_count=1` is the known setup-time conv-weight materialization
transition, and transition logs are dominated by required host uploads and
layout repacks rather than route hard-fails. This is not a new timing baseline
because the run was logging-heavy.

The channel-cat cleanup now also has a generic two-input rank-4 dim-1 float
buffer route, `aten::cat.buffer_channel_pair`, backed by
`cat_dim1_4d_buffer_float`. The route is guarded by dtype/rank/dim, equal
batch/spatial sizes, `N=1`, positive channels with total `C <= 4096`,
spatial sizes up to `224x224`, Vulkan buffer storage, and buffer-compute
support; it is not a PaddleOCR-named production route. Focused tests cover
buffer-view parity, odd-channel pairs, route hit logging, and visible copy
accounting. A one-repeat PaddleOCR RX 9070 op-hit sample under
`agent_space/paddle_hymt_perf_goal_current/paddleocr_gpu0_pair_channel_cat_hits/`
observed two `aten::cat.buffer_channel_pair` hits, so this is a small reusable
dispatch cleanup rather than the dominant bottleneck. The remaining PaddleOCR
packed-weight pressure is now visible in
`packed_weight_query_aggregate` rows: the current RX 9070 one-repeat sample
under
`agent_space/paddle_hymt_perf_goal_current/packed_weight_aggregate_rx9070/`
shows two large sliding-window conv shapes rejected by the existing
`store_skip_large` policy. The follow-up `LargeSlidingWindowPackedWeightResidency`
policy keeps the old 2 MB skip on conservative adapters, but on
non-conservative adapters lets float, non-quantized, buffer-direct
`Conv2dSlidingWindow` handles up to 8 MB enter the existing source/version/
bias/device identity cache. It does not add shape-only reuse and it does not
widen the GTX/RX 6700 XT large-cache policy. Focused artifacts under
`agent_space/paddle_hymt_large_sliding_window_cache_rx9070/` show RX 9070 warm
PaddleOCR repeats now hit those two large rows, while GTX 1080 and RX 6700 XT
one-repeat guardrails still skip them and complete successfully. Smaller
conv/depthwise rows still miss by source/storage identity and then store
persistent packed weights. HY-MT one-token on the same artifact stores 225
persistent linear packed weights with zero hits; a two-token sample then
records 225 hits and no extra stores, so repeated decode does reuse the packed
handles after token 1. The first-token row still accumulates about 7.16 GB of
persistent linear packed-weight residency on RX 9070, above the nominal 2 GB
packed-weight cache limit because the current trimmer only evicts transient
entries. The remaining per-token HY-MT fallbacks are generation-control
metadata (`mul`, `add`, comparison, `isin`, `all`, bitwise/logical ops, and
scalar extraction), not core tensor math.
PaddleOCR's broader remaining pressure is still packed-weight/prepack identity,
host uploads, retire/copy traffic, and larger multi-input channel-cat
materialization.
The follow-up channel-cat attribution confirms that the largest remaining
PaddleOCR cat-copy cluster is not a missing shape row: the pair route already
hits for the 224x224 case, and the dynamic `CatAxisDirectBuffer` family still
materializes a concatenated output through per-input `buffer_to_buffer`
dispatches. Do not broaden channel-cat rows as a performance fix unless the
implementation reduces real movement. The next performance-bearing direction
is a private `ChannelCatToConvInputContract`-style consumer handoff/fusion
proof, guarded by single consumer, no public/host/readback escape, stable
layout/allocation identity, and compatible Vulkan conv ownership. A
behavior-neutral readiness surface now records `aten::cat` tensor provenance
for buffer channel-cat outputs and emits
`ChannelCatToConvInputContract.v0` rows when those outputs feed float buffer
conv. These rows keep `behavior_enabled=0` and
`copy_elision_authorized=0`; compatible rows currently reject on
`missing_single_consumer_non_escape_proof` rather than removing the cat
materialization. A one-repeat RX 9070 PaddleOCR smoke under
`agent_space/paddleocr_cat_to_conv_readiness/` completed with
`cpu_fallback_count=0`, `sync_readback_count=1`, and 10 cat-to-conv readiness
rows; all 10 were bridge-shape-ready and all 10 remained unauthorized.
The local packed-weight source-identity follow-up fixes one generic cache miss
class: repeated `weight_cpu.to("vulkan")` uploads followed by Vulkan linear now
record CPU-upload provenance on the Vulkan destination, and transposed
metadata views inherit that root through the existing alias path. The focused
test `test_vulkan_packed_weight_cache_reuses_reuploaded_cpu_source` now reports
five lookups, four hits, one miss, and one store instead of five misses and five
stores; the matching Conv2d test now proves the same reuploaded CPU-source
reuse for sliding-window packed weights. This is a reusable
`PackedWeightSourceIdentity` fix, not a model-name route; PaddleOCR/HY-MT
model timing rows still need fresh guardrails before a model-level performance
claim. A follow-up HY-MT one-token smoke under
`agent_space/paddle_hymt_policy_gated_prepack/` validates the policy-gated
setup prepack behavior across all three adapters: RX 9070 prepacked 225 linear
modules and reused them during the timed generate row, while GTX 1080 and RX
6700 XT recorded `device_policy_avoids_large_persistent_weight_cache` and
completed without the previous large-persistent-prepack device-lost failure.
The remaining HY-MT row is still not clean: generation-control CPU fallbacks
and scalar readbacks remain at roughly the same level, so the next reusable
target is the small-control tensor contract family, not broader prepack.
Direct Vulkan `aten::linear` now routes through the existing packed-linear
`run_addmm_context` implementation, closing the cache-capable inference-mode
hole where `F.linear` cloned raw Vulkan linear weights during the timed row.
Focused tests cover prepacked 2D no-bias and 3D bias rows with zero CPU
fallback, sync readback, or buffer-copy events. On RX 9070, the HY-MT
one-token smoke drops buffer-copy accounting from about 977 copies / 7.24 GB
to about 303 copies / 9.9 MB and reuses all 225 setup-packed linear weights
without retaining raw unpacked weights. GTX 1080 and RX 6700 XT still
intentionally avoid the large persistent cache, so they continue to
transient-pack/copy about 7.17 GB of linear weights and need a no-cache
raw/direct-weight or inference-owned packed-linear plan rather than a
cache-policy broadening.

`conv2d_buffer_float_3x3_s1p1` keeps the existing 8x8x1 default workgroup for
`Kernel3x3Stride1Pad1`. Focused DAv2 multi-GPU evidence rejected both a blanket
16x4x1 default and a bounded large-spatial/high-output-channel heuristic as
default behavior: RX 9070 `vits_280` improved, but RX 9070 `vits_140` regressed
and the GTX 1080 `vits_280` segmented guardrail was worse/noisy. This is
cataloged as performance-plan evidence rather than promoted. The previous 16x8
workgroup candidate remains cataloged as slower, and the separate 768x768
stride-2 pad-1 shader-routing canary is kept as negative ignored-artifact
evidence rather than promoted.

A focused GPU timestamp attribution pass on DAv2 `vitb_140` found the RX 9070
vs RX 6700 XT gap is mostly buffer-conv kernel throughput, not the graph,
transition, submit, retire, copy, or readback layer. On that row, RX 9070 spent
about 170.7 ms in `conv2d_buffer_float_3x3_s1p1` family kernels versus about
112.8 ms on RX 6700 XT, and about 118.2 ms in generic `conv2d_buffer_float`
versus about 62.3 ms on RX 6700 XT. GTX 1080 underperformance is broader but
still dominated by buffer conv plus copy/buffer movement. The next performance
target is a proper `VulkanConvPlanKey`/candidate-plan tuning path for buffer
conv families, not another static workgroup default.

`VulkanConvPlanKey.v0` is now available as reporting-only infrastructure for
float-buffer conv submissions, including the fused `3x3_s1p1_add` conv path.
The snapshot records the selected kernel, contract name/family/tuple when
present, dtype/storage/layout classes, offsets, global/local workgroup,
candidate count, cacheability, tunability, and a compact device/capability
profile (`vendor_id`, `device_id`, `driver_version`, Vulkan API version,
subgroup limits, synchronization/timeline support, and cooperative-matrix
availability). It does not change route selection, shader selection, workgroup
selection, descriptor binding, or fallback/readback behavior. The next conv
performance task is to collect focused multi-GPU plan-key evidence and build a
bounded candidate-plan tuning layer on top of these rows.

`PYTORCH_VULKAN_CONV_PLAN_WORKGROUP_CANARY` is an opt-in candidate-plan
experiment hook for the `Kernel3x3Stride1Pad1` float-buffer conv family. It
currently admits `3x3_s1p1_16x4` and `3x3_s1p1_16x8` as canary workgroups while
leaving the default `8x8x1` plan unchanged. The hook exists so multi-GPU
evidence can be collected by `VulkanConvPlanKey.v0`; it is not a production
promotion and must not be enabled by default without the manifest-backed bounded
promotion evidence. A focused 18-row sweep over DAv2 `vits_140` and `vitb_140`
on RX 9070, GTX 1080, and RX 6700 XT kept both candidates rejected for default
promotion: all rows were correctness-clean and hit the expected plan-key
workgroup, but both candidates had mixed timing wins/regressions across models
and adapters.

`scripts/benchmarks/vulkan_conv_plan_tuning.py` is the first offline tuning
result tool for these rows. It consumes candidate-sweep JSON, emits
`VulkanConvPlanTuningResult.v0`, and validates decisions keyed by
`VulkanConvPlanKey.v0` plus the recorded device/capability profile. It is
behavior-neutral and does not load tuning results into runtime route selection.
The tool can also emit `--granularity plan-key` artifacts that split candidate
evidence by exact normalized plan key and capability profile; those entries are
still row-level timing evidence until per-kernel timing is attached, so they are
not promotion proof by themselves.

GPU timestamp rows now carry a `runtime=conv_plan|...` profile label for
float-buffer conv submissions when timestamp profiling is enabled. The label
records the selected kernel, input shape, output channel count, weight shape,
conv attrs, and global/local workgroup so per-kernel timing can be joined back
to exact `VulkanConvPlanKey.v0` rows. This is reporting-only infrastructure:
route selection, shader selection, workgroup selection, descriptor binding, and
fallback/readback behavior remain unchanged. The offline tuning tool can parse
these logs into `VulkanConvPlanTimestampSummary.v0` grouped by the normalized
conv-plan label fields.

`VulkanRuntimeAttributionReport.v0` is now the general benchmark attribution
tool for separating timestamped GPU shader work from submit, retire, copy,
readback, and fallback counters. Depth Anything V2 benchmark runs can set
`--vulkan-gpu-profile-phase single_image_forward_device_resident` with
`PYTORCH_VULKAN_GPU_TIMESTAMP_LOG=<path>` to synchronize after warmup, truncate
the timestamp log, and run only the selected measurement phase. Timestamp rows
record dispatch-time `recent_op`, `submit_phase`, `stack_phase`, and
`stack_block` fields, including stack-owned external recording dispatches.
The same benchmark now accepts `--vulkan-device-index` and records the selected
Vulkan adapter through `torch.vulkan`, so RX 9070, GTX 1080, and RX 6700 XT
guardrails can be run without wrapper scripts or implicit default-device
assumptions.
The attribution script groups GPU time by kernel class, runtime label, submit
phase, stack phase, and recent op, while joining the same benchmark phase's
CPU fallback, sync readback, buffer-copy, submit-origin, and retire-drain
counters. This is measurement infrastructure only; it does not change execution
routes, shader selection, copies, readbacks, submits, or fallback policy.
The benchmark can also dump CPU timeline summaries at measurement-phase begin
and end when `PYTORCH_VULKAN_CPU_TIMELINE_SUMMARY_LOG` is set. The begin dump
clears setup/warmup attribution and the end dump captures the timed phase; this
is reporting-only and does not affect execution. Stack-owned external recording
dispatches are now included in that opt-in summary with
`external_recording=1`; this closes the previous blind spot where the timed
phase showed decoder/generic CPU submission work but hid the stack segment
recording/binding cost. A focused RX 9070 `vits_140` wide4 run with this
instrumentation measured about 68.4 ms mean / 68.1 ms median / 72.2 ms p95
over five device-resident repeats, with zero timed CPU fallback, sync readback,
or buffer copies. The external-recording rows account for roughly 20 ms per
request across the top stack kernels, so the next sub-50 control-plane target is
descriptor/recording reuse or replay-readiness proof rather than another
retire-drain cleanup.

Descriptor-update allocation overhead is now reduced on the generic Vulkan
descriptor path. `DescriptorSet` reserves its binding vector to the shader
layout size, and `get_bind_handle()` uses an inline-capacity write list for the
common descriptor-update case instead of allocating a fresh heap vector per
dispatch. This changes no route, contract admission, shader, submit, copy,
fallback, or readback behavior. A focused RX 9070 `vits_140` wide4 repeat-30
run measured about 64.2 ms mean / 64.1 ms median / 65.7 ms p95
device-resident forward, versus the preceding 65.9 ms / 66.1 ms / 67.1 ms
retire-deferral baseline, with bridge sanity max_abs
`1.1846423149108887e-06`, zero timed sync readback, and the same submit,
retire, and stack-planned counters.

Stack descriptor dependency diagnostics are now gated off on the default
wide4 bridge lane. The heavy live-descriptor, pre-dispatch proof-table, and
barrier-canary descriptor rows are still emitted when
`PYTORCH_VULKAN_STACK_DEP_GRAPH`, `PYTORCH_VULKAN_STACK_DIAGNOSTIC_ROWS`, or a
selected barrier canary requests them, but successful timing runs no longer
build those rows by default. This changes no route, shader, submit, fallback,
readback, copy, or graph semantics. Focused stage attribution showed that
external-recording descriptor proof rows accounted for about 18 ms/request of
CPU work after descriptor-update allocation flattening; with default row
gating, an RX 9070 `vits_140` wide4 warmup-3/repeat-10 run measured about
57.6 ms mean / 57.8 ms median / 58.9 ms p95 device-resident forward with
bridge sanity max_abs `1.1846423149108887e-06`, zero timed CPU fallback, sync
readback, and buffer copies. A separate graph smoke still materialized the
pre-dispatch proof and submit-epoch rows when `PYTORCH_VULKAN_STACK_DEP_GRAPH`
was set. The earlier repeat-20/30 Windows stack-overflow failure around the
thirteenth repeated forward is now cataloged as an independent stream-sync
pool-lifetime issue rather than a descriptor-diagnostic gating issue.

The current RX 9070 `vits_140` retained-pool wide4 bridge lane remains the best
measured safe lane. The repeated fixed-feature decoder/bridge stack overflow
was traced to normal context command/descriptor pool lifetime at the public
stream-sync boundary: `synchronize_stream()` waited the current stream and
polled retires, but did not recycle the normal context command and descriptor
pools. Repeated device-resident DPT decoder forwards over stable Vulkan
features could therefore grow descriptor-set pressure until the default
1024-set descriptor pool was exhausted around the fourteenth repeat, while
fresh-input runs survived because upload-side fence waits incidentally flushed
the pools. Current-stream synchronization now flushes the normal command and
descriptor pools after a successful current-stream wait, matching the existing
`synchronize_device()` / fence-wait cleanup semantics without broad recovery
flushes. A focused synthetic fixed-feature DPTHead loop now completes 32
repeats, and the original `vits_140` warmup-0/repeat-30 device-resident
benchmark completes with about 62.3 ms mean / 60.9 ms median / 72.1 ms p95,
with zero timed CPU fallback and sync readback. A phase-isolated GPU timestamp
profile still shows about 43-47 ms of kernel work per forward. The largest GPU
row was the FP32 `fc2` linear family (`mm_buffer_float_bias`, about
13.9 ms/forward), followed by decoder/other convs, `fc1_gelu`, qkv/proj
linears, attention BMM, and LayerNorm. The broad tiled-linear canary remains
rejected; the later exact vec2 FC2 row is now promoted only through a narrow
linear plan contract.

Stack-region flatness is now explicit at the runtime-control boundary. Stack
planned recording and external recording begin/end paths emit
`stack_region_recording_depth_guard` rows under existing sync/CPU-timeline logs
when a nested or underflowed recording scope is rejected. Central submit,
pending-retire drain, retire-cleanup, and external-recording cleanup paths emit
`stack_region_control_plane_depth_guard` rows only if they are reentered while
already active. Inference replay record/warmup callbacks also fail closed on
nested callbacks with a `reject_nested_replay_callback` reason. Normal
`vits_140` wide4 bridge smoke stays quiet, so these guards are diagnostic
contracts for future compiled/segmented work rather than a route change or a
performance claim.

`StackRegionDependencyGraph` dumping is now guarded by the same flatness model.
When `PYTORCH_VULKAN_STACK_DEP_GRAPH` is requested, opportunistic graph writes
inside central submit, pending-retire drain, retire-cleanup, or
external-recording cleanup scopes do not recursively serialize the full graph.
Instead they record `StackRegionGraphDumpSkip.v0` rows with
`stack_region_graph_dump_skipped_reentrant_submit_or_cleanup`; the normal graph
dump at stack dependency-scope end still serializes outside those control-plane
scopes. Recursive graph serialization itself is also counted and skipped. This
is behavior-neutral diagnostic protection only: it changes no route, contract,
submit, retire, copy, readback, or shader behavior.

Stack-region exit cleanup now has an explicit behavior-neutral
`StackRegionControlPlaneWorkBatch.v0` scaffold.
`end_stack_planned_recording_and_submit()` still closes/submits the stack
region, snapshots pending-retire transfer sources, retires stack-internal
temporary batches, retires stack-region handoff batches, and finalizes owner
state under the same lock and in the same order as before. The difference is
that those post-submit actions are now prepared as a heap-owned exit work batch
with a typed ordered action list and drained immediately by an iterative
switch. The batch emits prepared and `drained_inline` rows with
`drain_mode=prepared_not_drained` for the prepared row,
`drain_mode=iterative_inline` for the drained row, `drain_action_count=6`,
`drained_action_count=0` before the drain and `drained_action_count=6` after,
`executor_mode=not_started` before executor entry,
`executor_mode=context_control_plane_inline` after the executor drains the
batch, `executor_depth_before=0`, `executor_depth=1`,
`executor_depth_after=0`, `executor_reentry_status=not_reentrant`,
`executor_reentry_rejected=0`, `executor_depth_guard=raii`,
`diagnostic_payload_publish_mode=deferred_after_context_unlock`,
`before_handoff_retained_state_payload_captured`, and
`after_finalize_retained_state_payload_captured`,
`retained_state_live_log_reread_count=0`,
`retained_state_deferred_payload_count`,
`submit_topology_preserved=1`, `phase_boundary_submits_preserved=1`,
`submit_elision_enabled=0`, and `deferred_submit_enabled=0`. This is the first
flattening scaffold for the remaining deep/compiled-session stack-overflow
class. The rows are now also exposed through
`stack_region_control_plane_work_batch_rows` in `StackRegionDependencyGraph.v0`
full and summary-only dumps, so graph artifacts can report the exit batch
without consulting the raw dry-run snapshot. The rows now include
`source_snapshot_state`, stack-internal-temp batch count/bytes, and
stack-region handoff batch count/bytes so the next flattening task can choose a
real cleanup payload by pressure. This does not defer cleanup, remove submits,
change command-buffer topology, or claim a timing win.

A post-guard RX 9070 `vits_140` GPU timestamp pass with
`segmented_stack_wide4_to_exit` showed about 44.2 ms of timestamped GPU work per
forward while the timestamp-instrumented wall time was inflated to about
115.7 ms mean. The top GPU row remained `fc2 | mm_buffer_float_bias` at about
15.2 ms/forward, followed by decoder/other convs, `fc1_gelu`, qkv/proj
linears, attention BMM, and LayerNorm. The original exact FC2 tiled canary
remains rejected as slower. The later exact FC2 vec2 and QKV tiled default
rows are now fail-closed as well: a post-37c8efe DAv2 `vits_140` bridge
regression check showed that keeping either exact tiled row active still failed
bridge sanity (`max_abs` about 1.62 with FC2 tiled active and QKV tiled
disabled; `max_abs` about 0.97 with QKV tiled active and FC2 tiled disabled).
Disabling all exact tiled vision linear rows restored bridge sanity at
`max_abs=1.1846423149108887e-06`. The isolated per-op linear tests remain
numerically clean, but that is no longer accepted as sufficient evidence for
stack-output bridge execution. `VisionFc2ExactTiledVec2LinearPlanContract` and
`VisionQkvExactTiledLinearPlanContract` must stay off the default route until a
replacement generated `LinearPlanContract` or kernel proves full stack-bridge
parity.

Stack-planned submit cleanup now batches pending-retire buffers and images into
one timeline-gated `RetiredResource` callback per stack-planned submission while
preserving the existing per-resource `note_vulkan_retired_resource` accounting.
`sync_counters()` appends `retire_cleanup_callback_count` so host cleanup
flattening is visible separately from retired resource count. A focused
warmup-3/repeat-30 RX 9070 `vits_140` bridge run measured about 66.6 ms mean /
66.3 ms median / 68.7 ms p95 device-resident forward, stayed
correctness-clean (`max_abs=1.1846423149108887e-06`) with zero timed CPU
fallback, sync readback, or buffer copies, and reported
`retired_resource_count=15415` versus `retire_cleanup_callback_count=4668`.
This is a generic cleanup-control-plane fix, not a submit-reduction claim: the
timed forward still reports four stack-planned submits, three retire-drain
submits, one pre-stack flush, and one explicit synchronize per request.

Normal submit cleanup now uses the same timeline-gated callback batching model
as stack-planned submit cleanup. `retire_deferred_cleanup()` still records
per-resource retire accounting and keeps the same submit/timeline ownership, but
ordinary submits now move their pending buffers/images into one callback instead
of one `RetiredResource` callback per resource. A focused warmup-0/repeat-30 RX
9070 `vits_140` device-resident run measured about 58.4 ms mean / 57.6 ms
median / 61.4 ms p95, with timed CPU fallback zero, sync readback zero, and the
same submit-origin and retire-drain counts as the pre-batch run. The cleanup
callback counter dropped from 38,745 to 2,491 over the 30-repeat measurement.
A separate no-skip-output three-repeat sanity completed with `performance_valid`
true and zero timed CPU fallback/sync readback, but the no-bridge benchmark path
does not emit a model-vs-reference `max_abs` field.

`PYTORCH_VULKAN_STACK_REGION_EXTERNAL_RECORDING_POOL_LEASE=per_stack` was an
opt-in stack-owned external recording pool-lease experiment. Focused
`vits_140` evidence showed it was slower than the retained persistent-pool path
and did not fix the `compiled_session_bridge` Windows stack-overflow exit
`-1073741571`, so the live mechanism is retired and the result remains
historical evidence only.

A focused 18-row timestamp sweep over DAv2 `vits_140` and `vitb_140` on RX
9070, GTX 1080, and RX 6700 XT kept both `3x3_s1p1_16x4` and
`3x3_s1p1_16x8` rejected as broad default promotions. Both candidates were
correct and hit the expected local workgroup, but each improved three
model/device rows and regressed three rows. `16x8` improved `vitb_140` whole-row
timing on all three devices while regressing `vits_140`, and the vits/vitb
target conv labels are disjoint, but exact vitb label deltas are still mixed by
device for several labels. The next promotion path is therefore a
device/driver/capability-keyed tuning cache or a narrower exact-plan policy,
not a static workgroup default.

An attempted seven-row exact 16x4 policy for labels that improved on all three
timestamp-sweep devices was also rejected and backed out before commit. It
routed only the accepted exact labels to 16x4 and left adjacent labels at 8x8,
but the post-policy default check regressed whole-row timing on GTX 1080 and RX
6700 XT `vitb_140`, while `vits_140` timing moved despite no promoted labels.
Keep this as rejected evidence; the next runtime promotion mechanism should be a
tuning cache that can validate the full row before selecting per-plan locals.

An attempted forced float-buffer tiled linear canary for label-inferred
vision-backbone linears was rejected and backed out before commit. The first
probe did not hit tiled kernels; after fixing the activation predicate, the
probe did hit `aten::linear.buffer_float_tiled_bias[_gelu]` rows but failed
`vits_140` bridge sanity at `max_abs=0.21523737907409668` and measured about
71.1 ms mean device-resident forward. Do not reintroduce a broad force-tiled
linear gate. A future linear plan needs a parity-proven kernel or narrower
contract before timing.

A narrower exact fc2 tiled-linear canary is now recorded as slower-but-correct
evidence rather than a promotion. With
`PYTORCH_VULKAN_LINEAR_TILED_CANARY=vision_fc2_exact_151x1536x384`, the
`vits_140` bridge route hits only `aten::linear.buffer_float_tiled_bias` for the
`[151,1536] -> [151,384]` bias/no-post-op rows, bridge sanity remains clean at
`max_abs=1.6391277313232422e-06`, and CPU fallback/sync readback remain zero.
However, the clean five-repeat RX 9070 run regressed to about 93.5 ms
device-resident mean versus the 64.3 ms wide4 baseline. Keep this as rejected
linear plan evidence; sub-50 work should not promote the current tiled fc2
kernel.

The follow-up exact FC2 vec2 tiled-linear row is historical canary evidence, not
an accepted default route. The exact `[151,1536] x [384,1536] + [384]` FC2 row
previously selected `aten::linear.buffer_float_tiled_bias_vec2` and produced a
sub-50 ms `vits_140` canary, but later full DAv2 stack-output bridge regression
checks showed that keeping FC2 tiled active still failed bridge sanity
(`max_abs` about 1.62). The exact QKV tiled row similarly failed full bridge
sanity (`max_abs` about 0.97). Both `VisionFc2ExactTiledVec2LinearPlanContract`
and `VisionQkvExactTiledLinearPlanContract` are therefore fail-closed by default:
focused fail-closed tests assert the old tiled ops are not selected, adjacent
negative tests keep nearby rows on the conservative path, and isolated per-op
linear parity is no longer sufficient evidence for promotion. Future sub-50
work needs a replacement FP32 linear plan or generated `LinearPlanContract` that
passes full stack-output bridge parity before timing claims.

The native `vulkan_prepack::run_vision_stack_captures_decoder_preprocess_bridge`
path enforces the same max-12-block proven-depth guard as the benchmark control
plane by default. Direct native callers with deeper stack-output bridge contexts
fail closed with `stack_output_bridge_depth_exceeds_proven_rowset` and point to
`StackOutputBridgeDeepSplitPlanRuntime.v0` instead of rediscovering the `vitl`
stack-overflow path. An opt-in
`PYTORCH_VULKAN_STACK_OUTPUT_BRIDGE_DEEP_SPLIT=native_private_baton` runtime now
splits deeper stacks into <=12-block native chunks, keeps the inter-chunk baton
device-private, and feeds the captured tensors directly to the decoder-preprocess
bridge. The former benchmark-mediated `python_private_baton` mode is retired and
falls through the generic unsupported-mode depth guard.

`docs/vulkan/PERFORMANCE_EVIDENCE.md` and
`test/vulkan_contract_proofs/performance_plan_evidence_manifest.json` now hold
the checked-in performance-plan evidence ledger. This is separate from accepted
shape rows: it records accepted fixes, opt-in canaries, slower-but-correct
plan candidates, correctness-blocked paths, and unsafe topologies so later
agents do not repeat the same diagnostics as if they were new work.
`test/vulkan_contract_proofs/stack_performance_canary_decision_table.json`
summarizes the active DAv2-driven performance canary decisions in a compact
table; consult it before adding another segmented-recording, compiled-session,
retire-handoff, conv-plan, or linear-plan canary.
The latest segment-mode evidence keeps `segmented_stack_wide4_to_exit` as the
best current `vits_140` bridge canary. The valid but slower wide3 and
prefix3-6 tail modes have been retired from live production, benchmark, and
test orchestration. The rejected
`segmented_stack_wide6_to_exit` route has been retired from live production,
benchmark, and test orchestration. Its historical result remains evidence: a
focused five-repeat RX 9070 run selected two full segments covering blocks 0-5
and 6-11 at 62/72 planned dispatches each, kept bridge sanity clean at max_abs
`1.6391277313232422e-06`, and kept CPU fallback, sync readback, and buffer
copies at zero. Reducing stack-planned submits from 20 to 15 over five repeats
did not improve latency: wide6 measured about 79.9 ms mean device-resident
forward while the matching wide4 run measured about 64.3 ms, with both modes
still reporting 15 stack-owner retire-drain submits. Repeated context-owned
stack-output bridge timing remains unsafe
because it can hit Windows stack overflow after a one-repeat sanity pass.
The DAv2 benchmark now fails this unsafe repeated context-owned bridge topology
before native execution and writes a JSON failure artifact pointing to the
performance evidence manifest. One-repeat bridge sanity checks and bounded
segmented stack-owned modes remain allowed.
Three stack-exit pending-retire handoff probes are also cataloged as rejected
`vits_140` evidence. `private_bridge_capture_handoff` kept bridge sanity clean
and preserved zero CPU fallback, sync readback, and copy counters, but measured
about 69.0 ms mean and still reported 15 stack-owner retire-drain submits. It
is also the wrong release boundary for promotion: private bridge captures must
release after decoder bridge consumer completion, not at backbone stack exit.
`residual2_norm1_carry_handoff` likewise preserved correctness and zero
fallback/readback/copy counters, but measured about 73.2 ms mean and did not
reduce the timed retire-drain submits. The segment-completion cleanup handoff
canary moved exact external-recording cleanup pending-retire entries into the
stack-exit handoff batch under
`StackRegionSegmentCompletionRetireHandoffContract.v0`, but it also kept the
same 15 stack-owner retire-drain submits and regressed to about 75.3 ms mean
device-resident forward. Do not repeat stack-exit pending-retire handoff as a
standalone latency path unless a later submit-plan change makes the transfer
reduce actual queue submits. The runtime env gate for the segment-completion
handoff is retired; the cleanup rows remain metadata-only. The next ownership
task is segment-local
completion ownership or a bridge-scoped private capture release owner, not
another stack-exit handoff.
A focused bridge-release probe at the post-decoder-consumer boundary was also
valid but did not transfer pending retires: the release-owner rows remained
`transfers_pending_retires=0`, and the five-repeat run measured about 70.3 ms
mean. That result is cataloged as a no-op/rejected path. A future behavior
canary needs a distinct bridge-scoped handoff batch with explicit restore and
close-submit retire ownership before it moves entries.
The narrow stack-owned segment retire-drain deferral is now the default under
the same active-stack-planned-recording, current-thread, resource-count, and
byte-budget predicates that previously gated the canary. The old rejected
evidence remains valid for the pre-cleanup-batching path: before stack-planned
cleanup callbacks were flattened, deferring the submit shifted cost to later
lifetime work and regressed timing. After cleanup batching, a focused
warmup-3/repeat-30 RX 9070 `vits_140` bridge run stayed correctness-clean,
kept timed CPU fallback, sync readback, and buffer copies at zero, and removed
the three timed `retire_queue_drain` queue submits per request. The run measured
about 65.9 ms mean / 66.1 ms median / 67.1 ms p95 device-resident forward.
The fast path records the deferred pending resources through the existing
retire-drain and stack-retire-blocker counters instead of hiding them; the same
run reported 1800 deferred resources / 42.7 MB over 30 timed forwards. Setting
`PYTORCH_VULKAN_STACK_REGION_RETIRE_DRAIN_DEFER=disabled` restores the old
retire-drain submit behavior for diagnostics. Focused warmup-1/repeat-3
`vitb_140` and `vitl_140` bridge smokes also stayed correctness-clean with zero
timed retire-drain submits, CPU fallback, sync readback, or buffer copies.
`StackProgramOwnedTempStabilityContract.v0` is now exposed as reporting-only
stack replay infrastructure. It records that current stack-internal temp
descriptors are stable for per-forward re-recording, but remain fail-closed for
command replay because allocator-backed program-temp slot identity is not yet
proven across forwards. Benchmark debug snapshots include
`stack_program_owned_temp_stability` rows next to
`stack_replay_binding_mode`, so future replay work can consume this proof row
instead of rediscovering the same `program_owned_temps_not_stable_for_command_replay`
blocker.
`StackProgramOwnedTempLiveIdentityJoin.v0` adds the next fail-closed replay
diagnostic: it joins planned program-owned temp descriptors to live descriptor
rows only at the broad `(phase, block, binding)` level and reports whether that
join is stable enough for replay. Current `vits_140` evidence shows the broad
join exists, but allocation identities are unstable and overbroad, so command
replay still requires a stable low-level program-temp slot id or replay-owned
temp allocator before it can be authorized.
`StackProgramOwnedTempSlotIdentity.v0` now reports the planner-owned temp slot
namespace directly from the stack descriptor table. Current rows prove stable
plan slot ids, descriptor indices, and shapes for program-owned temps, but keep
command replay fail-closed because those plan slots are not yet backed by stable
allocator identity or exact live descriptor-slot joins across forwards.
The same snapshot now also reports replay-program-owned tensor slots when an
existing `VisionBackboneProgram`/replay-bundle path is observed. Those rows have
stable replay-program slot ids and allocation identity, but remain
non-authorizing because they are not yet joined to the stack planned-recording
descriptor/live binding rows.
The Depth Anything V2 benchmark previously carried a benchmark-only
`compiled_session_bridge` canary over
`run_depth_anything_v2_compiled_session_bridge`. Its `vits_140` evidence ended
in Windows stack overflow `-1073741571`; the selectable mode and its unreachable
caller branch are now deleted. The public compiled-session backend remains
Migration-gated until graph-owned recorded partitions replace it.
The empty bridge-private capture pending-retire handoff batch scaffold is now
retired. It had clear, restore, retire, and retained-state reporting code, but
no producer ever moved a resource into it. The separate
`PrivateBridgeCaptureHandoffContract.v0` and release-owner proof rows remain as
fail-closed migration evidence; a future graph-owned release implementation
must provide real producer and completion ownership rather than restoring the
inactive batch.
Depth Anything V2 benchmark artifacts now also include a compact
`vulkan_stack_region_segment_plan` summary when `StackRegionSegmentPlan.v0`
rows are present in the debug snapshot. Segmented stack-owned recording modes
now enable the recording-domain observation rows automatically, so opt-in
canary artifacts carry the segment-plan evidence without requiring a separate
graph-dump path. This summary records the observed segment-plan modes,
statuses, fail reasons, dispatch budgets, and sampled segment rows for the
measured shape/topology. It is evidence lookup metadata only and must not be
used as a production route table.
`tools/vulkan_contract_codegen/query_performance_evidence.py` searches the
checked-in performance evidence manifest and can summarize this per-run
segment-plan field from benchmark artifacts before a diagnostic sweep is
rerun.
A focused `vits_182` wide4 graph-catalog run now records that bridge sanity
and `StackRegionSegmentPlan.v0` evidence pass for the next DAv2 input size.
A separate no-graph three-repeat timing run measured about 74.8 ms mean /
76.1 ms median / 78.3 ms p95 for device-resident forward with bridge sanity
passing at max_abs `2.507120370864868e-06`. This is opt-in canary evidence,
not a production default.
The same wide4 canary is also cataloged for `vits_280` / `280x420`: segment
plan evidence passes with the same 20 accepted / 36 rejected row split, and a
separate no-graph three-repeat timing run measured about 91.9 ms mean /
91.7 ms median / 107.5 ms p95 device-resident forward with bridge sanity
passing at max_abs `4.291534423828125e-06`.
For `vits_420` / `420x630`, the same wide4 canary also passes bridge sanity
and segment-plan evidence, with a separate no-graph three-repeat timing run at
about 127.7 ms mean / 127.9 ms median / 130.9 ms p95 device-resident forward
and max_abs `8.702278137207031e-06`.
For `vits_560` / `560x840`, the wide4 canary remains valid with the same
segment-plan row pattern; the no-graph three-repeat timing run measured about
316.7 ms mean / 326.4 ms median / 347.8 ms p95 device-resident forward with
bridge sanity max_abs `1.0013580322265625e-05`.
For `vits_700` / `700x1050`, the same canary remains valid with matching
segment-plan rows; the no-graph three-repeat timing run measured about
446.8 ms mean / 397.9 ms median / 550.2 ms p95 device-resident forward with
bridge sanity max_abs `1.52587890625e-05`.
The six `vits` rows are now grouped in the performance evidence manifest as a
finite wide4 canary rowset. This is review memory for the current DAv2 vits
performance lane, not a ShapeEnvelope expansion, a production dispatch table,
or evidence for `vitb`, `vitl`, Lotus, PaddleOCR, HY-MT, or Gemma.
`test/vulkan_contract_proofs/stack_region_segment_plan_manifest.json` records
that finite rowset as a `StackRegionSegmentPlan.v0` governance boundary and
links it back to the per-row performance evidence ids. Future `vitb` or `vitl`
segment-plan work must add separate rowsets rather than widening the `vits`
rowset by inference.
A focused `vitb_140` run now has its own one-row wide4 canary rowset: bridge
sanity passed with max_abs `3.5762786865234375e-06`, CPU fallback/readback were
zero, the graph catalog showed the same 20 accepted / 36 rejected
`StackRegionSegmentPlan.v0` rows, and a separate no-graph three-repeat timing
run measured about 110.2 ms mean / 109.2 ms median / 112.4 ms p95
device-resident forward. `vitl_140` no longer needs to rediscover the old
process-level stack overflow to make progress: the default path still fails
closed for `block_count > 12`, but the opt-in native private-baton deep-split
runtime can run the 24-block stack as two 12-block native chunks. A focused
`vitl_140` smoke wrote a valid artifact with bridge sanity passing at max_abs
`0.0001220703125`, `cpu_fallback=0`, and `sync_readback=0`; a separate
10-repeat run also completed without Windows stack overflow and kept the same
correctness/counter state, measuring about 237.5 ms mean / 221.2 ms median /
305.8 ms p95 for device-resident forward. The DAv2 benchmark safe path now
auto-selects the native private-baton deep split when the Vulkan stack-output
device bridge is requested, no explicit deep-split env is set, and the stack is
deeper than the max-12-block proven single-chunk rowset. A no-env RX 9070
`vitl_140` warmup-1/repeat-3 run measured about 171.4 ms mean / 173.4 ms median
/ 176.1 ms p95 with bridge sanity max_abs `0.00013685226440429688`, and a
no-env `vitl_182` warmup-1/repeat-2 smoke measured about 199.9 ms mean with
max_abs `0.0001678466796875`; both runs selected
`auto_native_private_baton_for_deep_stack`, kept `cpu_fallback=0`, and kept
`sync_readback=0`. A post-commit `vitl_140` guardrail on all three local
Vulkan adapters also passed with the auto policy: RX 9070 measured about
174.7 ms mean / 180.1 ms p95, GTX 1080 measured about 670.4 ms mean /
674.5 ms p95, and RX 6700 XT measured about 225.2 ms mean / 226.2 ms p95;
all three kept timed fallback/readback/copies at zero and reported
`runtime_auto_selected=true`. Direct native callers still fail closed unless they request
`PYTORCH_VULKAN_STACK_OUTPUT_BRIDGE_DEEP_SPLIT=native_private_baton`, and an
explicit benchmark env setting such as `none` is preserved rather than
overridden. The benchmark-local
`python_private_baton` canary is retired because it overflowed inside
`run_vision_backbone_stack_private_capture_debug`; its rejection remains
historical evidence rather than live benchmark orchestration. Do not infer broad
`vitl` support from `vits` or `vitb`, and do not retry the Python-mediated baton
path as the next runtime proof.
A post-`a3433b74fbd5` recheck with
`PYTORCH_VULKAN_STACK_OUTPUT_BRIDGE_DEEP_SPLIT=native_private_baton`,
`segmented_stack_wide4_to_exit`, and `compiled_session_bridge` passed
`vitl_140` repeats 1 and 2 with no Windows stack overflow, `cpu_fallback=0`,
`sync_readback=0`, and the deferred stack-exit diagnostic publication row
(`diagnostic_payload_publish_mode=deferred_after_context_unlock`,
`retained_state_live_log_reread_count=0`). This confirms the control-plane
publication flattening did not regress the accepted native deep-split canary.

## DAv2 Stack Region Policy Lock

The unsafe sub-50 ms DAv2 `vits_140` path is rejected. Retire-time or
lifetime-only removal of native-layernorm phase-boundary submits corrupted
capture-sensitive stack outputs and must not be pursued as the next
optimization. The old coalescing proof showed that consumer existence and
retire lifetime proof are not sufficient: the missing contract is dispatch
ordering at the producer-consumer edge before command recording.

The historical DAv2 stack-bridge baseline was approximately 82-103 ms for
`vits_140` depending on whether the private decoder bridge planned-recording
canary was enabled. That benchmark-only canary has been retired under the
graph-first cleanup policy; it is not a supported-default baseline. The opt-in
generic stack-captures-to-decoder bridge is
correct, copy-free, CPU-fallback-free, and sync-readback-free. A previous
`recover_after_vulkan_failure_if_needed()` bug forced `Context::flush()` at
stack entry and every block entry even when no Vulkan failure recovery was
pending; the helper now flushes only when
`vulkan_post_failure_recovery_required()` is set. Thrown Vulkan API errors
mark that recovery flag; non-throwing guard diagnostics such as replay-risk
reports do not. The bridge must remain guarded; public `Tensor[]` capture
behavior remains the safe default when a same-region consumer is not
registered. Bridge sanity validation now inserts a setup-phase Vulkan
synchronize between the optimized bridge output and the original reference
forward; deep `vitl` rows can otherwise reuse the same stack-owner context
before the bridge path has fully closed, causing Windows stack overflow during
the reference pass. This boundary is outside timed iterations and is reported
as `reference_boundary_synchronized` in the sanity metadata. A focused
30-repeat `vits_140` run with the recovery guard and the historical
decoder bridge planned-recording canary measured about 77.7 ms mean / 77.4 ms
median / 82.9 ms p95 for device-resident forward, with max_abs
`8.344650268554688e-07`, `cpu_fallback=0`, and `sync_readback=0`.
The public `vulkan_prepack::synchronize()` path now follows the same recovery
split: successful execution submits and waits the current stream, while full
`Context::flush()` remains reserved for a pending post-failure recovery flag.
This preserves synchronization semantics without clearing broad context cleanup
state at every benchmark timing boundary. The successful stream-sync path now
also resets idle persistent external-recording command/descriptor pools after
the current stream has completed, so segmented stack-owned recording does not
accumulate stale recording pool state across repeated benchmark iterations.
Historical wide3 evidence after the LayerNorm statistic-buffer retire proof
records a matching 30-repeat
`vits_140` run with `StackScopeRetireHandoffContract.v0`, decoder bridge
planned recording, and `segmented_stack_wide3_to_exit` at
about 75.5 ms mean / 74.7 ms median / 86.0 ms p95. The bridge sanity check
passed with max_abs `1.6391277313232422e-06`, CPU fallback stayed zero, and
sync readback stayed zero. The comparable context-owned stack path measured
about 92.0 ms mean / 90.2 ms median / 104.2 ms p95. The result remains useful
evidence, but the exact slower-than-wide4 route is no longer live.

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
The canary reports recorded work with a separate stack-owned dispatch counter
rather than `submit_count_`, keeping the normal context phase-boundary submit
logic untouched.
`PYTORCH_VULKAN_STACK_REGION_OWNED_COMMAND_BUFFER=segmented_stack_entry_to_exit`
is a narrower follow-up canary for private-bridge stacks that can be split at
capture boundaries into segments of four blocks or fewer. It opens one
stack-owned recording scope per segment, allows at most two scopes, and submits
each scope at its local stack-exit boundary. Stacks with an over-budget segment
or too many segments still fail closed to the context-owned path. Exploratory
full `vits_140` runs exposed stack-overflow risk for both unsegmented recording
and four capture-aligned scopes, so DAv2 remains on the context-owned path.
`PYTORCH_VULKAN_STACK_REGION_OWNED_COMMAND_BUFFER=segmented_stack_prefix_to_exit`
is an opt-in mixed-topology canary for that gap. It records only the first two
eligible private-bridge segments externally and leaves the remaining tail on the
existing context-owned path. This is a partial recording experiment, not a
full-stack DAv2 performance path. Selected external segments also fail closed
when their planned dispatch count exceeds the current small-scope canary budget.
The previous dispatch-derived prefix experiment completed one forward but hit
stack overflow under repeated `vits_140` inference, so it was removed. After
the persistent external recording pool reset owner landed, the first bounded
replacement is available only as
`PYTORCH_VULKAN_STACK_REGION_OWNED_COMMAND_BUFFER=segmented_stack_dispatch_budget_prefix_to_exit`.
It selects the first two dispatch-budget candidate segments, leaves the
remaining tail on the context-owned path, and keeps submit elision disabled.
It is not a full-stack DAv2 performance path; the six-segment candidate
sequence remains blocked on repeat-stable multi-scope ownership proof.
Repeated model-level graph dumps can now opt into
`PYTORCH_VULKAN_STACK_DEP_GRAPH_MODE=summary_only`. This keeps the root
summary plus segment-plan, recording-domain, cleanup-boundary, cleanup-retire,
and related ownership rows, while omitting the heavy dispatch/resource/live
binding arrays. The mode is diagnostic-only: it does not change recording,
submit, cleanup, retire, or segment-admission behavior. It exists because
repeated `vits_140` bridge graph dumps can otherwise grow into hundreds of MB
before producing timing results.
Detailed stack retire/lifetime attribution rows are now opt-in through either
`PYTORCH_VULKAN_STACK_DEP_GRAPH=<path>` or
`PYTORCH_VULKAN_STACK_DIAGNOSTIC_ROWS=1`. Normal timed runs keep aggregate
numeric counters but do not build the heavy string-keyed retire-blocker,
region-lifetime-attribution, or subresource dry-run row maps. This preserves
the graph/proof surface for explicit diagnostic runs while keeping ordinary
performance measurements from paying for diagnostic row construction.
`StackOwnerFrequencySubmitPlan.v0` is now the graph-gated attribution row for
`normal_cmd_submit_frequency` submits that occur while the submit phase is
`stack_owner`. It records the region id/state, command-buffer recording id,
submit epoch, pending dispatch count, planned-recording ownership state,
external region-owned command-buffer state, recent Vulkan op label, allocation
label, and fail-closed blocker. The row is behavior-neutral: it preserves the
frequency submit and does not authorize submit elision. Its purpose is to decide
whether the next canary should target frequency-submit suppression under planned
stack ownership, a bridge/decoder segment, or a coverage gap where dispatches
escaped planned recording.
The private decoder-bridge planned-recording canary has been retired. It had no
supported caller and only batched the post-stack LayerNorm and decoder-preprocess
island inside the benchmark-only stack lane. Its historical 77.7 ms DAv2 result
and exact measured configuration remain in the performance manifests, but the
environment selector, live recording scope, and mechanism-only proof field do
not remain in production code. Graph-owned command partitions are the migration
target for this fixed submission cost.
`StackRegionSegmentPlan.v0` is the behavior-neutral graph surface for that
planner. It emits a summary row for every segmented canary request and
per-segment rows when candidate segments are computed. The rows record generic
inputs and budgets only: private-bridge policy, runtime capture indices,
plan-capture indices, block count, segment ends, total and per-segment planned
dispatch counts from the stack shape plan, four-block segment budget, two-scope
budget, and fail-closed reason such as
`segment_scope_limit_exceeded`. They do not change command-buffer topology,
open recording scopes by themselves, move cleanup resources, defer submits, or
authorize submit elision.
For private bridge runs, the graph now also emits
`dispatch_budget_candidate_*` segment-plan rows that derive hypothetical
block-boundary segments from the same planned dispatch budget, but these rows
are candidate-only. They set
`owned_command_buffer_mode=dispatch_budget_candidate_only`,
`segment_selected_for_recording=0`, and
`planned_dispatch_count_admission_predicate=0`; they do not expose the rejected
dispatch-derived prefix canary again. Candidate plans that need more than the
current two-scope canary budget report
`dispatch_budget_candidate_scope_limit_exceeded`, plus
`candidate_sequence_requires_multi_scope_owner=1` and
`candidate_sequence_blocker=multi_scope_repeat_stability_unproven`. This is
now the real `vits_140` evidence row for six two-block candidate segments: the
segments are individually under dispatch budget, but the sequence is blocked
on repeat-stable multi-scope ownership proof rather than a larger hard-coded
scope limit.
`PYTORCH_VULKAN_STACK_REGION_OWNED_COMMAND_BUFFER=segmented_stack_dispatch_budget_single_segment_to_exit`
is a narrower canary for the same dispatch-budget segment planner. It selects
only the first dispatch-budget candidate segment, records the remaining segment
boundaries through context-owned planned scopes, preserves all submits, and
does not change the two-scope prefix budget. Its purpose is to isolate the
cost and cleanup behavior of one region-owned external segment before any
multi-scope expansion.
`PYTORCH_VULKAN_STACK_REGION_OWNED_COMMAND_BUFFER=segmented_stack_dispatch_budget_prefix3_to_exit`
is the next exact canary and raises only the dispatch-budget prefix selection
from two to three selected external segments. It is not a numeric scope-limit
override: default prefix mode remains capped at two segments, all selected
segments still enforce the four-block and 24-dispatch budgets, tail work stays
on the existing context-owned planned-recording path, and submit elision stays
disabled.
The former prefix3-6 tail modes coalesced unselected candidate segments into a
single context-owned scope through stack exit, while the former wide3 mode
recorded four three-block external segments under a 36-dispatch budget. All
five exact modes remained correctness-clean in the recorded RX 9070 sweep but
were slower than retained wide4. They are now retired; the checked-in evidence
manifest preserves their timing and rejection result, while the generic
dispatch-budget planner and live `segmented_stack_dispatch_budget_prefix3_to_exit`
route retain the reusable planning behavior.
`PYTORCH_VULKAN_STACK_REGION_OWNED_COMMAND_BUFFER=segmented_stack_wide4_to_exit`
is the next fixed wider-segment probe. It records three four-block external
segments for 12-block private-bridge stacks when each segment stays under a
48-dispatch budget. It exists to test whether one fewer stack-owned segment
close submit is safe after `wide3`; it is still opt-in, exposes no numeric
scope override, and keeps submit elision, deferred submit, and pending-retire
transfer disabled. A 10-repeat `vits_140` canary run stayed valid with bridge
sanity passing, measured about 71.0 ms mean / 70.7 ms median / 75.1 ms p95,
and reduced the exact device-resident loop to about nine queue submits: four
stack-planned submits, three retire-queue-drain submits, one pre-stack flush,
and one explicit backend sync.
The benchmark harness now emits `vulkan_measurement_phase_counters` alongside
the existing aggregate `vulkan_phase_counters`. The new rows are deltas for
each single-image measurement loop and keep the legacy aggregate phase intact.
For the historical `vits_140` wide3 device-resident forward, those rows show the
remaining per-forward structure is about 10 queue submits: five stack-planned
submits, three retire-queue-drain submits, one pre-stack flush, and one
explicit backend sync. The retained wide4 lane supersedes that exact canary;
the submit breakdown remains evidence for the next structural budget.
External recording cleanup logical-boundary rows are also stamped with segment
identity when a segmented stack-owned scope is active, so cleanup resource
counts and bytes can be joined to a segment without relying on row order. The
stamp is metadata-only and does not make cleanup resource count a segment
admission predicate. `StackRegionExternalRecordingCleanupRetire.v0` now records
the matching stack-exit cleanup-retire scheduling event for stack-owned
external recordings: buffer/image counts, retained cleanup bytes, timeline
validity, whether the batch was scheduled on the stack-exit submission or
cleared because no valid timeline was available, and persistent external
recording pool-pressure counters. The new counters report cumulative and
per-scope persistent command-buffer acquisitions plus descriptor-set
allocations observed while the external command buffer was active. The
cleanup-retire row itself remains metadata-only: it does not transfer pending
retires, remove submits, defer submits, or make cleanup resource count an
admission predicate. Its purpose is to make repeated stack-owned recording
cleanup and pool pressure visible after the rejected dispatch-derived prefix
experiment exposed repeat-instability risk. Persistent external recording
pools now reset at global completion/fence-wait flush points when no external
recording is active; cleanup-retire rows report
`external_pool_reset_required=1`,
`external_pool_reset_owner_available=1`,
`external_pool_reset_point=global_completion_flush`, and
`persistent_pool_reset_proven=1`. The reset is not performed at cleanup-retire
emission itself, and the rows still do not make cleanup resource count or pool
pressure an admission predicate for larger multi-scope canaries.

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
must be owned by a planned region-wide recording path and consume real barrier
insertion plus a current-run proof match. The rejected current-topology
submit-deletion canary has been retired; eligible rows are migration evidence,
not permission to skip a submit.

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
bridge output sanity. The behavior branch was backed out, and the inactive
environment route, guard rows, live-submit binding, and exact-join diagnostics
were subsequently retired. The retained rejection is that the current
`PhaseSubmitEpochVisibilityContract` predicates do not prove value preservation
when the phase submit itself is removed. Generic boundary and submit-level
proof surfaces remain as migration evidence for a region-owned recording path.
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
current-topology submit-deletion path has been retired, and the canary records
`authorizes_submit_elision=0` because it is a
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
visible without transferring pending retires or enabling submit elision. The
retired `stack_internal_until_stack_exit` diagnostic override allowed the
smaller stack-exit source to supersede the preserved phase-submit source. Its
selected synthetic `residual2@0 -> norm1@1` result remained partial and
fail-closed on `pending_retire_transfer_source_incomplete`, so the exact
selector and its work-batch field were deleted. The generic source snapshots,
coverage diagnostics, and preserved-phase handoff remain.
`PYTORCH_VULKAN_STACK_SCOPE_RETIRE_HANDOFF=1` or
`PYTORCH_VULKAN_STACK_SCOPE_RETIRE_HANDOFF=stack_scope_retire_handoff` is the
contract-facing `StackScopeRetireHandoffContract.v0` spelling for the proven
QKV stack-temp class. It also admits proven stack-scope activation rows for
`stack_norm1_output`, `stack_proj_output`, `stack_residual1_output`,
`stack_norm2_output`, and `stack_fc2_output` when their TensorAllocation
provenance is a direct Vulkan buffer with positive shape, has
last-use/non-escape/internal-temp proof, no requested/final/alias/runtime
escape, the producer phase matches the role, and the expected same-block
consumer phase matches the local stack contract (`qkv_linear`, `residual1`,
`norm2`, `fc1_gelu`, or `residual2`, respectively). `stack_residual2_output`,
requested/final outputs, aliases, raw stack-internal cleanup, metadata/uniform,
and unscoped LayerNorm buffer-width cleanup remain excluded. The contract does
not add shape admission, change shaders, move pending-retire entries into the
region handoff batch, defer submits, or authorize submit elision; it only lets
these proven internal stack tensors join the existing stack-internal retire
batch. A focused DAv2 `vits_140` bridge run after the recovery-flush guard
showed old-path pending retire bytes drop from 3,576,384,928 to 2,104,055,200
and QKV hypothetical bytes drop from 1,472,329,728 to zero under the contract
spelling, with bridge sanity passing, CPU fallback zero, and sync readback
zero. `retire_queue_drain` submit count and `stack_scope_end` count were
unchanged, so this is a retire-pressure reduction, not submit-count reduction.
After the stack-scope activation expansion, a historical focused 10-repeat DAv2
`vits_140` context-owned bridge run with decoder bridge planned recording
reported `single_image_forward_device_resident` mean 78.64 ms, median
78.68 ms, p95 81.55 ms, bridge sanity `max_abs=1.639e-06`, CPU fallback zero,
sync readback zero, and buffer copies zero. The measurement window accepted
2,280 stack-internal retire-batch rows / 634,344,960 bytes across
`stack_attention_output`, `stack_fc1_gelu_output`, `stack_fc2_output`,
`stack_norm1_output`, `stack_norm2_output`, `stack_proj_output`,
`stack_qkv_output`, and `stack_residual1_output`; the only rejected stack role
remained `stack_residual2_output`, because those rows are requested/final
capture outputs.
A follow-up graph/benchmark proof keeps that rejection explicit while exposing
the private bridge dependency. `StackOutputToDeviceConsumerBridgeContract.v0`
registrations are emitted before the captured tensor object exists, so their
producer identity fields now report
`producer_identity_unavailable_registration_emitted_before_capture_tensor`
instead of silently omitting identity. The runtime allocation identity is still
recovered from stack-lifetime rows. The DAv2 dry-run summary now consumes both
`explicit_synchronize` and `retire_queue_drain` stack-owner phase-boundary rows;
a focused one-repeat `vits_140` `segmented_stack_wide4_to_exit` run with
`StackScopeRetireHandoffContract.v0` reported 12 proven private bridge
dependency records for blocks 2, 5, and 8, zero missing runtime identity, and
12 candidate phase-boundary/retire-drain queue-submit records. Block 11 remains
excluded as final output. This is proof-only: the next behavior target is a
private bridge capture handoff / decoder-consumer ownership contract, not
admitting `stack_residual2_output` into the generic stack-internal retire batch.
`PrivateBridgeCaptureHandoffContract.v0` is the first behavior-neutral row for
that target. It records, per bridge capture slot, the raw residual2 capture
identity after the stack returns, the post-bridge-LayerNorm identity, and the
prefix-stripped decoder-preprocess input view identity. The row is generic to a
private bridge capture chain: it carries allocation id/generation/range,
storage offset, shape, consumer slot, and public/host/readback blockers, while
recording `transfers_pending_retires=0` and `submit_elision_enabled=0`. This
distinguishes requested private bridge captures from stack-internal temps and
sets up a future decoder-consumer ownership canary without changing execution.
The same schema emits a second bridge-exit row after decoder preprocessing has
consumed the prefix-stripped views. That row records
`decoder_consumer_completed_before_bridge_exit=1`, but still does not release,
retire, defer, or submit anything. This behavioral release-boundary proof
survives the retired recording-scope mechanism.
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
`BatchNormInferenceContract`, `ChannelCatContract`, `GQARepeatContract`,
`KVCacheAppendContract`,
`LinearGeluBridgeContract`, `NoOverlapConvTranspose2DContract`, and
`SafeViewReshapeContract`, `SmallMetadataPaddedConv2DContract`,
`SmallSpatialPointwiseConvContract`, `MaskedTinySDPAContract`,
`ElementwiseBroadcastContract`, and `TransformerGQASDPAContract`,
`VisionSelfAttentionSDPAContract`, `DiffusionSDPAContract`,
`DiffusionCrossAttentionContract`, `SDPAExecutionPolicyContract`, and
`SDPAScoreSoftmaxContract` are split into family-specific sources. The semantic
`TokenPrefixCatAddDirectBuffer` family also has its own source. The former
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
and `AttentionProbabilityMaterializationContract`.
The comparison tool
`tools/vulkan_contract_codegen/compare_contract_admission.py` reports admitted
row deltas, cardinality increases, exact-row debt changes, and stale dependency
digests; it is governance-only and does not change runtime route behavior.

`TokenPrefixCatAddDirectBuffer` covers rank-3 prefix-token concat plus
position-add by semantic runtime guards: fp32 Vulkan buffers, prefix length
`1`, dim `1`, positive token count, matching batch and feature dimensions, and
output sequence `1 + token_count`. The route writes a real contiguous Vulkan
output. Corpus tests retain the former DAv2 token-count and feature-dimension
combinations without using them as an admission boundary.

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

`FeatureMapToTokensDirectBuffer` replaced the bounded patch-embed
feature-map-to-token row contract. The private benchmark wrapper now admits
fp32 rank-4 Vulkan buffers by `[N,C,H,W] -> [N,H*W,C]` semantics and hard-fails
unsupported dtype, rank, storage class, width packing, storage offset, or
buffer-compute capability. Corpus-shape parity coverage retains all 24 former
row combinations, and dynamic tests cover the former adjacent shape bounds,
including `(40,62)`, without keeping a production exact-row matcher or
generated fixture.

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
Direct attention score/probability scratch tensors now have a shape-plan-derived
last-use proof. For direct-attention vision stack plans with positive head and
token counts, `[heads,tokens,tokens]` `stack_attention_output` buffers are
classified as same-block attention-internal resources whose final consumer is
the attention program itself. This is a lifetime/provenance proof only: it does
not change attention kernels, SDPA admission, transition materialization policy,
submit elision, or pending-retire handoff behavior.

Stack-scoped LayerNorm internal statistic buffers now have a shape-plan-derived
last-use proof for `[tokens,1]` `stack_norm1_output` and `stack_norm2_output`
resources. Under `PYTORCH_VULKAN_STACK_SCOPE_RETIRE_HANDOFF`, only those
exact TensorAllocation rows can join the stack-internal retire batch after the
same formal internal/non-escaping/final-consumer proof used by the existing
stack-scope retire handoff. The legacy
`PYTORCH_VULKAN_STACK_REGION_BATCH_QKV_RETIRES` spelling is retired; use the
contract-facing stack-scope handoff spelling for any future retire-batch proof.
Unscoped LayerNorm buffer-width cleanup remains
split out of the generic
`stack_internal_temp_raw_generation_range_missing_last_consumer` diagnostic
bucket as `layernorm_buffer_width_unscoped_cleanup` when the allocation label is
one of the generic buffer-width LayerNorm families. Those rows still report no
formal last-use proof, remain unsafe retire candidates, and continue to count
against missing-consumer ordering proof until they have their own scoped
producer/consumer contract.

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
- HY-MT small decode GQA first used the direct GQA buffer SDPA plan for admitted
  `TransformerGQASDPAContract` `SmallNonCausalGQA` rows; the newer
  `DirectDecodeGQASDPADirectBuffer` path removes the finite source-length and
  head-dim row gate for legal non-causal decode GQA shapes. A focused RX 9070
  16-token sanity run before the dynamic expansion reduced device-resident
  generate from the earlier 52.6 s class to 29.7 s, `cpu_fallback` from 1927 to
  967, and tensor CPU readback submits from 8355 to 6435. KV-cache append
  small-sequence broadening remains blocked: a scratch `S=1..115`
  sequence-append candidate reduced readback pressure but changed
  generation-token behavior, so it was not promoted.
- PaddleOCR RX 9070 screenshot: stable in the Task179 single row. It reported
  `cpu_fallback=1`, `sync_readback=1`, `tensor_cpu_readback=1824`, and
  `conv_prepack_upload=140`; the earlier first-attempt DeviceLost did not
  reproduce in that run.
- Gemma E2B: still blocked before useful route coverage by model-weight Vulkan
  OOM while moving
  `gemma4forconditionalgeneration.model.language_model.embed_tokens_per_layer.weight`.
- Lotus: Task181 encountered a source-tree build without the compiled DTensor C
  API `_DTensor_OpSchema_post_init`. The current 2026-07-17 Visual Studio build
  has `BUILD_PYTHON`, distributed, Gloo, C10D-Gloo, and libuv enabled; the
  loaded runtime exports both `_distributed_c10d` and
  `_DTensor_OpSchema_post_init`, and the model-suite DTensor preflight passes.
  This clears the recorded environment blocker but does not establish a fresh
  end-to-end Lotus result. Lotus remains excluded from backend regression
  budgets until that model run is recollected.

Benchmark-local distributed shims must stay import-only and single-process.
`_c10d_functional.wait_tensor` may be an identity shim for telemetry imports;
collective and DTensor op schema stubs must raise if executed. Do not add
benchmark-local fakes for compiled `torch._C` DTensor APIs. If the Lotus
preflight regresses, repair and deploy the real `torch_python` build rather than
changing the Vulkan backend or faking the missing API.
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
- `agent_space/task181_lotus_shim_validation.md`: historical benchmark-local
  Lotus shim validation and the now-cleared DTensor C API blocker.

These files are diagnostic inputs. Production code must not depend on
`agent_space`.

## Current Contract Groups

- `SmallSpatialPointwiseConvContract`: finite projection rows, now split into
  a family-specific source. The `SparseProjectionRows` slice has a JSON
  contract spec backed by `ShapeEnvelope` v1 `sparse_rowsets` with all 67
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
  The `depth_projection_384_18x10_192` and
  `depth_projection_384_18x10_384` rows are exact sparse rows for the vits
  decoder projection pair and do not promote `18x10` into the factorized
  spatial set.
  `SmallSpatialPointwiseConvContract` `GenericDynamicHW` is the first adaptive
  counterexample to the exact-row operating model: it admits legal fp32,
  rank-4, direct-buffer 1x1 pointwise convolutions with runtime batch/H/W
  under semantic 1x1/direct-buffer guards, then
  runs the existing dynamic-shape `conv2d_buffer_float_1x1` shader instead of
  requiring a sparse `(input_c, input_h, input_w, output_c)` row. Batch-one
  width-packed cases may select the existing as-linear plan from this dynamic
  admission. Sparse pointwise rows remain evidence and regression fixtures, not
  the required admission mechanism for every unseen legal H/W.
  `ElementwiseBroadcastDirectBuffer` applies the same operating model to
  fp32 Vulkan buffer add/mul/sub: rank, dtype, layout, attributes, and
  broadcast compatibility are semantic requirements, but exact shapes are not.
  `OCRProjection` sparse rows are retained as finite evidence fixtures around
  the dynamic pointwise family, including the observed batch-3 and batch-6 OCR
  cases. They are no longer production admission gates for legal fp32
  direct-buffer 1x1 runtime shapes.
  This does not add a model-name route.
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
  remain handwritten. The direct GQA buffer execution plan is now permitted for
  matched causal prefill rows, matched `SmallNonCausalGQA` decode rows, the
  semantic `DirectDecodeGQASDPADirectBuffer` family for legal non-causal decode
  GQA runtime shapes, and the semantic
  `DirectCausalPrefillGQASDPADirectBuffer` family for legal causal prefill GQA
  and equal-head MHA runtime shapes. q>1 non-causal GQA, unequal-head MHA
  within bounded target/source lengths uses the semantic
  `SmallNonCausalGQASDPADirectBuffer` family, while equal-head non-causal MHA
  with direct-buffer-compatible lane-aligned head/value dims uses
  `DirectNonCausalMHASDPADirectBuffer`. Unequal-head MHA without `enable_gqa`,
  masked, dropout, over-budget, and
  materialized/repeat execution policies remain rejected.
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
  explicit tuple contracts are now evidence for the original diffusion rows,
  while `CrossAttentionRuntimeShape` and
  `SquareSelfAttentionRuntimeShape` admit legal mask-free fp32 rank-4 runtime
  diffusion attention by semantic head/sequence/head-dim/score-budget guards.
  Runtime square admission is `head_dim=64` plus single-head `head_dim=512`
  when the square sequence is width-pack compatible for the materialized key
  transpose. The existing `head_dim=512` rows remain evidence, not production
  allowlist gates; non-compatible `512` sequences still require a broader
  direct-buffer materialization command plan.
  Square runtime admission is paired with
  `DiffusionMaterializedSquareRuntimeShape`, which preserves score
  pre-materialization and post-softmax clone behavior.
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
  remain handwritten. `RecognizerNonCausalMHARuntimeShape` and
  `TransformerDecodeGQACloneOnlyRuntimeShape` now cover their runtime semantic
  slices without exact row admission; keep remaining exact rows until broader
  layout-transition and materialization behavior is proven.
- `SDPAScoreSoftmaxContract`: finite float rank-3 square score-softmax
  evidence plus runtime square score admission. The `DiffusionSquareScores`
  slice covers heads `{1, 5}` and sequence `{504, 640}` with a JSON contract
  spec backed by `ShapeEnvelope` v1, checked-in positive/adjacent-negative
  runtime cases, and generic ShapeEnvelope C++ simple-bound helper output in
  `generated/ExecutionContractsSDPAScoreSoftmaxSpec.h`.
  `DiffusionSquareScoresRuntimeShape` admits runtime square fp32 score tensors
  within head/sequence/score-element budgets. The
  `VisionSelfAttentionScores` slice is a bounded production materialization
  edge for the six existing VisionSelfAttention score rows `[BH,T,T]` where
  `BH in {6,12,16}` and `T in {151,261}`; it consumes the generated
  `VisionSelfAttentionSDPAContract` rowset as source of truth and keeps direct
  softmax-probability-to-value-BMM disabled. Softmax route ordering, guard
  fallback labels, buffer softmax policy, and match-result assembly remain
  handwritten. Keep the temporary exception until broader score-softmax/layout
  behavior is proven.
- `SmallMetadataPaddedConv2DContract`: the original exact PaddleOCR
  `MaterializedBufferInput2x2` row is now evidence only for a semantic
  `RuntimeMaterializedBufferInput2x2` matcher. Production admission is based on
  batch-one fp32 width-packed non-direct small-channel 2x2 layout/materialization
  guards with random-shape coverage, not exact input height/width/output-channel
  rows. Batched inputs, other kernels, grouped convs, and direct-output
  ownership remain outside the family.
- Embedding lookup: `EmbeddingLookupDirectBuffer` owns the supported native
  route by runtime dtype, rank, layout, semantic-option, dispatch-limit, and
  index-bounds checks. The broader route made the finite token-batch and
  small-bounded matcher unreachable, so that exact contract, generated spec,
  duplicate dispatch, and mechanism-only tests were deleted. Device-produced
  indices without a bounds proof still fall back under the temporary
  correctness exception.
- `CatAxisContract`: umbrella for last-dim, channel-dim, and rank-3 cat
  patterns. The `ChannelCatContract` rank-4 dim-1 buffer slice has a JSON
  contract spec with generated positive and adjacent negative runtime coverage,
  but production admission for fp32 buffer-backed rank-4 dim-1 cats now flows
  through `CatAxisDirectBuffer` semantic runtime validation. Input count,
  spatial size, batch, and total channels are runtime descriptor values. The
  old generated rowset remains review evidence while the current implementation
  keeps the channel multiple-of-4 layout constraint.
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
- `GQARepeatContract`: the old
  `Batch1Heads4Factor4Sequence100To116Dim128` slice is now evidence for the
  materialization route, while `GenericRuntimeShape` admits fp32 rank-4
  Vulkan buffer K/V tensors by runtime batch/head/source/head-dim and repeat
  factor semantics. SDPA may select the materialized repeat route only for
  non-causal, mask-free GQA with both K and V matched and a downstream
  rectangular rank-3 score tensor accepted by
  `RectangularScoresRuntimeShape`. Random unseen-shape tests force query
  lengths above the direct-GQA small-shape limit and require the
  `aten::scaled_dot_product_attention.bounded_gqa_repeat_materialize` op hit.
  The generated helper still provides exact-row metadata for regression
  fixtures, but exact source-length rows are no longer production admission for
  the materialization shader.
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
- `LinearGeluBridgeContract`: semantic optimized bridge for deferred
  linear->GELU fusion. Generic linear and matmul execution is still owned by
  `LinearOrMatmulDirectBuffer`; the bridge now uses `GenericRuntimeShape`
  admission for legal rank-2/rank-3 fp32 Vulkan linear shapes when input
  flattening matches the packed weight input dimension, output features are
  positive, bias is present, `out=` is absent, alpha/beta are `1`, and GELU
  approximation is `none` or `tanh`. The old
  `BackboneMlpHidden384To1536` ShapeEnvelope slice remains checked-in evidence
  and regression coverage, not a runtime shape gate. Non-GELU consumers still
  materialize the deferred linear output before proceeding.
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
  keeps family-specific shape checks for BatchNormInference, ChannelCat,
  KVCacheAppend, LinearGeluBridge, GQARepeat, MaskedTinySDPA,
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
  ChannelCat and both SafeViewReshape direct-buffer slices have
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
- Embedding lookup no longer has an exact ShapeEnvelope or generated C++
  helper. Behavioral tests cover the sole dynamic semantic route for both
  host-validated CPU indices and provenance-validated Vulkan indices, plus
  invalid-index error and fallback behavior.
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
  per-row metadata, input/weight channel equality, the 56 correlated
  projection rows, exact lookup by input/output channel and spatial shape, and
  the generated 144-shape factorized depth-vision projection helper. The sparse
  rows now include sixteen exact mid-resolution depth-vision projection rows for
  spatial pairs `(30,45)` and `(40,62)` with only the proven channel/output
  pairs, plus the exact PaddleOCR OCR row `(512,3,80,512)`. Those spatial
  pairs were not added to the 144-shape factorized helper.
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
- The current source-tree runtime passes the Lotus compiled-DTensor preflight,
  but no fresh end-to-end Lotus telemetry has been collected from that build.
  Do not fake compiled `torch._C` DTensor APIs if the preflight regresses; fix
  and deploy the real `torch_python` runtime before treating Lotus as backend
  evidence.
- PaddleOCR completed the Task179 RX 9070 screenshot row with one known CPU
  fallback and one sync readback, but that is still telemetry-only and not
  cross-adapter gate-ready. Rerun the real-model matrix after the next backend
  behavior change or before claiming or raising a model gate.

## Build Context

On this Windows machine, use the existing Visual Studio CMake build tree from
`build/CMakeCache.txt`. The local cache records Visual Studio 17 2022, x64,
Release, `USE_VULKAN=ON`, `USE_VULKAN_API=ON`, strict SPIR-V, Vulkan 1.3, and
SPIR-V 1.6 targets.
