# Vulkan Graph Evidence

`scripts/benchmarks/vulkan_graph_export_evidence.py` records a real external
model run through the checked-in `torch.vulkan.export_and_lower` executor. It
is a corpus harness, not a backend route: the model-specific construction and
input policy live in an external adapter.

The adapter is a callable specified as `MODULE:FACTORY` or `FILE.py:FACTORY`.
It receives `external_root` and `checkpoint` paths and returns a mapping with:

```python
{
    "model": model.eval(),
    "normal_inputs": (cpu_tensor,),
    "alternate_inputs": (cpu_tensor,),
    "dynamic_shapes": export_dynamic_shapes,
    "out_of_range_inputs": (cpu_tensor,),
    "state_replay_input_from_output": ((target_leaf, source_leaf),),
}
```

`normal_inputs` and `alternate_inputs` must satisfy the same exported dynamic
guards. `out_of_range_inputs` is optional, but when supplied it must be
rejected by the exported guard. The harness runs CPU export, graph lowering,
eager Vulkan and CPU parity references, and a readback-separated repeated
graph reference run. It forces the remaining eager deferred canaries off for
the graph execution.

`state_replay_input_from_output` is optional. Each row maps one flattened leaf
of `alternate_inputs` to a flattened output leaf from the normal graph program.
When present, the harness runs the normal program, replaces the declared
alternate CPU state leaves with its still-device-resident outputs, and invokes
the alternate guard variant. It verifies source-state parity, target-output
parity, zero implicit host boundaries, and source-output lifetime after the
target invocation. This is an explicit state protocol; it cannot infer state
from model names or silently copy state through CPU.

Some model frontends export a guard-specialized program even when the adapter
supplies a broader symbolic policy. If the normal program rejects the legal
alternate input at `_guards_fn`, the harness records that rejection and creates
one explicit export-and-lower guard variant for the alternate input. The result
is reported as `recompiled_guard_variant` with both program identities. This
is a graph-program cache variant, not a Vulkan exact-shape admission row. Any
other execution error remains a failure; the harness never uses CPU fallback.

Example:

```text
python scripts/benchmarks/vulkan_graph_export_evidence.py \
  --adapter C:\work\dav2_adapter.py:build \
  --external-root C:\external\Depth-Anything \
  --checkpoint C:\external\Depth-Anything\checkpoints\depth_anything_v2_vits.pth \
  --output-dir C:\results\dav2_vits_graph \
  --source-git-sha <full-HEAD-SHA> \
  --device-index 1 \
  --planning-model-domain vision --planning-execution-phase none \
  --planning-prefer-packed-layout-propagation \
  --planning-fixed-shape-graph-input \
  --eager-atol 0.0 --eager-rtol 0.0 \
  --cpu-atol 0.004 --cpu-rtol 0.0 \
  --latency-warmup-repeats 3 --latency-measurement-repeats 10
```

`--device-index` selects the adapter before constructing the external model or
any Vulkan tensor. The artifact records that index, device name and type, and
reported total memory with the existing driver and loaded-DLL identity. This
makes constrained-adapter evidence explicit instead of depending on process
device order.

The caller-owned output directory receives measured census and parity
artifacts. The checked-in DAv2 and PaddleOCR evidence records the v8 executor
at source commit `4b688faac338f3784a1286a327292735a3b334b0` and
`torch_cpu.dll` SHA-256
`537802036062d3277a4d74ad7a27f28a76a16f7f4a4022c1ff1e132052989a9f`.
The DAv2 census lowers all 12 `linear_gelu_none` candidates with no rejection;
PaddleOCR remains the control with no such candidates. Both corpora report zero
unsupported nodes, exact graph-versus-eager Vulkan parity, and zero runtime CPU
fallback, sync readback, or deferred-value creation. DAv2 executes its complete
404-instruction C++ plan with all 20 graph-region calls under the outer owner.
PaddleOCR encodes its omitted `avg_pool2d` stride as a schema-typed empty list
recipe and executes its complete 290-instruction C++ plan. Both record
top-level invocation generations, completed nonzero final timeline tokens,
bounded owner checkpoints, graph-invocation counters, and submit origins.
Future measurements are caller-owned until they are deliberately reviewed and
replaced. The harness
requires an explicit source SHA when `git` is not on `PATH`, so a sanitized
runtime cannot emit unproven provenance.

The checked exact-SHA HY-MT prefill artifacts at
`019faaebf1593fd2f2fcbd8e5cec66a8202fd62e` on the 8 GB GTX 1080 capture
3,160 nodes, lower 225, report zero lower-time unsupported nodes, execute the
complete immutable C++ plan, and return 65 tensor outputs. Static inference
identity lowering consumes all 64 proven fresh detaches, so each plan contains
2,668 instructions, 2,402 IValue slots, 268 ordered effects, and 129 typed list
arguments. Both the four-token case and the guard-recompiled five-token case
stay within the recorded eager/CPU tolerances with zero graph fallback, sync
readback, or deferred-value creation. Each two-run evidence case records 176
graph-owner checkpoint flushes, four host uploads, 130 explicit evidence output
readbacks, 310 total queue submits, and no retire-drain submits. Graph first and
repeat high-water ranges from 4.0% below to 0.04% above supported eager.

The same checked artifacts contain three warmups and 30 alternating samples
per supported surface. Four-token medians are 2,193.66 ms graph versus
1,742.38 ms eager, with p95 at 2,232.31 ms versus 1,759.69 ms. Five-token
medians are 1,404.20 ms graph versus 1,786.36 ms eager, with p95 at 1,461.69 ms
versus 1,797.07 ms. The distribution is therefore mixed: it proves the
measurement gate and a shape-dependent result, not graph latency no-regression.
Graph diagnostics remain explicitly `LLM`/`Prefill` with the `LLM` route lane
and zero timed fallback/readback. Plain eager still mixes `Generic` and `LLM`,
selects the legacy `DepthDiffusion` attention lane, and records five fallbacks
plus one readback per timed invocation. Lane parity and the HY-MT latency
deletion gate remain open. The reviewed artifacts are checked in as
`test/vulkan_graph/evidence/hymt_prefill_export_census.json` and
`test/vulkan_graph/evidence/hymt_prefill_export_parity.json`; the external
model adapter remains caller-owned.

The checked exact-SHA HY-MT decode artifacts at
`79bf8d01ef0db5c01997042071e12434eac1b443` use separate first-step and
second-step guard variants on the same 8 GB GTX 1080. Each variant captures
3,224 nodes, lowers 225 linear contexts, and compiles a complete
2,732-instruction/2,530-value C++ plan with 66 inputs and 65 tensor outputs.
The 66 leaves are the token, attention mask, and 64 flattened key/value cache
tensors. Both variants stay within the eager and CPU tolerances, keep graph
fallback/readback/deferred-value counters at zero, and keep first/repeat peak
memory from 4.03% below to 0.02% above eager.

The explicit replay row maps all 64 first-step cache outputs to the second
guard variant's cache inputs. Across the chained pair it records 68 host
uploads: 66 initial inputs followed by only the next token and attention mask.
The state handoff itself records no host upload or output readback. Two graph
scopes capture two final tokens, issue 172 owner checkpoints and 240 total
submits, and retain the first-step 65 outputs after the second step completes.
Replayed state, second-step output, and preserved first-step output all remain
within CPU tolerance, with zero implicit host boundary counters.

The 30-sample latency result remains a rejection gate. First-step medians are
2,188.41 ms graph versus 1,324.03 ms eager; second-step medians are 1,897.01 ms
versus 1,341.53 ms. Graph p95 is 2,199.57/1,923.29 ms versus eager at
1,337.71/1,356.77 ms. Plain eager also retains its `DepthDiffusion` lane and
five fallbacks plus one readback per timed invocation, while graph execution is
explicit `LLM`/`Decode` and clean. The state protocol is accepted; latency and
lane parity are not. The reviewed files are checked in as
`test/vulkan_graph/evidence/hymt_decode_export_census.json` and
`test/vulkan_graph/evidence/hymt_decode_export_parity.json`.

The same caller-owned probe identifies 64 `aten::detach_` candidates. Every
candidate has a single-user producer chain rooted at `aten::lift_fresh_copy`,
so the generic preparation pass rewrites all 64 to functional `aten::detach`
with no rejection. Placeholder aliases and branched fresh values are covered
as adjacent negatives and remain mutable. This removes the last representation
blocker for the current prefill and transfers per-node execution plus top-level
submission/completion ownership to C++; it does not claim that memory or
descriptor ownership has transferred.

The exact-SHA v8 DAv2 evidence admits `aten::sym_size.int` through its immutable
CompositeImplicitAutograd registration and executes graph-classified integer
`add`, `sub`, `mul`, and `floordiv` with checked C++ semantics. The full DAv2
graph crosses those former representation blockers plus multi-schema-return
SSA and bounded list projection. Its two `aten::relu_` candidates are fed by
single-use, non-aliasing functional `aten::conv2d` results, and the generic
fresh-ReLU pass functionalizes both with no rejection. Normal and alternate
shapes compile and execute a 404-instruction, 425-value C++ plan with two
ordered effects, eight graph-scalar instructions, 20 list projections, 53 list
arguments, and one output. Both cases retain exact graph-versus-eager parity,
stay within the existing CPU tolerance, and report zero CPU fallback, sync
readback, or deferred-value creation. Placeholder, aliasing-view, and branched
producer tests remain fail-closed. The positive corpus result is exact-SHA
checked-in evidence; the adjacent rejection cases are unit-level contract
proofs. The matching exact-SHA v8 PaddleOCR measurement executes both shapes as
a complete 290-instruction, 294-value C++ plan with one ordered effect, 14 list
arguments, and one output. It retains exact graph-versus-eager parity, stays
within the existing CPU tolerance, and reports zero fallback, readback, or
deferred-value creation. Each PaddleOCR shape runs twice and records two scopes,
two final-token captures, 22 owner checkpoint flushes, two input uploads, two
output readbacks, 26 total queue submits, and no normal-frequency or
retire-drain submits. DAv2 reports `submission_owned=true` for the complete
plan and both shapes. Each two-run shape records two outer scopes and final
tokens, 20 owner checkpoint flushes, zero retire-drain submits, and 24 total
submits. The preceding supported artifacts recorded 26/30 flushes/submits,
then 38/42 before scratch-token inheritance; the earlier graph artifact
recorded 16 scopes, 56 retire-drain submits, and 92 total submits. Graph versus
eager Vulkan remains
exact, CPU tolerance remains satisfied, and runtime fallback, readback, and
deferred-value counters remain zero. The current DAv2 repeated-run samples are
supplemented by supported-default distributions with three warmups and 30
alternating samples per surface. DAv2 graph medians are 40.14 ms and 41.91 ms
versus eager medians of 114.08 ms and 116.76 ms; PaddleOCR graph medians are
42.61 ms and 54.94 ms versus eager medians of 135.79 ms and 145.76 ms. Graph p95
is below eager p95 in all four cases, and timed fallback/readback counters stay
zero. This clears the recorded-shape latency no-regression bar against the two
supported defaults. The checked HY-MT evidence retains its complete plan and
owned checkpoint/token evidence, but its mixed distribution and eager lane
mismatch keep the HY-MT deletion gates open as described above.

An exact-SHA `ed4975687b6` RX 9070 fixed-cost pass reuses the supported DAv2
graph boundary and isolates timestamp and CPU-timeline instrumentation after
warmup. `vits_140` records 309 GPU events and 23.9 ms of summed GPU work per
inference; `vits_280` records 303 events and 32.7 ms. Both issue 24
`pending_command_flush` checkpoints per unprofiled inference, with zero timed
fallback or readback. The CPU summary attributes about 3.4 ms and 2.9 ms per
inference to measured dispatch recording plus submit calls. Current-SHA
30-repeat uninstrumented medians are 49.06 ms and 49.09 ms, so the larger input
consumes idle submission/queue slack rather than extending wall latency. GPU
timestamp wall times are intentionally excluded because profiling adds a reset
submit and substantial collection overhead. The earlier checked-in 41.0/41.2
ms distributions were its supported deletion baseline. The next optimization
question was generic graph checkpoint batching, not another shape-specific
operator route. Caller-owned raw reports are under
`agent_space/dav2_graph_fixed_cost_ed4975687b6/`.

The exact-SHA `25b66ba0b8b` cadence keeps eager's frequency at 16 and gives
graph execution its own frequency of 24. DAv2 drops from 24 to 19 graph
checkpoints per inference, PaddleOCR records 14, and caller-owned HY-MT drops
from 168 to 114. Exact supported-default memory remains within the 5% gate for
all three corpora. Same-binary DAv2 30-repeat medians are 40.20 ms and 36.78 ms,
18.1% and 25.1% below the `ed4975687b6` attribution medians. Candidate
frequencies of 64 and 32 were rejected: DAv2 at 64 and PaddleOCR at 32 exceeded
the 5% repeat-with-live-output peak-memory gate. The accepted evidence therefore
supports bounded fixed-cost reduction, not unrestricted checkpoint deferral.
Exact-SHA `b157c550fc5` next-submission token inheritance for bounded
conv-region scratch removes the region-exit checkpoint and drops DAv2 from 19
to 13 pending submissions per inference. Normal and alternate graph medians are
42.10 ms and 40.97 ms against eager at 116.13 ms and 121.64 ms, with graph
repeat-with-live-output memory 0.8% and 1.8% above eager. PaddleOCR remained at
14 submissions and caller-owned HY-MT at 114; all recorded memory phases stayed
inside the 5% gate. Those DAv2 and PaddleOCR manifests were the supported
deletion baseline at that stage. Historical raw artifacts are under
`agent_space/graph_checkpoint24_exact_25b66ba0b8b/` and
`agent_space/dav2_graph_checkpoint24_79080a576b0/`; the inheritance artifacts
are under `agent_space/graph_submission_inheritance_exact_b157c550fc5/`.

Exact-SHA `4b688faac33` uses SSA last-use, non-escape, unique Vulkan-storage, and
live-alias guards to reuse dead `aten::relu` inputs, then widens the graph
cadence from 24 to 32 jobs. DAv2 records 10 submissions per inference with
30-sample graph medians of 40.14/41.91 ms and peak memory 0.9% to 3.2% above
eager. PaddleOCR records 11 submissions with graph medians of 42.61/54.94 ms
and peak memory 1.4% to 4.3% above eager. The checked manifests expose nine
DAv2 and 53 PaddleOCR candidate instructions plus the accepted-reuse counters.
Caller-owned HY-MT records 88 submissions and stays between 4.0% below and
0.05% above eager peak memory. All cases preserve recorded correctness and zero
graph fallback/readback. The DAv2 and PaddleOCR manifests are now the checked
supported-default deletion baseline; current raw exact-SHA artifacts are under
`agent_space/graph_dead_relu_reuse_checkpoint32_exact_4b688faac33/`.

Exact-SHA `1fb325d1d0c` removes the executor's per-instruction counter-vector and
boxed-stack allocations while preserving the vector-returning Python diagnostic
surface. Its 30-sample DAv2 control keeps the 10-submission cadence, identical
memory phases, exact graph/eager parity, and zero fallback/readback. Graph
medians are 39.10/40.07 ms versus eager at 110.64/111.37 ms; normalized ratios
are effectively unchanged from the checked baseline, so the evidence establishes
no regression rather than an isolated latency win. Raw files are under
`agent_space/graph_executor_fixed_alloc_exact_1fb325d1d0c/dav2/`.

Exact-SHA `8b60bf3ba4a` extends that ownership across invocations. The immutable
plan preallocates 425 boxed value slots, byte liveness, a dispatcher stack with
capacity eight, and 33 of 53 typed list arguments. The other 20 list arguments
belong to list-returning instructions and remain transient to prevent input/output
container aliasing. Repeated success and repeated-exception tests exercise the
same workspace. The exact DAv2 run preserves 10 pending submissions per
inference, exact graph/eager parity, zero fallback/readback, and first/repeat
high-water between 0.9% and 3.2% above eager. Graph medians are 44.21/42.09 ms
versus eager at 133.32/122.63 ms. Absolute host load varied enough across runs
that this establishes structural fixed-cost ownership and no regression, not a
separate latency win. Raw files are under
`agent_space/graph_executor_workspace_exact_8b60bf3ba4a2/dav2/`.

Exact-SHA `46ece5d7dc9` adds a reported static-inference identity pass before
tensor placement. It removes `aten::dropout` only for valid static probability
and either disabled training or zero probability; unit regressions preserve
training dropout and invalid-probability validation. DAv2 lowers all 48
candidates, reducing the immutable plan from 404 instructions/425 values to
356/377. Both shapes retain exact graph/eager parity, zero fallback/readback,
10 pending submissions per inference, and first/repeat high-water from 0.9% to
3.2% above eager. Canonical graph medians are 44.73/50.12 ms versus eager at
138.50/142.87 ms; their averaged normalized ratio is effectively unchanged from
the previous exact artifact. This proves fixed control-plane removal and no
regression rather than a separate speedup. Checked and raw files come from
`agent_space/graph_static_inference_identity_exact_46ece5d7dc93/dav2_checked/`.

Exact-SHA `e536f16cf36` consumes the fresh single-user detach-functionalization
proof as a static inference identity. It does not erase arbitrary detach aliases.
The caller-owned GTX 1080 HY-MT artifact lowers 64/64 proven detaches and shrinks
both four- and five-token plans from 2,732 instructions/2,466 values to
2,668/2,402. Numerical errors exactly match the earlier artifact, graph
fallback/readback remains zero, the 88-checkpoint cadence is unchanged, and
first/repeat high-water stays between 4.0% below and 0.035% above eager. One
sample per surface is insufficient for a latency claim; the HY-MT distribution
and lane-parity gates remain open. Raw files are under
`agent_space/hymt_static_detach_identity_exact_e536f16cf36/`.

The checked exact-SHA `019faaebf1593fd2f2fcbd8e5cec66a8202fd62e`
follow-up preserves the same 2,668-instruction plans, numerical errors,
88-checkpoint cadence, graph fallback/readback result, and memory envelope. Its
30-sample distribution is mixed: graph is 25.9% slower for the four-token
guard and 21.4% faster for the five-token guard. This closes the missing
measurement-evidence item without claiming latency parity. Plain eager's
`DepthDiffusion` lane and five-fallback/one-readback behavior remain the
explicit blockers.

Exact-SHA `c8332a964bb` validates the immutable C++ plan and its complete SSA
release schedule once at construction. Runtime execution consumes per-instruction
release IDs directly instead of revalidating every instruction/argument and then
rescanning argument/output recipes for last uses. The exact DAv2 artifact keeps
356 instructions, 377 values, exact graph/eager parity, zero fallback/readback,
10 pending submissions per inference, and first/repeat high-water between 0.9%
and 3.2% above eager. Across two 30-repeat passes, combined graph/eager median
ratios are 0.348/0.344; host load moved enough that this is not an isolated
latency claim. The change advances structural resource lifetime ownership but
does not clear replay, compiled-session, or stack deletion gates. Raw files are
under `agent_space/graph_release_schedule_exact_c8332a964bb/`.

A worktree 64-job cadence on the same implementation cuts DAv2 to five pending
submissions per inference but fails the deletion gate: normal first/repeat peak
memory is 5.6%/6.1% above eager and alternate is 8.5%/9.9% above eager. Its
42.12/43.07 ms graph medians are also slower than the 32-job control. The wider
cadence is rejected and was removed; raw evidence is under
`agent_space/graph_checkpoint64_post_reuse_worktree/dav2/`.

The intermediate 48-job cadence is rejected as well. It cuts DAv2 from 10 to 7
pending submissions per inference, but graph medians regress to 44.57/48.77 ms
from the 32-job control's 39.10/40.07 ms. Normal first/repeat peak memory is
2.8%/4.0% above eager, while alternate first/repeat peak memory is 5.0%/6.9%
above eager and fails the supported 5% gate. The result confirms that cadence
widening alone loses useful CPU/GPU overlap; recorded partitions or stronger
resource reuse must preserve the supported boundaries. Raw evidence is under
`agent_space/graph_checkpoint48_probe_worktree/dav2/`.

Checked-in corpus and unit-level v8 evidence add a real normal-Context ownership
scope across ordinary instructions, lifted copies, and bounded graph regions.
Multi-instruction plans capture the final timeline token per invocation,
preserve an earlier unread output across a later generation, and report
completion after readback.
Lifted-copy tests retain direct-buffer layout and the same ownership scope.
Large-linear tests force multiple owner checkpoints while the Python executor
is disabled. Command-free metadata plans advance generation without
fabricating a token.

The machine-readable evidence records graph census and lowerings, including
input normalization, static and lifted constants, fresh-detach and fresh-ReLU
functionalization, proven-identity indexing, static GQA repetition, and
explicit tensor placement. It also records an immutable-plan summary including
the graph-scalar and list-projection instruction counts, guard outcomes,
program key, Vulkan runtime identity and DLL hashes, timing, graph-invocation
and submit-origin counters, fallback/readback/deferred-value counters, and
CPU/eager Vulkan parity.

Each case also records three allocator high-water phases through the existing
`vulkan_memory_residency_snapshot` surface: supported eager, the first graph
run, and a repeated graph run while the prior graph output remains live. Reset
sets the high-water mark to the phase's current live bytes; it does not free or
hide persistent weights, contexts, or allocator state. Every phase records its
baseline live bytes, end live bytes, absolute high-water bytes, and peak delta
from the baseline. Eager temporaries are released and synchronized before the
graph phase. Absolute high-water comparisons are therefore meaningful within
the same case process, while peak deltas identify incremental activation and
temporary pressure. After capturing the first graph high-water mark, the
harness settles completed work before resetting the repeat-phase high-water
mark; the first graph output remains live throughout that repeat. Any queue
submission created by this boundary remains visible in the already-active
submit-origin counters. Checked-in evidence requires the first graph phase and
the repeat-with-prior-output-live phase to stay within 5% of their same-process
supported eager high-water mark. DAv2 is 0.7% to 1.4% below eager across the
recorded graph phases; PaddleOCR ranges from 0.2% to 3.3% above. These
fields do not authorize model-name production dispatch, exact-shape admission,
or executor performance tuning.

New measurements record structured residency evidence for the same three
phases. Allocator rows are aggregated by storage kind, lifetime state, semantic
role, and allocation label with counts and bytes; allocation IDs and pointer
identities are omitted.
Packed-weight cache summaries record live and persistent bytes plus lookup,
store, and eviction counters, while linear-pack summaries record created,
reused, packed, raw-weight, and unpacked-retention totals. Each supported eager
and graph run also captures route lanes and resolved runtime-policy fields. The
graph rows must report `inferred_from_label=0` before label inference can become
delete-ready; the eager rows preserve the supported-default comparison rather
than silently applying a harness-only planning scope.

Each case separately measures supported-default latency with preuploaded Vulkan
inputs and Vulkan outputs that are not read back in the timed region. Plain
concrete eager and `VulkanGraphProgram` receive the same device-resident inputs,
alternate which surface runs first in each measurement round, and synchronize
after every warmup and measured invocation. The artifact records every sample
plus mean, median, standard deviation, minimum, maximum, p90, and p95. The
default is three warmups and ten measurements per surface. This phase runs
after parity, submission, repeated-output, and memory snapshots have been
captured, so its explicit synchronization does not contaminate those counters.
The raw samples are checked in as a recorded-shape no-regression gate: graph
median and p95 must not exceed the matching supported eager values. A subsystem
deletion still requires a reviewed deletion unit to name this artifact and
prove that its other corpus, ownership, and resource gates are satisfied.

The default tolerances are zero. A nonzero tolerance is an explicit corpus
evidence choice and is written into the measured parity artifact. Graph versus
eager Vulkan should remain exact unless a documented backend precision contract
requires otherwise.

## GELU `none` precision contract

Eager Vulkan currently executes `aten::gelu(approximate="none")` with the tanh
kernel. A graph-owned `linear_gelu_none` region must therefore match eager
Vulkan exactly; it does not introduce a second approximation. Unit coverage
uses CPU tolerances of `atol=5e-4` and `rtol=5e-3` for this existing eager
kernel gap. The full DAv2 artifact retains its explicit corpus tolerance of
`atol=0.004`, `rtol=0.0` and records a maximum CPU difference of about
`0.00358`, while graph versus eager Vulkan remains exact. If eager Vulkan gains
true non-approximate GELU semantics, tighten this contract and refresh the
corpus evidence with the implementation change.
