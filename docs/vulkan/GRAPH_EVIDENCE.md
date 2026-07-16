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
}
```

`normal_inputs` and `alternate_inputs` must satisfy the same exported dynamic
guards. `out_of_range_inputs` is optional, but when supplied it must be
rejected by the exported guard. The harness runs CPU export, graph lowering,
eager Vulkan and CPU parity references, and a readback-separated repeated
graph reference run. It forces the remaining eager deferred canaries off for
the graph execution.

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
at source commit `25b66ba0b8bcb641ddafc2be091f55884eb17077` and
`torch_cpu.dll` SHA-256
`11579e6b7f39c5a28ad140a59d0c89aa16956142745a3630a57c1b85aa03a824`.
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

A caller-owned exact-SHA HY-MT prefill integration artifact at `25b66ba0b8b`
on GTX 1080 captures 3,160 nodes, lowers 225, reports zero lower-time
unsupported nodes, executes the complete immutable C++ plan, and returns 65
tensor outputs. The plan contains 2,732 instructions, 2,466 IValue slots, 268
ordered effects, and 129 typed list arguments. Both the four-token case and the
guard-recompiled five-token case stay within the recorded eager/CPU tolerances
with zero graph fallback, sync readback, or deferred-value creation. The
five-token unaligned boolean causal-mask broadcast stays in the generic native
buffer path. Each two-run case records 228 graph-owner checkpoint flushes, four
host uploads, 130 evidence output readbacks, 362 total queue submits, and no
retire-drain submits. Graph peak memory ranges from 4.0% below to 0.03% above
supported eager. Graph diagnostics carry explicit `LLM`/`Prefill` semantics
with zero label inference, but supported eager still reports the legacy
`DepthDiffusion` lane. The
artifact therefore advances correctness, guard, submission, and residency
coverage without clearing the lane or latency-distribution deletion gates. The
raw caller-owned files are under
`agent_space/graph_checkpoint24_exact_25b66ba0b8b/hymt/`.

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
two final-token captures, 28 owner checkpoint flushes, two input uploads, two
output readbacks, 32 total queue submits, and no normal-frequency or
retire-drain submits. DAv2 reports `submission_owned=true` for the complete
plan and both shapes. Each two-run shape records two outer scopes and final
tokens, 38 owner checkpoint flushes, zero retire-drain submits, and 42 total
submits. The prior supported graph artifact recorded 16 scopes, 56
retire-drain submits, and 92 total submits. Graph versus eager Vulkan remains
exact, CPU tolerance remains satisfied, and runtime fallback, readback, and
deferred-value counters remain zero. The current DAv2 repeated-run samples are
supplemented by supported-default distributions with three warmups and ten
alternating samples per surface. DAv2 graph medians are 38.6 ms and 43.7 ms
versus eager medians of 111.4 ms and 118.1 ms; PaddleOCR graph medians are
44.9 ms and 51.6 ms versus eager medians of 135.4 ms and 137.9 ms. Graph p95
is below eager p95 in all four cases, and timed fallback/readback counters stay
zero. This clears the recorded-shape latency no-regression bar against the two
supported defaults. The caller-owned HY-MT worktree probe retains its complete
plan and records the owned checkpoint/token evidence described above, but does
not yet provide the matching checked-in distribution.

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
The next target is generic next-submission token inheritance for bounded
conv-region scratch. Raw accepted artifacts are under
`agent_space/graph_checkpoint24_exact_25b66ba0b8b/` and
`agent_space/dav2_graph_checkpoint24_79080a576b0/`.

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
