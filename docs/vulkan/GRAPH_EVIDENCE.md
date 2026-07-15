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
  --eager-atol 0.0 --eager-rtol 0.0 \
  --cpu-atol 0.004 --cpu-rtol 0.0
```

The caller-owned output directory receives measured census and parity
artifacts. The checked-in DAv2 and PaddleOCR evidence records the v7 executor
at source commit `b0b484a7dfa233fb8ba881cbd0d0cbc83b8bc9a4`. The DAv2 census lowers all 12
`linear_gelu_none` candidates with no rejection; PaddleOCR remains the control
with no such candidates. Both corpora report zero unsupported nodes, exact
graph-versus-eager Vulkan parity, and zero runtime CPU fallback, sync readback,
or deferred-value creation. DAv2 executes its complete 404-instruction C++
plan. PaddleOCR encodes its omitted `avg_pool2d` stride as a schema-typed empty
list recipe and executes its complete 290-instruction C++ plan. Future
measurements are caller-owned until they are deliberately reviewed and
replaced. The harness
requires an explicit source SHA when `git` is not on `PATH`, so a sanitized
runtime cannot emit unproven provenance.

A caller-owned four-token HY-MT prefill integration probe on 2026-07-15
captures 3,160 nodes, lowers 225, reports zero lower-time unsupported nodes,
executes the complete immutable C++ plan, and returns 65 tensor outputs. The
plan contains 2,732 instructions, 2,466 IValue slots, 268 ordered effects, and
129 typed list arguments. A v7 regression probe reports zero graph-scalar and
list-projection instructions, as expected for this static prefill. Runtime
counters remain zero for CPU fallback, sync readback, and deferred-value
creation, with no Vulkan behavior overrides. This proves that static constants,
boolean mask construction, identity indexing,
GQA repetition, boolean-masked SDPA, and boxed C++ dispatch compose through one
real prefill. It is not a checked-in parity artifact: it does not compare
output values, exercise alternate dynamic guards, repeat live outputs, or
measure submits, memory, or latency, and cannot satisfy a subsystem deletion
gate by itself.

The same caller-owned probe identifies 64 `aten::detach_` candidates. Every
candidate has a single-user producer chain rooted at `aten::lift_fresh_copy`,
so the generic preparation pass rewrites all 64 to functional `aten::detach`
with no rejection. Placeholder aliases and branched fresh values are covered
as adjacent negatives and remain mutable. This removes the last representation
blocker for the current prefill and transfers per-node execution to C++; it
does not claim that memory, descriptor, submission, or completion ownership
has transferred.

The exact-SHA v7 DAv2 evidence admits `aten::sym_size.int` through its immutable
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
proofs. The matching exact-SHA v7 PaddleOCR measurement executes both shapes as
a complete 290-instruction, 294-value C++ plan with one ordered effect, 14 list
arguments, and one output. It retains exact graph-versus-eager parity, stays
within the existing CPU tolerance, and reports zero fallback, readback, or
deferred-value creation. The caller-owned v7 HY-MT probe retains its complete
plan and zero counters.

The machine-readable evidence records graph census and lowerings, including
input normalization, static and lifted constants, fresh-detach and fresh-ReLU
functionalization, proven-identity indexing, static GQA repetition, and
explicit tensor placement. It also records an immutable-plan summary including
the graph-scalar and list-projection instruction counts, guard outcomes,
program key, Vulkan runtime identity and DLL hashes, timing,
fallback/readback/deferred-value counters, and CPU/eager Vulkan parity. It does
not authorize model-name production dispatch, exact-shape admission, or
executor performance tuning.

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
