# Vulkan Backend Findings

Date: 2026-04-18

This note summarizes the main findings from the Vulkan backend work in this
workspace. The work started as model-zoo bring-up and gradually turned into a
backend architecture, cleanup, and performance exercise. The broad goal stayed
stable throughout:

- expose real backend gaps with real models
- move fixes into the C++ backend instead of Python glue
- make Vulkan execution more planner-owned and replay-owned
- push the vision path toward a reusable buffer-first architecture

## 1. Executive Summary

The short version is:

- The backend does not need a large refactor right now.
- The planner / execution-program / replay architecture is the right base.
- The model-zoo driven cleanup materially improved capability and surfaced the
  real remaining issues.
- DAv2 is still the best performance case study and the hardest remaining
  target.
- The next meaningful speedup is unlikely to come from isolated per-op tuning
  alone; it needs fixed-shape execution planning, broader packed-layout
  propagation, and fewer intermediate writes.

## 2. Build And Environment Findings

Known working build setup:

- build root: `pytorch/build`
- generator: `Visual Studio 17 2022`
- Vulkan SDK: local SDK at `VulkanSDK/1.4.341.1`
- do not force `BLAS=MKL`
- sanitize the environment so only one effective `PATH` reaches CMake/MSBuild
- build `torch_cpu` with `/m:1`

Important build failure modes:

- the old incremental `pytorch/build` could become poisoned and needed a clean
  regenerate
- `/m:8` could fail on generated project references such as
  `ZERO_CHECK -> GetNativeManifest`
- duplicate `Path` / `PATH` entries could poison MSBuild
- after rebuilding, the active Python checkout used
  `pytorch/torch/lib/torch_cpu.dll` and `pytorch/torch/lib/c10.dll`, so staged
  binaries had to be copied from `pytorch/build/bin/Release/`

Practical rule:

- a successful build is not enough; verify that the locally imported Python
  package actually loads the rebuilt DLLs

## 3. Major Architecture That Landed

The backend now has a real planning/runtime stack under:

- `Request.h`
- `Capabilities.h`
- `Runtime.h`
- `Scheduler.h`
- `ExecutionObjects.h`
- `ExecutionPrograms.h`
- `InferenceGraphs.h`

Important architectural pieces now exist:

- backend-owned execution objects
  - `KVCacheObject`
  - `ScratchArena`
- capability-driven kernel family selection
- scheduler `boundary_plan` instead of only split booleans
- execution-program ownership
- inference-graph / replay ownership above program objects

Vision / DAv2-specific architecture that landed:

- `VisionBackboneProgram`
- `VisionDecoderProgram`
- block-local replay
- stack-level replay bundling for backbone blocks
- decoder replay bundling

LLM-specific architecture that landed:

- Qwen linear attention backend ownership
- gated-delta backend work and recurrent-buffer shader support

This is enough structure to keep improving the backend without stopping for a
full redesign.

## 4. Cleanup And Review Findings

The cleanup work was worthwhile. The main architectural review conclusion is
that the codebase needed targeted cleanup and ownership fixes, not a wholesale
rewrite.

Review and cleanup findings that mattered:

- packed-weight residency cache ownership mattered
  - the bad pattern was storing strong references to source tensors and trimming
    only against resident GPU bytes
  - the corrected direction is weak ownership of source tensors so the cache
    does not pin entire modules in memory
- replay bridges need stricter correctness gates than eager code
  - one example was the SDPA replay path, where float-only assumptions can leak
    into public bridges if dtype gating is too loose
- replay bundle keys must reflect the real execution identity
  - shared-root bundle caches are unsafe if they key only on a broad label and
    ignore context identity or relevant shape differences
- duplicated behavior forks hurt testability
  - the user feedback here was correct: too many runtime paths or feature flags
    make the backend harder to validate because not all combinations are tested

Related simplification rule that emerged:

- keep profiling and logging toggles
- remove or avoid long-lived behavior forks unless they are directly on the
  active optimization path

## 5. Model Zoo And Verification Snapshot

The model-zoo work was valuable because it exposed backend gaps that unit tests
did not make obvious.

Representative earlier zoo snapshot:

- LLM zoo Vulkan OK:
  - Qwen
  - Llama
  - Falcon
  - Ministral
  - PowerMoE
- LLM zoo Vulkan FAIL at that checkpoint:
  - Gemma4 E2B
  - Phi-3 mini
- depth zoo:
  - all `8/8` completed
- specialized zoo:
  - `0/7` Vulkan passes at that earlier checkpoint

Important gap categories surfaced by the zoos:

- missing operators such as `aten::isin.Tensor_Tensor_out`
- fill semantics on buffer-backed logical views
- storage-less tensor / data-pointer semantics
- dtype/layout consistency on diffusion and vision paths
- copy and materialization churn

Verification also improved materially over the thread:

- earlier checkpoint: `129/130` Vulkan tests passing
- later checkpoint: `179/179` Vulkan unittests passing
- later targeted regression coverage was added for BF16 buffer-cast cases

Interpretation:

- the backend is much more capable now
- the remaining problems shifted away from "missing whole ops" and toward
  execution ownership, layout churn, replay correctness, and end-to-end
  performance

## 6. Current Architecture Judgment

Second-opinion summary:

- no large refactor is required right now
- the major backend abstractions are already in place
- the current architecture is good enough to support the next round of
  performance work
- the biggest risk is not architectural absence; it is architectural leakage,
  where expensive paths still escape the owned program/replay region

Practical implication:

- keep improving the existing planner/program/replay stack
- do not stop for a broad rewrite unless a future change proves that the
  execution-program ownership model itself is the bottleneck

## 7. DAv2 And Vision Path Findings

### 7.1 Broad Direction

The main DAv2 direction became:

- reduce texture dependence
- execute the backbone and decoder on a reusable buffer-first program path
- reduce eager dispatch and intermediate materialization with replay/program
  ownership

The texture path was kept available, but the buffer-first path became the main
compatibility direction.

### 7.2 Buffer-Path Conversion Work

Large parts of the vision path were converted or refined for buffer execution:

- binary ops
- reshape / view / contiguity handling
- feature-map/token transforms
- softmax and matmul cleanup
- more buffer equivalents for operations that previously forced texture use

Observed benefit:

- better compatibility
- fewer shape/materialization breaks
- clearer generic path for models beyond DAv2

Observed downside:

- some steps regressed isolated microbenchmarks because they traded a narrow,
  fast texture-specialized path for a more generic buffer path

Decision taken during the work:

- keep the buffer-first direction anyway, because compatibility and ownership
  were the higher-order constraints

### 7.3 Replay / Program Work

Important work that landed:

- block-local backbone programs
- block replay
- stack replay bundles for multiple DINOv2 blocks
- decoder replay bundling
- execution labels and graph labels for vision phases

Important positive result:

- block-level replay probes showed that once a DINO block stays inside replay,
  eager churn around packing and materialization drops sharply

That result still matters. It says the core architecture is sound; the problem
is capture coverage and execution ownership, not the replay idea itself.

### 7.4 Important Replay Caveat

One late investigation changed the interpretation of the earlier fast DAv2
number.

What was discovered:

- the old faster path used nested attention replay under an outer vision
  replay/program path
- `Context::submit_prepared_command_buffer(...)` submits immediately and is not
  captured into the outer recording scope

Practical consequence:

- nested replay submission inside outer replay recording is architecturally
  unsafe
- the earlier `~0.148 s` replay-era DAv2 result is not a trustworthy target for
  the exact same implementation pattern
- the later hangs and replay corruption symptoms are consistent with that

Design lesson:

- nested phases must be first-class inside the owning program/replay
- do not treat public replay bridges as safe to call from within another owned
  replay unless the capture semantics are explicit and correct

## 8. CUDA, DirectML, And Other Backend Comparison Findings

The CUDA and DirectML comparisons were useful mainly as direction-setting
references.

High-level takeaways:

- DirectML on the RX 9070 was materially faster than Vulkan on DAv2
- CUDA on the GTX 1080 was also materially faster than Vulkan
- the remaining Vulkan gap is not mainly a missing SDPA path anymore

The slower areas were broader:

- linear / matmul
- layernorm
- elementwise chains
- shape/view/materialization churn
- general eager runtime overhead around the backbone and decoder/head

The useful cross-backend idea is not "copy one kernel." It is:

- fixed-shape execution ownership
- persistent packed layouts
- less intermediate memory traffic
- fewer boundaries between model phases

That lines up with what DirectML and systems like `llama.cpp` generally do
well: they reduce planning overhead and memory churn around stable shapes.

## 9. What Helped, What Regressed, And What Was A Dead End

### Helped And Landed

- planner/runtime extraction into reusable execution objects and programs
- buffer-first generic backend work for vision models
- vision backbone and decoder ownership
- replay bundle work for stacked backbone blocks
- Qwen backend ownership of linear attention
- backend helpers for masks, slicing, rotary, and other model-support glue
- targeted cleanup that removed or reduced Python-side compatibility debt

### Helped But Only Incrementally

- per-op fusion in the decoder/head
- reducing intermediate writes in a few local spots
- direct program bridges that improved correctness but not peak throughput
- kernel-quality tweaks that helped single operators but did not move the full
  model enough

### Explicit Dead Ends Or Non-Final Paths

- keeping model behavior in Python when a backend helper could own it
- trusting the old nested-replay DAv2 fast number as a safe performance target
- broad runtime feature forks that fragment validation
- naive per-op padding/materialization inside a hot linear path

The pattern here is consistent:

- incremental local tuning helps, but it does not get DAv2 from `~0.14 s` to
  `~0.05 s`
- the next step needs a graph- or program-level ownership change

## 10. Cooperative-Matrix / BF16 Experiment

This was the main late performance experiment.

### 10.1 Groundwork That Landed

Several pieces of backend groundwork were worth keeping:

- cooperative-matrix capability discovery in the Vulkan runtime/planning stack
- required-subgroup-size pipeline support
- BF16 cooperative-matrix shader support for buffer linear
- float-to-BF16 buffer-cast plumbing
- targeted regression tests for the cast path

These changes are useful even though the first end-to-end DAv2 attempt was not
shippable.

### 10.2 Hardware Capability Result

The runtime capability probe on the RX 9070 showed the hardware is not the
blocker.

Relevant findings from the capability log:

- cooperative matrix support present
- shader BF16 support present
- subgroup size control present
- required subgroup size `32` supported
- cooperative matrix tile sizes `16x16x16` available

So the basic hardware prerequisites are there.

### 10.3 Isolated Smoke Result

The isolated op-hit smoke confirmed that the cooperative matrix linear path can
actually fire in a controlled case:

- `aten::linear.buffer_bfloat16_cooperative_matrix`

That matters because it rules out a purely mechanical wiring failure.

### 10.4 End-To-End DAv2 Result

The first end-to-end BF16 backbone experiment was not a win.

Observed benchmark snapshot for the experimental route:

- `single_image_forward_device_resident ~= 0.3179 s`
- `single_image_forward_with_readback ~= 0.2502 s`

The main reason is that DAv2's backbone token height is `2073`, so the packed
BF16 path kept falling back to plain `aten::linear.buffer_bfloat16` instead of
staying on the cooperative kernel. That destroyed the hoped-for benefit.

### 10.5 Accuracy Snapshot

The experiment was still useful because it showed the math path itself was not
obviously broken.

Same-repo CPU vs Vulkan comparison on `demo01` showed:

- raw MAE `0.00178176`
- normalized MAE `0.00033064`
- normalized correlation `0.999998756`

Interpretation:

- the BF16 backbone experiment was accurate enough to treat as a performance
  problem, not an immediate correctness failure

### 10.6 What Was Backed Out

The packed-context BF16 routing and the naive row-padding attempt were backed
out.

Reason:

- the row-padding idea was implemented too locally, inside the hot linear path
- it caused bad materialization behavior and still did not solve the real
  end-to-end alignment problem

What remained in the tree:

- cast plumbing
- BF16 linear helper support
- cooperative-matrix shader path
- targeted tests

What was restored:

- the packed linear replay path went back to the sound float buffer-native
  checkpoint

## 11. Environment Variable Policy

One clear simplification decision from the thread:

- do not keep many long-lived behavior switches

The preferred policy is:

- keep profiling and logging environment variables
- keep a feature gate only if it is directly on the active optimization path
- remove or collapse flags that create extra runtime paths without enough test
  coverage

This is less flexible, but it is better engineering. Multiple partially tested
backend routes make regression analysis much harder.

## 12. What Is Costly Right Now

The expensive part of DAv2 is not one isolated operator.

The main cost centers are:

- backbone linear / matmul work that still does not stay on the best available
  kernel family
- layernorm and other buffer math that still adds noticeable overhead
- decoder/head chains that still materialize intermediate results too often
- shape/view conversions and execution boundaries around otherwise stable shapes

Practical interpretation:

- more isolated micro-optimizations can still help
- but the big remaining cost is memory traffic and ownership boundaries

## 13. Current Safe Checkpoint

After backing out the non-shippable BF16 backbone route, the tree returned to
the last sound DAv2 checkpoint.

Useful benchmark artifact from that checkpoint:

- `single_image_forward_device_resident ~= 0.1492 s`
- `single_image_forward_with_readback ~= 0.1289 s`
- `full_corpus_end_to_end mean ~= 0.2402 s`

Current state summary:

- the backend is structurally much better than at the start of the work
- the codebase does not need a broad refactor
- the remaining DAv2 gap is still large relative to the `~0.05 s` goal
- the next win needs to come from better execution ownership, not just more
  local kernel polish

## 14. Recommended Next Steps

If this work resumes from the current checkpoint, the highest-value next steps
are:

1. Move fixed-shape alignment into the execution-program / replay level for the
   vision backbone.
2. Plan padded or aligned token buffers once, so `qkv`, `proj`, `fc1`, and
   `fc2` can stay on the cooperative-matrix path instead of falling back per op.
3. Make nested attention a first-class owned phase inside the backbone program
   rather than routing it through a public replay bridge.
4. Propagate packed layouts across more layer boundaries so fewer ops need
   repack/materialize steps.
5. Keep reducing intermediate writes in the decoder/head and only add fusion
   when it genuinely removes memory traffic.
6. Validate every architectural speedup with both a benchmark and a CPU accuracy
   check.

This is the key design conclusion from the late work:

- the next meaningful optimization is fixed-shape graph/program ownership, not
  another narrow local tweak

## 15. Key Artifacts And Scripts

Important scripts:

- `scripts/benchmarks/benchmark_llm_model_zoo.py`
- `scripts/benchmarks/benchmark_depth_model_zoo.py`
- `scripts/benchmarks/benchmark_specialized_model_zoo_run.py`
- `scripts/benchmarks/benchmark_depth_anything.py`

Important artifacts and notes:

- `comparison/depth_anything_v2_performance_notes_20260416.md`
- `comparison/benchmark_depth_anything_vulkan_bf16_backbone_probe_r1_20260418.json`
- `comparison/benchmark_depth_anything_vulkan_post_bf16_experiment_revert_r1_20260418.json`
- `comparison/coop_bf16_depth_demo01_same_repo_compare_20260418/summary.json`
- `comparison/cooperative_matrix_runtime_caps_20260418.log`
- `comparison/bfloat16_linear_coop_op_hits_20260418.log`

## 16. Short Version

If only six things are carried forward from this thread, they should be these:

1. The Vulkan backend moved from ad hoc fixes toward real
   planner/program/replay architecture.
2. The codebase does not need a large refactor right now; targeted ownership and
   cleanup work is the correct scale.
3. The buffer-first path should remain the main compatibility direction for
   vision work.
4. The old very fast DAv2 replay number is not a safe target to chase because
   nested replay semantics were wrong there.
5. The cooperative-matrix / BF16 experiment proved the hardware path is viable,
   but per-op fallback and alignment issues made the first end-to-end route a
   regression.
6. The next real DAv2 optimization is fixed-shape execution-program/replay
   planning with aligned buffers and fewer intermediate writes.
