# Vulkan Backend Findings

Date: 2026-04-18

This note summarizes the main findings from the long Vulkan backend work in
this workspace. The goal was to use model zoos to expose backend gaps, move
fixes into the C++/backend instead of Python glue, and push the backend toward
real end-to-end model execution rather than fragile fallback paths.

## 1. Project Goal

Primary direction:

- Use real model zoos to expose backend gaps.
- Fix gaps in C++/backend code instead of piling up Python compatibility shims.
- Move the Vulkan backend toward reusable planning/runtime architecture rather
  than model-specific patches.
- Prefer a buffer-first execution path for compatibility, while keeping the old
  texture path available as an option rather than deleting it.

## 2. Build And Environment Findings

Known working build setup:

- Build root: `pytorch/build`
- Generator: `Visual Studio 17 2022`
- Vulkan SDK: local SDK at `VulkanSDK/1.4.341.1`
- Do not force `BLAS=MKL`
- Sanitize the environment so only one effective `PATH` reaches CMake/MSBuild
- Build `torch_cpu` with `/m:1`

Important build failure modes:

- The old incremental `pytorch/build` could become poisoned and needed a clean
  regenerate.
- `/m:8` could fail on generated project references such as
  `ZERO_CHECK -> GetNativeManifest`.
- Duplicate `Path` / `PATH` environment entries could poison MSBuild.
- After rebuilding, the active Python checkout used
  `pytorch/torch/lib/torch_cpu.dll` and `pytorch/torch/lib/c10.dll`, so staged
  binaries had to be copied from `pytorch/build/bin/Release/`.

Practical rule:

- A successful build is not enough. Verify the DLLs actually loaded by the local
  Python package and restage them if needed.

## 3. Major Architecture That Landed

The planner/runtime was pulled into a real Vulkan planning stack under:

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
- scheduler `boundary_plan` instead of only boolean split hints
- execution-program layer
- inference-graph / replay layer above program objects

Vision/DAv2-specific architecture that was built on top:

- `VisionBackboneProgram`
- `VisionDecoderProgram`
- block-local and stack-level replay support
- replay-bundle support for chaining multiple backbone blocks
- decoder replay bundling

LLM-specific architecture that landed:

- Qwen linear attention backend path in
  - `QwenLinearAttention.cpp`
  - `QwenLinearAttention.h`
- gated delta backend work in
  - `GatedDelta.cpp`
  - `gated_delta_recurrent_buffer.glsl`

## 4. Backend Areas Touched Most Often

Hot backend areas from this work:

- `Mm.cpp`
- `Softmax.cpp`
- `Indexing.cpp`
- `Copy.cpp`
- `Convolution.cpp`
- `Packing.cpp`
- `Register.cpp`
- `VisionBlocks.cpp`

The repeated pattern was:

- expose a gap through a real model
- add a generic backend op or helper
- stop carrying the behavior in Python if possible

## 5. Model Zoo And Verification Snapshot

Early model-zoo snapshot from the rebuild handoff:

- LLM zoo Vulkan OK:
  - Qwen
  - Llama
  - Falcon
  - Ministral
  - PowerMoE
- LLM zoo Vulkan FAIL:
  - Gemma4 E2B
  - Phi-3 mini
- Depth zoo:
  - all 8/8 completed
  - some still used fallback-backed modes
- Specialized zoo:
  - 0/7 Vulkan passes at that point

Major cross-model backend gaps identified through the zoos:

- missing `aten::isin.Tensor_Tensor_out`
- missing `fill_.Scalar` on buffer-backed logical views
- storage-less tensor / data-pointer semantics
- diffusion UNet dtype/layout consistency
- copy/materialization churn

Python glue that was explicitly identified as debt:

- `qwen_vulkan_compat.py`
- `smoke_gemma4_text.py`
- model-side compat in Phi and PowerMoE

Verification state evolved over the thread:

- Earlier snapshot: `129/130` Vulkan tests passing, one remaining embedding test
  failure.
- Later replay-fix checkpoint: full Vulkan unittest pass at `179/179`.

Interpretation:

- The backend got materially more capable over the course of the work.
- The remaining issues shifted from "missing whole operators" toward replay,
  execution ownership, layout churn, and model-zoo integration.

## 6. DAv2 And Vision Path Findings

### 6.1 Broad Direction

The main DAv2 direction became:

- reduce texture dependence
- make the backbone and decoder execute on a reusable buffer-first program path
- use replay and program ownership to reduce eager dispatch/materialization

The texture path was kept available, but the main development direction became
buffer-first execution.

### 6.2 Buffer-Path Conversion Work

Large parts of the vision path were converted or refined for buffer execution:

- binary ops
- reshapes/views/contiguity semantics
- feature-map/token transforms
- softmax and mm path cleanup
- more buffer equivalents for operations that previously forced texture use

Observed benefit:

- better compatibility
- fewer shape/materialization path breaks
- clearer generic path for models beyond DAv2

Observed downside:

- some changes regressed isolated operator performance, especially when they
  traded a fast texture-specialized path for a more generic buffer path

Decision taken during the thread:

- keep the buffer-first direction even when an intermediate step regressed a
  microbenchmark, because compatibility and architectural consistency were the
  short-term priority

### 6.3 Vision Replay / Program Work

Important work that landed:

- block-local backbone programs
- block replay
- stack replay bundle for multiple DINOv2 blocks
- decoder replay bundle work
- execution labels and graph labels for vision phases

Important positive result:

- block-level replay probes showed that once a single DINO block is truly
  running through replay, the eager churn around linear pack/materialize drops
  sharply

This supported the broader architecture direction:

- the replay/program idea itself is valid
- the remaining gap is how much of the full model actually stays inside the
  captured region and how cleanly nested phases compose

### 6.4 Best Historical DAv2 Number

One notable benchmark artifact achieved:

- `single_image_forward_only ~= 0.1483 s`
- `single_image_end_to_end ~= 0.1820 s`

Artifact:

- `temp/depth_anything_v2_vits_vulkan_skip_output_after_attention_runtime.json`

At the time, this looked like a major improvement and was treated as a good
directional result.

### 6.5 Important Later Reinterpretation

Later investigation showed a serious caveat:

- the old faster replay-era DAv2 path used nested attention replay under an
  outer vision replay/program path
- `Context::submit_prepared_command_buffer(...)` submits immediately and is not
  captured into the outer external recording scope

Practical consequence:

- nested replay submit inside outer replay recording is architecturally unsafe
- the old `~0.148 s` result is therefore not a trustworthy target for the exact
  same implementation pattern
- this lines up with later hangs and replay corruption symptoms

This is the most important design-level lesson from the late DAv2 work:

- do not rely on nested replay submission from inside another replay recording
- make nested phases first-class inside the owning program/replay instead

## 7. April 16-18 DAv2 Replay Investigation

This was the main late-stage investigation.

### 7.1 What Was Observed

Symptoms:

- replay changes appeared to hurt DAv2 performance badly
- some benchmark runs looked stuck for a long time
- eventually a real hang was localized inside the DINOv2 bundled backbone path

Localization:

- the hang was inside
  `torch.ops.vulkan_prepack.run_vision_backbone_stack_replay_bundle_bridge(...)`
- more specifically, later probing showed it was stalling in the nested
  attention-runtime path under the bundled backbone execution

### 7.2 What Was Tried

Tried:

- route long-sequence vision attention to `AttentionRuntime`
- use the public SDPA Vulkan entry from the vision attention path
- observe op-hit logs and benchmark output

What happened:

- the path was active
- DAv2 could show a very fast number
- but later the same structure produced hangs

### 7.3 Direct Program Bridge Experiment

A direct non-replay attention-runtime bridge was then tried.

Idea:

- avoid nested replay by running the attention runtime program directly

Result:

- the hang disappeared
- but steady-state DAv2 performance dropped to roughly:
  - `single_image_forward_only ~= 0.337 s`
  - `single_image_end_to_end ~= 0.367 s`

The direct bridge was therefore:

- better for correctness/stability than nested replay
- not enough to recover the earlier fast benchmark result

### 7.4 Current Interpretation

Most likely explanation:

- the old faster path was benefiting from an invalid nesting pattern
- the newer corrected path is slower because it is no longer relying on nested
  replay submission behavior

Current useful conclusion:

- the replay architecture is still worth keeping
- but attention must become a first-class owned phase inside the vision
  backbone program/replay rather than being routed through the public SDPA
  replay bridge from within that outer program

## 8. CUDA / DirectML Comparison Findings

Cross-backend comparison work was used mainly as a reference point.

Key observations:

- DirectML on the RX 9070 was materially faster than Vulkan on DAv2
- CUDA on the GTX 1080 was also materially faster than Vulkan
- local comparison note:
  - `comparison/depth_anything_v2_performance_notes_20260416.md`

Important interpretation from that note:

- the remaining Vulkan gap is not primarily SDPA anymore
- the bigger gaps are around:
  - layernorm
  - linear/matmul
  - elementwise chains
  - shape/view/materialization churn
  - general eager runtime overhead around the backbone

Practical takeaway:

- closing the remaining gap is more about execution ownership, capture coverage,
  and memory/reuse policy than about a single missing attention kernel

## 9. Environment And Install Findings

Transformers / Gemma findings:

- the normal `.venv` had a `transformers 5.6.0.dev0` install that could import
  Gemma4 code, but the import failed because a user-roaming `torchvision`
  package tried to register `torchvision::nms` against the local custom PyTorch
  build where that operator did not exist
- practical symptom:
  - `RuntimeError: operator torchvision::nms does not exist`

DirectML environment finding:

- the `.venv-directml` LLM zoo environment was broken because that transformers
  install disabled usable PyTorch behavior with `torch 2.3.1`

This matters because some model-zoo failures were environment failures rather
than Vulkan backend failures.

## 10. What Was Tried So Far

### Helped And Landed

- Extracting planner/runtime into reusable execution objects, programs, and
  inference graphs
- Buffer-first generic op work for vision models
- Vision backbone and decoder program ownership
- Replay-bundle work for stacked backbone blocks and decoder fusion
- Qwen backend ownership of linear attention
- Backend helpers for masks, slicing, rotary, and other model-support glue

### Helped But With Caveats

- Attention-runtime integration for vision
  - useful directionally
  - later revealed unsafe nested replay interaction
- Stack-level replay for DAv2
  - correct architectural direction
  - still needs better ownership of nested attention phases

### Explicit Dead Ends Or Non-Final Paths

- Keeping model behavior in Python when a backend helper could own it
- Treating the old `~0.148 s` DAv2 replay number as a safe baseline without
  checking replay nesting semantics
- Global direct-program attention bridge as a universal replacement
  - it avoided hangs
  - but did not by itself recover good DAv2 performance

## 11. Current State

Best concise state summary as of this note:

- The Vulkan backend is much more capable and structurally cleaner than at the
  start of the work.
- The planner/program/replay architecture is real and useful.
- The buffer-first direction is now the main compatibility path.
- Qwen was a major success.
- DAv2 remains the main performance case study.
- The late DAv2 investigation found that nested replay under outer replay is
  unsafe and likely explains both hangs and misleadingly fast earlier numbers.
- The currently corrected DAv2 path runs and benchmarks cleanly, but it is
  slower than the old suspect replay-era number.

## 12. Recommended Next Steps

If resuming this work, the highest-value next steps are:

1. Make attention a first-class owned phase inside `VisionBackboneProgram` /
   replay rather than routing backbone attention through the public SDPA replay
   bridge.
2. Keep reducing uncaptured work around backbone entry/exit and intermediate
   feature extraction.
3. Continue shrinking Python glue where the backend can own the behavior.
4. Re-run the relevant model-zoo entries after each architectural change,
   especially DAv2 and at least one LLM sanity case.
5. Treat any benchmark improvement as suspect until it has both:
   - a stable non-hanging replay structure
   - an accuracy check versus CPU

## 13. Key Artifacts And Scripts

Important scripts:

- `scripts/benchmarks/benchmark_llm_model_zoo.py`
- `scripts/benchmarks/benchmark_depth_model_zoo.py`
- `scripts/benchmarks/benchmark_specialized_model_zoo_run.py`
- `scripts/benchmarks/benchmark_depth_anything.py`

Important artifacts and notes mentioned repeatedly:

- `summary.json`
- `depth_model_zoo_vulkan_20260409_after_rebuild.json`
- `specialized_model_zoo_20260409_after_rebuild.json`
- `temp/depth_anything_v2_vits_vulkan_skip_output_after_attention_runtime.json`
- `comparison/depth_anything_v2_performance_notes_20260416.md`

## 14. Short Version

If you only remember five things from this thread, remember these:

1. The backend moved from ad hoc fixes toward real program/replay/planner
   architecture.
2. The model zoo work was valuable because it exposed generic backend gaps that
   were not obvious from unit tests alone.
3. The buffer-first path should remain the main compatibility direction.
4. The old very fast DAv2 replay number is not a safe target to blindly chase,
   because nested replay semantics were wrong there.
5. The next real DAv2 performance step is first-class attention ownership inside
   the backbone replay/program, not more layering of replay inside replay.
