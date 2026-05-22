# Cross-Model Benchmark Suite

This directory now has a small shared benchmark harness for smoke/profiling
coverage outside the Depth Anything V2 owner path.

## Entry Points

- `benchmark_depth_anything.py`: keep as-is. This remains the canonical DAv2
  Vulkan stack-owner benchmark and exposes the richest Vulkan diagnostics.
- `benchmark_model_suite.py`: shared cross-model smoke/profiling entry point for
  Lotus, HY-MT, PaddleOCR, and Gemma.
- `benchmark_suite_common.py`: shared JSON schema, device probes, timing helpers,
  and Vulkan counter snapshots.

The suite is intentionally conservative. Unsupported backend/model combinations
produce structured skip/failure rows instead of silently falling back to another
route.

## JSON Schema

`benchmark_model_suite.py` emits:

- `schema_version`
- `accelerator_probe`
- `records[]`

Each record includes:

- task and exact model id
- backend, device index, and device metadata
- dtype/precision
- input metadata
- warmup/repeat counts
- setup and device-resident or pipeline timing summaries
- Vulkan debug counters when the backend is Vulkan
- output sanity metrics or structured skip/failure reason

## Hardware Policy

- RX 9070: primary Vulkan benchmark and optimization signal.
- RX 6700 XT: secondary Vulkan/DirectML compatibility target.
- GTX 1080: CUDA comparison and compatibility/floor target where Vulkan is
  available.

The suite records adapter/device lists through PyTorch, torch-directml when
installed, CUDA, and `vulkaninfo` when present.

## Model Coverage

- Lotus: `jingheya/lotus-depth-d-v1-1`, via Diffusers when installed. Vulkan is
  currently reported as a structured skip because Diffusers pipelines do not map
  to PyTorch Vulkan tensors.
- HY-MT: `tencent/HY-MT1.5-1.8B`, via Transformers causal generation. Vulkan is
  a structured skip until Transformers model execution can target PyTorch Vulkan.
- PaddleOCR 3.5 Transformers: pipeline smoke on CPU when PaddleOCR is installed.
  Non-CPU torch backends are structured skips until the PaddleOCR backend is
  explicitly mapped to the requested torch backend.
- Gemma: `google/gemma-4-E2B-it`, with `google/gemma-4-E2B` usable through
  `--gemma-model-id` when the instruction-tuned model is inaccessible.

## Cleanup Inventory

- Keep as-is:
  - `benchmark_depth_anything.py`
  - `compare_depth_anything_desktop_outputs.py`
  - `compare_depth_anything_desktop_profiles.py`
  - `compare_vulkan_value_trace.py`
- Fold later into shared harness:
  - `benchmark_depth_anything_desktop_forward.py`
  - `smoke_depth_extractor_models.py`
- Shared helpers:
  - `bench_common.py`
  - `depth_anything_common.py`
  - `benchmark_suite_common.py`
- Obsolete/remove:
  - none without review. No benchmark script was removed in this pass.

## Recommended Use

Probe only:

```powershell
python scripts\benchmarks\benchmark_model_suite.py --probe-only --out agent_space\probe.json
```

CPU smoke rows:

```powershell
python scripts\benchmarks\benchmark_model_suite.py --backends cpu --warmup 0 --repeats 1 --out agent_space\model_suite_cpu.json
```

RX 9070 Vulkan compatibility rows:

```powershell
python scripts\benchmarks\benchmark_model_suite.py --backends vulkan --warmup 0 --repeats 1 --out agent_space\model_suite_vulkan.json
```

Run DAv2 separately after any harness changes because it remains the primary
Vulkan stack-owner performance target.
