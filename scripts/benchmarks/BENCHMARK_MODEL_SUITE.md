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

Large model downloads are disabled by default. Local-cache-only runs use
`agent_space/hf_home` unless `--cache-dir` is provided. A missing local cache,
gated model, unavailable package, or unsupported backend is reported as a
structured `status=skip` row.

PaddleX/PaddleOCR model cache is forced to `agent_space/paddlex_cache` through
`PADDLE_PDX_CACHE_HOME` in the harness. Do not use `--allow-downloads` for
PaddleOCR unless downloading model files into that repo-local cache is intended.

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
- dependency versions and cache paths in `accelerator_probe`

The suite also includes `torch_ops`, a small local PyTorch `conv2d + relu`
backend smoke. It runs a CPU reference and reports CPU-vs-device numerical error
for Vulkan-capable venvs. This task is intentionally independent of external
model packages so each benchmark venv can verify that it imports the local
PyTorch build and can see the Vulkan devices before model-specific code runs.

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

## Cache and Dependency Workflow

Use separate benchmark virtual environments instead of trying to make one Python
install satisfy every model stack. The environments live under
`agent_space/venvs` and are not committed.

Create and probe the empty environments:

```powershell
python scripts\benchmarks\prepare_model_suite_envs.py --env all --create --probe --out agent_space\model_suite_env_probe.json
```

Install non-torch dependencies into one environment only when that task is being
enabled. The installer streams pip output as it runs and does not upgrade pip
unless `--upgrade-pip` is explicitly provided:

```powershell
python scripts\benchmarks\prepare_model_suite_envs.py --env paddleocr --install --probe --out agent_space\model_suite_env_probe_paddleocr.json
python scripts\benchmarks\prepare_model_suite_envs.py --env diffusers --install --probe --out agent_space\model_suite_env_probe_diffusers.json
python scripts\benchmarks\prepare_model_suite_envs.py --env transformers --install --probe --out agent_space\model_suite_env_probe_transformers.json
```

Then run the suite with the matching interpreter:

```powershell
agent_space\venvs\paddleocr\Scripts\python.exe scripts\benchmarks\benchmark_model_suite.py --tasks paddleocr --backends cpu --warmup 0 --repeats 1 --out agent_space\model_suite_paddleocr_cpu.json
agent_space\venvs\diffusers\Scripts\python.exe scripts\benchmarks\benchmark_model_suite.py --tasks lotus --backends cpu --warmup 0 --repeats 1 --out agent_space\model_suite_lotus_cpu.json
agent_space\venvs\transformers\Scripts\python.exe scripts\benchmarks\benchmark_model_suite.py --tasks hy_mt gemma --backends cpu --warmup 0 --repeats 1 --max-new-tokens 4 --out agent_space\model_suite_text_cpu.json
```

Verify PyTorch CPU/Vulkan coverage from each venv:

```powershell
agent_space\venvs\paddleocr\Scripts\python.exe scripts\benchmarks\benchmark_model_suite.py --tasks torch_ops --backends cpu vulkan --warmup 1 --repeats 3 --out agent_space\model_suite_torch_ops_paddleocr_venv.json
agent_space\venvs\diffusers\Scripts\python.exe scripts\benchmarks\benchmark_model_suite.py --tasks torch_ops --backends cpu vulkan --warmup 1 --repeats 3 --out agent_space\model_suite_torch_ops_diffusers_venv.json
agent_space\venvs\transformers\Scripts\python.exe scripts\benchmarks\benchmark_model_suite.py --tasks torch_ops --backends cpu vulkan --warmup 1 --repeats 3 --out agent_space\model_suite_torch_ops_transformers_venv.json
```

The base task venvs intentionally do not install PyTorch by default. They import
the source-tree build so Vulkan probing remains tied to the local backend under
test. For CPU framework coverage, use the installed-wheel CPU venv variants:

```powershell
python scripts\benchmarks\prepare_model_suite_envs.py --env diffusers_cpu --create --install --probe --out agent_space\model_suite_env_probe_diffusers_cpu.json
python scripts\benchmarks\prepare_model_suite_envs.py --env transformers_cpu --create --install --probe --out agent_space\model_suite_env_probe_transformers_cpu.json
python scripts\benchmarks\prepare_model_suite_envs.py --env paddleocr_cpu --create --install --probe --out agent_space\model_suite_env_probe_paddleocr_cpu.json
```

Those CPU venvs install a wheel-provided `torch` package for model framework
compatibility. They are not used for Vulkan backend performance decisions; the
source-tree interpreter remains the control path for existing DAv2 Vulkan runs
and the base venv `torch_ops` Vulkan smoke. Run CPU framework rows with
`--torch-import-mode installed`:

```powershell
agent_space\venvs\transformers_cpu\Scripts\python.exe scripts\benchmarks\benchmark_model_suite.py --torch-import-mode installed --tasks hy_mt gemma --backends cpu --warmup 0 --repeats 1 --max-new-tokens 4 --out agent_space\model_suite_text_cpu_installed_torch.json
agent_space\venvs\diffusers_cpu\Scripts\python.exe scripts\benchmarks\benchmark_model_suite.py --torch-import-mode installed --tasks lotus --backends cpu --warmup 0 --repeats 1 --out agent_space\model_suite_lotus_cpu_installed_torch.json
agent_space\venvs\paddleocr_cpu\Scripts\python.exe scripts\benchmarks\benchmark_model_suite.py --torch-import-mode installed --tasks paddleocr --backends cpu --warmup 0 --repeats 1 --out agent_space\model_suite_paddleocr_cpu_installed_torch.json
```

After download-prep, HY-MT and Gemma require the installed CPU text venv with
`transformers>=5.8.0`, `protobuf`, and `hf_xet`. Lotus model files can be cached
under `agent_space/hf_home`, but current Diffusers releases still do not expose
the model's `LotusDPipeline` class; this is reported as
`lotus_pipeline_class_unavailable_in_diffusers` until the suite gains a real
Lotus custom-pipeline adapter.

## Vulkan Model Mapping Status

The suite now attempts model-framework Vulkan mapping instead of returning early
for Transformers and PaddleOCR rows. HY-MT and Gemma load through the source-tree
PyTorch interpreter so the requested device is `torch.device("vulkan")`; the
source-tree build has Python `torch.distributed` stubs but no
`torch._C._distributed_c10d` extension (`USE_GLOO=OFF`, `USE_MPI=OFF`,
`USE_NCCL=OFF`). Transformers 5.9 imports continuous-batching generation helpers
that require that distributed extension even for single-process generation. The
harness installs a benchmark-local import shim for those continuous-batching
modules only. It does not implement real collectives and raises if continuous
batching is actually requested. Each JSON row records
`distributed_c10d_status` as `real_distributed_c10d`,
`distributed_import_shim`, or `missing_distributed_c10d`.

With that shim, HY-MT reaches the first Vulkan operator blocker:
`aten::isin.Tensor_Tensor_out` during generation special-token preparation.
Gemma reaches model construction and then fails while moving the large embedding
weights to Vulkan with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.

PaddleOCR initializes through PaddleX's accepted CPU control path, then the
harness patches PaddleX's Transformers predictor device hook so loaded
Transformers-engine modules are moved to `vulkan`. The source PaddleOCR venv
installs `torchvision` with `--no-deps` so it does not replace the source-tree
PyTorch build. The current first PaddleOCR Vulkan backend blocker is
`aten::convolution` routed to `KnownBadLargePointwiseConv` for
`input=[1, 512, 7, 7]` and `weight=[512, 512, 1, 1]` on RX 9070.

Installed-wheel CPU venvs are still the executable model-framework coverage
path. Source-tree venvs remain the Vulkan backend coverage path through
`torch_ops`, DAv2, and the model-mapping rows above. The next step for model
coverage is targeted Vulkan op support or model-side benchmark adaptation for
the listed blockers, not command-buffer replay or broad allocator work.

Local-cache-only smoke, no downloads:

```powershell
python scripts\benchmarks\benchmark_model_suite.py --backends cpu --warmup 0 --repeats 1 --out agent_space\model_suite_cpu.json
```

Use an explicit repo-local cache:

```powershell
python scripts\benchmarks\benchmark_model_suite.py --backends cpu --cache-dir agent_space\hf_home --warmup 0 --repeats 1 --out agent_space\model_suite_cpu.json
```

Download-prep runs must be explicit and should be reviewed before use because
the models are large:

```powershell
python scripts\benchmarks\benchmark_model_suite.py --backends cpu --tasks hy_mt gemma --allow-downloads --cache-dir agent_space\hf_home --warmup 0 --repeats 1 --max-new-tokens 4 --out agent_space\model_suite_download_prep.json
```

Debug tracebacks are omitted by default to keep JSON compact. Add
`--debug-traceback` only when diagnosing a harness or dependency failure.

## Current Environment Notes

The current source-tree PyTorch imports successfully, but Diffusers and
Transformers package checks expect installed wheel metadata and distributed
extension metadata. The harness applies a local Diffusers availability patch for
Lotus runs after importing the source-tree PyTorch. On this environment Lotus,
HY-MT, and Gemma still report installed-torch metadata blockers because
Diffusers/Transformers import paths require `torch._C._distributed_c10d`. That
is a benchmark environment blocker, not a Vulkan backend result. Use an
installed local PyTorch wheel or a compatible benchmark virtual environment
before treating those model rows as backend coverage.

PaddleOCR 3.5 uses the task-specific `agent_space/venvs/paddleocr` environment.
The PaddleOCR Transformers backend also requires the `transformers` package in
that same venv. PaddleOCR's current Transformers-engine documentation requires
`transformers>=5.8.0`; older 4.x Transformers builds do not recognize PaddleX
PP-LCNet model metadata such as `PPLCNetImageProcessor` / `pp_lcnet`. Because
the venv imports the source-tree PyTorch, it also needs Python-side PyTorch
dependencies such as `sympy` when Transformers imports PyTorch modeling helpers.
With downloads disabled and no repo-local PaddleX model cache, PaddleOCR reports
`paddleocr_model_cache_unavailable_downloads_disabled`. The cache check requires
the known PaddleOCR pipeline model directories under
`agent_space/paddlex_cache/official_models`, not just a partial PaddleX cache.
Once the cache exists, missing backend dependencies report
`paddleocr_transformers_dependency_missing`; installed-torch metadata blockers
report `paddleocr_transformers_requires_installed_torch_metadata`; missing
TorchVision support in the installed CPU env reports
`paddleocr_transformers_missing_torchvision`; missing
source-tree PyTorch Python dependencies report
`paddleocr_transformers_source_tree_torch_dependency_missing`. If Transformers
imports PyTorch distributed modeling helpers that are not available in the local
source-tree build, the row reports
`paddleocr_transformers_source_tree_torch_distributed_missing`. If the cached
PaddleX model declares a custom image processor that is not registered in the
installed PaddleOCR/PaddleX/Transformers packages, the row reports
`paddleocr_transformers_model_processor_unregistered`.

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
