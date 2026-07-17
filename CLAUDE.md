# Repository Operating Contract

This is a private PyTorch fork whose active product work is the Windows Vulkan
backend used by DeepDesktop. Preserve upstream PyTorch conventions, but make
decisions from this fork's graph-first Vulkan architecture, supported runtime,
and checked evidence.

## Work From Current Evidence

- Inspect the worktree, loaded runtime, CMake cache, and relevant artifacts
  before relying on conversation memory or historical documentation.
- `docs/vulkan/CURRENT_STATE.md` contains both current decisions and historical
  evidence. A historical blocker is not a current blocker until it reproduces
  with the loaded binaries.
- Distinguish these claims explicitly: imports, runs end to end, runs without
  fallback/readback, passes graph parity, meets the performance gate, and is
  production-ready. Evidence for one does not prove the next.
- When reporting a model blocked, record the command, current commit, loaded
  DLL hashes, configuration, and current error. Never repeat an old blocker
  merely because it remains in a document.

## Workspace And Tools

- Preserve user changes in a dirty worktree. Do not reset, discard, or rewrite
  unrelated files.
- Use `agent_space/` for logs, generated evidence, temporary scripts, model
  downloads, and throwaway experiments. It is git-ignored; never commit it.
- Search with `rg`/`rg --files`. Do not recursively scan venvs, model caches, or
  large downloaded corpora unless the task specifically requires it.
- Prefer the repository `.venv`. If a tool or package is missing, inspect the
  repository root, parent, and purpose-specific venvs under `agent_space/`
  before asking for an environment. Do not silently install packages or switch
  the build to a different Python ABI.
- Do not run more than one heavy build, benchmark, or model process at a time on
  this Windows machine. Inspect existing `cmake`, MSBuild, compiler, and linker
  processes before building; stop only duplicates confirmed to belong to the
  current task.

## Vulkan Source Of Truth

Before changing Vulkan production code, read:

- `docs/vulkan/PROJECT_CHARTER.md`
- `docs/vulkan/CURRENT_STATE.md`
- `docs/vulkan/ROADMAP.md`
- `docs/vulkan/GRAPH_RUNTIME.md`
- `docs/vulkan/REVIEW_CHECKLIST.md`
- `docs/vulkan/TEMPORARY_EXCEPTIONS.md`
- `docs/vulkan/CLEANUP_POLICY.md`
- `docs/vulkan/cleanup_ledger.json`

Read the sections relevant to the change again after long-running work or a
context handoff. Update the current-state documents when evidence or a blocker
changes; do not leave a superseded blocker presented as current.

## Architecture

- The performance architecture is graph-first: CPU `torch.export`, semantic
  contract lowering, immutable `VulkanGraphProgram` plans, program-owned
  resources, and eventually recorded command partitions.
- Eager Vulkan is the simple correctness substrate. Do not add speculative
  deferred placeholders, replay bridges, per-consumer materialization
  protocols, or model orchestration to make eager faster.
- DAv2, Lotus, HY-MT, PaddleOCR, and Gemma E2B are a coverage corpus. Model
  names may appear in harnesses, tests, evidence, and docs, but not in generic
  production dispatch.
- Prefer `KernelFamilyContract`, `RegionContract`, and
  `LayoutTransitionContract`. Admit legal runtime shapes semantically. Exact
  rows are evidence or temporary bounded policy, not the default answer to a
  new shape.
- Do not hide CPU fallback/readback, fake storage or `data_ptr`, or introduce
  permanent flags for incomplete behavior.
- Legacy stack, replay, compiled-session, and eager inference-graph code may be
  changed only for correctness, a migration hook, or an evidence-gated
  deletion. Do not expand those systems.
- Runtime-generated shader work belongs to graph codegen. Production execution
  must not shell out to a manually configured shader compiler.

## Windows Build Contract

The existing Visual Studio build tree is an incremental starting point, not an
authority. Never infer runtime capability solely from `build/CMakeCache.txt`.

Before building Vulkan work, inspect the cache and require:

```text
BUILD_PYTHON=ON
USE_VULKAN=ON
USE_DISTRIBUTED=ON
USE_GLOO=ON
USE_C10D_GLOO=ON
USE_LIBUV=ON
```

Require MPI, NCCL, and TensorPipe to remain disabled when those options are
present in the Windows configuration. Also verify that `Python_EXECUTABLE` is
the intended repository venv and that the configured `libuv_ROOT` exists. C10
is part of the core build; do not treat it as an optional component.

- Prefer the existing Visual Studio 17 2022 generator. Ninja and editable pip
  builds are not reliable substitutes on this machine.
- If a required option, dependency root, Python path, or generated source is
  missing or stale, reconfigure the same Visual Studio tree. Use
  `scripts/deepdesktop/windows/configure-vulkan-msvc.ps1` as the canonical
  local configuration and change CMake/configuration code when the product
  contract requires it. Do not preserve a bad cache to avoid reconfiguration.
- Build with one MSBuild worker unless the user explicitly requests otherwise:

```powershell
cmake --build build --config Release --target <target> -- /m:1 /nodeReuse:false
```

- Build `torch_cpu` for ordinary Vulkan C++ implementation changes.
- Build `torch_python` whenever Python bindings, `torch/csrc`, build options,
  generated bindings, distributed support, or configuration changed. A
  `torch_cpu`-only build cannot prove that a Python/DTensor binding exists.
- After a successful build, deploy the matching Release DLLs into `torch/lib`
  and compare SHA-256 hashes. Deploy `torch_cpu.dll` for backend changes and
  `torch_python.dll` whenever that target was rebuilt. Do not run source-tree
  tests against older deployed DLLs.
- Validate the imported runtime, not just files on disk:

```powershell
python -c "import torch; print(torch.__file__); print(torch._C.__file__)"
python -c "import torch; assert torch._C._has_vulkan"
```

For distributed/DTensor model coverage, also require:

```powershell
python -c "import torch; assert hasattr(torch._C, '_distributed_c10d'); assert hasattr(torch._C, '_DTensor_OpSchema_post_init')"
python scripts/benchmarks/benchmark_model_suite.py --validate-lotus-dtensor-preflight
```

If the cache flags pass but the runtime checks fail, the build or deployment is
stale. Rebuild/deploy `torch_python`; do not add benchmark shims for compiled
APIs and do not label Lotus blocked on DTensor until the preflight fails in the
current runtime.

## Testing And Evidence

Use PyTorch's test utilities:

```python
from torch.testing._internal.common_utils import run_tests, TestCase
```

- Use `assertEqual` for tensor equality, `@parametrize` for input families, and
  `instantiate_device_type_tests` for on-device numerical implementations when
  feasible.
- Dynamic semantic families need randomized legal-shape parity with a printed
  reproduction seed plus real dtype/layout/semantic negatives.
- Preserve behavioral regression tests for bug classes even when the mechanism
  that originally caused the bug is deleted.
- Match verification to risk. Documentation-only changes do not require a
  model matrix. Production C++ changes require a successful target build,
  matching deployed hash, and focused runtime tests.
- Run tests against the just-built source runtime. Report the exact build and
  test commands, failures, skips, and unavailable tooling honestly.

Before promoting a graph performance/default change, record against supported
plain eager and `VulkanGraphProgram` in the same process:

- correctness and graph/eager parity;
- unsupported-node, fallback, readback, and deferred-value counts;
- repeated execution while prior outputs remain live;
- submit-origin/checkpoint counts and GPU timestamp attribution where relevant;
- first and repeat peak memory, with the current 5% no-regression gate; and
- three warmups plus 30 alternating samples per surface, including median and
  p95.

Opt-in canaries, replay lanes, compiled sessions, and benchmark-only bridges
are historical evidence, not supported baselines.

Keep raw model/performance artifacts in `agent_space/` and identify them with
the exact source commit, adapter, driver, input, route, and deployed DLL hash.

## Performance Work

- Attribute wall time before changing kernels. The current DAv2 evidence shows
  a substantial fixed submission/driver/queue floor, so prioritize generic
  submission cadence, dispatch, allocation, lifetime, and recording costs when
  larger inputs have similar wall time.
- Prefer generic improvements that help multiple corpus models. Do not add
  shape-specific routes merely because a benchmark shape is visible.
- Test one policy change at a time. Accept it only after correctness, memory,
  and latency gates. Record rejected candidates and their revisit condition,
  then remove experimental environment toggles and dead branches.
- Consult and update
  `test/vulkan_contract_proofs/performance_plan_evidence_manifest.json` for an
  accepted, rejected, canary, or correctness-blocked performance plan.

## Cleanup Policy

Cleanup is a background track, not a sprint that blocks graph executor or
corpus work. Forward progress creates deletion eligibility.

- Git is the archive. Preserve the supported result, evidence, and concise
  rejection/replacement reason, not inactive implementations.
- Every discovered schema, custom class, `PYTORCH_VULKAN_*` read, and public
  Python entry point belongs to exactly one ledger state: Active, Migration,
  Compatibility, or Delete-ready. There is no quarantine/default state.
- Migration code is deleted only after its named graph replacement satisfies
  the ledger gate against supported defaults. Do not use old canary timings as
  a deletion bar.
- Mechanism-only tests die with the mechanism; bug-class behavioral tests
  survive.
- Delete subsystem-specific docs in the same unit as the subsystem.
- Commit cleanup in coherent verified waves, with a ledger tombstone that
  prevents retired paths or dedicated symbols from silently returning.

For any cleanup surface change, run:

```powershell
.\.venv\Scripts\python.exe tools/vulkan_cleanup/generate_surface_inventory.py --write
.\.venv\Scripts\python.exe tools/vulkan_cleanup/generate_surface_inventory.py
$env:PYTHONPATH=(Resolve-Path .).Path
.\.venv\Scripts\python.exe test/test_vulkan_cleanup.py
.\.venv\Scripts\python.exe test/test_vulkan.py TestVulkanGovernance
```

Regenerate and validate contract accepted/proof manifests when deleting or
changing generated contract admission. Do not manually edit generated output
when its generator owns the change.

## Device And Release Contract

- Never persist a Vulkan device index as physical identity. Expose physical
  UUID, Windows LUID, normalized PCI address, and pipeline-cache UUID. Worker
  selection accepts physical UUID or valid Windows LUID before global Vulkan
  context/device creation, and the selected worker sees exactly that physical
  device as `vulkan:0`.
- Graph, compiled, tuning, and pipeline caches are disposable derived data.
  Invalidate them on backend commit/schema, shader digest, physical UUID,
  pipeline-cache UUID, driver/capability profile, graph/state/input signature,
  or planning-context mismatch as applicable.
- Production wheels come from a clean exact-commit release build. Torch and
  torchvision are co-built and tested for each supported Python ABI and
  platform; do not combine this fork with an arbitrary PyPI torchvision wheel.
- A release manifest records full source commits, wheel SHA-256 values,
  platform/ABI, compiler/runtime identity, Vulkan/SPIR-V build flags, backend
  schema/shader identity, and signatures. Verify the installed
  `torch.version.git_version` equals the pinned full commit.

## Documentation

- Keep architectural decisions concise and current. Clearly label historical
  measurements and rejected experiments.
- When fresh evidence changes a blocker or model gate, update the relevant
  current-state section in the same coherent change.
- Do not claim five-model, platform, or production readiness from a narrow
  operator test or import preflight.

## Linting

- Use only `spin` commands: start with `spin help`, normally run
  `spin quicklint` for changed files, and use `spin quickfix` only when wanted.
- Do not invoke `lintrunner` directly. That repository workflow is obsolete.
- If `spin` itself depends on unavailable external tooling, report the exact
  failure; do not silently substitute another linter or install tools without
  authorization.

## Git And Commits

- Do not commit unless the user or active goal explicitly authorizes commits.
  A persistent instruction to commit coherent chunks remains authorization.
- Review `git status`, `git diff`, and `git diff --check` before committing.
- Keep commits coherent and verified. Do not mix unrelated worktree changes.
- Commit messages should explain the logical change, not enumerate files.
- Disclose AI authorship informally in the commit body, for example
  `Authored by Codex.` Never add an AI `Co-authored-by:` trailer.
- Preserve `Pull-Request:` and `ghstack-source-id:` trailers. For ghstack, do
  not amend or push directly unless asked; submit through `ghstack`, using
  `--no-stack` for an intentional single-commit update.

## Repository Safety

- `.ci/docker/` is content-hashed. Do not touch it unless a Docker image rebuild
  is intentional.
- Avoid destructive git and filesystem operations. Verify absolute targets
  before recursive delete/move operations.
- Match existing PyTorch style. Prefer clear explicit state, concise comments,
  simple abstractions, and ASCII in new code comments.
- Use `torch._dynamo.config.patch` for temporary Dynamo configuration changes.
- Use `torch.cuda._utils._check_cuda_bindings` for `cuda.bindings` error checks.
- For B950 in an expected multi-line string, put `# noqa: B950` on the closing
  triple-quote line rather than changing the expected string.
- Use structured tracing for diagnostics that must survive production jobs;
  local scratch logs may supplement it but must not be the only signal.
