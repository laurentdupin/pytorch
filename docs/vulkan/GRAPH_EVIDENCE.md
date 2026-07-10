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

Example:

```text
python scripts/benchmarks/vulkan_graph_export_evidence.py \
  --adapter C:\work\dav2_adapter.py:build \
  --external-root C:\external\Depth-Anything \
  --checkpoint C:\external\Depth-Anything\checkpoints\depth_anything_v2_vits.pth \
  --output-dir C:\results\dav2_vits_graph \
  --eager-atol 0.0 --eager-rtol 0.0 \
  --cpu-atol 0.002 --cpu-rtol 0.0
```

The caller-owned output directory receives measured
`dav2_vits_export_census.json` and `dav2_vits_export_parity.json`. Checked-in
files under `test/vulkan_graph/evidence/` are schema-valid templates only;
they never claim local-device measurements or include machine paths.

The machine-readable evidence records graph census and lowerings, guard
outcomes, program key, Vulkan runtime identity and DLL hashes, timing,
fallback/readback/deferred-value counters, and CPU/eager Vulkan parity. It
does not authorize model-name production dispatch, exact-shape admission, or
Python-executor performance tuning.

The default tolerances are zero. A nonzero tolerance is an explicit corpus
evidence choice and is written into the measured parity artifact. Graph versus
eager Vulkan should remain exact unless a documented backend precision contract
requires otherwise.
