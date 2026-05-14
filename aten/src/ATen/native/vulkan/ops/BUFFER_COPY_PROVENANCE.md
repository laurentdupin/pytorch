# Vulkan Buffer Copy Provenance

DAv2 no longer has one dominant non-attention copy source after the qtile
attention path. Buffer copy provenance is therefore logged before any copy
elision is attempted.

Enable logging with:

```
PYTORCH_VULKAN_BUFFER_COPY_LOG=path/to/buffer_copy.log
```

The log records every Vulkan buffer-to-buffer materialization path currently
covered by:

- `aten::copy_.buffer_to_buffer`
- `materialize_to_contiguous_buffer`
- direct buffer tensor copies used by replay/program helpers

Each line includes:

- reason
- logical and physical byte count
- source and destination sizes/strides
- storage offsets
- direct-buffer flags
- whether the copy is a logical no-op
- producer and consumer labels where known

The counters are exposed through:

```
torch.ops.vulkan_prepack.buffer_copy_counters()
```

Counter layout:

```
0 total
1 total_bytes
2 explicit_copy
3 contiguous
4 view_materialization
5 reshape_materialization
6 permute_materialization
7 transpose_materialization
8 layout_conversion
9 attention_materialization
10 linear_materialization
11 conv_materialization
12 decoder_materialization
13 backbone_materialization
14 logical_noop_copy
```

Use the DAv2 profile summarizer to aggregate the log:

```
python scripts/diagnostics/summarize_vulkan_dav2_profile.py \
    --buffer-copy-log comparison/dav2_buffer_copy.log \
    --top 40
```

For stable benchmark runs, prefer the in-process aggregate profiler. It avoids
per-copy text output and is intended for warmup/repeat profiles:

```
PYTORCH_VULKAN_BUFFER_COPY_AGGREGATE=1
```

The aggregate snapshot is exposed through:

```
torch.ops.vulkan_prepack.buffer_copy_aggregate_snapshot()
torch.ops.vulkan_prepack.reset_buffer_copy_aggregate()
```

Each aggregate row is keyed by:

- reason
- producer and consumer labels
- producer and consumer roles, when tensor provenance or allocation labels are
  available
- dtype
- source and destination sizes/strides
- direct-buffer flags

Dump the snapshot after a benchmark and summarize it with:

```
python scripts/diagnostics/summarize_vulkan_dav2_profile.py \
    --buffer-copy-aggregate comparison/dav2_buffer_copy_aggregate.txt \
    --top 40
```

In the 2026-05 DAv2 qtile profile, logical no-op copies remained zero. The top
copy traffic was distinct-buffer `explicit_copy` and `tensor_to_contiguous`
traffic. Q/K/V-shaped attention materialization was visible, but it was not a
top-five copy source by bytes in the stable aggregate profile, so the packed-QKV
attention path was not added in that pass.

The next classifier pass adds producer and consumer roles to separate broad
`explicit_copy` traffic. In the stable DAv2 profile, the largest `[1,T,1536]`
copies were classified as `aten::gelu` output copied into a clone destination,
with a matching unknown producer copy of the same shape. That confirms the large
shape is MLP-adjacent, but it is not safe to elide as a raw copy because the
destination is a distinct buffer. A fused fc1+GELU bridge should only be enabled
when the producer/consumer path can hand the fused result to fc2 without forcing
extra clone lifetime or allocation pressure.

Clone requirement profiling is exposed through:

```
torch.ops.vulkan_prepack.clone_requirement_snapshot()
torch.ops.vulkan_prepack.reset_clone_requirement_snapshot()
```

The DAv2 MLP `[1,T,1536]` GELU clone is classified as `fc2_input_preparation`.
The real DAv2 run that produced this clone did not go through the existing
`VisionBackboneProgram` owner, so fc1 and fc2 packed contexts were not available
together at the backend boundary. The safe next implementation is therefore a
proper DAv2 MLP/backbone owner that can allocate an fc2-compatible GELU scratch
inside the region. A local clone alias or global clone elision is still
forbidden.

The one-block DAv2 owner path now makes that backend boundary real and caches the
created block context outside measured forward loops. Phase-scoped fallback
accounting showed:

- context unpack readbacks are `owner_context_create`
- the positional embedding materialization is `positional_embedding_setup`
- the owner block forward is fallback-free

The owner MLP body already calls `run_linear_gelu_context` before `fc2`, so the
next copy-provenance comparison should measure eager Vulkan versus one-block
owner with that existing fused MLP path before adding a separate GELU scratch
implementation.

Copy elision should only be added when the provenance log proves that the copy is
a true logical no-op, or that the downstream Vulkan consumer accepts the producer
layout without changing aliasing or output semantics.
