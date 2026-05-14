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

Copy elision should only be added when the provenance log proves that the copy is
a true logical no-op, or that the downstream Vulkan consumer accepts the producer
layout without changing aliasing or output semantics.
