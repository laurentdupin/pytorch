# Vulkan Zero Fast Paths

## Byte Buffer Zero

The DAv2 one-block owner allocates Vulkan byte buffers while preparing packed
linear contexts. Before this path, `aten::zero_` only handled float buffers on
device, so byte buffers fell back to CPU with
`unsupported_shape_storage_or_dtype`.

`zero_buffer_uint8.glsl` zeros Vulkan `Byte` buffer tensors in-place using the
same metadata indexing and width-pack padding handling as the uint8
buffer-to-buffer copy shader. It is used when:

- storage is `BUFFER`
- dtype is `Byte`
- the adapter supports int8 buffer arithmetic
- rank is at most 4

The path is shader-only. `vkCmdFillBuffer` is not used here because it has offset
and size alignment requirements and depends on transfer destination buffer usage.

Diagnostics:

- `torch.ops.vulkan_prepack.zero_counters()`
- `torch.ops.vulkan_prepack.reset_zero_counters()`
- `PYTORCH_VULKAN_ZERO_PLAN_LOG`

The DAv2 owner setup tensors that triggered the fallback were byte buffers with
large one-dimensional shapes used by block0 qkv/proj/fc1/fc2 context setup.
