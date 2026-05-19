# Vulkan DAv2 Stack Execution Program Readiness

The DAv2 Vulkan benchmark enters a single stack owner for the 12-block backbone.
The stack owner is intentionally safe: it does not use replay, compiled-session
capture, or nested replay. Stack-owned attention is a first-class phase and no
longer creates decomposed `[6,T,T]` scores/probability carrier tensors.

## Manifest

`vulkan_prepack::stack_execution_manifest()` returns diagnostic rows for the
stack-owned execution sequence. Each row records:

```
ordinal
block
phase
op
kernel
input_shapes
output_shapes
dtype
uses_dynamic_shape
allocates_output
writes_preexisting_output
escapes_stack
requested_intermediate
requires_cpu_data
uses_fallback
submits_command_buffer
requires_host_sync
uses_runtime_capture
uses_replay
safe_to_capture
```

The manifest covers the expected stack phases:

```
norm1
qkv_linear
qkv_transform
attention
proj_linear
residual1
norm2
fc1_gelu
fc2
residual2
intermediate_capture
```

The diagnostic DAv2 run recorded 6528 manifest rows across the benchmark
workload:

```
norm1                 576
qkv_linear            576
qkv_transform         576
attention            1152
proj_linear           576
residual1             576
norm2                 576
fc1_gelu              576
fc2                   576
residual2             576
intermediate_capture  192
```

Every requested intermediate row is marked as `escapes_stack=1` and
`requested_intermediate=1`. Internal rows are not marked as escaping.

## Readiness Rules

`vulkan_prepack::stack_capture_readiness()` returns:

```
fixed_shapes
no_cpu_fallback
no_host_sync
no_nested_replay
no_active_capture
requested_intermediates_marked
internal_outputs_owned
known_lifetimes
safe_to_capture
```

The diagnostic run reported:

```
fixed_shapes=0
no_cpu_fallback=1
no_host_sync=1
no_nested_replay=1
no_active_capture=1
requested_intermediates_marked=1
internal_outputs_owned=1
known_lifetimes=1
safe_to_capture=0
```

Capture is not enabled yet because the current canonical DAv2 benchmark stack
still uses runtime token lengths. The same stack context sees both `T=2073` and
`T=2110`, and every manifest row is therefore marked `uses_dynamic_shape=1`.
A single fixed-shape command/program capture would need either a shape-keyed
program cache with correct resource rebinding or a separate fixed-shape
programmed sequence for each token length.

## Current Decision

No stack replay or command-buffer capture is merged in this pass. The readiness
manifest proves the safety properties around fallback, host sync, nested replay,
intermediate escapes, and lifetimes, but it also identifies the exact blocker:
fixed shapes are not established at the stack program boundary.

The next program-level step should be a shape-keyed stack program plan for
`T=2073` and `T=2110`, including explicit resource rebinding and invalidation
rules. That should be implemented before any command-buffer replay path is made
canonical.
