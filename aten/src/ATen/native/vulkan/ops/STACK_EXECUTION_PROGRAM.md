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

## Shape-Keyed Plans

The stack context now owns a cache of fixed-shape execution plans keyed by:

```
tokens
hidden
num_heads
head_dim
mlp_hidden
num_blocks
dtype
device_capability_key
layout_policy_version
attention_policy_version
owner_program_version
requested_intermediate_mask
direct_attention
q4_subgroup_available
```

The whole stack context can still observe multiple token lengths, so the dynamic
stack readiness may continue to report `fixed_shapes=0`. Each shape plan is
fixed independently. The canonical DAv2 benchmark produces separate plans for
the two observed patch-grid token counts:

```
T=2073
T=2110
```

These values come from DAv2 preprocessing at input size 518. Images in the
example corpus produce patch grids of `37x56` and `37x57`; adding the class token
gives `2073` and `2110` tokens. The token length changes across images, not
within a block.

The following diagnostic APIs expose the plan layer:

```
vulkan_prepack::stack_shape_plan_keys()
vulkan_prepack::stack_shape_plan_readiness()
vulkan_prepack::stack_shape_plan_counters()
vulkan_prepack::reset_stack_shape_plan_counters()
vulkan_prepack::validate_stack_shape_plan_binding(...)
```

Plan rows are also appended to `stack_execution_manifest()` as
`stack_shape_plan_manifest` rows. These rows have `uses_dynamic_shape=0` and
`fixed_shapes=1`.

## Binding And Invalidation

Plans store operation order, shapes, roles, policy versions, and context shape
metadata. They do not store runtime tensor pointers, command buffers, or stale
input/output resources.

Runtime binding validation currently rejects:

```
tokens_mismatch
hidden_mismatch
dtype_mismatch
requested_intermediates_mismatch
plan_not_found
```

The invalidation model also reserves counters for device/capability and context
identity changes. Those become mandatory before command-buffer replay can bind
resources from cached plans.

## Current Decision

No command-buffer replay is merged in this pass. The shape-keyed plan cache
turns the previous fixed-shape blocker into per-plan readiness: each observed
token length can be fixed-shape even though the parent stack context is dynamic.
The next pass can attempt a narrow programmed sequence or command-buffer capture
against these shape plans, with resource rebinding validated before execution.
