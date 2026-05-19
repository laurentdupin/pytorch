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

## Resource Binding Manifest

Shape plans now also produce a command-capture resource binding manifest:

```
vulkan_prepack::stack_resource_binding_manifest()
vulkan_prepack::reset_stack_resource_binding_manifest()
vulkan_prepack::stack_replay_readiness()
vulkan_prepack::stack_replay_binding_mode()
vulkan_prepack::stack_replay_counters()
vulkan_prepack::reset_stack_replay_counters()
```

Rows classify the logical resources for each stack plan step:

```
runtime input tensor
requested intermediate outputs
internal activations
q/k/v attention buffers
attention output
linear outputs
residual outputs
packed linear weights and biases
norm weights and biases
```

The manifest distinguishes persistent resources from runtime-bound resources and
internal temporaries. Persistent packed weights and norm parameters are stable
across forwards. Runtime inputs, requested outputs, and internal temporary
buffers are not persistent and would need descriptor rebinding or command
re-recording.

The current backend records descriptor sets inside each compute job. The
resource manifest is a logical classification layer; the descriptor binding
table below records set and binding indices for planned stack steps. This is
enough to prove re-record readiness, but not enough to safely replay a
previously recorded command buffer with new runtime tensors.

Replay readiness reports:

```
fixed_shape_plan=1
resources_classified=1
runtime_bindings_validated=1
descriptor_table_complete=1
descriptor_indices_known=1
descriptors_rebindable=1
persistent_resources_stable=1
internal_temps_owned=1
escaping_outputs_marked=1
no_cpu_fallback=1
no_host_sync=1
no_nested_replay=1
no_queue_idle=1
command_capture_safe=0
```

The binding mode is:

```
re_record_command_buffer_per_forward
```

No command-buffer replay is merged. The remaining blocker is replay-stable
descriptor ownership: captured command buffers would otherwise retain descriptor
sets that refer to old runtime inputs, internal temporaries, and escaping
outputs. The next pass should use the descriptor table for planned command
recording before attempting command replay.

## Descriptor Binding Table

Shape plans now build a planned descriptor binding table:

```
vulkan_prepack::stack_descriptor_binding_table()
vulkan_prepack::stack_descriptor_binding_validation()
vulkan_prepack::reset_stack_descriptor_binding_table()
```

The table is derived from the fixed stack plan and the current shader argument
conventions. Each row records the plan key, step ordinal, block, phase, op,
kernel, resource role, resource kind, lifetime, binding mode, descriptor set,
binding index, descriptor type, shape, dtype, and whether the descriptor is
runtime-varying.

The Vulkan API path still allocates and writes concrete descriptor sets per
compute job:

```
Context::submit_compute_job
DescriptorPool::get_descriptor_set
DescriptorSet::bind
DescriptorSet::get_bind_handle
CommandBuffer::bind_descriptors
```

The table models the same set `0` binding order without recording a command
buffer. Runtime inputs, requested outputs, metadata tied to runtime tensors,
and internal temps require descriptor updates when commands are re-recorded.
Packed weights, persistent biases, and norm parameters are marked persistent.

Validation currently reports:

```
table_complete=1
all_descriptor_indices_known=1
all_runtime_resources_rebindable=1
all_persistent_resources_stable=1
all_internal_temps_owned_or_rebindable=1
ready_for_re_record_per_forward=1
ready_for_command_replay=0
```

This means the shape plan is ready to drive planned command recording each
forward. It does not prove that a previously recorded command buffer can be
replayed with new resources. Command replay remains blocked because
program-owned temporaries are not yet stable replay resources and descriptor
updates without command re-recording have not been implemented.

## Planned Per-Forward Recording Readiness

`stack_planned_recording_readiness()` reports whether the shape plan and
descriptor table are sufficient to record one stack command buffer per forward
with current descriptors. This is intentionally separate from replay readiness.

Current result:

```
shape_plan_ready=1
descriptor_table_complete=1
ready_for_re_record_per_forward=1
no_cpu_fallback=1
no_host_sync=1
no_nested_replay=1
no_active_capture=1
command_recording_scope_available=0
barriers_recordable=0
descriptors_recordable=1
resources_lifetime_tracked=1
safe_to_record_stack_per_forward=0
```

The attempted stack-wide command scope exposed a command mutex re-entry hazard
in the current `Context` submission path. The safe result for this pass is
therefore diagnostics only: planned recording is rejected, the existing stack
owner remains canonical, and no command-buffer replay or persistent command
buffer is introduced. The next implementation needs a non-reentrant stack
recording API in `Context` that can append compute jobs and preserve barriers
without taking the per-job command mutex again.

## Current Decision

No command-buffer replay is merged. Planned per-forward recording is not merged
as an execution path because the command recording scope is not safe yet. The
next pass should fix the `Context` recording-scope prerequisite before trying
to batch the stack into one command submission.
