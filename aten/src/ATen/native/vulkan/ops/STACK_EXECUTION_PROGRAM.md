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

That rejected attempt held the command mutex across stack execution and then
re-entered the same mutex from each `submit_compute_job` call. The accepted
Context prerequisite is a non-reentrant stack recording state:

```
begin_stack_planned_recording()
    briefly locks the command mutex
    flushes any prior pending command buffer
    marks the current thread as the stack recording owner
    releases the mutex before stack execution

submit_compute_job()
    locks once per compute job as before
    records descriptors, barriers, profiling timestamps, and dispatch in order
    skips `cmdSubmitFrequency` flushing while stack recording is active

end_stack_planned_recording_and_submit()
    locks once at stack exit
    submits the accumulated command buffer
    clears the stack recording state
```

The command mutex is never held across Python/operator execution, and no
recursive mutex is introduced. Per-job barriers and descriptor writes remain in
the same recording path as the safe owner path, so the change batches command
submission without changing math or shader selection.

The resulting readiness is:

```
shape_plan_ready=1
descriptor_table_complete=1
ready_for_re_record_per_forward=1
no_cpu_fallback=1
no_host_sync=1
no_nested_replay=1
no_active_capture=1
command_recording_scope_available=1
barriers_recordable=1
descriptors_recordable=1
resources_lifetime_tracked=1
safe_to_record_stack_per_forward=1
```

## Current Decision

Planned per-forward recording is the canonical path when the fixed shape plan
and descriptor table validate. It re-records with current runtime resources
every forward and submits once at the stack boundary. No command-buffer replay
or persistent captured command buffer is merged.

True replay remains blocked by replay-stable program-owned temporaries and a
descriptor update model that can safely rebind those resources without
re-recording.

## Submit-Origin Diagnostics

`submit_origin_counters()` classifies actual queue submissions at the Context
submission point. It is separate from logical `submit_compute_job` counts. The
origin fields are:

```
total_queue_submits
normal_cmd_submit_frequency
stack_planned_recording_submit
pre_stack_flush
post_stack_flush
explicit_synchronize
tensor_cpu_readback
fallback_readback
retire_queue_drain
profiling_timestamp_reset
profiling_timestamp_readback
shutdown
debug_validation
unknown
```

Planned recording also reports `premature_stack_submit_count` and
`suppressed_frequency_flush_count` in `stack_planned_recording_counters()`.
The expected healthy stack-owned run has:

```
stack_planned_recording_submit == recording_scope_submit_count
premature_stack_submit_count == 0
suppressed_frequency_flush_count > 0
unknown == 0
```

This proves the local stack behavior even when whole-benchmark submit totals
are dominated by setup, decoder/head, readback, retire, or profiling work
outside the stack owner. `submit_origin_phase_counters()` further classifies
actual queue submissions by diagnostic phase. The phase label is not used for
routing.

The no-timestamp DAv2 submit-origin profile with planned stack recording
reported:

```
total_queue_submits=4773
stream_submit_count=4773
normal_cmd_submit_frequency=273
stack_planned_recording_submit=92
explicit_synchronize=1150
tensor_cpu_readback=560
retire_queue_drain=2698
unknown=0

recording_scope_begin=92
recording_scope_submit=92
recorded_stack_compute_jobs=15068
premature_stack_submits=0
suppressed_frequency_flushes=868
```

With `cmdSubmitFrequency=16`, the old frequency-batched stack estimate is
`ceil(15068 / 16) = 942` submits. Planned recording submits the 92 stack scopes,
for an estimated local stack submit reduction of 850 submits. Whole-benchmark
submits remain high because retire, explicit synchronization, tensor CPU
readback, and normal frequency submits remain.

The submit-phase follow-up profile showed the largest remaining bucket is not a
missed planned-recording flush:

```
retire_queue_drain stack_owner=2208
retire_queue_drain unknown=490
explicit_synchronize stack_owner=1104
tensor_cpu_readback model_setup=506
tensor_cpu_readback readback=46
normal_cmd_submit_frequency unknown=273
```

`retire_drain_counters()` reported `queue_submit_count=2698`,
`poll_only_count=219`, and `blocking_wait_count=0`. The reason breakdown was:

```
stack_scope_end=2208
resource_pressure=491
readback_preparation=46
setup_phase=172
```

Most retire submits are stack-owner native layer norm lifetime boundaries. They
submit pending work so short-lived uniform/metadata resources can be retired by
timeline; the profile does not prove these are safely replaceable with polling.
No retire/sync/readback cleanup was made in this pass.

The next retire pass added call-site and resource-role accounting. It showed
that native layer norm metadata is high-count but not the byte or submit-driver
source:

```
retire_call_site stack_owner_norm1 submit=1104 pending_bytes=294435440
retire_call_site stack_owner_norm2 submit=1104 pending_bytes=31627193088
retire_call_site context_flush_pending submit=490 pending_bytes=1810223264

retired_resource stack_internal_temp stack_owner_phase_boundary
  count=11090 bytes=45560691008
retired_resource stack_internal_temp stack_owner_norm2
  count=9088 bytes=31625501952
retired_resource native_layer_norm_metadata stack_owner_norm2
  count=6624 bytes=423936
retired_resource native_layer_norm_metadata stack_owner_phase_boundary
  count=6072 bytes=388608
retired_resource native_layer_norm_uniform stack_owner_norm2
  count=1104 bytes=17664
```

The native layer norm metadata/uniform contents are shape/dispatch metadata and
stable per shape, but they are tiny compared with stack-internal tensor buffers.
Persisting them alone would not remove the norm retire submits because the same
call sites also retire large stack-internal buffers. No metadata persistence or
ring was implemented. The next proven target is stack-internal temp lifetime
planning around norm2/residual boundaries, not command-buffer replay.

The next pass refined the broad `stack_internal_temp` bucket into phase-derived
roles and added `stack_temp_lifetime_safety_snapshot()`. The no-timestamp DAv2
profile still showed the same retire-submit shape:

```
total_queue_submits=4773
retire_queue_drain=2698
explicit_synchronize=1150
tensor_cpu_readback=560
normal_cmd_submit_frequency=273
stack_planned_recording_submit=92
unknown=0

retire_call_site stack_owner_norm1 submit=1104 pending_bytes=294435440
retire_call_site stack_owner_norm2 submit=1104 pending_bytes=31627193088
retire_call_site context_flush_pending submit=490 pending_bytes=1810223264
```

The top stack temp rows were:

```
stack_internal_temp stack_owner_phase_boundary
  count=7820 bytes=25025479680
stack_internal_temp stack_owner_norm2
  count=5520 bytes=24731062272
stack_fc2_output stack_owner_phase_boundary
  count=1129 bytes=18965065728
stack_attention_output stack_owner_norm2
  count=1104 bytes=3533008896
stack_proj_output stack_owner_norm2
  count=128 bytes=1643913216
stack_qkv_output stack_owner_norm2
  count=128 bytes=1643913216
stack_fc1_gelu_output stack_owner_phase_boundary
  count=117 bytes=1502674944
```

Every concrete stack temp role in the lifetime snapshot is currently classified
as `unsafe_unknown_consumer`. This is intentional: the cleanup path that creates
retire entries receives raw `VulkanBuffer`/`VulkanImage` resources after tensor
storage release, not a proven producer/consumer graph, shape, dtype, escape
state, or requested-intermediate/final-output bit for each retired allocation.
Some resources are also still only classifiable as `stack_internal_temp` because
they reach the retire queue outside a specific stack phase. Therefore no
stack-internal-temp retire batching was implemented. The safe next step is to
carry explicit stack tensor lifetime/provenance into the retire entry before any
batching change.

Pending retire entries now carry that stack provenance. Stack-created tensor
storage records the active stack phase, block index, producer role, lifetime
class, logical shape/strides, dtype, storage flags, escape/requested/final flags,
and alias/view state. `retired_resource_aggregate_snapshot()` and
`stack_temp_lifetime_safety_snapshot()` include this metadata for stack rows.

The no-timestamp DAv2 profile after provenance propagation still reports the
same submit totals:

```
total_queue_submits=4773
retire_queue_drain=2698
explicit_synchronize=1150
tensor_cpu_readback=560
normal_cmd_submit_frequency=273
stack_planned_recording_submit=92
unknown=0
blocking_wait=0
```

The stack temp bytes now split by concrete role:

```
stack_qkv_output        bytes=22842028800 count=6896
stack_fc1_gelu_output  bytes=15775975680 count=2480
stack_fc2_output       bytes=10108794624 count=4688
stack_residual2_output bytes=7079187072  count=11044
stack_residual1_output bytes=7066450560  count=11040
stack_attention_output bytes=7066427520  count=10032
stack_proj_output      bytes=5177001984  count=3584
stack_norm2_output     bytes=3606613248  count=3312
stack_norm1_output     bytes=3606613248  count=3312
```

All of these rows remain `unsafe_unknown_consumer`. The metadata now proves
producer, shape, dtype, and storage class, but it still does not prove the last
consumer recorded before the stack planned-recording submit. Retire batching was
therefore not enabled. The next lifetime step is to attach last-use/consumer
proof from the shape plan or stack execution manifest to each stack temp.

The stack owner now installs a diagnostic last-use proof scope derived from the
fixed shape plan while a stack forward is running. Each proof row records the
producer phase/block/role, the expected final consumer phase/block, whether that
consumer is recorded before the stack planned-recording submit, and whether the
resource is internal and non-escaping. Pending retire provenance uses this proof
only for diagnostics; no retire batching or replay behavior is enabled.

The no-timestamp DAv2 profile after last-use proof propagation kept the same
submit totals:

```
total_queue_submits=4773
retire_queue_drain=2698
explicit_synchronize=1150
tensor_cpu_readback=560
normal_cmd_submit_frequency=273
stack_planned_recording_submit=92
unknown=0
blocking_wait=0
```

Some fixed-plan internal temps are now provably safe to defer in the diagnostic
snapshot:

```
stack_qkv_output       safe_to_defer_until_stack_submit count=4416 bytes=21198053376
stack_fc1_gelu_output  safe_to_defer_until_stack_submit count=1104 bytes=14132035584
stack_residual1_output safe_to_defer_until_stack_submit count=2208 bytes=7066017792
stack_fc2_output       safe_to_defer_until_stack_submit count=1104 bytes=3533008896
stack_norm1_output     safe_to_defer_until_stack_submit count=1104 bytes=3533008896
stack_norm2_output     safe_to_defer_until_stack_submit count=1104 bytes=3533008896
stack_proj_output      safe_to_defer_until_stack_submit count=1104 bytes=3533008896
stack_attention_output safe_to_defer_until_stack_submit count=1104 bytes=3533008896
```

The same profile still has mixed unsafe rows for several roles, requested
intermediate escapes for `stack_residual2_output`, phase-boundary block outputs,
and generic `stack_internal_temp` rows that do not yet match a shape-plan proof.
For that reason this remains a proof-only pass. The next safe implementation
target is a narrow retire-batching change for one proof-complete resource class,
with same-role unsafe rows explicitly excluded.
