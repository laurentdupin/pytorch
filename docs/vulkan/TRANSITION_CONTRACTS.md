# Vulkan Transition Contracts

Transition contracts describe movement or materialization between a producer and
consumer after a kernel/shape contract has made an op legal. They are distinct
from `ExecutionContracts`: execution contracts admit an op family, while
transition contracts classify edges such as metadata views, descriptor views,
device copies, layout repacks, semantic materializations, host uploads,
readbacks, fallback staging, and region lifetime decisions.

The initial implementation is behavior-neutral. `TransitionContracts.h` defines
the controlled reason taxonomy and `TransitionPlanner.h` defines a minimal
request/admission record. `TransitionPlanner.cpp` can classify and log observed
transitions, but it does not redirect execution, remove copies, defer submits,
or change route policy.

## Reason Taxonomy

The first taxonomy is intentionally broad enough to count both required and
suspect transitions:

- `metadata_view_only`
- `descriptor_view_only`
- `required_semantic_clone`
- `required_semantic_cat`
- `required_contiguous_materialization`
- `required_consumer_layout`
- `required_layout_repack`
- `required_dtype_cast`
- `required_host_upload`
- `required_final_readback`
- `required_debug_readback`
- `required_correctness_materialization`
- `temporary_region_scratch_copy`
- `avoidable_redundant_copy`
- `avoidable_layout_churn`
- `unexpected_cpu_staging`
- `unexpected_intermediate_readback`
- `fallback_materialization`
- `unsupported_stride_for_consumer`
- `unsupported_storage_offset_for_consumer`
- `missing_lifetime_proof`
- `budget_blocked`
- `unknown_transition_reason`

Unknown reasons are allowed in the skeleton, but they must stay visible and
countable so follow-up tasks can replace them with scoped proof.

## JSONL Logging

Set `PYTORCH_VULKAN_TRANSITION_LOG=<path>` to write one JSON object per observed
transition. Events use `event="vulkan_transition"` and include:

- `phase`
- `reason`
- `kind`
- `outcome`
- `bytes`
- `host_transfer`
- `physical_copy`
- `sync_required`
- `queue_submit_required`
- `producer_schema` / `consumer_schema`
- `producer_contract` / `consumer_contract`
- source and destination dtype, shape, stride, layout, and storage fields where
  the call site knows them

This log is separate from contract admission diagnostics, op-hit logs, tensor
state logs, and phase counters. It is for attribution and proof planning only.

## Current Transition Specs

`AttentionProbabilityMaterializationContract` is the first transition contract
spec tied to real model-corpus traffic. It covers the bounded
softmax-probability materialization edge where `aten::_softmax` probabilities
are cloned into a direct/materialized Vulkan buffer before value BMM. The
transition reason is `required_correctness_materialization`, the kind is
`semantic_materialization`, `physical_copy=true`, `host_transfer=false`, and
the copy/readback budget is zero CPU fallback, zero sync readback, and zero host
transfer.

The checked-in spec remains in
`test/vulkan_contract_specs/attention_probability_materialization_contract.json`
and its generated helper remains
`aten/src/ATen/native/vulkan/planning/generated/ExecutionContractsAttentionProbabilityMaterializationSpec.h`.
Rows with `materialization_policy ==
vulkan_clone_probability_before_value_bmm` are the remaining clone-required
transition-contract rows. The original Lotus `[10,126,126]` proof row remains
clone-required. The six already-admitted `VisionSelfAttentionSDPAContract`
low-resolution rows with probability scores `[BH,T,T]`, `BH in {6,12,16}`,
`T in {151,261}`, and value dim `64` are now recorded as
`direct_safe_recorded_no_forced_materialization` and skip the probability clone
only under the bounded VisionSelfAttention and transition-contract guards.

The transition log attaches `producer_contract =
AttentionProbabilityMaterializationContract` and `consumer_contract =
DecomposedAttentionProbabilityToValueBmm` to remaining matching clone-required
events so future work can measure and review this edge as a named transition
contract.

`HostUploadTransitionContract`, `MetadataViewTransitionContract`,
`FinalReadbackContract`, `IntermediateReadbackTransitionContract`,
`SafeContiguousMaterializationContract`, `FallbackMaterializationContract`, and
`LayoutRepackTransitionContract` are classification-only reason-bucket specs
for existing transition logs. They live in
`test/vulkan_contract_specs/*_transition_contract.json`, have
`source_status = schema_only`, and do not admit backend routes or change copy,
submit, fallback, materialization, or readback behavior.

`HostUploadTransitionContract` covers `required_host_upload` /
`host_transfer` events from CPU tensors into Vulkan tensors. These events are
physical host transfers and remain counted as uploads; the spec only gives the
collector a source-of-truth bucket for reporting them.

`MetadataViewTransitionContract` covers `metadata_view_only` /
`metadata_view` events such as `MetadataViewCreated` and
`TypedMetadataViewCreated`. These events must remain metadata-only:
`physical_copy=false`, `host_transfer=false`, `sync_required=false`, and
`queue_submit_required=false`.

`FinalReadbackContract` covers `required_final_readback` / `host_transfer`
events from Vulkan tensors to CPU tensors for final user-visible observation.
These events remain synchronous readbacks and host transfers; the spec only
prevents matching events from being reported as missing transition-contract
buckets.

`IntermediateReadbackTransitionContract` covers
`unexpected_intermediate_readback` / `host_transfer` events from Vulkan tensors
to CPU tensors before final output observation. These events remain visible
readbacks and are not eliminated or reclassified as acceptable route behavior.

`SafeContiguousMaterializationContract` covers
`required_contiguous_materialization` / `layout_materialization` events from
`materialize_to_contiguous_buffer` into `buffer_to_buffer`. These are
device-side physical copies with no host transfer or sync readback; the spec is
only a collector bucket for existing safe contiguous materialization evidence.

`FallbackMaterializationContract` covers generic `fallback_materialization` /
`fallback` events such as visible owner-context unpack readbacks. These events
remain counted fallback materialization, host transfers, and sync readbacks;
the spec only gives the collector a source-of-truth bucket for reporting them.

`SmallControlTensorFallbackContract` and
`SmallControlScalarExtractionContract` refine the generic fallback bucket for
tiny Python control-plane tensors. They classify bounded Bool/Long/Int tensors
with `numel <= 16`, plus Float only for comparison-control rows, when execution
falls back through known generation-control ops such as `isin`, `any/all`,
comparison, binary/bitwise control ops, `masked_fill`, `fill_`, `to`, `max`, or
`cat`. Scalar extraction covers `_local_scalar_dense` on single-element control
tensors. These contracts do not authorize native Bool or Long execution, do not
hide fallback/readback counters, and keep larger or unlisted tensors in the
generic fallback bucket.

`ConvWeightLayoutRepackTransitionContract` is a more specific
classification-only bucket for `vulkan_prepack::conv2d_context ->
vulkan_weight_cpu_materialization` events. It records that the legacy conv2d
image-packed path reads actual Vulkan weight values back to CPU so
`pack_weights` can rearrange them into the shader-packed layout and upload the
packed weights back to Vulkan. The transition log attaches the contract name
and source tensor metadata where available, but the CPU readback remains
visible and unchanged. Explicit `Conv2dPackedContext::unpack()` and pickle
semantics must keep returning real tensors.

`LayoutRepackTransitionContract` covers `required_layout_repack` /
`layout_materialization` events such as `aten::cat -> buffer_to_buffer`
device-side repacks. These events remain physical device copies with no host
transfer or sync readback; the spec does not remove or redirect the repack.

`PointwiseConvInputLayoutTransitionContract` is a schema-only proof bucket for
1x1 convolution input descriptor-view evidence behind
`SmallSpatialPointwiseConvContract`. It distinguishes storage-offset-zero
width-packed rows that may use the existing as-linear descriptor-view path from
nonzero-storage-offset token-slice metadata views that currently stay on the
generic pointwise path because descriptor-view parity is not proven. The spec
does not broaden pointwise-conv admission, select as-linear for token slices,
or add hidden materialization.

## Rollout

The first wired sites classify existing copy and materialization observations:

- device-device buffer copies from the existing Vulkan buffer-copy counter site;
- CPU-to-Vulkan uploads;
- Vulkan-to-CPU readbacks, split by submit phase when available;
- CPU fallback staging/readback observations;
- layout and metadata-view transitions from the existing layout-transition
  helper.

Future transition-contract work should add precise producer/consumer contracts
and richer logical/physical descriptors before changing behavior. Any behavior
change must remain separately validated with parity, fallback/readback counters,
and model-corpus evidence.
