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
