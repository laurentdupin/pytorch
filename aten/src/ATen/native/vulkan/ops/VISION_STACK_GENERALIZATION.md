# Vulkan Vision Stack Generalization

This note separates the DAv2 benchmark glue from the reusable Vulkan vision
stack owner machinery. The stack owner is intended to be shape and capability
driven; it must not select a backend path from a model name.

## DAv2-specific code

DAv2-specific assumptions are allowed in benchmark glue, tests, and docs:

- `scripts/benchmarks/benchmark_depth_anything.py` builds the DAv2 benchmark
  wrapper, capture layer list, and image preprocessing path.
- `run_depth_anything_v2_*` TorchBind schemas and bridge helpers are model glue
  around generic vision transformer depth infrastructure.
- Documentation files record DAv2 timing, token lengths, and observed bottleneck
  shapes.
- Tests may use DAv2-like synthetic sizes such as hidden `384`, heads `6`,
  head dim `64`, MLP hidden `768` or `1536`, and token lengths `2073`/`2110`.

These names must not be used as production dispatch criteria in generic Vulkan
ops.

## Backend-general mechanisms

The reusable stack infrastructure lives primarily in `VisionBlocks.cpp` and
`VisionBlocks.h`:

- `VisionBackboneStackContext` owns a list of block contexts plus stack-level
  hidden, head, and MLP metadata.
- `VulkanVisionStackShapeKey` keys fixed-shape plans by shape, dtype,
  capability, layout policy, attention policy, owner-program version, requested
  intermediate mask, and direct-attention capability.
- `VulkanVisionStackShapePlan` records fixed operation order, shapes, lifetimes,
  and step metadata without storing runtime tensor pointers or command buffers.
- Execution manifests, resource binding manifests, capture readiness, and replay
  readiness are diagnostics for future descriptor-table or command-recording work.
- Stack-owned direct attention is selected from tensor rank, dtype, head/value
  dimensions, layout, and hardware subgroup capability. Generic decomposed
  attention remains available outside the stack owner.

## Shape key fields

The current stack shape key distinguishes:

- token count
- hidden size
- number of heads
- head dimension
- MLP hidden size
- block count
- input dtype
- device capability key
- layout policy version
- attention policy version
- owner-program version
- requested intermediate mask
- direct-attention policy
- q4 subgroup availability

The key intentionally does not use raw tensor pointers or model names.

## Capability fields

The stack key currently records the q4 subgroup capability through
`q4_subgroup_available` and a coarse `device_capability_key`. Descriptor-table
or command-recording work should expand this only with stable device and layout
policy information, such as subgroup size requirements, direct-buffer layout
policy, and attention policy revisions.

## Unsupported shapes and dtypes

Unsupported inputs must reject or use a generic safe path; they must not silently
run the DAv2 q4 path.

- Non-FP32 stack inputs are rejected by the stack owner.
- Head/value dimensions other than `64` do not use stack-owned q4 direct
  attention. The stack may still execute through the existing safe generic
  attention path when supported.
- Different token lengths create distinct shape keys. Binding validation rejects
  attempts to bind one token length to a different fixed-shape plan.
- Requested intermediate outputs are marked as escaping and must be preserved
  independently of the token length.

## Generalization tests

The Vulkan test suite includes synthetic non-DAv2 stack fixtures that validate:

- a two-block `T=601` stack creates a fixed, safe shape plan and matches
  sequential block-owner execution
- `T=601` and `T=607` create distinct keys and reject mismatched binding
- a head-dim `80` stack avoids the q4 direct-attention path
- unsupported FP16 input rejects before stack execution
- requested intermediates are preserved for a three-block synthetic stack

## Remaining overfit risks

- Some older route-policy and diagnostic helper names still include `dav2`; they
  should be reviewed before broadening those paths beyond their current validated
  shape classes.
- The depth-anything compiled-session bridge remains model glue around generic
  vision-transformer-depth program construction.
- Future descriptor binding tables must use the shape/capability key and runtime
  binding validation rather than assuming only `T=2073` and `T=2110`.
