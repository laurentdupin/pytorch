# Vulkan Replay Retirement

## Goal

The old replay and compiled-session paths are now quarantined while the backend
moves toward runtime-generated command lists:

```text
lazy op/region collection
semantic region planner
runtime shader/program generation
generated command list
explicit barriers/transitions/lifetime ownership
optional plan cache
```

Replay must not be used as the default answer for new performance work. New
work should either target the dynamic program runtime, the generated command-list
planner, or a contract/transition/lifetime proof that those systems consume.

The approved successor is now `VulkanGraphProgram`, described in
`docs/vulkan/GRAPH_RUNTIME.md`. New command-list work must be generated from an
exported graph partition with program-owned memory and descriptors. Do not add
another eager op-stream capture or replay bridge while migrating.

## Current Inventory

### Quarantined benchmark surface

`scripts/benchmarks/benchmark_depth_anything.py` no longer exposes
`compiled_session_bridge` as a selectable
`--vulkan-stack-output-device-bridge-mode`. The mode remains named only as a
deprecated replay mode so old evidence and docs can be interpreted.

### Public replay bridge APIs

The current public `vulkan_prepack` replay/compiled-session bridge APIs are
frozen by governance coverage:

```text
vulkan_prepack::run_attention_runtime_buffer_math_replay_bridge
vulkan_prepack::run_depth_anything_v2_compiled_session_bridge
vulkan_prepack::run_depth_anything_v2_image_compiled_session_bridge
vulkan_prepack::run_vision_backbone_decoder_replay_bundle_bridge
vulkan_prepack::run_vision_backbone_stack_compiled_session_bridge
vulkan_prepack::run_vision_backbone_stack_norm_compiled_session_bridge
vulkan_prepack::run_vision_backbone_stack_norm_replay_bundle_bridge
vulkan_prepack::run_vision_backbone_stack_replay_bundle_bridge
```

Do not add another public replay bridge API. If a generated command-list
successor needs a public test hook, name it after generated regions or command
plans, not replay.

### Kept for diagnostics and safety

These pieces remain for now because they are diagnostics, state validation, or
stale-tensor safety checks rather than a benchmark execution shortcut:

```text
PYTORCH_VULKAN_REPLAY_LOG
PYTORCH_VULKAN_COMPILED_SESSION_LOG
ReplayTensorState stale-view validation
stack_replay_readiness()
stack_replay_binding_mode()
stack_replay_counters()
reset_stack_replay_counters()
```

They should be renamed or deleted once generated command-list diagnostics cover
the same failure modes.

## Retirement Rules

- No new replay or compiled-session benchmark modes.
- No new public `vulkan_prepack::*replay*bridge` or
  `vulkan_prepack::*compiled_session*bridge` APIs.
- No replay route may become a default model path.
- Existing replay bridge APIs are migration targets, not expansion points.
- Runtime-generated command-list work may consume replay diagnostics only as
  evidence of blockers that must not recur.
- Deleting replay code is allowed once its caller is migrated or the evidence
  table records it as dead.

## Next Migration Targets

1. Replace `run_attention_runtime_buffer_math_replay_bridge` with a generated
   SDPA/attention command-list region.
2. Replace vision stack replay bundle bridges with generated stack-region
   command lists.
3. Replace compiled-session bridge tests with generated command-list parity
   tests.
4. Retire replay tensor stamps after generated-region output ownership has
   equivalent stale-view and escape validation.
