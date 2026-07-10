# Vulkan Legacy Sunset Table

This table records deletion gates for legacy eager and replay mechanisms. It is
kept separate from `CURRENT_STATE.md` so that current architecture remains
compact.

| Legacy mechanism | Default policy | Delete when |
| --- | --- | --- |
| Eager image-normalize, query-scale, decomposed-attention, layer-scale, and add-layer-norm bridges | Concrete eager execution only | Equivalent graph rewrites have corpus parity and no fallback/readback regression |
| Runtime deferred elementwise placeholder | Explicit diagnostic opt-in only | Graph-generated elementwise fusion owns program resources and passes repeated-run lifetime tests |
| Linear/GELU deferred registry | No default producer | Graph linear/GELU rewrite has parity; retain only the explicit fused context operation |
| Linear pending-flush deferral | Explicit diagnostic opt-in only | Graph program contexts and timeline completion own retained packed resources |
| Stack proof/canary machinery | Quarantined | Generated graph regions provide explicit values, transitions, output generations, and timeline retirement parity |
| Compiled-session and replay bridges | Quarantined | Bounded graph-owned command partitions replace every caller and preserve stale-output diagnostics |
| Replay/stack environment toggles | Diagnostic only | Their replacement has a compact graph scoreboard and focused parity coverage |
| Model-oriented VisionBlocks orchestration | Existing eager fallback only | Generic export lowering and graph regions cover the same semantic subgraphs without model dispatch |

No item may be deleted before its replacement has correctness parity, explicit
fallback/readback accounting, dynamic-shape coverage, and repeated-run output
ownership validation.
