# Vulkan Backend Design Notes

This directory records the main findings, experiments, and current working
interpretation from the Vulkan backend work in this workspace.

Files:

- `vulkan_backend_findings_20260418.md`
  - Consolidated handoff note covering the architecture that landed, cleanup
    findings, DAv2 performance investigation, the cooperative-matrix / BF16
    experiment, and the recommended next step.

Related notes outside this directory:

- `comparison/depth_anything_v2_performance_notes_20260416.md`
  - Detailed DAv2 cross-backend comparison note with benchmark artifacts.

Reading order:

1. Start with `vulkan_backend_findings_20260418.md`.
2. Use the comparison note for deeper cross-backend timing detail.

Mirror note:

- `design/` and `pytorch/design/` currently carry the same handoff note.
- If one copy changes, the other should be updated in the same edit to avoid
  drift.

Scope note:

- This is a design/history summary, not a live source of truth for every
  benchmark number.
- Some sections capture conclusions from a specific checkpoint and label the
  artifact that produced them.
- When in doubt, rerun the benchmark or test named in the note.
