# Vulkan Backend Design Notes

This directory records the main findings, experiments, and current working
interpretation from the Vulkan backend work in this workspace.

Files:

- `vulkan_backend_findings_20260418.md`
  - Main handoff-style summary of what was tried, what landed, what regressed,
    and what still looks worth doing.

Related notes outside this directory:

- `comparison/depth_anything_v2_performance_notes_20260416.md`
  - Detailed cross-backend DAv2 performance note with artifact references.

Reading order:

1. Start with `vulkan_backend_findings_20260418.md`.
2. Use the comparison note for deeper DAv2 performance detail and cross-backend
   numbers.

Scope note:

- This is a design/history summary, not a fresh truth source for every number.
- Some sections are snapshots from earlier dates in the thread and are labeled
  that way.
- When in doubt, rerun the benchmark or test named in the note.
