# Vulkan Agent Task Template

Use this template for bounded Vulkan backend tasks.

## Task Card

Title:

Workspace: `C:\Users\Frere\Source\GitRepos\pytorch`

Goal:

Scope:

- Allowed files:
- Out of scope:
- Do not commit unless explicitly asked.

Required reading:

- `AGENTS.md`
- `aten/src/ATen/native/vulkan/AGENTS.md`
- `docs/vulkan/PROJECT_CHARTER.md`
- `docs/vulkan/CURRENT_STATE.md`
- `docs/vulkan/ROADMAP.md`
- `docs/vulkan/REVIEW_CHECKLIST.md`
- `docs/vulkan/TEMPORARY_EXCEPTIONS.md`
- Task-specific artifacts:

Contract direction:

- Coverage model(s):
- Proposed contract:
- Exact tuple/envelope metadata, if any:
  - `contract_name`:
  - `family_name`:
  - `tuple_id`:
  - `evidence_id`:
  - `guard_id`:
  - `fallback_policy`:
  - `materialization_policy`:
- Positive evidence:
- Negative/guard evidence:
- Temporary exception, if any:

Stop conditions:

- Stop after finding a new blocker unless this card explicitly authorizes
  fixing it.
- Stop before broadening route legality beyond the named contract.
- Stop before changing benchmarks unless this card explicitly authorizes it.

Validation:

- `git diff --check`
- static scan/diff review confirming no unintended shape or envelope changes:
- focused tests:
- `git status --short --untracked-files=all`
- If docs are the only changes, no build is needed.

Final report:

- files created/updated
- key decisions captured
- validation result
- git status
