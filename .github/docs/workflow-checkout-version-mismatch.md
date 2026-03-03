# GitHub Actions Workflow/Checkout Version Mismatch in PyTorch CI

## The Problem

When scripts or files referenced by GitHub Actions workflow YAML are moved on `main`, in-flight PRs that branched before the move break with `No such file or directory` errors — even though the PR didn't touch any workflow files.

This affects most PR-triggered workflows in `pytorch/pytorch`.

## Context: How `pull_request` Workflows Work

For `pull_request` events, GitHub creates a synthetic merge commit at `refs/pull/N/merge` — the result of merging the PR branch into the base branch. GitHub then:

1. Reads the workflow YAML from this merge commit
2. Resolves and inlines all reusable workflows (`uses: ./.github/workflows/...`) from the same commit
3. Sets `GITHUB_SHA` to this merge commit and `GITHUB_REF` to `refs/pull/N/merge`
4. `actions/checkout@v4` (with default settings) checks out this same merge commit

GitHub builds the workflow as a **self-contained execution plan** from the merge commit — the root workflow YAML and all reusable workflows are resolved and inlined *before any job starts*. But scripts, composite actions, and other files referenced by `run:` commands are **not** inlined — they are read from disk at runtime, from whatever `actions/checkout` put in the workspace. With default checkout, both come from the same commit, so everything is self-consistent.

## Root Cause

The workflow YAML is taken from the merge commit, but `checkout-pytorch` checks out the PR HEAD. When scripts are moved on `main`, the merge commit has the new paths in its workflow YAML while the PR HEAD still has the scripts at their old locations.

PyTorch's `checkout-pytorch` composite action (`.github/actions/checkout-pytorch/action.yml`, lines 59 and 107) overrides the default checkout to use `pull_request.head.sha` instead of the merge commit (a deliberate choice for deterministic CI — testing the exact PR HEAD rather than a synthetic merge that shifts as `main` advances). The workflow YAML being executed still comes from the merge commit.

So the workflow runs `bash .github/scripts/moved/foo.sh` (the path from the merged YAML), but the checkout has the file at its old location `.github/scripts/foo.sh`.

## When It Manifests

The mismatch only causes failures when **all three conditions** are true:

1. A file referenced by a workflow's `run:` command is moved/renamed on `main`
2. An open PR was branched before the move (or hasn't rebased since)
3. The checkout uses `ref: pull_request.head.sha` instead of the default merge commit

In effect, GitHub inlines the root workflow YAML and all reusable workflows into a single execution plan at planning time — everything in that plan (inline `run:` commands, step names, env vars) is fixed from the merge commit. Everything else — composite actions, scripts, and other files those commands reference — is resolved from the checked-out workspace at runtime.

ciflow-triggered workflows are not affected — they use `push` events on tags, so the workflow YAML and checkout both resolve to the same commit with no merge commit involved.

