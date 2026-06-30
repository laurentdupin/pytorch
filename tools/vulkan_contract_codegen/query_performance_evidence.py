#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST_RELATIVE = Path(
    "test/vulkan_contract_proofs/performance_plan_evidence_manifest.json"
)


def _flatten_text(value: Any) -> list[str]:
    if isinstance(value, dict):
        flattened: list[str] = []
        for key, nested in value.items():
            flattened.append(str(key))
            flattened.extend(_flatten_text(nested))
        return flattened
    if isinstance(value, list):
        flattened = []
        for nested in value:
            flattened.extend(_flatten_text(nested))
        return flattened
    if value is None:
        return []
    return [str(value)]


def _entry_search_text(entry: dict[str, Any]) -> str:
    return "\n".join(_flatten_text(entry)).casefold()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != "VulkanPerformancePlanEvidenceManifest.v0":
        raise RuntimeError(f"unexpected performance evidence schema in {path}")
    return manifest


def query_manifest_entries(
    manifest: dict[str, Any],
    *,
    terms: list[str] | None = None,
    statuses: list[str] | None = None,
    model: str | None = None,
    variant: str | None = None,
    input_shape: str | None = None,
    scope: str | None = None,
) -> list[dict[str, Any]]:
    terms = [term.casefold() for term in (terms or []) if term]
    statuses = [status.casefold() for status in (statuses or []) if status]
    model_query = model.casefold() if model else None
    variant_query = variant.casefold() if variant else None
    input_query = input_shape.casefold() if input_shape else None
    scope_query = scope.casefold() if scope else None
    matches: list[dict[str, Any]] = []
    for entry in manifest.get("entries", []):
        if not isinstance(entry, dict):
            continue
        provenance = entry.get("model_provenance", {})
        if statuses and str(entry.get("status", "")).casefold() not in statuses:
            continue
        if model_query and model_query not in str(
            provenance.get("model", "")
        ).casefold():
            continue
        if variant_query and variant_query not in str(
            provenance.get("variant", "")
        ).casefold():
            continue
        if input_query and input_query not in str(
            provenance.get("input", "")
        ).casefold():
            continue
        if scope_query:
            scope_text = "\n".join(
                str(item) for item in entry.get("contract_or_topology_scope", [])
            ).casefold()
            if scope_query not in scope_text:
                continue
        search_text = _entry_search_text(entry)
        if any(term not in search_text for term in terms):
            continue
        matches.append(entry)
    return matches


def summarize_artifact_segment_plan(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        artifact = json.load(handle)
    segment_plan = artifact.get("vulkan_stack_region_segment_plan")
    if not isinstance(segment_plan, dict):
        return {
            "path": str(path),
            "available": False,
            "reason": "missing_vulkan_stack_region_segment_plan",
        }
    keys = (
        "available",
        "contract_name",
        "row_count",
        "observed_row_count",
        "accepted_row_count",
        "rejected_row_count",
        "status_counts",
        "fail_reason_counts",
        "owned_command_buffer_mode_counts",
        "segment_planned_dispatch_limit_counts",
        "max_planned_dispatch_count",
        "max_segment_planned_dispatch_count",
    )
    summary = {"path": str(path)}
    for key in keys:
        if key in segment_plan:
            summary[key] = segment_plan[key]
    return summary


def _entry_summary(entry: dict[str, Any]) -> dict[str, Any]:
    provenance = entry.get("model_provenance", {})
    return {
        "id": entry.get("id"),
        "status": entry.get("status"),
        "head": entry.get("head"),
        "model": provenance.get("model"),
        "variant": provenance.get("variant"),
        "input": provenance.get("input"),
        "scope": entry.get("contract_or_topology_scope", []),
        "candidate": entry.get("candidate"),
        "decision": entry.get("decision"),
        "revisit_conditions": entry.get("revisit_conditions", []),
    }


def build_query_result(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    matches: list[dict[str, Any]],
    artifact_paths: list[Path],
) -> dict[str, Any]:
    return {
        "manifest": str(manifest_path),
        "schema": manifest.get("schema"),
        "match_count": len(matches),
        "matches": [_entry_summary(entry) for entry in matches],
        "artifact_segment_plans": [
            summarize_artifact_segment_plan(path) for path in artifact_paths
        ],
    }


def format_text_result(result: dict[str, Any]) -> str:
    lines = [
        f"manifest: {result['manifest']}",
        f"matches: {result['match_count']}",
    ]
    for match in result["matches"]:
        lines.append(f"- {match['id']} [{match['status']}]")
        lines.append(
            "  model: "
            f"{match.get('model')} {match.get('variant')} {match.get('input')}"
        )
        lines.append(f"  scope: {', '.join(match.get('scope') or [])}")
        lines.append(f"  candidate: {match.get('candidate')}")
        lines.append(f"  decision: {match.get('decision')}")
        revisit = match.get("revisit_conditions") or []
        if revisit:
            lines.append(f"  revisit: {revisit[0]}")
    artifact_plans = result.get("artifact_segment_plans") or []
    if artifact_plans:
        lines.append("artifact segment plans:")
    for plan in artifact_plans:
        lines.append(f"- {plan['path']}")
        if not plan.get("available"):
            lines.append(f"  unavailable: {plan.get('reason', 'unknown')}")
            continue
        for key in (
            "row_count",
            "accepted_row_count",
            "rejected_row_count",
            "owned_command_buffer_mode_counts",
            "status_counts",
            "fail_reason_counts",
        ):
            if key in plan:
                lines.append(f"  {key}: {plan[key]}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search Vulkan performance-plan evidence before rerunning diagnostics."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--status", action="append", default=[])
    parser.add_argument("--model")
    parser.add_argument("--variant")
    parser.add_argument("--input", dest="input_shape")
    parser.add_argument("--scope")
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-match", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest or args.repo_root / DEFAULT_MANIFEST_RELATIVE
    manifest = load_manifest(manifest_path)
    matches = query_manifest_entries(
        manifest,
        terms=args.query,
        statuses=args.status,
        model=args.model,
        variant=args.variant,
        input_shape=args.input_shape,
        scope=args.scope,
    )
    result = build_query_result(
        manifest_path=manifest_path,
        manifest=manifest,
        matches=matches,
        artifact_paths=args.artifact,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(format_text_result(result))
    if args.require_match and not matches:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
