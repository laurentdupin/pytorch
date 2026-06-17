from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_SPEC_DIR = REPO_ROOT / "test" / "vulkan_contract_specs"
GENERATED_CPP_MANIFEST_FILE = "generated_cpp_manifest.json"

ROUTE_CONTRACTS = {
    "token_prefix_cat_add": "TokenPrefixCatAddContract",
    "vision_self_attention_sdpa_calls": "VisionSelfAttentionSDPAContract",
    "vision_self_attention_sdpa_buffer_math_ops": "VisionSelfAttentionSDPAContract",
    "pointwise_depth_vision_projection": "SmallSpatialPointwiseConvContract",
    "softmax_buffer_lastdim": "SDPAScoreSoftmaxContract",
}

CONTRACT_MISSING_BUCKETS = {
    "required_final_readback": "FinalReadbackContract",
    "required_host_upload": "HostUploadTransitionContract",
    "unexpected_intermediate_readback": "IntermediateReadbackTransitionContract",
    "required_contiguous_materialization": "SafeContiguousMaterializationContract",
    "required_layout_repack": "LayoutRepackTransitionContract",
    "fallback_materialization": "FallbackMaterializationContract",
    "metadata_view_only": "MetadataViewTransitionContract",
    "required_semantic_cat": "CatMaterializationContract",
    "required_semantic_clone": "CloneSemanticMaterializationContract",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_transition_reason_bucket_contracts(
    spec_dir: Path = CONTRACT_SPEC_DIR,
) -> dict[str, str]:
    contracts: dict[str, str] = {}
    if not spec_dir.exists():
        return contracts
    for path in sorted(spec_dir.glob("*.json")):
        if path.name == GENERATED_CPP_MANIFEST_FILE:
            continue
        spec = load_json(path)
        transition_contract = spec.get("transition_contract")
        if not isinstance(transition_contract, dict):
            continue
        if not transition_contract.get("collector_reason_bucket"):
            continue
        if transition_contract.get("contract_type") != "LayoutTransitionContract":
            raise ValueError(f"{path.name} is not a LayoutTransitionContract")
        reason = transition_contract.get("reason")
        contract_name = spec.get("contract_name")
        if not reason or not contract_name:
            raise ValueError(f"{path.name} missing transition reason or contract_name")
        expected_contract = CONTRACT_MISSING_BUCKETS.get(reason)
        if expected_contract is not None and expected_contract != contract_name:
            raise ValueError(
                f"{path.name} maps {reason} to {contract_name}, "
                f"expected {expected_contract}"
            )
        previous = contracts.get(reason)
        if previous is not None and previous != contract_name:
            raise ValueError(
                f"duplicate transition reason bucket {reason}: "
                f"{previous} and {contract_name}"
            )
        contracts[reason] = contract_name
    return contracts


def repo_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def rel_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def status_from_result(result: dict[str, Any] | None, fallback: str = "skip") -> str:
    if not result:
        return fallback
    if result.get("status") in {"ok", "fail", "skip", "env_blocked", "oom"}:
        return result["status"]
    if result.get("performance_valid") is True or result.get("torch_vulkan_available") is True:
        return "ok"
    if result.get("error") or result.get("exception"):
        return "fail"
    return fallback


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def timing_summary(result: dict[str, Any] | None, matrix_row: dict[str, Any] | None) -> dict[str, Any]:
    if matrix_row and isinstance(matrix_row.get("timing"), dict):
        return matrix_row["timing"]
    if not result:
        return {}
    return {
        "device_resident": result.get("single_image_forward_device_resident"),
        "end_to_end": result.get("single_image_end_to_end"),
        "with_readback": result.get("single_image_forward_with_readback"),
    }


def normalize_counter_delta(delta: dict[str, Any]) -> dict[str, Any]:
    if not delta:
        return {}
    if "buffer_copy_count" in delta:
        return dict(delta)

    buffer_copy = delta.get("buffer_copy_counters") or []
    submit_origin = delta.get("submit_origin_counters") or []
    retire = delta.get("retire_drain_counters") or []
    normalized = {
        "cpu_fallback_count": int(delta.get("cpu_fallback_count") or 0),
        "sync_readback_count": int(delta.get("sync_readback_count") or 0),
    }
    if len(buffer_copy) >= 5:
        normalized.update(
            {
                "buffer_copy_count": int(buffer_copy[0]),
                "buffer_copy_bytes": int(buffer_copy[1]),
                "buffer_copy_explicit": int(buffer_copy[2]),
                "buffer_copy_contiguous": int(buffer_copy[3]),
                "buffer_copy_view_materialization": int(buffer_copy[4]),
            }
        )
    if len(submit_origin) >= 7:
        normalized.update(
            {
                "total_queue_submits": int(submit_origin[0]),
                "tensor_cpu_readback_submits": int(submit_origin[2]),
            }
        )
    if len(retire) >= 6:
        normalized.update(
            {
                "retire_drains": int(retire[0]),
                "retire_submit_drains": int(retire[1]),
                "retire_poll_drains": int(retire[3]),
                "retire_pending_resources": int(retire[4]),
                "retire_pending_bytes": int(retire[5]),
            }
        )
    return normalized


def phase_counters_from_result(
    result: dict[str, Any] | None,
    matrix_row: dict[str, Any] | None,
) -> dict[str, Any]:
    if matrix_row and isinstance(matrix_row.get("phases"), dict):
        return {name: dict(values) for name, values in matrix_row["phases"].items()}

    phases: dict[str, Any] = {}
    phase_blob = (result or {}).get("vulkan_phase_counters") or {}
    for phase in phase_blob.get("phases") or []:
        name = phase.get("name")
        if name:
            phases[name] = normalize_counter_delta(phase.get("delta") or {})
    if phase_blob.get("total"):
        phases["total"] = normalize_counter_delta(phase_blob["total"])
    return phases


def summarize_transitions(
    transition_jsonl: Path | None,
    missing_artifacts: list[dict[str, str]],
    reason_bucket_contracts: dict[str, str] | None = None,
) -> dict[str, Any]:
    if transition_jsonl is None or not transition_jsonl.exists():
        missing_artifacts.append(
            {
                "kind": "missing_artifact",
                "name": "transition_jsonl",
                "path": rel_path(transition_jsonl) or "",
                "impact": "transition event coverage is unavailable for this row",
            }
        )
        return {
            "events_by_reason_phase": [],
            "events_by_contract": {},
            "unknown_transition_reason_count": 0,
            "unknown_producer_consumer": {
                "unknown_producer_schema": 0,
                "unknown_consumer_schema": 0,
                "unknown_producer_contract": 0,
                "unknown_consumer_contract": 0,
            },
            "top_events": [],
        }

    reason_bucket_contracts = reason_bucket_contracts or {}
    aggregates: dict[tuple[Any, ...], dict[str, Any]] = {}
    by_contract: Counter[str] = Counter()
    unknown_transition_reasons = 0
    unknown_producer_consumer = Counter()

    with transition_jsonl.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            reason = event.get("reason") or "unknown_transition_reason"
            if reason == "unknown_transition_reason":
                unknown_transition_reasons += 1
            for field in (
                "producer_schema",
                "consumer_schema",
                "producer_contract",
                "consumer_contract",
            ):
                if event.get(field) in {None, "", "unknown"}:
                    unknown_producer_consumer[field] += 1
            explicit_contract_counted = False
            for field in ("producer_contract", "consumer_contract"):
                contract = event.get(field)
                if contract and contract != "unknown" and contract.endswith("Contract"):
                    by_contract[contract] += 1
                    explicit_contract_counted = True
            if not explicit_contract_counted:
                reason_contract = reason_bucket_contracts.get(reason)
                if reason_contract:
                    by_contract[reason_contract] += 1
            key = (
                event.get("phase") or "unknown",
                reason,
                event.get("kind") or "unknown",
                bool(event.get("physical_copy")),
                bool(event.get("host_transfer")),
                bool(event.get("sync_required")),
                bool(event.get("queue_submit_required")),
                event.get("producer_schema") or "unknown",
                event.get("consumer_schema") or "unknown",
                event.get("producer_contract") or "unknown",
                event.get("consumer_contract") or "unknown",
            )
            current = aggregates.setdefault(
                key,
                {
                    "phase": key[0],
                    "reason": key[1],
                    "kind": key[2],
                    "physical_copy": key[3],
                    "host_transfer": key[4],
                    "sync_required": key[5],
                    "queue_submit_required": key[6],
                    "producer_schema": key[7],
                    "consumer_schema": key[8],
                    "producer_contract": key[9],
                    "consumer_contract": key[10],
                    "count": 0,
                    "bytes": 0,
                },
            )
            current["count"] += 1
            if isinstance(event.get("bytes"), int) and event["bytes"] > 0:
                current["bytes"] += event["bytes"]

    events = sorted(
        aggregates.values(),
        key=lambda item: (-item["count"], -item["bytes"], item["phase"], item["reason"]),
    )
    return {
        "events_by_reason_phase": events,
        "events_by_contract": dict(sorted(by_contract.items())),
        "unknown_transition_reason_count": unknown_transition_reasons,
        "unknown_producer_consumer": dict(sorted(unknown_producer_consumer.items())),
        "top_events": events[:20],
    }


def summarize_region_lifetime(
    dry_run_summary: Path | None,
    missing_artifacts: list[dict[str, str]],
) -> dict[str, Any]:
    if dry_run_summary is None or not dry_run_summary.exists():
        missing_artifacts.append(
            {
                "kind": "missing_artifact",
                "name": "region_lifetime_dry_run",
                "path": rel_path(dry_run_summary) or "",
                "impact": "region/lifetime eligibility is unavailable for this row",
            }
        )
        return {
            "dry_run_enabled": False,
            "all_safe_group_eligible": 0,
            "would_remove_submit_drains": 0,
            "actual_removed_submit_drains": 0,
            "blockers_by_class": {},
            "blocker_bytes_by_class": {},
            "peak_extra_live_bytes_estimate": 0,
        }

    data = load_json(dry_run_summary)
    aggregate = data.get("aggregate") if isinstance(data, dict) else None
    source = aggregate if isinstance(aggregate, dict) else data
    return {
        "dry_run_enabled": True,
        "all_safe_group_eligible": int(source.get("all_safe_group_eligible") or 0),
        "would_remove_submit_drains": int(source.get("would_remove_submit_drains") or 0),
        "actual_removed_submit_drains": int(source.get("actual_removed_submit_drains") or 0),
        "blockers_by_class": source.get("blockers_by_class") or {},
        "blocker_bytes_by_class": source.get("blocker_bytes_by_class") or {},
        "peak_extra_live_bytes_estimate": int(
            source.get("peak_extra_live_bytes_estimate") or 0
        ),
    }


def route_contract_summary(matrix_row: dict[str, Any] | None) -> dict[str, Any]:
    route_counts = (matrix_row or {}).get("route_counts") or {}
    by_contract: Counter[str] = Counter()
    admissions: list[dict[str, Any]] = []
    for route, count in sorted(route_counts.items()):
        contract = ROUTE_CONTRACTS.get(route)
        if not contract or not count:
            continue
        by_contract[contract] += int(count)
        admissions.append(
            {
                "phase": "total",
                "op_schema": route,
                "contract_name": contract,
                "family": "route_counter",
                "tuple_id": "unknown",
                "route_label": route,
                "outcome": "admitted",
                "reject_reason": "",
                "shapes": {},
                "dtype": "unknown",
                "layout": "unknown",
                "count": int(count),
            }
        )
    return {
        "admissions": admissions,
        "rejections": [],
        "by_contract": dict(sorted(by_contract.items())),
        "uncontracted_ops": [],
    }


def budget_check(observed: int | float, budget: int | float | None = None) -> dict[str, Any]:
    status = "informational" if budget is None else ("pass" if observed <= budget else "fail")
    return {"observed": observed, "budget": budget, "status": status}


def make_budgets(phases: dict[str, Any]) -> dict[str, Any]:
    timed = phases.get("timed_forward") or phases.get("total") or {}
    return {
        "cpu_fallback": budget_check(int(timed.get("cpu_fallback_count") or 0), 0),
        "sync_readback": budget_check(int(timed.get("sync_readback_count") or 0), None),
        "host_transfer": budget_check(int(timed.get("tensor_cpu_readback_submits") or 0), None),
        "device_copy_bytes": budget_check(int(timed.get("buffer_copy_bytes") or 0), None),
        "queue_submits": budget_check(int(timed.get("total_queue_submits") or 0), None),
    }


def find_matrix_row(matrix: dict[str, Any] | None, match: dict[str, Any]) -> dict[str, Any] | None:
    if not matrix:
        return None
    for row in matrix.get("rows") or []:
        if all(row.get(key) == value for key, value in match.items()):
            return row
    return None


def build_row(
    row_cfg: dict[str, Any],
    reason_bucket_contracts: dict[str, str] | None = None,
) -> dict[str, Any]:
    missing_artifacts: list[dict[str, str]] = []
    result_path = repo_path(row_cfg.get("result_json"))
    result = load_json(result_path) if result_path and result_path.exists() else None
    if result_path and result is None:
        missing_artifacts.append(
            {
                "kind": "missing_artifact",
                "name": "result_json",
                "path": rel_path(result_path) or "",
                "impact": "row status and counters are partial",
            }
        )

    matrix_path = repo_path(row_cfg.get("matrix_json"))
    matrix = load_json(matrix_path) if matrix_path and matrix_path.exists() else None
    matrix_row = find_matrix_row(matrix, row_cfg.get("matrix_match") or {})
    if matrix_path and matrix is None:
        missing_artifacts.append(
            {
                "kind": "missing_artifact",
                "name": "matrix_json",
                "path": rel_path(matrix_path) or "",
                "impact": "summary row counters are unavailable",
            }
        )
    elif matrix_path and matrix_row is None:
        missing_artifacts.append(
            {
                "kind": "missing_artifact",
                "name": "matrix_row",
                "path": rel_path(matrix_path) or "",
                "impact": "configured matrix row was not found",
            }
        )

    phases = phase_counters_from_result(result, matrix_row)
    transitions = summarize_transitions(
        repo_path(row_cfg.get("transition_jsonl")),
        missing_artifacts,
        reason_bucket_contracts,
    )
    region_lifetime = summarize_region_lifetime(
        repo_path(row_cfg.get("region_lifetime_summary")),
        missing_artifacts,
    )
    op_contracts = route_contract_summary(matrix_row)
    status = status_from_result(
        matrix_row or result,
        row_cfg.get("status", "skip"),
    )
    timing_valid = bool(first_present(
        (matrix_row or {}).get("timing_valid"),
        (result or {}).get("performance_valid"),
        False,
    ))
    model = dict(row_cfg.get("model") or {})
    if result:
        model.setdefault("python_executable", result.get("python_executable"))
        model.setdefault("input_path", result.get("image") or result.get("input"))
        model.setdefault("model_checkpoint", result.get("checkpoint"))
        model.setdefault("variant", result.get("encoder"))
        if result.get("input_size") is not None:
            model.setdefault("resolution_or_shape", result.get("input_size"))
    elif matrix_row:
        model.setdefault("input_path", matrix_row.get("input"))
        model.setdefault("variant", matrix_row.get("model"))
        model.setdefault("resolution_or_shape", matrix_row.get("resolution"))

    blockers = []
    if status != "ok":
        blockers.append(
            {
                "kind": row_cfg.get("blocker_kind", "unknown"),
                "message": row_cfg.get("blocker_message", f"row status is {status}"),
                "original_vs_downstream": "not_probe",
                "details": row_cfg.get("blocker_details") or {},
            }
        )

    unknowns = {
        "unknown_transition_reasons": transitions["unknown_transition_reason_count"],
        "unknown_producer_consumer": sum(transitions["unknown_producer_consumer"].values()),
        "uncontracted_op_shapes": len(op_contracts["uncontracted_ops"]),
        "unclassified_lifetime_resources": sum(
            region_lifetime.get("blockers_by_class", {}).values()
        )
        if region_lifetime.get("dry_run_enabled")
        else 0,
    }
    artifacts = {
        "result_json": rel_path(result_path) if result_path and result_path.exists() else "",
        "matrix_json": rel_path(matrix_path) if matrix_path and matrix_path.exists() else "",
        "transition_jsonl": rel_path(repo_path(row_cfg.get("transition_jsonl")))
        if row_cfg.get("transition_jsonl") and repo_path(row_cfg.get("transition_jsonl")).exists()
        else "",
    }
    if missing_artifacts:
        artifacts["missing_artifacts"] = json.dumps(missing_artifacts, sort_keys=True)

    return {
        "row_id": row_cfg["row_id"],
        "model": model,
        "status": status,
        "timing_valid": timing_valid,
        "environment": {
            "missing_artifacts": missing_artifacts,
            "device_info": (result or {}).get("device_info") or {},
        },
        "timings": timing_summary(result, matrix_row),
        "phase_counters": phases,
        "op_contracts": op_contracts,
        "transitions": transitions,
        "region_lifetime": region_lifetime,
        "budgets": make_budgets(phases),
        "unknowns": unknowns,
        "blockers": blockers,
        "artifacts": artifacts,
    }


def aggregate_rows(
    rows: list[dict[str, Any]],
    reason_bucket_contracts: dict[str, str] | None = None,
) -> dict[str, Any]:
    reason_bucket_contracts = reason_bucket_contracts or {}
    rows_ok = sum(1 for row in rows if row["status"] == "ok")
    transition_unknown = sum(row["unknowns"]["unknown_transition_reasons"] for row in rows)
    transition_contracts = Counter()
    op_contracts = Counter()
    missing_artifacts = 0
    observed_reasons = Counter()
    for row in rows:
        missing_artifacts += len(row["environment"].get("missing_artifacts") or [])
        transition_contracts.update(row["transitions"].get("events_by_contract") or {})
        op_contracts.update(row["op_contracts"].get("by_contract") or {})
        for event in row["transitions"].get("events_by_reason_phase") or []:
            observed_reasons[event["reason"]] += int(event["count"])
    missing_contract_buckets = sorted(
        {
            CONTRACT_MISSING_BUCKETS[reason]
            for reason in observed_reasons
            if reason in CONTRACT_MISSING_BUCKETS
            and reason not in reason_bucket_contracts
        }
    )
    recommendations = []
    if missing_artifacts:
        recommendations.append("fill missing optional artifacts before treating coverage as complete")
    if missing_contract_buckets:
        recommendations.append(
            "consider transition specs for: " + ", ".join(missing_contract_buckets)
        )
    if not recommendations:
        recommendations.append("run the collector over the full five-model matrix")
    return {
        "rows_total": len(rows),
        "rows_ok": rows_ok,
        "rows_failed": len(rows) - rows_ok,
        "coverage": {
            "op_contracts_by_count": dict(sorted(op_contracts.items())),
            "transition_contracts_by_count": dict(sorted(transition_contracts.items())),
            "observed_transition_reasons": dict(sorted(observed_reasons.items())),
            "missing_contract_buckets": missing_contract_buckets,
            "missing_optional_artifacts": missing_artifacts,
            "unknown_transition_reasons": transition_unknown,
        },
        "budgets": {
            "rows_with_cpu_fallback_budget_failure": sum(
                1 for row in rows if row["budgets"]["cpu_fallback"]["status"] == "fail"
            )
        },
        "recommendations": recommendations,
    }


def validate_artifact_shape(artifact: dict[str, Any]) -> None:
    for key in ("schema_version", "run", "rows", "aggregate"):
        if key not in artifact:
            raise ValueError(f"missing top-level key: {key}")
    for row in artifact["rows"]:
        for key in (
            "row_id",
            "model",
            "status",
            "timing_valid",
            "budgets",
            "op_contracts",
            "transitions",
            "unknowns",
            "blockers",
        ):
            if key not in row:
                raise ValueError(f"{row.get('row_id', '<unknown>')} missing {key}")


def write_markdown(path: Path, artifact: dict[str, Any]) -> None:
    lines = [
        "# Five-Model Contract Validation Demo",
        "",
        f"- Head: `{artifact['run']['git_head']}`",
        f"- Mode: `{artifact['run']['mode']}`",
        f"- Rows: `{artifact['aggregate']['rows_total']}`",
        f"- Rows ok: `{artifact['aggregate']['rows_ok']}`",
        f"- Missing optional artifacts: `{artifact['aggregate']['coverage']['missing_optional_artifacts']}`",
        "",
        "## Rows",
        "",
        "| row | status | timing valid | op contracts | transition contracts | "
        "unknown transition reasons | missing artifacts |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for row in artifact["rows"]:
        op_contracts = ", ".join(
            f"{name}:{count}" for name, count in row["op_contracts"]["by_contract"].items()
        ) or "-"
        transition_contracts = ", ".join(
            f"{name}:{count}" for name, count in row["transitions"]["events_by_contract"].items()
        ) or "-"
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {} | {} |".format(
                row["row_id"],
                row["status"],
                "yes" if row["timing_valid"] else "no",
                op_contracts,
                transition_contracts,
                row["unknowns"]["unknown_transition_reasons"],
                len(row["environment"].get("missing_artifacts") or []),
            )
        )
    lines.extend(
        [
            "",
            "## Missing Contract Buckets",
            "",
        ]
    )
    buckets = artifact["aggregate"]["coverage"].get("missing_contract_buckets") or []
    if buckets:
        lines.extend(f"- `{bucket}`" for bucket in buckets)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Missing optional inputs are stored in each row under "
            "`environment.missing_artifacts` and mirrored as a JSON string in "
            "`artifacts.missing_artifacts` for tools that only read artifact "
            "paths.",
            "- Probe rows are supported by the schema, but the supplied config "
            "keeps probe mode disabled.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_transition_contract_classification() -> None:
    reason_bucket_contracts = load_transition_reason_bucket_contracts()
    required = {
        "fallback_materialization": "FallbackMaterializationContract",
        "required_final_readback": "FinalReadbackContract",
        "required_host_upload": "HostUploadTransitionContract",
        "required_layout_repack": "LayoutRepackTransitionContract",
        "unexpected_intermediate_readback": "IntermediateReadbackTransitionContract",
        "required_contiguous_materialization": "SafeContiguousMaterializationContract",
        "metadata_view_only": "MetadataViewTransitionContract",
    }
    for reason, contract_name in required.items():
        if reason_bucket_contracts.get(reason) != contract_name:
            raise AssertionError(f"{reason} is not covered by {contract_name}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        transition_log = Path(tmp_dir) / "transition.jsonl"
        events = [
            {
                "event": "vulkan_transition",
                "phase": "model_setup",
                "reason": "required_host_upload",
                "kind": "host_transfer",
                "outcome": "classified",
                "bytes": 4096,
                "host_transfer": True,
                "physical_copy": True,
                "sync_required": True,
                "queue_submit_required": True,
                "producer_schema": "cpu_tensor",
                "consumer_schema": "vulkan_tensor",
                "producer_contract": "unknown",
                "consumer_contract": "unknown",
            },
            {
                "event": "vulkan_transition",
                "phase": "layout_transition",
                "reason": "metadata_view_only",
                "kind": "metadata_view",
                "outcome": "classified",
                "bytes": 4096,
                "host_transfer": False,
                "physical_copy": False,
                "sync_required": False,
                "queue_submit_required": False,
                "producer_schema": "aten::view",
                "consumer_schema": "MetadataViewCreated",
                "producer_contract": "unknown",
                "consumer_contract": "unknown",
            },
            {
                "event": "vulkan_transition",
                "phase": "readback",
                "reason": "required_final_readback",
                "kind": "host_transfer",
                "outcome": "classified",
                "bytes": 4096,
                "host_transfer": True,
                "physical_copy": True,
                "sync_required": True,
                "queue_submit_required": True,
                "producer_schema": "vulkan_tensor",
                "consumer_schema": "cpu_tensor",
                "producer_contract": "unknown",
                "consumer_contract": "unknown",
            },
            {
                "event": "vulkan_transition",
                "phase": "model_setup",
                "reason": "unexpected_intermediate_readback",
                "kind": "host_transfer",
                "outcome": "classified",
                "bytes": 4096,
                "host_transfer": True,
                "physical_copy": True,
                "sync_required": True,
                "queue_submit_required": True,
                "producer_schema": "vulkan_tensor",
                "consumer_schema": "cpu_tensor",
                "producer_contract": "unknown",
                "consumer_contract": "unknown",
            },
            {
                "event": "vulkan_transition",
                "phase": "model_setup",
                "reason": "required_contiguous_materialization",
                "kind": "layout_materialization",
                "outcome": "classified",
                "bytes": 4096,
                "host_transfer": False,
                "physical_copy": True,
                "sync_required": False,
                "queue_submit_required": True,
                "producer_schema": "materialize_to_contiguous_buffer",
                "consumer_schema": "buffer_to_buffer",
                "producer_contract": "unknown",
                "consumer_contract": "unknown",
            },
            {
                "event": "vulkan_transition",
                "phase": "layout_transition",
                "reason": "required_layout_repack",
                "kind": "layout_materialization",
                "outcome": "classified",
                "bytes": 4096,
                "host_transfer": False,
                "physical_copy": True,
                "sync_required": False,
                "queue_submit_required": True,
                "producer_schema": "vulkan_tensor",
                "consumer_schema": "vulkan_tensor",
                "producer_contract": "unknown",
                "consumer_contract": "unknown",
            },
            {
                "event": "vulkan_transition",
                "phase": "owner_context_create",
                "reason": "fallback_materialization",
                "kind": "fallback",
                "outcome": "classified",
                "bytes": 4096,
                "host_transfer": True,
                "physical_copy": True,
                "sync_required": True,
                "queue_submit_required": True,
                "producer_schema": "vulkan_prepack::vision_context",
                "consumer_schema": "unpack_qkv_weight_readback",
                "producer_contract": "unknown",
                "consumer_contract": "unknown",
            },
        ]
        with transition_log.open("w", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event, sort_keys=True) + "\n")

        missing_artifacts: list[dict[str, str]] = []
        transitions = summarize_transitions(
            transition_log,
            missing_artifacts,
            reason_bucket_contracts,
        )
    if missing_artifacts:
        raise AssertionError(f"unexpected missing artifacts: {missing_artifacts}")
    expected_counts = {
        "FallbackMaterializationContract": 1,
        "FinalReadbackContract": 1,
        "HostUploadTransitionContract": 1,
        "IntermediateReadbackTransitionContract": 1,
        "LayoutRepackTransitionContract": 1,
        "MetadataViewTransitionContract": 1,
        "SafeContiguousMaterializationContract": 1,
    }
    for contract_name, expected_count in expected_counts.items():
        actual_count = transitions["events_by_contract"].get(contract_name)
        if actual_count != expected_count:
            raise AssertionError(
                f"{contract_name} count mismatch: {actual_count} != {expected_count}"
            )

    row = {
        "status": "ok",
        "environment": {"missing_artifacts": []},
        "transitions": transitions,
        "op_contracts": {"by_contract": {}},
        "unknowns": {"unknown_transition_reasons": 0},
        "budgets": {"cpu_fallback": {"status": "pass"}},
    }
    aggregate = aggregate_rows([row], reason_bucket_contracts)
    missing_buckets = aggregate["coverage"]["missing_contract_buckets"]
    for contract_name in expected_counts:
        if contract_name in missing_buckets:
            raise AssertionError(f"{contract_name} unexpectedly reported missing")
    if missing_buckets:
        raise AssertionError(f"unexpected missing buckets: {missing_buckets}")
    print(
        "validated transition contract classification "
        f"reason_buckets={json.dumps(required, sort_keys=True)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize Vulkan benchmark outputs into the five-model contract validation schema."
    )
    parser.add_argument("--config", help="Collector config JSON.")
    parser.add_argument("--schema", default="agent_space/five_model_contract_validation_schema.json")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    parser.add_argument("--rows", nargs="*", help="Optional row ids to include.")
    parser.add_argument(
        "--validate-transition-contract-classification",
        action="store_true",
        help="Validate reason-bucket transition contract classification.",
    )
    args = parser.parse_args()

    if args.validate_transition_contract_classification:
        validate_transition_contract_classification()
        return

    if not args.config or not args.output_json:
        parser.error("--config and --output-json are required unless validating")

    config_path = repo_path(args.config)
    config = load_json(config_path)
    reason_bucket_contracts = load_transition_reason_bucket_contracts()
    selected = set(args.rows or [])
    rows = []
    for row_cfg in config.get("rows", []):
        if selected and row_cfg["row_id"] not in selected:
            continue
        rows.append(build_row(row_cfg, reason_bucket_contracts))
    if not rows:
        raise ValueError("no rows selected")

    artifact = {
        "schema_version": 1,
        "run": {
            "git_head": git_head(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": config.get("mode", "normal_no_probe"),
            "adapter": config.get("adapter") or {},
            "inputs": [row["model"] for row in rows],
            "tool_versions": {"collector_config": rel_path(config_path)},
        },
        "rows": rows,
        "aggregate": aggregate_rows(rows, reason_bucket_contracts),
    }
    validate_artifact_shape(artifact)
    write_json(repo_path(args.output_json), artifact)
    if args.output_md:
        write_markdown(repo_path(args.output_md), artifact)


if __name__ == "__main__":
    main()
