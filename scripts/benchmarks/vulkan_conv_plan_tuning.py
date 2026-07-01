#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "VulkanConvPlanTuningResult.v0"
TIMESTAMP_SCHEMA = "VulkanConvPlanTimestampSummary.v0"
PLAN_KEY_SCHEMA = "VulkanConvPlanKey.v0"
VALID_DECISIONS = {
    "accepted",
    "rejected_mixed",
    "rejected_slower",
    "correctness_blocked",
}

REQUIRED_SHAPE_FIELDS = (
    "input",
    "output_channels",
    "weight",
    "stride",
    "padding",
    "dilation",
    "groups",
)
REQUIRED_LAYOUT_FIELDS = (
    "input_dtype",
    "weight_dtype",
    "output_dtype",
    "input_storage",
    "weight_storage",
    "output_storage",
    "input_layout",
    "weight_layout",
    "output_layout",
    "input_direct",
    "output_direct",
    "weight_packed",
    "bias",
    "pointwise",
    "depthwise",
    "sliding_window",
    "input_offset",
    "weight_offset",
    "output_offset",
)
REQUIRED_CAPABILITY_PROFILE_FIELDS = (
    "context_device_index",
    "vendor_id",
    "device_id",
    "driver_version",
    "api_version",
    "subgroup_size",
    "min_subgroup_size",
    "max_subgroup_size",
    "max_compute_workgroup_subgroups",
    "has_subgroup_size_control",
    "has_compute_full_subgroups",
    "has_cooperative_matrix",
    "cooperative_matrix_property_count",
    "has_timeline_semaphore",
    "has_synchronization2",
)
CONV_PLAN_TIMESTAMP_GROUP_FIELDS = (
    "kernel",
    "input",
    "output_channels",
    "weight",
    "stride",
    "padding",
    "dilation",
    "groups",
    "global",
    "local",
)


def parse_plan_key_snapshot_row(row: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in row.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def parse_conv_plan_runtime_label(label: str) -> dict[str, str] | None:
    parts = label.split("|")
    if not parts or parts[0] != "conv_plan":
        return None
    fields: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    missing = [
        field for field in CONV_PLAN_TIMESTAMP_GROUP_FIELDS if field not in fields
    ]
    if missing:
        raise ValueError(
            "conv_plan timestamp label missing fields: " + ",".join(missing)
        )
    return fields


def parse_conv_plan_timestamp_log_line(line: str) -> dict[str, Any] | None:
    fields = parse_plan_key_snapshot_row(line)
    runtime = fields.get("runtime")
    if runtime is None:
        return None
    label_fields = parse_conv_plan_runtime_label(runtime)
    if label_fields is None:
        return None
    try:
        duration_ns = int(fields["duration_ns"])
    except KeyError as exc:
        raise ValueError("conv_plan timestamp row missing duration_ns") from exc
    except ValueError as exc:
        raise ValueError(
            f"conv_plan timestamp row has invalid duration_ns={fields.get('duration_ns')}"
        ) from exc
    if duration_ns < 0:
        raise ValueError("conv_plan timestamp row has negative duration_ns")
    return {
        "fields": label_fields,
        "duration_ns": duration_ns,
    }


def plan_key_from_snapshot(
    fields: dict[str, str],
    *,
    candidate: str,
) -> dict[str, Any]:
    return {
        "schema": fields.get("schema"),
        "selected": fields.get("selected"),
        "reject": fields.get("reject"),
        "kernel": fields.get("kernel"),
        "role": fields.get("role"),
        "contract": fields.get("contract"),
        "contract_family": fields.get("contract_family"),
        "contract_tuple": fields.get("contract_tuple"),
        "shape": {field: fields.get(field) for field in REQUIRED_SHAPE_FIELDS},
        "layout": {field: fields.get(field) for field in REQUIRED_LAYOUT_FIELDS},
        "global": fields.get("global"),
        "local": fields.get("local"),
        "candidate": candidate,
        "candidate_count": fields.get("candidate_count"),
        "cacheable": fields.get("cacheable"),
        "tunable": fields.get("tunable"),
    }


def capability_profile_from_snapshot(fields: dict[str, str]) -> dict[str, str]:
    return {
        field: fields.get(field, "not_available")
        for field in REQUIRED_CAPABILITY_PROFILE_FIELDS
    }


def validate_tuning_result(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if payload.get("runtime_defaults_changed") is not False:
        errors.append("runtime_defaults_changed must be false")

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        errors.append("results must be a non-empty list")
        return errors
    if payload.get("result_count") != len(results):
        errors.append("result_count must match len(results)")

    seen_ids: set[str] = set()
    for index, result in enumerate(results):
        prefix = f"results[{index}]"
        result_id = result.get("id")
        if not isinstance(result_id, str) or not result_id:
            errors.append(f"{prefix}.id must be a non-empty string")
        elif result_id in seen_ids:
            errors.append(f"{prefix}.id is duplicated")
        else:
            seen_ids.add(result_id)

        decision = result.get("decision")
        if decision not in VALID_DECISIONS:
            errors.append(
                f"{prefix}.decision must be one of {sorted(VALID_DECISIONS)}"
            )

        plan_key = result.get("plan_key")
        if not isinstance(plan_key, dict):
            errors.append(f"{prefix}.plan_key must be an object")
        else:
            if plan_key.get("schema") != PLAN_KEY_SCHEMA:
                errors.append(f"{prefix}.plan_key.schema must be {PLAN_KEY_SCHEMA}")
            for field in ("contract", "kernel", "local", "candidate"):
                if not plan_key.get(field):
                    errors.append(f"{prefix}.plan_key.{field} is required")
            for group, required_fields in (
                ("shape", REQUIRED_SHAPE_FIELDS),
                ("layout", REQUIRED_LAYOUT_FIELDS),
            ):
                value = plan_key.get(group)
                if not isinstance(value, dict):
                    errors.append(f"{prefix}.plan_key.{group} must be an object")
                    continue
                for field in required_fields:
                    if value.get(field) in (None, ""):
                        errors.append(f"{prefix}.plan_key.{group}.{field} is required")

        capability_profile = result.get("capability_profile")
        if not isinstance(capability_profile, dict):
            errors.append(f"{prefix}.capability_profile must be an object")
        else:
            for forbidden in ("device_name", "name"):
                if forbidden in capability_profile:
                    errors.append(
                        f"{prefix}.capability_profile must not key by {forbidden}"
                    )
            for field in REQUIRED_CAPABILITY_PROFILE_FIELDS:
                if capability_profile.get(field) in (None, ""):
                    errors.append(f"{prefix}.capability_profile.{field} is required")

        evidence = result.get("evidence")
        if not isinstance(evidence, dict):
            errors.append(f"{prefix}.evidence must be an object")

        revisit_conditions = result.get("revisit_conditions")
        if not isinstance(revisit_conditions, list) or not revisit_conditions:
            errors.append(f"{prefix}.revisit_conditions must be a non-empty list")
        elif not all(isinstance(item, str) and item for item in revisit_conditions):
            errors.append(f"{prefix}.revisit_conditions entries must be strings")

    return errors


def validate_timestamp_summary(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != TIMESTAMP_SCHEMA:
        errors.append(f"schema must be {TIMESTAMP_SCHEMA}")
    if not payload.get("source_log"):
        errors.append("source_log is required")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        errors.append("rows must be a list")
        return errors
    if payload.get("row_count") != len(rows):
        errors.append("row_count must match len(rows)")
    total_duration = 0
    total_count = 0
    seen_keys: set[tuple[str, ...]] = set()
    for index, row in enumerate(rows):
        prefix = f"rows[{index}]"
        key = []
        for field in CONV_PLAN_TIMESTAMP_GROUP_FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or not value:
                errors.append(f"{prefix}.{field} is required")
                value = ""
            key.append(value)
        key_tuple = tuple(key)
        if key_tuple in seen_keys:
            errors.append(f"{prefix} duplicates a normalized conv-plan label")
        seen_keys.add(key_tuple)
        count = row.get("count")
        duration_sum = row.get("duration_ns_sum")
        duration_mean = row.get("duration_ns_mean")
        duration_max = row.get("duration_ns_max")
        if not isinstance(count, int) or count <= 0:
            errors.append(f"{prefix}.count must be a positive integer")
            count = 0
        if not isinstance(duration_sum, int) or duration_sum < 0:
            errors.append(f"{prefix}.duration_ns_sum must be a non-negative integer")
            duration_sum = 0
        if not isinstance(duration_max, int) or duration_max < 0:
            errors.append(f"{prefix}.duration_ns_max must be a non-negative integer")
        if not isinstance(duration_mean, (int, float)) or duration_mean < 0:
            errors.append(f"{prefix}.duration_ns_mean must be non-negative")
        elif count and abs(float(duration_mean) - (duration_sum / count)) > 1.0e-6:
            errors.append(f"{prefix}.duration_ns_mean does not match sum/count")
        total_count += count
        total_duration += duration_sum
    if payload.get("conv_plan_event_count") != total_count:
        errors.append("conv_plan_event_count must match row counts")
    if payload.get("total_conv_plan_duration_ns") != total_duration:
        errors.append("total_conv_plan_duration_ns must match row sums")
    return errors


def validate_payload(payload: dict[str, Any]) -> list[str]:
    schema = payload.get("schema")
    if schema == SCHEMA:
        return validate_tuning_result(payload)
    if schema == TIMESTAMP_SCHEMA:
        return validate_timestamp_summary(payload)
    return [f"schema must be {SCHEMA} or {TIMESTAMP_SCHEMA}"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _decision_from_sweep(candidate_decision: dict[str, Any]) -> str:
    if not candidate_decision.get("clean_correctness_and_expected_workgroup"):
        return "correctness_blocked"
    improved = int(candidate_decision.get("improved_row_count", 0))
    regressed = int(candidate_decision.get("regressed_row_count", 0))
    if improved > 0 and regressed > 0:
        return "rejected_mixed"
    if regressed > 0:
        return "rejected_slower"
    return "accepted"


def _find_plan_key_fields(
    row_json: Path,
    *,
    kernel: str,
    expected_local: list[int] | None,
) -> dict[str, str]:
    matches = _find_plan_key_field_rows(
        row_json,
        kernel=kernel,
        expected_local=expected_local,
    )
    if matches:
        return matches[0]
    raise ValueError(f"no {PLAN_KEY_SCHEMA} row for kernel={kernel} in {row_json}")


def _find_plan_key_field_rows(
    row_json: Path,
    *,
    kernel: str,
    expected_local: list[int] | None,
) -> list[dict[str, str]]:
    row = load_json(row_json)
    snapshot = row.get("vulkan_debug_counters", {}).get("conv_plan_key_snapshot", [])
    expected_local_text = (
        "[" + ",".join(str(value) for value in expected_local) + "]"
        if expected_local
        else None
    )
    fallback: list[dict[str, str]] = []
    matches: list[dict[str, str]] = []
    for line in snapshot:
        if not isinstance(line, str) or PLAN_KEY_SCHEMA not in line:
            continue
        fields = parse_plan_key_snapshot_row(line)
        if fields.get("kernel") != kernel:
            continue
        fallback.append(fields)
        if expected_local_text is None or fields.get("local") == expected_local_text:
            matches.append(fields)
    return matches if matches else fallback


def _row_delta_mean(row: dict[str, Any]) -> float | None:
    delta = row.get("delta_vs_default", {}).get("mean_ms")
    if isinstance(delta, (int, float)):
        return float(delta)
    return None


def _row_decision(row: dict[str, Any]) -> str:
    if row.get("correctness", {}).get("bridge_sanity_passed") is False:
        return "correctness_blocked"
    if not all(row.get("kernel_expected_local_ok", {}).values()):
        return "correctness_blocked"
    delta = _row_delta_mean(row)
    if delta is not None and delta < 0:
        return "accepted"
    return "rejected_slower"


def _combine_row_decisions(decisions: list[str]) -> str:
    if any(decision == "correctness_blocked" for decision in decisions):
        return "correctness_blocked"
    accepted = any(decision == "accepted" for decision in decisions)
    rejected = any(decision == "rejected_slower" for decision in decisions)
    if accepted and rejected:
        return "rejected_mixed"
    if accepted:
        return "accepted"
    return "rejected_slower"


def _plan_key_group_id(
    *,
    kernel: str,
    candidate: str,
    fields: dict[str, str],
    capability_profile: dict[str, str],
) -> str:
    return "|".join(
        [
            kernel,
            candidate,
            f"vendor={capability_profile.get('vendor_id', 'not_available')}",
            f"device={capability_profile.get('device_id', 'not_available')}",
            f"driver={capability_profile.get('driver_version', 'not_available')}",
            f"input={fields.get('input', 'not_available')}",
            f"weight={fields.get('weight', 'not_available')}",
            f"out={fields.get('output_channels', 'not_available')}",
            f"local={fields.get('local', 'not_available')}",
        ]
    )


def build_result_from_sweep_summary(
    summary_path: Path,
    *,
    kernel: str,
    candidates: set[str] | None,
    granularity: str = "candidate",
) -> dict[str, Any]:
    summary = load_json(summary_path)
    summary_dir = summary_path.parent
    plan_decisions = summary.get("plan_decisions", {})
    rows = summary.get("rows", [])

    if granularity == "plan-key":
        return build_plan_key_result_from_sweep_summary(
            summary,
            summary_path,
            summary_dir,
            kernel=kernel,
            candidates=candidates,
        )

    results: list[dict[str, Any]] = []
    for candidate, candidate_decision in sorted(plan_decisions.items()):
        if candidates is not None and candidate not in candidates:
            continue
        source_rows = [
            row for row in rows if row.get("plan") == candidate and row.get("row_json")
        ]
        if not source_rows:
            raise ValueError(f"candidate {candidate} has no source row_json")
        source_row = source_rows[0]
        row_json = Path(source_row["row_json"])
        if not row_json.is_absolute():
            row_json = summary_dir / row_json.name
        fields = _find_plan_key_fields(
            row_json,
            kernel=kernel,
            expected_local=source_row.get("expected_local"),
        )
        decision = _decision_from_sweep(candidate_decision)
        result_id = f"{kernel}:{candidate}:{decision}"
        results.append(
            {
                "id": result_id,
                "decision": decision,
                "plan_key": plan_key_from_snapshot(fields, candidate=candidate),
                "capability_profile": capability_profile_from_snapshot(fields),
                "evidence": {
                    "source_summary": str(summary_path),
                    "source_head": summary.get("head"),
                    "source_rows": [row.get("label") for row in source_rows],
                    "improved_rows": candidate_decision.get("improved_rows", []),
                    "regressed_rows": candidate_decision.get("regressed_rows", []),
                    "clean_correctness_and_expected_workgroup": candidate_decision.get(
                        "clean_correctness_and_expected_workgroup"
                    ),
                    "source_decision": candidate_decision.get("decision"),
                    "source_reason": candidate_decision.get("reason"),
                },
                "revisit_conditions": [
                    "Re-evaluate when VulkanConvPlanKey.v0 capability fields change.",
                    "Re-evaluate when a new benchmark rowset changes correctness or timing status.",
                    "Do not change runtime defaults from this offline artifact alone.",
                ],
            }
        )

    payload = {
        "schema": SCHEMA,
        "source_kind": "benchmark_sweep_summary",
        "source": str(summary_path),
        "granularity": granularity,
        "runtime_defaults_changed": False,
        "result_count": len(results),
        "results": results,
    }
    errors = validate_tuning_result(payload)
    if errors:
        raise ValueError("invalid tuning result:\n" + "\n".join(errors))
    return payload


def build_plan_key_result_from_sweep_summary(
    summary: dict[str, Any],
    summary_path: Path,
    summary_dir: Path,
    *,
    kernel: str,
    candidates: set[str] | None,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for row in summary.get("rows", []):
        candidate = row.get("plan")
        if not candidate or candidate == "default":
            continue
        if candidates is not None and candidate not in candidates:
            continue
        row_json_value = row.get("row_json")
        if not row_json_value:
            continue
        row_json = Path(row_json_value)
        if not row_json.is_absolute():
            row_json = summary_dir / row_json.name
        fields_rows = _find_plan_key_field_rows(
            row_json,
            kernel=kernel,
            expected_local=row.get("expected_local"),
        )
        if not fields_rows:
            continue
        row_decision = _row_decision(row)
        emitted_group_ids: set[str] = set()
        for fields in fields_rows:
            capability_profile = capability_profile_from_snapshot(fields)
            group_id = _plan_key_group_id(
                kernel=kernel,
                candidate=candidate,
                fields=fields,
                capability_profile=capability_profile,
            )
            if group_id in emitted_group_ids:
                continue
            emitted_group_ids.add(group_id)
            group = groups.setdefault(
                group_id,
                {
                    "id": group_id,
                    "plan_key": plan_key_from_snapshot(fields, candidate=candidate),
                    "capability_profile": capability_profile,
                    "row_decisions": [],
                    "source_rows": [],
                    "row_evidence": [],
                },
            )
            group["row_decisions"].append(row_decision)
            group["source_rows"].append(row.get("label"))
            group["row_evidence"].append(
                {
                    "source_row": row.get("label"),
                    "device_index": row.get("device_index"),
                    "model": row.get("model"),
                    "input_size": row.get("input_size"),
                    "delta_mean_ms": _row_delta_mean(row),
                    "mean_ms": row.get("timing_device_resident_ms", {}).get("mean"),
                    "correctness": row.get("correctness", {}),
                    "cpu_fallback": row.get("cpu_fallback"),
                    "sync_readback": row.get("sync_readback"),
                    "expected_workgroup_observed": all(
                        row.get("kernel_expected_local_ok", {}).values()
                    ),
                }
            )

    results = []
    for group_id, group in sorted(groups.items()):
        decision = _combine_row_decisions(group["row_decisions"])
        results.append(
            {
                "id": f"{group_id}|decision={decision}",
                "decision": decision,
                "plan_key": group["plan_key"],
                "capability_profile": group["capability_profile"],
                "evidence": {
                    "source_summary": str(summary_path),
                    "source_head": summary.get("head"),
                    "source_rows": group["source_rows"],
                    "row_evidence": group["row_evidence"],
                },
                "revisit_conditions": [
                    "Re-evaluate when VulkanConvPlanKey.v0 capability fields change.",
                    "Re-evaluate when this exact plan key has stable per-kernel timing.",
                    "Do not change runtime defaults from this offline artifact alone.",
                ],
            }
        )

    payload = {
        "schema": SCHEMA,
        "source_kind": "benchmark_sweep_summary",
        "source": str(summary_path),
        "granularity": "plan-key",
        "runtime_defaults_changed": False,
        "result_count": len(results),
        "results": results,
    }
    errors = validate_tuning_result(payload)
    if errors:
        raise ValueError("invalid tuning result:\n" + "\n".join(errors))
    return payload


def build_timestamp_summary_from_log(log_path: Path) -> dict[str, Any]:
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    ignored_line_count = 0
    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = parse_conv_plan_timestamp_log_line(line)
            except ValueError as exc:
                raise ValueError(f"{log_path}:{line_number}: {exc}") from exc
            if parsed is None:
                ignored_line_count += 1
                continue
            fields = parsed["fields"]
            key = tuple(fields[field] for field in CONV_PLAN_TIMESTAMP_GROUP_FIELDS)
            group = groups.setdefault(
                key,
                {
                    **{field: fields[field] for field in CONV_PLAN_TIMESTAMP_GROUP_FIELDS},
                    "count": 0,
                    "duration_ns_sum": 0,
                    "duration_ns_max": 0,
                },
            )
            duration_ns = parsed["duration_ns"]
            group["count"] += 1
            group["duration_ns_sum"] += duration_ns
            group["duration_ns_max"] = max(group["duration_ns_max"], duration_ns)

    rows = []
    total_duration = 0
    total_count = 0
    for key in sorted(groups):
        row = groups[key]
        row["duration_ns_mean"] = row["duration_ns_sum"] / row["count"]
        total_duration += row["duration_ns_sum"]
        total_count += row["count"]
        rows.append(row)

    payload = {
        "schema": TIMESTAMP_SCHEMA,
        "source_log": str(log_path),
        "conv_plan_event_count": total_count,
        "ignored_line_count": ignored_line_count,
        "total_conv_plan_duration_ns": total_duration,
        "row_count": len(rows),
        "rows": rows,
    }
    errors = validate_timestamp_summary(payload)
    if errors:
        raise ValueError("invalid timestamp summary:\n" + "\n".join(errors))
    return payload


def cmd_validate(args: argparse.Namespace) -> int:
    payload = load_json(Path(args.path))
    errors = validate_payload(payload)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"{args.path}: ok")
    return 0


def cmd_from_sweep(args: argparse.Namespace) -> int:
    candidates = set(args.candidate) if args.candidate else None
    payload = build_result_from_sweep_summary(
        Path(args.summary),
        kernel=args.kernel,
        candidates=candidates,
        granularity=args.granularity,
    )
    write_json(Path(args.out), payload)
    print(f"wrote {args.out}")
    return 0


def cmd_from_timestamp_log(args: argparse.Namespace) -> int:
    payload = build_timestamp_summary_from_log(Path(args.log))
    write_json(Path(args.out), payload)
    print(f"wrote {args.out}")
    return 0


def cmd_self_test(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        row_path = root / "row.json"
        write_json(
            row_path,
            {
                "vulkan_debug_counters": {
                    "conv_plan_key_snapshot": [
                        (
                            "schema=VulkanConvPlanKey.v0 "
                            "selected=FloatBufferConv reject=None "
                            "kernel=conv2d_buffer_float_3x3_s1p1 "
                            "role=other_3x3_s1p1 contract=none "
                            "contract_family=none contract_tuple=none "
                            "input=[1,64,140,210] output_channels=32 "
                            "weight=[32,64,3,3] stride=[1,1] "
                            "padding=[1,1] dilation=[1,1] groups=1 "
                            "input_dtype=6 weight_dtype=6 output_dtype=6 "
                            "input_storage=1 weight_storage=1 output_storage=1 "
                            "input_layout=2 weight_layout=2 output_layout=2 "
                            "input_direct=1 output_direct=1 weight_packed=1 "
                            "bias=1 pointwise=0 depthwise=0 sliding_window=1 "
                            "input_offset=0 weight_offset=0 output_offset=0 "
                            "global=[32,29400,1] local=[16,4,1] "
                            "candidate_count=3 context_device_index=0 "
                            "vendor_id=4098 device_id=29631 "
                            "driver_version=252313600 api_version=4206831 "
                            "subgroup_size=64 min_subgroup_size=32 "
                            "max_subgroup_size=64 "
                            "max_compute_workgroup_subgroups=8 "
                            "has_subgroup_size_control=1 "
                            "has_compute_full_subgroups=1 "
                            "has_cooperative_matrix=0 "
                            "cooperative_matrix_property_count=0 "
                            "has_timeline_semaphore=1 has_synchronization2=1 "
                            "cacheable=1 tunable=1"
                        )
                    ]
                }
            },
        )
        summary_path = root / "summary.json"
        write_json(
            summary_path,
            {
                "schema": "conv_workgroup_canary_sweep_summary.v1",
                "head": "self_test",
                "rows": [
                    {
                        "label": "self_test_candidate",
                        "plan": "3x3_s1p1_16x4",
                        "row_json": str(row_path),
                        "expected_local": [16, 4, 1],
                        "delta_vs_default": {"mean_ms": -1.0},
                        "correctness": {"bridge_sanity_passed": True},
                        "kernel_expected_local_ok": {
                            "conv2d_buffer_float_3x3_s1p1": True,
                        },
                    }
                ],
                "plan_decisions": {
                    "3x3_s1p1_16x4": {
                        "clean_correctness_and_expected_workgroup": True,
                        "improved_row_count": 1,
                        "regressed_row_count": 0,
                    }
                },
            },
        )
        payload = build_result_from_sweep_summary(
            summary_path,
            kernel="conv2d_buffer_float_3x3_s1p1",
            candidates=None,
            granularity="plan-key",
        )
        errors = validate_tuning_result(payload)
        if errors:
            raise AssertionError("\n".join(errors))
        if payload["results"][0]["decision"] != "accepted":
            raise AssertionError("self-test candidate should be accepted")
        if payload["results"][0]["capability_profile"]["vendor_id"] != "4098":
            raise AssertionError("capability profile was not captured")
        timestamp_log = root / "gpu_timestamps.log"
        timestamp_log.write_text(
            "\n".join(
                [
                    (
                        "gpu_timestamp reason=submit name=conv "
                        "runtime=conv_plan|kernel=conv2d_buffer_float_3x3_s1p1"
                        "|input=[1x64x140x210]|output_channels=32"
                        "|weight=[32x64x3x3]|stride=[1x1]|padding=[1x1]"
                        "|dilation=[1x1]|groups=1|global=210x140x32"
                        "|local=16x4x1 start_ns=10 end_ns=40 duration_ns=30 "
                        "global=[210,140,32] local=[16,4,1]"
                    ),
                    (
                        "gpu_timestamp reason=submit name=conv "
                        "runtime=conv_plan|kernel=conv2d_buffer_float_3x3_s1p1"
                        "|input=[1x64x140x210]|output_channels=32"
                        "|weight=[32x64x3x3]|stride=[1x1]|padding=[1x1]"
                        "|dilation=[1x1]|groups=1|global=210x140x32"
                        "|local=16x4x1 start_ns=50 end_ns=90 duration_ns=40 "
                        "global=[210,140,32] local=[16,4,1]"
                    ),
                    (
                        "gpu_timestamp reason=submit name=other "
                        "runtime=attention start_ns=100 end_ns=105 duration_ns=5"
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        timestamp_summary = build_timestamp_summary_from_log(timestamp_log)
        timestamp_errors = validate_timestamp_summary(timestamp_summary)
        if timestamp_errors:
            raise AssertionError("\n".join(timestamp_errors))
        if timestamp_summary["total_conv_plan_duration_ns"] != 70:
            raise AssertionError("timestamp duration sum was not captured")
        if timestamp_summary["rows"][0]["duration_ns_mean"] != 35:
            raise AssertionError("timestamp duration mean was not captured")
    print("validated Vulkan conv plan tuning result")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate or build offline Vulkan conv plan tuning results."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("path")
    validate.set_defaults(func=cmd_validate)

    from_sweep = subparsers.add_parser("from-sweep-summary")
    from_sweep.add_argument("--summary", required=True)
    from_sweep.add_argument("--out", required=True)
    from_sweep.add_argument("--kernel", required=True)
    from_sweep.add_argument("--candidate", action="append", default=[])
    from_sweep.add_argument(
        "--granularity",
        choices=("candidate", "plan-key"),
        default="candidate",
    )
    from_sweep.set_defaults(func=cmd_from_sweep)

    from_timestamp_log = subparsers.add_parser("from-timestamp-log")
    from_timestamp_log.add_argument("--log", required=True)
    from_timestamp_log.add_argument("--out", required=True)
    from_timestamp_log.set_defaults(func=cmd_from_timestamp_log)

    self_test = subparsers.add_parser("self-test")
    self_test.set_defaults(func=cmd_self_test)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
