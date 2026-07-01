#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "VulkanConvPlanTuningResult.v0"
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


def parse_plan_key_snapshot_row(row: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in row.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


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


def cmd_validate(args: argparse.Namespace) -> int:
    payload = load_json(Path(args.path))
    errors = validate_tuning_result(payload)
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

    self_test = subparsers.add_parser("self-test")
    self_test.set_defaults(func=cmd_self_test)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
