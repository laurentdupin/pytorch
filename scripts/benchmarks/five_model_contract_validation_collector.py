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

SUBMIT_ORIGIN_COUNTER_INDEX = {
    "total_queue_submits": 0,
    "normal_cmd_submit_frequency_submits": 1,
    "stack_planned_recording_submits": 2,
    "pre_stack_flush_submits": 3,
    "post_stack_flush_submits": 4,
    "explicit_synchronize_submits": 5,
    "tensor_cpu_readback_submits": 6,
    "fallback_readback_submits": 7,
    "retire_queue_drain_submits": 8,
    "conv_prepack_upload_submits": 13,
}

CONV_PLAN_COUNTER_FIELDS = [
    "total",
    "pointwise_1x1_hit",
    "pointwise_1x1_as_linear_hit",
    "known_bad_large_pointwise",
    "cpu_fallback",
    "reject_layout",
    "reject_dtype",
]

POINTWISE_CONV_ROUTE_COUNTER_FIELDS = [
    "total_1x1",
    "specialized_1x1_hit",
    "generic_1x1_hit",
    "reject_not_direct_buffer",
    "reject_input_not_buffer",
    "reject_input_not_direct_buffer",
    "reject_output_not_direct_buffer",
    "reject_storage_offset",
    "reject_dtype",
    "reject_groups",
    "reject_stride_padding_dilation",
    "reject_weight_layout",
    "reject_bias",
    "reject_shape",
]

LINEAR_PLAN_COUNTER_FIELDS = [
    "total",
    "coop_hit",
    "coop_tail_m_hit",
    "reject_m_tail",
    "reject_k_tail",
    "reject_n_tail",
    "reject_layout",
    "reject_dtype",
    "reject_capability",
    "fallback_plain_bf16",
    "fallback_float",
]

PLAN_NOT_AVAILABLE = "not_available"

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


def load_specific_transition_contracts(
    spec_dir: Path = CONTRACT_SPEC_DIR,
) -> dict[tuple[str, str, str], str]:
    contracts: dict[tuple[str, str, str], str] = {}
    if not spec_dir.exists():
        return contracts
    for path in sorted(spec_dir.glob("*.json")):
        if path.name == GENERATED_CPP_MANIFEST_FILE:
            continue
        spec = load_json(path)
        transition_contract = spec.get("transition_contract")
        if not isinstance(transition_contract, dict):
            continue
        if not transition_contract.get("collector_event_bucket"):
            continue
        if transition_contract.get("contract_type") != "LayoutTransitionContract":
            raise ValueError(f"{path.name} is not a LayoutTransitionContract")
        reason = transition_contract.get("reason")
        producer_schema = transition_contract.get("producer_schema")
        consumer_schema = transition_contract.get("consumer_schema")
        contract_name = spec.get("contract_name")
        if not (reason and producer_schema and consumer_schema and contract_name):
            raise ValueError(
                f"{path.name} missing transition event match fields or contract_name"
            )
        key = (reason, producer_schema, consumer_schema)
        previous = contracts.get(key)
        if previous is not None and previous != contract_name:
            raise ValueError(
                f"duplicate transition event bucket {key}: "
                f"{previous} and {contract_name}"
            )
        contracts[key] = contract_name
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


def key_value_fields(row: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in row.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key] = value
    return fields


def int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_int_list(value: Any) -> list[int] | str:
    if not isinstance(value, str):
        return PLAN_NOT_AVAILABLE
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return PLAN_NOT_AVAILABLE
    inner = text[1:-1].strip()
    if not inner:
        return []
    result: list[int] = []
    for part in inner.split(","):
        try:
            result.append(int(part))
        except ValueError:
            return PLAN_NOT_AVAILABLE
    return result


def named_counter_snapshot(values: list[Any], names: list[str]) -> dict[str, int]:
    return {
        name: int_value(values[index])
        for index, name in enumerate(names)
        if index < len(values)
    }


def list_counter_field(delta: dict[str, Any], name: str) -> list[Any]:
    value = delta.get(name)
    if value is None:
        value = delta.get(f"{name}_delta")
    return value if isinstance(value, list) else []


def phase_counter_delta(phase: Any) -> dict[str, Any]:
    if not isinstance(phase, dict):
        return {}
    delta = phase.get("delta")
    return delta if isinstance(delta, dict) else phase


def phase_counter_blob(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    direct = result.get("vulkan_phase_counters")
    if isinstance(direct, dict):
        return direct
    counters = result.get("counters")
    if isinstance(counters, dict) and isinstance(counters.get("vulkan_phase_counters"), dict):
        return counters["vulkan_phase_counters"]
    return {}


def phase_delta_by_name(result: dict[str, Any] | None, phase_name: str) -> dict[str, Any]:
    blob = phase_counter_blob(result)
    if phase_name == "total" and blob.get("total"):
        return phase_counter_delta(blob["total"])
    for phase in blob.get("phases") or []:
        if isinstance(phase, dict) and phase.get("name") == phase_name:
            return phase_counter_delta(phase)
    return {}


def normalize_model_suite_status(status: str | None, failure: Any) -> str:
    if status == "ok":
        return "ok"
    failure_text = json.dumps(failure or {}, sort_keys=True).lower()
    if "out_of_device_memory" in failure_text or "oom" in failure_text:
        return "oom"
    if "dtensor" in failure_text or "importerror" in failure_text:
        return "env_blocked"
    if status == "failure":
        return "fail"
    if status == "skip":
        return "skip"
    return status or "skip"


def unwrap_model_suite_result(raw: Any, source_path: Path | None = None) -> Any:
    if not isinstance(raw, dict) or not isinstance(raw.get("records"), list):
        return raw
    records = [record for record in raw["records"] if isinstance(record, dict)]
    if not records:
        return raw
    record = records[0]
    environment = dict(raw.get("environment") or {})
    environment.update(record.get("environment") or {})
    counters = record.get("counters") or {}
    output_sanity = record.get("output_sanity") or {}
    failure = record.get("failure") or {}
    status = normalize_model_suite_status(record.get("status"), failure)
    result = {
        "schema_version": raw.get("schema_version"),
        "source": "benchmark_model_suite_record",
        "raw_result_json": rel_path(source_path),
        "python_executable": environment.get("python") or environment.get("python_executable"),
        "torch_version": environment.get("torch_version"),
        "status": status,
        "performance_valid": status == "ok" and bool(record.get("timings")),
        "task": record.get("task"),
        "model_name": record.get("model_name"),
        "model_id": record.get("model_id"),
        "backend": record.get("backend"),
        "dtype": record.get("dtype"),
        "input": record.get("input"),
        "device_info": {
            "device": record.get("device"),
            "device_index": record.get("device_index"),
            "backend": record.get("backend"),
        },
        "timings": record.get("timings") or {},
        "failure": failure,
        "output_sanity": output_sanity,
        "environment": environment,
        "vulkan_phase_counters": counters.get("vulkan_phase_counters") or {},
        "vulkan_debug_counters": counters.get("vulkan_debug") or {},
    }
    return result


def timing_summary(result: dict[str, Any] | None, matrix_row: dict[str, Any] | None) -> dict[str, Any]:
    if matrix_row and isinstance(matrix_row.get("timing"), dict):
        return matrix_row["timing"]
    if not result:
        return {}
    if isinstance(result.get("timings"), dict):
        return result["timings"]
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
    named_submit_origins = Counter()
    for row in list_counter_field(delta, "submit_origin_phase_counters"):
        fields = key_value_fields(str(row))
        origin = fields.get("origin")
        if origin:
            named_submit_origins[origin] += int_value(fields.get("count"))
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
    for name, index in SUBMIT_ORIGIN_COUNTER_INDEX.items():
        if len(submit_origin) > index:
            normalized[name] = int(submit_origin[index])
    if named_submit_origins:
        origin_to_field = {
            "normal_cmd_submit_frequency": "normal_cmd_submit_frequency_submits",
            "stack_planned_recording_submit": "stack_planned_recording_submits",
            "pre_stack_flush": "pre_stack_flush_submits",
            "post_stack_flush": "post_stack_flush_submits",
            "explicit_synchronize": "explicit_synchronize_submits",
            "tensor_cpu_readback": "tensor_cpu_readback_submits",
            "fallback_readback": "fallback_readback_submits",
            "retire_queue_drain": "retire_queue_drain_submits",
            "conv_prepack_upload": "conv_prepack_upload_submits",
        }
        for origin, field in origin_to_field.items():
            if origin in named_submit_origins:
                normalized[field] = int(named_submit_origins[origin])
        normalized["submit_origin_named_counts"] = dict(sorted(named_submit_origins.items()))
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
    phase_blob = phase_counter_blob(result)
    for phase in phase_blob.get("phases") or []:
        name = phase.get("name")
        if name:
            phases[name] = normalize_counter_delta(phase_counter_delta(phase))
    if phase_blob.get("total"):
        phases["total"] = normalize_counter_delta(phase_counter_delta(phase_blob["total"]))
    return phases


def summarize_transitions(
    transition_jsonl: Path | None,
    missing_artifacts: list[dict[str, str]],
    reason_bucket_contracts: dict[str, str] | None = None,
    specific_transition_contracts: dict[tuple[str, str, str], str] | None = None,
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
    specific_transition_contracts = specific_transition_contracts or {}
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
                producer_schema = event.get("producer_schema") or "unknown"
                consumer_schema = event.get("consumer_schema") or "unknown"
                event_contract = specific_transition_contracts.get(
                    (reason, producer_schema, consumer_schema)
                )
                reason_contract = reason_bucket_contracts.get(reason)
                if event_contract:
                    by_contract[event_contract] += 1
                elif reason_contract:
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


def sample_rows(rows: list[Any], limit: int = 3) -> list[Any]:
    return rows[:limit]


def summarize_sidecar_jsonl(path: Path) -> dict[str, Any]:
    records = 0
    candidate_contracts: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            contract = (
                event.get("contract_name")
                or event.get("candidate_contract")
                or event.get("candidate_contract_family")
            )
            if contract:
                candidate_contracts[str(contract)] += 1
            outcome = event.get("outcome") or event.get("status") or event.get("classification")
            if outcome:
                outcomes[str(outcome)] += 1
    return {
        "path": rel_path(path),
        "records": records,
        "candidate_contracts": dict(sorted(candidate_contracts.items())),
        "outcomes": dict(sorted(outcomes.items())),
    }


def discover_sidecar(path: Path | None, suffix: str) -> Path | None:
    if path is None:
        return None
    directory = path.parent
    for candidate in sorted(directory.glob(f"*{suffix}")):
        if candidate.is_file():
            return candidate
    return None


def summarize_probe_sidecars(
    row_cfg: dict[str, Any],
    result_path: Path | None,
    missing_artifacts: list[dict[str, str]],
) -> dict[str, Any]:
    sidecar_specs = {
        "probe_jsonl": (".probe.jsonl", "jsonl"),
        "admission_jsonl": (".admission.jsonl", "jsonl"),
        "probe_summary_json": (".probe_summary.json", "json"),
        "op_hit_log": (".op_hits.log", "log"),
    }
    sidecars: dict[str, Any] = {}
    for name, (suffix, kind) in sidecar_specs.items():
        configured = row_cfg.get(name)
        path = repo_path(configured) if configured else discover_sidecar(result_path, suffix)
        if configured and (path is None or not path.exists()):
            missing_artifacts.append(
                {
                    "kind": "missing_artifact",
                    "name": name,
                    "path": rel_path(path) or "",
                    "impact": f"{name} evidence is unavailable for this row",
                }
            )
            continue
        if path is None or not path.exists():
            continue
        if kind == "jsonl":
            sidecars[name] = summarize_sidecar_jsonl(path)
        elif kind == "json":
            data = load_json(path)
            sidecars[name] = {
                "path": rel_path(path),
                "top_level_keys": sorted(data.keys()) if isinstance(data, dict) else [],
            }
        else:
            sidecars[name] = summarize_op_hit_log(path)
    sidecars["present_count"] = sum(1 for key in sidecars if key != "present_count")
    return sidecars


def summarize_op_hit_log(path: Path) -> dict[str, Any]:
    line_count = 0
    pointwise_classes: Counter[str] = Counter()
    pointwise_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            line_count += 1
            fields = key_value_fields(stripped)
            if fields.get("op") not in {"pointwise_route", None}:
                continue
            if fields.get("op") is None and not stripped.startswith("pointwise_route"):
                continue
            contract = fields.get("contract")
            family = fields.get("contract_family")
            if contract != "SmallSpatialPointwiseConvContract" or family not in {
                "DepthVisionProjection",
                "OCRProjection",
            }:
                continue
            selected = fields.get("selected") or PLAN_NOT_AVAILABLE
            input_offset = int_field(fields, "input_offset")
            old_generic_retained = fields.get("old_generic_retained") == "1"
            if selected == "as_linear":
                classification = "descriptor_view_only"
            elif (
                selected == "generic"
                and old_generic_retained
                and int_value(input_offset) > 0
            ):
                classification = "generic_conv_preserved_due_unproven_layout"
            elif selected == "generic" and old_generic_retained:
                classification = "generic_conv_preserved_due_unproven_layout"
            else:
                classification = "pointwise_input_layout_not_classified"
            pointwise_classes[classification] += 1
            if len(pointwise_rows) < 20:
                pointwise_rows.append(
                    {
                        "classification": classification,
                        "contract": contract,
                        "contract_family": family,
                        "contract_tuple": fields.get("contract_tuple")
                        or PLAN_NOT_AVAILABLE,
                        "selected": selected,
                        "selected_plan": fields.get("selected_plan")
                        or PLAN_NOT_AVAILABLE,
                        "fallback_plan": fields.get("fallback_plan")
                        or PLAN_NOT_AVAILABLE,
                        "reject": fields.get("reject") or PLAN_NOT_AVAILABLE,
                        "input": parse_int_list(fields.get("input")),
                        "weight": parse_int_list(fields.get("weight")),
                        "input_offset": input_offset,
                        "input_direct": bool_field(fields, "input_direct"),
                        "output_direct": bool_field(fields, "output_direct"),
                    }
                )
    summary: dict[str, Any] = {"path": rel_path(path), "lines": line_count}
    if pointwise_classes:
        summary["pointwise_input_layout_transition_evidence"] = {
            "contract_name": "PointwiseConvInputLayoutTransitionContract",
            "schema_version": 0,
            "classes": dict(sorted(pointwise_classes.items())),
            "sample_rows": pointwise_rows,
        }
    return summary


def summarize_model_suite_evidence(
    result: dict[str, Any] | None,
    row_cfg: dict[str, Any],
    result_path: Path | None,
    missing_artifacts: list[dict[str, str]],
) -> dict[str, Any]:
    total_delta = phase_delta_by_name(result, "total")
    output_sanity = (result or {}).get("output_sanity") or {}
    kernel = {
        "conv_aggregate_rows": len(list_counter_field(total_delta, "conv_aggregate_snapshot")),
        "conv_aggregate_sample": sample_rows(
            list_counter_field(total_delta, "conv_aggregate_snapshot")
        ),
        "linear_aggregate_rows": len(list_counter_field(total_delta, "linear_aggregate_snapshot")),
        "linear_aggregate_sample": sample_rows(
            list_counter_field(total_delta, "linear_aggregate_snapshot")
        ),
        "clone_requirement_rows": len(list_counter_field(total_delta, "clone_requirement_snapshot")),
        "clone_requirement_sample": sample_rows(
            list_counter_field(total_delta, "clone_requirement_snapshot")
        ),
        "conv_plan_counters": list_counter_field(total_delta, "conv_plan_counters"),
        "linear_plan_counters": list_counter_field(total_delta, "linear_plan_counters"),
        "pointwise_conv_route_counters": list_counter_field(
            total_delta, "pointwise_conv_route_counters"
        ),
        "attention_plan_counters": list_counter_field(total_delta, "attention_plan_counters"),
    }
    submit_origin_named_counts = normalize_counter_delta(total_delta).get(
        "submit_origin_named_counts", {}
    )
    diagnostics: dict[str, Any] = {}
    if isinstance(output_sanity.get("fallback_readback_attribution"), dict):
        attribution = output_sanity["fallback_readback_attribution"]
        diagnostics["fallback_readback_attribution"] = {
            "categories": sorted((attribution.get("categories") or {}).keys()),
            "total_dispatches": attribution.get("total_dispatches"),
            "vulkan_dispatches": attribution.get("vulkan_dispatches"),
        }
    if isinstance(output_sanity.get("grid_sample_calls"), list):
        diagnostics["grid_sample_calls"] = len(output_sanity["grid_sample_calls"])
    if isinstance(output_sanity.get("paddleocr_postprocess_cpu_metadata_tensors"), list):
        diagnostics["paddleocr_postprocess_cpu_metadata_tensors"] = len(
            output_sanity["paddleocr_postprocess_cpu_metadata_tensors"]
        )

    return {
        "source": (result or {}).get("source") or "result_json",
        "torch_version": (result or {}).get("torch_version")
        or ((result or {}).get("environment") or {}).get("torch_version"),
        "kernel": kernel,
        "submit_origin_named_counts": submit_origin_named_counts,
        "sidecars": summarize_probe_sidecars(row_cfg, result_path, missing_artifacts),
        "diagnostics": diagnostics,
    }


def row_counter_context(phases: dict[str, Any]) -> dict[str, Any]:
    total = phases.get("total") or {}
    timed = phases.get("timed_forward") or {}
    source = timed if timed else total
    return {
        "source_phase": "timed_forward" if timed else "total",
        "cpu_fallback_count": int(source.get("cpu_fallback_count") or 0),
        "sync_readback_count": int(source.get("sync_readback_count") or 0),
        "tensor_cpu_readback_submits": int(source.get("tensor_cpu_readback_submits") or 0),
        "buffer_copy_count": int(source.get("buffer_copy_count") or 0),
        "buffer_copy_bytes": int(source.get("buffer_copy_bytes") or 0),
        "total_queue_submits": int(source.get("total_queue_submits") or 0),
        "retire_drains": int(source.get("retire_drains") or 0),
        "conv_prepack_upload_submits": int(source.get("conv_prepack_upload_submits") or 0),
    }


def int_field(fields: dict[str, str], name: str) -> int | str:
    if name not in fields:
        return PLAN_NOT_AVAILABLE
    return int_value(fields[name])


def bool_field(fields: dict[str, str], name: str) -> bool | str:
    value = int_field(fields, name)
    if value == PLAN_NOT_AVAILABLE:
        return PLAN_NOT_AVAILABLE
    return bool(value)


def infer_conv_candidate_contract(fields: dict[str, str]) -> str:
    role = fields.get("role") or ""
    if "depth_vision_projection" in role or role == "small_spatial_pointwise_conv":
        return "SmallSpatialPointwiseConvContract"
    if fields.get("pointwise") == "1":
        return "SmallSpatialPointwiseConvContract"
    return PLAN_NOT_AVAILABLE


def infer_linear_candidate_contract(fields: dict[str, str]) -> str:
    role = fields.get("role") or ""
    if fields.get("post_op") == "1" or "gelu" in role:
        return "LinearGeluBridgeContract"
    return PLAN_NOT_AVAILABLE


def conv_plan_key(fields: dict[str, str]) -> str:
    return (
        "conv2d"
        f"|selected={fields.get('selected', PLAN_NOT_AVAILABLE)}"
        f"|kernel={fields.get('kernel', PLAN_NOT_AVAILABLE)}"
        f"|role={fields.get('role', PLAN_NOT_AVAILABLE)}"
        f"|input={fields.get('input', PLAN_NOT_AVAILABLE)}"
        f"|weight={fields.get('weight', PLAN_NOT_AVAILABLE)}"
        f"|out={fields.get('output_channels', PLAN_NOT_AVAILABLE)}"
        f"|stride={fields.get('stride', PLAN_NOT_AVAILABLE)}"
        f"|padding={fields.get('padding', PLAN_NOT_AVAILABLE)}"
        f"|dilation={fields.get('dilation', PLAN_NOT_AVAILABLE)}"
        f"|groups={fields.get('groups', PLAN_NOT_AVAILABLE)}"
        f"|bias={fields.get('bias', PLAN_NOT_AVAILABLE)}"
    )


def linear_plan_key(fields: dict[str, str]) -> str:
    return (
        "linear"
        f"|kernel={fields.get('kernel', PLAN_NOT_AVAILABLE)}"
        f"|submit={fields.get('submit_kernel', PLAN_NOT_AVAILABLE)}"
        f"|role={fields.get('role', PLAN_NOT_AVAILABLE)}"
        f"|m={fields.get('m', PLAN_NOT_AVAILABLE)}"
        f"|k={fields.get('k', PLAN_NOT_AVAILABLE)}"
        f"|n={fields.get('n', PLAN_NOT_AVAILABLE)}"
        f"|bias={fields.get('bias', PLAN_NOT_AVAILABLE)}"
        f"|post_op={fields.get('post_op', PLAN_NOT_AVAILABLE)}"
        f"|packed={fields.get('weight_packed', PLAN_NOT_AVAILABLE)}"
    )


def common_plan_availability() -> dict[str, str]:
    return {
        "program_cache": PLAN_NOT_AVAILABLE,
        "pipeline_cache": PLAN_NOT_AVAILABLE,
        "descriptor_plan_cache": PLAN_NOT_AVAILABLE,
        "scratch_allocation_reuse": PLAN_NOT_AVAILABLE,
    }


def normalize_conv_plan_evidence(
    row_id: str,
    model: dict[str, Any],
    row: str,
    plan_counters: dict[str, int],
    pointwise_counters: dict[str, int],
    row_counters: dict[str, Any],
) -> dict[str, Any]:
    fields = key_value_fields(row)
    count = int_field(fields, "count")
    input_bytes = int_field(fields, "input_bytes")
    output_bytes = int_field(fields, "output_bytes")
    weight_bytes = int_field(fields, "weight_bytes")
    byte_values = [
        value for value in (input_bytes, output_bytes, weight_bytes) if isinstance(value, int)
    ]
    return {
        "schema_version": 0,
        "source_row": row_id,
        "source_model": model,
        "source_kind": "conv_aggregate_snapshot",
        "op_family": "conv2d",
        "contract_name": PLAN_NOT_AVAILABLE,
        "candidate_contract_family": infer_conv_candidate_contract(fields),
        "plan_key": conv_plan_key(fields),
        "selected_route": fields.get("selected") or PLAN_NOT_AVAILABLE,
        "plan_label": fields.get("kernel") or PLAN_NOT_AVAILABLE,
        "route_label": fields.get("role") or PLAN_NOT_AVAILABLE,
        "reject_reason": fields.get("reject") or PLAN_NOT_AVAILABLE,
        "shapes": {
            "input": parse_int_list(fields.get("input")),
            "weight": parse_int_list(fields.get("weight")),
            "output_channels": int_field(fields, "output_channels"),
        },
        "attrs": {
            "stride": parse_int_list(fields.get("stride")),
            "padding": parse_int_list(fields.get("padding")),
            "dilation": parse_int_list(fields.get("dilation")),
            "groups": int_field(fields, "groups"),
            "bias": bool_field(fields, "bias"),
            "pointwise": bool_field(fields, "pointwise"),
            "depthwise": bool_field(fields, "depthwise"),
            "sliding_window": bool_field(fields, "sliding_window"),
        },
        "layout": {
            "input_direct": bool_field(fields, "input_direct"),
            "output_direct": bool_field(fields, "output_direct"),
            "weight_packed": bool_field(fields, "weight_packed"),
        },
        "evidence_counters": {
            "dispatch_count": count,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "weight_bytes": weight_bytes,
            "total_observed_bytes": sum(byte_values) if byte_values else PLAN_NOT_AVAILABLE,
            "row_counter_context": row_counters,
            "conv_plan_counters": plan_counters,
            "pointwise_conv_route_counters": pointwise_counters,
        },
        "prepack_upload": {
            "weight_packed": bool_field(fields, "weight_packed"),
            "conv_prepack_upload_submits": row_counters.get("conv_prepack_upload_submits", 0),
        },
        "cache_counters": common_plan_availability(),
    }


def normalize_linear_plan_evidence(
    row_id: str,
    model: dict[str, Any],
    row: str,
    plan_counters: dict[str, int],
    row_counters: dict[str, Any],
) -> dict[str, Any]:
    fields = key_value_fields(row)
    count = int_field(fields, "count")
    input_bytes = int_field(fields, "input_bytes")
    output_bytes = int_field(fields, "output_bytes")
    weight_bytes = int_field(fields, "weight_bytes")
    byte_values = [
        value for value in (input_bytes, output_bytes, weight_bytes) if isinstance(value, int)
    ]
    return {
        "schema_version": 0,
        "source_row": row_id,
        "source_model": model,
        "source_kind": "linear_aggregate_snapshot",
        "op_family": "linear",
        "contract_name": PLAN_NOT_AVAILABLE,
        "candidate_contract_family": infer_linear_candidate_contract(fields),
        "plan_key": linear_plan_key(fields),
        "selected_route": fields.get("submit_kernel") or PLAN_NOT_AVAILABLE,
        "plan_label": fields.get("kernel") or PLAN_NOT_AVAILABLE,
        "route_label": fields.get("role") or PLAN_NOT_AVAILABLE,
        "reject_reason": PLAN_NOT_AVAILABLE,
        "shapes": {
            "m": int_field(fields, "m"),
            "k": int_field(fields, "k"),
            "n": int_field(fields, "n"),
        },
        "attrs": {
            "bias": bool_field(fields, "bias"),
            "post_op": int_field(fields, "post_op"),
            "input_dtype": int_field(fields, "input_dtype"),
            "weight_dtype": int_field(fields, "weight_dtype"),
            "bias_dtype": int_field(fields, "bias_dtype"),
            "output_dtype": int_field(fields, "output_dtype"),
        },
        "layout": {
            "input_direct": bool_field(fields, "input_direct"),
            "output_direct": bool_field(fields, "output_direct"),
            "weight_packed": bool_field(fields, "weight_packed"),
            "input_offset": int_field(fields, "input_offset"),
            "weight_offset": int_field(fields, "weight_offset"),
            "output_offset": int_field(fields, "output_offset"),
        },
        "evidence_counters": {
            "dispatch_count": count,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "weight_bytes": weight_bytes,
            "total_observed_bytes": sum(byte_values) if byte_values else PLAN_NOT_AVAILABLE,
            "row_counter_context": row_counters,
            "linear_plan_counters": plan_counters,
        },
        "prepack_upload": {
            "weight_packed": bool_field(fields, "weight_packed"),
            "linear_prepack_upload_submits": PLAN_NOT_AVAILABLE,
        },
        "cache_counters": common_plan_availability(),
    }


def summarize_execution_plan_evidence(
    result: dict[str, Any] | None,
    row_id: str,
    model: dict[str, Any],
    phases: dict[str, Any],
) -> dict[str, Any]:
    total_delta = phase_delta_by_name(result, "total")
    row_counters = row_counter_context(phases)
    conv_plan_counters = named_counter_snapshot(
        list_counter_field(total_delta, "conv_plan_counters"),
        CONV_PLAN_COUNTER_FIELDS,
    )
    pointwise_counters = named_counter_snapshot(
        list_counter_field(total_delta, "pointwise_conv_route_counters"),
        POINTWISE_CONV_ROUTE_COUNTER_FIELDS,
    )
    linear_plan_counters = named_counter_snapshot(
        list_counter_field(total_delta, "linear_plan_counters"),
        LINEAR_PLAN_COUNTER_FIELDS,
    )
    conv_rows = [
        normalize_conv_plan_evidence(
            row_id,
            model,
            str(row),
            conv_plan_counters,
            pointwise_counters,
            row_counters,
        )
        for row in list_counter_field(total_delta, "conv_aggregate_snapshot")
    ]
    linear_rows = [
        normalize_linear_plan_evidence(
            row_id,
            model,
            str(row),
            linear_plan_counters,
            row_counters,
        )
        for row in list_counter_field(total_delta, "linear_aggregate_snapshot")
    ]
    pointwise_summary: dict[str, Any] = {}
    if any(pointwise_counters.values()):
        pointwise_summary = {
            "schema_version": 0,
            "source_row": row_id,
            "source_model": model,
            "source_kind": "pointwise_conv_route_counters",
            "op_family": "pointwise_conv",
            "contract_name": "SmallSpatialPointwiseConvContract",
            "plan_key": "pointwise_conv_route_counters",
            "selected_route": "counter_snapshot",
            "evidence_counters": {
                "pointwise_conv_route_counters": pointwise_counters,
                "conv_plan_counters": conv_plan_counters,
                "row_counter_context": row_counters,
            },
            "cache_counters": common_plan_availability(),
        }
    all_rows = conv_rows + linear_rows
    top_rows = sorted(
        all_rows,
        key=lambda item: (
            -int_value(item["evidence_counters"].get("dispatch_count")),
            -int_value(item["evidence_counters"].get("total_observed_bytes")),
            item["plan_key"],
        ),
    )[:20]
    return {
        "schema_version": 0,
        "scope": "collector_only_existing_counter_snapshots",
        "behavior_change": False,
        "summary": {
            "conv_plan_rows": len(conv_rows),
            "linear_plan_rows": len(linear_rows),
            "pointwise_counter_rows": 1 if pointwise_summary else 0,
            "top_plan_rows": len(top_rows),
        },
        "plan_counters": {
            "conv": conv_plan_counters,
            "pointwise_conv": pointwise_counters,
            "linear": linear_plan_counters,
        },
        "pointwise_summary": pointwise_summary,
        "top_plan_rows": top_rows,
    }


def embedded_region_lifetime_summary(result: dict[str, Any] | None) -> dict[str, Any] | None:
    total_delta = phase_delta_by_name(result, "total")
    if not total_delta:
        return None
    dry_run_rows = list_counter_field(total_delta, "stack_subresource_lifetime_dry_run_snapshot")
    retire_blocker_rows = list_counter_field(total_delta, "stack_retire_drain_blocker_snapshot")
    submit_attribution_rows = list_counter_field(
        total_delta, "region_lifetime_submit_attribution_snapshot"
    )
    retired_resource_rows = list_counter_field(total_delta, "retired_resource_aggregate_snapshot")
    dry_run_counters = list_counter_field(
        total_delta, "stack_subresource_lifetime_dry_run_counters"
    )
    if not (
        dry_run_rows
        or retire_blocker_rows
        or submit_attribution_rows
        or retired_resource_rows
        or dry_run_counters
    ):
        return None
    return {
        "dry_run_enabled": bool(dry_run_rows or any(int_value(value) for value in dry_run_counters)),
        "all_safe_group_eligible": 0,
        "would_remove_submit_drains": 0,
        "actual_removed_submit_drains": 0,
        "blockers_by_class": {},
        "blocker_bytes_by_class": {},
        "peak_extra_live_bytes_estimate": 0,
        "embedded_region_lifetime_available": True,
        "stack_subresource_lifetime_dry_run_rows": len(dry_run_rows),
        "stack_subresource_lifetime_dry_run_sample": sample_rows(dry_run_rows),
        "stack_retire_drain_blocker_rows": len(retire_blocker_rows),
        "stack_retire_drain_blocker_sample": sample_rows(retire_blocker_rows),
        "region_lifetime_submit_attribution_rows": len(submit_attribution_rows),
        "region_lifetime_submit_attribution_sample": sample_rows(
            submit_attribution_rows
        ),
        "retired_resource_aggregate_rows": len(retired_resource_rows),
        "retired_resource_aggregate_sample": sample_rows(retired_resource_rows),
    }


def summarize_region_lifetime(
    dry_run_summary: Path | None,
    missing_artifacts: list[dict[str, str]],
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    embedded = embedded_region_lifetime_summary(result)
    if dry_run_summary is None and embedded is not None:
        return embedded

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


STACK_REGION_GRAPH_ROW_KEYS = (
    "stack_region_boundary_submit_plan_live_rows",
    "stack_region_barrier_only_canary_live_rows",
    "stack_region_pre_dispatch_proof_table_rows",
    "stack_region_boundary_optimization_plan_rows",
    "stack_region_submit_elision_canary_rows",
    "stack_region_single_recording_canary_rows",
    "stack_region_exit_submit_runtime_point_rows",
    "stack_region_pending_retire_transfer_records",
    "stack_region_pending_retire_transfer_owner_records",
    "region_lifetime_rows",
)


def graph_row_fields(row: Any) -> dict[str, str]:
    if isinstance(row, dict):
        fields = row.get("fields")
        if isinstance(fields, dict):
            return {str(key): str(value) for key, value in fields.items()}
        raw = row.get("raw")
        if isinstance(raw, str):
            return key_value_fields(raw)
    if isinstance(row, str):
        return key_value_fields(row)
    return {}


def counter_from_graph_rows(rows: list[Any], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = graph_row_fields(row).get(field)
        if value:
            counter[value] += 1
    return dict(sorted(counter.items()))


def summarize_stack_region_graph(
    graph_path: Path | None,
    missing_artifacts: list[dict[str, str]],
) -> dict[str, Any]:
    if graph_path is None:
        return {
            "available": False,
            "configured": False,
            "path": None,
            "row_counts": {},
        }
    if not graph_path.exists():
        missing_artifacts.append(
            {
                "kind": "missing_artifact",
                "name": "stack_graph_json",
                "path": rel_path(graph_path) or "",
                "impact": "stack-region graph ownership evidence is unavailable for this row",
            }
        )
        return {
            "available": False,
            "configured": True,
            "path": rel_path(graph_path),
            "row_counts": {},
        }

    data = load_json(graph_path)
    summary = data.get("summary") if isinstance(data, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    row_counts: dict[str, int] = {}
    for key in STACK_REGION_GRAPH_ROW_KEYS:
        rows = data.get(key) if isinstance(data, dict) else None
        if isinstance(rows, list):
            row_counts[key] = len(rows)

    single_recording_rows = (
        data.get("stack_region_single_recording_canary_rows")
        if isinstance(data, dict)
        else []
    )
    if not isinstance(single_recording_rows, list):
        single_recording_rows = []
    pending_transfer_rows = (
        data.get("stack_region_pending_retire_transfer_records")
        if isinstance(data, dict)
        else []
    )
    if not isinstance(pending_transfer_rows, list):
        pending_transfer_rows = []
    pending_owner_rows = (
        data.get("stack_region_pending_retire_transfer_owner_records")
        if isinstance(data, dict)
        else []
    )
    if not isinstance(pending_owner_rows, list):
        pending_owner_rows = []

    return {
        "available": True,
        "configured": True,
        "path": rel_path(graph_path),
        "schema": data.get("schema") if isinstance(data, dict) else None,
        "behavior_neutral": bool((data or {}).get("behavior_neutral"))
        if isinstance(data, dict)
        else False,
        "summary": {
            "dispatch_nodes": int(summary.get("dispatch_nodes") or 0),
            "resource_nodes": int(summary.get("resource_nodes") or 0),
            "dependency_edge_rows": int(summary.get("dependency_edge_rows") or 0),
            "boundary_nodes": int(summary.get("boundary_nodes") or 0),
            "single_recording_canary_rows": int(
                summary.get("stack_region_single_recording_canary_rows") or 0
            ),
            "single_recording_canary_submits_removed": int(
                summary.get("single_recording_canary_submits_removed") or 0
            ),
            "single_recording_canary_submits_removed_outside_selected_boundary": int(
                summary.get(
                    "single_recording_canary_submits_removed_outside_selected_boundary"
                )
                or 0
            ),
            "submit_elision_enabled": bool(summary.get("submit_elision_enabled")),
            "single_recording_canary_enabled": bool(
                summary.get("single_recording_canary_enabled")
            ),
        },
        "row_counts": row_counts,
        "single_recording_canary": {
            "status_counts": counter_from_graph_rows(
                single_recording_rows, "status"
            ),
            "guard_fail_reason_counts": counter_from_graph_rows(
                single_recording_rows, "guard_fail_reason"
            ),
            "close_submit_owner_status_counts": counter_from_graph_rows(
                single_recording_rows, "region_exit_close_submit_owner_status"
            ),
            "submits_removed_counts": counter_from_graph_rows(
                single_recording_rows, "submits_removed"
            ),
        },
        "pending_retire_transfer": {
            "source_match_status_counts": counter_from_graph_rows(
                pending_transfer_rows, "source_match_status"
            ),
            "source_coverage_status_counts": counter_from_graph_rows(
                pending_transfer_rows, "region_exit_bound_source_coverage_status"
            ),
            "owner_status_counts": counter_from_graph_rows(
                pending_owner_rows, "owner_status"
            ),
            "owner_top_blocker_counts": counter_from_graph_rows(
                pending_owner_rows, "top_blocker"
            ),
        },
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
    specific_transition_contracts: dict[tuple[str, str, str], str] | None = None,
) -> dict[str, Any]:
    missing_artifacts: list[dict[str, str]] = []
    result_path = repo_path(row_cfg.get("result_json"))
    result = (
        unwrap_model_suite_result(load_json(result_path), result_path)
        if result_path and result_path.exists()
        else None
    )
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
        specific_transition_contracts,
    )
    region_lifetime = summarize_region_lifetime(
        repo_path(row_cfg.get("region_lifetime_summary")),
        missing_artifacts,
        result,
    )
    stack_graph = summarize_stack_region_graph(
        repo_path(
            first_present(
                row_cfg.get("stack_graph_json"),
                row_cfg.get("stack_region_graph_json"),
                row_cfg.get("graph_json"),
            )
        ),
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
        model.setdefault("model_id", result.get("model_id"))
        model.setdefault("task", result.get("task"))
        if result.get("input_size") is not None:
            model.setdefault("resolution_or_shape", result.get("input_size"))
    elif matrix_row:
        model.setdefault("input_path", matrix_row.get("input"))
        model.setdefault("variant", matrix_row.get("model"))
        model.setdefault("resolution_or_shape", matrix_row.get("resolution"))
    model_suite_evidence = summarize_model_suite_evidence(
        result,
        row_cfg,
        result_path,
        missing_artifacts,
    )
    execution_plan_evidence = summarize_execution_plan_evidence(
        result,
        row_cfg["row_id"],
        model,
        phases,
    )

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
        "stack_graph_json": stack_graph.get("path") or "",
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
            "model_suite_evidence": model_suite_evidence,
            "execution_plan_evidence": execution_plan_evidence,
            "stack_region_graph": stack_graph,
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
    model_suite_evidence = Counter()
    stack_graph_evidence = Counter()
    stack_graph_single_recording_guard_reasons = Counter()
    stack_graph_pending_retire_coverage = Counter()
    for row in rows:
        missing_artifacts += len(row["environment"].get("missing_artifacts") or [])
        transition_contracts.update(row["transitions"].get("events_by_contract") or {})
        op_contracts.update(row["op_contracts"].get("by_contract") or {})
        evidence = row["environment"].get("model_suite_evidence") or {}
        kernel = evidence.get("kernel") or {}
        model_suite_evidence["conv_aggregate_rows"] += int(
            kernel.get("conv_aggregate_rows") or 0
        )
        model_suite_evidence["linear_aggregate_rows"] += int(
            kernel.get("linear_aggregate_rows") or 0
        )
        model_suite_evidence["clone_requirement_rows"] += int(
            kernel.get("clone_requirement_rows") or 0
        )
        model_suite_evidence["probe_sidecar_count"] += int(
            (evidence.get("sidecars") or {}).get("present_count") or 0
        )
        for event in row["transitions"].get("events_by_reason_phase") or []:
            observed_reasons[event["reason"]] += int(event["count"])
        graph = row["environment"].get("stack_region_graph") or {}
        if graph.get("available"):
            stack_graph_evidence["available_rows"] += 1
            graph_summary = graph.get("summary") or {}
            stack_graph_evidence["dispatch_nodes"] += int(
                graph_summary.get("dispatch_nodes") or 0
            )
            stack_graph_evidence["resource_nodes"] += int(
                graph_summary.get("resource_nodes") or 0
            )
            stack_graph_evidence["dependency_edge_rows"] += int(
                graph_summary.get("dependency_edge_rows") or 0
            )
            stack_graph_evidence["single_recording_canary_rows"] += int(
                graph_summary.get("single_recording_canary_rows") or 0
            )
            stack_graph_evidence["single_recording_canary_submits_removed"] += int(
                graph_summary.get("single_recording_canary_submits_removed") or 0
            )
            single_recording = graph.get("single_recording_canary") or {}
            stack_graph_single_recording_guard_reasons.update(
                single_recording.get("guard_fail_reason_counts") or {}
            )
            pending_retire = graph.get("pending_retire_transfer") or {}
            stack_graph_pending_retire_coverage.update(
                pending_retire.get("source_coverage_status_counts") or {}
            )
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
            "model_suite_evidence_rows": dict(sorted(model_suite_evidence.items())),
            "stack_region_graph_evidence": dict(sorted(stack_graph_evidence.items())),
            "stack_region_graph_single_recording_guard_reasons": dict(
                sorted(stack_graph_single_recording_guard_reasons.items())
            ),
            "stack_region_graph_pending_retire_coverage": dict(
                sorted(stack_graph_pending_retire_coverage.items())
            ),
        },
        "budgets": {
            "rows_with_cpu_fallback_budget_failure": sum(
                1 for row in rows if row["budgets"]["cpu_fallback"]["status"] == "fail"
            )
        },
        "execution_plan_evidence": aggregate_execution_plan_evidence(rows),
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


def aggregate_execution_plan_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_family_rows = Counter()
    family_rows = Counter()
    family_dispatches = Counter()
    plan_clusters: dict[str, dict[str, Any]] = {}
    for row in rows:
        evidence = row["environment"].get("execution_plan_evidence") or {}
        summary = evidence.get("summary") or {}
        source_family_rows["conv2d"] += int(summary.get("conv_plan_rows") or 0)
        source_family_rows["linear"] += int(summary.get("linear_plan_rows") or 0)
        source_family_rows["pointwise_conv"] += int(
            summary.get("pointwise_counter_rows") or 0
        )
        for item in evidence.get("top_plan_rows") or []:
            family = item.get("op_family") or "unknown"
            dispatch_count = int_value(
                (item.get("evidence_counters") or {}).get("dispatch_count")
            )
            bytes_count = int_value(
                (item.get("evidence_counters") or {}).get("total_observed_bytes")
            )
            family_rows[family] += 1
            family_dispatches[family] += dispatch_count
            key = item.get("plan_key") or "unknown"
            cluster = plan_clusters.setdefault(
                key,
                {
                    "plan_key": key,
                    "op_family": family,
                    "selected_route": item.get("selected_route") or PLAN_NOT_AVAILABLE,
                    "plan_label": item.get("plan_label") or PLAN_NOT_AVAILABLE,
                    "route_label": item.get("route_label") or PLAN_NOT_AVAILABLE,
                    "contract_name": item.get("contract_name") or PLAN_NOT_AVAILABLE,
                    "candidate_contract_family": item.get("candidate_contract_family")
                    or PLAN_NOT_AVAILABLE,
                    "rows": [],
                    "dispatch_count": 0,
                    "total_observed_bytes": 0,
                },
            )
            cluster["dispatch_count"] += dispatch_count
            cluster["total_observed_bytes"] += bytes_count
            cluster["rows"].append(row["row_id"])
    top_clusters = sorted(
        plan_clusters.values(),
        key=lambda item: (
            -int_value(item.get("dispatch_count")),
            -int_value(item.get("total_observed_bytes")),
            item.get("plan_key") or "",
        ),
    )[:20]
    for cluster in top_clusters:
        cluster["rows"] = sorted(set(cluster["rows"]))
    return {
        "schema_version": 0,
        "behavior_change": False,
        "source_plan_rows_by_family": dict(sorted(source_family_rows.items())),
        "top_plan_rows_by_family": dict(sorted(family_rows.items())),
        "dispatches_by_family": dict(sorted(family_dispatches.items())),
        "top_plan_key_clusters": top_clusters,
    }


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
        "model-suite evidence | execution plans | lifetime evidence | graph evidence | "
        "unknown transition reasons | missing artifacts |",
        "|---|---|---:|---|---|---|---|---|---|---:|---:|",
    ]
    for row in artifact["rows"]:
        op_contracts = ", ".join(
            f"{name}:{count}" for name, count in row["op_contracts"]["by_contract"].items()
        ) or "-"
        transition_contracts = ", ".join(
            f"{name}:{count}" for name, count in row["transitions"]["events_by_contract"].items()
        ) or "-"
        evidence = row["environment"].get("model_suite_evidence") or {}
        kernel = evidence.get("kernel") or {}
        suite_summary = (
            "conv:{conv} linear:{linear} clone:{clone} sidecars:{sidecars}".format(
                conv=kernel.get("conv_aggregate_rows", 0),
                linear=kernel.get("linear_aggregate_rows", 0),
                clone=kernel.get("clone_requirement_rows", 0),
                sidecars=(evidence.get("sidecars") or {}).get("present_count", 0),
            )
        )
        plan_evidence = row["environment"].get("execution_plan_evidence") or {}
        plan_summary = (
            "conv:{conv} linear:{linear} pointwise:{pointwise}".format(
                conv=(plan_evidence.get("summary") or {}).get("conv_plan_rows", 0),
                linear=(plan_evidence.get("summary") or {}).get("linear_plan_rows", 0),
                pointwise=(plan_evidence.get("summary") or {}).get(
                    "pointwise_counter_rows", 0
                ),
            )
        )
        lifetime_summary = (
            "dry:{dry} blockers:{blockers} retired:{retired}".format(
                dry=row["region_lifetime"].get(
                    "stack_subresource_lifetime_dry_run_rows", 0
                ),
                blockers=row["region_lifetime"].get("stack_retire_drain_blocker_rows", 0),
                retired=row["region_lifetime"].get("retired_resource_aggregate_rows", 0),
            )
        )
        graph = row["environment"].get("stack_region_graph") or {}
        graph_summary = graph.get("summary") or {}
        graph_evidence_summary = (
            "dispatch:{dispatch} resource:{resource} single:{single} removed:{removed}".format(
                dispatch=graph_summary.get("dispatch_nodes", 0),
                resource=graph_summary.get("resource_nodes", 0),
                single=graph_summary.get("single_recording_canary_rows", 0),
                removed=graph_summary.get(
                    "single_recording_canary_submits_removed", 0
                ),
            )
            if graph.get("available")
            else "-"
        )
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["row_id"],
                row["status"],
                "yes" if row["timing_valid"] else "no",
                op_contracts,
                transition_contracts,
                suite_summary,
                plan_summary,
                lifetime_summary,
                graph_evidence_summary,
                row["unknowns"]["unknown_transition_reasons"],
                len(row["environment"].get("missing_artifacts") or []),
            )
        )
    lines.extend(
        [
            "",
            "## Top Execution Plan Evidence",
            "",
            "| family | dispatches | bytes | selected route | plan label | candidate contract | rows |",
            "|---|---:|---:|---|---|---|---|",
        ]
    )
    top_plans = (
        (artifact["aggregate"].get("execution_plan_evidence") or {}).get(
            "top_plan_key_clusters"
        )
        or []
    )
    if top_plans:
        for plan in top_plans[:10]:
            rows = ", ".join(f"`{row}`" for row in plan.get("rows") or [])
            lines.append(
                "| {} | {} | {} | `{}` | `{}` | `{}` | {} |".format(
                    plan.get("op_family") or "unknown",
                    plan.get("dispatch_count") or 0,
                    plan.get("total_observed_bytes") or 0,
                    plan.get("selected_route") or PLAN_NOT_AVAILABLE,
                    plan.get("plan_label") or PLAN_NOT_AVAILABLE,
                    plan.get("candidate_contract_family") or PLAN_NOT_AVAILABLE,
                    rows or "-",
                )
            )
    else:
        lines.append("| - | 0 | 0 | - | - | - | - |")
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
    specific_transition_contracts = load_specific_transition_contracts()
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
    conv_weight_key = (
        "fallback_materialization",
        "vulkan_prepack::conv2d_context",
        "vulkan_weight_cpu_materialization",
    )
    if (
        specific_transition_contracts.get(conv_weight_key)
        != "ConvWeightLayoutRepackTransitionContract"
    ):
        raise AssertionError(
            "conv weight layout repack is not covered by "
            "ConvWeightLayoutRepackTransitionContract"
        )

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
            {
                "event": "vulkan_transition",
                "phase": "model_setup",
                "reason": "fallback_materialization",
                "kind": "fallback",
                "outcome": "classified",
                "bytes": 4096,
                "host_transfer": True,
                "physical_copy": True,
                "sync_required": True,
                "queue_submit_required": True,
                "producer_schema": "vulkan_prepack::conv2d_context",
                "consumer_schema": "vulkan_weight_cpu_materialization",
                "producer_contract": "unknown",
                "consumer_contract": "unknown",
                "source_dtype": "Float",
                "source_sizes": "[8,4,3,3]",
                "source_strides": "[36,9,3,1]",
                "destination_layout": "legacy_shader_packed_conv_weight",
                "destination_storage": "TEXTURE_2D",
                "detail": (
                    "packer_path=pack_weights;actual_values_required=1;"
                    "explicit_unpack_preserved=1;pickle_unpack_preserved=1"
                ),
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
            specific_transition_contracts,
        )
    if missing_artifacts:
        raise AssertionError(f"unexpected missing artifacts: {missing_artifacts}")
    expected_counts = {
        "FallbackMaterializationContract": 1,
        "ConvWeightLayoutRepackTransitionContract": 1,
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


def validate_model_suite_ingestion() -> None:
    fallback_delta = normalize_counter_delta(
        {
            "submit_origin_counters": [
                100,
                1,
                2,
                3,
                4,
                5,
                66,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
            ]
        }
    )
    if fallback_delta.get("tensor_cpu_readback_submits") != 66:
        raise AssertionError("submit_origin_counters index 6 fallback was not used")

    delta = {
        "cpu_fallback_count": 0,
        "sync_readback_count": 1,
        "buffer_copy_counters": [3, 1024, 1, 1, 1],
        "submit_origin_counters": [100, 1, 2, 3, 4, 5, 66, 7, 8, 9, 10, 11, 12, 13],
        "submit_origin_phase_counters_delta": [
            "submit_origin_phase origin=tensor_cpu_readback phase=timed_forward count=1173",
            "submit_origin_phase origin=retire_queue_drain phase=timed_forward count=35",
        ],
        "retire_drain_counters": [9, 8, 0, 1, 2, 2048],
        "conv_aggregate_snapshot_delta": ["conv_aggregate selected=FloatBufferConv count=2"],
        "linear_aggregate_snapshot_delta": ["linear_aggregate kernel=mm_buffer_float count=4"],
        "clone_requirement_snapshot_delta": ["clone_requirement reason=storage_offset count=1"],
        "conv_plan_counters": [1, 2, 3],
        "linear_plan_counters": [4, 5],
        "pointwise_conv_route_counters": [6],
        "stack_retire_drain_blocker_snapshot_delta": [
            "stack_retire_drain_blocker callsite=context_flush_pending count=1"
        ],
        "stack_subresource_lifetime_dry_run_snapshot_delta": [
            "stack_subresource_lifetime_dry_run class=metadata_uniform count=1"
        ],
        "region_lifetime_submit_attribution_snapshot_delta": [
            (
                "region_lifetime_submit_attribution group=1 "
                "origin=retire_queue_drain phase=stack_owner "
                "callsite=stack_owner_norm2 count=1 bytes=2048"
            )
        ],
        "retired_resource_aggregate_snapshot_delta": [
            "retired_resource kind=buffer role=unknown count=1"
        ],
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        result_path = tmp_path / "model_suite_result.json"
        result_payload = {
            "schema_version": 1,
            "environment": {
                "python": "agent_space/venvs/example/Scripts/python.exe",
                "torch_version": "stale-runtime-for-collector-test",
            },
            "records": [
                {
                    "status": "ok",
                    "task": "collector_self_test",
                    "model_name": "example",
                    "model_id": "example_model",
                    "backend": "vulkan",
                    "dtype": "float32",
                    "device": "vulkan:0",
                    "device_index": 0,
                    "input": {"shape": [1, 3, 16, 16]},
                    "timings": {"timed_forward": {"mean_s": 1.0}},
                    "counters": {
                        "vulkan_phase_counters": {
                            "phases": [{"name": "timed_forward", "delta": delta}],
                            "total": {"name": "total", "delta": delta},
                        }
                    },
                    "output_sanity": {
                        "fallback_readback_attribution": {
                            "total_dispatches": 2,
                            "vulkan_dispatches": 1,
                            "categories": {"metadata": {"count": 1}},
                        },
                        "grid_sample_calls": [{"index": 0}],
                        "paddleocr_postprocess_cpu_metadata_tensors": [{"shape": [1]}],
                    },
                }
            ],
        }
        write_json(result_path, result_payload)
        (tmp_path / "row.probe.jsonl").write_text(
            json.dumps({"candidate_contract": "ExampleContract", "outcome": "recorded"})
            + "\n",
            encoding="utf-8",
        )
        (tmp_path / "row.admission.jsonl").write_text(
            json.dumps({"contract_name": "ExampleContract", "outcome": "admitted"})
            + "\n",
            encoding="utf-8",
        )
        write_json(tmp_path / "row.probe_summary.json", {"status": "ok"})
        write_json(
            tmp_path / "row.stack_graph.json",
            {
                "schema": "StackRegionDependencyGraph.v0",
                "behavior_neutral": True,
                "summary": {
                    "dispatch_nodes": 3,
                    "resource_nodes": 2,
                    "dependency_edge_rows": 1,
                    "boundary_nodes": 1,
                    "stack_region_single_recording_canary_rows": 1,
                    "single_recording_canary_submits_removed": 0,
                    "single_recording_canary_submits_removed_outside_selected_boundary": 0,
                    "submit_elision_enabled": False,
                    "single_recording_canary_enabled": False,
                },
                "stack_region_single_recording_canary_rows": [
                    {
                        "fields": {
                            "status": "single_recording_owner_close_submit_canary_guard_failed",
                            "guard_fail_reason": "pending_dispatch_barrier_coverage_incomplete",
                            "region_exit_close_submit_owner_status": "region_exit_close_submit_owner_preserved_phase_submit_batch_fail_closed",
                            "submits_removed": "0",
                        }
                    }
                ],
                "stack_region_pending_retire_transfer_records": [
                    {
                        "fields": {
                            "source_match_status": "pending_retire_transfer_source_partially_bound_to_region_exit_submit",
                            "region_exit_bound_source_coverage_status": "pending_retire_transfer_source_coverage_partial",
                        }
                    }
                ],
                "stack_region_pending_retire_transfer_owner_records": [
                    {
                        "fields": {
                            "owner_status": "pending_retire_transfer_owner_blocked_by_transfer_plan",
                            "top_blocker": "pending_retire_transfer_owner_unavailable",
                        }
                    }
                ],
            },
        )
        (tmp_path / "row.op_hits.log").write_text(
            "\n".join(
                [
                    "aten::example",
                    (
                        "pointwise_route selected=as_linear "
                        "selected_plan=FloatBufferPointwise1x1AsLinear "
                        "fallback_plan=FloatBufferConv reject=KnownBadLargePointwiseConv "
                        "contract=SmallSpatialPointwiseConvContract "
                        "contract_family=OCRProjection contract_tuple=ocr_projection "
                        "old_generic_retained=0 input=[1,512,14,14] "
                        "output=[1,192,14,14] weight=[192,512,1,1] "
                        "input_direct=1 output_direct=1 input_offset=0"
                    ),
                    (
                        "pointwise_route selected=generic "
                        "selected_plan=FloatBufferPointwise1x1 "
                        "fallback_plan=FloatBufferConv reject=KnownBadLargePointwiseConv "
                        "contract=SmallSpatialPointwiseConvContract "
                        "contract_family=DepthVisionProjection "
                        "contract_tuple=depth_vision_factorized_projection_108 "
                        "old_generic_retained=1 input=[1,384,20,31] "
                        "output=[1,192,20,31] weight=[192,384,1,1] "
                        "input_direct=0 output_direct=1 input_offset=384"
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        row = build_row(
            {
                "row_id": "self_test",
                "result_json": str(result_path),
                "stack_graph_json": str(tmp_path / "row.stack_graph.json"),
            },
            {},
        )

    timed = row["phase_counters"]["timed_forward"]
    if timed.get("tensor_cpu_readback_submits") != 1173:
        raise AssertionError("named tensor_cpu_readback submit origin was not used")
    if timed.get("retire_queue_drain_submits") != 35:
        raise AssertionError("named retire_queue_drain submit origin was not used")
    evidence = row["environment"]["model_suite_evidence"]
    kernel = evidence["kernel"]
    if kernel["conv_aggregate_rows"] != 1 or kernel["linear_aggregate_rows"] != 1:
        raise AssertionError("conv/linear aggregate snapshots were not consumed")
    if kernel["clone_requirement_rows"] != 1:
        raise AssertionError("clone requirement snapshot was not consumed")
    if evidence["sidecars"]["present_count"] != 4:
        raise AssertionError("probe/admission/op-hit sidecars were not discovered")
    pointwise_layout_evidence = evidence["sidecars"]["op_hit_log"].get(
        "pointwise_input_layout_transition_evidence"
    )
    if not pointwise_layout_evidence:
        raise AssertionError("pointwise input layout evidence was not parsed")
    expected_layout_classes = {
        "descriptor_view_only": 1,
        "generic_conv_preserved_due_unproven_layout": 1,
    }
    if pointwise_layout_evidence["classes"] != expected_layout_classes:
        raise AssertionError(
            "pointwise input layout classes were not normalized: "
            f"{pointwise_layout_evidence['classes']}"
        )
    if not row["region_lifetime"].get("embedded_region_lifetime_available"):
        raise AssertionError("embedded region/lifetime evidence was not consumed")
    if row["region_lifetime"].get("region_lifetime_submit_attribution_rows") != 1:
        raise AssertionError("submit attribution evidence was not consumed")
    if any(
        item.get("name") == "region_lifetime_dry_run"
        for item in row["environment"].get("missing_artifacts") or []
    ):
        raise AssertionError("embedded lifetime evidence was still reported missing")
    graph = row["environment"].get("stack_region_graph") or {}
    if not graph.get("available"):
        raise AssertionError("stack-region graph evidence was not consumed")
    if graph["summary"]["single_recording_canary_rows"] != 1:
        raise AssertionError("single-recording graph rows were not summarized")
    guard_reasons = graph["single_recording_canary"]["guard_fail_reason_counts"]
    if guard_reasons.get("pending_dispatch_barrier_coverage_incomplete") != 1:
        raise AssertionError("single-recording guard reasons were not counted")
    coverage = graph["pending_retire_transfer"]["source_coverage_status_counts"]
    if coverage.get("pending_retire_transfer_source_coverage_partial") != 1:
        raise AssertionError("pending-retire graph coverage was not counted")
    print("validated model-suite collector ingestion")


def validate_execution_plan_evidence() -> None:
    delta = {
        "cpu_fallback_count": 0,
        "sync_readback_count": 0,
        "buffer_copy_counters": [2, 2048, 1, 1, 0],
        "submit_origin_counters": [7, 1, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 2],
        "retire_drain_counters": [1, 1, 0, 0, 0, 0],
        "conv_plan_counters": [4, 2, 0, 0, 0, 0, 0],
        "pointwise_conv_route_counters": [2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "linear_plan_counters": [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
        "conv_aggregate_snapshot": [
            (
                "conv_aggregate selected=FloatBufferConv reject=None "
                "kernel=conv2d_buffer_float role=other_pointwise_1x1 "
                "input=[1,16,8,8] output_channels=32 weight=[32,16,1,1] "
                "stride=[1,1] padding=[0,0] dilation=[1,1] groups=1 "
                "input_direct=1 output_direct=1 weight_packed=1 bias=1 "
                "pointwise=1 depthwise=0 sliding_window=0 input_bytes=4096 "
                "output_bytes=8192 weight_bytes=2048 count=2"
            )
        ],
        "linear_aggregate_snapshot": [
            (
                "linear_aggregate role=fc1_gelu kernel=mm_buffer_float_gelu "
                "submit_kernel=aten::linear.buffer_float_bias_gelu label=test.fc1 "
                "m=64 k=16 n=64 input_dtype=6 weight_dtype=6 bias_dtype=6 "
                "output_dtype=6 post_op=1 bias=1 input_direct=1 output_direct=1 "
                "weight_packed=1 input_offset=0 weight_offset=0 output_offset=0 "
                "input_bytes=4096 weight_bytes=4096 output_bytes=16384 count=5"
            )
        ],
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        result_path = Path(tmp_dir) / "result.json"
        write_json(
            result_path,
            {
                "records": [
                    {
                        "status": "ok",
                        "task": "plan_evidence_self_test",
                        "model_name": "example",
                        "model_id": "example_model",
                        "backend": "vulkan",
                        "device": "vulkan:0",
                        "device_index": 0,
                        "timings": {"timed_forward": {"mean_s": 1.0}},
                        "counters": {
                            "vulkan_phase_counters": {
                                "phases": [{"name": "timed_forward", "delta": delta}],
                                "total": {"name": "total", "delta": delta},
                            }
                        },
                    }
                ]
            },
        )
        row = build_row({"row_id": "plan_self_test", "result_json": str(result_path)}, {})
        aggregate = aggregate_rows([row], {})

    evidence = row["environment"]["execution_plan_evidence"]
    if evidence["summary"]["conv_plan_rows"] != 1:
        raise AssertionError("conv execution plan evidence was not normalized")
    if evidence["summary"]["linear_plan_rows"] != 1:
        raise AssertionError("linear execution plan evidence was not normalized")
    if evidence["summary"]["pointwise_counter_rows"] != 1:
        raise AssertionError("pointwise route counter evidence was not normalized")
    conv_rows = [
        item for item in evidence["top_plan_rows"] if item["op_family"] == "conv2d"
    ]
    if not conv_rows:
        raise AssertionError("conv evidence row missing")
    conv_row = conv_rows[0]
    if conv_row["shapes"]["input"] != [1, 16, 8, 8]:
        raise AssertionError("conv input shape was not parsed")
    if conv_row["evidence_counters"]["conv_plan_counters"]["pointwise_1x1_hit"] != 2:
        raise AssertionError("conv plan counters were not named")
    linear_rows = [
        item for item in evidence["top_plan_rows"] if item["op_family"] == "linear"
    ]
    if not linear_rows:
        raise AssertionError("linear evidence row missing")
    if linear_rows[0]["candidate_contract_family"] != "LinearGeluBridgeContract":
        raise AssertionError("linear candidate contract family was not inferred")
    aggregate_plans = aggregate["execution_plan_evidence"]["top_plan_key_clusters"]
    if not aggregate_plans:
        raise AssertionError("aggregate plan-key clusters were not emitted")
    print("validated execution plan evidence")


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
    parser.add_argument(
        "--validate-model-suite-ingestion",
        action="store_true",
        help="Validate benchmark_model_suite wrapper and sidecar ingestion.",
    )
    parser.add_argument(
        "--validate-execution-plan-evidence",
        action="store_true",
        help="Validate normalized conv/linear/pointwise execution plan evidence.",
    )
    args = parser.parse_args()

    if args.validate_transition_contract_classification:
        validate_transition_contract_classification()
        return
    if args.validate_model_suite_ingestion:
        validate_model_suite_ingestion()
        return
    if args.validate_execution_plan_evidence:
        validate_execution_plan_evidence()
        return

    if not args.config or not args.output_json:
        parser.error("--config and --output-json are required unless validating")

    config_path = repo_path(args.config)
    config = load_json(config_path)
    reason_bucket_contracts = load_transition_reason_bucket_contracts()
    specific_transition_contracts = load_specific_transition_contracts()
    selected = set(args.rows or [])
    rows = []
    for row_cfg in config.get("rows", []):
        if selected and row_cfg["row_id"] not in selected:
            continue
        rows.append(
            build_row(row_cfg, reason_bucket_contracts, specific_transition_contracts)
        )
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
