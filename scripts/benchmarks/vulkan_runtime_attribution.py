#!/usr/bin/env python3
# mypy: allow-untyped-defs

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "VulkanRuntimeAttributionReport.v0"

SUBMIT_ORIGIN_COUNTER_NAMES = (
    "total",
    "normal_cmd_submit_frequency",
    "stack_planned_recording_submit",
    "pre_stack_flush",
    "post_stack_flush",
    "explicit_synchronize",
    "tensor_cpu_readback",
    "fallback_readback",
    "retire_queue_drain",
    "profiling_timestamp_reset",
    "profiling_timestamp_readback",
    "shutdown",
    "debug_validation",
    "conv_prepack_upload",
    "unknown",
)

RETIRE_DRAIN_COUNTER_NAMES = (
    "total",
    "queue_submit_count",
    "blocking_wait_count",
    "poll_only_count",
    "pending_resource_count_total",
    "pending_bytes_total",
    "explicit_drain",
    "shutdown",
    "resource_pressure",
    "descriptor_pool_pressure",
    "command_buffer_recycle",
    "readback_preparation",
    "synchronize",
    "stack_scope_end",
    "decoder_phase",
    "setup_phase",
    "debug_validation",
    "unknown",
)

BUFFER_COPY_COUNTER_NAMES = (
    "total",
    "total_bytes",
    "explicit_copy",
    "contiguous",
    "view_materialization",
    "reshape_materialization",
    "permute_materialization",
    "transpose_materialization",
    "layout_conversion",
    "attention_materialization",
    "linear_materialization",
    "conv_materialization",
    "decoder_materialization",
    "backbone_materialization",
    "logical_noop_copy",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_key_value_line(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value.rstrip(",")
    return fields


def parse_runtime_label(label: str) -> tuple[str, dict[str, str]]:
    parts = label.split("|")
    category = parts[0] if parts and parts[0] else "unknown"
    fields: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    return category, fields


def classify_kernel(name: str, runtime_label: str) -> str:
    label = f"{name} {runtime_label}".lower()
    if "copy" in label or "buffer_to_buffer" in label or "image_to" in label:
        return "copy_or_layout"
    if "zero" in label or "fill" in label or "clear" in label:
        return "zero_or_fill"
    if "conv" in label:
        return "conv"
    if (
        "softmax" in label
        or "attention" in label
        or "sdpa" in label
        or "bmm" in label
    ):
        return "attention"
    if "upsample" in label or "interpolate" in label:
        return "upsample"
    if "mm" in label or "linear" in label or "matmul" in label:
        return "mm_or_linear"
    if "norm" in label:
        return "norm"
    if "binary" in label or "add" in label or "mul" in label or "sub" in label:
        return "elementwise"
    return "other"


def parse_gpu_timestamp_log(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    ignored = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = parse_key_value_line(line)
            if fields.get("gpu_timestamp") is None and not line.startswith(
                "gpu_timestamp "
            ):
                ignored += 1
                continue
            runtime = fields.get("runtime")
            duration_text = fields.get("duration_ns")
            if runtime is None or duration_text is None:
                ignored += 1
                continue
            try:
                duration_ns = int(duration_text)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid duration_ns={duration_text}"
                ) from exc
            if duration_ns < 0:
                raise ValueError(f"{path}:{line_number}: negative duration_ns")
            category, label_fields = parse_runtime_label(runtime)
            submit_phase = fields.get(
                "submit_phase",
                fields.get("phase", label_fields.get("phase", "not_available")),
            )
            stack_phase = fields.get(
                "stack_phase",
                label_fields.get("stack_phase", "not_available"),
            )
            stack_block = fields.get(
                "stack_block",
                label_fields.get("stack_block", "not_available"),
            )
            rows.append(
                {
                    "source_log": str(path),
                    "line_number": line_number,
                    "reason": fields.get("reason", "unknown"),
                    "name": fields.get("name", "unknown"),
                    "runtime_label": runtime,
                    "category": category,
                    "kernel": label_fields.get("kernel", fields.get("name", category)),
                    "kernel_class": classify_kernel(fields.get("name", ""), runtime),
                    "recent_op": fields.get("recent_op", "not_available"),
                    "submit_phase": submit_phase,
                    "stack_phase": stack_phase,
                    "stack_block": stack_block,
                    "duration_ns": duration_ns,
                    "has_phase_metadata": any(
                        key in fields or key in label_fields
                        for key in (
                            "phase",
                            "submit_phase",
                            "stack_phase",
                            "measurement_phase",
                        )
                    ),
                    "has_iteration_metadata": any(
                        key in fields or key in label_fields
                        for key in ("iteration", "repeat", "timed_iteration")
                    ),
                }
            )
    return rows, ignored


def summarize_timestamp_rows(
    rows: list[dict[str, Any]],
    *,
    timed_iteration_count: int | None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["category"]),
            str(row["kernel"]),
            str(row["name"]),
            str(row["reason"]),
            str(row["runtime_label"]),
        )
        group = groups.setdefault(
            key,
            {
                "key": (
                    f"{row['category']}:{row['kernel']}:{row['name']}:"
                    f"{row['runtime_label']}"
                ),
                "category": row["category"],
                "kernel": row["kernel"],
                "name": row["name"],
                "reason": row["reason"],
                "runtime_label": row["runtime_label"],
                "count": 0,
                "duration_ns_sum": 0,
                "duration_ns_max": 0,
                "source_logs": [],
            },
        )
        group["count"] += 1
        group["duration_ns_sum"] += int(row["duration_ns"])
        group["duration_ns_max"] = max(
            int(group["duration_ns_max"]),
            int(row["duration_ns"]),
        )
        if row["source_log"] not in group["source_logs"]:
            group["source_logs"].append(row["source_log"])

    out = []
    for group in groups.values():
        count = int(group["count"])
        total_ms = int(group["duration_ns_sum"]) / 1.0e6
        mean_ms = total_ms / count if count else 0.0
        result = {
            **group,
            "total_gpu_ms": total_ms,
            "mean_gpu_ms": mean_ms,
            "max_gpu_ms": int(group["duration_ns_max"]) / 1.0e6,
            "estimated_per_timed_iteration_gpu_ms": (
                total_ms / timed_iteration_count
                if timed_iteration_count and timed_iteration_count > 0
                else None
            ),
            "estimated_per_timed_iteration_count": (
                count / timed_iteration_count
                if timed_iteration_count and timed_iteration_count > 0
                else None
            ),
        }
        out.append(result)
    out.sort(key=lambda item: (-float(item["total_gpu_ms"]), item["category"]))
    return out


def summarize_by_key(
    rows: list[dict[str, Any]],
    key_name: str,
    *,
    timed_iteration_count: int | None,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(key_name, "not_available"))
        group = groups.setdefault(
            key,
            {
                "key": key,
                "count": 0,
                "duration_ns_sum": 0,
                "duration_ns_max": 0,
            },
        )
        group["count"] += 1
        group["duration_ns_sum"] += int(row["duration_ns"])
        group["duration_ns_max"] = max(
            int(group["duration_ns_max"]),
            int(row["duration_ns"]),
        )
    out = []
    for group in groups.values():
        total_ms = int(group["duration_ns_sum"]) / 1.0e6
        count = int(group["count"])
        out.append(
            {
                **group,
                "total_gpu_ms": total_ms,
                "mean_gpu_ms": total_ms / count if count else 0.0,
                "max_gpu_ms": int(group["duration_ns_max"]) / 1.0e6,
                "estimated_per_timed_iteration_gpu_ms": (
                    total_ms / timed_iteration_count
                    if timed_iteration_count and timed_iteration_count > 0
                    else None
                ),
            }
        )
    out.sort(key=lambda item: (-float(item["total_gpu_ms"]), item["key"]))
    return out


def summarize_stack_phase(
    rows: list[dict[str, Any]],
    *,
    timed_iteration_count: int | None,
) -> list[dict[str, Any]]:
    keyed_rows = []
    for row in rows:
        copy = dict(row)
        copy["stack_phase_block"] = (
            f"{row.get('stack_phase', 'not_available')}:"
            f"block:{row.get('stack_block', 'not_available')}"
        )
        keyed_rows.append(copy)
    return summarize_by_key(
        keyed_rows,
        "stack_phase_block",
        timed_iteration_count=timed_iteration_count,
    )


def timed_iteration_count_from_benchmark(payload: dict[str, Any]) -> tuple[int | None, str]:
    forward = payload.get("single_image_forward_device_resident")
    if isinstance(forward, dict) and isinstance(forward.get("count"), int):
        return int(forward["count"]), "single_image_forward_device_resident.count"
    repeats = payload.get("repeats")
    if isinstance(repeats, int):
        return repeats, "repeats"
    return None, "unavailable"


def named_counter_dict(value: Any, names: tuple[str, ...]) -> dict[str, int] | None:
    if isinstance(value, list):
        return {
            names[index] if index < len(names) else f"index_{index}": int(item)
            for index, item in enumerate(value)
            if isinstance(item, int)
        }
    if isinstance(value, dict):
        if all(isinstance(item, int) for item in value.values()):
            return {str(key): int(item) for key, item in value.items()}
        if isinstance(value.get("field_counts"), dict):
            return None
    return None


def collect_counter_summary(payload: dict[str, Any]) -> dict[str, Any]:
    counters = payload.get("vulkan_debug_counters")
    if not isinstance(counters, dict):
        return {
            "source": "missing",
            "cpu_fallback_count": None,
            "sync_readback_count": None,
            "submit_origin_counters": None,
            "retire_drain_counters": None,
            "buffer_copy_counters": None,
        }
    return {
        "source": "vulkan_debug_counters",
        "cpu_fallback_count": counters.get("cpu_fallback_count"),
        "sync_readback_count": counters.get("sync_readback_count"),
        "submit_origin_counters": named_counter_dict(
            counters.get("submit_origin_counters"),
            SUBMIT_ORIGIN_COUNTER_NAMES,
        ),
        "retire_drain_counters": named_counter_dict(
            counters.get("retire_drain_counters"),
            RETIRE_DRAIN_COUNTER_NAMES,
        ),
        "buffer_copy_counters": named_counter_dict(
            counters.get("buffer_copy_counters"),
            BUFFER_COPY_COUNTER_NAMES,
        ),
        "raw_counter_snapshot_mode": counters.get("snapshot_mode"),
    }


def measurement_phase_delta(payload: dict[str, Any], phase: str) -> dict[str, Any]:
    rows = payload.get("vulkan_measurement_phase_counters")
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if not isinstance(row, dict) or row.get("name") != phase:
            continue
        delta = row.get("delta")
        return delta if isinstance(delta, dict) else {}
    return {}


def collect_measurement_counter_summary(
    payload: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    delta = measurement_phase_delta(payload, phase)
    if not delta:
        return {
            "source": "missing",
            "phase": phase,
            "submit_origin_counters": None,
            "retire_drain_counters": None,
            "buffer_copy_counters": None,
        }
    return {
        "source": "vulkan_measurement_phase_counters",
        "phase": phase,
        "submit_origin_counters": named_counter_dict(
            delta.get("submit_origin_counters"),
            SUBMIT_ORIGIN_COUNTER_NAMES,
        ),
        "retire_drain_counters": named_counter_dict(
            delta.get("retire_drain_counters"),
            RETIRE_DRAIN_COUNTER_NAMES,
        ),
        "buffer_copy_counters": named_counter_dict(
            delta.get("buffer_copy_counters"),
            BUFFER_COPY_COUNTER_NAMES,
        ),
        "cpu_fallback_count": delta.get("cpu_fallback_count"),
        "sync_readback_count": delta.get("sync_readback_count"),
    }


def phase_filter_quality(
    timestamp_rows: list[dict[str, Any]],
    benchmark_payload: dict[str, Any],
    phase: str,
) -> tuple[str, list[str]]:
    limitations: list[str] = []
    if not timestamp_rows:
        limitations.append("no_gpu_timestamp_rows_available")
        return "no_timestamp_rows", limitations
    profile = benchmark_payload.get("vulkan_gpu_timestamp_profile")
    if isinstance(profile, dict):
        if profile.get("target_phase") != phase:
            limitations.append("benchmark_gpu_timestamp_profile_phase_mismatch")
        elif profile.get("phase_filter_quality") == "isolated_after_warmup":
            return "isolated_after_warmup", limitations
    if not all(row["has_phase_metadata"] for row in timestamp_rows):
        limitations.append("timestamp_rows_do_not_all_carry_phase_metadata")
    if not all(row["has_iteration_metadata"] for row in timestamp_rows):
        limitations.append("timestamp_rows_do_not_all_carry_iteration_metadata")
    if "vulkan_measurement_phase_counters" not in benchmark_payload:
        limitations.append("benchmark_json_has_no_measurement_phase_counter_window")
    if limitations:
        return "unfiltered_log_estimate", limitations
    return "strict_phase_metadata_present", []


def build_report(
    *,
    benchmark_json_path: Path,
    timestamp_log_paths: list[Path],
    phase: str = "single_image_forward_device_resident",
    timed_iteration_count_override: int | None = None,
) -> dict[str, Any]:
    benchmark_payload = load_json(benchmark_json_path)
    timestamp_rows: list[dict[str, Any]] = []
    ignored_line_count = 0
    for path in timestamp_log_paths:
        rows, ignored = parse_gpu_timestamp_log(path)
        timestamp_rows.extend(rows)
        ignored_line_count += ignored
    timed_count, timed_source = timed_iteration_count_from_benchmark(
        benchmark_payload
    )
    if timed_iteration_count_override is not None:
        timed_count = timed_iteration_count_override
        timed_source = "cli_override"
    quality, limitations = phase_filter_quality(
        timestamp_rows,
        benchmark_payload,
        phase,
    )
    groups = summarize_timestamp_rows(
        timestamp_rows,
        timed_iteration_count=timed_count,
    )
    timestamp_total_gpu_ms = sum(
        int(row["duration_ns"]) for row in timestamp_rows
    ) / 1.0e6
    return {
        "schema": SCHEMA,
        "runtime_behavior_changed": False,
        "source_benchmark_json": str(benchmark_json_path),
        "source_timestamp_logs": [str(path) for path in timestamp_log_paths],
        "phase": phase,
        "gpu_timestamp_profile": benchmark_payload.get(
            "vulkan_gpu_timestamp_profile"
        ),
        "timed_iteration_count": timed_count,
        "timed_iteration_count_source": timed_source,
        "phase_filter_quality": quality,
        "limitations": limitations,
        "ignored_timestamp_line_count": ignored_line_count,
        "timestamp_event_count": len(timestamp_rows),
        "timestamp_total_gpu_ms": timestamp_total_gpu_ms,
        "timestamp_estimated_per_timed_iteration_gpu_ms": (
            timestamp_total_gpu_ms / timed_count
            if timed_count and timed_count > 0
            else None
        ),
        "group_count": len(groups),
        "groups": groups,
        "kernel_class_groups": summarize_by_key(
            timestamp_rows,
            "kernel_class",
            timed_iteration_count=timed_count,
        ),
        "submit_phase_groups": summarize_by_key(
            timestamp_rows,
            "submit_phase",
            timed_iteration_count=timed_count,
        ),
        "stack_phase_groups": summarize_stack_phase(
            timestamp_rows,
            timed_iteration_count=timed_count,
        ),
        "recent_op_groups": summarize_by_key(
            timestamp_rows,
            "recent_op",
            timed_iteration_count=timed_count,
        ),
        "counters": collect_counter_summary(benchmark_payload),
        "measurement_counters": collect_measurement_counter_summary(
            benchmark_payload,
            phase,
        ),
        "benchmark_summary": {
            "benchmark_name": benchmark_payload.get("benchmark_name"),
            "device": benchmark_payload.get("device"),
            "encoder": benchmark_payload.get("encoder"),
            "input_size": benchmark_payload.get("input_size"),
            "performance_valid": benchmark_payload.get("performance_valid"),
            "performance_invalid_reasons": benchmark_payload.get(
                "performance_invalid_reasons"
            ),
        },
    }


def markdown_table(title: str, rows: list[dict[str, Any]], limit: int) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| key | count | GPU ms | GPU ms/iter |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows[:limit]:
        key = str(row.get("key") or row.get("category") or row.get("kernel"))
        per_iter = row.get("estimated_per_timed_iteration_gpu_ms")
        lines.append(
            "| {key} | {count} | {total:.3f} | {per_iter} |".format(
                key=key.replace("|", "\\|"),
                count=row.get("count", 0),
                total=float(row.get("total_gpu_ms", 0.0)),
                per_iter=(
                    f"{float(per_iter):.3f}" if per_iter is not None else "n/a"
                ),
            )
        )
    lines.append("")
    return lines


def write_markdown(path: Path, report: dict[str, Any], limit: int) -> None:
    per_iter = report.get("timestamp_estimated_per_timed_iteration_gpu_ms")
    per_iter_text = f"{float(per_iter):.3f} ms" if per_iter is not None else "n/a"
    lines = [
        f"# {SCHEMA}",
        "",
        f"- phase: `{report.get('phase')}`",
        f"- phase filter quality: `{report.get('phase_filter_quality')}`",
        f"- timestamp events: `{report.get('timestamp_event_count')}`",
        f"- timed iterations: `{report.get('timed_iteration_count')}`",
        f"- GPU timestamp total: `{report.get('timestamp_total_gpu_ms'):.3f} ms`",
        f"- GPU timestamp per timed iteration: `{per_iter_text}`",
        "",
    ]
    limitations = report.get("limitations") or []
    if limitations:
        lines.extend(["## Limitations", ""])
        lines.extend(f"- `{item}`" for item in limitations)
        lines.append("")
    lines.extend(
        markdown_table(
            "GPU Time By Kernel Class",
            report.get("kernel_class_groups", []),
            limit,
        )
    )
    lines.extend(markdown_table("GPU Time By Kernel/Runtime", report["groups"], limit))
    lines.extend(
        markdown_table(
            "GPU Time By Submit Phase",
            report.get("submit_phase_groups", []),
            limit,
        )
    )
    lines.extend(
        markdown_table(
            "GPU Time By Stack Phase",
            report.get("stack_phase_groups", []),
            limit,
        )
    )
    lines.extend(
        markdown_table(
            "GPU Time By Recent Op",
            report.get("recent_op_groups", []),
            limit,
        )
    )
    measurement = report.get("measurement_counters") or {}
    lines.extend(["## Measurement Counters", ""])
    for name in (
        "cpu_fallback_count",
        "sync_readback_count",
        "submit_origin_counters",
        "retire_drain_counters",
        "buffer_copy_counters",
    ):
        lines.append(f"- `{name}`: `{measurement.get(name)}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_report(args: argparse.Namespace) -> int:
    payload = build_report(
        benchmark_json_path=Path(args.benchmark_json),
        timestamp_log_paths=[Path(path) for path in args.timestamp_log],
        phase=args.phase,
        timed_iteration_count_override=args.timed_iteration_count,
    )
    write_json(Path(args.out), payload)
    if args.markdown:
        write_markdown(Path(args.markdown), payload, args.markdown_top)
    print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a behavior-neutral Vulkan runtime attribution report."
    )
    parser.add_argument("--benchmark-json", required=True)
    parser.add_argument("--timestamp-log", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--phase",
        default="single_image_forward_device_resident",
        help="Benchmark measurement phase represented by the timestamp log.",
    )
    parser.add_argument("--timed-iteration-count", type=int)
    parser.add_argument("--markdown")
    parser.add_argument("--markdown-top", type=int, default=20)
    args = parser.parse_args(argv)
    return cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
