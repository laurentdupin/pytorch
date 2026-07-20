from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import gc
import importlib
import importlib.util
import os
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.utils._pytree as pytree

from torch.vulkan._graph_evidence import (
    ExternalGraphEvidenceSetupError,
    MEASURED_STATUS,
    parse_input_shape,
    require_external_assets,
    sha256_file,
    source_git_sha,
    write_evidence,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
_GRAPH_PROGRAM_INVOCATION_COUNTER_NAMES = (
    "scope_begun",
    "normal_submit_token_capture",
    "aborted_submit",
    "rejected_incompatible_state",
    "bounded_region_host_sync_rejected",
    "scratch_captured",
    "scratch_reused",
    "scratch_transient_overflow",
    "scratch_retire_enqueued",
    "scratch_immediate_release",
    "resource_arena_immediate_release",
    "resource_arena_retire_enqueued",
    "resource_arena_unsafe_slot_leak",
    "resource_arena_retirement_failure",
)
_SUBMIT_ORIGIN_COUNTER_NAMES = (
    "total_queue_submits",
    "normal_cmd_submit_frequency",
    "stack_planned_recording_submit",
    "pre_stack_flush",
    "post_stack_flush",
    "explicit_synchronize",
    "tensor_cpu_readback",
    "host_upload",
    "fallback_readback",
    "retire_queue_drain",
    "profiling_timestamp_reset",
    "profiling_timestamp_readback",
    "shutdown",
    "debug_validation",
    "conv_prepack_upload",
    "pending_command_flush",
    "unknown",
)
_ROUTE_DIAGNOSTIC_FIELDS = (
    "op",
    "lane",
    "decision",
    "reason",
    "family",
    "telemetry",
    "hard_fail",
)
_RUNTIME_POLICY_DIAGNOSTIC_FIELDS = (
    "workload",
    "source_workload",
    "model_domain",
    "execution_phase",
    "tensor_role",
    "fixed_shape_graph",
    "prefer_packed_layout_propagation",
    "backend_route",
    "linear_kernel_family",
    "norm_kernel_family",
    "attention_kernel_family",
    "attention_execution_strategy",
    "inferred_from_label",
)
_LINEAR_PACK_RESIDENCY_FIELDS = (
    "count",
    "created",
    "reused",
    "packed_bytes",
    "raw_weight_bytes",
    "raw_bias_bytes",
    "raw_weight_vulkan",
    "retain_unpacked",
)
_LONG_SESSION_SOAK_GATE_DURATION_SECONDS = 600
_LONG_SESSION_SOAK_GATE_MINIMUM_INVOCATIONS = 3000
_LONG_SESSION_SOAK_MEMORY_LIMIT_RATIO = 1.05
_LONG_SESSION_SOAK_RECAPTURE_INTERVAL = 250
_LONG_SESSION_SOAK_MEMORY_SAMPLE_INTERVAL = 50
_LONG_SESSION_SOAK_DEVICE_NAME = "AMD Radeon RX 9070"


def _named_counter_snapshot(
    names: tuple[str, ...], values: list[int], label: str
) -> dict[str, int]:
    if len(names) != len(values):
        raise RuntimeError(
            f"{label} counter schema has {len(names)} names for {len(values)} values"
        )
    return dict(zip(names, values, strict=True))


def _artifact_prefix(value: str) -> str:
    if (
        not value
        or value != value.strip()
        or "/" in value
        or "\\" in value
        or not value.isascii()
        or not value[0].isalnum()
        or not all(character.isalnum() or character in "_-" for character in value)
    ):
        raise argparse.ArgumentTypeError(
            "--artifact-prefix must start with an ASCII letter or digit and contain "
            "only ASCII letters, digits, underscores, or hyphens"
        )
    return value


def _nonnegative_repeat_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("repeat count must be an integer") from error
    if count < 0:
        raise argparse.ArgumentTypeError("repeat count must be nonnegative")
    return count


def _positive_repeat_count(value: str) -> int:
    count = _nonnegative_repeat_count(value)
    if count == 0:
        raise argparse.ArgumentTypeError("measurement repeat count must be positive")
    return count


def _device_index(value: str) -> int:
    try:
        index = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("device index must be an integer") from error
    if index < 0:
        raise argparse.ArgumentTypeError("device index must be nonnegative")
    return index


def _artifact_output_paths(output_dir: Path, prefix: str) -> tuple[Path, Path]:
    return (
        output_dir / f"{prefix}_export_census.json",
        output_dir / f"{prefix}_export_parity.json",
    )


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _load_adapter(spec: str) -> Callable[[Path, Path], Mapping[str, Any]]:
    try:
        module_spec, factory_name = spec.rsplit(":", 1)
    except ValueError as error:
        raise ValueError("--adapter must be MODULE:FACTORY or FILE.py:FACTORY") from error
    module_path = Path(module_spec)
    if module_path.is_file():
        loaded_spec = importlib.util.spec_from_file_location(
            "vulkan_graph_evidence_adapter", module_path
        )
        if loaded_spec is None or loaded_spec.loader is None:
            raise ValueError(f"Unable to load adapter file {module_path}")
        module = importlib.util.module_from_spec(loaded_spec)
        loaded_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ValueError(f"Adapter factory {factory_name!r} is not callable")
    return factory


def _adapter_identity(spec: str) -> str:
    module_spec, factory_name = spec.rsplit(":", 1)
    module_path = Path(module_spec)
    if module_path.suffix == ".py":
        return f"{module_path.name}:{factory_name}"
    return spec


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    return [
        leaf
        for leaf in pytree.tree_leaves(value)
        if isinstance(leaf, torch.Tensor)
    ]


def _require_cpu_inputs(value: Any, field: str) -> tuple[Any, ...]:
    args = value if isinstance(value, tuple) else (value,)
    if not args or any(tensor.device.type != "cpu" for tensor in _tensor_leaves(args)):
        raise ValueError(f"Adapter {field} must contain CPU tensors")
    return args


def _input_shapes(args: tuple[Any, ...]) -> list[list[int]]:
    return [list(tensor.shape) for tensor in _tensor_leaves(args)]


def _state_replay_mapping(value: Any) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, tuple | list) or not value:
        raise ValueError(
            "Adapter state_replay_input_from_output must be a non-empty sequence"
        )
    mapping: list[tuple[int, int]] = []
    for index, pair in enumerate(value):
        if (
            not isinstance(pair, tuple | list)
            or len(pair) != 2
            or any(
                type(leaf_index) is not int or leaf_index < 0
                for leaf_index in pair
            )
        ):
            raise ValueError(
                "Adapter state replay mapping row "
                f"{index} must contain two nonnegative integer leaf indices"
            )
        mapping.append((pair[0], pair[1]))
    input_indices = [input_index for input_index, _ in mapping]
    output_indices = [output_index for _, output_index in mapping]
    if len(set(input_indices)) != len(input_indices):
        raise ValueError("Adapter state replay input leaf indices must be unique")
    if len(set(output_indices)) != len(output_indices):
        raise ValueError("Adapter state replay output leaf indices must be unique")
    return tuple(mapping)


def _planning_context(
    args: argparse.Namespace,
    inputs: tuple[Any, ...],
) -> torch.vulkan.VulkanGraphPlanningContext:
    fixed_shape = None
    if args.planning_fixed_shape_graph_input:
        fixed_shape = tuple(_input_shapes(inputs)[0])
    return torch.vulkan.VulkanGraphPlanningContext(
        model_domain=args.planning_model_domain,
        execution_phase=args.planning_execution_phase,
        prefer_packed_layout_propagation=(
            args.planning_prefer_packed_layout_propagation
        ),
        fixed_shape_graph_input_sizes=fixed_shape,
    )


def _parity(actual: Any, expected: Any) -> dict[str, float | int]:
    actual_tensors = _tensor_leaves(actual)
    expected_tensors = _tensor_leaves(expected)
    if len(actual_tensors) != len(expected_tensors):
        raise RuntimeError(
            "Output tensor leaf count mismatch: "
            f"{len(actual_tensors)} != {len(expected_tensors)}"
        )
    maximum = 0.0
    mean_values: list[float] = []
    for actual_tensor, expected_tensor in zip(actual_tensors, expected_tensors):
        difference = (actual_tensor.cpu() - expected_tensor.cpu()).abs()
        maximum = max(maximum, float(difference.max().item()))
        mean_values.append(float(difference.mean().item()))
    return {
        "tensor_leaf_count": len(actual_tensors),
        "max_abs": maximum,
        "mean_abs": statistics.fmean(mean_values) if mean_values else 0.0,
    }


def _assert_close(actual: Any, expected: Any, atol: float, rtol: float) -> None:
    actual_tensors = _tensor_leaves(actual)
    expected_tensors = _tensor_leaves(expected)
    if len(actual_tensors) != len(expected_tensors):
        raise RuntimeError("Output tensor leaf counts differ")
    for actual_tensor, expected_tensor in zip(actual_tensors, expected_tensors):
        torch.testing.assert_close(
            actual_tensor.cpu(), expected_tensor.cpu(), atol=atol, rtol=rtol
        )


def _device_runtime_identity(device: torch.device) -> dict[str, Any]:
    properties = torch.vulkan.get_device_properties(device)
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    files: dict[str, Any] = {}
    for name in ("torch_cpu.dll", "torch_python.dll", "c10.dll"):
        path = torch_lib / name
        files[name] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    extension = Path(torch._C.__file__).resolve()
    files[extension.name] = {
        "path": str(extension),
        "sha256": sha256_file(extension),
    }
    return {
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "device": str(device),
        "device_index": properties.index,
        "device_name": properties.name,
        "device_type": properties.type,
        "total_memory_bytes": properties.total_memory,
        "vendor_id": properties.vendor_id,
        "device_id": properties.device_id,
        "driver_version": properties.driver_version,
        "api_version": properties.api_version,
        "loaded_files": files,
    }


def _execution_plan_summary(
    program: torch.vulkan.VulkanGraphProgram,
) -> dict[str, Any] | None:
    report = getattr(program, "cpp_plan_report", None)
    if report is None:
        return None
    summary = {
        "mode": getattr(program, "execution_mode", "python_correctness_executor"),
        "status": report.status,
        "reason": report.reason,
        "plan_class": report.plan_class,
        "plan_version": report.plan_version,
        "planning_model_domain": report.planning_model_domain,
        "planning_execution_phase": report.planning_execution_phase,
        "planning_prefer_packed_layout_propagation": (
            report.planning_prefer_packed_layout_propagation
        ),
        "planning_fixed_shape_graph_input_sizes": (
            report.planning_fixed_shape_graph_input_sizes
        ),
        "input_count": report.input_count,
        "instruction_count": report.instruction_count,
        "effect_instruction_count": report.effect_instruction_count,
        "graph_scalar_instruction_count": (
            report.graph_scalar_instruction_count
        ),
        "list_projection_instruction_count": (
            report.list_projection_instruction_count
        ),
        "list_argument_count": report.list_argument_count,
        "invocation_value_slot_count": report.invocation_value_slot_count,
        "invocation_list_slot_count": report.invocation_list_slot_count,
        "invocation_stack_capacity": report.invocation_stack_capacity,
        "dead_input_reuse_instruction_count": (
            report.dead_input_reuse_instruction_count
        ),
        "resource_slot_count": report.resource_slot_count,
        "resource_value_count": report.resource_value_count,
        "resource_writer_instruction_count": (
            report.resource_writer_instruction_count
        ),
        "resource_arena_flight_depth": report.resource_arena_flight_depth,
        "recorded_partition_count": report.recorded_partition_count,
        "recorded_partition_instruction_count": (
            report.recorded_partition_instruction_count
        ),
        "resource_alias_extended_lifetime_count": (
            report.resource_alias_extended_lifetime_count
        ),
        "resource_alias_escape_rejection_count": (
            report.resource_alias_escape_rejection_count
        ),
        "value_count": report.value_count,
        "output_count": report.output_count,
        "submission_owned": report.submission_owned,
    }
    plan = getattr(program, "cpp_plan", None)
    if plan is not None:
        summary.update(
            {
                "invocation_generation": plan.invocation_generation(),
                "last_submission_value": plan.last_submission_value(),
                "last_submission_complete": plan.last_submission_complete(),
                "dead_input_reuse_count": plan.dead_input_reuse_count(),
                "resource_arena_generation_count": (
                    plan.resource_arena_generation_count()
                ),
                "resource_arena_capture_count": (
                    plan.resource_arena_capture_count()
                ),
                "resource_arena_reuse_count": plan.resource_arena_reuse_count(),
                "resource_arena_spill_count": plan.resource_arena_spill_count(),
                "resource_write_count": plan.resource_write_count(),
                "resource_writer_bypass_count": (
                    plan.resource_writer_bypass_count()
                ),
                "recorded_partition_prime_count": (
                    plan.recorded_partition_prime_count()
                ),
                "recorded_partition_capture_count": (
                    plan.recorded_partition_capture_count()
                ),
                "recorded_partition_replay_count": (
                    plan.recorded_partition_replay_count()
                ),
                "recorded_partition_failure_count": (
                    plan.recorded_partition_failure_count()
                ),
                "recorded_partition_represented_dispatch_count": (
                    plan.recorded_partition_represented_dispatch_count()
                ),
            }
        )
    return summary


def _graph_counts(program: torch.vulkan.VulkanGraphProgram) -> dict[str, Any]:
    census = program.census
    execution_plan = _execution_plan_summary(program)

    def target_counts(classification: str) -> dict[str, int]:
        counts: dict[str, int] = Counter(
            node.target
            for node in census.nodes
            if node.classification == classification
        )
        return dict(sorted(counts.items()))

    return {
        "captured_node_count": census.captured_node_count,
        "call_function_node_count": census.call_function_node_count,
        "statically_lowered": census.lowered_vulkan_node_count,
        "graph_owned_prepacked_contexts": sum(
            getattr(report, "created_context_count", 0)
            for report in _lowering_report_objects(program).values()
            if report is not None
        ),
        "eager_vulkan_dispatch": census.direct_vulkan_node_count,
        "composite_runtime_verified": census.composite_node_count,
        "conditionally_supported": 0,
        "unsupported_at_lower_time": census.unsupported_node_count,
        "direct_vulkan_by_target": target_counts("direct_vulkan"),
        "composite_by_target": target_counts("composite"),
        "lowered_vulkan_by_target": target_counts("lowered_vulkan"),
        "unsupported_by_target": target_counts("unsupported"),
        "partition_candidates": {
            "status": (
                execution_plan["status"]
                if execution_plan is not None
                else "not_planned_python_correctness_executor"
            ),
            "vulkan_only_candidate_count": int(
                execution_plan is not None and execution_plan["mode"] == "cpp_plan"
            ),
        },
    }


def _lowering_report_objects(
    program: torch.vulkan.VulkanGraphProgram,
) -> dict[str, Any]:
    return {
        "input_normalization": getattr(program, "input_normalization", None),
        "static_factory_constants": getattr(
            program, "static_factory_constants", None
        ),
        "lifted_tensor_constants": getattr(
            program, "lifted_tensor_constants", None
        ),
        "fresh_detach_functionalization": getattr(
            program, "fresh_detach_functionalization", None
        ),
        "fresh_relu_functionalization": getattr(
            program, "fresh_relu_functionalization", None
        ),
        "static_inference_identities": getattr(
            program, "static_inference_identities", None
        ),
        "static_identity_advanced_indices": getattr(
            program, "static_identity_advanced_indices", None
        ),
        "static_gqa_repeats": getattr(program, "static_gqa_repeats", None),
        "static_sdpa_fusions": getattr(program, "static_sdpa_fusions", None),
        "tensor_placement": getattr(program, "tensor_placement", None),
        "linear_lowering": getattr(program, "linear_lowering", None),
        "static_linear_gelu_regions": getattr(
            program, "static_linear_gelu_regions", None
        ),
        "conv2d_lowering": getattr(program, "conv2d_lowering", None),
        "layernorm_lowering": getattr(program, "layernorm_lowering", None),
        "static_add_layernorm_regions": getattr(
            program, "static_add_layernorm_regions", None
        ),
        "static_conv2d_relu_conv2d_regions": getattr(
            program, "static_conv2d_relu_conv2d_regions", None
        ),
        "static_conv2d_relu_regions": getattr(
            program, "static_conv2d_relu_regions", None
        ),
        "vulkan_graph_regions": getattr(program, "vulkan_graph_regions", None),
    }


def _lowering_reports(program: torch.vulkan.VulkanGraphProgram) -> dict[str, Any]:
    return {
        name: _jsonable(report)
        for name, report in _lowering_report_objects(program).items()
    }


def _graph_structure(program: torch.vulkan.VulkanGraphProgram) -> dict[str, int]:
    nodes = tuple(program.graph_module.graph.nodes)
    return {
        "graph_inputs": sum(node.op == "placeholder" for node in nodes),
        "graph_outputs": sum(node.op == "output" for node in nodes),
        "graph_constants": sum(node.op == "get_attr" for node in nodes),
        "normalized_node_count": len(nodes),
    }


def _row_fields(row: str, prefix: str) -> dict[str, str]:
    return {
        key: value
        for token in row[len(prefix) :].split()
        if "=" in token
        for key, value in (token.split("=", 1),)
    }


def _aggregate_diagnostic_rows(
    rows: list[str],
    prefix: str,
    field_names: tuple[str, ...],
    required_field: str,
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        if not row.startswith(prefix):
            continue
        fields = _row_fields(row, prefix)
        if required_field not in fields:
            continue
        missing = [field for field in field_names if field not in fields]
        if missing:
            raise RuntimeError(
                f"{prefix.strip()} diagnostic row is missing fields: {missing}"
            )
        counts[tuple(fields[field] for field in field_names)] += 1
    return [
        {
            **dict(zip(field_names, values, strict=True)),
            "count": count,
        }
        for values, count in sorted(counts.items())
    ]


def _planning_diagnostic_summary(
    route_rows: list[str], runtime_policy_rows: list[str]
) -> dict[str, Any]:
    route_decisions = _aggregate_diagnostic_rows(
        route_rows,
        "vulkan_route ",
        _ROUTE_DIAGNOSTIC_FIELDS,
        "lane",
    )
    runtime_policies = _aggregate_diagnostic_rows(
        runtime_policy_rows,
        "runtime_policy ",
        _RUNTIME_POLICY_DIAGNOSTIC_FIELDS,
        "source_workload",
    )
    return {
        "route_lanes": sorted({row["lane"] for row in route_decisions}),
        "route_decisions": route_decisions,
        "runtime_model_domains": sorted(
            {row["model_domain"] for row in runtime_policies}
        ),
        "runtime_execution_phases": sorted(
            {row["execution_phase"] for row in runtime_policies}
        ),
        "runtime_inferred_from_label_count": sum(
            row["count"]
            for row in runtime_policies
            if row["inferred_from_label"] == "1"
        ),
        "runtime_policies": runtime_policies,
    }


@contextlib.contextmanager
def _planning_diagnostic_capture(directory: Path, phase: str):
    route_path = directory / f"{phase}_route.log"
    runtime_policy_path = directory / f"{phase}_runtime_policy.log"
    environment = {
        "PYTORCH_VULKAN_ROUTE_LOG": str(route_path),
        "PYTORCH_VULKAN_RUNTIME_POLICY_LOG": str(runtime_policy_path),
    }
    previous = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    try:
        yield route_path, runtime_policy_path
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _captured_planning_diagnostics(
    paths: tuple[Path, Path]
) -> dict[str, Any]:
    route_path, runtime_policy_path = paths
    route_rows = (
        route_path.read_text(encoding="utf-8").splitlines()
        if route_path.is_file()
        else []
    )
    runtime_policy_rows = (
        runtime_policy_path.read_text(encoding="utf-8").splitlines()
        if runtime_policy_path.is_file()
        else []
    )
    return _planning_diagnostic_summary(route_rows, runtime_policy_rows)


def _int_summary_row(rows: list[str], prefix: str) -> dict[str, int]:
    summary = next((row for row in rows if row.startswith(prefix)), None)
    if summary is None:
        raise RuntimeError(f"Vulkan snapshot is missing {prefix.strip()!r}")
    fields = _row_fields(summary, prefix)
    return {key: int(value) for key, value in sorted(fields.items())}


def _allocation_group_summary(
    records: list[dict[str, str]], field_name: str
) -> list[dict[str, int | str]]:
    counts: Counter[str] = Counter()
    allocated_bytes: Counter[str] = Counter()
    for record in records:
        value = record[field_name]
        counts[value] += 1
        allocated_bytes[value] += int(record["allocated_bytes"])
    return [
        {
            "value": value,
            "allocation_count": counts[value],
            "allocated_bytes": allocated_bytes[value],
        }
        for value in sorted(counts)
    ]


def _allocator_residency_summary(rows: list[str]) -> dict[str, Any]:
    prefix = "vulkan_memory_residency "
    records = [
        _row_fields(row, prefix) for row in rows if row.startswith(prefix)
    ]
    required_fields = {
        "kind",
        "state",
        "role",
        "requested_bytes",
        "allocated_bytes",
        "owns_memory",
        "label",
    }
    for record in records:
        missing = required_fields.difference(record)
        if missing:
            raise RuntimeError(
                "Vulkan memory residency row is missing fields: "
                f"{sorted(missing)}"
            )
    return {
        "allocation_count": len(records),
        "requested_bytes": sum(int(row["requested_bytes"]) for row in records),
        "allocated_bytes": sum(int(row["allocated_bytes"]) for row in records),
        "owned_allocated_bytes": sum(
            int(row["allocated_bytes"])
            for row in records
            if row["owns_memory"] == "1"
        ),
        "by_kind": _allocation_group_summary(records, "kind"),
        "by_state": _allocation_group_summary(records, "state"),
        "by_role": _allocation_group_summary(records, "role"),
        "by_label": _allocation_group_summary(records, "label"),
    }


def _linear_pack_residency_summary(rows: list[str]) -> dict[str, int]:
    prefix = "linear_pack_residency "
    totals = {field: 0 for field in _LINEAR_PACK_RESIDENCY_FIELDS}
    row_count = 0
    for row in rows:
        if not row.startswith(prefix):
            continue
        fields = _row_fields(row, prefix)
        missing = [
            field for field in _LINEAR_PACK_RESIDENCY_FIELDS if field not in fields
        ]
        if missing:
            raise RuntimeError(
                f"Linear pack residency row is missing fields: {missing}"
            )
        row_count += 1
        for field in _LINEAR_PACK_RESIDENCY_FIELDS:
            totals[field] += int(fields[field])
    return {"row_count": row_count, **totals}


def _memory_usage_snapshot() -> dict[str, Any]:
    rows = list(torch.ops.vulkan_prepack.vulkan_memory_residency_snapshot())
    fields = _int_summary_row(rows, "vulkan_memory_summary ")
    return {
        "live_bytes": fields["live_bytes"],
        "high_water_bytes": fields["high_water_bytes"],
        "allocator_residency": _allocator_residency_summary(rows),
    }


def _begin_memory_phase() -> int:
    torch.ops.vulkan_prepack.reset_vulkan_memory_residency_snapshot()
    torch.ops.vulkan_prepack.reset_packed_weight_residency_snapshot()
    torch.ops.vulkan_prepack.reset_linear_pack_residency_snapshot()
    return _memory_usage_snapshot()["live_bytes"]


def _finish_memory_phase(baseline_live_bytes: int) -> dict[str, Any]:
    snapshot = _memory_usage_snapshot()
    return {
        "baseline_live_bytes": baseline_live_bytes,
        "end_live_bytes": snapshot["live_bytes"],
        "high_water_bytes": snapshot["high_water_bytes"],
        "peak_delta_bytes": max(
            0, snapshot["high_water_bytes"] - baseline_live_bytes
        ),
        "residency": {
            "allocator": snapshot["allocator_residency"],
            "packed_weight_cache": _int_summary_row(
                list(torch.ops.vulkan_prepack.packed_weight_residency_snapshot()),
                "packed_weight_residency_summary ",
            ),
            "linear_pack": _linear_pack_residency_summary(
                list(torch.ops.vulkan_prepack.linear_pack_residency_snapshot())
            ),
        },
    }


def _counter_delta(
    before: dict[str, int], after: dict[str, int]
) -> dict[str, int]:
    if before.keys() != after.keys():
        raise ValueError("Counter snapshots use different schemas")
    return {name: after[name] - before[name] for name in before}


def _soak_memory_limit(reference_bytes: int) -> int:
    return int(reference_bytes * _LONG_SESSION_SOAK_MEMORY_LIMIT_RATIO)


def _evaluate_long_session_soak_gate(record: dict[str, Any]) -> dict[str, Any]:
    configuration = record["configuration"]
    measurement = record["measurement"]
    memory = measurement["memory"]
    runtime_counters = measurement["runtime_counters"]
    graph_invocation_counters = measurement["submission_counters"][
        "graph_program_invocation"
    ]
    submit_counters = measurement["submission_counters"]["submit_origin"]
    preflight_peak = memory["replacement_preflight"]["high_water_bytes"]
    soak_phase = memory["soak"]
    checks = {
        "rx_9070_adapter": (
            record["device_name"] == _LONG_SESSION_SOAK_DEVICE_NAME
        ),
        "ten_minute_duration": (
            measurement["elapsed_seconds"]
            >= _LONG_SESSION_SOAK_GATE_DURATION_SECONDS
        ),
        "minimum_invocations": (
            measurement["invocation_count"]
            >= _LONG_SESSION_SOAK_GATE_MINIMUM_INVOCATIONS
        ),
        "periodic_guard_recapture": (
            measurement["recapture_count"]
            == max(
                0,
                (measurement["invocation_count"] - 1)
                // _LONG_SESSION_SOAK_RECAPTURE_INTERVAL,
            )
        ),
        "all_outputs_checked": (
            measurement["parity_check_count"]
            == measurement["invocation_count"]
        ),
        "zero_cpu_fallback": runtime_counters["cpu_fallback"] == 0,
        "zero_unexpected_sync_readback": (
            runtime_counters["sync_readback"] == 0
        ),
        "zero_resource_arena_unsafe_slot_leak": (
            graph_invocation_counters["resource_arena_unsafe_slot_leak"] == 0
        ),
        "zero_resource_arena_retirement_failure": (
            graph_invocation_counters["resource_arena_retirement_failure"] == 0
        ),
        "one_final_readback_per_frame": (
            submit_counters["tensor_cpu_readback"]
            == measurement["invocation_count"]
        ),
        "final_live_bytes_bounded": (
            soak_phase["end_live_bytes"]
            <= _soak_memory_limit(soak_phase["baseline_live_bytes"])
        ),
        "replacement_peak_bounded": (
            soak_phase["high_water_bytes"]
            <= _soak_memory_limit(preflight_peak)
        ),
    }
    qualified = (
        configuration["requested_duration_seconds"]
        >= _LONG_SESSION_SOAK_GATE_DURATION_SECONDS
        and record["device_name"] == _LONG_SESSION_SOAK_DEVICE_NAME
    )
    passed = all(checks.values())
    status = "diagnostic_only"
    if qualified:
        status = "passed" if passed else "failed"
    return {
        "status": status,
        "qualified_gate_run": qualified,
        "all_checks_passed": passed,
        "checks": checks,
        "limits": {
            "device_name": _LONG_SESSION_SOAK_DEVICE_NAME,
            "duration_seconds": _LONG_SESSION_SOAK_GATE_DURATION_SECONDS,
            "minimum_invocations": _LONG_SESSION_SOAK_GATE_MINIMUM_INVOCATIONS,
            "recapture_interval_invocations": (
                _LONG_SESSION_SOAK_RECAPTURE_INTERVAL
            ),
            "memory_limit_ratio": _LONG_SESSION_SOAK_MEMORY_LIMIT_RATIO,
            "maximum_final_live_bytes": _soak_memory_limit(
                soak_phase["baseline_live_bytes"]
            ),
            "maximum_replacement_peak_bytes": _soak_memory_limit(
                preflight_peak
            ),
        },
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize_latency_samples(samples: list[float]) -> dict[str, Any]:
    if not samples:
        raise ValueError("latency samples must not be empty")
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "mean_seconds": statistics.fmean(samples),
        "median_seconds": statistics.median(ordered),
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "stdev_seconds": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "p90_seconds": _percentile(ordered, 0.90),
        "p95_seconds": _percentile(ordered, 0.95),
        "samples_seconds": samples,
    }


def _latency_runtime_snapshot() -> dict[str, int]:
    return {
        "cpu_fallback": int(torch.ops.vulkan_prepack.cpu_fallback_count()),
        "sync_readback": int(torch.ops.vulkan_prepack.sync_readback_count()),
    }


def _measure_latency_pair(
    eager_run: Callable[[], Any],
    graph_run: Callable[[], Any],
    warmup_repeats: int,
    measurement_repeats: int,
) -> dict[str, Any]:
    surfaces = (
        ("supported_eager", eager_run),
        ("vulkan_graph_program", graph_run),
    )
    torch.ops.vulkan_prepack.synchronize()
    for index in range(warmup_repeats):
        ordered_surfaces = surfaces if index % 2 == 0 else tuple(reversed(surfaces))
        for _, run in ordered_surfaces:
            output = run()
            torch.ops.vulkan_prepack.synchronize()
            del output
    torch.ops.vulkan_prepack.synchronize()

    samples: dict[str, list[float]] = {name: [] for name, _ in surfaces}
    runtime_counters = {
        name: {"cpu_fallback": 0, "sync_readback": 0} for name, _ in surfaces
    }
    for index in range(measurement_repeats):
        ordered_surfaces = surfaces if index % 2 == 0 else tuple(reversed(surfaces))
        for name, run in ordered_surfaces:
            counters_before = _latency_runtime_snapshot()
            start = time.perf_counter()
            output = run()
            torch.ops.vulkan_prepack.synchronize()
            samples[name].append(time.perf_counter() - start)
            counters_after = _latency_runtime_snapshot()
            for counter_name in runtime_counters[name]:
                runtime_counters[name][counter_name] += (
                    counters_after[counter_name] - counters_before[counter_name]
                )
            del output

    eager_summary = _summarize_latency_samples(samples["supported_eager"])
    graph_summary = _summarize_latency_samples(samples["vulkan_graph_program"])
    eager_summary["runtime_counters"] = runtime_counters["supported_eager"]
    graph_summary["runtime_counters"] = runtime_counters["vulkan_graph_program"]
    median_ratio = (
        graph_summary["median_seconds"] / eager_summary["median_seconds"]
    )
    return {
        "method": "alternating_completed_device_resident_invocations",
        "input_boundary": "preuploaded_vulkan_inputs_to_completed_vulkan_outputs",
        "output_readback_in_timed_region": False,
        "synchronization": "vulkan_prepack::synchronize_after_each_measurement",
        "measurement_order": "supported_eager_first_on_even_rounds",
        "warmup_repeats_per_surface": warmup_repeats,
        "measurement_repeats_per_surface": measurement_repeats,
        "supported_eager": eager_summary,
        "vulkan_graph_program": graph_summary,
        "median_ratio_graph_over_eager": median_ratio,
        "median_delta_percent": (median_ratio - 1.0) * 100.0,
    }


def _measure_case_latency(
    program: torch.vulkan.VulkanGraphProgram,
    eager_model: torch.nn.Module,
    args: tuple[Any, ...],
    warmup_repeats: int,
    measurement_repeats: int,
) -> dict[str, Any]:
    with torch.inference_mode():
        eager_latency_args = pytree.tree_map(
            lambda value: value.to("vulkan")
            if isinstance(value, torch.Tensor)
            else value,
            args,
        )
        normalized_placeholders = {
            node.placeholder_name
            for node in program.input_normalization.nodes
            if node.status == "lowered" and node.placeholder_name is not None
        }
        placeholder_names = tuple(
            str(node.target)
            for node in program.graph_module.graph.nodes
            if node.op == "placeholder"
        )
        graph_latency_args = tuple(
            value
            if placeholder_name in normalized_placeholders
            else pytree.tree_map(
                lambda leaf: leaf.to("vulkan")
                if isinstance(leaf, torch.Tensor)
                else leaf,
                value,
            )
            for placeholder_name, value in zip(placeholder_names, args)
        )
        plan = getattr(program, "cpp_plan", None)
        generation_before = (
            plan.invocation_generation() if plan is not None else None
        )
        result = _measure_latency_pair(
            lambda: eager_model(*eager_latency_args),
            lambda: program(*graph_latency_args),
            warmup_repeats,
            measurement_repeats,
        )
        if normalized_placeholders:
            result["method"] = "alternating_completed_supported_surface_invocations"
            result["input_boundary"] = (
                "supported_eager_preuploaded_vulkan_inputs_and_graph_contract_"
                "inputs_to_completed_vulkan_outputs"
            )
        generation_after = (
            plan.invocation_generation() if plan is not None else None
        )
        result["graph_invocation_generation_before"] = generation_before
        result["graph_invocation_generation_after"] = generation_after
        del eager_latency_args
        del graph_latency_args
    return result


def _run_long_session_soak(
    args: argparse.Namespace,
    model: torch.nn.Module,
    normal_program: torch.vulkan.VulkanGraphProgram,
    normal_args: tuple[Any, ...],
    alternate_args: tuple[Any, ...],
    dynamic_shapes: Any,
    device: torch.device,
    cpu_atol: float,
    cpu_rtol: float,
) -> dict[str, Any]:
    device_name = torch.vulkan.get_device_properties(device).name
    compile_seconds: list[float] = []

    def compile_variant(
        variant_args: tuple[Any, ...],
    ) -> torch.vulkan.VulkanGraphProgram:
        start = time.perf_counter()
        variant = torch.vulkan.export_and_lower(
            model,
            variant_args,
            dynamic_shapes=dynamic_shapes,
            device=device,
            planning_context=_planning_context(args, variant_args),
        )
        compile_seconds.append(time.perf_counter() - start)
        return variant

    def readback_and_check(
        program: torch.vulkan.VulkanGraphProgram,
        variant_args: tuple[Any, ...],
        expected: Any,
    ) -> None:
        output = program(*variant_args)
        runtime_counters = {
            "cpu_fallback": program.last_cpu_fallback_count,
            "sync_readback": program.last_sync_readback_count,
            "deferred_values_created": program.last_deferred_values_created,
        }
        if any(runtime_counters.values()):
            raise RuntimeError(
                "Long-session graph invocation crossed an implicit host boundary: "
                f"{runtime_counters}"
            )
        cpu_output = pytree.tree_map(
            lambda value: value.cpu()
            if isinstance(value, torch.Tensor)
            else value,
            output,
        )
        _assert_close(cpu_output, expected, cpu_atol, cpu_rtol)
        del cpu_output
        del output

    with torch.inference_mode():
        expected = {
            "normal": model(*normal_args),
            "alternate": model(*alternate_args),
        }
        torch.ops.vulkan_prepack.synchronize()
        preflight_baseline = _begin_memory_phase()
        active_variant = compile_variant(alternate_args)
        readback_and_check(active_variant, alternate_args, expected["alternate"])
        replacement = compile_variant(normal_args)
        readback_and_check(replacement, normal_args, expected["normal"])
        active_variant = replacement
        del replacement
        gc.collect()
        torch.ops.vulkan_prepack.synchronize()
        replacement_preflight = _finish_memory_phase(preflight_baseline)

        runtime_before = _latency_runtime_snapshot()
        graph_invocation_before = _named_counter_snapshot(
            _GRAPH_PROGRAM_INVOCATION_COUNTER_NAMES,
            list(torch.ops.vulkan_prepack.graph_program_invocation_counters()),
            "graph program invocation",
        )
        submit_before = _named_counter_snapshot(
            _SUBMIT_ORIGIN_COUNTER_NAMES,
            list(torch.ops.vulkan_prepack.submit_origin_counters()),
            "submit origin",
        )
        soak_baseline = _begin_memory_phase()
        memory_samples: list[dict[str, Any]] = []
        invocation_count = 0
        parity_check_count = 0
        recapture_count = 0
        start = time.perf_counter()
        while time.perf_counter() - start < args.long_session_soak_seconds:
            run_program = normal_program
            run_args = normal_args
            expected_output = expected["normal"]
            if (
                invocation_count > 0
                and invocation_count % _LONG_SESSION_SOAK_RECAPTURE_INTERVAL == 0
            ):
                recapture_case = (
                    "alternate" if recapture_count % 2 == 0 else "normal"
                )
                run_args = (
                    alternate_args if recapture_case == "alternate" else normal_args
                )
                expected_output = expected[recapture_case]
                replacement = compile_variant(run_args)
                active_variant = replacement
                del replacement
                gc.collect()
                run_program = active_variant
                recapture_count += 1
            readback_and_check(run_program, run_args, expected_output)
            invocation_count += 1
            parity_check_count += 1
            if (
                invocation_count % _LONG_SESSION_SOAK_MEMORY_SAMPLE_INTERVAL == 0
            ):
                gc.collect()
                torch.ops.vulkan_prepack.synchronize()
                snapshot = _memory_usage_snapshot()
                memory_samples.append(
                    {
                        "invocation": invocation_count,
                        "elapsed_seconds": time.perf_counter() - start,
                        "live_bytes": snapshot["live_bytes"],
                        "high_water_bytes": snapshot["high_water_bytes"],
                    }
                )

        gc.collect()
        torch.ops.vulkan_prepack.synchronize()
        elapsed_seconds = time.perf_counter() - start
        soak_memory = _finish_memory_phase(soak_baseline)
        if (
            not memory_samples
            or memory_samples[-1]["invocation"] != invocation_count
        ):
            memory_samples.append(
                {
                    "invocation": invocation_count,
                    "elapsed_seconds": elapsed_seconds,
                    "live_bytes": soak_memory["end_live_bytes"],
                    "high_water_bytes": soak_memory["high_water_bytes"],
                }
            )
        runtime_after = _latency_runtime_snapshot()
        graph_invocation_after = _named_counter_snapshot(
            _GRAPH_PROGRAM_INVOCATION_COUNTER_NAMES,
            list(torch.ops.vulkan_prepack.graph_program_invocation_counters()),
            "graph program invocation",
        )
        submit_after = _named_counter_snapshot(
            _SUBMIT_ORIGIN_COUNTER_NAMES,
            list(torch.ops.vulkan_prepack.submit_origin_counters()),
            "submit origin",
        )

    record = {
        "schema": "VulkanGraphLongSessionSoak.v0",
        "device_name": device_name,
        "configuration": {
            "requested_duration_seconds": args.long_session_soak_seconds,
            "steady_state_case": "normal",
            "per_frame_output_readback": True,
            "parity_check_every_invocation": True,
            "recapture_interval_invocations": (
                _LONG_SESSION_SOAK_RECAPTURE_INTERVAL
            ),
            "memory_sample_interval_invocations": (
                _LONG_SESSION_SOAK_MEMORY_SAMPLE_INTERVAL
            ),
        },
        "measurement": {
            "elapsed_seconds": elapsed_seconds,
            "invocation_count": invocation_count,
            "parity_check_count": parity_check_count,
            "recapture_count": recapture_count,
            "recapture_compile_seconds": compile_seconds,
            "runtime_counters": _counter_delta(runtime_before, runtime_after),
            "submission_counters": {
                "graph_program_invocation": _counter_delta(
                    graph_invocation_before, graph_invocation_after
                ),
                "submit_origin": _counter_delta(submit_before, submit_after),
            },
            "memory": {
                "replacement_preflight": replacement_preflight,
                "soak": soak_memory,
                "samples": memory_samples,
            },
        },
    }
    record["gate"] = _evaluate_long_session_soak_gate(record)
    return record


def _run_case(
    name: str,
    program: torch.vulkan.VulkanGraphProgram,
    eager_model: torch.nn.Module,
    cpu_model: torch.nn.Module,
    args: tuple[Any, ...],
    eager_atol: float,
    eager_rtol: float,
    cpu_atol: float,
    cpu_rtol: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix=f"vulkan_{name}_diagnostics_") as temp_dir:
        diagnostic_dir = Path(temp_dir)
        with torch.inference_mode():
            cpu_expected = cpu_model(*args)
            eager_memory_baseline = _begin_memory_phase()
            with _planning_diagnostic_capture(
                diagnostic_dir, "supported_eager"
            ) as eager_diagnostic_paths:
                eager_args = pytree.tree_map(
                    lambda value: value.to("vulkan")
                    if isinstance(value, torch.Tensor)
                    else value,
                    args,
                )
                eager_output = eager_model(*eager_args)
                eager_expected = pytree.tree_map(
                    lambda value: value.cpu()
                    if isinstance(value, torch.Tensor)
                    else value,
                    eager_output,
                )
            eager_diagnostics = _captured_planning_diagnostics(
                eager_diagnostic_paths
            )
            eager_memory = _finish_memory_phase(eager_memory_baseline)
            del eager_output
            del eager_args
            torch.ops.vulkan_prepack.synchronize()
            torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
            torch.ops.vulkan_prepack.reset_submit_origin_counters()
            graph_first_memory_baseline = _begin_memory_phase()
            with _planning_diagnostic_capture(
                diagnostic_dir, "vulkan_graph_program_first"
            ) as graph_first_diagnostic_paths:
                first_start = time.perf_counter()
                graph_output = program(*args)
                first_run_seconds = time.perf_counter() - first_start
                graph_cpu = pytree.tree_map(
                    lambda value: value.cpu()
                    if isinstance(value, torch.Tensor)
                    else value,
                    graph_output,
                )
            graph_first_diagnostics = _captured_planning_diagnostics(
                graph_first_diagnostic_paths
            )
            graph_eager = _parity(graph_cpu, eager_expected)
            graph_cpu_parity = _parity(graph_cpu, cpu_expected)
            _assert_close(graph_cpu, eager_expected, eager_atol, eager_rtol)
            _assert_close(graph_cpu, cpu_expected, cpu_atol, cpu_rtol)
            graph_first_memory = _finish_memory_phase(graph_first_memory_baseline)
            torch.ops.vulkan_prepack.synchronize()
            graph_repeat_memory_baseline = _begin_memory_phase()
            with _planning_diagnostic_capture(
                diagnostic_dir, "vulkan_graph_program_repeat"
            ) as graph_repeat_diagnostic_paths:
                repeat_start = time.perf_counter()
                repeat_output = program(*args)
                _ = pytree.tree_map(
                    lambda value: value.cpu()
                    if isinstance(value, torch.Tensor)
                    else value,
                    repeat_output,
                )
                repeated_run_seconds = time.perf_counter() - repeat_start
            graph_repeat_diagnostics = _captured_planning_diagnostics(
                graph_repeat_diagnostic_paths
            )
            graph_repeat_memory = _finish_memory_phase(
                graph_repeat_memory_baseline
            )
    submission_counters = {
        "graph_program_invocation": _named_counter_snapshot(
            _GRAPH_PROGRAM_INVOCATION_COUNTER_NAMES,
            list(torch.ops.vulkan_prepack.graph_program_invocation_counters()),
            "graph program invocation",
        ),
        "submit_origin": _named_counter_snapshot(
            _SUBMIT_ORIGIN_COUNTER_NAMES,
            list(torch.ops.vulkan_prepack.submit_origin_counters()),
            "submit origin",
        ),
    }
    counters = {
        "runtime_cpu_fallback": program.last_cpu_fallback_count,
        "runtime_sync_readback_escape": program.last_sync_readback_count,
        "deferred_values_created": program.last_deferred_values_created,
    }
    if any(counters.values()):
        raise RuntimeError(f"{name} graph runtime counters are nonzero: {counters}")
    memory = {
        "eager": eager_memory,
        "graph_first": graph_first_memory,
        "graph_repeat_with_prior_output_live": graph_repeat_memory,
    }
    planning_diagnostics = {
        "supported_eager": eager_diagnostics,
        "vulkan_graph_program_first": graph_first_diagnostics,
        "vulkan_graph_program_repeat": graph_repeat_diagnostics,
    }
    return (
        {
            "name": name,
            "input_shape": _input_shapes(args),
            "timing": {
                "first_run_seconds": first_run_seconds,
                "repeated_run_seconds_reference_only": repeated_run_seconds,
            },
            "guard": {"status": "accepted"},
            "execution_plan": _execution_plan_summary(program),
            "runtime_counters": counters,
            "submission_counters": submission_counters,
            "memory": memory,
            "planning_diagnostics": planning_diagnostics,
        },
        {
            "name": name,
            "input_shape": _input_shapes(args),
            "timing": {
                "first_run_seconds": first_run_seconds,
                "repeated_run_seconds_reference_only": repeated_run_seconds,
            },
            "guard": {"status": "accepted"},
            "execution_plan": _execution_plan_summary(program),
            "submission_counters": submission_counters,
            "memory": memory,
            "planning_diagnostics": planning_diagnostics,
            "graph_vs_eager_vulkan": graph_eager,
            "graph_vs_cpu": graph_cpu_parity,
            "tolerance": {
                "graph_vs_eager_vulkan": {
                    "atol": eager_atol,
                    "rtol": eager_rtol,
                },
                "graph_vs_cpu": {"atol": cpu_atol, "rtol": cpu_rtol},
            },
        },
    )


def _run_state_replay(
    mapping: tuple[tuple[int, int], ...],
    source_program: torch.vulkan.VulkanGraphProgram,
    target_program: torch.vulkan.VulkanGraphProgram,
    cpu_model: torch.nn.Module,
    source_args: tuple[Any, ...],
    target_args: tuple[Any, ...],
    cpu_atol: float,
    cpu_rtol: float,
) -> dict[str, Any]:
    if source_program.cpp_plan is None or target_program.cpp_plan is None:
        raise RuntimeError("State replay requires compiled C++ graph plans")
    with torch.inference_mode():
        source_expected = cpu_model(*source_args)
        target_expected = cpu_model(*target_args)
        target_input_leaves, target_input_spec = pytree.tree_flatten(target_args)
        source_generation_before = source_program.cpp_plan.invocation_generation()
        target_generation_before = target_program.cpp_plan.invocation_generation()
        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        torch.ops.vulkan_prepack.reset_submit_origin_counters()
        source_output = source_program(*source_args)
        source_output_leaves, source_output_spec = pytree.tree_flatten(source_output)
        replay_input_leaves = list(target_input_leaves)
        mapping_rows = []
        for input_index, output_index in mapping:
            if input_index >= len(replay_input_leaves):
                raise RuntimeError(
                    f"State replay input leaf index {input_index} is out of range"
                )
            if output_index >= len(source_output_leaves):
                raise RuntimeError(
                    f"State replay output leaf index {output_index} is out of range"
                )
            expected_input = replay_input_leaves[input_index]
            source_value = source_output_leaves[output_index]
            if not isinstance(expected_input, torch.Tensor) or not isinstance(
                source_value, torch.Tensor
            ):
                raise RuntimeError("State replay mappings must connect Tensor leaves")
            if source_value.device.type != "vulkan":
                raise RuntimeError("State replay source Tensor must remain on Vulkan")
            if (
                source_value.shape != expected_input.shape
                or source_value.dtype != expected_input.dtype
            ):
                raise RuntimeError(
                    "State replay source and target Tensor metadata do not match"
                )
            replay_input_leaves[input_index] = source_value
            mapping_rows.append(
                {
                    "input_leaf_index": input_index,
                    "output_leaf_index": output_index,
                    "shape": list(source_value.shape),
                    "dtype": str(source_value.dtype),
                }
            )
        replay_args = pytree.tree_unflatten(replay_input_leaves, target_input_spec)
        if not isinstance(replay_args, tuple):
            raise RuntimeError("State replay input tree must reconstruct a tuple")
        target_output = target_program(*replay_args)
        submission_counters = {
            "graph_program_invocation": _named_counter_snapshot(
                _GRAPH_PROGRAM_INVOCATION_COUNTER_NAMES,
                list(torch.ops.vulkan_prepack.graph_program_invocation_counters()),
                "graph program invocation",
            ),
            "submit_origin": _named_counter_snapshot(
                _SUBMIT_ORIGIN_COUNTER_NAMES,
                list(torch.ops.vulkan_prepack.submit_origin_counters()),
                "submit origin",
            ),
        }
        replayed_state = tuple(
            source_output_leaves[output_index] for _, output_index in mapping
        )
        expected_state = tuple(
            target_input_leaves[input_index] for input_index, _ in mapping
        )
        state_parity = _parity(replayed_state, expected_state)
        target_parity = _parity(target_output, target_expected)
        source_parity_after_target = _parity(source_output, source_expected)
        _assert_close(replayed_state, expected_state, cpu_atol, cpu_rtol)
        _assert_close(target_output, target_expected, cpu_atol, cpu_rtol)
        _assert_close(source_output, source_expected, cpu_atol, cpu_rtol)
    source_counters = {
        "cpu_fallback": source_program.last_cpu_fallback_count,
        "sync_readback": source_program.last_sync_readback_count,
        "deferred_values_created": source_program.last_deferred_values_created,
    }
    target_counters = {
        "cpu_fallback": target_program.last_cpu_fallback_count,
        "sync_readback": target_program.last_sync_readback_count,
        "deferred_values_created": target_program.last_deferred_values_created,
    }
    if any(source_counters.values()) or any(target_counters.values()):
        raise RuntimeError(
            "State replay crossed an implicit host boundary: "
            f"source={source_counters}, target={target_counters}"
        )
    return {
        "status": "passed",
        "protocol": "explicit_output_to_input_tensor_leaves",
        "mapped_leaf_count": len(mapping),
        "mapping": mapping_rows,
        "source_output_tree_spec": str(source_output_spec),
        "target_input_tree_spec": str(target_input_spec),
        "source_program_key": _jsonable(source_program.key),
        "target_program_key": _jsonable(target_program.key),
        "source_invocation_generation_before": source_generation_before,
        "source_invocation_generation_after": (
            source_program.cpp_plan.invocation_generation()
        ),
        "target_invocation_generation_before": target_generation_before,
        "target_invocation_generation_after": (
            target_program.cpp_plan.invocation_generation()
        ),
        "source_runtime_counters": source_counters,
        "target_runtime_counters": target_counters,
        "submission_counters": submission_counters,
        "replayed_state_vs_cpu": state_parity,
        "target_output_vs_cpu": target_parity,
        "source_output_after_target_vs_cpu": source_parity_after_target,
        "source_output_preserved_after_target": True,
        "tolerance": {"atol": cpu_atol, "rtol": cpu_rtol},
    }


def _out_of_range_guard(
    program: torch.vulkan.VulkanGraphProgram,
    args: tuple[Any, ...] | None,
) -> dict[str, Any]:
    if args is None:
        return {"status": "not_requested"}
    try:
        program(*args)
    except Exception as error:
        return {
            "status": "guard_rejected",
            "exception_type": type(error).__name__,
            "message": str(error),
        }
    return {"status": "unexpectedly_accepted"}


def _is_export_guard_rejection(error: Exception) -> bool:
    return isinstance(error, torch.vulkan.VulkanGraphExecutionError) and (
        "Guard failed:" in str(error)
        or "_guards_fn" in str(error)
        or "fixed graph input shape mismatch" in str(error)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record generic torch.export Vulkan graph evidence."
    )
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--artifact-prefix", default="dav2_vits", type=_artifact_prefix)
    parser.add_argument("--source-git-sha", default=None)
    parser.add_argument("--device-index", type=_device_index, default=None)
    parser.add_argument("--eager-atol", type=float, default=0.0)
    parser.add_argument("--eager-rtol", type=float, default=0.0)
    parser.add_argument("--cpu-atol", type=float, default=0.0)
    parser.add_argument("--cpu-rtol", type=float, default=0.0)
    parser.add_argument(
        "--planning-model-domain",
        choices=("generic", "vision", "llm"),
        default="generic",
    )
    parser.add_argument(
        "--planning-execution-phase",
        choices=("none", "prefill", "decode", "backbone", "decoder"),
        default="none",
    )
    parser.add_argument(
        "--planning-prefer-packed-layout-propagation",
        action="store_true",
    )
    parser.add_argument(
        "--planning-fixed-shape-graph-input",
        action="store_true",
    )
    parser.add_argument(
        "--latency-warmup-repeats", type=_nonnegative_repeat_count, default=3
    )
    parser.add_argument(
        "--latency-measurement-repeats", type=_positive_repeat_count, default=10
    )
    parser.add_argument(
        "--long-session-soak-seconds",
        type=_nonnegative_repeat_count,
        default=0,
        help=(
            "Run the graph long-session soak for this many seconds. Only runs of "
            "at least 600 seconds on the RX 9070 can satisfy the standing gate."
        ),
    )
    parser.add_argument("--normal-input-shape", default=None)
    parser.add_argument("--alternate-input-shape", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_sha = args.source_git_sha or source_git_sha(REPO_ROOT)
    if source_sha is None:
        raise RuntimeError(
            "Source Git SHA is required; pass --source-git-sha when git is not "
            "available on PATH"
        )
    device_count = torch.vulkan.device_count()
    if device_count < 1:
        raise RuntimeError("No Vulkan devices are available")
    device_index = (
        torch.vulkan.current_device()
        if args.device_index is None
        else args.device_index
    )
    if device_index >= device_count:
        raise ValueError(
            f"--device-index {device_index} is outside [0, {device_count})"
        )
    torch.vulkan.set_device(device_index)
    device = torch.device("vulkan", device_index)
    external_root, checkpoint = require_external_assets(
        args.external_root, args.checkpoint
    )
    factory = _load_adapter(args.adapter)
    adapter = dict(factory(external_root, checkpoint))
    model = adapter.get("model")
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Adapter must return an eval torch.nn.Module as model")
    model.eval()
    normal = _require_cpu_inputs(adapter.get("normal_inputs"), "normal_inputs")
    alternate = _require_cpu_inputs(
        adapter.get("alternate_inputs"), "alternate_inputs"
    )
    out_of_range = adapter.get("out_of_range_inputs")
    if out_of_range is not None:
        out_of_range = _require_cpu_inputs(out_of_range, "out_of_range_inputs")
    if args.normal_input_shape and _input_shapes(normal) != [
        list(parse_input_shape(args.normal_input_shape))
    ]:
        raise ValueError("--normal-input-shape disagrees with adapter normal_inputs")
    if args.alternate_input_shape and _input_shapes(alternate) != [
        list(parse_input_shape(args.alternate_input_shape))
    ]:
        raise ValueError(
            "--alternate-input-shape disagrees with adapter alternate_inputs"
        )
    dynamic_shapes = adapter.get("dynamic_shapes")
    state_replay_mapping = _state_replay_mapping(
        adapter.get("state_replay_input_from_output")
    )
    planning_context = _planning_context(args, normal)
    original_export_start = time.perf_counter()
    original_export = torch.export.export(
        model, normal, dynamic_shapes=dynamic_shapes, strict=False
    )
    original_export_seconds = time.perf_counter() - original_export_start
    lower_start = time.perf_counter()
    program = torch.vulkan.export_and_lower(
        model,
        normal,
        dynamic_shapes=dynamic_shapes,
        device=device,
        planning_context=planning_context,
    )
    lower_seconds = time.perf_counter() - lower_start
    normal_program = program
    eager_model = copy.deepcopy(model).to(device).eval()
    normal_census, normal_parity = _run_case(
        "normal",
        program,
        eager_model,
        model,
        normal,
        args.eager_atol,
        args.eager_rtol,
        args.cpu_atol,
        args.cpu_rtol,
    )
    try:
        alternate_program = program
        alternate_census, alternate_parity = _run_case(
            "alternate",
            program,
            eager_model,
            model,
            alternate,
            args.eager_atol,
            args.eager_rtol,
            args.cpu_atol,
            args.cpu_rtol,
        )
    except Exception as error:
        if not _is_export_guard_rejection(error):
            raise
        variant_start = time.perf_counter()
        alternate_planning_context = _planning_context(args, alternate)
        alternate_program = torch.vulkan.export_and_lower(
            model,
            alternate,
            dynamic_shapes=dynamic_shapes,
            device=device,
            planning_context=alternate_planning_context,
        )
        variant_seconds = time.perf_counter() - variant_start
        alternate_census, alternate_parity = _run_case(
            "alternate",
            alternate_program,
            eager_model,
            model,
            alternate,
            args.eager_atol,
            args.eager_rtol,
            args.cpu_atol,
            args.cpu_rtol,
        )
        alternate_census["guard"] = {
            "status": "recompiled_guard_variant",
            "rejected_program_message": str(error),
            "variant_compile_seconds": variant_seconds,
            "variant_program_key": _jsonable(alternate_program.key),
        }
        alternate_parity["guard"] = alternate_census["guard"]
    out_of_range_guard = _out_of_range_guard(program, out_of_range)
    execution_plan = _execution_plan_summary(program)
    normal_latency = _measure_case_latency(
        normal_program,
        eager_model,
        normal,
        args.latency_warmup_repeats,
        args.latency_measurement_repeats,
    )
    alternate_latency = _measure_case_latency(
        alternate_program,
        eager_model,
        alternate,
        args.latency_warmup_repeats,
        args.latency_measurement_repeats,
    )
    for case in (normal_census, normal_parity):
        case["timing"]["supported_default_latency"] = normal_latency
    for case in (alternate_census, alternate_parity):
        case["timing"]["supported_default_latency"] = alternate_latency
    state_replay = None
    if state_replay_mapping:
        state_replay = _run_state_replay(
            state_replay_mapping,
            normal_program,
            alternate_program,
            model,
            normal,
            alternate,
            args.cpu_atol,
            args.cpu_rtol,
        )
    long_session_soak = None
    if args.long_session_soak_seconds:
        long_session_soak = _run_long_session_soak(
            args,
            model,
            normal_program,
            normal,
            alternate,
            dynamic_shapes,
            device,
            args.cpu_atol,
            args.cpu_rtol,
        )
    if out_of_range_guard["status"] == "unexpectedly_accepted":
        raise RuntimeError("Out-of-range input was accepted by exported guards")
    lowering_reports = _lowering_reports(program)
    common = {
        "schema": "VulkanGraphExportEvidence.v1",
        "status": MEASURED_STATUS,
        "source_git_sha": source_sha,
        "external_assets": {
            "adapter": _adapter_identity(args.adapter),
            "checkpoint_basename": checkpoint.name,
            "checkpoint_sha256": sha256_file(checkpoint),
        },
        "runtime": _device_runtime_identity(device),
        "speculative_eager_bridges": "disabled",
        "clone_workaround_used": False,
        "program_key": _jsonable(program.key),
        "graph_structure": _graph_structure(program),
        "graph_census": _graph_counts(program),
        "execution_plan": execution_plan,
        **lowering_reports,
    }
    if state_replay is not None:
        common["state_replay"] = state_replay
    if long_session_soak is not None:
        common["long_session_soak"] = long_session_soak
    census = {
        **common,
        "artifact_type": "export_census",
        "timing": {
            "cpu_export_seconds": original_export_seconds,
            "export_and_lower_seconds": lower_seconds,
        },
        "original_exported_node_count": len(tuple(original_export.graph.nodes)),
        "cases": [normal_census, alternate_census],
        "out_of_range_guard": out_of_range_guard,
    }
    parity = {
        **common,
        "artifact_type": "parity",
        "cases": [normal_parity, alternate_parity],
        "out_of_range_guard": out_of_range_guard,
    }
    census_path, parity_path = _artifact_output_paths(
        args.output_dir, args.artifact_prefix
    )
    write_evidence(census_path, census)
    write_evidence(parity_path, parity)
    if (
        long_session_soak is not None
        and long_session_soak["gate"]["qualified_gate_run"]
        and long_session_soak["gate"]["status"] != "passed"
    ):
        failed_checks = [
            name
            for name, passed in long_session_soak["gate"]["checks"].items()
            if not passed
        ]
        raise RuntimeError(
            "Long-session graph soak failed registered gates: "
            + ", ".join(failed_checks)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
