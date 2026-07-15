from __future__ import annotations

import argparse
import copy
import dataclasses
import importlib
import importlib.util
import os
import statistics
import sys
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
    for name in ("torch_cpu.dll", "c10.dll"):
        path = torch_lib / name
        files[name] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    return {
        "device": str(device),
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
        "static_identity_advanced_indices": getattr(
            program, "static_identity_advanced_indices", None
        ),
        "static_gqa_repeats": getattr(program, "static_gqa_repeats", None),
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


def _memory_usage_snapshot() -> dict[str, int]:
    rows = torch.ops.vulkan_prepack.vulkan_memory_residency_snapshot()
    prefix = "vulkan_memory_summary "
    summary = next((row for row in rows if row.startswith(prefix)), None)
    if summary is None:
        raise RuntimeError("Vulkan memory snapshot is missing its summary row")
    fields = dict(
        field.split("=", 1)
        for field in summary[len(prefix) :].split()
    )
    return {
        "live_bytes": int(fields["live_bytes"]),
        "high_water_bytes": int(fields["high_water_bytes"]),
    }


def _begin_memory_phase() -> int:
    torch.ops.vulkan_prepack.reset_vulkan_memory_residency_snapshot()
    return _memory_usage_snapshot()["live_bytes"]


def _finish_memory_phase(baseline_live_bytes: int) -> dict[str, int]:
    snapshot = _memory_usage_snapshot()
    return {
        "baseline_live_bytes": baseline_live_bytes,
        "end_live_bytes": snapshot["live_bytes"],
        "high_water_bytes": snapshot["high_water_bytes"],
        "peak_delta_bytes": max(
            0, snapshot["high_water_bytes"] - baseline_live_bytes
        ),
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
        latency_args = pytree.tree_map(
            lambda value: value.to("vulkan")
            if isinstance(value, torch.Tensor)
            else value,
            args,
        )
        plan = getattr(program, "cpp_plan", None)
        generation_before = (
            plan.invocation_generation() if plan is not None else None
        )
        result = _measure_latency_pair(
            lambda: eager_model(*latency_args),
            lambda: program(*latency_args),
            warmup_repeats,
            measurement_repeats,
        )
        generation_after = (
            plan.invocation_generation() if plan is not None else None
        )
        result["graph_invocation_generation_before"] = generation_before
        result["graph_invocation_generation_after"] = generation_after
        del latency_args
    return result


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
    with torch.inference_mode():
        cpu_expected = cpu_model(*args)
        eager_memory_baseline = _begin_memory_phase()
        eager_args = pytree.tree_map(
            lambda value: value.to("vulkan")
            if isinstance(value, torch.Tensor)
            else value,
            args,
        )
        eager_output = eager_model(*eager_args)
        eager_expected = pytree.tree_map(
            lambda value: value.cpu() if isinstance(value, torch.Tensor) else value,
            eager_output,
        )
        eager_memory = _finish_memory_phase(eager_memory_baseline)
        del eager_output
        del eager_args
        torch.ops.vulkan_prepack.synchronize()
        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        torch.ops.vulkan_prepack.reset_submit_origin_counters()
        graph_first_memory_baseline = _begin_memory_phase()
        first_start = time.perf_counter()
        graph_output = program(*args)
        first_run_seconds = time.perf_counter() - first_start
        graph_cpu = pytree.tree_map(
            lambda value: value.cpu() if isinstance(value, torch.Tensor) else value,
            graph_output,
        )
        graph_eager = _parity(graph_cpu, eager_expected)
        graph_cpu_parity = _parity(graph_cpu, cpu_expected)
        _assert_close(graph_cpu, eager_expected, eager_atol, eager_rtol)
        _assert_close(graph_cpu, cpu_expected, cpu_atol, cpu_rtol)
        graph_first_memory = _finish_memory_phase(graph_first_memory_baseline)
        graph_repeat_memory_baseline = _begin_memory_phase()
        repeat_start = time.perf_counter()
        repeat_output = program(*args)
        _ = pytree.tree_map(
            lambda value: value.cpu() if isinstance(value, torch.Tensor) else value,
            repeat_output,
        )
        repeated_run_seconds = time.perf_counter() - repeat_start
        graph_repeat_memory = _finish_memory_phase(graph_repeat_memory_baseline)
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
        "Guard failed:" in str(error) or "_guards_fn" in str(error)
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
    parser.add_argument("--eager-atol", type=float, default=0.0)
    parser.add_argument("--eager-rtol", type=float, default=0.0)
    parser.add_argument("--cpu-atol", type=float, default=0.0)
    parser.add_argument("--cpu-rtol", type=float, default=0.0)
    parser.add_argument(
        "--latency-warmup-repeats", type=_nonnegative_repeat_count, default=3
    )
    parser.add_argument(
        "--latency-measurement-repeats", type=_positive_repeat_count, default=10
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
    device = torch.device("vulkan")
    original_export_start = time.perf_counter()
    original_export = torch.export.export(
        model, normal, dynamic_shapes=dynamic_shapes, strict=False
    )
    original_export_seconds = time.perf_counter() - original_export_start
    lower_start = time.perf_counter()
    program = torch.vulkan.export_and_lower(
        model, normal, dynamic_shapes=dynamic_shapes, device=device
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
        alternate_program = torch.vulkan.export_and_lower(
            model, alternate, dynamic_shapes=dynamic_shapes, device=device
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
