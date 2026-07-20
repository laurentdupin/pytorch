import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.vulkan._graph_evidence import (
    EVIDENCE_SCHEMA,
    ExternalGraphEvidenceSetupError,
    parse_input_shape,
    require_external_assets,
    template_payload,
    validate_evidence_payload,
)
from scripts.benchmarks.vulkan_graph_export_evidence import (
    _adapter_identity,
    _allocator_residency_summary,
    _artifact_output_paths,
    _artifact_prefix,
    _device_index,
    _evaluate_long_session_soak_gate,
    _execution_plan_summary,
    _graph_counts,
    _int_summary_row,
    _is_export_guard_rejection,
    _linear_pack_residency_summary,
    _lowering_reports,
    _named_counter_snapshot,
    _nonnegative_repeat_count,
    _planning_diagnostic_summary,
    _planning_context,
    _positive_repeat_count,
    _state_replay_mapping,
    _summarize_latency_samples,
)


@dataclass(frozen=True)
class _FakeLoweringReport:
    created_context_count: int


@dataclass(frozen=True)
class _FakeStaticLinearGeluNodeReport:
    node_name: str
    status: str
    reason: str
    linear_node_name: str
    context_attr: str
    plan_attr: str
    program_name: str
    program_version: str
    instruction_count: int
    input_ssa: int
    output_ssa: int
    input_use_count: int
    input_last_use: int
    static_context_slot: int
    direct_transition_only: bool
    replay_state_empty: bool
    region_family: str
    intermediate_ssa: int
    intermediate_use_count: int
    intermediate_last_use: int


@dataclass(frozen=True)
class _FakeStaticLinearGeluRegionReport:
    candidate_count: int
    lowered_count: int
    rejected_count: int
    skipped_count: int
    plan_factory: str
    nodes: tuple[_FakeStaticLinearGeluNodeReport, ...]


@dataclass(frozen=True)
class _FakeVulkanGraphRegionFamilyDiagnostics:
    family: str
    candidate_count: int
    lowered_count: int
    rejected_count: int
    skipped_count: int
    plan_factory: str
    nodes: tuple[object, ...]


@dataclass(frozen=True)
class _FakeVulkanGraphRegionReport:
    plan_class: str
    plan_version: str
    families: tuple[_FakeVulkanGraphRegionFamilyDiagnostics, ...]


def _fake_vulkan_graph_region_report() -> _FakeVulkanGraphRegionReport:
    linear_report = _fake_static_linear_gelu_report()
    return _FakeVulkanGraphRegionReport(
        plan_class="VulkanGraphRegionPlan",
        plan_version="v1",
        families=(
            _FakeVulkanGraphRegionFamilyDiagnostics(
                family="linear_gelu_tanh",
                candidate_count=1,
                lowered_count=1,
                rejected_count=0,
                skipped_count=0,
                plan_factory=(
                    "vulkan_prepack::create_vulkan_graph_region_plan_linear_gelu"
                ),
                nodes=linear_report.nodes,
            ),
        ),
    )


def _fake_static_linear_gelu_report() -> _FakeStaticLinearGeluRegionReport:
    return _FakeStaticLinearGeluRegionReport(
        candidate_count=1,
        lowered_count=1,
        rejected_count=0,
        skipped_count=0,
        plan_factory="vulkan_prepack::create_vulkan_graph_region_plan_linear_gelu",
        nodes=(
            _FakeStaticLinearGeluNodeReport(
                node_name="run_vulkan_graph_region_plan",
                status="lowered",
                reason="graph_owned_static_linear_tanh_gelu",
                linear_node_name="run_linear_context",
                context_attr="_vulkan_linear_context_a",
                plan_attr="_vulkan_graph_region_plan_a",
                program_name="VulkanGraphRegionPlan",
                program_version="v1",
                instruction_count=2,
                input_ssa=0,
                output_ssa=2,
                input_use_count=1,
                input_last_use=0,
                static_context_slot=0,
                direct_transition_only=True,
                replay_state_empty=True,
                region_family="linear_gelu_tanh",
                intermediate_ssa=1,
                intermediate_use_count=1,
                intermediate_last_use=1,
            ),
        ),
    )


@dataclass(frozen=True)
class _FakeStaticAddLayernormNodeReport:
    node_name: str
    status: str
    reason: str
    add_node_name: str
    layernorm_node_name: str
    context_attr: str
    plan_attr: str
    normalized_shape: tuple[int, ...]
    program_name: str
    program_version: str
    fused_instruction: str
    instruction_count: int
    residual_input_ssa: int
    addend_input_ssa: int
    residual_output_ssa: int
    normalized_output_ssa: int
    residual_input_use_count: int
    residual_input_last_use: int
    addend_input_use_count: int
    addend_input_last_use: int
    static_context_slot: int
    context_ownership_outcome: str
    direct_transition_only: bool
    replay_state_empty: bool
    persistent_output_state: bool


@dataclass(frozen=True)
class _FakeStaticAddLayernormRegionReport:
    candidate_count: int
    lowered_count: int
    rejected_count: int
    skipped_count: int
    plan_factory: str
    nodes: tuple[_FakeStaticAddLayernormNodeReport, ...]


def _fake_static_add_layernorm_report() -> _FakeStaticAddLayernormRegionReport:
    return _FakeStaticAddLayernormRegionReport(
        candidate_count=1,
        lowered_count=1,
        rejected_count=0,
        skipped_count=0,
        plan_factory="vulkan_prepack::create_graph_add_layernorm_plan",
        nodes=(
            _FakeStaticAddLayernormNodeReport(
                node_name="run_graph_add_layernorm_plan",
                status="lowered",
                reason="graph_owned_static_add_layernorm",
                add_node_name="add",
                layernorm_node_name="run_layernorm_context",
                context_attr="_vulkan_layernorm_context_a",
                plan_attr="_vulkan_static_add_layernorm_plan_a",
                normalized_shape=(4,),
                program_name="StaticAddLayernormRegion",
                program_version="v1",
                fused_instruction="add_layernorm_fused_or_composed_vulkan",
                instruction_count=1,
                residual_input_ssa=0,
                addend_input_ssa=1,
                residual_output_ssa=2,
                normalized_output_ssa=3,
                residual_input_use_count=1,
                residual_input_last_use=0,
                addend_input_use_count=1,
                addend_input_last_use=0,
                static_context_slot=0,
                context_ownership_outcome=(
                    "transferred_removed_original_context_attr"
                ),
                direct_transition_only=True,
                replay_state_empty=True,
                persistent_output_state=False,
            ),
        ),
    )


@dataclass(frozen=True)
class _FakeStaticConv2dReluNodeReport:
    node_name: str
    status: str
    reason: str
    conv2d_node_name: str
    context_attr: str
    plan_attr: str
    program_name: str
    program_version: str
    instruction_count: int
    input_ssa: int
    output_ssa: int
    input_use_count: int
    input_last_use: int
    static_context_slot: int
    direct_transition_only: bool
    replay_state_empty: bool


@dataclass(frozen=True)
class _FakeStaticConv2dReluRegionReport:
    candidate_count: int
    lowered_count: int
    rejected_count: int
    skipped_count: int
    plan_factory: str
    nodes: tuple[_FakeStaticConv2dReluNodeReport, ...]


def _fake_static_conv2d_relu_report() -> _FakeStaticConv2dReluRegionReport:
    return _FakeStaticConv2dReluRegionReport(
        candidate_count=1,
        lowered_count=1,
        rejected_count=0,
        skipped_count=0,
        plan_factory="vulkan_prepack::create_graph_conv2d_relu_plan",
        nodes=(
            _FakeStaticConv2dReluNodeReport(
                node_name="run_graph_conv2d_relu_plan",
                status="lowered",
                reason="graph_owned_static_conv2d_relu",
                conv2d_node_name="run_conv2d_context",
                context_attr="_vulkan_conv2d_context_a",
                plan_attr="_vulkan_static_conv2d_relu_plan_a",
                program_name="StaticConv2dReluRegion",
                program_version="v1",
                instruction_count=1,
                input_ssa=0,
                output_ssa=1,
                input_use_count=1,
                input_last_use=0,
                static_context_slot=0,
                direct_transition_only=True,
                replay_state_empty=True,
            ),
        ),
    )


@dataclass(frozen=True)
class _FakeStaticConv2dReluConv2dNodeReport:
    node_name: str
    status: str
    reason: str
    first_conv2d_node_name: str
    relu_node_name: str
    second_conv2d_node_name: str
    first_context_attr: str
    second_context_attr: str
    plan_attr: str
    program_name: str
    program_version: str
    instruction_count: int
    input_ssa: int
    intermediate_ssa: int
    output_ssa: int
    input_use_count: int
    input_last_use: int
    intermediate_use_count: int
    intermediate_last_use: int
    first_static_context_slot: int
    second_static_context_slot: int
    bounded_submission_owned: bool
    program_private_scratch: bool
    scratch_ring_capacity: int
    timeline_gated_release: bool
    direct_transition_only: bool
    replay_state_empty: bool


@dataclass(frozen=True)
class _FakeStaticConv2dReluConv2dRegionReport:
    candidate_count: int
    lowered_count: int
    rejected_count: int
    skipped_count: int
    plan_factory: str
    nodes: tuple[_FakeStaticConv2dReluConv2dNodeReport, ...]
    excluded_relu_node_names: tuple[str, ...]


def _fake_static_conv2d_relu_conv2d_report(
) -> _FakeStaticConv2dReluConv2dRegionReport:
    return _FakeStaticConv2dReluConv2dRegionReport(
        candidate_count=1,
        lowered_count=1,
        rejected_count=0,
        skipped_count=0,
        plan_factory=(
            "vulkan_prepack::create_vulkan_graph_region_plan_conv2d_relu_conv2d"
        ),
        nodes=(
            _FakeStaticConv2dReluConv2dNodeReport(
                node_name="run_vulkan_graph_region_plan",
                status="lowered",
                reason="graph_owned_static_conv2d_relu_conv2d",
                first_conv2d_node_name="run_conv2d_context",
                relu_node_name="relu",
                second_conv2d_node_name="run_conv2d_context_1",
                first_context_attr="_vulkan_conv2d_context_a",
                second_context_attr="_vulkan_conv2d_context_b",
                plan_attr="_vulkan_graph_region_plan_a",
                program_name="VulkanGraphRegionPlan",
                program_version="v1",
                instruction_count=2,
                input_ssa=0,
                intermediate_ssa=1,
                output_ssa=2,
                input_use_count=1,
                input_last_use=0,
                intermediate_use_count=1,
                intermediate_last_use=1,
                first_static_context_slot=0,
                second_static_context_slot=1,
                bounded_submission_owned=True,
                program_private_scratch=True,
                scratch_ring_capacity=2,
                timeline_gated_release=True,
                direct_transition_only=True,
                replay_state_empty=True,
            ),
        ),
        excluded_relu_node_names=("relu_0",),
    )


class TestVulkanGraphEvidence(TestCase):
    def _long_session_soak_record(self) -> dict[str, Any]:
        return {
            "device_name": "AMD Radeon RX 9070",
            "configuration": {"requested_duration_seconds": 600},
            "measurement": {
                "elapsed_seconds": 600.0,
                "invocation_count": 3000,
                "parity_check_count": 3000,
                "recapture_count": 11,
                "runtime_counters": {
                    "cpu_fallback": 0,
                    "sync_readback": 0,
                },
                "submission_counters": {
                    "graph_program_invocation": {
                        "resource_arena_unsafe_slot_leak": 0,
                        "resource_arena_retirement_failure": 0,
                    },
                    "submit_origin": {"tensor_cpu_readback": 3000},
                },
                "memory": {
                    "replacement_preflight": {"high_water_bytes": 1100},
                    "soak": {
                        "baseline_live_bytes": 1000,
                        "end_live_bytes": 1050,
                        "high_water_bytes": 1155,
                    },
                },
            },
        }

    def test_long_session_soak_gate_accepts_registered_boundary(self):
        gate = _evaluate_long_session_soak_gate(
            self._long_session_soak_record()
        )
        self.assertEqual(gate["status"], "passed")
        self.assertTrue(gate["qualified_gate_run"])
        self.assertTrue(gate["all_checks_passed"])
        self.assertEqual(set(gate["checks"].values()), {True})

    def test_short_long_session_soak_is_diagnostic_only(self):
        record = self._long_session_soak_record()
        record["configuration"]["requested_duration_seconds"] = 60
        record["measurement"]["elapsed_seconds"] = 60.0
        record["measurement"]["invocation_count"] = 100
        record["measurement"]["parity_check_count"] = 100
        gate = _evaluate_long_session_soak_gate(record)
        self.assertEqual(gate["status"], "diagnostic_only")
        self.assertFalse(gate["qualified_gate_run"])

    def test_qualified_long_session_soak_rejects_memory_drift(self):
        record = self._long_session_soak_record()
        record["measurement"]["memory"]["soak"]["end_live_bytes"] = 1051
        gate = _evaluate_long_session_soak_gate(record)
        self.assertEqual(gate["status"], "failed")
        self.assertTrue(gate["qualified_gate_run"])
        self.assertFalse(gate["checks"]["final_live_bytes_bounded"])

    def test_qualified_long_session_soak_rejects_unsafe_resource_leak(self):
        record = self._long_session_soak_record()
        record["measurement"]["submission_counters"][
            "graph_program_invocation"
        ]["resource_arena_unsafe_slot_leak"] = 1
        gate = _evaluate_long_session_soak_gate(record)
        self.assertEqual(gate["status"], "failed")
        self.assertTrue(gate["qualified_gate_run"])
        self.assertFalse(
            gate["checks"]["zero_resource_arena_unsafe_slot_leak"]
        )

    def test_state_replay_mapping_is_explicit_and_one_to_one(self):
        self.assertEqual(_state_replay_mapping(None), ())
        self.assertEqual(
            _state_replay_mapping(((2, 1), [3, 2])),
            ((2, 1), (3, 2)),
        )
        for value, message in (
            ((), "non-empty sequence"),
            (((2, 1), (2, 3)), "input leaf indices must be unique"),
            (((2, 1), (3, 1)), "output leaf indices must be unique"),
            (((True, 1),), "nonnegative integer leaf indices"),
            (((-1, 1),), "nonnegative integer leaf indices"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                _state_replay_mapping(value)

    def test_device_and_planning_diagnostics_are_structured(self):
        self.assertEqual(_device_index("1"), 1)
        with self.assertRaises(argparse.ArgumentTypeError):
            _device_index("-1")
        with self.assertRaises(argparse.ArgumentTypeError):
            _device_index("not-an-index")

        route_row = (
            "vulkan_route op=aten::convolution lane=AdjacentDepthVision "
            "decision=VulkanBufferDirectKernel reason=None "
            "family=buffer_float_conv2d telemetry=SelectedBufferFloatConv2d "
            "hard_fail=0 shape={input=[1, 4, 8, 8]}"
        )
        runtime_policy_row = (
            "runtime_policy workload=VisionDecoder "
            "source_workload=Convolution model_domain=Vision "
            "execution_phase=Decoder tensor_role=Input fixed_shape_graph=1 "
            "prefer_packed_layout_propagation=1 backend_route=Vulkan "
            "linear_kernel_family=TexturePacked "
            "norm_kernel_family=TextureWidth "
            "attention_kernel_family=TextureMath "
            "attention_execution_strategy=GenericMath "
            "inferred_from_label=0"
        )

        summary = _planning_diagnostic_summary(
            [route_row, route_row],
            [runtime_policy_row, "runtime_policy workload=VisionDecoder builds=2"],
        )

        self.assertEqual(summary["route_lanes"], ["AdjacentDepthVision"])
        self.assertEqual(summary["route_decisions"][0]["count"], 2)
        self.assertEqual(summary["runtime_model_domains"], ["Vision"])
        self.assertEqual(summary["runtime_execution_phases"], ["Decoder"])
        self.assertEqual(summary["runtime_inferred_from_label_count"], 0)
        self.assertEqual(summary["runtime_policies"][0]["count"], 1)

    def test_residency_snapshots_exclude_allocation_identities(self):
        memory_rows = [
            "vulkan_memory_summary live_bytes=384 high_water_bytes=512",
            (
                "vulkan_memory_residency id=1 generation=1 kind=buffer "
                "state=live role=packed_weight requested_bytes=128 "
                "allocated_bytes=128 owns_memory=1 label=linear"
            ),
            (
                "vulkan_memory_residency id=2 generation=2 kind=image "
                "state=pending_retire role=activation requested_bytes=200 "
                "allocated_bytes=256 owns_memory=1 label=convolution"
            ),
        ]
        allocator = _allocator_residency_summary(memory_rows)
        self.assertEqual(allocator["allocation_count"], 2)
        self.assertEqual(allocator["requested_bytes"], 328)
        self.assertEqual(allocator["allocated_bytes"], 384)
        self.assertEqual(
            allocator["by_state"],
            [
                {
                    "value": "live",
                    "allocation_count": 1,
                    "allocated_bytes": 128,
                },
                {
                    "value": "pending_retire",
                    "allocation_count": 1,
                    "allocated_bytes": 256,
                },
            ],
        )
        self.assertNotIn("id", allocator)
        self.assertEqual(
            _int_summary_row(memory_rows, "vulkan_memory_summary "),
            {"high_water_bytes": 512, "live_bytes": 384},
        )

        linear_pack = _linear_pack_residency_summary(
            [
                (
                    "linear_pack_residency shape=4x4 count=1 created=1 "
                    "reused=0 packed_bytes=64 raw_weight_bytes=64 "
                    "raw_bias_bytes=16 raw_weight_vulkan=0 retain_unpacked=0"
                ),
                (
                    "linear_pack_residency shape=8x8 count=2 created=1 "
                    "reused=1 packed_bytes=256 raw_weight_bytes=256 "
                    "raw_bias_bytes=32 raw_weight_vulkan=1 retain_unpacked=1"
                ),
            ]
        )
        self.assertEqual(linear_pack["row_count"], 2)
        self.assertEqual(linear_pack["count"], 3)
        self.assertEqual(linear_pack["created"], 2)
        self.assertEqual(linear_pack["reused"], 1)
        self.assertEqual(linear_pack["packed_bytes"], 320)

    def test_planning_context_binds_each_case_primary_input_shape(self):
        args = SimpleNamespace(
            planning_model_domain="vision",
            planning_execution_phase="none",
            planning_prefer_packed_layout_propagation=True,
            planning_fixed_shape_graph_input=True,
        )

        context = _planning_context(args, (torch.randn(2, 3, 4),))

        self.assertEqual(context.model_domain, "vision")
        self.assertEqual(context.execution_phase, "none")
        self.assertTrue(context.prefer_packed_layout_propagation)
        self.assertEqual(context.fixed_shape_graph_input_sizes, (2, 3, 4))

    def test_named_counter_snapshot_rejects_schema_drift(self):
        self.assertEqual(
            _named_counter_snapshot(("first", "second"), [3, 5], "test"),
            {"first": 3, "second": 5},
        )
        with self.assertRaisesRegex(RuntimeError, "2 names for 1 values"):
            _named_counter_snapshot(("first", "second"), [3], "test")

    def test_graph_counts_include_all_graph_owned_lowering_families(self):
        census = SimpleNamespace(
            captured_node_count=100,
            call_function_node_count=80,
            lowered_vulkan_node_count=79,
            direct_vulkan_node_count=10,
            composite_node_count=5,
            unsupported_node_count=0,
            nodes=(
                SimpleNamespace(
                    target="aten::layer_norm",
                    classification="lowered_vulkan",
                ),
                SimpleNamespace(
                    target="aten::layer_norm",
                    classification="lowered_vulkan",
                ),
                SimpleNamespace(
                    target="aten::add",
                    classification="direct_vulkan",
                ),
                SimpleNamespace(
                    target="aten::relu",
                    classification="direct_vulkan",
                ),
                SimpleNamespace(
                    target="aten::sym_size.int",
                    classification="composite",
                ),
            ),
        )
        program = SimpleNamespace(
            census=census,
            linear_lowering=_FakeLoweringReport(created_context_count=48),
            static_linear_gelu_regions=_fake_static_linear_gelu_report(),
            conv2d_lowering=_FakeLoweringReport(created_context_count=31),
            layernorm_lowering=_FakeLoweringReport(created_context_count=28),
            static_add_layernorm_regions=_fake_static_add_layernorm_report(),
            static_conv2d_relu_conv2d_regions=(
                _fake_static_conv2d_relu_conv2d_report()
            ),
            static_conv2d_relu_regions=_fake_static_conv2d_relu_report(),
        )
        counts = _graph_counts(program)
        self.assertEqual(counts["statically_lowered"], 79)
        self.assertEqual(counts["graph_owned_prepacked_contexts"], 107)
        self.assertEqual(
            counts["direct_vulkan_by_target"],
            {"aten::add": 1, "aten::relu": 1},
        )
        self.assertEqual(
            list(counts["direct_vulkan_by_target"]),
            ["aten::add", "aten::relu"],
        )
        self.assertEqual(
            counts["composite_by_target"], {"aten::sym_size.int": 1}
        )
        self.assertEqual(
            counts["lowered_vulkan_by_target"], {"aten::layer_norm": 2}
        )
        self.assertEqual(
            counts["partition_candidates"],
            {
                "status": "not_planned_python_correctness_executor",
                "vulkan_only_candidate_count": 0,
            },
        )

    def test_graph_counts_and_evidence_describe_compiled_cpp_plan(self):
        census = SimpleNamespace(
            captured_node_count=6,
            call_function_node_count=4,
            lowered_vulkan_node_count=1,
            direct_vulkan_node_count=2,
            composite_node_count=1,
            unsupported_node_count=0,
            nodes=(),
        )
        report = SimpleNamespace(
            status="compiled",
            reason="immutable_ivalue_ssa_resource_plan",
            plan_class="VulkanGraphPlan",
            plan_version="v9",
            planning_model_domain="vision",
            planning_execution_phase="none",
            planning_prefer_packed_layout_propagation=False,
            planning_fixed_shape_graph_input_sizes=None,
            input_count=2,
            instruction_count=4,
            effect_instruction_count=1,
            graph_scalar_instruction_count=2,
            list_projection_instruction_count=1,
            list_argument_count=1,
            invocation_value_slot_count=5,
            invocation_list_slot_count=1,
            invocation_stack_capacity=4,
            dead_input_reuse_instruction_count=3,
            resource_slot_count=2,
            resource_value_count=3,
            resource_writer_instruction_count=2,
            resource_arena_flight_depth=2,
            recorded_partition_count=1,
            recorded_partition_instruction_count=3,
            resource_alias_extended_lifetime_count=1,
            resource_alias_escape_rejection_count=4,
            value_count=5,
            output_count=2,
            submission_owned=True,
        )
        program = SimpleNamespace(
            census=census,
            execution_mode="cpp_plan",
            cpp_plan=SimpleNamespace(
                invocation_generation=lambda: 2,
                last_submission_value=lambda: 37,
                last_submission_complete=lambda: True,
                dead_input_reuse_count=lambda: 7,
                resource_arena_generation_count=lambda: 2,
                resource_arena_capture_count=lambda: 2,
                resource_arena_reuse_count=lambda: 5,
                resource_arena_spill_count=lambda: 1,
                resource_write_count=lambda: 9,
                resource_writer_bypass_count=lambda: 0,
                recorded_partition_prime_count=lambda: 2,
                recorded_partition_capture_count=lambda: 2,
                recorded_partition_replay_count=lambda: 5,
                recorded_partition_failure_count=lambda: 0,
                recorded_partition_represented_dispatch_count=lambda: 21,
            ),
            cpp_plan_report=report,
        )

        self.assertEqual(
            _execution_plan_summary(program),
            {
                "mode": "cpp_plan",
                "status": "compiled",
                "reason": "immutable_ivalue_ssa_resource_plan",
                "plan_class": "VulkanGraphPlan",
                "plan_version": "v9",
                "planning_model_domain": "vision",
                "planning_execution_phase": "none",
                "planning_prefer_packed_layout_propagation": False,
                "planning_fixed_shape_graph_input_sizes": None,
                "input_count": 2,
                "instruction_count": 4,
                "effect_instruction_count": 1,
                "graph_scalar_instruction_count": 2,
                "list_projection_instruction_count": 1,
                "list_argument_count": 1,
                "invocation_value_slot_count": 5,
                "invocation_list_slot_count": 1,
                "invocation_stack_capacity": 4,
                "dead_input_reuse_instruction_count": 3,
                "resource_slot_count": 2,
                "resource_value_count": 3,
                "resource_writer_instruction_count": 2,
                "resource_arena_flight_depth": 2,
                "recorded_partition_count": 1,
                "recorded_partition_instruction_count": 3,
                "resource_alias_extended_lifetime_count": 1,
                "resource_alias_escape_rejection_count": 4,
                "value_count": 5,
                "output_count": 2,
                "submission_owned": True,
                "invocation_generation": 2,
                "last_submission_value": 37,
                "last_submission_complete": True,
                "dead_input_reuse_count": 7,
                "resource_arena_generation_count": 2,
                "resource_arena_capture_count": 2,
                "resource_arena_reuse_count": 5,
                "resource_arena_spill_count": 1,
                "resource_write_count": 9,
                "resource_writer_bypass_count": 0,
                "recorded_partition_prime_count": 2,
                "recorded_partition_capture_count": 2,
                "recorded_partition_replay_count": 5,
                "recorded_partition_failure_count": 0,
                "recorded_partition_represented_dispatch_count": 21,
            },
        )
        self.assertEqual(
            _graph_counts(program)["partition_candidates"],
            {
                "status": "compiled",
                "vulkan_only_candidate_count": 1,
            },
        )

    def test_lowering_reports_keep_single_family_programs_additive(self):
        linear_only = SimpleNamespace(
            linear_lowering=_FakeLoweringReport(created_context_count=2),
        )
        conv2d_only = SimpleNamespace(
            conv2d_lowering=_FakeLoweringReport(created_context_count=3),
        )
        self.assertEqual(
            _lowering_reports(linear_only),
            {
                "input_normalization": None,
                "static_factory_constants": None,
                "lifted_tensor_constants": None,
                "fresh_detach_functionalization": None,
                "fresh_relu_functionalization": None,
                "static_inference_identities": None,
                "static_identity_advanced_indices": None,
                "static_gqa_repeats": None,
                "static_sdpa_fusions": None,
                "tensor_placement": None,
                "linear_lowering": {"created_context_count": 2},
                "static_linear_gelu_regions": None,
                "conv2d_lowering": None,
                "layernorm_lowering": None,
                "static_add_layernorm_regions": None,
                "static_conv2d_relu_conv2d_regions": None,
                "static_conv2d_relu_regions": None,
                "vulkan_graph_regions": None,
            },
        )
        self.assertEqual(
            _lowering_reports(conv2d_only),
            {
                "input_normalization": None,
                "static_factory_constants": None,
                "lifted_tensor_constants": None,
                "fresh_detach_functionalization": None,
                "fresh_relu_functionalization": None,
                "static_inference_identities": None,
                "static_identity_advanced_indices": None,
                "static_gqa_repeats": None,
                "static_sdpa_fusions": None,
                "tensor_placement": None,
                "linear_lowering": None,
                "static_linear_gelu_regions": None,
                "conv2d_lowering": {"created_context_count": 3},
                "layernorm_lowering": None,
                "static_add_layernorm_regions": None,
                "static_conv2d_relu_conv2d_regions": None,
                "static_conv2d_relu_regions": None,
                "vulkan_graph_regions": None,
            },
        )

    def test_graph_preparation_reports_serialize_additively(self):
        preparation_reports = {
            "input_normalization": {"normalized_input_count": 1},
            "static_factory_constants": {"created_constant_count": 2},
            "lifted_tensor_constants": {"created_constant_count": 3},
            "fresh_detach_functionalization": {"functionalized_count": 4},
            "fresh_relu_functionalization": {"functionalized_count": 2},
            "static_inference_identities": {"lowered_count": 48},
            "static_identity_advanced_indices": {"lowered_count": 5},
            "static_gqa_repeats": {"lowered_count": 6},
            "static_sdpa_fusions": {"lowered_count": 12},
            "tensor_placement": {"buffer_direct_count": 7},
        }
        reports = _lowering_reports(SimpleNamespace(**preparation_reports))
        self.assertEqual(
            {name: reports[name] for name in preparation_reports},
            preparation_reports,
        )
        self.assertEqual(reports["linear_lowering"], None)

    def test_static_linear_gelu_regions_serialize_without_counting_context_twice(
        self,
    ):
        census = SimpleNamespace(
            captured_node_count=2,
            call_function_node_count=1,
            lowered_vulkan_node_count=1,
            direct_vulkan_node_count=0,
            composite_node_count=0,
            unsupported_node_count=0,
            nodes=(),
        )
        program = SimpleNamespace(
            census=census,
            linear_lowering=_FakeLoweringReport(created_context_count=1),
            static_linear_gelu_regions=_fake_static_linear_gelu_report(),
            conv2d_lowering=_FakeLoweringReport(created_context_count=1),
            layernorm_lowering=_FakeLoweringReport(created_context_count=0),
            static_add_layernorm_regions=_fake_static_add_layernorm_report(),
            static_conv2d_relu_conv2d_regions=(
                _fake_static_conv2d_relu_conv2d_report()
            ),
            static_conv2d_relu_regions=_fake_static_conv2d_relu_report(),
            vulkan_graph_regions=_fake_vulkan_graph_region_report(),
        )
        reports = _lowering_reports(program)
        static_report = reports["static_linear_gelu_regions"]
        layernorm_report = reports["layernorm_lowering"]
        static_add_layernorm_report = reports["static_add_layernorm_regions"]
        static_multi_conv_report = reports["static_conv2d_relu_conv2d_regions"]
        static_conv_report = reports["static_conv2d_relu_regions"]
        graph_regions = reports["vulkan_graph_regions"]
        self.assertEqual(_graph_counts(program)["graph_owned_prepacked_contexts"], 2)
        self.assertEqual(layernorm_report, {"created_context_count": 0})
        self.assertEqual(graph_regions["plan_class"], "VulkanGraphRegionPlan")
        self.assertEqual(graph_regions["plan_version"], "v1")
        self.assertEqual(graph_regions["families"][0]["family"], "linear_gelu_tanh")
        self.assertEqual(
            graph_regions["families"][0]["nodes"][0]["instruction_count"], 2
        )
        self.assertEqual(static_report["candidate_count"], 1)
        self.assertEqual(static_report["lowered_count"], 1)
        self.assertEqual(static_report["rejected_count"], 0)
        self.assertEqual(static_report["skipped_count"], 0)
        self.assertEqual(
            static_report["plan_factory"],
            "vulkan_prepack::create_vulkan_graph_region_plan_linear_gelu",
        )
        node = static_report["nodes"][0]
        self.assertEqual(node["status"], "lowered")
        self.assertEqual(node["reason"], "graph_owned_static_linear_tanh_gelu")
        self.assertEqual(node["linear_node_name"], "run_linear_context")
        self.assertEqual(node["context_attr"], "_vulkan_linear_context_a")
        self.assertEqual(
            node["plan_attr"], "_vulkan_graph_region_plan_a"
        )
        self.assertEqual(node["program_name"], "VulkanGraphRegionPlan")
        self.assertEqual(node["program_version"], "v1")
        self.assertEqual(node["instruction_count"], 2)
        self.assertEqual(node["input_ssa"], 0)
        self.assertEqual(node["intermediate_ssa"], 1)
        self.assertEqual(node["output_ssa"], 2)
        self.assertEqual(node["input_use_count"], 1)
        self.assertEqual(node["input_last_use"], 0)
        self.assertEqual(node["intermediate_use_count"], 1)
        self.assertEqual(node["intermediate_last_use"], 1)
        self.assertEqual(node["static_context_slot"], 0)
        self.assertEqual(node["region_family"], "linear_gelu_tanh")
        self.assertTrue(node["direct_transition_only"])
        self.assertTrue(node["replay_state_empty"])
        self.assertEqual(static_add_layernorm_report["candidate_count"], 1)
        self.assertEqual(static_add_layernorm_report["lowered_count"], 1)
        self.assertEqual(static_add_layernorm_report["rejected_count"], 0)
        self.assertEqual(static_add_layernorm_report["skipped_count"], 0)
        self.assertEqual(
            static_add_layernorm_report["plan_factory"],
            "vulkan_prepack::create_graph_add_layernorm_plan",
        )
        add_layernorm_node = static_add_layernorm_report["nodes"][0]
        self.assertEqual(add_layernorm_node["status"], "lowered")
        self.assertEqual(
            add_layernorm_node["reason"],
            "graph_owned_static_add_layernorm",
        )
        self.assertEqual(add_layernorm_node["add_node_name"], "add")
        self.assertEqual(
            add_layernorm_node["layernorm_node_name"],
            "run_layernorm_context",
        )
        self.assertEqual(
            add_layernorm_node["program_name"],
            "StaticAddLayernormRegion",
        )
        self.assertEqual(add_layernorm_node["program_version"], "v1")
        self.assertEqual(
            add_layernorm_node["fused_instruction"],
            "add_layernorm_fused_or_composed_vulkan",
        )
        self.assertEqual(add_layernorm_node["residual_input_ssa"], 0)
        self.assertEqual(add_layernorm_node["addend_input_ssa"], 1)
        self.assertEqual(add_layernorm_node["residual_output_ssa"], 2)
        self.assertEqual(add_layernorm_node["normalized_output_ssa"], 3)
        self.assertEqual(
            add_layernorm_node["context_ownership_outcome"],
            "transferred_removed_original_context_attr",
        )
        self.assertTrue(add_layernorm_node["direct_transition_only"])
        self.assertTrue(add_layernorm_node["replay_state_empty"])
        self.assertFalse(add_layernorm_node["persistent_output_state"])
        self.assertEqual(static_multi_conv_report["candidate_count"], 1)
        self.assertEqual(static_multi_conv_report["lowered_count"], 1)
        self.assertEqual(static_multi_conv_report["rejected_count"], 0)
        self.assertEqual(static_multi_conv_report["skipped_count"], 0)
        self.assertEqual(
            static_multi_conv_report["excluded_relu_node_names"], ["relu_0"]
        )
        self.assertEqual(
            static_multi_conv_report["plan_factory"],
            "vulkan_prepack::create_vulkan_graph_region_plan_conv2d_relu_conv2d",
        )
        multi_node = static_multi_conv_report["nodes"][0]
        self.assertEqual(multi_node["status"], "lowered")
        self.assertEqual(
            multi_node["reason"], "graph_owned_static_conv2d_relu_conv2d"
        )
        self.assertEqual(multi_node["program_name"], "VulkanGraphRegionPlan")
        self.assertEqual(multi_node["program_version"], "v1")
        self.assertEqual(multi_node["instruction_count"], 2)
        self.assertEqual(multi_node["input_ssa"], 0)
        self.assertEqual(multi_node["intermediate_ssa"], 1)
        self.assertEqual(multi_node["output_ssa"], 2)
        self.assertEqual(multi_node["input_use_count"], 1)
        self.assertEqual(multi_node["input_last_use"], 0)
        self.assertEqual(multi_node["intermediate_use_count"], 1)
        self.assertEqual(multi_node["intermediate_last_use"], 1)
        self.assertEqual(multi_node["first_static_context_slot"], 0)
        self.assertEqual(multi_node["second_static_context_slot"], 1)
        self.assertTrue(multi_node["bounded_submission_owned"])
        self.assertTrue(multi_node["program_private_scratch"])
        self.assertEqual(multi_node["scratch_ring_capacity"], 2)
        self.assertTrue(multi_node["timeline_gated_release"])
        self.assertTrue(multi_node["direct_transition_only"])
        self.assertTrue(multi_node["replay_state_empty"])
        self.assertEqual(static_conv_report["candidate_count"], 1)
        self.assertEqual(static_conv_report["lowered_count"], 1)
        self.assertEqual(static_conv_report["rejected_count"], 0)
        self.assertEqual(static_conv_report["skipped_count"], 0)
        self.assertEqual(
            static_conv_report["plan_factory"],
            "vulkan_prepack::create_graph_conv2d_relu_plan",
        )
        conv_node = static_conv_report["nodes"][0]
        self.assertEqual(conv_node["status"], "lowered")
        self.assertEqual(conv_node["reason"], "graph_owned_static_conv2d_relu")
        self.assertEqual(conv_node["conv2d_node_name"], "run_conv2d_context")
        self.assertEqual(conv_node["context_attr"], "_vulkan_conv2d_context_a")
        self.assertEqual(
            conv_node["plan_attr"], "_vulkan_static_conv2d_relu_plan_a"
        )
        self.assertEqual(conv_node["program_name"], "StaticConv2dReluRegion")
        self.assertEqual(conv_node["program_version"], "v1")
        self.assertEqual(conv_node["instruction_count"], 1)
        self.assertEqual(conv_node["input_ssa"], 0)
        self.assertEqual(conv_node["output_ssa"], 1)
        self.assertEqual(conv_node["input_use_count"], 1)
        self.assertEqual(conv_node["input_last_use"], 0)
        self.assertEqual(conv_node["static_context_slot"], 0)
        self.assertTrue(conv_node["direct_transition_only"])
        self.assertTrue(conv_node["replay_state_empty"])

    def test_checked_in_corpus_evidence_is_schema_valid_and_measured(self):
        evidence_dir = Path(__file__).parent / "vulkan_graph" / "evidence"
        for name, artifact_type in (
            ("dav2_vits_export_census.json", "export_census"),
            ("dav2_vits_export_parity.json", "parity"),
            ("hymt_decode_export_census.json", "export_census"),
            ("hymt_decode_export_parity.json", "parity"),
            ("hymt_prefill_export_census.json", "export_census"),
            ("hymt_prefill_export_parity.json", "parity"),
            ("paddleocr_recognition_export_census.json", "export_census"),
            ("paddleocr_recognition_export_parity.json", "parity"),
        ):
            payload = json.loads((evidence_dir / name).read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], EVIDENCE_SCHEMA)
            self.assertEqual(payload["artifact_type"], artifact_type)
            self.assertEqual(payload["status"], "measured")
            self.assertEqual(len(payload["cases"]), 2)
            self.assertIsInstance(payload["source_git_sha"], str)
            self.assertEqual(validate_evidence_payload(payload), [])

    def test_checked_in_hymt_evidence_keeps_open_gates_explicit(self):
        evidence_dir = Path(__file__).parent / "vulkan_graph" / "evidence"
        payloads = tuple(
            json.loads((evidence_dir / name).read_text(encoding="utf-8"))
            for name in (
                "hymt_prefill_export_census.json",
                "hymt_prefill_export_parity.json",
            )
        )
        for payload in payloads:
            self.assertEqual(
                payload["source_git_sha"],
                "019faaebf1593fd2f2fcbd8e5cec66a8202fd62e",
            )
            self.assertEqual(
                payload["runtime"]["loaded_files"]["torch_cpu.dll"]["sha256"],
                "b2ba673689b6110d06861884456a50808e294e62ee4c64e195b3d5f5256cd2c1",
            )
            plan = payload["execution_plan"]
            self.assertEqual(plan["status"], "compiled")
            self.assertEqual(plan["plan_version"], "v8")
            self.assertEqual(plan["instruction_count"], 2668)
            self.assertEqual(plan["value_count"], 2402)
            self.assertEqual(plan["output_count"], 65)
            self.assertTrue(plan["submission_owned"])
            self.assertEqual(
                (
                    payload["fresh_detach_functionalization"]["candidate_count"],
                    payload["fresh_detach_functionalization"][
                        "functionalized_count"
                    ],
                    payload["fresh_detach_functionalization"]["rejected_count"],
                ),
                (64, 64, 0),
            )
        for case in payloads[0]["cases"]:
            self.assertEqual(set(case["runtime_counters"].values()), {0})

        parity = payloads[1]
        cases = {case["name"]: case for case in parity["cases"]}
        self.assertEqual(cases["normal"]["guard"]["status"], "accepted")
        self.assertEqual(
            cases["alternate"]["guard"]["status"],
            "recompiled_guard_variant",
        )
        for case in cases.values():
            self.assertLessEqual(
                case["graph_vs_eager_vulkan"]["max_abs"],
                case["tolerance"]["graph_vs_eager_vulkan"]["atol"],
            )
            self.assertLessEqual(
                case["graph_vs_cpu"]["max_abs"],
                case["tolerance"]["graph_vs_cpu"]["atol"],
            )
            self.assertEqual(
                case["submission_counters"]["submit_origin"][
                    "pending_command_flush"
                ],
                176,
            )
            eager_peak = case["memory"]["eager"]["high_water_bytes"]
            for phase in (
                "graph_first",
                "graph_repeat_with_prior_output_live",
            ):
                self.assertLessEqual(
                    case["memory"][phase]["high_water_bytes"],
                    eager_peak * 1.05,
                )

            eager_planning = case["planning_diagnostics"]["supported_eager"]
            graph_planning = case["planning_diagnostics"][
                "vulkan_graph_program_first"
            ]
            self.assertEqual(eager_planning["route_lanes"], ["DepthDiffusion"])
            self.assertEqual(graph_planning["route_lanes"], ["LLM"])
            self.assertEqual(
                eager_planning["runtime_model_domains"],
                ["Generic", "LLM"],
            )
            self.assertEqual(graph_planning["runtime_model_domains"], ["LLM"])

            latency = case["timing"]["supported_default_latency"]
            self.assertEqual(latency["warmup_repeats_per_surface"], 3)
            self.assertEqual(latency["measurement_repeats_per_surface"], 30)
            self.assertEqual(
                latency["supported_eager"]["runtime_counters"],
                {"cpu_fallback": 150, "sync_readback": 30},
            )
            self.assertEqual(
                latency["vulkan_graph_program"]["runtime_counters"],
                {"cpu_fallback": 0, "sync_readback": 0},
            )
            for surface in ("supported_eager", "vulkan_graph_program"):
                samples = latency[surface]["samples_seconds"]
                self.assertEqual(len(samples), 30)
                self.assertEqual(
                    latency[surface]["median_seconds"],
                    statistics.median(samples),
                )

        self.assertGreater(
            cases["normal"]["timing"]["supported_default_latency"][
                "median_ratio_graph_over_eager"
            ],
            1.0,
        )
        self.assertLess(
            cases["alternate"]["timing"]["supported_default_latency"][
                "median_ratio_graph_over_eager"
            ],
            1.0,
        )

    def test_checked_in_hymt_decode_evidence_replays_device_state(self):
        evidence_dir = Path(__file__).parent / "vulkan_graph" / "evidence"
        payloads = tuple(
            json.loads((evidence_dir / name).read_text(encoding="utf-8"))
            for name in (
                "hymt_decode_export_census.json",
                "hymt_decode_export_parity.json",
            )
        )
        for payload in payloads:
            self.assertEqual(
                payload["source_git_sha"],
                "79bf8d01ef0db5c01997042071e12434eac1b443",
            )
            self.assertEqual(
                payload["runtime"]["loaded_files"]["torch_cpu.dll"]["sha256"],
                "b2ba673689b6110d06861884456a50808e294e62ee4c64e195b3d5f5256cd2c1",
            )
            plan = payload["execution_plan"]
            self.assertEqual(plan["status"], "compiled")
            self.assertEqual(plan["plan_version"], "v8")
            self.assertEqual(plan["planning_model_domain"], "llm")
            self.assertEqual(plan["planning_execution_phase"], "decode")
            self.assertEqual(plan["input_count"], 66)
            self.assertEqual(plan["instruction_count"], 2732)
            self.assertEqual(plan["value_count"], 2530)
            self.assertEqual(plan["output_count"], 65)
            self.assertTrue(plan["submission_owned"])
            self.assertEqual(
                (
                    payload["fresh_detach_functionalization"]["candidate_count"],
                    payload["fresh_detach_functionalization"][
                        "functionalized_count"
                    ],
                    payload["fresh_detach_functionalization"]["rejected_count"],
                ),
                (64, 64, 0),
            )

        parity = payloads[1]
        cases = {case["name"]: case for case in parity["cases"]}
        self.assertEqual(cases["normal"]["guard"]["status"], "accepted")
        self.assertEqual(
            cases["alternate"]["guard"]["status"],
            "recompiled_guard_variant",
        )
        self.assertEqual(
            cases["normal"]["submission_counters"]["submit_origin"][
                "pending_command_flush"
            ],
            173,
        )
        self.assertEqual(
            cases["alternate"]["submission_counters"]["submit_origin"][
                "pending_command_flush"
            ],
            172,
        )
        for case in cases.values():
            self.assertEqual(len(case["input_shape"]), 66)
            self.assertLessEqual(
                case["graph_vs_eager_vulkan"]["max_abs"],
                case["tolerance"]["graph_vs_eager_vulkan"]["atol"],
            )
            self.assertLessEqual(
                case["graph_vs_cpu"]["max_abs"],
                case["tolerance"]["graph_vs_cpu"]["atol"],
            )
            eager_peak = case["memory"]["eager"]["high_water_bytes"]
            for phase in (
                "graph_first",
                "graph_repeat_with_prior_output_live",
            ):
                self.assertLessEqual(
                    case["memory"][phase]["high_water_bytes"],
                    eager_peak * 1.05,
                )
            self.assertEqual(
                case["planning_diagnostics"]["supported_eager"]["route_lanes"],
                ["DepthDiffusion"],
            )
            self.assertEqual(
                case["planning_diagnostics"]["vulkan_graph_program_first"][
                    "route_lanes"
                ],
                ["LLM"],
            )
            latency = case["timing"]["supported_default_latency"]
            self.assertEqual(latency["warmup_repeats_per_surface"], 3)
            self.assertEqual(latency["measurement_repeats_per_surface"], 30)
            self.assertEqual(
                latency["supported_eager"]["runtime_counters"],
                {"cpu_fallback": 150, "sync_readback": 30},
            )
            self.assertEqual(
                latency["vulkan_graph_program"]["runtime_counters"],
                {"cpu_fallback": 0, "sync_readback": 0},
            )
            self.assertGreater(latency["median_ratio_graph_over_eager"], 1.0)
            for surface in ("supported_eager", "vulkan_graph_program"):
                self.assertEqual(len(latency[surface]["samples_seconds"]), 30)

        replay = parity["state_replay"]
        self.assertEqual(replay["status"], "passed")
        self.assertEqual(
            replay["protocol"],
            "explicit_output_to_input_tensor_leaves",
        )
        self.assertEqual(replay["mapped_leaf_count"], 64)
        self.assertEqual(
            (
                replay["mapping"][0]["input_leaf_index"],
                replay["mapping"][0]["output_leaf_index"],
            ),
            (2, 1),
        )
        self.assertEqual(
            (
                replay["mapping"][-1]["input_leaf_index"],
                replay["mapping"][-1]["output_leaf_index"],
            ),
            (65, 64),
        )
        self.assertNotEqual(
            replay["source_program_key"],
            replay["target_program_key"],
        )
        for prefix in ("source", "target"):
            self.assertEqual(
                set(replay[f"{prefix}_runtime_counters"].values()),
                {0},
            )
            self.assertEqual(
                replay[f"{prefix}_invocation_generation_after"]
                - replay[f"{prefix}_invocation_generation_before"],
                1,
            )
        submit = replay["submission_counters"]["submit_origin"]
        self.assertEqual(submit["host_upload"], 68)
        self.assertEqual(submit["pending_command_flush"], 172)
        self.assertEqual(submit["tensor_cpu_readback"], 0)
        self.assertEqual(submit["fallback_readback"], 0)
        self.assertEqual(submit["retire_queue_drain"], 0)
        self.assertEqual(submit["total_queue_submits"], 240)
        self.assertTrue(replay["source_output_preserved_after_target"])
        for field in (
            "replayed_state_vs_cpu",
            "target_output_vs_cpu",
            "source_output_after_target_vs_cpu",
        ):
            self.assertLessEqual(
                replay[field]["max_abs"],
                replay["tolerance"]["atol"],
            )

    def test_checked_in_corpus_evidence_records_graph_plan_progress(self):
        evidence_dir = Path(__file__).parent / "vulkan_graph" / "evidence"

        def load(name):
            return json.loads((evidence_dir / name).read_text(encoding="utf-8"))

        dav2_census = load("dav2_vits_export_census.json")
        dav2_parity = load("dav2_vits_export_parity.json")
        paddle_census = load("paddleocr_recognition_export_census.json")
        paddle_parity = load("paddleocr_recognition_export_parity.json")
        payloads = (dav2_census, dav2_parity, paddle_census, paddle_parity)
        self.assertEqual(
            {
                payload["source_git_sha"]
                for payload in (dav2_census, dav2_parity)
            },
            {"46ece5d7dc93a558837102d40fe5a7e20380397d"},
        )
        self.assertEqual(
            {
                payload["source_git_sha"]
                for payload in (paddle_census, paddle_parity)
            },
            {"4b688faac338f3784a1286a327292735a3b334b0"},
        )
        self.assertEqual(
            {
                payload["runtime"]["loaded_files"]["torch_cpu.dll"]["sha256"]
                for payload in (dav2_census, dav2_parity)
            },
            {
                "a1829062fbba8a8b6082d435344c231863246221ac8242ee0472ab5700a304f1"
            },
        )
        self.assertEqual(
            {
                payload["runtime"]["loaded_files"]["torch_cpu.dll"]["sha256"]
                for payload in (paddle_census, paddle_parity)
            },
            {
                "537802036062d3277a4d74ad7a27f28a76a16f7f4a4022c1ff1e132052989a9f"
            },
        )
        for payload in payloads:
            self.assertEqual(payload["execution_plan"]["plan_version"], "v8")
            self.assertEqual(payload["execution_plan"]["status"], "compiled")
            self.assertEqual(
                payload["execution_plan"]["reason"],
                "immutable_ivalue_ssa_plan",
            )
        for payload in (dav2_census, dav2_parity):
            self.assertEqual(
                (
                    payload["execution_plan"]["instruction_count"],
                    payload["execution_plan"]["effect_instruction_count"],
                    payload["execution_plan"][
                        "graph_scalar_instruction_count"
                    ],
                    payload["execution_plan"][
                        "list_projection_instruction_count"
                    ],
                    payload["execution_plan"]["list_argument_count"],
                    payload["execution_plan"]["value_count"],
                    payload["execution_plan"]["output_count"],
                ),
                (356, 2, 8, 20, 53, 377, 1),
            )
            self.assertEqual(
                payload["execution_plan"]["invocation_value_slot_count"],
                377,
            )
            self.assertEqual(
                payload["execution_plan"]["invocation_list_slot_count"],
                33,
            )
            self.assertEqual(
                payload["execution_plan"]["invocation_stack_capacity"],
                8,
            )
            self.assertTrue(payload["execution_plan"]["submission_owned"])
            self.assertEqual(
                payload["execution_plan"]["invocation_generation"], 2
            )
            self.assertGreater(
                payload["execution_plan"]["last_submission_value"], 0
            )
            self.assertTrue(
                payload["execution_plan"]["last_submission_complete"]
            )
            self.assertEqual(
                payload["execution_plan"][
                    "dead_input_reuse_instruction_count"
                ],
                9,
            )
            self.assertEqual(
                payload["execution_plan"]["dead_input_reuse_count"], 4
            )
            self.assertEqual(
                (
                    payload["fresh_relu_functionalization"][
                        "candidate_count"
                    ],
                    payload["fresh_relu_functionalization"][
                        "functionalized_count"
                    ],
                    payload["fresh_relu_functionalization"]["rejected_count"],
                ),
                (2, 2, 0),
            )
            self.assertEqual(
                (
                    payload["static_inference_identities"]["candidate_count"],
                    payload["static_inference_identities"]["lowered_count"],
                    payload["static_inference_identities"]["skipped_count"],
                ),
                (48, 48, 0),
            )
        for payload in (paddle_census, paddle_parity):
            self.assertEqual(
                (
                    payload["execution_plan"]["instruction_count"],
                    payload["execution_plan"]["effect_instruction_count"],
                    payload["execution_plan"][
                        "graph_scalar_instruction_count"
                    ],
                    payload["execution_plan"][
                        "list_projection_instruction_count"
                    ],
                    payload["execution_plan"]["list_argument_count"],
                    payload["execution_plan"]["value_count"],
                    payload["execution_plan"]["output_count"],
                ),
                (290, 1, 0, 0, 14, 294, 1),
            )
            self.assertTrue(payload["execution_plan"]["submission_owned"])
            self.assertEqual(
                payload["execution_plan"]["invocation_generation"], 4
            )
            self.assertGreater(
                payload["execution_plan"]["last_submission_value"], 0
            )
            self.assertTrue(
                payload["execution_plan"]["last_submission_complete"]
            )
            self.assertEqual(
                payload["execution_plan"][
                    "dead_input_reuse_instruction_count"
                ],
                53,
            )
            self.assertEqual(
                payload["execution_plan"]["dead_input_reuse_count"], 212
            )
            self.assertEqual(
                payload["fresh_relu_functionalization"]["candidate_count"],
                0,
            )
        self.assertEqual(
            (
                dav2_census["static_linear_gelu_regions"]["candidate_count"],
                dav2_census["static_linear_gelu_regions"]["lowered_count"],
                dav2_census["static_linear_gelu_regions"]["rejected_count"],
            ),
            (12, 12, 0),
        )
        self.assertEqual(
            (
                paddle_census["static_linear_gelu_regions"]["candidate_count"],
                paddle_census["static_linear_gelu_regions"]["lowered_count"],
                paddle_census["static_linear_gelu_regions"]["rejected_count"],
            ),
            (0, 0, 0),
        )
        for census in (dav2_census, paddle_census):
            self.assertEqual(
                census["graph_census"]["unsupported_at_lower_time"], 0
            )
            for case in census["cases"]:
                self.assertEqual(set(case["runtime_counters"].values()), {0})
        for payload in payloads:
            for case in payload["cases"]:
                memory = case["memory"]
                self.assertEqual(
                    set(memory),
                    {
                        "eager",
                        "graph_first",
                        "graph_repeat_with_prior_output_live",
                    },
                )
                for phase in memory.values():
                    self.assertEqual(
                        set(phase),
                        {
                            "baseline_live_bytes",
                            "end_live_bytes",
                            "high_water_bytes",
                            "peak_delta_bytes",
                            "residency",
                        },
                    )
                    self.assertEqual(
                        set(phase["residency"]),
                        {
                            "allocator",
                            "linear_pack",
                            "packed_weight_cache",
                        },
                    )
                    self.assertGreaterEqual(
                        phase["high_water_bytes"],
                        phase["baseline_live_bytes"],
                    )
                    self.assertGreaterEqual(
                        phase["high_water_bytes"], phase["end_live_bytes"]
                    )
                    self.assertEqual(
                        phase["peak_delta_bytes"],
                        phase["high_water_bytes"]
                        - phase["baseline_live_bytes"],
                    )
                eager_peak = memory["eager"]["high_water_bytes"]
                for graph_phase in (
                    "graph_first",
                    "graph_repeat_with_prior_output_live",
                ):
                    self.assertLessEqual(
                        memory[graph_phase]["high_water_bytes"],
                        eager_peak * 1.05,
                    )
                latency = case["timing"]["supported_default_latency"]
                self.assertEqual(
                    set(latency),
                    {
                        "method",
                        "input_boundary",
                        "output_readback_in_timed_region",
                        "synchronization",
                        "measurement_order",
                        "warmup_repeats_per_surface",
                        "measurement_repeats_per_surface",
                        "supported_eager",
                        "vulkan_graph_program",
                        "median_ratio_graph_over_eager",
                        "median_delta_percent",
                        "graph_invocation_generation_before",
                        "graph_invocation_generation_after",
                    },
                )
                self.assertEqual(
                    latency["method"],
                    "alternating_completed_device_resident_invocations",
                )
                self.assertEqual(
                    latency["input_boundary"],
                    "preuploaded_vulkan_inputs_to_completed_vulkan_outputs",
                )
                self.assertFalse(latency["output_readback_in_timed_region"])
                self.assertEqual(latency["warmup_repeats_per_surface"], 3)
                self.assertEqual(latency["measurement_repeats_per_surface"], 30)
                for surface in ("supported_eager", "vulkan_graph_program"):
                    summary = latency[surface]
                    self.assertEqual(
                        set(summary),
                        {
                            "count",
                            "mean_seconds",
                            "median_seconds",
                            "min_seconds",
                            "max_seconds",
                            "stdev_seconds",
                            "p90_seconds",
                            "p95_seconds",
                            "samples_seconds",
                            "runtime_counters",
                        },
                    )
                    samples = summary["samples_seconds"]
                    self.assertEqual(summary["count"], 30)
                    self.assertEqual(len(samples), 30)
                    self.assertTrue(all(sample > 0.0 for sample in samples))
                    self.assertEqual(summary["mean_seconds"], statistics.fmean(samples))
                    self.assertEqual(
                        summary["median_seconds"], statistics.median(samples)
                    )
                    self.assertEqual(summary["min_seconds"], min(samples))
                    self.assertEqual(summary["max_seconds"], max(samples))
                    self.assertEqual(
                        summary["stdev_seconds"], statistics.pstdev(samples)
                    )
                    self.assertEqual(
                        summary["runtime_counters"],
                        {"cpu_fallback": 0, "sync_readback": 0},
                    )
                eager_latency = latency["supported_eager"]
                graph_latency = latency["vulkan_graph_program"]
                self.assertLessEqual(
                    graph_latency["median_seconds"], eager_latency["median_seconds"]
                )
                self.assertLessEqual(
                    graph_latency["p95_seconds"], eager_latency["p95_seconds"]
                )
                median_ratio = (
                    graph_latency["median_seconds"] / eager_latency["median_seconds"]
                )
                self.assertAlmostEqual(
                    latency["median_ratio_graph_over_eager"], median_ratio
                )
                self.assertAlmostEqual(
                    latency["median_delta_percent"], (median_ratio - 1.0) * 100.0
                )
                self.assertEqual(
                    latency["graph_invocation_generation_after"]
                    - latency["graph_invocation_generation_before"],
                    33,
                )
        expected_dav2_graph_counters = {
            "scope_begun": 2,
            "normal_submit_token_capture": 2,
            "aborted_submit": 0,
            "rejected_incompatible_state": 0,
            "bounded_region_host_sync_rejected": 0,
            "scratch_captured": 8,
            "scratch_reused": 8,
            "scratch_transient_overflow": 0,
            "scratch_retire_enqueued": 0,
            "scratch_immediate_release": 0,
        }
        expected_dav2_submit_origins = {
            "total_queue_submits": 24,
            "normal_cmd_submit_frequency": 0,
            "stack_planned_recording_submit": 0,
            "pre_stack_flush": 0,
            "post_stack_flush": 0,
            "explicit_synchronize": 0,
            "tensor_cpu_readback": 2,
            "host_upload": 2,
            "fallback_readback": 0,
            "retire_queue_drain": 0,
            "profiling_timestamp_reset": 0,
            "profiling_timestamp_readback": 0,
            "shutdown": 0,
            "debug_validation": 0,
            "conv_prepack_upload": 0,
            "pending_command_flush": 20,
            "unknown": 0,
        }
        for payload in (dav2_census, dav2_parity):
            for case in payload["cases"]:
                self.assertTrue(case["execution_plan"]["submission_owned"])
                self.assertEqual(
                    case["execution_plan"]["invocation_generation"], 2
                )
                self.assertGreater(
                    case["execution_plan"]["last_submission_value"], 0
                )
                self.assertTrue(
                    case["execution_plan"]["last_submission_complete"]
                )
                self.assertEqual(
                    case["submission_counters"]["graph_program_invocation"],
                    expected_dav2_graph_counters,
                )
                self.assertEqual(
                    case["submission_counters"]["submit_origin"],
                    expected_dav2_submit_origins,
                )
        expected_paddle_graph_counters = {
            "scope_begun": 2,
            "normal_submit_token_capture": 2,
            "aborted_submit": 0,
            "rejected_incompatible_state": 0,
            "bounded_region_host_sync_rejected": 0,
            "scratch_captured": 0,
            "scratch_reused": 0,
            "scratch_transient_overflow": 0,
            "scratch_retire_enqueued": 0,
            "scratch_immediate_release": 0,
        }
        expected_paddle_submit_origins = {
            "total_queue_submits": 26,
            "normal_cmd_submit_frequency": 0,
            "stack_planned_recording_submit": 0,
            "pre_stack_flush": 0,
            "post_stack_flush": 0,
            "explicit_synchronize": 0,
            "tensor_cpu_readback": 2,
            "host_upload": 2,
            "fallback_readback": 0,
            "retire_queue_drain": 0,
            "profiling_timestamp_reset": 0,
            "profiling_timestamp_readback": 0,
            "shutdown": 0,
            "debug_validation": 0,
            "conv_prepack_upload": 0,
            "pending_command_flush": 22,
            "unknown": 0,
        }
        for payload in (paddle_census, paddle_parity):
            self.assertEqual(
                [
                    case["execution_plan"]["invocation_generation"]
                    for case in payload["cases"]
                ],
                [2, 4],
            )
            for case in payload["cases"]:
                self.assertTrue(case["execution_plan"]["submission_owned"])
                self.assertGreater(
                    case["execution_plan"]["last_submission_value"], 0
                )
                self.assertTrue(
                    case["execution_plan"]["last_submission_complete"]
                )
                self.assertEqual(
                    case["submission_counters"]["graph_program_invocation"],
                    expected_paddle_graph_counters,
                )
                self.assertEqual(
                    case["submission_counters"]["submit_origin"],
                    expected_paddle_submit_origins,
                )
        for parity in (dav2_parity, paddle_parity):
            for case in parity["cases"]:
                self.assertEqual(case["graph_vs_eager_vulkan"]["max_abs"], 0.0)

    def test_template_helper_does_not_claim_machine_measurement(self):
        payload = template_payload("export_census")
        self.assertEqual(payload["status"], "template_not_measured")
        self.assertEqual(validate_evidence_payload(payload), [])
        payload["external_assets"]["checkpoint"] = r"C:\machine\model.pth"
        self.assertIn("absolute paths", validate_evidence_payload(payload)[0])

    def test_external_assets_are_required_and_shapes_are_checked(self):
        with self.assertRaisesRegex(
            ExternalGraphEvidenceSetupError, "--external-root"
        ):
            require_external_assets(None, None)
        self.assertEqual(parse_input_shape("1,3,140,280"), (1, 3, 140, 280))
        with self.assertRaisesRegex(ValueError, "four positive"):
            parse_input_shape("1,3,0,280")

    def test_artifact_prefix_produces_safe_corpus_specific_filenames(self):
        output_dir = Path("artifacts")
        self.assertEqual(_artifact_prefix("dav2_vits"), "dav2_vits")
        self.assertEqual(_artifact_prefix("paddleocr"), "paddleocr")
        self.assertEqual(
            _artifact_output_paths(output_dir, "paddleocr"),
            (
                output_dir / "paddleocr_export_census.json",
                output_dir / "paddleocr_export_parity.json",
            ),
        )
        for prefix in ("", "../paddleocr", r"paddleocr\\run", "paddle ocr", "."):
            with self.assertRaises(argparse.ArgumentTypeError):
                _artifact_prefix(prefix)

    def test_latency_repeat_counts_and_summary_are_explicit(self):
        self.assertEqual(_nonnegative_repeat_count("0"), 0)
        self.assertEqual(_positive_repeat_count("10"), 10)
        with self.assertRaises(argparse.ArgumentTypeError):
            _nonnegative_repeat_count("-1")
        with self.assertRaises(argparse.ArgumentTypeError):
            _positive_repeat_count("0")

        summary = _summarize_latency_samples([0.04, 0.01, 0.03, 0.02])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["samples_seconds"], [0.04, 0.01, 0.03, 0.02])
        self.assertAlmostEqual(summary["mean_seconds"], 0.025)
        self.assertAlmostEqual(summary["median_seconds"], 0.025)
        self.assertAlmostEqual(summary["p90_seconds"], 0.037)
        self.assertAlmostEqual(summary["p95_seconds"], 0.0385)

    def test_guard_variant_only_handles_export_guard_rejections(self):
        self.assertTrue(
            _is_export_guard_rejection(
                torch.vulkan.VulkanGraphExecutionError(
                    "Vulkan graph node '_guards_fn' failed: Guard failed: x"
                )
            )
        )
        self.assertTrue(
            _is_export_guard_rejection(
                torch.vulkan.VulkanGraphExecutionError(
                    "fixed graph input shape mismatch: expected (2, 3), "
                    "got (4, 3)"
                )
            )
        )
        self.assertFalse(
            _is_export_guard_rejection(
                torch.vulkan.VulkanGraphExecutionError("Vulkan dispatch failed")
            )
        )
        self.assertEqual(
            _adapter_identity(r"C:\scratch\dav2_adapter.py:build"),
            "dav2_adapter.py:build",
        )

    def test_measured_parity_cases_require_and_accept_shared_case_fields(self):
        payload = {
            "schema": EVIDENCE_SCHEMA,
            "artifact_type": "parity",
            "status": "measured",
            "cases": [
                {
                    "name": "normal",
                    "input_shape": [[1, 3, 140, 140]],
                    "timing": {"first_run_seconds": 1.0},
                    "guard": {"status": "accepted"},
                },
                {
                    "name": "alternate",
                    "input_shape": [[1, 3, 140, 280]],
                    "timing": {"first_run_seconds": 1.0},
                    "guard": {"status": "recompiled_guard_variant"},
                },
            ],
        }
        self.assertEqual(validate_evidence_payload(payload), [])


if __name__ == "__main__":
    run_tests()
