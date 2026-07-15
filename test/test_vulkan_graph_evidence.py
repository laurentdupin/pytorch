import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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
    _artifact_output_paths,
    _artifact_prefix,
    _execution_plan_summary,
    _graph_counts,
    _is_export_guard_rejection,
    _lowering_reports,
    _named_counter_snapshot,
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
            reason="immutable_ivalue_ssa_plan",
            plan_class="VulkanGraphPlan",
            plan_version="v8",
            input_count=2,
            instruction_count=4,
            effect_instruction_count=1,
            graph_scalar_instruction_count=2,
            list_projection_instruction_count=1,
            list_argument_count=1,
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
            ),
            cpp_plan_report=report,
        )

        self.assertEqual(
            _execution_plan_summary(program),
            {
                "mode": "cpp_plan",
                "status": "compiled",
                "reason": "immutable_ivalue_ssa_plan",
                "plan_class": "VulkanGraphPlan",
                "plan_version": "v8",
                "input_count": 2,
                "instruction_count": 4,
                "effect_instruction_count": 1,
                "graph_scalar_instruction_count": 2,
                "list_projection_instruction_count": 1,
                "list_argument_count": 1,
                "value_count": 5,
                "output_count": 2,
                "submission_owned": True,
                "invocation_generation": 2,
                "last_submission_value": 37,
                "last_submission_complete": True,
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
                "static_identity_advanced_indices": None,
                "static_gqa_repeats": None,
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
                "static_identity_advanced_indices": None,
                "static_gqa_repeats": None,
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
            "static_identity_advanced_indices": {"lowered_count": 5},
            "static_gqa_repeats": {"lowered_count": 6},
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

    def test_checked_in_corpus_evidence_records_graph_plan_progress(self):
        evidence_dir = Path(__file__).parent / "vulkan_graph" / "evidence"

        def load(name):
            return json.loads((evidence_dir / name).read_text(encoding="utf-8"))

        dav2_census = load("dav2_vits_export_census.json")
        dav2_parity = load("dav2_vits_export_parity.json")
        paddle_census = load("paddleocr_recognition_export_census.json")
        paddle_parity = load("paddleocr_recognition_export_parity.json")
        payloads = (dav2_census, dav2_parity, paddle_census, paddle_parity)
        source_shas = {payload["source_git_sha"] for payload in payloads}
        self.assertEqual(
            source_shas,
            {"bc331fcfc9d3b69d5d57845453da6320e74fa6e9"},
        )
        torch_cpu_shas = {
            payload["runtime"]["loaded_files"]["torch_cpu.dll"]["sha256"]
            for payload in payloads
        }
        self.assertEqual(
            torch_cpu_shas,
            {
                "50790da4ed9f7f60bfa0a5419e28142cafc0de33d1b688de1a90750aeb2584c2"
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
                (404, 2, 8, 20, 53, 425, 1),
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
            "total_queue_submits": 52,
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
            "pending_command_flush": 48,
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
            "total_queue_submits": 46,
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
            "pending_command_flush": 42,
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

    def test_guard_variant_only_handles_export_guard_rejections(self):
        self.assertTrue(
            _is_export_guard_rejection(
                torch.vulkan.VulkanGraphExecutionError(
                    "Vulkan graph node '_guards_fn' failed: Guard failed: x"
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
