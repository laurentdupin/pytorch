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
    _graph_counts,
    _is_export_guard_rejection,
    _lowering_reports,
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


@dataclass(frozen=True)
class _FakeStaticLinearGeluRegionReport:
    candidate_count: int
    lowered_count: int
    rejected_count: int
    skipped_count: int
    plan_factory: str
    nodes: tuple[_FakeStaticLinearGeluNodeReport, ...]


def _fake_static_linear_gelu_report() -> _FakeStaticLinearGeluRegionReport:
    return _FakeStaticLinearGeluRegionReport(
        candidate_count=1,
        lowered_count=1,
        rejected_count=0,
        skipped_count=0,
        plan_factory="vulkan_prepack::create_graph_linear_gelu_plan",
        nodes=(
            _FakeStaticLinearGeluNodeReport(
                node_name="run_graph_linear_gelu_plan",
                status="lowered",
                reason="graph_owned_static_linear_tanh_gelu",
                linear_node_name="run_linear_context",
                context_attr="_vulkan_linear_context_a",
                plan_attr="_vulkan_static_linear_gelu_plan_a",
                program_name="StaticLinearGeluRegion",
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
        plan_factory="vulkan_prepack::create_graph_conv2d_relu_conv2d_plan",
        nodes=(
            _FakeStaticConv2dReluConv2dNodeReport(
                node_name="run_graph_conv2d_relu_conv2d_plan",
                status="lowered",
                reason="graph_owned_static_conv2d_relu_conv2d",
                first_conv2d_node_name="run_conv2d_context",
                relu_node_name="relu",
                second_conv2d_node_name="run_conv2d_context_1",
                first_context_attr="_vulkan_conv2d_context_a",
                second_context_attr="_vulkan_conv2d_context_b",
                plan_attr="_vulkan_static_conv2d_relu_conv2d_plan_a",
                program_name="StaticConv2dReluConv2dRegion",
                program_version="v3",
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
                "linear_lowering": {"created_context_count": 2},
                "static_linear_gelu_regions": None,
                "conv2d_lowering": None,
                "layernorm_lowering": None,
                "static_add_layernorm_regions": None,
                "static_conv2d_relu_conv2d_regions": None,
                "static_conv2d_relu_regions": None,
            },
        )
        self.assertEqual(
            _lowering_reports(conv2d_only),
            {
                "linear_lowering": None,
                "static_linear_gelu_regions": None,
                "conv2d_lowering": {"created_context_count": 3},
                "layernorm_lowering": None,
                "static_add_layernorm_regions": None,
                "static_conv2d_relu_conv2d_regions": None,
                "static_conv2d_relu_regions": None,
            },
        )

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
        )
        reports = _lowering_reports(program)
        static_report = reports["static_linear_gelu_regions"]
        layernorm_report = reports["layernorm_lowering"]
        static_add_layernorm_report = reports["static_add_layernorm_regions"]
        static_multi_conv_report = reports["static_conv2d_relu_conv2d_regions"]
        static_conv_report = reports["static_conv2d_relu_regions"]
        self.assertEqual(_graph_counts(program)["graph_owned_prepacked_contexts"], 2)
        self.assertEqual(layernorm_report, {"created_context_count": 0})
        self.assertEqual(static_report["candidate_count"], 1)
        self.assertEqual(static_report["lowered_count"], 1)
        self.assertEqual(static_report["rejected_count"], 0)
        self.assertEqual(static_report["skipped_count"], 0)
        self.assertEqual(
            static_report["plan_factory"],
            "vulkan_prepack::create_graph_linear_gelu_plan",
        )
        node = static_report["nodes"][0]
        self.assertEqual(node["status"], "lowered")
        self.assertEqual(node["reason"], "graph_owned_static_linear_tanh_gelu")
        self.assertEqual(node["linear_node_name"], "run_linear_context")
        self.assertEqual(node["context_attr"], "_vulkan_linear_context_a")
        self.assertEqual(
            node["plan_attr"], "_vulkan_static_linear_gelu_plan_a"
        )
        self.assertEqual(node["program_name"], "StaticLinearGeluRegion")
        self.assertEqual(node["program_version"], "v1")
        self.assertEqual(node["instruction_count"], 1)
        self.assertEqual(node["input_ssa"], 0)
        self.assertEqual(node["output_ssa"], 1)
        self.assertEqual(node["input_use_count"], 1)
        self.assertEqual(node["input_last_use"], 0)
        self.assertEqual(node["static_context_slot"], 0)
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
            "vulkan_prepack::create_graph_conv2d_relu_conv2d_plan",
        )
        multi_node = static_multi_conv_report["nodes"][0]
        self.assertEqual(multi_node["status"], "lowered")
        self.assertEqual(
            multi_node["reason"], "graph_owned_static_conv2d_relu_conv2d"
        )
        self.assertEqual(multi_node["program_name"], "StaticConv2dReluConv2dRegion")
        self.assertEqual(multi_node["program_version"], "v3")
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

    def test_checked_in_dav2_evidence_is_schema_valid_and_measured(self):
        evidence_dir = Path(__file__).parent / "vulkan_graph" / "evidence"
        for name, artifact_type in (
            ("dav2_vits_export_census.json", "export_census"),
            ("dav2_vits_export_parity.json", "parity"),
        ):
            payload = json.loads((evidence_dir / name).read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], EVIDENCE_SCHEMA)
            self.assertEqual(payload["artifact_type"], artifact_type)
            self.assertEqual(payload["status"], "measured")
            self.assertEqual(len(payload["cases"]), 2)
            self.assertIsInstance(payload["source_git_sha"], str)
            self.assertEqual(validate_evidence_payload(payload), [])

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
