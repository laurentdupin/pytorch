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
    _graph_counts,
    _is_export_guard_rejection,
    _lowering_reports,
)


@dataclass(frozen=True)
class _FakeLoweringReport:
    created_context_count: int


class TestVulkanGraphEvidence(TestCase):
    def test_graph_counts_include_all_graph_owned_lowering_families(self):
        census = SimpleNamespace(
            captured_node_count=100,
            call_function_node_count=80,
            lowered_vulkan_node_count=79,
            direct_vulkan_node_count=10,
            composite_node_count=5,
            unsupported_node_count=0,
            nodes=(),
        )
        program = SimpleNamespace(
            census=census,
            linear_lowering=_FakeLoweringReport(created_context_count=48),
            conv2d_lowering=_FakeLoweringReport(created_context_count=31),
        )
        counts = _graph_counts(program)
        self.assertEqual(counts["statically_lowered"], 79)
        self.assertEqual(counts["graph_owned_prepacked_contexts"], 79)

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
                "conv2d_lowering": None,
            },
        )
        self.assertEqual(
            _lowering_reports(conv2d_only),
            {
                "linear_lowering": None,
                "conv2d_lowering": {"created_context_count": 3},
            },
        )

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
