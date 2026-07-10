import json
from pathlib import Path

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.vulkan._graph_evidence import (
    EVIDENCE_SCHEMA,
    ExternalGraphEvidenceSetupError,
    parse_input_shape,
    require_external_assets,
    template_payload,
    validate_evidence_payload,
)


class TestVulkanGraphEvidence(TestCase):
    def test_checked_in_templates_are_schema_valid_and_unmeasured(self):
        evidence_dir = Path(__file__).parent / "vulkan_graph" / "evidence"
        for name, artifact_type in (
            ("dav2_vits_export_census.json", "export_census"),
            ("dav2_vits_export_parity.json", "parity"),
        ):
            payload = json.loads((evidence_dir / name).read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], EVIDENCE_SCHEMA)
            self.assertEqual(payload["artifact_type"], artifact_type)
            self.assertEqual(payload["status"], "template_not_measured")
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


if __name__ == "__main__":
    run_tests()
