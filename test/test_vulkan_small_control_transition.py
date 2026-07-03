# Owner(s): ["module: vulkan"]

import json
import os
import sys
import tempfile
import unittest

TEST_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_FILE_DIR)
LOCAL_BUILD_BIN_DIR = os.path.join(REPO_ROOT, "build", "bin", "Release")
LOCAL_TORCH_LIB_DIR = os.path.join(REPO_ROOT, "torch", "lib")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if sys.platform == "win32":
    for dll_dir in (LOCAL_TORCH_LIB_DIR, LOCAL_BUILD_BIN_DIR):
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)
    existing_path = os.environ.get("PATH", "")
    path_entries = existing_path.split(os.pathsep) if existing_path else []
    local_dll_dirs = [
        path for path in (LOCAL_TORCH_LIB_DIR, LOCAL_BUILD_BIN_DIR)
        if os.path.isdir(path) and path not in path_entries
    ]
    if local_dll_dirs:
        os.environ["PATH"] = os.pathsep.join(local_dll_dirs + path_entries)

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipUnless(torch.is_vulkan_available(), "Vulkan not available")
class TestVulkanSmallControlTransition(TestCase):
    def test_bool_any_all_stays_fail_closed(self):
        cases = [
            torch.tensor([True, False, False, False], dtype=torch.bool),
            torch.tensor([False, True, False, False], dtype=torch.bool),
            torch.tensor([False, False, True, False], dtype=torch.bool),
            torch.tensor([False, False, False, True], dtype=torch.bool),
            torch.ones((1, 14), dtype=torch.bool),
            torch.zeros((1, 14), dtype=torch.bool),
            torch.zeros((1025,), dtype=torch.bool),
        ]

        for values in cases:
            torch.ops.vulkan_prepack.reset_fallback_counters()
            vulkan_values = values.to("vulkan")

            self.assertEqual(vulkan_values.any().cpu(), values.any())
            self.assertEqual(vulkan_values.all().cpu(), values.all())
            self.assertGreater(torch.ops.vulkan_prepack.cpu_fallback_count(), 0)
            self.assertEqual(torch.ops.vulkan_prepack.sync_readback_count(), 0)

    def test_argmax_last_dim_stays_fail_closed(self):
        cases = [
            (torch.arange(17, dtype=torch.float32).reshape(1, 17), 16),
            (torch.zeros((1, 14), dtype=torch.float32), 7),
        ]
        cases[1][0][0, 7] = 3.0

        for values, expected in cases:
            torch.ops.vulkan_prepack.reset_fallback_counters()
            result = values.to("vulkan").argmax(dim=-1)

            self.assertEqual(result.cpu(), torch.tensor([expected]))
            self.assertGreater(torch.ops.vulkan_prepack.cpu_fallback_count(), 0)
            self.assertEqual(torch.ops.vulkan_prepack.sync_readback_count(), 0)

    def test_transition_log_classifies_small_control_tensor_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "transition.jsonl")
            previous_log = os.environ.get("PYTORCH_VULKAN_TRANSITION_LOG")
            os.environ["PYTORCH_VULKAN_TRANSITION_LOG"] = log_path
            try:
                x = torch.tensor([1, 2, 3], dtype=torch.long).to("vulkan")
                mask = x == 2
                scalar = torch.tensor([1], dtype=torch.long).to("vulkan").item()
                large = torch.arange(17, dtype=torch.long).to("vulkan")
                large_mask = large == 2
                control_float = torch.randn(1, 14).to("vulkan")
                float_mask = control_float > 0

                torch.testing.assert_close(
                    mask.cpu(), torch.tensor([False, True, False])
                )
                torch.testing.assert_close(large_mask.cpu(), torch.arange(17) == 2)
                torch.testing.assert_close(float_mask.cpu(), control_float.cpu() > 0)
                self.assertEqual(scalar, 1)
            finally:
                if previous_log is None:
                    os.environ.pop("PYTORCH_VULKAN_TRANSITION_LOG", None)
                else:
                    os.environ["PYTORCH_VULKAN_TRANSITION_LOG"] = previous_log

            with open(log_path, encoding="utf-8") as log_file:
                records = [
                    json.loads(line)
                    for line in log_file
                    if line.strip()
                ]

            self.assertTrue(
                any(
                    record.get("producer_contract")
                    == "SmallControlTensorFallbackContract"
                    and record.get("consumer_contract")
                    == "PythonControlPlaneTensorConsumer"
                    and "control_tensor=1" in record.get("detail", "")
                    and (
                        "host_residency_contract="
                        "SmallControlHostResidencyContract.v0"
                    )
                    in record.get("detail", "")
                    and "host_result_reuploaded_to_vulkan=1"
                    in record.get("detail", "")
                    and "host_residency_authorized=0" in record.get("detail", "")
                    and (
                        "host_residency_top_blocker="
                        "consumer_chain_proof_missing"
                    )
                    in record.get("detail", "")
                    for record in records
                )
            )
            self.assertTrue(
                any(
                    record.get("producer_contract")
                    == "SmallControlScalarExtractionContract"
                    and record.get("consumer_contract")
                    == "PythonControlPlaneScalarConsumer"
                    and record.get("sync_required")
                    and (
                        "host_residency_contract="
                        "SmallControlHostResidencyContract.v0"
                    )
                    in record.get("detail", "")
                    and "host_scalar_boundary_preserved=1"
                    in record.get("detail", "")
                    and "host_residency_authorized=0" in record.get("detail", "")
                    and (
                        "host_residency_top_blocker="
                        "python_scalar_boundary_required"
                    )
                    in record.get("detail", "")
                    for record in records
                )
            )
            self.assertTrue(
                any(
                    record.get("producer_contract")
                    == "SmallControlTensorFallbackContract"
                    and record.get("producer_schema") == "aten::comparison"
                    and record.get("source_dtype") == "Float"
                    and record.get("source_sizes") == "[1,14]"
                    for record in records
                )
            )
            self.assertTrue(
                any(
                    record.get("producer_schema") == "aten::comparison"
                    and record.get("source_dtype") == "Long"
                    and record.get("source_sizes") == "[17]"
                    and record.get("producer_contract") == "unknown"
                    for record in records
                )
            )


if __name__ == "__main__":
    run_tests()
