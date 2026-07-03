# Owner(s): ["module: vulkan"]

import os
import re
import sys
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


def _field(row, name):
    match = re.search(rf"\b{name}=([0-9]+)", row)
    if match is None:
        raise AssertionError(f"{name} missing from row: {row}")
    return int(match.group(1))


def _large_conv_aggregate_row():
    rows = torch.ops.vulkan_prepack.packed_weight_residency_snapshot()
    for row in rows:
        if (
            "packed_weight_query_aggregate" in row
            and "kind=Conv2dSlidingWindow" in row
            and "logical_weight_shape=[128,128,5,5]" in row
        ):
            return row
    raise AssertionError("\n".join(rows))


def _is_conservative_large_weight_cache_adapter():
    name = torch.vulkan.get_device_name(torch.vulkan.current_device())
    return "GTX" in name or "6700 XT" in name


@unittest.skipUnless(torch.is_vulkan_available(), "Vulkan not available")
class TestVulkanLargeSlidingWindowResidency(TestCase):
    def test_large_sliding_window_weight_residency_uses_identity_cache(self):
        torch.manual_seed(0)
        torch.ops.vulkan_prepack.reset_packed_weight_residency_snapshot()

        conv = torch.nn.Conv2d(128, 128, 5, padding=2).eval().to("vulkan")
        x_cpu = torch.randn(1, 128, 8, 8)
        x = x_cpu.to("vulkan")

        with torch.inference_mode():
            for _ in range(3):
                y = conv(x)
        torch.ops.vulkan_prepack.synchronize()

        expected = torch.nn.functional.conv2d(
            x_cpu,
            conv.weight.cpu(),
            conv.bias.cpu(),
            padding=2)
        self.assertEqual(y.cpu(), expected)

        row = _large_conv_aggregate_row()
        if _is_conservative_large_weight_cache_adapter():
            self.assertGreater(_field(row, "store_skip_large"), 0)
            self.assertEqual(_field(row, "stores"), 0)
            return

        self.assertEqual(_field(row, "lookups"), 3)
        self.assertEqual(_field(row, "misses"), 1)
        self.assertEqual(_field(row, "hits"), 2)
        self.assertEqual(_field(row, "stores"), 1)
        self.assertEqual(_field(row, "persistent_stores"), 1)
        self.assertEqual(_field(row, "store_skip_large"), 0)

        with torch.no_grad():
            conv.weight.add_(0.01)
        with torch.inference_mode():
            y = conv(x)
        torch.ops.vulkan_prepack.synchronize()
        expected = torch.nn.functional.conv2d(
            x_cpu,
            conv.weight.cpu(),
            conv.bias.cpu(),
            padding=2)
        self.assertEqual(y.cpu(), expected)

        row = _large_conv_aggregate_row()
        self.assertEqual(_field(row, "lookups"), 4)
        self.assertEqual(_field(row, "misses"), 2)
        self.assertEqual(_field(row, "hits"), 2)
        self.assertEqual(_field(row, "stores"), 2)

        other = torch.nn.Conv2d(128, 128, 5, padding=2).eval().to("vulkan")
        with torch.inference_mode():
            y = other(x)
        torch.ops.vulkan_prepack.synchronize()
        expected = torch.nn.functional.conv2d(
            x_cpu,
            other.weight.cpu(),
            other.bias.cpu(),
            padding=2)
        self.assertEqual(y.cpu(), expected)

        row = _large_conv_aggregate_row()
        self.assertEqual(_field(row, "lookups"), 5)
        self.assertEqual(_field(row, "misses"), 3)
        self.assertEqual(_field(row, "hits"), 2)
        self.assertEqual(_field(row, "stores"), 3)


if __name__ == "__main__":
    run_tests()
