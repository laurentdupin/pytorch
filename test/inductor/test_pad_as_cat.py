# Owner(s): ["module: inductor"]
"""Tests for cat multi-consumer optimization (pytorch#125075)."""

import re

import torch
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


def _count_triton_kernels(code: str) -> int:
    return len(re.findall(r"def triton_\w+\(", code))


class TestCatMultiConsumer(TestCase):

    @requires_gpu()
    def test_cat_to_fp16(self):
        """Multi-consumer cat avoids duplicate computation."""

        def fn(x):
            z = torch.cat([x, torch.zeros([6, 768], device=GPU_TYPE)], dim=0)
            y = x.to(torch.float16)
            return z, y

        x = torch.randn(1024, 768, device=GPU_TYPE)
        compiled = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled, x)
        ref = fn(x)

        self.assertEqual(result[0], ref[0])
        self.assertEqual(result[1], ref[1])
        kernel_count = _count_triton_kernels(code)
        self.assertLessEqual(
            kernel_count,
            2,
            f"Expected at most 2 kernels (fused + fill), got {kernel_count}.",
        )

    @requires_gpu()
    def test_single_consumer_cat_unchanged(self):
        """Single-consumer cat unchanged."""

        def fn(x):
            return torch.cat([x, torch.zeros([6, 768], device=GPU_TYPE)], dim=0)

        x = torch.randn(1024, 768, device=GPU_TYPE)
        compiled = torch.compile(fn)
        result = compiled(x)
        ref = fn(x)

        self.assertEqual(result, ref)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
