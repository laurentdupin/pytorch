# Owner(s): ["module: inductor"]
"""Tests for cat multi-consumer optimization (pytorch#125075)."""

import torch
from torch._inductor import metrics
from torch._inductor.test_case import TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


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
        metrics.reset()
        result = compiled(x)
        ref = fn(x)

        self.assertEqual(result[0], ref[0])
        self.assertEqual(result[1], ref[1])

        # Without the optimization x would be read twice (once by cat, once
        # by to_fp16). With the optimization ConcatKernel shares x so it is
        # read only once.
        x_bytes = x.nelement() * x.element_size()
        z_bytes = (1024 + 6) * 768 * x.element_size()
        y_bytes = 1024 * 768 * torch.float16.itemsize
        unoptimized_bytes = 2 * x_bytes + z_bytes + y_bytes
        self.assertLess(
            metrics.num_bytes_accessed,
            unoptimized_bytes,
            "Optimization should avoid reading x twice.",
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
