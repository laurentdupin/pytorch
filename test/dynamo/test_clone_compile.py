# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo
import torch._dynamo.testing
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCloneCompile(TestCase):
    def test_clone_no_cache_collision(self):
        """g1 exceeds recompile limit, g2 should not be affected."""
        recompile_limit = 2

        def fn(x):
            return x + 1

        cnt1 = torch._dynamo.testing.CompileCounter()
        cnt2 = torch._dynamo.testing.CompileCounter()

        g1 = torch.compile(
            fn,
            backend=cnt1,
            dynamic=False,
            clone=True,
            recompile_limit=recompile_limit,
        )
        g2 = torch.compile(
            fn,
            backend=cnt2,
            dynamic=True,
            clone=True,
            recompile_limit=recompile_limit,
        )

        # Exhaust g1's recompile limit by passing different shapes.
        # With dynamic=False, each new shape triggers a recompile until the
        # limit is hit, after which Dynamo falls back to eager.
        for i in range(1, recompile_limit + 2):
            g1(torch.randn(i))
        self.assertEqual(cnt1.frame_count, recompile_limit)

        # g2 should compile fine — its cache is independent of g1's.
        # With dynamic=True, it compiles a dynamic kernel that handles
        # varying shapes without recompilation.
        g2(torch.randn(10))
        self.assertEqual(cnt2.frame_count, 1)

        g2(torch.randn(20))
        g2(torch.randn(30))
        self.assertEqual(cnt2.frame_count, 1)


if __name__ == "__main__":
    run_tests()
