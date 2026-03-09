# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import unittest
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    FSDPMeshInfo,
    ShardPlacementResult,
)
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
from torch.distributed.tensor import init_device_mesh, Shard
from torch.distributed.tensor.experimental import implicit_replication
from torch.testing._internal.common_distributed import (
    skip_if_lt_x_gpu,
    skip_if_rocm_arch_multiprocess,
)
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    get_devtype,
    patch_all_gather,
    patch_reduce_scatter,
)
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    MI200_ARCH,
    run_tests,
    TEST_HPU,
)


device_type = torch.device(get_devtype())
device_module = torch.get_device_module(device_type)


def _time_fn(fn: Callable):
    start_event = device_module.Event(enable_timing=True)
    end_event = device_module.Event(enable_timing=True)
    dist.barrier()
    device_module.synchronize()
    start_event.record()
    fn()
    end_event.record()
    device_module.synchronize()
    return start_event.elapsed_time(end_event)


def _make_delayed_collectives(comm_sleep_ms: int):
    """Returns (delayed_all_gather, delayed_reduce_scatter) that add a sleep
    on a shared comm stream before the original collective."""
    orig_all_gather = dist.all_gather_into_tensor
    orig_reduce_scatter = dist.reduce_scatter_tensor
    comm_stream = device_module.Stream()

    def delay_collective():
        # Share a stream so that all-gather and reduce-scatter block each
        # other like in `ProcessGroupNCCL`
        comm_stream.wait_stream(device_module.current_stream())
        with device_module.stream(comm_stream):
            device_module._sleep(int(comm_sleep_ms * get_cycles_per_ms()))
        device_module.current_stream().wait_stream(comm_stream)

    def delayed_all_gather(*args, **kwargs):
        delay_collective()
        return orig_all_gather(*args, **kwargs)

    def delayed_reduce_scatter(*args, **kwargs):
        delay_collective()
        return orig_reduce_scatter(*args, **kwargs)

    return delayed_all_gather, delayed_reduce_scatter


class TestFullyShardOverlap(FSDPTest):
    """
    NOTE: Testing stream overlap in PyTorch CI is tricky.

    One approach is to use CUDA sleeps to emulate kernels in each stream;
    however, ``torch.cuda._sleep`` requires inputs in units of cycles. The
    ``get_cycles_per_ms`` function to convert from ms to cycles is computed
    once and cached thereafter, which means that if there is variation later,
    the cached value may not be accurate. This leads to flakiness in CI.

    To address this, we relax the tests as simple sanity checks that the
    overlapped times are less than a non-overlapped baseline, but we do not
    test that the overlapped time is less than a precisely calculated time.
    """

    @property
    def world_size(self) -> int:
        return min(2, torch.get_device_module(device_type).device_count())

    @skip_if_rocm_arch_multiprocess(MI200_ARCH)
    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_fully_shard_training_overlap(self):
        torch.manual_seed(42)

        # Use non-trivial comm. time but still shorter than compute time
        dim, num_linears, compute_sleep_ms, comm_sleep_ms = (4, 3, 25, 10)
        model = nn.Sequential(
            *[LinearWithSleep(dim, compute_sleep_ms) for _ in range(num_linears)]
        )
        ref_model = copy.deepcopy(model).to(device_type)
        for lin in model:
            if len(list(lin.parameters())) != 1:
                raise AssertionError("Expects only one weight")
            fully_shard(lin, reshard_after_forward=True)
        fully_shard(model, reshard_after_forward=True)

        delayed_all_gather, delayed_reduce_scatter = _make_delayed_collectives(
            comm_sleep_ms
        )

        inp = torch.randn((2, dim), device=device_type.type)
        loss = model(inp).sum()  # warmup CUDA and allocator
        loss.backward()

        def ref_fwd():
            with patch_all_gather(delayed_all_gather):
                # Run dummy all-gathers per weight (which is one FSDP group)
                for lin in ref_model:
                    dummy_ag_output = torch.empty_like(lin.weight)
                    dummy_ag_input = torch.chunk(dummy_ag_output, self.world_size)[
                        self.rank
                    ]
                    dist.all_gather_into_tensor(dummy_ag_output, dummy_ag_input)
                return ref_model(inp)

        def fwd():
            with patch_all_gather(delayed_all_gather):
                model(inp)

        ref_fwd_time = _time_fn(ref_fwd)
        fwd_time = _time_fn(fwd)
        # Forward: only 1st all-gather is exposed
        # NOTE: Do not enforce the expected forward time due to flakiness in CI
        # expected_fwd_time = comm_sleep_ms + num_linears * compute_sleep_ms + buffer_ms
        self.assertLessEqual(fwd_time, ref_fwd_time)

        def ref_fwd_bwd():
            with patch_all_gather(delayed_all_gather):
                # Run dummy all-gathers per weight (which is one FSDP group)
                for lin in ref_model:
                    dummy_ag_output = torch.empty_like(lin.weight)
                    dummy_ag_input = torch.chunk(dummy_ag_output, self.world_size)[
                        self.rank
                    ]
                    dist.all_gather_into_tensor(dummy_ag_output, dummy_ag_input)
                loss = ref_model(inp).sum()
                # Run dummy all-gathers per weight again since we are
                # resharding after forward
                for lin in ref_model:
                    dummy_ag_output = torch.empty_like(lin.weight)
                    dummy_ag_input = torch.chunk(dummy_ag_output, self.world_size)[
                        self.rank
                    ]
                    dist.all_gather_into_tensor(dummy_ag_output, dummy_ag_input)
                loss.backward()
                # Run dummy reduce-scatters per weight
                for lin in ref_model:
                    dummy_rs_input = torch.empty_like(lin.weight)
                    dummy_rs_output = torch.chunk(dummy_rs_input, self.world_size)[
                        self.rank
                    ]
                    dist.reduce_scatter_tensor(dummy_rs_output, dummy_rs_input)

        def fwd_bwd():
            with (
                patch_all_gather(delayed_all_gather),
                patch_reduce_scatter(delayed_reduce_scatter),
            ):
                loss = model(inp).sum()
                loss.backward()

        ref_fwd_bwd_time = _time_fn(ref_fwd_bwd)
        fwd_bwd_time = _time_fn(fwd_bwd)
        # Backward: only 1st all-gather and last reduce-scatter are exposed;
        # double the backward compute since computing two gradients per layer
        # NOTE: Do not enforce the expected forward-backward time due to
        # flakiness in CI
        # expected_bwd_time = (
        #     comm_sleep_ms * 2 + num_linears * 2 * compute_sleep_ms + buffer_ms * 2
        # )
        self.assertLessEqual(fwd_bwd_time, ref_fwd_bwd_time)

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_fully_shard_post_optim_event_overlap(self):
        torch.manual_seed(42)

        # Use non-trivial comm. time but still shorter than compute time
        dim, compute_sleep_ms, comm_sleep_ms = (4, 25, 10)
        # Define the model to have a high-compute linear followed by a
        # low-compute linear, where only the low-compute linear uses FSDP
        model = nn.Sequential(
            LinearWithSleep(dim, compute_sleep_ms), nn.Linear(dim, dim)
        ).to(device_type)
        fully_shard(model[1], reshard_after_forward=False)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        orig_all_gather_into_tensor = dist.all_gather_into_tensor

        def delayed_all_gather(*args, **kwargs):
            torch.get_device_module(device_type)._sleep(
                int(comm_sleep_ms * get_cycles_per_ms())
            )
            return orig_all_gather_into_tensor(*args, **kwargs)

        inp = torch.randn((2, dim), device=device_type)

        def run_train_steps(num_iters: int, use_post_optim_event: bool):
            for _ in range(num_iters):
                optim.zero_grad()
                with patch_all_gather(delayed_all_gather):
                    loss = model(inp).sum()
                loss.backward()
                with implicit_replication():
                    optim.step()
                if use_post_optim_event:
                    post_optim_event = (
                        torch.get_device_module(device_type)
                        .current_stream()
                        .record_event()
                    )
                    model[1].set_post_optim_event(post_optim_event)

        run_train_steps(1, False)  # warmup CUDA and allocator
        num_iters = 5
        baseline_time = _time_fn(functools.partial(run_train_steps, num_iters, False))
        test_time = _time_fn(functools.partial(run_train_steps, num_iters, True))

        buffer_ms = 4  # CPU delays and copies
        # Baseline: FSDP all-gather is exposed since the FSDP module waits for
        # the current stream and hence the high-compute linear
        self.assertLessEqual(
            baseline_time,
            num_iters * (3 * compute_sleep_ms + comm_sleep_ms + buffer_ms),
        )
        # Test: FSDP all-gather is overlapped with the high-compute linear
        # since the FSDP module only waits for the post-optim event (except on
        # the 1st iteration when no event has been recorded)
        expected_test_time = (
            num_iters * (3 * compute_sleep_ms + buffer_ms) + comm_sleep_ms
        )
        self.assertLessEqual(test_time, expected_test_time)
        # Since `get_cycles_per_ms` uses lru cache, there may be some variance
        # between the initially determined cycles vs. the current cycles per
        # ms, so we relax the baseline check to just that it is greater than
        # the test time rather than the expected test time
        self.assertGreater(baseline_time, test_time)


class Matmul(torch.autograd.Function):
    # Use CUDA sleeps to emulate compute time
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, sleep_ms: int):
        ctx.save_for_backward(input, weight)
        ctx.sleep_ms = sleep_ms
        torch.get_device_module(device_type)._sleep(int(sleep_ms * get_cycles_per_ms()))
        return input @ weight

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input, weight) = ctx.saved_tensors
        torch.get_device_module(device_type)._sleep(
            int(2 * ctx.sleep_ms * get_cycles_per_ms())
        )
        grad_input = grad_output @ weight.T
        grad_weight = input.T @ grad_output
        return grad_input, grad_weight, None


class LinearWithSleep(nn.Module):
    def __init__(self, dim: int, sleep_ms: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((dim, dim)))
        self.sleep_ms = sleep_ms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(Matmul.apply(x, self.weight, self.sleep_ms))


class DualParamLinearWithSleep(nn.Module):
    """Module with two weights for testing per-param mesh overlap.

    weight_a and weight_b can be routed to different FSDP param groups
    via shard_placement_fn.
    """

    def __init__(self, dim: int, sleep_ms: int):
        super().__init__()
        self.weight_a = nn.Parameter(torch.randn((dim, dim)))
        self.weight_b = nn.Parameter(torch.randn((dim, dim)))
        self.sleep_ms = sleep_ms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(
            Matmul.apply(x, self.weight_a + self.weight_b, self.sleep_ms)
        )


def _make_delayed_reduce_scatter(comm_sleep_ms: int):
    """Returns a delayed reduce_scatter that sleeps on the current stream."""
    orig_rs = dist.reduce_scatter_tensor

    def delayed_rs(*args, **kwargs):
        device_module._sleep(int(comm_sleep_ms * get_cycles_per_ms()))
        return orig_rs(*args, **kwargs)

    return delayed_rs


def _median(times: list[float]) -> float:
    times = sorted(times)
    return times[len(times) // 2]


@contextlib.contextmanager
def _separate_ag_pgs(
    model: nn.Module, world_size: int, ep_degree: int
) -> None:
    """Monkey-patch AG to use separate NCCL PGs from RS.

    Within ProcessGroupNCCL, all collectives on the same PG share one NCCL
    stream.  When we inject a CUDA sleep before the RS to simulate slow
    communication, that sleep also blocks the AG on the same PG.  Creating
    fresh PGs for AG gives them their own NCCL streams, so AG and RS can
    overlap just as they would with real (non-simulated) collectives.
    """
    ag_dp_pg = dist.new_group(list(range(world_size)))
    ag_efsdp_pg = None
    my_rank = dist.get_rank()
    for i in range(ep_degree):
        ranks = list(range(i, world_size, ep_degree))
        g = dist.new_group(ranks)
        if my_rank in ranks:
            ag_efsdp_pg = g

    pg_map: dict[dist.ProcessGroup, dist.ProcessGroup] = {}
    for mod in model:
        state = _get_module_fsdp_state(mod)
        if state is None:
            continue
        for param_group in state._fsdp_param_groups:
            orig_pg = param_group.mesh_info.shard_process_group
            if orig_pg not in pg_map:
                pg_map[orig_pg] = (
                    ag_dp_pg if orig_pg.size() == world_size else ag_efsdp_pg
                )

    orig_fget = FSDPParamGroup._all_gather_process_group.fget

    @property
    def _patched_ag_pg(self):
        orig_pg = orig_fget(self)
        return pg_map.get(orig_pg, orig_pg)

    FSDPParamGroup._all_gather_process_group = _patched_ag_pg
    try:
        yield
    finally:
        FSDPParamGroup._all_gather_process_group = property(orig_fget)


class TestFullyShardPerParamMeshOverlap(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(8, torch.get_device_module(device_type).device_count())

    @skip_if_rocm_arch_multiprocess(MI200_ARCH)
    @skip_if_lt_x_gpu(8)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_fully_shard_per_param_mesh_training_overlap(self):
        """Test per-PG reduce-scatter state for per-param mesh FSDP.

        With per-param mesh, each module has two param groups using different
        process groups.  Per-PG reduce-scatter state (HEAD) avoids intra-module
        cross-PG stalls on the compute stream, allowing the RS to overlap with
        the next module's backward compute.  Shared state (HEAD~1) forces each
        module's second param group to wait for the first's RS, adding
        ``comm_sleep`` per module to the critical path.

        We use ``reshard_after_forward=True`` and give AG and RS separate NCCL
        PGs via ``_separate_ag_pgs`` so the simulated RS delay doesn't also
        block the backward AG (which would happen if they shared one NCCL
        stream).
        """
        torch.manual_seed(42)

        dim, num_modules = 8, 5
        compute_sleep_ms, comm_sleep_ms = 100, 50
        ep_degree = 2

        efsdp_size = self.world_size // ep_degree
        dp_mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("dp",),
        )
        sparse_mesh = dp_mesh._unflatten(
            0,
            (efsdp_size, ep_degree),
            ("efsdp", "ep"),
        )
        dp_mesh_info = FSDPMeshInfo(mesh=dp_mesh, shard_mesh_dim=0)
        efsdp_mesh_info = FSDPMeshInfo(mesh=sparse_mesh["efsdp"], shard_mesh_dim=0)

        model = nn.Sequential(
            *[
                DualParamLinearWithSleep(dim, compute_sleep_ms)
                for _ in range(num_modules)
            ]
        )

        for mod in model:

            def _shard_placement_fn(param, _mod=mod):
                if param is _mod.weight_b:
                    return ShardPlacementResult(
                        placement=Shard(0), mesh_info=efsdp_mesh_info
                    )
                return ShardPlacementResult(placement=Shard(0), mesh_info=dp_mesh_info)

            fully_shard(
                mod,
                mesh=dp_mesh,
                reshard_after_forward=True,
                shard_placement_fn=_shard_placement_fn,
            )
        fully_shard(model, mesh=dp_mesh, reshard_after_forward=True)

        with _separate_ag_pgs(model, self.world_size, ep_degree):
            inp = torch.randn((2, dim), device=device_type.type)

            def fwd_bwd():
                model(inp).sum().backward()

            # Warmup
            for _ in range(3):
                fwd_bwd()
                model.zero_grad()

            # Measure no-delay baseline
            baseline_times = [_time_fn(fwd_bwd) for _ in range(3)]
            for _ in range(3):
                model.zero_grad()

            # Measure with delayed reduce-scatter
            delayed_rs = _make_delayed_reduce_scatter(comm_sleep_ms)
            with patch_reduce_scatter(delayed_rs):
                for _ in range(3):
                    fwd_bwd()
                    model.zero_grad()
                delayed_times = [_time_fn(fwd_bwd) for _ in range(3)]
                for _ in range(3):
                    model.zero_grad()

        baseline = _median(baseline_times)
        delayed = _median(delayed_times)
        overhead = delayed - baseline
        # 2 param groups * N modules * comm_sleep = total RS sleep injected
        total_rs = 2 * num_modules * comm_sleep_ms
        # With per-PG state the RS overlaps with compute and the overhead
        # is roughly 2*comm_sleep (the tail of the last module).  With
        # shared state the intra-module stall adds ~(N-1)*comm_sleep to the
        # critical path.  The 0.5 threshold sits between these two regimes.
        self.assertLess(
            overhead,
            0.5 * total_rs,
            f"RS overhead {overhead:.0f}ms >= 50% of total RS {total_rs}ms; "
            f"per-PG RS state may not be preventing cross-PG stalls",
        )


if __name__ == "__main__":
    run_tests()
