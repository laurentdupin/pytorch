"""
PyTorch Linux AArch64 tests.
Corresponds to test_linux_aarch64 in test.sh.
"""

from __future__ import annotations

import functools

from cli.lib.pytorch.base import (
    CoreTestPlan,
    TestStep,
)
from cli.lib.pytorch.base import run_test


def _aarch64_core(shard_id: int, num_shards: int) -> None:
    # test.sh: test_linux_aarch64 — core tests
    run_test(
        "--include"
        " test_modules test_utils test_mkldnn test_mkldnn_fusion test_openmp test_torch test_dynamic_shapes"
        " test_transformers test_multiprocessing test_numpy_interop test_autograd test_binary_ufuncs test_complex test_spectral_ops"
        " test_foreach test_reductions test_unary_ufuncs test_tensor_creation_ops test_ops profiler/test_memory_profiler"
        " distributed/elastic/timer/api_test distributed/elastic/timer/local_timer_example distributed/elastic/timer/local_timer_test"
        " test_linalg"
        " test_jit test_jit_autocast test_ops_jit"
        " test_nn nn/test_convolution functorch/test_ops functorch/test_aotdispatch",
        f"--shard {shard_id} {num_shards}",
        "--verbose",
    )


def _aarch64_dynamo(shard_id: int, num_shards: int) -> None:
    # test.sh: test_linux_aarch64 — dynamo tests
    run_test(
        "--include"
        " dynamo/test_compile dynamo/test_backends dynamo/test_comptime dynamo/test_config"
        " dynamo/test_functions dynamo/test_fx_passes_pre_grad dynamo/test_interop dynamo/test_model_output dynamo/test_modules"
        " dynamo/test_optimizers dynamo/test_recompile_ux dynamo/test_recompiles",
        f"--shard {shard_id} {num_shards}",
        "--verbose",
    )


def _aarch64_inductor(shard_id: int, num_shards: int) -> None:
    # test.sh: test_linux_aarch64 — inductor tests
    run_test(
        "--include"
        " inductor/test_torchinductor inductor/test_benchmark_fusion inductor/test_codecache"
        " inductor/test_config inductor/test_control_flow inductor/test_coordinate_descent_tuner inductor/test_fx_fusion"
        " inductor/test_group_batch_fusion inductor/test_inductor_freezing inductor/test_inductor_utils"
        " inductor/test_inplacing_pass inductor/test_kernel_benchmark inductor/test_layout_optim"
        " inductor/test_max_autotune inductor/test_memory_planning inductor/test_metrics inductor/test_multi_kernel inductor/test_pad_mm"
        " inductor/test_pattern_matcher inductor/test_perf inductor/test_profiler inductor/test_select_algorithm inductor/test_smoke"
        " inductor/test_split_cat_fx_passes inductor/test_compile inductor/test_torchinductor"
        " inductor/test_torchinductor_codegen_dynamic_shapes inductor/test_torchinductor_dynamic_shapes inductor/test_memory"
        " inductor/test_triton_cpu_backend inductor/test_triton_extension_backend inductor/test_mkldnn_pattern_matcher inductor/test_cpu_cpp_wrapper"
        " inductor/test_cpu_select_algorithm inductor/test_cpu_repro"
        " inductor/test_aot_inductor inductor/test_fused_attention",
        f"--shard {shard_id} {num_shards}",
        "--verbose",
    )


# main python test step
def _aarch64_steps(build_env: str, shard_id: int, num_shards: int) -> list[TestStep]:
    return [
        TestStep(test_id="core",     fn=functools.partial(_aarch64_core,     shard_id, num_shards)),
        TestStep(test_id="dynamo",   fn=functools.partial(_aarch64_dynamo,   shard_id, num_shards)),
        TestStep(test_id="inductor", fn=functools.partial(_aarch64_inductor, shard_id, num_shards)),
    ]


AARCH64_PLANS: dict[str, CoreTestPlan] = {
    "pytorch_linux_aarch64": CoreTestPlan(
        group_id="pytorch_linux_aarch64",
        title="PyTorch Linux AArch64 Tests",
        # test.sh: elif [[ "${BUILD_ENVIRONMENT}" == *aarch64* && "${TEST_CONFIG}" == 'default' ]]
        run_on=["aarch64"],
        test_configs=["default"],
        get_steps_fn=_aarch64_steps,
    ),
}
