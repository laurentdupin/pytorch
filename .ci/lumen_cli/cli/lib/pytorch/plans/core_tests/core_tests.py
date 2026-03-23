"""
Core PyTorch tests — python test/run_test.py based.
Corresponds to jit_legacy / numpy_2 TEST_CONFIG in test.sh.
"""

from __future__ import annotations

import subprocess
import sys

from cli.lib.pytorch.base import (
    CoreTestPlan,
    TestStep,
)
from cli.lib.pytorch.base import run_test
from cli.lib.pytorch.plans.core_tests.pytorch_linux_aarch64 import AARCH64_PLANS


def _legacy_jit() -> None:
    run_test(
        "--include test_jit_legacy test_jit_fuser_legacy",
        "--verbose",
    )


def _setup_numpy_2() -> None:
    # test.sh: if [[ "${TEST_CONFIG}" == *numpy_2* ]]
    pandas_ver = subprocess.run(
        [sys.executable, "-c", "import pandas; print(pandas.__version__)"],
        capture_output=True, text=True,
    ).stdout.strip()
    pkgs = ["--pre", "numpy==2.0.2", "scipy==1.13.1", "numba==0.60.0"]
    if pandas_ver:
        pkgs += [f"pandas=={pandas_ver}", "--force-reinstall"]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs)


def _numpy_2() -> None:
    run_test(
        "--include"
        " dynamo/test_functions.py"
        " dynamo/test_unspec.py"
        " test_binary_ufuncs.py"
        " test_fake_tensor.py"
        " test_linalg.py"
        " test_numpy_interop.py"
        " test_tensor_creation_ops.py"
        " test_torch.py"
        " torch_np/test_basic.py",
    )


CORE_TEST_PLANS: dict[str, CoreTestPlan] = {
    **AARCH64_PLANS,
    "pytorch_jit_legacy": CoreTestPlan(
        group_id="pytorch_jit_legacy",
        title="PyTorch JIT Legacy Tests",
        # test.sh: elif [[ "$TEST_CONFIG" == 'jit_legacy' ]]; then test_python_legacy_jit
        test_configs=["jit_legacy"],
        steps=[
            TestStep(test_id="jit_legacy", fn=_legacy_jit),
        ],
    ),
    "pytorch_numpy_2": CoreTestPlan(
        group_id="pytorch_numpy_2",
        title="PyTorch NumPy 2 Tests",
        # test.sh: if [[ "${TEST_CONFIG}" == *numpy_2* ]]
        test_configs=["numpy_2"],
        setup_fn=_setup_numpy_2,
        steps=[
            TestStep(test_id="numpy_2", fn=_numpy_2),
        ],
    ),
}
