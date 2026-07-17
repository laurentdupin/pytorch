import copy
import gc
import os
import operator
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.export._unlift import GuardsFn
import torch.utils._pytree as pytree
import torch.vulkan._graph as vulkan_graph
import torch.vulkan._graph_lowering as vulkan_graph_lowering
from scripts.benchmarks.vulkan_graph_export_evidence import _measure_case_latency
from torch.testing._internal.common_utils import run_tests, TestCase


# Vulkan aten::gelu("none") uses the tanh kernel; this covers its CPU gap.
VULKAN_GELU_NONE_CPU_RTOL = 5e-3
VULKAN_GELU_NONE_CPU_ATOL = 5e-4


def _linear_context_attrs(program):
    return {
        str(node.target)
        for node in program.graph_module.graph.nodes
        if node.op == "get_attr"
        and str(node.target).startswith("_vulkan_linear_context_")
    }


def _conv2d_context_attrs(program):
    return {
        str(node.target)
        for node in program.graph_module.graph.nodes
        if node.op == "get_attr"
        and str(node.target).startswith("_vulkan_conv2d_context_")
    }


def _layernorm_context_attrs(program):
    return {
        str(node.target)
        for node in program.graph_module.graph.nodes
        if node.op == "get_attr"
        and str(node.target).startswith("_vulkan_layernorm_context_")
    }


def _layernorm_context_attrs_from_module(graph_module):
    return {
        str(node.target)
        for node in graph_module.graph.nodes
        if node.op == "get_attr"
        and str(node.target).startswith("_vulkan_layernorm_context_")
    }


def _static_linear_gelu_plan_attrs(program):
    return {
        str(node.target)
        for node in program.graph_module.graph.nodes
        if node.op == "get_attr"
        and str(node.target).startswith("_vulkan_graph_region_plan_")
    }


def _static_add_layernorm_plan_attrs(program):
    return {
        str(node.target)
        for node in program.graph_module.graph.nodes
        if node.op == "get_attr"
        and str(node.target).startswith("_vulkan_static_add_layernorm_plan_")
    }


def _graph_scalar_error_plan(operator_name, left, right):
    return torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
        ["shape_scalar", "scale"],
        [operator_name, "aten::mul"],
        ["", "Scalar"],
        [[[-1], [-2]], [[0], [1]]],
        [[0, 0], [0, 0]],
        [[1], [2]],
        [left, right],
        1,
        [2],
    )


def _static_add_layernorm_plan_attrs_from_module(graph_module):
    return {
        str(node.target)
        for node in graph_module.graph.nodes
        if node.op == "get_attr"
        and str(node.target).startswith("_vulkan_static_add_layernorm_plan_")
    }


def _static_conv2d_relu_plan_attrs(program):
    return {
        str(node.target)
        for node in program.graph_module.graph.nodes
        if node.op == "get_attr"
        and str(node.target).startswith("_vulkan_static_conv2d_relu_plan_")
    }


def _static_conv2d_relu_plan_attrs_from_module(graph_module):
    return {
        str(node.target)
        for node in graph_module.graph.nodes
        if node.op == "get_attr"
        and str(node.target).startswith("_vulkan_static_conv2d_relu_plan_")
    }


def _static_conv2d_relu_conv2d_plan_attrs(program):
    return {
        str(node.target)
        for node in program.graph_module.graph.nodes
        if node.op == "get_attr"
        and str(node.target).startswith("_vulkan_graph_region_plan_")
    }


def _static_conv2d_relu_conv2d_plan_attrs_from_module(graph_module):
    return {
        str(node.target)
        for node in graph_module.graph.nodes
        if node.op == "get_attr"
        and str(node.target).startswith("_vulkan_graph_region_plan_")
    }


def _graph_program_invocation_counters():
    return list(torch.ops.vulkan_prepack.graph_program_invocation_counters())


def _raise_graph_node_error(tensor):
    del tensor
    raise RuntimeError("expected graph node failure")


@unittest.skipUnless(torch.vulkan.is_available(), "Vulkan is not available")
class TestVulkanGraph(TestCase):
    def test_graph_planning_context_is_explicit_and_program_keyed(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 6)

            def forward(self, tensor):
                return self.linear(tensor)

        torch.manual_seed(9)
        model = Linear().eval()
        tensor = torch.randn(2, 3, 4)
        planning_context = torch.vulkan.VulkanGraphPlanningContext(
            model_domain="vision",
            execution_phase="backbone",
            prefer_packed_layout_propagation=True,
            fixed_shape_graph_input_sizes=(2, 3, 4),
        )
        program = torch.vulkan.export_and_lower(
            model,
            tensor,
            planning_context=planning_context,
        )

        self.assertEqual(program.planning_context, planning_context)
        self.assertEqual(program.key.planning_context, planning_context)
        self.assertEqual(
            program.cpp_plan_report.planning_model_domain,
            "vision",
        )
        self.assertEqual(
            program.cpp_plan_report.planning_execution_phase,
            "backbone",
        )
        self.assertTrue(
            program.cpp_plan_report.planning_prefer_packed_layout_propagation
        )
        self.assertEqual(
            program.cpp_plan_report.planning_fixed_shape_graph_input_sizes,
            (2, 3, 4),
        )
        self.assertEqual(program.cpp_plan.planning_model_domain(), 1)
        self.assertEqual(program.cpp_plan.planning_execution_phase(), 3)
        self.assertTrue(
            program.cpp_plan.planning_prefer_packed_layout_propagation()
        )
        self.assertEqual(
            program.cpp_plan.planning_fixed_shape_graph_input_sizes(),
            [2, 3, 4],
        )
        self.assertEqual(
            program(tensor).cpu(),
            model(tensor),
            rtol=1e-4,
            atol=1e-4,
        )
        with self.assertRaisesRegex(
            torch.vulkan.VulkanGraphExecutionError,
            "fixed graph input shape mismatch",
        ):
            program(torch.randn(1, 3, 4))
        with self.assertRaisesRegex(ValueError, "fixed graph input shape mismatch"):
            torch.vulkan.export_and_lower(
                model,
                tensor,
                planning_context=torch.vulkan.VulkanGraphPlanningContext(
                    fixed_shape_graph_input_sizes=(1, 3, 4),
                ),
            )

        generic_program = torch.vulkan.export_and_lower(model, tensor)
        self.assertNotEqual(program.key.graph_hash, generic_program.key.graph_hash)

    def test_explicit_generic_graph_context_suppresses_label_inference(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(64, 64)

            def forward(self, tensor):
                return self.linear(tensor)

        model = Linear().eval()
        tensor = torch.randn(2, 64)
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_log_path = os.path.join(temp_dir, "runtime_policy.log")
            with patch.dict(
                os.environ,
                {"PYTORCH_VULKAN_RUNTIME_POLICY_LOG": policy_log_path},
            ):
                previous_label = torch.ops.vulkan_prepack.swap_runtime_label("llama")
                try:
                    program = torch.vulkan.export_and_lower(model, tensor)
                    self.assertEqual(
                        program(tensor).cpu(),
                        model(tensor),
                        rtol=1e-4,
                        atol=1e-4,
                    )
                finally:
                    torch.ops.vulkan_prepack.swap_runtime_label(previous_label)

            with open(policy_log_path, encoding="utf-8") as policy_log:
                policy_rows = [
                    row
                    for row in policy_log.read().splitlines()
                    if row.startswith("runtime_policy ")
                ]
        self.assertTrue(policy_rows)
        for row in policy_rows:
            self.assertIn("model_domain=Generic", row)
            self.assertIn("execution_phase=None", row)
            self.assertIn("inferred_from_label=0", row)

    def test_graph_route_lane_uses_explicit_planning_context(self):
        class Conv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(4, 6, 3, padding=1)

            def forward(self, tensor):
                return self.conv(tensor)

        torch.manual_seed(10)
        model = Conv().eval()
        tensor = torch.randn(1, 4, 8, 8)
        planning_context = torch.vulkan.VulkanGraphPlanningContext(
            model_domain="vision",
            execution_phase="decoder",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            route_log_path = os.path.join(temp_dir, "route.log")
            with patch.dict(
                os.environ,
                {"PYTORCH_VULKAN_ROUTE_LOG": route_log_path},
            ):
                program = torch.vulkan.export_and_lower(
                    model,
                    tensor,
                    planning_context=planning_context,
                )
                actual = program(tensor).cpu()

            with open(route_log_path, encoding="utf-8") as route_log:
                convolution_rows = [
                    row
                    for row in route_log.read().splitlines()
                    if " op=aten::convolution " in row
                ]

        self.assertEqual(actual, model(tensor), rtol=1e-4, atol=1e-4)
        self.assertTrue(convolution_rows)
        for row in convolution_rows:
            self.assertIn("lane=AdjacentDepthVision", row)

    def test_graph_planning_context_rejects_incompatible_semantics(self):
        with self.assertRaisesRegex(ValueError, "incompatible with model_domain"):
            torch.vulkan.VulkanGraphPlanningContext(
                model_domain="llm",
                execution_phase="backbone",
            )
        with self.assertRaisesRegex(
            ValueError,
            "non-empty tuple of positive integers",
        ):
            torch.vulkan.VulkanGraphPlanningContext(
                fixed_shape_graph_input_sizes=(2, 0, 4),
            )

    def test_static_linear_tanh_gelu_region_matches_cpu_and_transfers_context(self):
        class LinearGelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 6)

            def forward(self, tensor):
                return torch.nn.functional.gelu(
                    self.linear(tensor), approximate="tanh"
                )

        torch.manual_seed(10)
        model = LinearGelu().eval()
        tensor = torch.randn(2, 3, 4)
        program = torch.vulkan.export_and_lower(model, tensor)
        self.assertTrue(program.cpp_plan_report.submission_owned)
        self.assertTrue(program.cpp_plan.submission_owned())
        eager_vulkan = copy.deepcopy(model).to("vulkan").eval()
        with torch.inference_mode():
            eager_vulkan_output = eager_vulkan(tensor.to("vulkan")).cpu()

        torch.testing.assert_close(
            program(tensor).cpu(), model(tensor), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            program(tensor).cpu(), eager_vulkan_output, rtol=1e-4, atol=1e-4
        )
        report = program.static_linear_gelu_regions
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.lowered_count, 1)
        self.assertEqual(report.rejected_count, 0)
        self.assertEqual(report.skipped_count, 0)
        self.assertEqual(
            report.plan_factory,
            "vulkan_prepack::create_vulkan_graph_region_plan_linear_gelu",
        )
        node = report.nodes[0]
        self.assertEqual(node.program_name, "VulkanGraphRegionPlan")
        self.assertEqual(node.program_version, "v1")
        self.assertEqual(node.instruction_count, 2)
        self.assertEqual(node.input_ssa, 0)
        self.assertEqual(node.output_ssa, 2)
        self.assertEqual(node.input_use_count, 1)
        self.assertEqual(node.input_last_use, 0)
        self.assertEqual(node.static_context_slot, 0)
        self.assertTrue(node.direct_transition_only)
        self.assertTrue(node.replay_state_empty)
        self.assertEqual(len(_static_linear_gelu_plan_attrs(program)), 1)
        self.assertFalse(_linear_context_attrs(program))
        self.assertFalse(program.graph_module.state_dict())
        self.assertEqual(program.vulkan_graph_regions.plan_class, "VulkanGraphRegionPlan")
        self.assertEqual(program.vulkan_graph_regions.plan_version, "v1")
        self.assertEqual(
            program.vulkan_graph_regions.families[0].family,
            "linear_gelu_tanh",
        )
        lowered = [
            node
            for node in program.census.nodes
            if node.reason == "graph_owned_vulkan_graph_region_plan"
        ]
        self.assertEqual(len(lowered), 1)
        self.assertEqual(lowered[0].classification, "lowered_vulkan")
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_static_linear_tanh_gelu_region_reuses_dynamic_shapes(self):
        class LinearGelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 7)

            def forward(self, tensor):
                return torch.nn.functional.gelu(
                    self.linear(tensor), approximate="tanh"
                )

        torch.manual_seed(11)
        model = LinearGelu().eval()
        example = torch.randn(2, 3, 5)
        batch = torch.export.Dim("batch", min=1, max=4)
        tokens = torch.export.Dim("tokens", min=1, max=8)
        program = torch.vulkan.export_and_lower(
            model,
            example,
            dynamic_shapes=({0: batch, 1: tokens},),
        )
        self.assertEqual(program.execution_mode, "cpp_plan")
        self.assertEqual(program.cpp_plan_report.status, "compiled")
        self.assertEqual(program.cpp_plan_report.plan_version, "v9")
        self.assertEqual(
            program.cpp_plan_report.list_projection_instruction_count,
            1,
        )
        plan_attrs = _static_linear_gelu_plan_attrs(program)
        self.assertEqual(len(plan_attrs), 1)
        self.assertFalse(_linear_context_attrs(program))
        self.assertFalse(program.graph_module.state_dict())
        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            for shape in ((1, 1, 5), (4, 8, 5)):
                tensor = torch.randn(shape)
                torch.testing.assert_close(
                    program(tensor).cpu(),
                    model(tensor),
                    rtol=1e-4,
                    atol=1e-4,
                )
        self.assertEqual(_static_linear_gelu_plan_attrs(program), plan_attrs)
        self.assertEqual(program.static_linear_gelu_regions.lowered_count, 1)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_cpp_graph_plan_list_projection_checks_runtime_index(self):
        class LinearGelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 7)

            def forward(self, tensor):
                return torch.nn.functional.gelu(
                    self.linear(tensor), approximate="tanh"
                )

        model = LinearGelu().eval()
        tensor = torch.randn(2, 3, 5)
        program = torch.vulkan.export_and_lower(model, tensor)
        plan_attr = next(iter(_static_linear_gelu_plan_attrs(program)))
        region_plan = getattr(program.graph_module, plan_attr)

        def list_projection_plan(index):
            return torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
                ["region", "getitem"],
                [
                    "vulkan_prepack::run_vulkan_graph_region_plan",
                    "vulkan_graph::list_getitem",
                ],
                ["", ""],
                [[[0], [-1]], [[1], [-2]]],
                [[1, 0], [0, 0]],
                [[1], [2]],
                [region_plan, index],
                1,
                [2],
            )

        vulkan_tensor = tensor.to("vulkan")
        negative_output = torch.ops.vulkan_prepack.run_vulkan_graph_plan.default(
            [vulkan_tensor],
            list_projection_plan(-1),
        )
        torch.testing.assert_close(
            negative_output[0].cpu(),
            model(tensor),
            rtol=1e-4,
            atol=1e-4,
        )
        failing_plan = list_projection_plan(1)
        for _ in range(2):
            with self.assertRaisesRegex(
                RuntimeError,
                "VulkanGraphPlan.v9 node 'getitem'.*"
                "index 1 is out of range for length 1",
            ):
                torch.ops.vulkan_prepack.run_vulkan_graph_plan.default(
                    [vulkan_tensor],
                    failing_plan,
                )

    def test_static_linear_biasless_tanh_gelu_region_preserves_tanh_semantics(self):
        class LinearGelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 6, bias=False)

            def forward(self, tensor):
                return torch.nn.functional.gelu(
                    self.linear(tensor), approximate="tanh"
                )

        torch.manual_seed(15)
        model = LinearGelu().eval()
        tensor = torch.randn(2, 3, 4)
        program = torch.vulkan.export_and_lower(model, tensor)
        eager_vulkan = copy.deepcopy(model).to("vulkan").eval()
        with torch.inference_mode():
            eager_vulkan_output = eager_vulkan(tensor.to("vulkan")).cpu()

        graph_output = program(tensor).cpu()
        self.assertEqual(program.static_linear_gelu_regions.lowered_count, 1)
        self.assertEqual(len(_static_linear_gelu_plan_attrs(program)), 1)
        torch.testing.assert_close(
            graph_output, model(tensor), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            graph_output, eager_vulkan_output, rtol=1e-4, atol=1e-4
        )
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_static_linear_default_gelu_region_matches_cpu_and_eager_vulkan(self):
        class LinearGelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, tensor):
                return torch.nn.functional.gelu(self.linear(tensor))

        torch.manual_seed(12)
        model = LinearGelu().eval()
        with torch.no_grad():
            model.linear.weight.copy_(torch.eye(4))
            model.linear.bias.zero_()
        tensor = torch.tensor([[-3.0, -1.0, 1.0, 3.0]])
        cpu_output = model(tensor)
        program = torch.vulkan.export_and_lower(model, tensor)
        eager_vulkan = copy.deepcopy(model).to("vulkan").eval()
        with torch.inference_mode():
            eager_vulkan_output = eager_vulkan(tensor.to("vulkan")).cpu()
        op_hit_log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "vulkan_graph_linear_gelu_none_op_hit_test.log",
        )
        previous_op_hit_log = os.environ.get("PYTORCH_VULKAN_OP_HIT_LOG")
        if os.path.exists(op_hit_log_path):
            os.remove(op_hit_log_path)
        os.environ["PYTORCH_VULKAN_OP_HIT_LOG"] = op_hit_log_path
        try:
            graph_output = program(tensor).cpu()
            with open(op_hit_log_path, encoding="utf-8") as op_hit_file:
                op_hit_text = op_hit_file.read()
        finally:
            if previous_op_hit_log is None:
                os.environ.pop("PYTORCH_VULKAN_OP_HIT_LOG", None)
            else:
                os.environ["PYTORCH_VULKAN_OP_HIT_LOG"] = previous_op_hit_log
            if os.path.exists(op_hit_log_path):
                os.remove(op_hit_log_path)
        torch.testing.assert_close(
            graph_output,
            cpu_output,
            rtol=VULKAN_GELU_NONE_CPU_RTOL,
            atol=VULKAN_GELU_NONE_CPU_ATOL,
        )
        torch.testing.assert_close(
            graph_output, eager_vulkan_output, rtol=1e-5, atol=1e-5
        )
        self.assertIn("op=aten::linear.", op_hit_text)
        self.assertIn("op=aten::gelu.buffer_float", op_hit_text)
        self.assertIn("post=none", op_hit_text)
        self.assertNotIn("post=gelu", op_hit_text)
        self.assertNotIn("vulkan_prepack::run_linear_gelu_context", op_hit_text)
        report = program.static_linear_gelu_regions
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.lowered_count, 1)
        self.assertEqual(
            report.plan_factory,
            "vulkan_prepack::create_vulkan_graph_region_plan_linear_gelu_none",
        )
        self.assertEqual(len(_static_linear_gelu_plan_attrs(program)), 1)
        self.assertFalse(_linear_context_attrs(program))
        self.assertFalse(program.graph_module.state_dict())
        self.assertEqual(report.nodes[0].region_family, "linear_gelu_none")
        self.assertEqual(
            report.nodes[0].reason,
            "graph_owned_static_linear_none_gelu",
        )
        families = {
            family.family: family for family in program.vulkan_graph_regions.families
        }
        self.assertEqual(families["linear_gelu_none"].lowered_count, 1)
        self.assertEqual(families["linear_gelu_tanh"].lowered_count, 0)
        self.assertEqual(
            families["linear_gelu_none"].plan_factory,
            "vulkan_prepack::create_vulkan_graph_region_plan_linear_gelu_none",
        )
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_static_linear_none_gelu_region_reuses_dynamic_shapes(self):
        class LinearGelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 7)

            def forward(self, tensor):
                return torch.nn.functional.gelu(self.linear(tensor))

        torch.manual_seed(16)
        model = LinearGelu().eval()
        example = torch.randn(2, 3, 5)
        batch = torch.export.Dim("batch", min=1, max=4)
        tokens = torch.export.Dim("tokens", min=1, max=8)
        program = torch.vulkan.export_and_lower(
            model,
            example,
            dynamic_shapes=({0: batch, 1: tokens},),
        )
        plan_attrs = _static_linear_gelu_plan_attrs(program)
        self.assertEqual(len(plan_attrs), 1)
        self.assertFalse(_linear_context_attrs(program))
        self.assertFalse(program.graph_module.state_dict())
        for shape in ((1, 1, 5), (4, 8, 5)):
            tensor = torch.randn(shape)
            torch.testing.assert_close(
                program(tensor).cpu(),
                model(tensor),
                rtol=VULKAN_GELU_NONE_CPU_RTOL,
                atol=VULKAN_GELU_NONE_CPU_ATOL,
            )
        self.assertEqual(_static_linear_gelu_plan_attrs(program), plan_attrs)
        self.assertEqual(program.static_linear_gelu_regions.lowered_count, 1)
        self.assertEqual(
            program.static_linear_gelu_regions.nodes[0].region_family,
            "linear_gelu_none",
        )
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_static_linear_biasless_none_gelu_region_matches_cpu_and_eager_vulkan(
        self,
    ):
        class LinearGelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 6, bias=False)

            def forward(self, tensor):
                return torch.nn.functional.gelu(self.linear(tensor))

        torch.manual_seed(17)
        model = LinearGelu().eval()
        tensor = torch.randn(2, 3, 4)
        program = torch.vulkan.export_and_lower(model, tensor)
        eager_vulkan = copy.deepcopy(model).to("vulkan").eval()
        with torch.inference_mode():
            eager_vulkan_output = eager_vulkan(tensor.to("vulkan")).cpu()
        graph_output = program(tensor).cpu()
        torch.testing.assert_close(
            graph_output,
            model(tensor),
            rtol=VULKAN_GELU_NONE_CPU_RTOL,
            atol=VULKAN_GELU_NONE_CPU_ATOL,
        )
        torch.testing.assert_close(
            graph_output, eager_vulkan_output, rtol=1e-4, atol=1e-4
        )
        self.assertEqual(program.static_linear_gelu_regions.lowered_count, 1)
        self.assertEqual(
            program.static_linear_gelu_regions.nodes[0].region_family,
            "linear_gelu_none",
        )
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_static_linear_gelu_multi_use_linear_output_stays_unfused(self):
        class LinearGeluResidual(torch.nn.Module):
            def __init__(self, approximate):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)
                self.approximate = approximate

            def forward(self, tensor):
                linear = self.linear(tensor)
                return torch.nn.functional.gelu(
                    linear, approximate=self.approximate
                ) + linear

        for approximate in ("tanh", "none"):
            with self.subTest(approximate=approximate):
                torch.manual_seed(13)
                model = LinearGeluResidual(approximate).eval()
                tensor = torch.randn(2, 3)
                program = torch.vulkan.export_and_lower(model, tensor)
                torch.testing.assert_close(
                    program(tensor).cpu(), model(tensor), rtol=1e-4, atol=1e-4
                )
                report = program.static_linear_gelu_regions
                self.assertEqual(report.lowered_count, 0)
                self.assertEqual(report.skipped_count, 1)
                self.assertEqual(
                    report.nodes[0].reason, "linear_output_has_multiple_users"
                )
                self.assertEqual(
                    report.nodes[0].region_family,
                    f"linear_gelu_{approximate}",
                )
                self.assertFalse(_static_linear_gelu_plan_attrs(program))
                self.assertEqual(len(_linear_context_attrs(program)), 1)

    def test_cpp_graph_plan_executes_tensor_ssa_without_python_nodes(self):
        class LinearGeluResidual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)

            def forward(self, tensor):
                linear = self.linear(tensor)
                return torch.nn.functional.gelu(
                    linear, approximate="tanh"
                ) + linear

        torch.manual_seed(18)
        model = LinearGeluResidual().eval()
        first_input = torch.randn(2, 3)
        second_input = torch.randn(2, 3)
        program = torch.vulkan.export_and_lower(model, first_input)

        self.assertEqual(program.execution_mode, "cpp_plan")
        self.assertIsInstance(program.cpp_plan, torch.ScriptObject)
        report = program.cpp_plan_report
        self.assertEqual(report.status, "compiled")
        self.assertEqual(report.reason, "immutable_ivalue_ssa_resource_plan")
        self.assertEqual(report.plan_class, "VulkanGraphPlan")
        self.assertEqual(report.plan_version, "v9")
        self.assertEqual(report.input_count, 1)
        self.assertEqual(report.instruction_count, 3)
        self.assertEqual(report.effect_instruction_count, 0)
        self.assertEqual(report.graph_scalar_instruction_count, 0)
        self.assertEqual(report.list_argument_count, 0)
        self.assertEqual(report.invocation_value_slot_count, 4)
        self.assertEqual(report.invocation_list_slot_count, 0)
        self.assertGreaterEqual(report.invocation_stack_capacity, 2)
        self.assertEqual(report.value_count, 4)
        self.assertEqual(report.output_count, 1)
        self.assertTrue(report.submission_owned)
        self.assertEqual(report.value_use_counts, (1, 2, 1, 0))
        self.assertEqual(report.value_last_uses, (0, 2, 2, 2))
        self.assertEqual(program.cpp_plan.input_count(), 1)
        self.assertEqual(program.cpp_plan.instruction_count(), 3)
        self.assertEqual(program.cpp_plan.effect_instruction_count(), 0)
        self.assertEqual(program.cpp_plan.graph_scalar_instruction_count(), 0)
        self.assertEqual(program.cpp_plan.list_argument_count(), 0)
        self.assertEqual(
            program.cpp_plan.invocation_value_slot_count(),
            report.value_count,
        )
        self.assertEqual(program.cpp_plan.invocation_list_slot_count(), 0)
        self.assertGreaterEqual(program.cpp_plan.invocation_stack_capacity(), 2)
        self.assertEqual(program.cpp_plan.value_count(), 4)
        self.assertEqual(program.cpp_plan.output_count(), 1)
        self.assertTrue(program.cpp_plan.submission_owned())
        self.assertEqual(program.cpp_plan.invocation_generation(), 0)
        self.assertEqual(program.cpp_plan.last_submission_value(), 0)
        self.assertTrue(program.cpp_plan.last_submission_complete())
        self.assertEqual(
            tuple(program.cpp_plan.value_use_counts()),
            report.value_use_counts,
        )
        self.assertEqual(
            tuple(program.cpp_plan.value_last_uses()),
            report.value_last_uses,
        )

        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            first_output = program(first_input)
            second_output = program(second_input)

        torch.testing.assert_close(
            first_output.cpu(), model(first_input), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            second_output.cpu(), model(second_input), rtol=1e-4, atol=1e-4
        )
        self.assertEqual(program.run_count, 2)
        self.assertEqual(program.last_executed_nodes, report.node_names)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)
        self.assertEqual(program.cpp_plan.invocation_generation(), 2)
        self.assertGreater(program.cpp_plan.last_submission_value(), 0)
        self.assertTrue(program.cpp_plan.last_submission_complete())
        self.assertEqual(
            _graph_program_invocation_counters(),
            [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )

    def test_cpp_graph_plan_preserves_outputs_before_last_instruction(self):
        class SinCos(torch.nn.Module):
            def forward(self, tensor):
                first = torch.sin(tensor)
                return first, torch.cos(first)

        model = SinCos().eval()
        tensor = torch.randn(2, 3)
        program = torch.vulkan.export_and_lower(model, tensor)

        self.assertEqual(program.execution_mode, "cpp_plan")
        self.assertEqual(program.cpp_plan_report.status, "compiled")
        self.assertEqual(program.cpp_plan_report.plan_version, "v9")
        self.assertEqual(program.cpp_plan_report.instruction_count, 2)
        self.assertEqual(program.cpp_plan_report.output_count, 2)
        self.assertEqual(program.cpp_plan.output_count(), 2)
        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            actual = tuple(value.cpu() for value in program(tensor))
        for value, reference in zip(actual, model(tensor)):
            torch.testing.assert_close(value, reference)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_cpp_graph_plan_owns_empty_submission_boundary(self):
        model = torch.nn.Flatten().eval()
        tensor = torch.randn(2, 3)
        program = torch.vulkan.export_and_lower(model, tensor)

        self.assertEqual(program.execution_mode, "cpp_plan")
        self.assertTrue(program.cpp_plan_report.submission_owned)
        self.assertEqual(program.cpp_plan_report.instruction_count, 1)
        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        output = program(tensor)

        self.assertEqual(program.cpp_plan.invocation_generation(), 1)
        self.assertEqual(program.cpp_plan.last_submission_value(), 0)
        self.assertTrue(program.cpp_plan.last_submission_complete())
        self.assertEqual(
            _graph_program_invocation_counters(),
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        self.assertEqual(output.cpu(), model(tensor))

    def test_cpp_graph_plan_owns_pool_and_softmax_submission(self):
        class PoolSoftmax(torch.nn.Module):
            def forward(self, tensor):
                pooled = torch.nn.functional.max_pool2d(tensor, 2)
                return torch.softmax(pooled, dim=-1)

        model = PoolSoftmax().eval()
        tensor = torch.randn(1, 4, 16, 16)
        program = torch.vulkan.export_and_lower(model, tensor)

        self.assertEqual(program.execution_mode, "cpp_plan")
        self.assertTrue(program.cpp_plan_report.submission_owned)
        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        output = program(tensor)

        self.assertEqual(program.cpp_plan.invocation_generation(), 1)
        self.assertGreater(program.cpp_plan.last_submission_value(), 0)
        self.assertEqual(
            _graph_program_invocation_counters(),
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        self.assertEqual(output.cpu(), model(tensor))
        self.assertTrue(program.cpp_plan.last_submission_complete())

    def test_cpp_graph_plan_batches_frequency_checkpoints(self):
        class RepeatedRelu(torch.nn.Module):
            def forward(self, tensor):
                value = tensor
                for _ in range(130):
                    value = torch.relu(value)
                return value

        model = RepeatedRelu().eval()
        tensor = torch.randn(1, 4, 4, 4)
        program = torch.vulkan.export_and_lower(model, tensor)
        device_tensor = tensor.to("vulkan")
        torch.ops.vulkan_prepack.synchronize()

        self.assertEqual(program.execution_mode, "cpp_plan")
        self.assertTrue(program.cpp_plan_report.submission_owned)
        self.assertEqual(program.cpp_plan_report.instruction_count, 130)
        self.assertEqual(
            program.cpp_plan.dead_input_reuse_instruction_count(), 130
        )
        self.assertEqual(program.cpp_plan.dead_input_reuse_count(), 0)
        torch.ops.vulkan_prepack.reset_submit_origin_counters()
        output = program(device_tensor)
        submit_origins = list(torch.ops.vulkan_prepack.submit_origin_counters())

        self.assertEqual(submit_origins[15], 5)
        self.assertEqual(submit_origins[0], 5)
        self.assertEqual(program.cpp_plan.dead_input_reuse_count(), 129)
        self.assertEqual(device_tensor.cpu(), tensor)
        self.assertEqual(output.cpu(), model(tensor))

        torch.ops.vulkan_prepack.reset_submit_origin_counters()
        eager_output = device_tensor
        for _ in range(17):
            eager_output = torch.relu(eager_output)
        eager_submit_origins = list(
            torch.ops.vulkan_prepack.submit_origin_counters()
        )

        self.assertEqual(eager_submit_origins[1], 1)
        self.assertEqual(eager_submit_origins[0], 1)
        eager_expected = tensor
        for _ in range(17):
            eager_expected = torch.relu(eager_expected)
        self.assertEqual(eager_output.cpu(), eager_expected)

    def test_cpp_graph_plan_dead_input_reuse_preserves_live_view(self):
        class LiveTransposeRelu(torch.nn.Module):
            def forward(self, tensor):
                value = tensor * 2.0
                view = value.transpose(0, 1)
                return view, torch.relu(value)

        model = LiveTransposeRelu().eval()
        tensor = torch.tensor([[-3.0, -2.0, -1.0], [0.0, 1.0, 2.0]])
        program = torch.vulkan.export_and_lower(model, tensor)

        self.assertEqual(
            program.cpp_plan.dead_input_reuse_instruction_count(), 1
        )
        output = tuple(value.cpu() for value in program(tensor))

        self.assertEqual(program.cpp_plan.dead_input_reuse_count(), 0)
        self.assertEqual(output, model(tensor))

    def test_cpp_graph_plan_owns_large_linear_checkpoint_submissions(self):
        class RepeatedLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1024, 1024, bias=False)

            def forward(self, tensor):
                value = tensor
                for _ in range(49):
                    linear = self.linear(value)
                    value = torch.nn.functional.gelu(
                        linear, approximate="tanh"
                    ) + linear
                return value

        model = RepeatedLinear().eval()
        tensor = torch.randn(1, 1024)
        program = torch.vulkan.export_and_lower(model, tensor)

        self.assertEqual(program.execution_mode, "cpp_plan")
        self.assertTrue(program.cpp_plan_report.submission_owned)
        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        torch.ops.vulkan_prepack.reset_submit_origin_counters()
        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            output = program(tensor)

        submit_origins = list(torch.ops.vulkan_prepack.submit_origin_counters())
        self.assertEqual(output.shape, (1, 1024))
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)
        self.assertEqual(program.cpp_plan.invocation_generation(), 1)
        self.assertGreater(program.cpp_plan.last_submission_value(), 0)
        self.assertGreaterEqual(submit_origins[15], 2)
        self.assertEqual(
            submit_origins[0], submit_origins[7] + submit_origins[15]
        )
        self.assertEqual(
            _graph_program_invocation_counters(),
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )

    def test_cpp_graph_plan_rejects_mutable_dispatch(self):
        with self.assertRaisesRegex(RuntimeError, "rejects mutable operator"):
            torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
                ["mutable_add"],
                ["aten::add_"],
                ["Tensor"],
                [[[0], [1], [-1]]],
                [[0, 0, 0]],
                [[2]],
                [1],
                2,
                [2],
            )

    def test_cpp_graph_plan_attributes_dispatch_failure_to_node(self):
        plan = torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
            ["invalid_mm"],
            ["aten::mm"],
            [""],
            [[[0], [1]]],
            [[0, 0]],
            [[2]],
            [],
            2,
            [2],
        )
        left = torch.randn(2, 3, device="vulkan")
        right = torch.randn(4, 2, device="vulkan")
        with self.assertRaisesRegex(
            RuntimeError,
            "VulkanGraphPlan.v9 node 'invalid_mm'.*failed",
        ):
            torch.ops.vulkan_prepack.run_vulkan_graph_plan.default(
                [left, right], plan
            )

    def test_cpp_graph_plan_executes_ordered_metadata_effect(self):
        class MetadataCheckedSin(torch.nn.Module):
            def forward(self, tensor):
                torch._assert_tensor_metadata(
                    tensor,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                )
                return torch.sin(tensor)

        model = MetadataCheckedSin().eval()
        first_input = torch.randn(2, 3)
        second_input = torch.randn(2, 3)
        program = torch.vulkan.export_and_lower(model, first_input)

        self.assertEqual(program.execution_mode, "cpp_plan")
        report = program.cpp_plan_report
        self.assertEqual(report.status, "compiled")
        self.assertEqual(report.reason, "immutable_ivalue_ssa_resource_plan")
        self.assertEqual(report.plan_version, "v9")
        self.assertEqual(report.input_count, 1)
        self.assertEqual(report.instruction_count, 2)
        self.assertEqual(report.effect_instruction_count, 1)
        self.assertEqual(report.graph_scalar_instruction_count, 0)
        self.assertEqual(report.list_argument_count, 0)
        self.assertEqual(report.value_count, 2)
        self.assertEqual(report.output_count, 1)
        self.assertEqual(report.value_use_counts, (2, 0))
        self.assertEqual(report.value_last_uses, (1, 1))
        self.assertEqual(program.cpp_plan.effect_instruction_count(), 1)
        self.assertEqual(program.cpp_plan.list_argument_count(), 0)

        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            first_output = program(first_input)
            second_output = program(second_input)

        torch.testing.assert_close(first_output.cpu(), model(first_input))
        torch.testing.assert_close(second_output.cpu(), model(second_input))
        self.assertEqual(program.run_count, 2)
        self.assertEqual(program.last_executed_nodes, report.node_names)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_cpp_graph_plan_reuses_program_owned_resources_after_completion(self):
        class ThreeLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Linear(4, 4)
                self.second = torch.nn.Linear(4, 4)
                self.output = torch.nn.Linear(4, 4)

            def forward(self, tensor):
                tensor = torch.sin(self.first(tensor))
                tensor = torch.sin(self.second(tensor))
                return self.output(tensor)

        torch.manual_seed(0)
        model = ThreeLinear().eval()
        first_input = torch.randn(2, 4)
        second_input = torch.randn(2, 4)
        program = torch.vulkan.export_and_lower(model, first_input)
        report = program.cpp_plan_report

        self.assertEqual(report.plan_version, "v9")
        self.assertEqual(report.resource_writer_instruction_count, 2)
        self.assertEqual(report.resource_value_count, 2)
        self.assertEqual(report.resource_slot_count, 1)
        self.assertEqual(report.resource_arena_flight_depth, 2)
        self.assertEqual(program.cpp_plan.resource_arena_generation_count(), 0)

        with torch.inference_mode():
            first_output = program(first_input)
            first_before = first_output.cpu()
            torch.ops.vulkan_prepack.synchronize()
            second_output = program(second_input)
            second_cpu = second_output.cpu()
            first_after = first_output.cpu()

        torch.testing.assert_close(first_before, model(first_input))
        torch.testing.assert_close(second_cpu, model(second_input))
        torch.testing.assert_close(first_after, first_before)
        self.assertEqual(program.cpp_plan.resource_arena_generation_count(), 1)
        self.assertEqual(program.cpp_plan.resource_arena_capture_count(), 1)
        self.assertGreaterEqual(program.cpp_plan.resource_arena_reuse_count(), 1)
        self.assertEqual(program.cpp_plan.resource_arena_spill_count(), 0)
        self.assertEqual(program.cpp_plan.resource_write_count(), 4)
        self.assertEqual(program.cpp_plan.resource_writer_bypass_count(), 0)

    def test_cpp_graph_plan_rejects_resource_alias_that_escapes(self):
        class LinearViewOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, tensor):
                return self.linear(tensor).view(4, 2)

        torch.manual_seed(0)
        model = LinearViewOutput().eval()
        tensor = torch.randn(2, 4)
        program = torch.vulkan.export_and_lower(model, tensor)
        report = program.cpp_plan_report

        self.assertEqual(report.resource_writer_instruction_count, 0)
        self.assertEqual(report.resource_value_count, 0)
        self.assertEqual(report.resource_slot_count, 0)
        self.assertEqual(report.resource_alias_extended_lifetime_count, 0)
        self.assertEqual(report.resource_alias_escape_rejection_count, 1)
        torch.testing.assert_close(program(tensor).cpu(), model(tensor))

    def test_cpp_graph_plan_extends_resource_lifetime_through_aliases(self):
        class TwoLinearLiveView(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Linear(4, 4)
                self.second = torch.nn.Linear(4, 4)

            def forward(self, tensor):
                first = self.first(tensor)
                alias = first.view(4, 2).view(2, 4)
                second = self.second(tensor)
                return torch.sin(alias + second)

        torch.manual_seed(0)
        model = TwoLinearLiveView().eval()
        tensor = torch.randn(2, 4)
        program = torch.vulkan.export_and_lower(model, tensor)
        report = program.cpp_plan_report

        self.assertEqual(report.resource_writer_instruction_count, 2)
        self.assertEqual(report.resource_value_count, 2)
        self.assertEqual(report.resource_slot_count, 2)
        self.assertEqual(report.resource_alias_extended_lifetime_count, 1)
        self.assertEqual(report.resource_alias_escape_rejection_count, 0)
        torch.testing.assert_close(program(tensor).cpu(), model(tensor))

    def test_cpp_graph_plan_reuses_resource_arena_after_partial_failure(self):
        context = torch.ops.vulkan_prepack.create_linear_context(
            torch.randn(4, 4), torch.randn(4)
        )
        plan = torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
            ["linear", "metadata_check", "sin"],
            [
                "vulkan_prepack::run_linear_context",
                "aten::_assert_tensor_metadata",
                "aten::sin",
            ],
            ["", "", ""],
            [
                [[0], [-1]],
                [[1], [-2], [-3], [-4], [-5], [-6]],
                [[1]],
            ],
            [[0, 0], [0, 0, 0, 0, 0, 0], [0]],
            [[1], [], [2]],
            [context, None, None, torch.float64, None, None],
            1,
            [2],
            0,
            0,
            False,
            None,
            [-1, 0, -1],
            [2, 4],
            [2],
            2,
        )
        tensor = torch.randn(2, 4, device="vulkan")

        self.assertEqual(plan.resource_slot_count(), 1)
        self.assertEqual(plan.resource_value_count(), 1)
        self.assertEqual(plan.resource_writer_instruction_count(), 1)
        for _ in range(2):
            with self.assertRaisesRegex(
                RuntimeError,
                "VulkanGraphPlan.v9 node 'metadata_check'.*failed",
            ):
                torch.ops.vulkan_prepack.run_vulkan_graph_plan.default(
                    [tensor], plan
                )
            torch.ops.vulkan_prepack.synchronize()

        self.assertEqual(plan.resource_arena_generation_count(), 1)
        self.assertEqual(plan.resource_arena_capture_count(), 1)
        self.assertGreaterEqual(plan.resource_arena_reuse_count(), 1)
        self.assertEqual(plan.resource_arena_spill_count(), 0)
        self.assertEqual(plan.resource_write_count(), 2)
        self.assertEqual(plan.resource_writer_bypass_count(), 0)

    def test_cpp_graph_plan_rejects_resource_slot_on_escaping_output(self):
        context = torch.ops.vulkan_prepack.create_linear_context(
            torch.randn(4, 4), torch.randn(4)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "escaping outputs cannot use an internal resource slot",
        ):
            torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
                ["linear"],
                ["vulkan_prepack::run_linear_context"],
                [""],
                [[[0], [-1]]],
                [[0, 0]],
                [[1]],
                [context],
                1,
                [1],
                0,
                0,
                False,
                None,
                [-1, 0],
                [2, 4],
                [2],
                2,
            )

    def test_cpp_graph_plan_rejects_overlapping_resource_slot_lifetimes(self):
        context = torch.ops.vulkan_prepack.create_linear_context(
            torch.randn(4, 4), torch.randn(4)
        )
        with self.assertRaisesRegex(RuntimeError, "resource slot lifetimes overlap"):
            torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
                ["first_linear", "second_linear", "add"],
                [
                    "vulkan_prepack::run_linear_context",
                    "vulkan_prepack::run_linear_context",
                    "aten::add",
                ],
                ["", "", "Tensor"],
                [
                    [[0], [-1]],
                    [[0], [-1]],
                    [[1], [2], [-2]],
                ],
                [[0, 0], [0, 0], [0, 0, 0]],
                [[1], [2], [3]],
                [context, 1],
                1,
                [3],
                0,
                0,
                False,
                None,
                [-1, 0, 0, -1],
                [2, 4],
                [2],
                2,
            )

    def test_cpp_graph_plan_rejects_malformed_resource_slot_ranks(self):
        context = torch.ops.vulkan_prepack.create_linear_context(
            torch.randn(4, 4), torch.randn(4)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "resource slot ranks must partition the flat sizes",
        ):
            torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
                ["linear", "sin"],
                ["vulkan_prepack::run_linear_context", "aten::sin"],
                ["", ""],
                [[[0], [-1]], [[1]]],
                [[0, 0], [0]],
                [[1], [2]],
                [context],
                1,
                [2],
                0,
                0,
                False,
                None,
                [-1, 0, -1],
                [2, 4],
                [3],
                2,
            )

    def test_cpp_graph_plan_releases_resource_arena_when_plan_dies(self):
        gc.collect()
        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        weight = torch.randn(4, 4)
        bias = torch.randn(4)
        context = torch.ops.vulkan_prepack.create_linear_context(weight, bias)
        plan = torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
            ["linear", "sin"],
            ["vulkan_prepack::run_linear_context", "aten::sin"],
            ["", ""],
            [[[0], [-1]], [[1]]],
            [[0, 0], [0]],
            [[1], [2]],
            [context],
            1,
            [2],
            0,
            0,
            False,
            None,
            [-1, 0, -1],
            [2, 4],
            [2],
            2,
        )
        tensor = torch.randn(2, 4, device="vulkan")
        output = torch.ops.vulkan_prepack.run_vulkan_graph_plan.default(
            [tensor], plan
        )[0]
        self.assertEqual(output.cpu().shape, (2, 4))
        del output
        torch.ops.vulkan_prepack.synchronize()
        del plan
        gc.collect()

        counters = _graph_program_invocation_counters()
        self.assertGreaterEqual(counters[10], 1)
        self.assertEqual(counters[11:], [0, 0, 0])

    def test_cpp_graph_plan_attributes_effect_failure_before_later_op(self):
        plan = torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
            ["metadata_check", "sin"],
            ["aten::_assert_tensor_metadata", "aten::sin"],
            ["", ""],
            [
                [[0], [-1], [-2], [-3], [-4], [-5]],
                [[0]],
            ],
            [[0, 0, 0, 0, 0, 0], [0]],
            [[], [1]],
            [None, None, torch.float64, None, None],
            1,
            [1],
        )
        tensor = torch.randn(2, 3, device="vulkan")
        with self.assertRaisesRegex(
            RuntimeError,
            "VulkanGraphPlan.v9 node 'metadata_check'.*failed",
        ):
            torch.ops.vulkan_prepack.run_vulkan_graph_plan.default(
                [tensor], plan
            )

    def test_cpp_graph_plan_carries_non_tensor_ivalue(self):
        plan = torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
            ["is_contiguous", "assert_contiguous", "sin"],
            ["aten::is_contiguous", "aten::_assert_scalar", "aten::sin"],
            ["", "", ""],
            [[[0]], [[1], [-1]], [[0]]],
            [[0], [0, 0], [0]],
            [[1], [], [2]],
            ["Vulkan input must be contiguous"],
            1,
            [2],
        )
        tensor = torch.randn(2, 3, device="vulkan")
        output = torch.ops.vulkan_prepack.run_vulkan_graph_plan.default(
            [tensor], plan
        )
        torch.testing.assert_close(output[0].cpu(), tensor.cpu().sin())
        self.assertEqual(plan.effect_instruction_count(), 1)
        self.assertEqual(plan.list_argument_count(), 0)
        self.assertEqual(tuple(plan.value_use_counts()), (2, 1, 0))
        self.assertEqual(tuple(plan.value_last_uses()), (2, 1, 2))

    def test_cpp_graph_plan_compiles_non_tensor_ivalue(self):
        class ContiguityObservedSin(torch.nn.Module):
            def forward(self, tensor):
                torch.ops.aten.is_contiguous.default(tensor)
                return torch.sin(tensor)

        model = ContiguityObservedSin().eval()
        tensor = torch.randn(2, 3)
        program = torch.vulkan.export_and_lower(model, tensor)

        self.assertEqual(program.execution_mode, "cpp_plan")
        report = program.cpp_plan_report
        self.assertEqual(report.status, "compiled")
        self.assertEqual(report.effect_instruction_count, 0)
        self.assertEqual(report.instruction_count, 2)
        self.assertEqual(report.value_count, 3)
        self.assertEqual(report.value_use_counts, (2, 0, 0))
        self.assertEqual(report.value_last_uses, (1, 0, 1))

        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            output = program(tensor)
        torch.testing.assert_close(output.cpu(), model(tensor))

    def test_cpp_graph_plan_executes_dynamic_tensor_list(self):
        class CatSin(torch.nn.Module):
            def forward(self, left, right):
                return torch.cat([left, right], dim=1).sin()

        model = CatSin().eval()
        first_left = torch.randn(2, 2)
        first_right = torch.randn(2, 3)
        second_left = torch.randn(2, 2)
        second_right = torch.randn(2, 3)
        program = torch.vulkan.export_and_lower(
            model,
            (first_left, first_right),
        )

        self.assertEqual(program.execution_mode, "cpp_plan")
        report = program.cpp_plan_report
        self.assertEqual(report.status, "compiled")
        self.assertEqual(report.plan_version, "v9")
        self.assertEqual(report.instruction_count, 2)
        self.assertEqual(report.effect_instruction_count, 0)
        self.assertEqual(report.graph_scalar_instruction_count, 0)
        self.assertEqual(report.list_argument_count, 1)
        self.assertEqual(report.invocation_value_slot_count, 4)
        self.assertEqual(report.invocation_list_slot_count, 1)
        self.assertGreaterEqual(report.invocation_stack_capacity, 2)
        self.assertEqual(report.value_count, 4)
        self.assertEqual(report.value_use_counts, (1, 1, 1, 0))
        self.assertEqual(report.value_last_uses, (0, 0, 1, 1))
        self.assertEqual(program.cpp_plan.list_argument_count(), 1)
        self.assertEqual(program.cpp_plan.invocation_value_slot_count(), 4)
        self.assertEqual(program.cpp_plan.invocation_list_slot_count(), 1)
        self.assertGreaterEqual(program.cpp_plan.invocation_stack_capacity(), 2)

        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            first_output = program(first_left, first_right)
            second_output = program(second_left, second_right)

        torch.testing.assert_close(
            first_output.cpu(), model(first_left, first_right)
        )
        torch.testing.assert_close(
            second_output.cpu(), model(second_left, second_right)
        )
        self.assertEqual(program.run_count, 2)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_cpp_graph_plan_executes_empty_typed_list(self):
        class DefaultStrideAvgPool(torch.nn.Module):
            def forward(self, tensor):
                return torch.ops.aten.avg_pool2d.default(tensor, [3, 2])

        model = DefaultStrideAvgPool().eval()
        tensor = torch.randn(2, 4, 32, 32)
        program = torch.vulkan.export_and_lower(model, tensor)

        self.assertEqual(program.execution_mode, "cpp_plan")
        report = program.cpp_plan_report
        self.assertEqual(report.status, "compiled")
        self.assertEqual(report.instruction_count, 1)
        self.assertEqual(report.list_argument_count, 1)
        self.assertEqual(program.cpp_plan.list_argument_count(), 1)
        self.assertEqual(program.cpp_plan.invocation_value_slot_count(), 2)
        self.assertEqual(program.cpp_plan.invocation_list_slot_count(), 1)
        self.assertGreaterEqual(program.cpp_plan.invocation_stack_capacity(), 2)
        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            output = program(tensor)
        torch.testing.assert_close(output.cpu(), model(tensor))
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_cpp_graph_plan_keeps_list_return_arguments_transient(self):
        plan = torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
            ["broadcast", "getitem"],
            ["aten::broadcast_tensors", "vulkan_graph::list_getitem"],
            ["", ""],
            [[[0, 1]], [[2], [-1]]],
            [[1], [0, 0]],
            [[2], [3]],
            [0],
            2,
            [3],
        )
        self.assertEqual(plan.list_argument_count(), 1)
        self.assertEqual(plan.invocation_list_slot_count(), 0)

        left = torch.randn(2, 1)
        right = torch.randn(1, 3)
        vulkan_inputs = [left.to("vulkan"), right.to("vulkan")]
        for _ in range(2):
            output = torch.ops.vulkan_prepack.run_vulkan_graph_plan.default(
                vulkan_inputs,
                plan,
            )
            torch.testing.assert_close(
                output[0].cpu(),
                torch.broadcast_tensors(left, right)[0],
            )

    def test_cpp_graph_plan_rejects_list_recipe_for_scalar_argument(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "declares a list recipe for non-list argument 'self'",
        ):
            torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
                ["sin"],
                ["aten::sin"],
                [""],
                [[[0]]],
                [[1]],
                [[1]],
                [],
                1,
                [1],
            )

    def test_cpp_graph_plan_rejects_empty_value_recipe(self):
        with self.assertRaisesRegex(RuntimeError, "invalid argument recipe"):
            torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
                ["sin"],
                ["aten::sin"],
                [""],
                [[[]]],
                [[0]],
                [[1]],
                [],
                1,
                [1],
            )

    def test_cpp_graph_plan_canonicalizes_scalar_tensor_literal(self):
        class ScalarTensorMul(torch.nn.Module):
            def forward(self, tensor):
                return torch.ops.aten.mul.Tensor(tensor, 1.5)

        model = ScalarTensorMul().eval()
        tensor = torch.randn(2, 3)
        program = torch.vulkan.export_and_lower(model, tensor)

        self.assertEqual(program.execution_mode, "cpp_plan")
        self.assertEqual(program.cpp_plan_report.status, "compiled")
        self.assertEqual(program.cpp_plan_report.plan_version, "v9")
        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            output = program(tensor)
        torch.testing.assert_close(output.cpu(), model(tensor))
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_cpp_graph_plan_elides_functionalized_fresh_detach_chain(self):
        class FreshDetachCat(torch.nn.Module):
            def forward(self, tensor):
                fresh = torch.tensor([1.0, 2.0])
                return torch.cat([tensor, fresh.detach_()])

        model = FreshDetachCat().eval()
        first_input = torch.randn(2)
        second_input = torch.randn(2)
        program = torch.vulkan.export_and_lower(model, first_input)

        functionalization = program.fresh_detach_functionalization
        self.assertEqual(functionalization.candidate_count, 2)
        self.assertEqual(functionalization.functionalized_count, 2)
        self.assertEqual(functionalization.rejected_count, 0)
        self.assertEqual(
            tuple(node.status for node in functionalization.nodes),
            ("functionalized", "functionalized"),
        )
        self.assertEqual(
            tuple(node.reason for node in functionalization.nodes),
            (
                "fresh_single_user_detach_chain",
                "fresh_single_user_detach_chain",
            ),
        )
        self.assertEqual(
            tuple(node.replacement_target for node in functionalization.nodes),
            ("aten::detach", "aten::detach"),
        )
        self.assertFalse(
            any(
                node.target == torch.ops.aten.detach_.default
                for node in program.graph_module.graph.nodes
            )
        )
        self.assertEqual(
            sum(
                node.target == torch.ops.aten.detach.default
                for node in program.graph_module.graph.nodes
            ),
            0,
        )
        identities = program.static_inference_identities
        self.assertEqual(identities.candidate_count, 2)
        self.assertEqual(identities.lowered_count, 2)
        self.assertEqual(identities.skipped_count, 0)
        self.assertEqual(
            tuple(node.reason for node in identities.nodes),
            (
                "functionalized_fresh_detach_under_inference",
                "functionalized_fresh_detach_under_inference",
            ),
        )
        self.assertEqual(
            tuple(node.operator_name for node in identities.nodes),
            ("aten::detach", "aten::detach"),
        )
        self.assertEqual(program.execution_mode, "cpp_plan")
        self.assertEqual(program.cpp_plan_report.status, "compiled")
        self.assertTrue(program.cpp_plan_report.submission_owned)

        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            first_output = program(first_input)
            second_output = program(second_input)

        self.assertEqual(first_output.cpu(), model(first_input))
        self.assertEqual(second_output.cpu(), model(second_input))
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)
        self.assertEqual(program.cpp_plan.invocation_generation(), 2)
        self.assertGreater(program.cpp_plan.last_submission_value(), 0)
        self.assertTrue(program.cpp_plan.last_submission_complete())
        self.assertEqual(
            _graph_program_invocation_counters(),
            [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )

    def test_fresh_detach_functionalization_rejects_input_alias(self):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        detached = graph.call_function(
            torch.ops.aten.detach_.default,
            (tensor,),
        )
        graph.output(detached)
        graph_module = torch.fx.GraphModule({}, graph)

        report = vulkan_graph_lowering.functionalize_fresh_detach_mutations(
            graph_module
        )

        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.functionalized_count, 0)
        self.assertEqual(report.rejected_count, 1)
        self.assertEqual(report.nodes[0].status, "rejected")
        self.assertEqual(
            report.nodes[0].reason,
            "input_is_not_a_fresh_detach_chain",
        )
        self.assertEqual(detached.target, torch.ops.aten.detach_.default)

    def test_fresh_detach_functionalization_rejects_branched_fresh_value(self):
        root = torch.nn.Module()
        root.register_buffer("constant", torch.ones(2), persistent=False)
        graph = torch.fx.Graph()
        constant = graph.get_attr("constant")
        fresh = graph.call_function(
            torch.ops.aten.lift_fresh_copy.default,
            (constant,),
        )
        detached = graph.call_function(
            torch.ops.aten.detach_.default,
            (fresh,),
        )
        other = graph.call_function(torch.ops.aten.sin.default, (fresh,))
        graph.output((detached, other))
        graph_module = torch.fx.GraphModule(root, graph)

        report = vulkan_graph_lowering.functionalize_fresh_detach_mutations(
            graph_module
        )

        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.functionalized_count, 0)
        self.assertEqual(report.rejected_count, 1)
        self.assertEqual(report.nodes[0].status, "rejected")
        self.assertEqual(
            report.nodes[0].reason,
            "fresh_chain_value_has_other_users",
        )
        self.assertEqual(detached.target, torch.ops.aten.detach_.default)

    def test_cpp_graph_plan_functionalizes_fresh_relu(self):
        class FreshRelu(torch.nn.Module):
            def forward(self, tensor):
                fresh = tensor.clone()
                return torch.ops.aten.relu_.default(fresh)

        model = FreshRelu().eval()
        tensor = torch.randn(2, 3)
        program = torch.vulkan.export_and_lower(model, tensor)

        functionalization = program.fresh_relu_functionalization
        self.assertEqual(functionalization.candidate_count, 1)
        self.assertEqual(functionalization.functionalized_count, 1)
        self.assertEqual(functionalization.rejected_count, 0)
        self.assertEqual(functionalization.nodes[0].status, "functionalized")
        self.assertEqual(
            functionalization.nodes[0].reason,
            "fresh_single_user_non_aliasing_tensor_result",
        )
        self.assertEqual(
            functionalization.nodes[0].source_operator_name,
            "aten.clone.default",
        )
        self.assertEqual(
            functionalization.nodes[0].replacement_target,
            "aten::relu",
        )
        self.assertFalse(
            any(
                node.target == torch.ops.aten.relu_.default
                for node in program.graph_module.graph.nodes
            )
        )
        self.assertEqual(program.execution_mode, "cpp_plan")
        self.assertEqual(program.cpp_plan_report.status, "compiled")
        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            output = program(tensor)
        self.assertEqual(output.cpu(), model(tensor))
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_fresh_relu_functionalization_rejects_input_alias(self):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        relu = graph.call_function(torch.ops.aten.relu_.default, (tensor,))
        graph.output(relu)
        graph_module = torch.fx.GraphModule({}, graph)

        report = vulkan_graph_lowering.functionalize_fresh_relu_mutations(
            graph_module
        )

        self.assertEqual(report.functionalized_count, 0)
        self.assertEqual(report.rejected_count, 1)
        self.assertEqual(
            report.nodes[0].reason,
            "source_is_not_an_operator_result",
        )
        self.assertEqual(relu.target, torch.ops.aten.relu_.default)

    def test_fresh_relu_functionalization_rejects_aliasing_source(self):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        view = graph.call_function(torch.ops.aten.view.default, (tensor, [6]))
        relu = graph.call_function(torch.ops.aten.relu_.default, (view,))
        graph.output(relu)
        graph_module = torch.fx.GraphModule({}, graph)

        report = vulkan_graph_lowering.functionalize_fresh_relu_mutations(
            graph_module
        )

        self.assertEqual(report.functionalized_count, 0)
        self.assertEqual(report.rejected_count, 1)
        self.assertEqual(report.nodes[0].reason, "source_result_may_alias")
        self.assertEqual(relu.target, torch.ops.aten.relu_.default)

    def test_fresh_relu_functionalization_rejects_branched_source(self):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        fresh = graph.call_function(torch.ops.aten.clone.default, (tensor,))
        relu = graph.call_function(torch.ops.aten.relu_.default, (fresh,))
        other = graph.call_function(torch.ops.aten.sin.default, (fresh,))
        graph.output((relu, other))
        graph_module = torch.fx.GraphModule({}, graph)

        report = vulkan_graph_lowering.functionalize_fresh_relu_mutations(
            graph_module
        )

        self.assertEqual(report.functionalized_count, 0)
        self.assertEqual(report.rejected_count, 1)
        self.assertEqual(report.nodes[0].reason, "fresh_value_has_other_users")
        self.assertEqual(relu.target, torch.ops.aten.relu_.default)

    def test_cpp_graph_plan_elides_static_inference_dropout(self):
        class InferenceDropout(torch.nn.Module):
            def forward(self, tensor):
                dropped = torch.ops.aten.dropout.default(tensor, 0.25, False)
                zero_probability = torch.ops.aten.dropout.default(
                    dropped, 0.0, True
                )
                return zero_probability + 1.0

        model = InferenceDropout().eval()
        tensor = torch.randn(2, 3)
        program = torch.vulkan.export_and_lower(model, tensor)

        report = program.static_inference_identities
        self.assertEqual(report.candidate_count, 2)
        self.assertEqual(report.lowered_count, 2)
        self.assertEqual(report.skipped_count, 0)
        self.assertEqual(
            tuple(node.status for node in report.nodes),
            ("lowered", "lowered"),
        )
        self.assertEqual(
            tuple(node.reason for node in report.nodes),
            (
                "static_dropout_training_disabled",
                "static_dropout_probability_zero",
            ),
        )
        self.assertEqual(
            tuple(node.probability for node in report.nodes),
            (0.25, 0.0),
        )
        self.assertEqual(
            tuple(node.training for node in report.nodes),
            (False, True),
        )
        self.assertFalse(
            any(
                node.target == torch.ops.aten.dropout.default
                for node in program.graph_module.graph.nodes
            )
        )
        self.assertEqual(program.execution_mode, "cpp_plan")
        self.assertEqual(program.cpp_plan_report.instruction_count, 1)
        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            output = program(tensor)
        self.assertEqual(output.cpu(), model(tensor))
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)

    def test_static_inference_dropout_preserves_training_semantics(self):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        dropout = graph.call_function(
            torch.ops.aten.dropout.default,
            (tensor, 0.25, True),
        )
        graph.output(dropout)
        graph_module = torch.fx.GraphModule({}, graph)

        report = vulkan_graph_lowering.lower_static_inference_identities(
            graph_module
        )

        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.lowered_count, 0)
        self.assertEqual(report.skipped_count, 1)
        self.assertEqual(report.nodes[0].reason, "dropout_training_enabled")
        self.assertEqual(dropout.target, torch.ops.aten.dropout.default)

    def test_static_inference_dropout_preserves_probability_validation(self):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        dropout = graph.call_function(
            torch.ops.aten.dropout.default,
            (tensor, 1.25, False),
        )
        graph.output(dropout)
        graph_module = torch.fx.GraphModule({}, graph)

        report = vulkan_graph_lowering.lower_static_inference_identities(
            graph_module
        )

        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.lowered_count, 0)
        self.assertEqual(report.skipped_count, 1)
        self.assertEqual(
            report.nodes[0].reason,
            "dropout_probability_out_of_range",
        )
        self.assertEqual(dropout.target, torch.ops.aten.dropout.default)

    def test_static_inference_identity_preserves_unproven_detach(self):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        detached = graph.call_function(torch.ops.aten.detach.default, (tensor,))
        graph.output(detached)
        graph_module = torch.fx.GraphModule({}, graph)

        report = vulkan_graph_lowering.lower_static_inference_identities(
            graph_module
        )

        self.assertEqual(report.candidate_count, 0)
        self.assertEqual(report.lowered_count, 0)
        self.assertEqual(report.skipped_count, 0)
        self.assertEqual(detached.target, torch.ops.aten.detach.default)

    def test_static_linear_gelu_tied_context_stays_unfused(self):
        class TiedLinearGelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)

            def forward(self, tensor):
                first = torch.nn.functional.gelu(
                    self.linear(tensor), approximate="tanh"
                )
                second = torch.nn.functional.gelu(
                    self.linear(tensor), approximate="tanh"
                )
                return first + second

        torch.manual_seed(14)
        model = TiedLinearGelu().eval()
        tensor = torch.randn(2, 3)
        program = torch.vulkan.export_and_lower(model, tensor)
        torch.testing.assert_close(
            program(tensor).cpu(), model(tensor), rtol=1e-4, atol=1e-4
        )
        report = program.static_linear_gelu_regions
        self.assertEqual(report.lowered_count, 0)
        self.assertEqual(report.skipped_count, 2)
        self.assertTrue(
            all(
                node.reason == "context_attr_has_multiple_references"
                for node in report.nodes
            )
        )
        self.assertFalse(_static_linear_gelu_plan_attrs(program))
        self.assertEqual(len(_linear_context_attrs(program)), 1)

    def test_static_linear_gelu_nonprivate_context_stays_unfused(self):
        for approximate in ("tanh", "none"):
            with self.subTest(approximate=approximate):
                graph = torch.fx.Graph()
                tensor = graph.placeholder("tensor")
                context = graph.get_attr("context")
                linear = graph.call_function(
                    torch.ops.vulkan_prepack.run_linear_context.default,
                    args=(tensor, context),
                )
                gelu = graph.call_function(
                    torch.ops.aten.gelu.default,
                    args=(linear,),
                    kwargs={"approximate": approximate},
                )
                graph.output(gelu)
                root = torch.nn.Module()
                root.context = object()
                graph_module = torch.fx.GraphModule(root, graph)

                report = vulkan_graph.lower_static_linear_gelu_regions(graph_module)
                self.assertEqual(report.candidate_count, 0)
                self.assertEqual(report.lowered_count, 0)
                self.assertFalse(
                    any(
                        str(node.target)
                        == "vulkan_prepack.run_vulkan_graph_region_plan.default"
                        for node in graph_module.graph.nodes
                    )
                )

    def test_static_linear_gelu_dynamic_context_stays_unfused(self):
        for approximate in ("tanh", "none"):
            with self.subTest(approximate=approximate):
                graph = torch.fx.Graph()
                tensor = graph.placeholder("tensor")
                context = graph.placeholder("context")
                linear = graph.call_function(
                    torch.ops.vulkan_prepack.run_linear_context.default,
                    args=(tensor, context),
                )
                gelu = graph.call_function(
                    torch.ops.aten.gelu.default,
                    args=(linear,),
                    kwargs={"approximate": approximate},
                )
                graph.output(gelu)
                graph_module = torch.fx.GraphModule({}, graph)

                report = vulkan_graph.lower_static_linear_gelu_regions(graph_module)
                self.assertEqual(report.candidate_count, 0)
                self.assertEqual(report.lowered_count, 0)
                self.assertFalse(
                    any(
                        str(node.target)
                        == "vulkan_prepack.run_vulkan_graph_region_plan.default"
                        for node in graph_module.graph.nodes
                    )
                )

    def test_static_layernorm_lowering_matches_cpu_and_eager_vulkan(self):
        class AffineLayerNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(4, eps=1e-5)

            def forward(self, tensor):
                return self.norm(tensor)

        torch.manual_seed(29)
        model = AffineLayerNorm().eval()
        tensor = torch.randn(2, 3, 4)
        expected = model(tensor)
        program = torch.vulkan.export_and_lower(model, tensor)
        self.assertTrue(program.cpp_plan_report.submission_owned)
        self.assertTrue(program.cpp_plan.submission_owned())
        eager_vulkan = copy.deepcopy(model).to("vulkan").eval()
        with torch.inference_mode():
            eager_vulkan_output = eager_vulkan(tensor.to("vulkan")).cpu()
        graph_output = program(tensor).cpu()

        torch.testing.assert_close(
            eager_vulkan_output, expected, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(graph_output, expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            graph_output, eager_vulkan_output, rtol=1e-4, atol=1e-4
        )
        report = program.layernorm_lowering
        self.assertEqual(report.layer_norm_node_count, 1)
        self.assertEqual(report.lowered_count, 1)
        self.assertEqual(report.rejected_count, 0)
        self.assertEqual(report.skipped_count, 0)
        self.assertEqual(report.created_context_count, 1)
        self.assertEqual(report.reused_context_count, 0)
        self.assertEqual(
            report.context_factory,
            "vulkan_prepack::create_layernorm_context",
        )
        node = report.nodes[0]
        self.assertEqual(node.reason, "static_cpu_snapshot_affine_parameters")
        self.assertEqual(node.normalized_shape, (4,))
        self.assertEqual(node.context_status, "created")
        context_attrs = _layernorm_context_attrs(program)
        self.assertEqual(len(context_attrs), 1)
        self.assertTrue(
            isinstance(
                getattr(program.graph_module, next(iter(context_attrs))),
                torch.ScriptObject,
            )
        )
        self.assertNotIn("norm.weight", program.graph_module.state_dict())
        self.assertNotIn("norm.bias", program.graph_module.state_dict())
        self.assertFalse(program.graph_module.state_dict())
        lowered = [
            node
            for node in program.census.nodes
            if node.reason == "graph_owned_layernorm_context"
        ]
        self.assertEqual(len(lowered), 1)
        self.assertEqual(lowered[0].classification, "lowered_vulkan")
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_static_layernorm_lowering_reuses_dynamic_shapes(self):
        class AffineLayerNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(4)

            def forward(self, tensor):
                return self.norm(tensor)

        torch.manual_seed(30)
        model = AffineLayerNorm().eval()
        batch = torch.export.Dim("batch", min=1, max=4)
        sequence = torch.export.Dim("sequence", min=1, max=8)
        program = torch.vulkan.export_and_lower(
            model,
            torch.randn(2, 3, 4),
            dynamic_shapes=({0: batch, 1: sequence},),
        )
        context_attrs = _layernorm_context_attrs(program)
        self.assertEqual(len(context_attrs), 1)
        for shape in ((1, 8, 4), (4, 1, 4)):
            tensor = torch.randn(shape)
            torch.testing.assert_close(
                program(tensor).cpu(), model(tensor), rtol=1e-4, atol=1e-4
            )
        self.assertEqual(_layernorm_context_attrs(program), context_attrs)
        self.assertEqual(program.layernorm_lowering.lowered_count, 1)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_static_layernorm_lowering_reuses_tied_affine_state(self):
        class TiedAffineLayerNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4))
                self.bias = torch.nn.Parameter(torch.randn(4))

            def forward(self, first, second):
                return torch.nn.functional.layer_norm(
                    first, (4,), self.weight, self.bias, 1e-5
                ) + torch.nn.functional.layer_norm(
                    second, (4,), self.weight, self.bias, 1e-5
                )

        torch.manual_seed(32)
        model = TiedAffineLayerNorm().eval()
        first = torch.randn(2, 3, 4)
        second = torch.randn(2, 3, 4)
        program = torch.vulkan.export_and_lower(model, (first, second))
        torch.testing.assert_close(
            program(first, second).cpu(),
            model(first, second),
            rtol=1e-4,
            atol=1e-4,
        )
        report = program.layernorm_lowering
        self.assertEqual(report.lowered_count, 2)
        self.assertEqual(report.created_context_count, 1)
        self.assertEqual(report.reused_context_count, 1)
        self.assertEqual(len(_layernorm_context_attrs(program)), 1)
        self.assertFalse(program.graph_module.state_dict())

    def test_static_layernorm_lowering_skips_unsupported_semantics(self):
        cases = (
            ("dynamic_affine", [4], torch.ones(4), torch.zeros(4), {}),
            ("missing_affine", [4], None, None, {}),
            (
                "malformed_normalized_shape",
                "four",
                torch.ones(4),
                torch.zeros(4),
                {},
            ),
            ("incompatible_affine", [4], torch.ones(3), torch.zeros(3), {}),
        )
        expected_reasons = {
            "dynamic_affine": "weight_not_static_get_attr",
            "missing_affine": "affine_weight_missing",
            "malformed_normalized_shape": (
                "normalized_shape_not_static_positive_int_sequence"
            ),
            "incompatible_affine": "affine_state_incompatible_with_normalized_shape",
        }
        for kind, normalized_shape, weight, bias, snapshot in cases:
            graph = torch.fx.Graph()
            tensor = graph.placeholder("tensor")
            tensor.meta["val"] = torch.empty(2, 4)
            root = torch.nn.Module()
            if kind == "dynamic_affine":
                weight_arg = graph.placeholder("weight")
                bias_arg = graph.placeholder("bias")
            elif kind == "missing_affine":
                weight_arg = None
                bias_arg = None
            else:
                root.register_buffer("weight", weight)
                root.register_buffer("bias", bias)
                weight_arg = graph.get_attr("weight")
                bias_arg = graph.get_attr("bias")
                snapshot = {"weight": weight, "bias": bias}
            layer_norm = graph.call_function(
                torch.ops.aten.layer_norm.default,
                args=(tensor, normalized_shape, weight_arg, bias_arg, 1e-5, False),
            )
            graph.output(layer_norm)
            graph_module = torch.fx.GraphModule(root, graph)
            report = vulkan_graph.lower_static_layernorm_to_vulkan_contexts(
                graph_module, snapshot
            )
            self.assertEqual(report.lowered_count, 0)
            self.assertEqual(report.rejected_count, 0)
            self.assertEqual(report.skipped_count, 1)
            self.assertEqual(report.nodes[0].reason, expected_reasons[kind])
            self.assertFalse(_layernorm_context_attrs_from_module(graph_module))

    def test_forged_or_missing_layernorm_context_is_unsupported(self):
        for missing in (False, True):
            graph = torch.fx.Graph()
            tensor = graph.placeholder("tensor")
            context = graph.get_attr("_vulkan_layernorm_context_forged")
            layer_norm = graph.call_function(
                torch.ops.vulkan_prepack.run_layernorm_context.default,
                args=(tensor, [4], context),
            )
            graph.output(layer_norm)
            root = torch.nn.Module()
            setattr(root, "_vulkan_layernorm_context_forged", object())
            graph_module = torch.fx.GraphModule(root, graph)
            if missing:
                delattr(graph_module, "_vulkan_layernorm_context_forged")
            census = vulkan_graph._build_census(graph_module)
            record = next(
                node
                for node in census.nodes
                if node.target == "vulkan_prepack::run_layernorm_context"
            )
            self.assertEqual(record.classification, "unsupported")
            self.assertEqual(
                record.reason,
                "run_layernorm_context_missing_graph_owned_context",
            )

    def test_static_add_layernorm_region_matches_cpu_and_eager_vulkan(self):
        class AddLayerNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(4, eps=1e-5)

            def forward(self, first, second):
                residual = first + second
                return residual, self.norm(residual)

        torch.manual_seed(36)
        model = AddLayerNorm().eval()
        first = torch.randn(2, 3, 4)
        second = torch.randn(2, 3, 4)
        expected = model(first, second)
        program = torch.vulkan.export_and_lower(model, (first, second))
        eager_vulkan = copy.deepcopy(model).to("vulkan").eval()
        with torch.inference_mode():
            eager_output = eager_vulkan(
                first.to("vulkan"), second.to("vulkan")
            )
            eager_output = tuple(value.cpu() for value in eager_output)
        graph_output = tuple(value.cpu() for value in program(first, second))

        for actual, reference, eager in zip(graph_output, expected, eager_output):
            torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(actual, eager, rtol=1e-4, atol=1e-4)
        report = program.static_add_layernorm_regions
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.lowered_count, 1)
        self.assertEqual(report.rejected_count, 0)
        self.assertEqual(report.skipped_count, 0)
        self.assertEqual(
            report.plan_factory,
            "vulkan_prepack::create_graph_add_layernorm_plan",
        )
        node = report.nodes[0]
        self.assertEqual(node.program_name, "StaticAddLayernormRegion")
        self.assertEqual(node.program_version, "v1")
        self.assertEqual(
            node.fused_instruction, "add_layernorm_fused_or_composed_vulkan"
        )
        self.assertEqual(node.instruction_count, 1)
        self.assertEqual(node.residual_input_ssa, 0)
        self.assertEqual(node.addend_input_ssa, 1)
        self.assertEqual(node.residual_output_ssa, 2)
        self.assertEqual(node.normalized_output_ssa, 3)
        self.assertEqual(
            node.context_ownership_outcome,
            "transferred_removed_original_context_attr",
        )
        self.assertTrue(node.direct_transition_only)
        self.assertTrue(node.replay_state_empty)
        self.assertFalse(node.persistent_output_state)
        plan_attrs = _static_add_layernorm_plan_attrs(program)
        self.assertEqual(len(plan_attrs), 1)
        self.assertTrue(
            isinstance(
                getattr(program.graph_module, next(iter(plan_attrs))),
                torch.ScriptObject,
            )
        )
        self.assertFalse(_layernorm_context_attrs(program))
        self.assertFalse(program.graph_module.state_dict())
        lowered = [
            graph_node
            for graph_node in program.census.nodes
            if graph_node.reason == "graph_owned_static_add_layernorm_plan"
        ]
        self.assertEqual(len(lowered), 1)
        self.assertEqual(lowered[0].classification, "lowered_vulkan")
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_static_add_layernorm_region_reuses_dynamic_shapes(self):
        class AddLayerNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(5)

            def forward(self, first, second):
                residual = first + second
                return residual, self.norm(residual)

        torch.manual_seed(37)
        model = AddLayerNorm().eval()
        batch = torch.export.Dim("batch", min=1, max=4)
        sequence = torch.export.Dim("sequence", min=1, max=8)
        program = torch.vulkan.export_and_lower(
            model,
            (torch.randn(2, 3, 5), torch.randn(2, 3, 5)),
            dynamic_shapes=(
                {0: batch, 1: sequence},
                {0: batch, 1: sequence},
            ),
        )
        self.assertEqual(program.execution_mode, "cpp_plan")
        self.assertEqual(program.cpp_plan_report.status, "compiled")
        self.assertEqual(program.cpp_plan_report.plan_version, "v9")
        plan_attrs = _static_add_layernorm_plan_attrs(program)
        self.assertEqual(len(plan_attrs), 1)
        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            for shape in ((1, 8, 5), (4, 1, 5)):
                first = torch.randn(shape)
                second = torch.randn(shape)
                graph_output = tuple(
                    value.cpu() for value in program(first, second)
                )
                for actual, reference in zip(
                    graph_output,
                    model(first, second),
                ):
                    torch.testing.assert_close(
                        actual,
                        reference,
                        rtol=1e-4,
                        atol=1e-4,
                    )
        self.assertEqual(_static_add_layernorm_plan_attrs(program), plan_attrs)
        self.assertEqual(program.static_add_layernorm_regions.lowered_count, 1)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_cpp_graph_plan_rejects_multi_return_index_out_of_range(self):
        context = torch.ops.vulkan_prepack.create_layernorm_context.default(
            torch.ones(4),
            torch.zeros(4),
            1e-5,
        )
        plan = torch.ops.vulkan_prepack.create_graph_add_layernorm_plan.default(
            context,
            [4],
        )
        graph = torch.fx.Graph()
        first = graph.placeholder("first")
        first.meta["val"] = torch.empty(2, 4)
        second = graph.placeholder("second")
        second.meta["val"] = torch.empty(2, 4)
        plan_node = graph.get_attr("_vulkan_static_add_layernorm_plan_test")
        region = graph.call_function(
            torch.ops.vulkan_prepack.run_graph_add_layernorm_plan.default,
            args=(first, second, plan_node),
        )
        selected = graph.call_function(operator.getitem, args=(region, 2))
        graph.output(selected)
        root = torch.nn.Module()
        root._vulkan_static_add_layernorm_plan_test = plan
        graph_module = torch.fx.GraphModule(root, graph)

        compilation = vulkan_graph.compile_vulkan_graph_plan(
            graph_module,
            {
                region.name: "lowered_vulkan",
                selected.name: "graph",
            },
        )

        self.assertIsNone(compilation.plan)
        self.assertEqual(
            compilation.report.reason,
            "multi_return_index_out_of_range:getitem",
        )

    def test_static_add_layernorm_region_keeps_extra_residual_consumers(self):
        class AddLayerNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(4)

            def forward(self, first, second):
                residual = first + second
                return residual, self.norm(residual), -residual

        torch.manual_seed(38)
        model = AddLayerNorm().eval()
        first = torch.randn(2, 3, 4)
        second = torch.randn(2, 3, 4)
        program = torch.vulkan.export_and_lower(model, (first, second))
        graph_output = tuple(value.cpu() for value in program(first, second))
        for actual, reference in zip(graph_output, model(first, second)):
            torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
        self.assertEqual(program.static_add_layernorm_regions.lowered_count, 1)
        self.assertEqual(len(_static_add_layernorm_plan_attrs(program)), 1)

    def test_static_add_layernorm_region_retains_shared_layernorm_context(self):
        class AddAndStandaloneLayerNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(4)

            def forward(self, first, second):
                residual = first + second
                return self.norm(residual), self.norm(second)

        torch.manual_seed(39)
        model = AddAndStandaloneLayerNorm().eval()
        first = torch.randn(2, 3, 4)
        second = torch.randn(2, 3, 4)
        program = torch.vulkan.export_and_lower(model, (first, second))
        graph_output = tuple(value.cpu() for value in program(first, second))
        for actual, reference in zip(graph_output, model(first, second)):
            torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
        self.assertEqual(program.static_add_layernorm_regions.lowered_count, 1)
        self.assertEqual(len(_static_add_layernorm_plan_attrs(program)), 1)
        self.assertEqual(len(_layernorm_context_attrs(program)), 1)
        self.assertEqual(
            program.static_add_layernorm_regions.nodes[0].context_ownership_outcome,
            "shared_context_retained_original_attr",
        )

    def test_static_add_layernorm_region_skips_dynamic_missing_and_forged_contexts(
        self,
    ):
        for kind in ("dynamic", "missing", "forged"):
            graph = torch.fx.Graph()
            first = graph.placeholder("first")
            second = graph.placeholder("second")
            add = graph.call_function(
                torch.ops.aten.add.Tensor, args=(first, second)
            )
            if kind == "dynamic":
                context = graph.placeholder("context")
                root = torch.nn.Module()
            else:
                context = graph.get_attr("_vulkan_layernorm_context_static")
                root = torch.nn.Module()
                setattr(root, "_vulkan_layernorm_context_static", object())
            layernorm = graph.call_function(
                torch.ops.vulkan_prepack.run_layernorm_context.default,
                args=(add, [4], context),
            )
            graph.output(layernorm)
            graph_module = torch.fx.GraphModule(root, graph)
            if kind == "missing":
                delattr(graph_module, "_vulkan_layernorm_context_static")
            report = vulkan_graph.lower_static_add_layernorm_regions(graph_module)
            self.assertEqual(report.candidate_count, 0)
            self.assertEqual(report.lowered_count, 0)
            self.assertEqual(report.rejected_count, 0)
            self.assertEqual(report.skipped_count, 1)
            self.assertEqual(
                report.nodes[0].reason,
                "layernorm_context_not_graph_owned",
            )
            self.assertFalse(_static_add_layernorm_plan_attrs_from_module(graph_module))

    def test_static_add_layernorm_region_skips_nonsemantic_adds(self):
        context = torch.ops.vulkan_prepack.create_layernorm_context.default(
            torch.ones(4), torch.zeros(4), 1e-5
        )
        cases = (
            ("alpha_not_one", torch.ops.aten.add.Tensor, ("first", "second", 2), {}),
            ("non_tensor_add", torch.ops.aten.add.Scalar, ("first", 1), {}),
        )
        expected_reasons = {
            "alpha_not_one": "add_alpha_not_one",
            "non_tensor_add": "input_not_aten_add_tensor",
        }
        for kind, target, argument_names, kwargs in cases:
            graph = torch.fx.Graph()
            first = graph.placeholder("first")
            second = graph.placeholder("second")
            arguments = tuple(
                {"first": first, "second": second}.get(value, value)
                for value in argument_names
            )
            add = graph.call_function(target, args=arguments, kwargs=kwargs)
            context_node = graph.get_attr("_vulkan_layernorm_context_static")
            layernorm = graph.call_function(
                torch.ops.vulkan_prepack.run_layernorm_context.default,
                args=(add, [4], context_node),
            )
            graph.output(layernorm)
            root = torch.nn.Module()
            setattr(root, "_vulkan_layernorm_context_static", context)
            graph_module = torch.fx.GraphModule(root, graph)
            report = vulkan_graph.lower_static_add_layernorm_regions(graph_module)
            self.assertEqual(report.candidate_count, 0)
            self.assertEqual(report.lowered_count, 0)
            self.assertEqual(report.rejected_count, 0)
            self.assertEqual(report.skipped_count, 1)
            self.assertEqual(report.nodes[0].reason, expected_reasons[kind])
            self.assertFalse(_static_add_layernorm_plan_attrs_from_module(graph_module))

    def test_static_add_layernorm_region_skips_multiple_layernorm_consumers(self):
        graph = torch.fx.Graph()
        first = graph.placeholder("first")
        second = graph.placeholder("second")
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(first, second))
        context = graph.get_attr("_vulkan_layernorm_context_static")
        first_layernorm = graph.call_function(
            torch.ops.vulkan_prepack.run_layernorm_context.default,
            args=(add, [4], context),
        )
        second_layernorm = graph.call_function(
            torch.ops.vulkan_prepack.run_layernorm_context.default,
            args=(add, [4], context),
        )
        graph.output((first_layernorm, second_layernorm))
        root = torch.nn.Module()
        setattr(
            root,
            "_vulkan_layernorm_context_static",
            torch.ops.vulkan_prepack.create_layernorm_context.default(
                torch.ones(4), torch.zeros(4), 1e-5
            ),
        )
        graph_module = torch.fx.GraphModule(root, graph)
        report = vulkan_graph.lower_static_add_layernorm_regions(graph_module)
        self.assertEqual(report.candidate_count, 0)
        self.assertEqual(report.lowered_count, 0)
        self.assertEqual(report.rejected_count, 0)
        self.assertEqual(report.skipped_count, 2)
        self.assertTrue(
            all(
                node.reason == "add_has_multiple_layernorm_consumers"
                for node in report.nodes
            )
        )
        self.assertFalse(_static_add_layernorm_plan_attrs_from_module(graph_module))

    def test_forged_or_missing_static_add_layernorm_plan_is_unsupported(self):
        for missing in (False, True):
            graph = torch.fx.Graph()
            first = graph.placeholder("first")
            second = graph.placeholder("second")
            plan = graph.get_attr("_vulkan_static_add_layernorm_plan_forged")
            region = graph.call_function(
                torch.ops.vulkan_prepack.run_graph_add_layernorm_plan.default,
                args=(first, second, plan),
            )
            graph.output(region)
            root = torch.nn.Module()
            setattr(root, "_vulkan_static_add_layernorm_plan_forged", object())
            graph_module = torch.fx.GraphModule(root, graph)
            if missing:
                delattr(graph_module, "_vulkan_static_add_layernorm_plan_forged")
            census = vulkan_graph._build_census(graph_module)
            record = next(
                node
                for node in census.nodes
                if node.target == "vulkan_prepack::run_graph_add_layernorm_plan"
            )
            self.assertEqual(record.classification, "unsupported")
            self.assertEqual(
                record.reason,
                "run_graph_add_layernorm_plan_missing_graph_owned_plan",
            )

    def test_forged_missing_or_nonprivate_graph_region_plan_is_unsupported(self):
        for plan_attr, missing in (
            ("_vulkan_graph_region_plan_forged", False),
            ("_vulkan_graph_region_plan_missing", True),
            ("not_private_graph_region_plan", False),
        ):
            graph = torch.fx.Graph()
            tensor = graph.placeholder("tensor")
            plan = graph.get_attr(plan_attr)
            region = graph.call_function(
                torch.ops.vulkan_prepack.run_vulkan_graph_region_plan.default,
                args=([tensor], plan),
            )
            graph.output(graph.call_function(operator.getitem, (region, 0)))
            root = torch.nn.Module()
            setattr(
                root,
                plan_attr,
                torch.ops.vulkan_prepack.create_layernorm_context.default(
                    torch.ones(4), torch.zeros(4), 1e-5
                ),
            )
            graph_module = torch.fx.GraphModule(root, graph)
            if missing:
                delattr(graph_module, plan_attr)
            census = vulkan_graph._build_census(graph_module)
            record = next(
                node
                for node in census.nodes
                if node.target == "vulkan_prepack::run_vulkan_graph_region_plan"
            )
            self.assertEqual(record.classification, "unsupported")
            self.assertEqual(
                record.reason,
                "run_vulkan_graph_region_plan_missing_graph_owned_plan",
            )
        for operator_name in (
            "vulkan_prepack::create_graph_linear_gelu_plan",
            "vulkan_prepack::run_graph_linear_gelu_plan",
            "vulkan_prepack::create_graph_conv2d_relu_conv2d_plan",
            "vulkan_prepack::run_graph_conv2d_relu_conv2d_plan",
        ):
            with self.assertRaises(RuntimeError):
                torch._C._dispatch_find_schema_or_throw(operator_name, "")

    def test_static_conv2d_relu_conv2d_region_matches_cpu_and_eager_vulkan(self):
        class ConvReluConv(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.conv0 = torch.nn.Conv2d(3, 4, 3, padding=1, bias=bias)
                self.conv1 = torch.nn.Conv2d(4, 2, 3, padding=1, bias=bias)

            def forward(self, tensor):
                return self.conv1(torch.relu(self.conv0(tensor)))

        tensor = torch.ones(2, 3, 8, 7)
        for bias in (False, True):
            model = ConvReluConv(bias).eval()
            with torch.no_grad():
                model.conv0.weight.fill_(0.01)
                model.conv1.weight.fill_(0.01)
                if bias:
                    model.conv0.bias.fill_(0.01)
                    model.conv1.bias.fill_(0.01)
            expected = model(tensor)
            self.assertTrue(torch.any(expected > 0))
            program = torch.vulkan.export_and_lower(model, tensor)
            self.assertTrue(program.cpp_plan_report.submission_owned)
            self.assertTrue(program.cpp_plan.submission_owned())
            eager_vulkan = copy.deepcopy(model).to("vulkan").eval()
            with torch.inference_mode():
                eager_vulkan_output = eager_vulkan(tensor.to("vulkan")).cpu()
            device_tensor = tensor.to("vulkan")
            torch.ops.vulkan_prepack.synchronize()
            torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
            torch.ops.vulkan_prepack.reset_submit_origin_counters()
            graph_output_device = program(device_tensor)
            submit_origins = list(
                torch.ops.vulkan_prepack.submit_origin_counters()
            )
            self.assertEqual(submit_origins[15], 1)
            self.assertEqual(submit_origins[0], 1)
            graph_output = graph_output_device.cpu()
            with torch.inference_mode():
                normal_context_output = torch.relu(tensor.to("vulkan")).cpu()
            torch.testing.assert_close(
                eager_vulkan_output, expected, rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(
                graph_output, expected, rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(
                graph_output, eager_vulkan_output, rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(normal_context_output, torch.relu(tensor))
            self.assertTrue(torch.any(graph_output > 0))
            self.assertEqual(program.conv2d_lowering.lowered_count, 2)
            self.assertEqual(
                program.static_conv2d_relu_conv2d_regions.candidate_count, 1
            )
            self.assertEqual(
                program.static_conv2d_relu_conv2d_regions.lowered_count, 1
            )
            self.assertEqual(
                program.static_conv2d_relu_conv2d_regions.rejected_count, 0
            )
            self.assertEqual(
                program.static_conv2d_relu_conv2d_regions.skipped_count, 0
            )
            node = program.static_conv2d_relu_conv2d_regions.nodes[0]
            self.assertEqual(node.program_name, "VulkanGraphRegionPlan")
            self.assertEqual(node.program_version, "v1")
            self.assertEqual(node.instruction_count, 2)
            self.assertEqual(node.input_ssa, 0)
            self.assertEqual(node.intermediate_ssa, 1)
            self.assertEqual(node.output_ssa, 2)
            self.assertEqual(node.input_use_count, 1)
            self.assertEqual(node.input_last_use, 0)
            self.assertEqual(node.intermediate_use_count, 1)
            self.assertEqual(node.intermediate_last_use, 1)
            self.assertEqual(node.first_static_context_slot, 0)
            self.assertEqual(node.second_static_context_slot, 1)
            self.assertTrue(node.bounded_submission_owned)
            self.assertTrue(node.program_private_scratch)
            self.assertEqual(node.scratch_ring_capacity, 2)
            self.assertTrue(node.timeline_gated_release)
            self.assertTrue(node.direct_transition_only)
            self.assertTrue(node.replay_state_empty)
            self.assertEqual(program.static_conv2d_relu_regions.lowered_count, 0)
            self.assertFalse(_conv2d_context_attrs(program))
            self.assertEqual(
                len(_static_conv2d_relu_conv2d_plan_attrs(program)), 1
            )
            self.assertFalse(program.graph_module.state_dict())
            lowered = [
                node
                for node in program.census.nodes
            if node.reason == "graph_owned_vulkan_graph_region_plan"
            ]
            self.assertEqual(len(lowered), 1)
            self.assertEqual(lowered[0].classification, "lowered_vulkan")
            self.assertEqual(program.last_cpu_fallback_count, 0)
            self.assertEqual(program.last_sync_readback_count, 0)
            self.assertEqual(program.last_deferred_values_created, 0)
            counters = _graph_program_invocation_counters()
            self.assertEqual(len(counters), 14)
            self.assertEqual(counters[0], 1)
            self.assertEqual(counters[1], 1)
            self.assertEqual(counters[2:5], [0, 0, 0])
            self.assertGreaterEqual(counters[5], 1)
            self.assertEqual(counters[6:10], [0, 0, 0, 0])
            self.assertEqual(counters[10:], [0, 0, 0, 0])

    def test_direct_conv2d_relu_conv2d_region_keeps_private_scope(self):
        class ConvReluConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = torch.nn.Conv2d(3, 4, 3, padding=1)
                self.conv1 = torch.nn.Conv2d(4, 2, 3, padding=1)

            def forward(self, tensor):
                return self.conv1(torch.relu(self.conv0(tensor)))

        torch.manual_seed(38)
        model = ConvReluConv().eval()
        tensor = torch.randn(2, 3, 8, 7)
        program = torch.vulkan.export_and_lower(model, tensor)
        plan_attr = next(
            iter(_static_conv2d_relu_conv2d_plan_attrs(program))
        )
        region_plan = getattr(program.graph_module, plan_attr)
        vulkan_tensor = tensor.to("vulkan")
        torch.ops.vulkan_prepack.synchronize()

        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        torch.ops.vulkan_prepack.reset_submit_origin_counters()
        output = torch.ops.vulkan_prepack.run_vulkan_graph_region_plan.default(
            [vulkan_tensor], region_plan
        )[0]
        submit_origins = list(torch.ops.vulkan_prepack.submit_origin_counters())

        torch.testing.assert_close(
            output.cpu(), model(tensor), rtol=1e-4, atol=1e-4
        )
        self.assertEqual(submit_origins[15], 1)
        self.assertEqual(submit_origins[0], 1)
        self.assertEqual(
            _graph_program_invocation_counters(),
            [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        )

    def test_conv2d_region_scratch_inherits_abort_submission(self):
        class ConvReluConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = torch.nn.Conv2d(3, 4, 3, padding=1)
                self.conv1 = torch.nn.Conv2d(4, 2, 3, padding=1)

            def forward(self, tensor):
                return self.conv1(torch.relu(self.conv0(tensor)))

        torch.manual_seed(39)
        model = ConvReluConv().eval()
        tensor = torch.randn(2, 3, 8, 7)
        program = torch.vulkan.export_and_lower(model, tensor)
        plan_attr = next(
            iter(_static_conv2d_relu_conv2d_plan_attrs(program))
        )
        region_plan = getattr(program.graph_module, plan_attr)

        def graph_plan(index):
            return torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
                ["region", "getitem"],
                [
                    "vulkan_prepack::run_vulkan_graph_region_plan",
                    "vulkan_graph::list_getitem",
                ],
                ["", ""],
                [[[0], [-1]], [[1], [-2]]],
                [[1, 0], [0, 0]],
                [[1], [2]],
                [region_plan, index],
                1,
                [2],
            )

        vulkan_tensor = tensor.to("vulkan")
        torch.ops.vulkan_prepack.synchronize()
        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        torch.ops.vulkan_prepack.reset_submit_origin_counters()
        with self.assertRaisesRegex(
            RuntimeError,
            "VulkanGraphPlan.v9 node 'getitem'.*"
            "index 1 is out of range for length 1",
        ):
            torch.ops.vulkan_prepack.run_vulkan_graph_plan.default(
                [vulkan_tensor], graph_plan(1)
            )
        abort_submit_origins = list(
            torch.ops.vulkan_prepack.submit_origin_counters()
        )
        self.assertEqual(abort_submit_origins[15], 1)
        self.assertEqual(abort_submit_origins[0], 1)

        torch.ops.vulkan_prepack.synchronize()
        torch.ops.vulkan_prepack.reset_submit_origin_counters()
        output = torch.ops.vulkan_prepack.run_vulkan_graph_plan.default(
            [vulkan_tensor], graph_plan(0)
        )[0]
        submit_origins = list(torch.ops.vulkan_prepack.submit_origin_counters())

        torch.testing.assert_close(
            output.cpu(), model(tensor), rtol=1e-4, atol=1e-4
        )
        self.assertEqual(submit_origins[15], 1)
        self.assertEqual(submit_origins[0], 1)
        self.assertEqual(
            _graph_program_invocation_counters(),
            [2, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        )

    def test_static_conv2d_relu_conv2d_region_rejects_host_sync_requirement(
        self,
    ):
        class ConvReluConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = torch.nn.Conv2d(3, 4, 3, padding=1)
                self.conv1 = torch.nn.Conv2d(4, 2, 3, padding=1)

            def forward(self, tensor):
                return self.conv1(torch.relu(self.conv0(tensor)))

        torch.manual_seed(37)
        model = ConvReluConv().eval()
        tensor = torch.randn(2, 3, 8, 7)
        expected = model(tensor)
        program = torch.vulkan.export_and_lower(model, tensor)
        self.assertTrue(program.cpp_plan_report.submission_owned)
        self.assertTrue(program.cpp_plan.submission_owned())
        eager_vulkan = copy.deepcopy(model).to("vulkan").eval()

        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        torch.ops.vulkan_prepack.set_graph_program_conv_host_sync_for_testing(True)
        try:
            with self.assertRaisesRegex(
                torch.vulkan.VulkanGraphExecutionError,
                "bounded_submission_host_sync_required",
            ):
                program(tensor)
        finally:
            torch.ops.vulkan_prepack.set_graph_program_conv_host_sync_for_testing(
                False
            )

        self.assertEqual(
            _graph_program_invocation_counters(),
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        graph_output = program(tensor).cpu()
        with torch.inference_mode():
            eager_vulkan_output = eager_vulkan(tensor.to("vulkan")).cpu()
        torch.testing.assert_close(graph_output, expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            graph_output, eager_vulkan_output, rtol=1e-4, atol=1e-4
        )
        counters = _graph_program_invocation_counters()
        self.assertGreaterEqual(counters[0], 1)
        self.assertGreaterEqual(counters[1], 1)
        self.assertEqual(counters[2:5], [0, 0, 0])
        self.assertGreaterEqual(counters[5], 1)

    def test_static_conv2d_relu_conv2d_region_reuses_dynamic_shapes(self):
        class ConvReluConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = torch.nn.Conv2d(3, 4, 3, padding=1)
                self.conv1 = torch.nn.Conv2d(4, 2, 3, padding=1)

            def forward(self, tensor):
                return self.conv1(torch.relu(self.conv0(tensor)))

        torch.manual_seed(19)
        model = ConvReluConv().eval()
        example = torch.randn(2, 3, 6, 7)
        batch = torch.export.Dim("batch", min=1, max=4)
        height = torch.export.Dim("height", min=5, max=12)
        width = torch.export.Dim("width", min=5, max=12)
        program = torch.vulkan.export_and_lower(
            model,
            example,
            dynamic_shapes=({0: batch, 2: height, 3: width},),
        )
        plan_attrs = _static_conv2d_relu_conv2d_plan_attrs(program)
        self.assertEqual(len(plan_attrs), 1)
        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        for shape in ((1, 3, 5, 11), (4, 3, 12, 5), (1, 3, 5, 11)):
            tensor = torch.randn(shape)
            torch.testing.assert_close(
                program(tensor).cpu(), model(tensor), rtol=1e-4, atol=1e-4
            )
        self.assertEqual(_static_conv2d_relu_conv2d_plan_attrs(program), plan_attrs)
        self.assertEqual(
            program.static_conv2d_relu_conv2d_regions.lowered_count, 1
        )
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)
        self.assertEqual(
            _graph_program_invocation_counters(),
            [3, 3, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0],
        )

    def test_static_conv2d_relu_conv2d_region_allows_two_unread_outputs(self):
        class ConvReluConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = torch.nn.Conv2d(3, 4, 3, padding=1)
                self.conv1 = torch.nn.Conv2d(4, 2, 3, padding=1)

            def forward(self, tensor):
                return self.conv1(torch.relu(self.conv0(tensor)))

        torch.manual_seed(23)
        model = ConvReluConv().eval()
        tensor = torch.randn(2, 3, 9, 11)
        expected = model(tensor)
        program = torch.vulkan.export_and_lower(model, tensor)
        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        first_output = program(tensor)
        second_output = program(tensor)
        torch.testing.assert_close(
            first_output.cpu(), expected, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            second_output.cpu(), expected, rtol=1e-4, atol=1e-4
        )
        counters = _graph_program_invocation_counters()
        self.assertEqual(counters[:5], [2, 2, 0, 0, 0])
        self.assertEqual(sum(counters[5:8]), 2)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_static_conv2d_relu_conv2d_region_releases_or_retires_scratch(self):
        class ConvReluConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = torch.nn.Conv2d(3, 24, 3, padding=1)
                self.conv1 = torch.nn.Conv2d(24, 12, 3, padding=1)

            def forward(self, tensor):
                return self.conv1(torch.relu(self.conv0(tensor)))

        torch.manual_seed(29)
        model = ConvReluConv().eval()
        tensor = torch.randn(2, 3, 256, 257)
        expected = model(tensor)
        program = torch.vulkan.export_and_lower(model, tensor)
        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        first_output = program(tensor)
        second_output = program(tensor)
        del program
        gc.collect()
        torch.testing.assert_close(
            first_output.cpu(), expected, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            second_output.cpu(), expected, rtol=1e-4, atol=1e-4
        )
        counters = _graph_program_invocation_counters()
        self.assertGreaterEqual(counters[5] + counters[6], 1)
        self.assertGreaterEqual(counters[8] + counters[9], 1)
        if counters[8] == 0:
            self.assertGreaterEqual(counters[9], 1)

    def test_static_conv2d_relu_conv2d_region_releases_completed_scratch(self):
        class ConvReluConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = torch.nn.Conv2d(3, 4, 3, padding=1)
                self.conv1 = torch.nn.Conv2d(4, 2, 3, padding=1)

            def forward(self, tensor):
                return self.conv1(torch.relu(self.conv0(tensor)))

        torch.manual_seed(31)
        model = ConvReluConv().eval()
        tensor = torch.randn(2, 3, 11, 13)
        expected = model(tensor)
        program = torch.vulkan.export_and_lower(model, tensor)
        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        output = program(tensor)
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-4, atol=1e-4)
        del program
        gc.collect()
        counters = _graph_program_invocation_counters()
        self.assertGreaterEqual(counters[5], 1)
        self.assertEqual(counters[8], 0)
        self.assertGreaterEqual(counters[9], 1)

    def test_static_conv2d_lowering_matches_cpu_and_releases_weights(self):
        class ConvRelu(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 3, padding=1, bias=bias)

            def forward(self, tensor):
                return torch.relu(self.conv(tensor))

        torch.manual_seed(7)
        tensor = torch.randn(2, 3, 8, 7)
        for bias in (False, True):
            model = ConvRelu(bias).eval()
            program = torch.vulkan.export_and_lower(model, tensor)
            eager_vulkan = copy.deepcopy(model).to("vulkan").eval()
            with torch.inference_mode():
                eager_vulkan_output = eager_vulkan(tensor.to("vulkan")).cpu()
            graph_output = program(tensor).cpu()
            expected = model(tensor)
            self.assertTrue(torch.any(expected > 0))
            torch.testing.assert_close(
                graph_output, expected, rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(
                eager_vulkan_output, expected, rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(
                graph_output, eager_vulkan_output, rtol=1e-4, atol=1e-4
            )
            self.assertTrue(torch.any(graph_output > 0))
            self.assertEqual(program.conv2d_lowering.conv2d_node_count, 1)
            self.assertEqual(program.conv2d_lowering.lowered_count, 1)
            self.assertEqual(program.conv2d_lowering.rejected_count, 0)
            self.assertEqual(program.conv2d_lowering.created_context_count, 1)
            self.assertEqual(program.conv2d_lowering.reused_context_count, 0)
            self.assertEqual(
                program.conv2d_lowering.context_factory,
                "vulkan_prepack::create_graph_conv2d_context",
            )
            report = program.static_conv2d_relu_regions
            self.assertEqual(report.candidate_count, 1)
            self.assertEqual(report.lowered_count, 1)
            self.assertEqual(report.rejected_count, 0)
            self.assertEqual(report.skipped_count, 0)
            self.assertEqual(
                report.plan_factory,
                "vulkan_prepack::create_graph_conv2d_relu_plan",
            )
            self.assertEqual(
                program.static_conv2d_relu_conv2d_regions.lowered_count, 0
            )
            self.assertEqual(
                program.static_conv2d_relu_conv2d_regions.excluded_relu_node_names,
                (),
            )
            node = report.nodes[0]
            self.assertEqual(node.program_name, "StaticConv2dReluRegion")
            self.assertEqual(node.program_version, "v1")
            self.assertEqual(node.instruction_count, 1)
            self.assertEqual(node.input_ssa, 0)
            self.assertEqual(node.output_ssa, 1)
            self.assertEqual(node.input_use_count, 1)
            self.assertEqual(node.input_last_use, 0)
            self.assertEqual(node.static_context_slot, 0)
            self.assertTrue(node.direct_transition_only)
            self.assertTrue(node.replay_state_empty)
            self.assertFalse(_conv2d_context_attrs(program))
            self.assertEqual(len(_static_conv2d_relu_plan_attrs(program)), 1)
            self.assertFalse(program.graph_module.state_dict())
            lowered = [
                node
                for node in program.census.nodes
                if node.reason == "graph_owned_static_conv2d_relu_plan"
            ]
            self.assertEqual(len(lowered), 1)
            self.assertEqual(lowered[0].classification, "lowered_vulkan")
            self.assertEqual(program.last_cpu_fallback_count, 0)
            self.assertEqual(program.last_sync_readback_count, 0)
            self.assertEqual(program.last_deferred_values_created, 0)

    def test_static_conv2d_relu_region_reuses_dynamic_shapes(self):
        class ConvRelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

            def forward(self, tensor):
                return torch.relu(self.conv(tensor))

        torch.manual_seed(16)
        model = ConvRelu().eval()
        example = torch.randn(2, 3, 6, 7)
        batch = torch.export.Dim("batch", min=1, max=4)
        height = torch.export.Dim("height", min=5, max=12)
        width = torch.export.Dim("width", min=5, max=12)
        program = torch.vulkan.export_and_lower(
            model,
            example,
            dynamic_shapes=({0: batch, 2: height, 3: width},),
        )
        plan_attrs = _static_conv2d_relu_plan_attrs(program)
        self.assertEqual(len(plan_attrs), 1)
        for shape in ((1, 3, 5, 11), (4, 3, 12, 5)):
            tensor = torch.randn(shape)
            torch.testing.assert_close(
                program(tensor).cpu(), model(tensor), rtol=1e-4, atol=1e-4
            )
        self.assertEqual(_static_conv2d_relu_plan_attrs(program), plan_attrs)
        self.assertEqual(program.static_conv2d_relu_regions.lowered_count, 1)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_static_conv2d_relu_conv2d_shared_context_stays_unfused(self):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        first_context = graph.get_attr("_vulkan_conv2d_context_first")
        second_context = graph.get_attr("_vulkan_conv2d_context_second")
        first = graph.call_function(
            torch.ops.vulkan_prepack.run_conv2d_context.default,
            args=(tensor, first_context),
        )
        relu = graph.call_function(torch.ops.aten.relu.default, args=(first,))
        second = graph.call_function(
            torch.ops.vulkan_prepack.run_conv2d_context.default,
            args=(relu, second_context),
        )
        graph.output(second)
        root = torch.nn.Module()
        context = object()
        setattr(root, "_vulkan_conv2d_context_first", context)
        setattr(root, "_vulkan_conv2d_context_second", context)
        graph_module = torch.fx.GraphModule(root, graph)

        report = vulkan_graph.lower_static_conv2d_relu_conv2d_regions(graph_module)
        self.assertEqual(report.candidate_count, 0)
        self.assertEqual(report.lowered_count, 0)
        self.assertEqual(report.skipped_count, 1)
        self.assertEqual(report.nodes[0].reason, "conv2d_contexts_must_be_distinct")
        self.assertFalse(
            _static_conv2d_relu_conv2d_plan_attrs_from_module(graph_module)
        )

    def test_static_conv2d_relu_conv2d_multi_use_stays_unfused(self):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        first_context = graph.get_attr("_vulkan_conv2d_context_first")
        second_context = graph.get_attr("_vulkan_conv2d_context_second")
        first = graph.call_function(
            torch.ops.vulkan_prepack.run_conv2d_context.default,
            args=(tensor, first_context),
        )
        relu = graph.call_function(torch.ops.aten.relu.default, args=(first,))
        second = graph.call_function(
            torch.ops.vulkan_prepack.run_conv2d_context.default,
            args=(relu, second_context),
        )
        extra = graph.call_function(torch.ops.aten.neg.default, args=(relu,))
        graph.output((second, extra))
        root = torch.nn.Module()
        setattr(root, "_vulkan_conv2d_context_first", object())
        setattr(root, "_vulkan_conv2d_context_second", object())
        graph_module = torch.fx.GraphModule(root, graph)

        report = vulkan_graph.lower_static_conv2d_relu_conv2d_regions(graph_module)
        self.assertEqual(report.candidate_count, 0)
        self.assertEqual(report.lowered_count, 0)
        self.assertEqual(report.skipped_count, 1)
        self.assertEqual(report.nodes[0].reason, "relu_output_has_multiple_users")
        self.assertEqual(
            report.excluded_relu_node_names,
            (report.nodes[0].relu_node_name,),
        )
        self.assertFalse(
            _static_conv2d_relu_conv2d_plan_attrs_from_module(graph_module)
        )

    def test_static_conv2d_relu_conv2d_ineligible_chain_stays_unfused(self):
        class ConvReluConvWithExtraReluUse(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = torch.nn.Conv2d(3, 4, 3, padding=1)
                self.conv1 = torch.nn.Conv2d(4, 2, 3, padding=1)

            def forward(self, tensor):
                relu = torch.relu(self.conv0(tensor))
                return self.conv1(relu), -relu

        torch.manual_seed(31)
        model = ConvReluConvWithExtraReluUse().eval()
        tensor = torch.randn(2, 3, 8, 7)
        program = torch.vulkan.export_and_lower(model, tensor)
        eager_vulkan = copy.deepcopy(model).to("vulkan").eval()
        with torch.inference_mode():
            eager_vulkan_output = eager_vulkan(tensor.to("vulkan"))
        graph_output = program(tensor)
        expected = model(tensor)
        for actual, reference, eager in zip(
            graph_output, expected, eager_vulkan_output
        ):
            torch.testing.assert_close(
                actual.cpu(), reference, rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(
                actual.cpu(), eager.cpu(), rtol=1e-4, atol=1e-4
            )
        multi_report = program.static_conv2d_relu_conv2d_regions
        single_report = program.static_conv2d_relu_regions
        self.assertEqual(multi_report.lowered_count, 0)
        self.assertEqual(multi_report.skipped_count, 1)
        self.assertEqual(
            multi_report.nodes[0].reason,
            "relu_output_has_multiple_users",
        )
        self.assertEqual(
            multi_report.excluded_relu_node_names,
            (multi_report.nodes[0].relu_node_name,),
        )
        self.assertEqual(single_report.lowered_count, 0)
        self.assertEqual(single_report.skipped_count, 1)
        self.assertEqual(
            single_report.nodes[0].reason,
            "excluded_by_static_conv2d_relu_conv2d_region",
        )
        self.assertFalse(_static_conv2d_relu_conv2d_plan_attrs(program))
        self.assertFalse(_static_conv2d_relu_plan_attrs(program))
        self.assertEqual(len(_conv2d_context_attrs(program)), 2)

    def test_static_conv2d_relu_conv2d_nonsemantic_paths_stay_unfused(self):
        cases = (
            (torch.ops.aten.relu_.default, "inplace"),
            (torch.ops.aten.relu.default, "dynamic_context"),
            (torch.ops.aten.relu.default, "nonprivate_context"),
            (torch.ops.aten.relu.default, "malformed_context_call"),
        )
        for relu_op, kind in cases:
            graph = torch.fx.Graph()
            tensor = graph.placeholder("tensor")
            first_context = graph.get_attr("_vulkan_conv2d_context_first")
            first = graph.call_function(
                torch.ops.vulkan_prepack.run_conv2d_context.default,
                args=(tensor, first_context),
            )
            relu = graph.call_function(relu_op, args=(first,))
            if kind == "dynamic_context":
                second_context = graph.placeholder("context")
                second_args = (relu, second_context)
            elif kind == "nonprivate_context":
                second_context = graph.get_attr("context")
                second_args = (relu, second_context)
            elif kind == "malformed_context_call":
                second_context = graph.get_attr("_vulkan_conv2d_context_second")
                second_args = (relu, second_context, tensor)
            else:
                second_context = graph.get_attr("_vulkan_conv2d_context_second")
                second_args = (relu, second_context)
            second = graph.call_function(
                torch.ops.vulkan_prepack.run_conv2d_context.default,
                args=second_args,
            )
            graph.output(second)
            root = torch.nn.Module()
            setattr(root, "_vulkan_conv2d_context_first", object())
            if kind == "nonprivate_context":
                root.context = object()
            else:
                setattr(root, "_vulkan_conv2d_context_second", object())
            graph_module = torch.fx.GraphModule(root, graph)

            report = vulkan_graph.lower_static_conv2d_relu_conv2d_regions(
                graph_module
            )
            self.assertEqual(report.lowered_count, 0)
            self.assertFalse(
                _static_conv2d_relu_conv2d_plan_attrs_from_module(graph_module)
            )
            if kind == "inplace":
                self.assertEqual(report.candidate_count, 0)
                self.assertEqual(report.skipped_count, 0)
            else:
                self.assertEqual(report.candidate_count, 0)
                self.assertEqual(report.skipped_count, 1)
                self.assertEqual(
                    report.nodes[0].reason,
                    "relu_output_not_graph_owned_conv2d",
                )

    def test_static_conv2d_relu_multi_use_output_stays_unfused(self):
        class ConvReluResidual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

            def forward(self, tensor):
                conv = self.conv(tensor)
                return torch.relu(conv) + conv

        torch.manual_seed(17)
        model = ConvReluResidual().eval()
        tensor = torch.randn(1, 3, 6, 5)
        program = torch.vulkan.export_and_lower(model, tensor)
        torch.testing.assert_close(
            program(tensor).cpu(), model(tensor), rtol=1e-4, atol=1e-4
        )
        report = program.static_conv2d_relu_regions
        self.assertEqual(report.lowered_count, 0)
        self.assertEqual(report.skipped_count, 1)
        self.assertEqual(report.nodes[0].reason, "conv2d_output_has_multiple_users")
        self.assertFalse(_static_conv2d_relu_plan_attrs(program))
        self.assertEqual(len(_conv2d_context_attrs(program)), 1)

    def test_static_conv2d_relu_tied_context_stays_unfused(self):
        class TiedConvRelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

            def forward(self, tensor):
                first = torch.relu(self.conv(tensor))
                second = torch.relu(self.conv(tensor))
                return first + second

        torch.manual_seed(18)
        model = TiedConvRelu().eval()
        tensor = torch.randn(1, 3, 6, 5)
        program = torch.vulkan.export_and_lower(model, tensor)
        torch.testing.assert_close(
            program(tensor).cpu(), model(tensor), rtol=1e-4, atol=1e-4
        )
        report = program.static_conv2d_relu_regions
        self.assertEqual(report.lowered_count, 0)
        self.assertEqual(report.skipped_count, 2)
        self.assertTrue(
            all(
                node.reason == "context_attr_has_multiple_references"
                for node in report.nodes
            )
        )
        self.assertFalse(_static_conv2d_relu_plan_attrs(program))
        self.assertEqual(len(_conv2d_context_attrs(program)), 1)

    def test_static_conv2d_relu_inplace_stays_unfused(self):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        context = graph.get_attr("_vulkan_conv2d_context_fake")
        conv2d = graph.call_function(
            torch.ops.vulkan_prepack.run_conv2d_context.default,
            args=(tensor, context),
        )
        relu = graph.call_function(torch.ops.aten.relu_.default, args=(conv2d,))
        graph.output(relu)
        root = torch.nn.Module()
        setattr(root, "_vulkan_conv2d_context_fake", object())
        graph_module = torch.fx.GraphModule(root, graph)

        report = vulkan_graph.lower_static_conv2d_relu_regions(graph_module)
        self.assertEqual(report.candidate_count, 0)
        self.assertEqual(report.lowered_count, 0)
        self.assertFalse(_static_conv2d_relu_plan_attrs_from_module(graph_module))

    def test_static_conv2d_relu_nonprivate_and_dynamic_contexts_stay_unfused(self):
        for context_op, context_target in (
            ("get_attr", "context"),
            ("placeholder", "context"),
        ):
            graph = torch.fx.Graph()
            tensor = graph.placeholder("tensor")
            context = getattr(graph, context_op)(context_target)
            conv2d = graph.call_function(
                torch.ops.vulkan_prepack.run_conv2d_context.default,
                args=(tensor, context),
            )
            relu = graph.call_function(
                torch.ops.aten.relu.default, args=(conv2d,))
            graph.output(relu)
            root = torch.nn.Module()
            if context_op == "get_attr":
                root.context = object()
            graph_module = torch.fx.GraphModule(root, graph)

            report = vulkan_graph.lower_static_conv2d_relu_regions(graph_module)
            self.assertEqual(report.candidate_count, 0)
            self.assertEqual(report.lowered_count, 0)
            self.assertFalse(_static_conv2d_relu_plan_attrs_from_module(graph_module))

    def test_static_conv2d_relu_unsupported_signature_stays_unfused(self):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        context = graph.get_attr("_vulkan_conv2d_context_fake")
        conv2d = graph.call_function(
            torch.ops.vulkan_prepack.run_conv2d_context.default,
            args=(tensor, context, tensor),
        )
        relu = graph.call_function(torch.ops.aten.relu.default, args=(conv2d,))
        graph.output(relu)
        root = torch.nn.Module()
        setattr(root, "_vulkan_conv2d_context_fake", object())
        graph_module = torch.fx.GraphModule(root, graph)

        report = vulkan_graph.lower_static_conv2d_relu_regions(graph_module)
        self.assertEqual(report.candidate_count, 0)
        self.assertEqual(report.lowered_count, 0)
        self.assertFalse(_static_conv2d_relu_plan_attrs_from_module(graph_module))

    def test_reuses_tied_conv2d_context(self):
        class SharedConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 3, 3, 3))
                self.bias = torch.nn.Parameter(torch.randn(4))

            def forward(self, tensor):
                first = torch.nn.functional.conv2d(
                    tensor, self.weight, self.bias, padding=1
                )
                second = torch.nn.functional.conv2d(
                    tensor, self.weight, self.bias, padding=1
                )
                return first + second

        torch.manual_seed(8)
        model = SharedConv().eval()
        tensor = torch.randn(1, 3, 6, 5)
        program = torch.vulkan.export_and_lower(model, tensor)
        self.assertEqual(program(tensor).cpu(), model(tensor))
        self.assertEqual(program.conv2d_lowering.conv2d_node_count, 2)
        self.assertEqual(program.conv2d_lowering.lowered_count, 2)
        self.assertEqual(program.conv2d_lowering.created_context_count, 1)
        self.assertEqual(program.conv2d_lowering.reused_context_count, 1)
        self.assertEqual(len(_conv2d_context_attrs(program)), 1)

    def test_rejects_dynamic_conv2d_weight_before_execution(self):
        class DynamicWeightConv2d(torch.nn.Module):
            def forward(self, tensor, weight):
                return torch.nn.functional.conv2d(tensor, weight)

        with self.assertRaisesRegex(
            torch.vulkan.VulkanGraphExecutionError,
            "weight_not_static_get_attr",
        ):
            torch.vulkan.export_and_lower(
                DynamicWeightConv2d().eval(),
                (torch.randn(1, 3, 6, 6), torch.randn(4, 3, 3, 3)),
            )

    def test_rejects_invalid_static_conv2d_parameter(self):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        tensor.meta["val"] = torch.empty(1, 3, 6, 6)
        weight = graph.get_attr("weight")
        conv2d = graph.call_function(
            torch.ops.aten.conv2d.default,
            args=(tensor, weight, None, [0, 1], [0, 0], [1, 1], 1),
        )
        graph.output(conv2d)
        graph_module = torch.fx.GraphModule(
            {"weight": torch.randn(4, 3, 3, 3)}, graph
        )
        report = vulkan_graph.lower_static_conv2d_to_vulkan_contexts(
            graph_module,
            {"weight": torch.randn(4, 3, 3, 3)},
        )
        self.assertEqual(report.conv2d_node_count, 1)
        self.assertEqual(report.lowered_count, 0)
        self.assertEqual(report.rejected_count, 1)
        self.assertEqual(report.nodes[0].reason, "stride_outside_supported_range")

    def test_changed_conv2d_state_uses_a_distinct_program_key_and_context(self):
        class Conv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

            def forward(self, tensor):
                return self.conv(tensor)

        torch.manual_seed(9)
        model = Conv().eval()
        tensor = torch.randn(1, 3, 6, 6)
        expected_first = model(tensor)
        first = torch.vulkan.export_and_lower(model, tensor)
        with torch.no_grad():
            model.conv.weight.add_(1.0)
        expected_second = model(tensor)
        second = torch.vulkan.export_and_lower(model, tensor)
        first_attr = next(iter(_conv2d_context_attrs(first)))
        second_attr = next(iter(_conv2d_context_attrs(second)))
        self.assertNotEqual(first.key.state_fingerprint, second.key.state_fingerprint)
        self.assertIsNot(
            getattr(first.graph_module, first_attr),
            getattr(second.graph_module, second_attr),
        )
        self.assertEqual(first(tensor).cpu(), expected_first)
        self.assertEqual(second(tensor).cpu(), expected_second)

    def test_export_and_lower_matches_cpu_and_preserves_model(self):
        class LinearRelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)

            def forward(self, tensor):
                return torch.relu(self.linear(tensor))

        torch.manual_seed(0)
        model = LinearRelu().eval()
        tensor = torch.randn(2, 3)
        expected = model(tensor)

        program = torch.vulkan.export_and_lower(model, tensor)
        actual = program(tensor)

        self.assertEqual(actual.cpu(), expected)
        self.assertEqual(actual.device.type, "vulkan")
        self.assertEqual(next(model.parameters()).device.type, "cpu")
        self.assertEqual(program.run_count, 1)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)
        self.assertGreater(len(program.last_executed_nodes), 0)
        self.assertEqual(program.census.unsupported_node_count, 0)
        self.assertGreater(program.census.direct_vulkan_node_count, 0)
        self.assertEqual(program.census.lowered_vulkan_node_count, 1)
        self.assertEqual(program.linear_lowering.linear_node_count, 1)
        self.assertEqual(program.linear_lowering.lowered_count, 1)
        self.assertEqual(program.linear_lowering.rejected_count, 0)
        self.assertEqual(program.linear_lowering.created_context_count, 1)
        self.assertEqual(program.linear_lowering.reused_context_count, 0)
        self.assertEqual(
            program.linear_lowering.context_factory,
            "vulkan_prepack::create_graph_linear_context",
        )
        self.assertEqual(len(program.key.state_fingerprint), 64)
        self.assertNotIn("linear.weight", program.graph_module.state_dict())
        self.assertNotIn("linear.bias", program.graph_module.state_dict())
        self.assertFalse(hasattr(program, "exported_program"))
        self.assertFalse(hasattr(program, "_exported_program"))

    def test_inference_grad_wrapper_is_inlined_before_vulkan_lowering(self):
        class NoGradPointwise(torch.nn.Module):
            @torch.no_grad()
            def forward(self, tensor):
                return torch.sin(tensor) + 1

        model = NoGradPointwise().eval()
        tensor = torch.randn(2, 3)
        program = torch.vulkan.export_and_lower(model, tensor)
        self.assertEqual(program.execution_mode, "cpp_plan")
        self.assertEqual(program.cpp_plan_report.status, "compiled")
        self.assertFalse(
            any(
                node.target is torch.ops.higher_order.wrap_with_set_grad_enabled
                for node in program.graph_module.graph.nodes
            )
        )
        self.assertEqual(program(tensor).cpu(), model(tensor))
        self.assertEqual(program.census.unsupported_node_count, 0)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)

    def test_static_contexts_replace_raw_weights_before_state_device_move(self):
        class EmbeddingProjection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 4)
                self.projection = torch.nn.Linear(4, 3)

            def forward(self, indices):
                return self.projection(self.embedding(indices))

        model = EmbeddingProjection().eval()
        indices = torch.tensor([[1, 3]], dtype=torch.long)
        expected = model(indices)
        program = torch.vulkan.export_and_lower(model, indices)
        state = program.graph_module.state_dict()
        self.assertEqual(set(state), {"embedding.weight"})
        self.assertEqual(state["embedding.weight"].device.type, "vulkan")
        self.assertEqual(model.embedding.weight.device.type, "cpu")
        self.assertEqual(model.projection.weight.device.type, "cpu")
        self.assertEqual(program(indices).cpu(), expected)

    def test_static_arange_expression_becomes_graph_owned_constant(self):
        class StaticArange(torch.nn.Module):
            def forward(self, tensor):
                positions = torch.arange(tensor.shape[0], device=tensor.device) + 0
                keys = positions.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                queries = positions.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                return keys <= queries

        model = StaticArange().eval()
        tensor = torch.randn(4, 2)
        expected = model(tensor)
        program = torch.vulkan.export_and_lower(model, tensor)
        report = program.static_factory_constants
        self.assertGreater(report.candidate_count, 2)
        self.assertEqual(report.lowered_count, report.candidate_count)
        self.assertEqual(report.rejected_count, 0)
        self.assertEqual(report.skipped_count, 0)
        self.assertGreater(report.created_constant_count, 1)
        self.assertGreater(report.reused_constant_count, 0)
        constant_attr = report.nodes[-1].constant_attr
        self.assertIsNotNone(constant_attr)
        self.assertEqual(getattr(program.graph_module, constant_attr).device.type, "vulkan")
        self.assertNotIn(constant_attr, program.graph_module.state_dict())
        self.assertFalse(
            any(
                node.op == "call_function"
                and (
                    vulkan_graph_lowering._is_static_arange_target(node.target)
                    or (
                        isinstance(node.target, torch._ops.OpOverload)
                        and node.target.name()
                        in vulkan_graph_lowering._STATIC_FACTORY_EXPRESSION_OPERATOR_NAMES
                    )
                )
                for node in program.graph_module.graph.nodes
            )
        )
        self.assertEqual(program(tensor).cpu(), expected)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)

    def test_lifted_tensor_literals_become_graph_owned_constants(self):
        class EmptyLiterals(torch.nn.Module):
            def forward(self, tensor):
                del tensor
                left = torch.tensor([], dtype=torch.float32)
                right = torch.tensor([], dtype=torch.float32)
                return left, right

        tensor = torch.randn(4)
        model = EmptyLiterals().eval()
        program = torch.vulkan.export_and_lower(model, tensor)
        report = program.lifted_tensor_constants
        self.assertEqual(report.candidate_count, 2)
        self.assertEqual(report.lowered_count, 2)
        self.assertEqual(report.rejected_count, 0)
        self.assertEqual(report.created_constant_count, 1)
        self.assertEqual(report.reused_constant_count, 1)
        constant_attr = report.nodes[0].constant_attr
        self.assertIsNotNone(constant_attr)
        self.assertEqual(getattr(program.graph_module, constant_attr).device.type, "vulkan")
        self.assertNotIn(constant_attr, program.graph_module.state_dict())
        actual = program(tensor)
        expected = model(tensor)
        self.assertEqual(tuple(value.cpu() for value in actual), expected)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)

    def test_lifted_bool_literal_uses_direct_buffer_placement(self):
        class BoolLiteral(torch.nn.Module):
            def forward(self, tensor):
                del tensor
                return torch.tensor([True, False, True, False])

        tensor = torch.randn(1)
        model = BoolLiteral().eval()
        program = torch.vulkan.export_and_lower(model, tensor)
        report = program.lifted_tensor_constants
        self.assertEqual(report.lowered_count, 1)
        self.assertTrue(program.cpp_plan_report.submission_owned)
        self.assertTrue(program.cpp_plan.submission_owned())
        constant_attr = report.nodes[0].constant_attr
        self.assertIsNotNone(constant_attr)
        self.assertEqual(
            program.tensor_placement.buffer_constant_attrs,
            (constant_attr,),
        )
        torch.ops.vulkan_prepack.reset_graph_program_invocation_counters()
        first = program(tensor)
        second = program(tensor)
        self.assertEqual(first.cpu(), model(tensor))
        self.assertEqual(second.cpu(), model(tensor))
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.cpp_plan.invocation_generation(), 2)
        self.assertGreater(program.cpp_plan.last_submission_value(), 0)
        self.assertTrue(program.cpp_plan.last_submission_complete())
        self.assertEqual(
            _graph_program_invocation_counters(),
            [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )

    def test_static_causal_mask_and_runtime_attention_mask_stay_on_device(self):
        class CausalAttentionMask(torch.nn.Module):
            def forward(self, attention_mask):
                batches = torch.arange(
                    attention_mask.shape[0],
                    device=attention_mask.device,
                )
                positions = torch.arange(
                    attention_mask.shape[-1],
                    device=attention_mask.device,
                )
                batch_indices = (
                    batches.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                )
                keys = positions.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                queries = positions.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                causal = keys <= queries
                runtime_mask = attention_mask.to(
                    device=attention_mask.device,
                    dtype=torch.bool,
                )
                runtime = runtime_mask[batch_indices, keys]
                return causal & runtime

        attention_mask = torch.tensor(
            [[1, 1, 1, 0], [1, 0, 1, 1]],
            dtype=torch.int64,
        )
        model = CausalAttentionMask().eval()
        program = torch.vulkan.export_and_lower(model, attention_mask)
        self.assertEqual(program.static_factory_constants.rejected_count, 0)
        self.assertGreater(program.static_factory_constants.lowered_count, 0)
        self.assertEqual(program.input_normalization.lowered_count, 1)
        index_report = program.static_identity_advanced_indices
        self.assertEqual(index_report.candidate_count, 1)
        self.assertEqual(index_report.lowered_count, 1)
        self.assertEqual(index_report.rejected_count, 0)
        self.assertEqual(index_report.skipped_count, 0)
        self.assertEqual(
            index_report.nodes[0].reason,
            "static_full_rank_identity_advanced_index",
        )
        self.assertNotIn("aten.index.Tensor", program.graph_module.code)
        self.assertEqual(
            program.tensor_placement.buffer_placeholder_names,
            ("attention_mask",),
        )
        self.assertEqual(program(attention_mask).cpu(), model(attention_mask))
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)

    def test_unaligned_causal_mask_broadcast_stays_on_device(self):
        class CausalAttentionMask(torch.nn.Module):
            def forward(self, attention_mask):
                positions = torch.arange(
                    attention_mask.shape[-1],
                    device=attention_mask.device,
                )
                keys = positions.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                queries = positions.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                causal = torch.tensor(True, device=attention_mask.device) & (
                    keys <= queries
                )
                runtime = attention_mask.view(1, 1, 1, -1)
                return causal & runtime

        attention_mask = torch.tensor(
            [[True, True, True, False, True]], dtype=torch.bool
        )
        model = CausalAttentionMask().eval()
        program = torch.vulkan.export_and_lower(model, attention_mask)

        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            first = program(attention_mask)
            second = program(attention_mask)

        self.assertEqual(first.cpu(), model(attention_mask))
        self.assertEqual(second.cpu(), model(attention_mask))
        self.assertEqual(program.execution_mode, "cpp_plan")
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_bool_graph_input_accepts_program_device_tensor(self):
        class BoolAnd(torch.nn.Module):
            def forward(self, mask):
                return mask & mask

        mask = torch.tensor(
            [[True, False, True, False]],
            dtype=torch.bool,
        )
        model = BoolAnd().eval()
        program = torch.vulkan.export_and_lower(model, mask)
        mask_vulkan = torch.ops.vulkan_prepack.upload_graph_tensor_to_buffer(
            mask, torch.device("vulkan")
        )
        self.assertEqual(program(mask_vulkan).cpu(), model(mask))
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)

    def test_boolean_masked_sdpa_stays_on_device(self):
        class BooleanMaskedAttention(torch.nn.Module):
            def forward(self, query, key, value, mask):
                return torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=0.0883883,
                )

        torch.manual_seed(0)
        query = torch.randn(1, 16, 4, 128)
        key = torch.randn(1, 16, 4, 128)
        value = torch.randn(1, 16, 4, 128)
        mask = torch.tril(torch.ones(1, 1, 4, 4, dtype=torch.bool))
        model = BooleanMaskedAttention().eval()
        expected = model(query, key, value, mask)
        program = torch.vulkan.export_and_lower(model, (query, key, value, mask))

        actual = program(query, key, value, mask).cpu()

        self.assertEqual(actual, expected, rtol=1e-3, atol=1e-3)
        self.assertEqual(program.census.unsupported_node_count, 0)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)

    def test_rejects_static_advanced_index_that_reorders_values(self):
        class ReverseMask(torch.nn.Module):
            def forward(self, attention_mask):
                batches = torch.arange(
                    attention_mask.shape[0],
                    device=attention_mask.device,
                )
                positions = torch.arange(
                    attention_mask.shape[-1] - 1,
                    -1,
                    -1,
                    device=attention_mask.device,
                )
                batch_indices = (
                    batches.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                )
                key_indices = positions.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                runtime_mask = attention_mask.to(torch.bool)
                return runtime_mask[batch_indices, key_indices]

        attention_mask = torch.tensor([[1, 0, 1, 1]], dtype=torch.int64)
        program = torch.vulkan.export_and_lower(ReverseMask().eval(), attention_mask)
        report = program.static_identity_advanced_indices
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.lowered_count, 0)
        self.assertEqual(report.rejected_count, 1)
        self.assertEqual(report.skipped_count, 0)
        self.assertEqual(
            report.nodes[0].reason,
            "static_index_is_not_identity_order",
        )
        self.assertIn("aten.index.Tensor", program.graph_module.code)

    def test_static_gqa_repeat_lowers_to_generic_kernel_family(self):
        class RepeatHeads(torch.nn.Module):
            def forward(self, tensor):
                batch, heads, tokens, width = tensor.shape
                return tensor.unsqueeze(2).expand(
                    batch,
                    heads,
                    3,
                    tokens,
                    width,
                ).reshape(batch, heads * 3, tokens, width)

        tensor = torch.randn(2, 3, 5, 7)
        model = RepeatHeads().eval()
        program = torch.vulkan.export_and_lower(model, tensor)
        report = program.static_gqa_repeats
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.lowered_count, 1)
        self.assertEqual(report.rejected_count, 0)
        self.assertEqual(report.nodes[0].repeat_factor, 3)
        self.assertEqual(
            report.nodes[0].reason,
            "static_gqa_repeat_kernel_family",
        )
        self.assertIn(
            "vulkan_prepack.repeat_attention_heads_for_gqa",
            program.graph_module.code,
        )
        self.assertNotIn("aten.expand.default", program.graph_module.code)
        self.assertEqual(program(tensor).cpu(), model(tensor))
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)

    def test_static_gqa_repeat_rejects_non_float_kernel_input(self):
        class RepeatHeads(torch.nn.Module):
            def forward(self, tensor):
                batch, heads, tokens, width = tensor.shape
                return tensor.unsqueeze(2).expand(
                    batch,
                    heads,
                    2,
                    tokens,
                    width,
                ).reshape(batch, heads * 2, tokens, width)

        tensor = torch.ones(1, 2, 3, 4, dtype=torch.int32)
        program = torch.vulkan.export_and_lower(RepeatHeads().eval(), tensor)
        report = program.static_gqa_repeats
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.lowered_count, 0)
        self.assertEqual(report.rejected_count, 1)
        self.assertEqual(report.nodes[0].reason, "gqa_repeat_requires_float32")
        self.assertIn("aten.expand.default", program.graph_module.code)

    def test_enable_grad_wrapper_stays_unsupported_for_inference(self):
        body_graph = torch.fx.Graph()
        body_input = body_graph.placeholder("tensor")
        body_output = body_graph.call_function(
            torch.ops.aten.sin.default, (body_input,)
        )
        body_graph.output((body_output,))
        body = torch.fx.GraphModule({}, body_graph)

        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        body_attr = graph.get_attr("body")
        wrapped = graph.call_function(
            torch.ops.higher_order.wrap_with_set_grad_enabled,
            (True, body_attr, tensor),
        )
        output = graph.call_function(operator.getitem, (wrapped, 0))
        graph.output(output)
        graph_module = torch.fx.GraphModule({"body": body}, graph)

        self.assertEqual(vulkan_graph._inline_inference_grad_wrappers(graph_module), 0)
        record = vulkan_graph._classify_node(graph_module, 2, wrapped)
        self.assertEqual(record.classification, "unsupported")
        self.assertEqual(record.reason, "unsupported_graph_node_kind")

    def test_dynamic_batch_reuses_program(self):
        class DynamicLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 7)

            def forward(self, tensor):
                return torch.relu(self.linear(tensor))

        torch.manual_seed(1)
        model = DynamicLinear().eval()
        example = torch.randn(2, 3, 5)
        batch = torch.export.Dim("batch", min=1, max=8)
        tokens = torch.export.Dim("tokens", min=1, max=16)
        program = torch.vulkan.export_and_lower(
            model,
            example,
            dynamic_shapes=({0: batch, 1: tokens},),
        )

        self.assertIn("s", program.key.input_signature[0])
        for batch_size, token_count in ((1, 1), (5, 11), (8, 16)):
            tensor = torch.randn(batch_size, token_count, 5)
            self.assertEqual(program(tensor).cpu(), model(tensor))
        self.assertEqual(program.run_count, 3)
        self.assertEqual(program.linear_lowering.lowered_count, 1)
        context_attrs = _linear_context_attrs(program)
        with self.assertRaises(torch.vulkan.VulkanGraphExecutionError):
            program(torch.randn(9, 16, 5))
        with self.assertRaises(torch.vulkan.VulkanGraphExecutionError):
            program(torch.randn(2, 3, 6))
        tensor = torch.randn(3, 7, 5)
        self.assertEqual(program(tensor).cpu(), model(tensor))
        self.assertEqual(program.run_count, 4)
        self.assertEqual(_linear_context_attrs(program), context_attrs)

    def test_cpp_graph_plan_executes_dynamic_symbolic_size(self):
        class DynamicView(torch.nn.Module):
            def forward(self, tensor):
                return tensor.view(tensor.shape[0], -1)

        batch = torch.export.Dim("batch", min=1, max=8)
        first_input = torch.randn(2, 3, 4)
        second_input = torch.randn(7, 3, 4)
        program = torch.vulkan.export_and_lower(
            DynamicView().eval(),
            first_input,
            dynamic_shapes=({0: batch},),
        )

        symbolic_size = next(
            node
            for node in program.census.nodes
            if node.target == "aten::sym_size.int"
        )
        self.assertEqual(symbolic_size.classification, "composite")
        self.assertEqual(
            symbolic_size.reason,
            "registered_CompositeImplicitAutograd",
        )
        self.assertEqual(program.execution_mode, "cpp_plan")
        report = program.cpp_plan_report
        self.assertEqual(report.status, "compiled")
        self.assertEqual(report.plan_version, "v9")
        self.assertEqual(report.instruction_count, 2)
        self.assertEqual(report.graph_scalar_instruction_count, 0)
        self.assertEqual(report.list_argument_count, 1)
        self.assertEqual(report.value_count, 3)

        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            first_output = program(first_input)
            second_output = program(second_input)

        self.assertEqual(first_output.cpu(), first_input.view(2, -1))
        self.assertEqual(second_output.cpu(), second_input.view(7, -1))
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_cpp_graph_plan_executes_dynamic_shape_scalar_arithmetic(self):
        class DynamicReshape(torch.nn.Module):
            def forward(self, tensor):
                rows = tensor.shape[1] * tensor.shape[2]
                return tensor.reshape(tensor.shape[0], rows, tensor.shape[3])

        example = torch.randn(2, 3, 4, 5)
        batch = torch.export.Dim("batch", min=1, max=4)
        height = torch.export.Dim("height", min=2, max=8)
        width = torch.export.Dim("width", min=2, max=8)
        program = torch.vulkan.export_and_lower(
            DynamicReshape().eval(),
            example,
            dynamic_shapes=({0: batch, 1: height, 2: width},),
        )
        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            for shape in ((1, 2, 7, 5), (4, 8, 2, 5)):
                tensor = torch.randn(shape)
                self.assertEqual(
                    program(tensor).cpu(),
                    tensor.reshape(shape[0], -1, 5),
                )
        scalar_nodes = [
            node
            for node in program.census.nodes
            if node.reason == "python_scalar_shape_arithmetic"
        ]
        self.assertTrue(scalar_nodes)
        self.assertTrue(
            all(node.classification == "graph" for node in scalar_nodes)
        )
        self.assertEqual(program.execution_mode, "cpp_plan")
        self.assertEqual(program.cpp_plan_report.status, "compiled")
        self.assertEqual(program.cpp_plan_report.plan_version, "v9")
        self.assertEqual(
            program.cpp_plan_report.graph_scalar_instruction_count,
            1,
        )
        self.assertEqual(program.cpp_plan.graph_scalar_instruction_count(), 1)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_cpp_graph_plan_checks_integer_shape_arithmetic(self):
        class DynamicScale(torch.nn.Module):
            def forward(self, tensor):
                scale = ((tensor.shape[0] - 5) // 2) * 3 + 1
                return torch.ops.aten.mul.Scalar(tensor, scale)

        batch = torch.export.Dim("batch", min=1, max=8)
        first_input = torch.randn(2, 3)
        second_input = torch.randn(7, 3)
        model = DynamicScale().eval()
        program = torch.vulkan.export_and_lower(
            model,
            first_input,
            dynamic_shapes=({0: batch},),
        )

        self.assertEqual(program.execution_mode, "cpp_plan")
        self.assertEqual(program.cpp_plan_report.plan_version, "v9")
        self.assertEqual(
            program.cpp_plan_report.graph_scalar_instruction_count,
            4,
        )
        with patch.object(
            vulkan_graph._VulkanGraphInterpreter,
            "run_node",
            side_effect=AssertionError("Python node execution is forbidden"),
        ):
            first_output = program(first_input)
            second_output = program(second_input)
        self.assertEqual(first_output.cpu(), model(first_input))
        self.assertEqual(second_output.cpu(), model(second_input))
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_cpp_graph_plan_rejects_float_graph_scalar_instruction(self):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        tensor.meta["val"] = torch.empty(2, 3)
        size = graph.call_function(
            torch.ops.aten.sym_size.int,
            args=(tensor, 0),
        )
        scale = graph.call_function(operator.mul, args=(size, 1.5))
        output = graph.call_function(
            torch.ops.aten.mul.Scalar,
            args=(tensor, scale),
        )
        graph.output(output)
        graph_module = torch.fx.GraphModule({}, graph)
        compilation = vulkan_graph.compile_vulkan_graph_plan(
            graph_module,
            {
                size.name: "composite",
                scale.name: "graph",
                output.name: "direct_vulkan",
            },
        )

        self.assertIsNone(compilation.plan)
        self.assertEqual(
            compilation.report.status,
            "python_correctness_executor",
        )
        self.assertEqual(
            compilation.report.reason,
            "argument_type_mismatch:mul:operand_1",
        )

    def test_cpp_graph_plan_checks_graph_scalar_division_by_zero(self):
        plan = _graph_scalar_error_plan(
            "vulkan_graph::int_floor_divide",
            1,
            0,
        )
        tensor = torch.randn(2, 3, device="vulkan")
        with self.assertRaisesRegex(
            RuntimeError,
            "VulkanGraphPlan.v9 node 'shape_scalar'.*divides by zero",
        ):
            torch.ops.vulkan_prepack.run_vulkan_graph_plan.default(
                [tensor], plan
            )

    def test_cpp_graph_plan_checks_graph_scalar_overflow(self):
        plan = _graph_scalar_error_plan(
            "vulkan_graph::int_add",
            2**63 - 1,
            1,
        )
        tensor = torch.randn(2, 3, device="vulkan")
        with self.assertRaisesRegex(
            RuntimeError,
            "VulkanGraphPlan.v9 node 'shape_scalar'.*overflows int64",
        ):
            torch.ops.vulkan_prepack.run_vulkan_graph_plan.default(
                [tensor], plan
            )

    def test_cpp_graph_plan_rejects_graph_scalar_list_recipe(self):
        with self.assertRaisesRegex(RuntimeError, "requires value arguments"):
            torch.ops.vulkan_prepack.create_vulkan_graph_plan.default(
                ["shape_scalar", "scale"],
                ["vulkan_graph::int_add", "aten::mul"],
                ["", "Scalar"],
                [[[-1], [-2]], [[0], [1]]],
                [[1, 0], [0, 0]],
                [[1], [2]],
                [1, 2],
                1,
                [2],
            )

    def test_python_tensor_arithmetic_stays_unsupported(self):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        tensor.meta["val"] = torch.empty(2)
        add = graph.call_function(operator.add, args=(tensor, 1))
        add.meta["val"] = torch.empty(2)
        graph.output(add)
        graph_module = torch.fx.GraphModule({}, graph)
        record = vulkan_graph._classify_node(graph_module, 1, add)
        self.assertEqual(record.classification, "unsupported")
        self.assertEqual(record.reason, "unsupported_graph_node_kind")

    def test_binds_runtime_kwargs_in_graph_placeholder_order(self):
        class LinearResidual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)

            def forward(self, tensor, *, residual):
                return torch.relu(self.linear(tensor) + residual)

        torch.manual_seed(2)
        model = LinearResidual().eval()
        tensor = torch.randn(2, 3)
        residual = torch.randn(2, 4)
        program = torch.vulkan.export_and_lower(
            model,
            tensor,
            example_kwargs={"residual": residual},
        )
        self.assertEqual(
            program(tensor, residual=residual).cpu(),
            model(tensor, residual=residual),
        )
        with self.assertRaisesRegex(
            torch.vulkan.VulkanGraphExecutionError,
            "input binding failed",
        ):
            program(tensor)

    def test_binds_nested_runtime_inputs_in_exported_pytree_order(self):
        class NestedResidual(torch.nn.Module):
            def forward(self, tensor, state):
                first, second = state
                return torch.relu(tensor + first + second)

        torch.manual_seed(3)
        model = NestedResidual().eval()
        tensor = torch.randn(2, 4)
        state = (torch.randn(2, 4), torch.randn(2, 4))
        program = torch.vulkan.export_and_lower(model, (tensor, state))
        self.assertEqual(program(tensor, state).cpu(), model(tensor, state))
        self.assertEqual(
            tuple(
                node.target
                for node in program.graph_module.graph.nodes
                if node.op == "placeholder"
            ),
            ("tensor", "state_0", "state_1"),
        )
        with self.assertRaisesRegex(
            torch.vulkan.VulkanGraphExecutionError,
            "input binding failed: runtime input tree",
        ):
            program(tensor, list(state))

    def test_replays_nested_explicit_state_across_guard_variants(self):
        class AppendState(torch.nn.Module):
            def forward(self, token, state):
                key, value = state
                next_key = torch.cat((key, token), dim=1)
                next_value = torch.cat((value, token + 1.0), dim=1)
                return token, (next_key, next_value)

        torch.manual_seed(4)
        model = AppendState().eval()
        first_token = torch.randn(1, 1, 4)
        first_state = (torch.randn(1, 2, 4), torch.randn(1, 2, 4))
        first_expected = model(first_token, first_state)
        first_program = torch.vulkan.export_and_lower(
            model,
            (first_token, first_state),
        )

        second_token = torch.randn(1, 1, 4)
        second_expected = model(second_token, first_expected[1])
        second_program = torch.vulkan.export_and_lower(
            model,
            (second_token, first_expected[1]),
        )
        self.assertNotEqual(first_program.key, second_program.key)

        first_output = first_program(first_token, first_state)
        second_output = second_program(second_token, first_output[1])
        self.assertEqual(
            pytree.tree_map(
                lambda tensor: tensor.cpu(),
                second_output,
            ),
            second_expected,
        )
        self.assertEqual(
            pytree.tree_map(
                lambda tensor: tensor.cpu(),
                first_output,
            ),
            first_expected,
        )
        for program in (first_program, second_program):
            self.assertEqual(program.cpp_plan_report.input_count, 3)
            self.assertEqual(program.cpp_plan_report.output_count, 3)
            self.assertEqual(program.last_cpu_fallback_count, 0)
            self.assertEqual(program.last_sync_readback_count, 0)
            self.assertEqual(program.last_deferred_values_created, 0)

    def test_rejects_dynamic_linear_weight_before_execution(self):
        class DynamicWeightLinear(torch.nn.Module):
            def forward(self, tensor, weight):
                return torch.ops.aten.linear.default(tensor, weight, None)

        with self.assertRaisesRegex(
            torch.vulkan.VulkanGraphExecutionError,
            "weight_not_static_get_attr",
        ):
            torch.vulkan.export_and_lower(
                DynamicWeightLinear().eval(),
                (torch.randn(2, 3), torch.randn(4, 3)),
            )

    def test_reuses_tied_linear_context_without_retaining_export_state(self):
        class SharedLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)

            def forward(self, tensor):
                return self.linear(tensor) + self.linear(tensor)

        torch.manual_seed(3)
        model = SharedLinear().eval()
        tensor = torch.randn(2, 3)
        program = torch.vulkan.export_and_lower(model, tensor)
        self.assertEqual(program.linear_lowering.linear_node_count, 2)
        self.assertEqual(program.linear_lowering.lowered_count, 2)
        self.assertEqual(program.linear_lowering.created_context_count, 1)
        self.assertEqual(program.linear_lowering.reused_context_count, 1)
        self.assertEqual(
            program.linear_lowering.context_factory,
            "vulkan_prepack::create_graph_linear_context",
        )
        context_attrs = _linear_context_attrs(program)
        self.assertEqual(len(context_attrs), 1)
        self.assertFalse(hasattr(program, "exported_program"))
        self.assertFalse(hasattr(program, "_exported_program"))
        self.assertEqual(program(tensor).cpu(), model(tensor))
        self.assertEqual(program(tensor).cpu(), model(tensor))
        self.assertEqual(program.run_count, 2)
        self.assertEqual(_linear_context_attrs(program), context_attrs)
        self.assertFalse(program.graph_module.state_dict())

    def test_rejects_scalar_only_output(self):
        class ScalarOnly(torch.nn.Module):
            def forward(self, tensor):
                return 1

        program = torch.vulkan.export_and_lower(ScalarOnly().eval(), torch.randn(2))
        with self.assertRaisesRegex(
            torch.vulkan.VulkanGraphExecutionError,
            "must contain at least one Vulkan tensor",
        ):
            program(torch.randn(2))

    def test_rejects_training_and_non_cpu_capture(self):
        model = torch.nn.Linear(3, 4)
        tensor = torch.randn(2, 3)
        with self.assertRaisesRegex(ValueError, "inference-only"):
            torch.vulkan.export_and_lower(model, tensor)

        model.eval()
        with self.assertRaisesRegex(ValueError, "CPU example tensors"):
            torch.vulkan.export_and_lower(model, tensor.to("vulkan"))

    def test_rejects_unsupported_node_before_execution(self):
        class Unsupported(torch.nn.Module):
            def forward(self, tensor):
                return torch.histc(tensor)

        with self.assertRaisesRegex(
            torch.vulkan.VulkanGraphExecutionError,
            "unsupported Vulkan nodes",
        ):
            torch.vulkan.export_and_lower(
                Unsupported().eval(),
                torch.randn(4, 4),
            )

    def test_rejects_implicit_cpu_fallback_during_execution(self):
        class IntegralCast(torch.nn.Module):
            def forward(self, tensor):
                return tensor.to(torch.int32)

        tensor = torch.randn(4)
        program = torch.vulkan.export_and_lower(IntegralCast().eval(), tensor)
        with self.assertRaisesRegex(
            torch.vulkan.VulkanGraphExecutionError,
            r"Vulkan graph node 'to' \(aten::to.dtype\) crossed an implicit "
            "host boundary: cpu_fallback=1, sync_readback=0, "
            "deferred_values_created=0",
        ):
            program(tensor)
        self.assertEqual(program.run_count, 0)
        self.assertEqual(program.last_cpu_fallback_count, 1)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)
        attribution = program.last_implicit_boundary
        self.assertIsNotNone(attribution)
        self.assertEqual(attribution.node_name, "to")
        self.assertEqual(attribution.target, "aten::to.dtype")
        self.assertEqual(attribution.cpu_fallback_count, 1)
        self.assertEqual(attribution.sync_readback_count, 0)
        self.assertEqual(attribution.deferred_values_created, 0)
        input_guard = program._exported_input_guard
        self.assertIsInstance(input_guard, GuardsFn)

        def fail_guard(*values):
            del values
            raise RuntimeError("expected input guard failure")

        input_guard.forward = fail_guard
        with self.assertRaisesRegex(
            torch.vulkan.VulkanGraphExecutionError,
            "exported input guard failed",
        ):
            program(tensor)
        self.assertEqual(program.last_executed_nodes, ())
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)
        self.assertIsNone(program.last_implicit_boundary)
        with self.assertRaisesRegex(
            torch.vulkan.VulkanGraphExecutionError,
            "input binding failed",
        ):
            program()
        self.assertEqual(program.last_executed_nodes, ())
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)
        self.assertIsNone(program.last_implicit_boundary)

    def test_normalizes_isolated_int64_unsqueeze_to_float32_at_graph_input(self):
        class NormalizeTimestep(torch.nn.Module):
            def forward(self, tensor):
                return tensor.unsqueeze(0).to(torch.float32) * 0.5

        tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
        model = NormalizeTimestep().eval()
        program = torch.vulkan.export_and_lower(model, tensor)
        with torch.inference_mode():
            cpu_output = model(tensor)
            eager_vulkan_output = model(tensor.to("vulkan")).cpu()
            graph_output = program(tensor).cpu()
        torch.testing.assert_close(graph_output, cpu_output)
        torch.testing.assert_close(graph_output, eager_vulkan_output)
        report = program.input_normalization
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.lowered_count, 1)
        self.assertEqual(report.rejected_count, 0)
        node = report.nodes[0]
        self.assertEqual(node.status, "lowered")
        self.assertEqual(node.placeholder_name, "tensor")
        self.assertEqual(node.source_dtype, torch.int64)
        self.assertEqual(node.target_dtype, torch.float32)
        self.assertEqual(node.erased_node_name, "to")
        self.assertEqual(node.chain_node_names, ("unsqueeze",))
        self.assertNotIn("aten.to.dtype", program.graph_module.code)
        self.assertIn("dtype=torch.int64", program.key.input_signature[0])
        self.assertIn("dtype = torch.float32", program.graph_module.code)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

        with self.assertRaisesRegex(
            torch.vulkan.VulkanGraphExecutionError,
            "requires CPU dtype torch.int64 before graph input normalization",
        ):
            program(tensor.to(torch.float32))
        with self.assertRaisesRegex(
            torch.vulkan.VulkanGraphExecutionError,
            "requires the original CPU dtype torch.int64",
        ):
            program(tensor.to("vulkan"))
        with self.assertRaisesRegex(
            torch.vulkan.VulkanGraphExecutionError,
            "requires the original CPU dtype torch.int64",
        ):
            program(tensor.to(torch.float32).to("vulkan"))

    def test_normalizes_isolated_int64_attention_mask_to_bool_at_graph_input(self):
        class NormalizeMask(torch.nn.Module):
            def forward(self, attention_mask):
                return attention_mask.to(
                    device=attention_mask.device,
                    dtype=torch.bool,
                )

        attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.int64)
        model = NormalizeMask().eval()
        program = torch.vulkan.export_and_lower(model, attention_mask)
        report = program.input_normalization
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.lowered_count, 1)
        self.assertEqual(report.rejected_count, 0)
        node = report.nodes[0]
        self.assertEqual(node.status, "lowered")
        self.assertEqual(node.reason, "isolated_int64_placeholder_to_bool")
        self.assertEqual(node.placeholder_name, "attention_mask")
        self.assertEqual(node.source_dtype, torch.int64)
        self.assertEqual(node.target_dtype, torch.bool)
        self.assertEqual(node.chain_node_names, ())
        self.assertNotIn("aten.to.device", program.graph_module.code)
        self.assertIn("dtype=torch.int64", program.key.input_signature[0])
        self.assertEqual(program(attention_mask).cpu(), model(attention_mask))
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)

    def test_normalizes_isolated_int64_identity_expand_unsqueeze_to_float32(
        self,
    ):
        class NormalizeTimestep(torch.nn.Module):
            def forward(self, tensor):
                return tensor.expand([1]).unsqueeze(1).to(torch.float32) * 0.5

        tensor = torch.tensor([3], dtype=torch.int64)
        model = NormalizeTimestep().eval()
        program = torch.vulkan.export_and_lower(model, tensor)
        with torch.inference_mode():
            cpu_output = model(tensor)
            eager_vulkan_output = model(tensor.to("vulkan")).cpu()
            graph_output = program(tensor).cpu()
        torch.testing.assert_close(graph_output, cpu_output)
        torch.testing.assert_close(graph_output, eager_vulkan_output)
        report = program.input_normalization
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.lowered_count, 1)
        self.assertEqual(report.rejected_count, 0)
        node = report.nodes[0]
        self.assertEqual(node.status, "lowered")
        self.assertEqual(node.placeholder_name, "tensor")
        self.assertEqual(node.source_dtype, torch.int64)
        self.assertEqual(node.target_dtype, torch.float32)
        self.assertEqual(node.erased_node_name, "to")
        self.assertEqual(node.chain_node_names, ("expand", "unsqueeze"))
        self.assertNotIn("aten.to.dtype", program.graph_module.code)
        self.assertIn("dtype=torch.int64", program.key.input_signature[0])
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_evidence_latency_preserves_normalized_graph_input_contract(self):
        class NormalizeMask(torch.nn.Module):
            def forward(self, attention_mask):
                return attention_mask.to(
                    device=attention_mask.device,
                    dtype=torch.bool,
                )

        attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.int64)
        model = NormalizeMask().eval()
        program = torch.vulkan.export_and_lower(model, attention_mask)

        result = _measure_case_latency(
            program,
            model,
            (attention_mask,),
            warmup_repeats=0,
            measurement_repeats=1,
        )

        self.assertEqual(
            result["method"],
            "alternating_completed_supported_surface_invocations",
        )
        self.assertEqual(
            result["input_boundary"],
            "supported_eager_preuploaded_vulkan_inputs_and_graph_contract_"
            "inputs_to_completed_vulkan_outputs",
        )
        self.assertEqual(
            result["vulkan_graph_program"]["runtime_counters"],
            {"cpu_fallback": 0, "sync_readback": 0},
        )

    def test_normalizes_identity_expand_with_multiplaceholder_export_guard(self):
        class NormalizeTimestep(torch.nn.Module):
            def forward(self, timestep, other):
                return (
                    timestep.expand([1]).unsqueeze(1).to(torch.float32) + other
                )

        timestep_value = (1 << 40) + 1
        timestep = torch.tensor([timestep_value], dtype=torch.int64)
        other = torch.ones(1, 1)
        model = NormalizeTimestep().eval()
        program = torch.vulkan.export_and_lower(model, (timestep, other))
        input_guard = program._exported_input_guard
        self.assertIsInstance(input_guard, GuardsFn)
        seen_guard_inputs = []

        def check_guard(guard_timestep, guard_other):
            seen_guard_inputs.append(
                (
                    guard_timestep.device,
                    guard_timestep.dtype,
                    guard_timestep.item(),
                    guard_other.device,
                )
            )
            if (
                guard_timestep.device.type != "cpu"
                or guard_timestep.dtype != torch.int64
                or guard_timestep.item() != timestep_value
                or guard_other.device.type != "cpu"
            ):
                raise RuntimeError("input guard did not receive original inputs")

        input_guard.forward = check_guard
        with torch.inference_mode():
            cpu_output = model(timestep, other)
            eager_vulkan_output = model(
                timestep.to("vulkan"), other.to("vulkan")
            ).cpu()
            graph_output = program(timestep, other).cpu()
        torch.testing.assert_close(graph_output, cpu_output)
        torch.testing.assert_close(graph_output, eager_vulkan_output)
        report = program.input_normalization
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.lowered_count, 1)
        self.assertEqual(report.rejected_count, 0)
        self.assertEqual(report.nodes[0].chain_node_names, ("expand", "unsqueeze"))
        self.assertNotIn("aten.to.dtype", program.graph_module.code)
        self.assertNotIn("_guards_fn", program.graph_module.code)
        self.assertEqual(
            seen_guard_inputs,
            [
                (
                    torch.device("cpu"),
                    torch.int64,
                    timestep_value,
                    torch.device("cpu"),
                )
            ],
        )
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

    def test_graph_input_normalization_export_guard_accepts_only_placeholders(self):
        def guard_graph_module(
            guard_module,
            non_placeholder_argument=False,
            guard_kwargs=None,
            guard_output_escapes=False,
        ):
            graph = torch.fx.Graph()
            timestep = graph.placeholder("timestep")
            other = graph.placeholder("other")
            guard_args = (timestep, other)
            if non_placeholder_argument:
                view = graph.call_function(
                    torch.ops.aten.unsqueeze.default,
                    (other, 0),
                )
                guard_args = (timestep, view)
            guard = graph.call_module(
                "_guards_fn",
                guard_args,
                {} if guard_kwargs is None else guard_kwargs,
            )
            graph.output(guard if guard_output_escapes else timestep)
            root = torch.nn.Module()
            root.add_module("_guards_fn", guard_module)
            return torch.fx.GraphModule(root, graph), timestep, guard

        graph_module, timestep, valid_guard = guard_graph_module(
            GuardsFn(),
        )
        self.assertTrue(
            vulkan_graph_lowering._is_export_guard_user(
                graph_module, valid_guard, timestep
            )
        )
        extracted_guard = vulkan_graph_lowering.extract_verified_exported_input_guard(
            graph_module
        )
        self.assertIsInstance(extracted_guard, GuardsFn)
        self.assertNotIn("_guards_fn", graph_module.code)
        self.assertFalse(hasattr(graph_module, "_guards_fn"))

        class SameNamedUserModule(torch.nn.Module):
            def forward(self, *args):
                return args[0]

        graph_module, timestep, same_named_user = guard_graph_module(
            SameNamedUserModule(),
        )
        self.assertFalse(
            vulkan_graph_lowering._is_export_guard_user(
                graph_module, same_named_user, timestep
            )
        )
        self.assertIsNone(
            vulkan_graph_lowering.extract_verified_exported_input_guard(
                graph_module
            )
        )
        self.assertIn("_guards_fn", graph_module.code)
        record = next(
            node
            for node in vulkan_graph._build_census(graph_module).nodes
            if node.name == same_named_user.name
        )
        self.assertEqual(record.classification, "unsupported")
        self.assertEqual(record.reason, "unverified_exported_input_guard")

        graph_module, timestep, invalid_guard = guard_graph_module(
            GuardsFn(),
            non_placeholder_argument=True,
        )
        self.assertFalse(
            vulkan_graph_lowering._is_export_guard_user(
                graph_module, invalid_guard, timestep
            )
        )
        self.assertIsNone(
            vulkan_graph_lowering.extract_verified_exported_input_guard(
                graph_module
            )
        )
        self.assertIn("_guards_fn", graph_module.code)

        graph_module, timestep, guard_with_kwargs = guard_graph_module(
            GuardsFn(),
            guard_kwargs={"invalid": True},
        )
        self.assertFalse(
            vulkan_graph_lowering._is_export_guard_user(
                graph_module, guard_with_kwargs, timestep
            )
        )
        self.assertIsNone(
            vulkan_graph_lowering.extract_verified_exported_input_guard(
                graph_module
            )
        )
        self.assertIn("_guards_fn", graph_module.code)

        graph_module, timestep, escaping_guard = guard_graph_module(
            GuardsFn(),
            guard_output_escapes=True,
        )
        self.assertFalse(
            vulkan_graph_lowering._is_export_guard_user(
                graph_module, escaping_guard, timestep
            )
        )
        self.assertIsNone(
            vulkan_graph_lowering.extract_verified_exported_input_guard(
                graph_module
            )
        )
        self.assertIn("_guards_fn", graph_module.code)

    def test_rejects_graph_input_normalization_nonidentity_expand(self):
        class NonIdentityExpand(torch.nn.Module):
            def forward(self, tensor):
                return tensor.expand([2]).unsqueeze(1).to(torch.float32)

        program = torch.vulkan.export_and_lower(
            NonIdentityExpand().eval(),
            torch.tensor([1], dtype=torch.int64),
        )
        report = program.input_normalization
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.lowered_count, 0)
        self.assertEqual(report.rejected_count, 1)
        self.assertEqual(report.nodes[0].reason, "expand_not_identity")
        self.assertEqual(report.nodes[0].chain_node_names, ("unsqueeze",))
        self.assertIn("aten.to.dtype", program.graph_module.code)

    def test_rejects_graph_input_normalization_with_observable_view_consumer(self):
        class ObservableView(torch.nn.Module):
            def forward(self, tensor):
                view = tensor.unsqueeze(0)
                return view, view.to(torch.float32)

        program = torch.vulkan.export_and_lower(
            ObservableView().eval(),
            torch.tensor([1, 2, 3], dtype=torch.int64),
        )
        report = program.input_normalization
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.lowered_count, 0)
        self.assertEqual(report.rejected_count, 1)
        self.assertEqual(
            report.nodes[0].reason,
            "placeholder_path_has_observable_consumer_or_alias",
        )
        self.assertIn("aten.to.dtype", program.graph_module.code)

    def test_rejects_graph_input_normalization_unsupported_cast_dtype(self):
        class UnsupportedCast(torch.nn.Module):
            def forward(self, tensor):
                return tensor.unsqueeze(0).to(torch.float16)

        program = torch.vulkan.export_and_lower(
            UnsupportedCast().eval(),
            torch.tensor([1, 2, 3], dtype=torch.int64),
        )
        report = program.input_normalization
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(report.lowered_count, 0)
        self.assertEqual(report.rejected_count, 1)
        self.assertEqual(
            report.nodes[0].reason,
            "to_dtype_target_not_supported_floating_dtype",
        )
        self.assertIn("aten.to.dtype", program.graph_module.code)

        class CopyCast(torch.nn.Module):
            def forward(self, tensor):
                return tensor.unsqueeze(0).to(torch.float32, copy=True)

        copy_program = torch.vulkan.export_and_lower(
            CopyCast().eval(),
            torch.tensor([1, 2, 3], dtype=torch.int64),
        )
        copy_report = copy_program.input_normalization
        self.assertEqual(copy_report.candidate_count, 1)
        self.assertEqual(copy_report.lowered_count, 0)
        self.assertEqual(copy_report.rejected_count, 1)
        self.assertEqual(
            copy_report.nodes[0].reason,
            "to_dtype_signature_not_static_default",
        )
        self.assertIn("aten.to.dtype", copy_program.graph_module.code)

    def test_graph_node_scope_ends_when_dispatch_raises(self):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        failure = graph.call_function(_raise_graph_node_error, (tensor,))
        graph.output(failure)
        graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
        interpreter = vulkan_graph._VulkanGraphInterpreter(
            graph_module,
            torch.device("vulkan"),
        )
        outer = torch.ops.vulkan_prepack.begin_graph_execution_scope()
        try:
            with self.assertRaisesRegex(
                torch.vulkan.VulkanGraphExecutionError,
                "Vulkan graph node '.*' .*expected graph node failure",
            ):
                interpreter.run(torch.randn(1).to("vulkan"))
        finally:
            self.assertEqual(
                torch.ops.vulkan_prepack.end_graph_execution_scope(outer),
                [0, 0, 0],
            )

    def test_graph_execution_scope_is_nested_and_lifo(self):
        outer = torch.ops.vulkan_prepack.begin_graph_execution_scope()
        inner = torch.ops.vulkan_prepack.begin_graph_execution_scope()
        with self.assertRaisesRegex(RuntimeError, "LIFO order"):
            torch.ops.vulkan_prepack.end_graph_execution_scope(outer)
        self.assertEqual(
            torch.ops.vulkan_prepack.end_graph_execution_scope(inner),
            [0, 0, 0],
        )
        self.assertEqual(
            torch.ops.vulkan_prepack.end_graph_execution_scope(outer),
            [0, 0, 0],
        )

if __name__ == "__main__":
    run_tests()
