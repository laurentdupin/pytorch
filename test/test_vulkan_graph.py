import copy
import os
import operator
import unittest
from unittest.mock import patch

import torch
import torch.vulkan._graph as vulkan_graph
from torch.testing._internal.common_utils import run_tests, TestCase


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
        and str(node.target).startswith("_vulkan_static_linear_gelu_plan_")
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
        and str(node.target).startswith("_vulkan_static_conv2d_relu_conv2d_plan_")
    }


def _static_conv2d_relu_conv2d_plan_attrs_from_module(graph_module):
    return {
        str(node.target)
        for node in graph_module.graph.nodes
        if node.op == "get_attr"
        and str(node.target).startswith("_vulkan_static_conv2d_relu_conv2d_plan_")
    }


@unittest.skipUnless(torch.vulkan.is_available(), "Vulkan is not available")
class TestVulkanGraph(TestCase):
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
            "vulkan_prepack::create_graph_linear_gelu_plan",
        )
        node = report.nodes[0]
        self.assertEqual(node.program_name, "StaticLinearGeluRegion")
        self.assertEqual(node.program_version, "v1")
        self.assertEqual(node.instruction_count, 1)
        self.assertEqual(node.input_ssa, 0)
        self.assertEqual(node.output_ssa, 1)
        self.assertEqual(node.input_use_count, 1)
        self.assertEqual(node.input_last_use, 0)
        self.assertEqual(node.static_context_slot, 0)
        self.assertTrue(node.direct_transition_only)
        self.assertTrue(node.replay_state_empty)
        self.assertEqual(len(_static_linear_gelu_plan_attrs(program)), 1)
        self.assertFalse(_linear_context_attrs(program))
        self.assertFalse(program.graph_module.state_dict())
        lowered = [
            node
            for node in program.census.nodes
            if node.reason == "graph_owned_static_linear_gelu_plan"
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
        plan_attrs = _static_linear_gelu_plan_attrs(program)
        self.assertEqual(len(plan_attrs), 1)
        for shape in ((1, 1, 5), (4, 8, 5)):
            tensor = torch.randn(shape)
            torch.testing.assert_close(
                program(tensor).cpu(), model(tensor), rtol=1e-4, atol=1e-4
            )
        self.assertEqual(_static_linear_gelu_plan_attrs(program), plan_attrs)
        self.assertEqual(program.static_linear_gelu_regions.lowered_count, 1)
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

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

    def test_static_linear_gelu_none_stays_unfused(self):
        class LinearGelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)

            def forward(self, tensor):
                return torch.nn.functional.gelu(
                    self.linear(tensor), approximate="none"
                )

        torch.manual_seed(12)
        model = LinearGelu().eval()
        tensor = torch.randn(2, 3)
        program = torch.vulkan.export_and_lower(model, tensor)
        torch.testing.assert_close(
            program(tensor).cpu(), model(tensor), rtol=1e-4, atol=1e-4
        )
        self.assertEqual(program.static_linear_gelu_regions.candidate_count, 0)
        self.assertEqual(program.static_linear_gelu_regions.lowered_count, 0)
        self.assertFalse(_static_linear_gelu_plan_attrs(program))
        self.assertEqual(len(_linear_context_attrs(program)), 1)

    def test_static_linear_gelu_multi_use_linear_output_stays_unfused(self):
        class LinearGeluResidual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)

            def forward(self, tensor):
                linear = self.linear(tensor)
                return torch.nn.functional.gelu(linear, approximate="tanh") + linear

        torch.manual_seed(13)
        model = LinearGeluResidual().eval()
        tensor = torch.randn(2, 3)
        program = torch.vulkan.export_and_lower(model, tensor)
        torch.testing.assert_close(
            program(tensor).cpu(), model(tensor), rtol=1e-4, atol=1e-4
        )
        report = program.static_linear_gelu_regions
        self.assertEqual(report.lowered_count, 0)
        self.assertEqual(report.skipped_count, 1)
        self.assertEqual(report.nodes[0].reason, "linear_output_has_multiple_users")
        self.assertFalse(_static_linear_gelu_plan_attrs(program))
        self.assertEqual(len(_linear_context_attrs(program)), 1)

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
            kwargs={"approximate": "tanh"},
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
                str(node.target) == "vulkan_prepack.run_graph_linear_gelu_plan.default"
                for node in graph_module.graph.nodes
            )
        )

    def test_static_linear_gelu_dynamic_context_stays_unfused(self):
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
            kwargs={"approximate": "tanh"},
        )
        graph.output(gelu)
        graph_module = torch.fx.GraphModule({}, graph)

        report = vulkan_graph.lower_static_linear_gelu_regions(graph_module)
        self.assertEqual(report.candidate_count, 0)
        self.assertEqual(report.lowered_count, 0)
        self.assertFalse(
            any(
                str(node.target) == "vulkan_prepack.run_graph_linear_gelu_plan.default"
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
            eager_vulkan = copy.deepcopy(model).to("vulkan").eval()
            with torch.inference_mode():
                eager_vulkan_output = eager_vulkan(tensor.to("vulkan")).cpu()
            graph_output = program(tensor).cpu()
            torch.testing.assert_close(
                eager_vulkan_output, expected, rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(
                graph_output, expected, rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(
                graph_output, eager_vulkan_output, rtol=1e-4, atol=1e-4
            )
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
            self.assertEqual(node.program_name, "StaticConv2dReluConv2dRegion")
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
                if node.reason == "graph_owned_static_conv2d_relu_conv2d_plan"
            ]
            self.assertEqual(len(lowered), 1)
            self.assertEqual(lowered[0].classification, "lowered_vulkan")
            self.assertEqual(program.last_cpu_fallback_count, 0)
            self.assertEqual(program.last_sync_readback_count, 0)
            self.assertEqual(program.last_deferred_values_created, 0)

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
        for shape in ((1, 3, 5, 11), (4, 3, 12, 5)):
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

    def test_dynamic_shape_scalar_arithmetic_is_graph_bookkeeping(self):
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
        for shape in ((1, 2, 7, 5), (4, 8, 2, 5)):
            tensor = torch.randn(shape)
            self.assertEqual(program(tensor).cpu(), tensor.reshape(shape[0], -1, 5))
        scalar_nodes = [
            node
            for node in program.census.nodes
            if node.reason == "python_scalar_shape_arithmetic"
        ]
        self.assertTrue(scalar_nodes)
        self.assertTrue(all(node.classification == "graph" for node in scalar_nodes))
        self.assertEqual(program.last_cpu_fallback_count, 0)
        self.assertEqual(program.last_sync_readback_count, 0)
        self.assertEqual(program.last_deferred_values_created, 0)

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
            "crossed an implicit host boundary.*cpu_fallback=1",
        ):
            program(tensor)
        self.assertEqual(program.run_count, 0)
        self.assertEqual(program.last_cpu_fallback_count, 1)

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

    def test_graph_rejects_deferred_value_registration(self):
        class Add(torch.nn.Module):
            def forward(self, tensor):
                return tensor + tensor

        tensor = torch.randn(2, 3)
        program = torch.vulkan.export_and_lower(Add().eval(), tensor)
        with patch.dict(
            os.environ,
            {"PYTORCH_VULKAN_RUNTIME_ELEMENTWISE_CHAIN_DEFER": "1"},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "cannot register a deferred value",
            ):
                program(tensor)
        self.assertEqual(program.last_deferred_values_created, 0)


if __name__ == "__main__":
    run_tests()
