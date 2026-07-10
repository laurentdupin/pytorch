import os
import unittest
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


def _linear_context_attrs(program):
    return {
        str(node.target)
        for node in program.graph_module.graph.nodes
        if node.op == "get_attr"
        and str(node.target).startswith("_vulkan_linear_context_")
    }


@unittest.skipUnless(torch.vulkan.is_available(), "Vulkan is not available")
class TestVulkanGraph(TestCase):
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
