# Owner(s): ["oncall: mobile"]

import os
import subprocess
import sys
import textwrap
import unittest
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing import FileCheck
import io

@unittest.skipUnless(torch.is_vulkan_available(),
                     "Vulkan backend must be available for these tests.")
class TestVulkanRewritePass(TestCase):
    @staticmethod
    def validate_transformed_module(
            # To please flake
            self,
            pattern_count_map,
            data_shape,
            prepack_removal=False,
            fuse_clamping_ops=False):
        module_instance = self
        scripted_model = torch.jit.script(module_instance)
        scripted_model.eval()
        input_data = torch.normal(1, 20, size=data_shape)
        scripted_model(input_data)
        torch._C._jit_pass_vulkan_insert_prepacked_ops(scripted_model._c)
        if fuse_clamping_ops or prepack_removal:
            scripted_model._c = torch._C._freeze_module(scripted_model._c)
        if fuse_clamping_ops:
            torch._C._jit_pass_vulkan_fuse_clamp_w_prepacked_conv(scripted_model._c)
        if prepack_removal:
            torch._C._jit_pass_vulkan_fold_prepacking_ops(scripted_model._c)

        buffer = io.BytesIO()
        torch.jit.save(scripted_model, buffer)
        buffer.seek(0)
        deserialized_scripted_model = torch.jit.load(buffer)
        for pattern, v in pattern_count_map.items():
            if (v == 0):
                FileCheck().check(pattern).run(deserialized_scripted_model.graph)
            elif (v == -1):
                FileCheck().check_not(pattern).run(deserialized_scripted_model.graph)
            else:
                FileCheck().check_count(pattern, v, exactly=True).run(deserialized_scripted_model.graph)

    def test_conv(self):
        # Conv params
        batch_size = 2
        input_channels_per_group = 6
        height = 16
        width = 16
        output_channels_per_group = 6
        groups = 4
        kernel_h = kernel_w = 3
        stride_h = stride_w = 1
        pad_h = pad_w = 1
        dilation = 1
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        strides = (stride_h, stride_w)
        paddings = (pad_h, pad_w)
        dilations = (dilation, dilation)
        conv_weight_shape = (output_channels, input_channels_per_group, kernel_h, kernel_w)
        conv_bias_shape = (output_channels)

        class Conv2D(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(conv_weight_shape), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.rand(conv_bias_shape), requires_grad=False)
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                return F.conv2d(x, self.weight, self.bias,
                                self.strides, self.paddings, self.dilations, self.groups)

        data_shape = (batch_size, input_channels, height, width)
        pattern_count_map = {"Tensor = aten::conv2d": -1,
                             "vulkan_prepack::create_conv2d_context": 1,
                             "vulkan_prepack::run_conv2d_context": 1}
        TestVulkanRewritePass.validate_transformed_module(Conv2D(), pattern_count_map, data_shape)

        class Conv2DRelu(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(conv_weight_shape), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.rand(conv_bias_shape), requires_grad=False)
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                o = F.conv2d(x, self.weight, self.bias,
                             self.strides, self.paddings, self.dilations, self.groups)
                o = F.relu(o)
                return o

        data_shape = (batch_size, input_channels, height, width)
        pattern_count_map = {"Tensor = aten::conv2d": -1,
                             "vulkan_prepack::create_conv2d_context": 1,
                             "vulkan_prepack::run_conv2d_context": 1}
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DRelu(), pattern_count_map, data_shape)

        pattern_count_map["aten::relu"] = 1
        pattern_count_map["vulkan_prepack::create_conv2d_context"] = -1
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DRelu(),
            pattern_count_map,
            data_shape,
            prepack_removal=True)
        pattern_count_map["aten::relu"] = -1
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DRelu(),
            pattern_count_map,
            data_shape,
            prepack_removal=True,
            fuse_clamping_ops=True)


        class Conv2DHardtanh(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(conv_weight_shape), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.rand(conv_bias_shape), requires_grad=False)
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                o = F.conv2d(x, self.weight, self.bias,
                             self.strides, self.paddings, self.dilations, self.groups)
                o = F.hardtanh(o)
                return o

        data_shape = (batch_size, input_channels, height, width)
        pattern_count_map = {"Tensor = aten::conv2d": -1,
                             "vulkan_prepack::create_conv2d_context": 1,
                             "vulkan_prepack::run_conv2d_context": 1}
        TestVulkanRewritePass.validate_transformed_module(Conv2DHardtanh(), pattern_count_map, data_shape)
        pattern_count_map["aten::hardtanh"] = 1
        pattern_count_map["vulkan_prepack::create_conv2d_context"] = -1
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DHardtanh(),
            pattern_count_map,
            data_shape,
            prepack_removal=True)
        pattern_count_map["aten::hardtanh"] = -1
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DRelu(),
            pattern_count_map,
            data_shape,
            prepack_removal=True,
            fuse_clamping_ops=True)

class DepthAnythingStyleResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class DepthAnythingStyleFeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.residual_1 = DepthAnythingStyleResidualConvUnit(features)
        self.residual_2 = DepthAnythingStyleResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1)

    def forward(self, *xs, size=None):
        output = xs[0]

        if len(xs) == 2:
            output = output + self.residual_1(xs[1])

        output = self.residual_2(output)
        if size is None:
            output = F.interpolate(
                output,
                scale_factor=2,
                mode="bilinear",
                align_corners=True)
        else:
            output = F.interpolate(
                output,
                size=size,
                mode="bilinear",
                align_corners=True)
        return self.out_conv(output)


class DepthAnythingStyleReadoutProject(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU())

    def forward(self, tokens, cls_token):
        readout = cls_token.unsqueeze(1).expand_as(tokens)
        return self.project(torch.cat((tokens, readout), dim=-1))


class DepthAnythingStyleMiniDPTHead(nn.Module):
    # Reduced-width DPT head modeled on Depth Anything V2's decoder topology.
    def __init__(
            self,
            embed_dim=16,
            features=8,
            out_channels=(8, 8, 8, 8),
            use_clstoken=False,
            scratch_bias=True):
        super().__init__()
        self.use_clstoken = use_clstoken
        self.projects = nn.ModuleList(
            [nn.Conv2d(embed_dim, channels, kernel_size=1)
             for channels in out_channels])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])
        if use_clstoken:
            self.readout_projects = nn.ModuleList(
                [DepthAnythingStyleReadoutProject(embed_dim)
                 for _ in out_channels])

        self.scratch_layers = nn.ModuleList(
            [nn.Conv2d(
                channels,
                features,
                kernel_size=3,
                padding=1,
                bias=scratch_bias)
             for channels in out_channels])

        self.refinenet4 = DepthAnythingStyleFeatureFusionBlock(features)
        self.refinenet3 = DepthAnythingStyleFeatureFusionBlock(features)
        self.refinenet2 = DepthAnythingStyleFeatureFusionBlock(features)
        self.refinenet1 = DepthAnythingStyleFeatureFusionBlock(features)

        self.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, features // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, 1, kernel_size=1),
            nn.ReLU(inplace=True))

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x
                x = self.readout_projects[i](x, cls_token)
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape(
                (x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch_layers[0](layer_1)
        layer_2_rn = self.scratch_layers[1](layer_2)
        layer_3_rn = self.scratch_layers[2](layer_3)
        layer_4_rn = self.scratch_layers[3](layer_4)

        path_4 = self.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.refinenet1(path_2, layer_1_rn)

        out = self.output_conv1(path_1)
        out = F.interpolate(
            out,
            (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear",
            align_corners=True)
        return self.output_conv2(out)

@unittest.skipUnless(torch.is_vulkan_available(),
                     "Vulkan backend must be available for these tests.")
class TestVulkanEagerRuntime(TestCase):
    def _to_vulkan(self, value):
        if torch.is_tensor(value):
            return value.to("vulkan")
        if isinstance(value, tuple):
            return tuple(self._to_vulkan(v) for v in value)
        if isinstance(value, list):
            return [self._to_vulkan(v) for v in value]
        return value

    def _assert_outputs_close(self, expected, actual, *, atol=1e-4, rtol=1e-4):
        if torch.is_tensor(expected):
            self.assertTrue(torch.is_tensor(actual))
            self.assertEqual(expected, actual.cpu(), atol=atol, rtol=rtol)
            return

        if isinstance(expected, tuple):
            self.assertIsInstance(actual, tuple)
            self.assertEqual(len(expected), len(actual))
            for expected_item, actual_item in zip(expected, actual):
                self._assert_outputs_close(
                    expected_item,
                    actual_item,
                    atol=atol,
                    rtol=rtol)
            return

        if isinstance(expected, list):
            self.assertIsInstance(actual, list)
            self.assertEqual(len(expected), len(actual))
            for expected_item, actual_item in zip(expected, actual):
                self._assert_outputs_close(
                    expected_item,
                    actual_item,
                    atol=atol,
                    rtol=rtol)
            return

        self.assertEqual(expected, actual)

    def _assert_vulkan_matches_cpu(self, fn, *args, atol=1e-4, rtol=1e-4):
        # Exercise the Vulkan backend in inference mode to avoid autograd
        # dispatch ambiguity with CompositeImplicitAutograd kernels.
        with torch.inference_mode():
            expected = fn(*args)
            actual = fn(*self._to_vulkan(args))

        self._assert_outputs_close(expected, actual, atol=atol, rtol=rtol)

    def _assert_known_limitation(self, fn, *args, exc_type=RuntimeError, message):
        with torch.inference_mode():
            fn(*args)
            with self.assertRaisesRegex(exc_type, message):
                fn(*self._to_vulkan(args))

    def _make_depth_anything_style_features(
            self,
            *,
            patch_h=4,
            patch_w=4,
            embed_dim=16,
            use_clstoken=False):
        batch_size = 1
        token_count = patch_h * patch_w
        features = []
        for _ in range(4):
            tokens = torch.randn(batch_size, token_count, embed_dim)
            if use_clstoken:
                cls_token = torch.randn(batch_size, embed_dim)
                features.append((tokens, cls_token))
            else:
                features.append((tokens,))
        return features

    def test_binary_and_unary_ops(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 8, 8)
        positive = torch.rand(2, 3, 8, 8)

        cases = [
            ("add_tensor", lambda t: t + t, (x,)),
            ("sub_scalar", lambda t: t - 0.25, (x,)),
            ("mul_tensor", lambda t: t * t, (x,)),
            ("div_scalar", lambda t: t / 2.0, (x,)),
            ("relu", F.relu, (x,)),
            ("hardtanh", lambda t: F.hardtanh(t, -0.5, 0.5), (x,)),
            ("sigmoid", torch.sigmoid, (x,)),
            ("exp", torch.exp, (x,)),
            ("sqrt", lambda t: torch.sqrt(t + 1e-3), (positive,)),
            ("clamp", lambda t: torch.clamp(t, -0.2, 0.3), (x,)),
        ]

        for name, fn, args in cases:
            with self.subTest(case=name):
                self._assert_vulkan_matches_cpu(fn, *args)

        with self.subTest(case="gelu_default"):
            self._assert_vulkan_matches_cpu(
                lambda t: F.gelu(t),
                x,
                atol=1e-3,
                rtol=1e-3)

    def test_reduction_and_shape_ops(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 8, 8)
        x_small = torch.randn(2, 3, 8)
        x_broadcast = torch.randn(1, 3, 1, 8)
        mask = torch.randn(2, 3, 8, 8) > 0

        cases = [
            ("mean_hw", lambda t: t.mean(dim=(2, 3)), (x,)),
            ("sum_hw", lambda t: t.sum(dim=(2, 3)), (x,)),
            ("permute_nhwc", lambda t: t.permute(0, 2, 3, 1), (x,)),
            ("transpose_channels_width", lambda t: t.transpose(1, 3), (x,)),
            ("slice_hw", lambda t: t[:, :, 1:5, 2:6], (x,)),
            ("select_height", lambda t: t.select(2, 3), (x,)),
            ("repeat_channels", lambda t: t.repeat(1, 2, 1, 1), (x,)),
            ("expand_broadcast", lambda t: t.expand(2, 3, 4, 8), (x_broadcast,)),
            ("cat_channels", lambda a, b: torch.cat([a, b], dim=1), (x, x)),
            ("stack_3d", lambda a, b: torch.stack([a, b], dim=0), (x_small, x_small)),
            ("softmax_channels", lambda t: torch.softmax(t, dim=1), (x,)),
            ("masked_fill", lambda t, m: t.masked_fill(m, 0.2), (x, mask)),
        ]

        for name, fn, args in cases:
            with self.subTest(case=name):
                self._assert_vulkan_matches_cpu(fn, *args)

    def test_view_ops_with_preexisting_vulkan_input_in_inference_mode(self):
        torch.manual_seed(0)
        x_cpu = torch.randn(8, 32)
        x_vulkan = x_cpu.to("vulkan")

        with torch.inference_mode():
            cases = [
                ("t", lambda t: t.t()),
                ("transpose", lambda t: t.transpose(0, 1)),
                ("permute", lambda t: t.permute(1, 0)),
                ("reshape", lambda t: t.reshape(4, 64)),
                ("view", lambda t: t.view(4, 64)),
                ("flatten", lambda t: t.flatten()),
                ("slice", lambda t: t[:, :16]),
                ("unsqueeze", lambda t: t.unsqueeze(0)),
            ]

            for name, fn in cases:
                with self.subTest(case=name):
                    expected = fn(x_cpu)
                    actual_vulkan = fn(x_vulkan)
                    self.assertFalse(actual_vulkan.is_inference())
                    actual = actual_vulkan.cpu()
                    self._assert_outputs_close(
                        expected,
                        actual,
                        atol=1e-4,
                        rtol=1e-4)

    def test_linear_algebra_ops(self):
        torch.manual_seed(0)
        a = torch.randn(4, 5)
        b = torch.randn(5, 3)
        batch_a = torch.randn(2, 4, 5)
        batch_b = torch.randn(2, 5, 3)
        batch_input = torch.randn(2, 4, 3)

        cases = [
            ("mm", torch.mm, (a, b)),
            ("addmm", torch.addmm, (torch.randn(4, 3), a, b)),
            ("bmm", torch.bmm, (batch_a, batch_b)),
            ("baddbmm", torch.baddbmm, (batch_input, batch_a, batch_b)),
        ]

        for name, fn, args in cases:
            with self.subTest(case=name):
                self._assert_vulkan_matches_cpu(fn, *args)

    def test_mm_and_addmm_with_transposed_vulkan_weight(self):
        torch.manual_seed(0)
        x_cpu = torch.randn(16, 32)
        weight_cpu = torch.randn(8, 32)
        bias_cpu = torch.randn(8)

        x_vulkan = x_cpu.to("vulkan")
        weight_vulkan = weight_cpu.to("vulkan")
        bias_vulkan = bias_cpu.to("vulkan")
        weight_vulkan_t = weight_vulkan.t()

        with torch.inference_mode():
            expected_mm = torch.mm(x_cpu, weight_cpu.t())
            actual_mm = torch.mm(x_vulkan, weight_vulkan_t).cpu()
            self._assert_outputs_close(
                expected_mm,
                actual_mm,
                atol=1e-4,
                rtol=1e-4)

            expected_addmm = torch.addmm(bias_cpu, x_cpu, weight_cpu.t())
            actual_addmm = torch.addmm(
                bias_vulkan,
                x_vulkan,
                weight_vulkan_t).cpu()
            self._assert_outputs_close(
                expected_addmm,
                actual_addmm,
                atol=1e-4,
                rtol=1e-4)

    def test_bmm_and_baddbmm_with_transposed_vulkan_weight(self):
        torch.manual_seed(0)
        batch_a_cpu = torch.randn(2, 4, 5)
        batch_b_cpu = torch.randn(2, 3, 5)
        bias_cpu = torch.randn(2, 4, 3)

        batch_a_vulkan = batch_a_cpu.to("vulkan")
        batch_b_vulkan = batch_b_cpu.to("vulkan")
        batch_b_vulkan_t = batch_b_vulkan.transpose(1, 2)
        bias_vulkan = bias_cpu.to("vulkan")

        with torch.inference_mode():
            expected_bmm = torch.bmm(batch_a_cpu, batch_b_cpu.transpose(1, 2))
            actual_bmm = torch.bmm(batch_a_vulkan, batch_b_vulkan_t).cpu()
            self._assert_outputs_close(
                expected_bmm,
                actual_bmm,
                atol=1e-4,
                rtol=1e-4)

            expected_baddbmm = torch.baddbmm(
                bias_cpu,
                batch_a_cpu,
                batch_b_cpu.transpose(1, 2))
            actual_baddbmm = torch.baddbmm(
                bias_vulkan,
                batch_a_vulkan,
                batch_b_vulkan_t).cpu()
            self._assert_outputs_close(
                expected_baddbmm,
                actual_baddbmm,
                atol=1e-4,
                rtol=1e-4)

    def test_nn_inference_ops(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 8, 8)
        conv_x = torch.randn(1, 3, 8, 8)
        conv_weight = torch.randn(4, 3, 3, 3)
        conv_bias = torch.randn(4)
        norm_x = torch.randn(2, 4, 8)
        norm_weight = torch.ones(8)
        norm_bias = torch.zeros(8)

        cases = [
            (
                "conv2d_functional",
                lambda t: F.conv2d(t, conv_weight, conv_bias, padding=1),
                (conv_x,),
                1e-4,
                1e-4,
            ),
            (
                "conv2d_functional_no_bias",
                lambda t: F.conv2d(t, conv_weight, None, padding=1),
                (conv_x,),
                1e-4,
                1e-4,
            ),
            ("avg_pool2d", lambda t: F.avg_pool2d(t, 2), (x,), 1e-4, 1e-4),
            ("max_pool2d", lambda t: F.max_pool2d(t, 2), (x,), 1e-4, 1e-4),
            (
                "adaptive_avg_pool2d",
                lambda t: F.adaptive_avg_pool2d(t, (1, 1)),
                (x,),
                1e-4,
                1e-4,
            ),
            (
                "upsample_nearest2d",
                lambda t: F.interpolate(t, scale_factor=2.0, mode="nearest"),
                (x,),
                1e-4,
                1e-4,
            ),
            (
                "upsample_bilinear2d",
                lambda t: F.interpolate(
                    t,
                    size=(10, 10),
                    mode="bilinear",
                    align_corners=False),
                (x,),
                1e-4,
                1e-4,
            ),
            (
                "layer_norm",
                lambda t: F.layer_norm(t, (8,), norm_weight, norm_bias, 1e-5),
                (norm_x,),
                1e-4,
                1e-4,
            ),
            (
                "native_layer_norm",
                lambda t: torch.native_layer_norm(
                    t,
                    (8,),
                    norm_weight,
                    norm_bias,
                    1e-5)[0],
                (norm_x,),
                1e-4,
                1e-4,
            ),
        ]

        for name, fn, args, atol, rtol in cases:
            with self.subTest(case=name):
                self._assert_vulkan_matches_cpu(fn, *args, atol=atol, rtol=rtol)

    def test_conv2d_module_with_vulkan_weights(self):
        torch.manual_seed(0)

        module_cpu = torch.nn.Conv2d(
            3,
            12,
            kernel_size=4,
            stride=4,
            bias=True).eval()
        module_vulkan = torch.nn.Conv2d(
            3,
            12,
            kernel_size=4,
            stride=4,
            bias=True).eval()
        module_vulkan.load_state_dict(module_cpu.state_dict())
        module_vulkan = module_vulkan.to("vulkan")

        module_cpu_nobias = torch.nn.Conv2d(
            3,
            12,
            kernel_size=4,
            stride=4,
            bias=False).eval()
        module_vulkan_nobias = torch.nn.Conv2d(
            3,
            12,
            kernel_size=4,
            stride=4,
            bias=False).eval()
        module_vulkan_nobias.load_state_dict(module_cpu_nobias.state_dict())
        module_vulkan_nobias = module_vulkan_nobias.to("vulkan")

        x_cpu = torch.randn(1, 3, 16, 20)
        x_vulkan = x_cpu.to("vulkan")

        with torch.inference_mode():
            expected = module_cpu(x_cpu)
            actual = module_vulkan(x_vulkan).cpu()
            self._assert_outputs_close(
                expected,
                actual,
                atol=1e-4,
                rtol=1e-4)

            actual_functional = F.conv2d(
                x_vulkan,
                module_vulkan.weight,
                module_vulkan.bias,
                stride=4).cpu()
            self._assert_outputs_close(
                expected,
                actual_functional,
                atol=1e-4,
                rtol=1e-4)

            expected_nobias = module_cpu_nobias(x_cpu)
            actual_nobias = module_vulkan_nobias(x_vulkan).cpu()
            self._assert_outputs_close(
                expected_nobias,
                actual_nobias,
                atol=1e-4,
                rtol=1e-4)

    def test_large_pointwise_conv2d_module_with_vulkan_weights(self):
        torch.manual_seed(0)

        x_cpu = torch.randn(1, 384, 7, 9)
        x_vulkan = x_cpu.to("vulkan")

        for out_channels in (192, 384):
            with self.subTest(out_channels=out_channels):
                module_cpu = torch.nn.Conv2d(
                    384,
                    out_channels,
                    kernel_size=1,
                    bias=True).eval()
                module_vulkan = torch.nn.Conv2d(
                    384,
                    out_channels,
                    kernel_size=1,
                    bias=True).eval()
                module_vulkan.load_state_dict(module_cpu.state_dict())
                module_vulkan = module_vulkan.to("vulkan")

                with torch.inference_mode():
                    expected = module_cpu(x_cpu)
                    actual = module_vulkan(x_vulkan).cpu()
                    self._assert_outputs_close(
                        expected,
                        actual,
                        atol=1e-4,
                        rtol=1e-4)

    def test_large_pointwise_conv_weight_roundtrip(self):
        torch.manual_seed(0)

        for out_channels in (192, 384):
            with self.subTest(out_channels=out_channels):
                weight_cpu = torch.randn(out_channels, 384, 1, 1)

                with torch.inference_mode():
                    weight_vulkan = weight_cpu.to("vulkan")
                    roundtrip = weight_vulkan.cpu()
                    self._assert_outputs_close(
                        weight_cpu,
                        roundtrip,
                        atol=1e-4,
                        rtol=1e-4)

    def test_large_spatial_conv2d_module_with_vulkan_weights(self):
        torch.manual_seed(0)

        x_cpu = torch.randn(1, 384, 37, 56)
        x_vulkan = x_cpu.to("vulkan")

        module_cpu = torch.nn.Conv2d(
            384,
            384,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True).eval()
        module_vulkan = torch.nn.Conv2d(
            384,
            384,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True).eval()
        module_vulkan.load_state_dict(module_cpu.state_dict())
        module_vulkan = module_vulkan.to("vulkan")

        with torch.inference_mode():
            expected = module_cpu(x_cpu)
            actual = module_vulkan(x_vulkan).cpu()
            self._assert_outputs_close(
                expected,
                actual,
                atol=1e-4,
                rtol=1e-4)

    def test_large_spatial_conv_weight_roundtrip(self):
        torch.manual_seed(0)

        weight_cpu = torch.randn(384, 384, 3, 3)

        with torch.inference_mode():
            weight_vulkan = weight_cpu.to("vulkan")
            roundtrip = weight_vulkan.cpu()
            self._assert_outputs_close(
                weight_cpu,
                roundtrip,
                atol=1e-4,
                rtol=1e-4)

    def test_small_conv_block(self):
        torch.manual_seed(0)

        class SmallConvBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
                self.linear_weight = torch.nn.Parameter(
                    torch.randn(8, 64),
                    requires_grad=False)
                self.linear_bias = torch.nn.Parameter(
                    torch.randn(8),
                    requires_grad=False)

            def forward(self, x):
                x = self.conv(x)
                x = F.relu(x)
                x = F.avg_pool2d(x, 2)
                x = x.flatten(1)
                return F.linear(x, self.linear_weight, self.linear_bias)

        module = SmallConvBlock().eval()
        x = torch.randn(1, 3, 8, 8)
        self._assert_vulkan_matches_cpu(module, x, atol=1e-4, rtol=1e-4)

    def test_linear_with_preexisting_vulkan_input(self):
        torch.manual_seed(0)
        module = torch.nn.Linear(32, 16).eval()
        x_cpu = torch.randn(1, 16, 32)
        x_vulkan = x_cpu.to("vulkan")

        with torch.inference_mode():
            expected = module(x_cpu)
            actual = module(x_vulkan).cpu()

            self._assert_outputs_close(
                expected,
                actual,
                atol=1e-4,
                rtol=1e-4)

            expected_nobias = F.linear(x_cpu, module.weight, None)
            actual_nobias = F.linear(x_vulkan, module.weight, None).cpu()
            self._assert_outputs_close(
                expected_nobias,
                actual_nobias,
                atol=1e-4,
                rtol=1e-4)

    def test_linear_module_with_vulkan_weights(self):
        torch.manual_seed(0)
        module_cpu = torch.nn.Linear(32, 16).eval()
        module_vulkan = torch.nn.Linear(32, 16).eval()
        module_vulkan.load_state_dict(module_cpu.state_dict())
        module_vulkan = module_vulkan.to("vulkan")

        x_cpu = torch.randn(1, 16, 32)
        x_vulkan = x_cpu.to("vulkan")

        with torch.inference_mode():
            expected = module_cpu(x_cpu)
            actual = module_vulkan(x_vulkan).cpu()
            self._assert_outputs_close(
                expected,
                actual,
                atol=1e-4,
                rtol=1e-4)

            expected_nobias = F.linear(x_cpu, module_cpu.weight, None)
            actual_nobias = F.linear(x_vulkan, module_vulkan.weight, None).cpu()
            self._assert_outputs_close(
                expected_nobias,
                actual_nobias,
                atol=1e-4,
                rtol=1e-4)

            linear_context = torch.ops.vulkan_prepack.create_linear_context(
                module_vulkan.weight.clone().t(),
                module_vulkan.bias)
            actual_prepack = torch.ops.vulkan_prepack.run_linear_context(
                x_vulkan,
                linear_context).cpu()
            self._assert_outputs_close(
                expected,
                actual_prepack,
                atol=1e-4,
                rtol=1e-4)

            labeled_context = torch.ops.vulkan_prepack.create_linear_context_labeled(
                module_vulkan.weight,
                module_vulkan.bias,
                "test_linear")
            actual_labeled = torch.ops.vulkan_prepack.run_linear_context(
                x_vulkan,
                labeled_context).cpu()
            self._assert_outputs_close(
                expected,
                actual_labeled,
                atol=1e-4,
                rtol=1e-4)

            expected_gelu = F.gelu(expected)
            actual_gelu = torch.ops.vulkan_prepack.run_linear_gelu_context(
                x_vulkan,
                labeled_context).cpu()
            self._assert_outputs_close(
                expected_gelu,
                actual_gelu,
                atol=1e-4,
                rtol=1e-4)

    def test_layer_norm_then_linear_in_inference_mode(self):
        torch.manual_seed(0)
        x_cpu = torch.randn(1, 16, 32)
        norm_weight = torch.randn(32)
        norm_bias = torch.randn(32)

        module_cpu = torch.nn.Linear(32, 64).eval()
        module_vulkan = torch.nn.Linear(32, 64).eval()
        module_vulkan.load_state_dict(module_cpu.state_dict())
        module_vulkan = module_vulkan.to("vulkan")

        x_vulkan = x_cpu.to("vulkan")

        with torch.inference_mode():
            expected = module_cpu(
                F.layer_norm(x_cpu, (32,), norm_weight, norm_bias, 1e-5))

            normalized_vulkan = F.layer_norm(
                x_vulkan,
                (32,),
                norm_weight,
                norm_bias,
                1e-5)
            actual = module_vulkan(normalized_vulkan).cpu()

            self._assert_outputs_close(
                expected,
                actual,
                atol=1e-4,
                rtol=1e-4)

    def test_repeated_transformer_block_in_inference_mode(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            repo_root
            if not existing_pythonpath
            else repo_root + os.pathsep + existing_pythonpath
        )

        script = textwrap.dedent(
            """
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class TinyTransformerBlock(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(64, eps=1e-6)
                    self.qkv = nn.Linear(64, 64 * 3, bias=True)
                    self.proj = nn.Linear(64, 64, bias=True)
                    self.norm2 = nn.LayerNorm(64, eps=1e-6)
                    self.fc1 = nn.Linear(64, 256, bias=True)
                    self.fc2 = nn.Linear(256, 64, bias=True)

                def forward(self, x):
                    residual = x
                    qkv = self.qkv(self.norm1(x)).reshape(1, 257, 3, 64)
                    q = qkv[:, :, 0].reshape(1, 257, 4, 16)
                    k = qkv[:, :, 1].reshape(1, 257, 4, 16)
                    v = qkv[:, :, 2].reshape(1, 257, 4, 16)
                    q = q.permute(0, 2, 1, 3).reshape(4, 257, 16) * (16 ** -0.5)
                    k = k.permute(0, 2, 1, 3).reshape(4, 257, 16)
                    v = v.permute(0, 2, 1, 3).reshape(4, 257, 16)
                    x = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=0.0,
                        is_causal=False,
                        scale=1.0).reshape(1, 4, 257, 16)
                    x = x.permute(0, 2, 1, 3).reshape(1, 257, 64)
                    x = residual + self.proj(x)
                    residual = x
                    x = self.norm2(x)
                    x = self.fc2(F.gelu(self.fc1(x)))
                    return residual + x

            block = TinyTransformerBlock().eval().to("vulkan")
            x = torch.randn(1, 257, 64, dtype=torch.float32).to("vulkan")

            with torch.inference_mode():
                for _ in range(12):
                    x = block(x)

                print(float(x.cpu().mean()))
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=(
                "Repeated inference transformer block crashed on Vulkan.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            ),
        )

    def test_scaled_dot_product_attention(self):
        torch.manual_seed(0)
        cases = [
            (
                "sdpa_4d",
                torch.randn(1, 2, 9, 8),
                torch.randn(1, 2, 7, 8),
                torch.randn(1, 2, 7, 8),
            ),
            (
                "sdpa_3d",
                torch.randn(2, 9, 8),
                torch.randn(2, 7, 8),
                torch.randn(2, 7, 8),
            ),
            (
                "sdpa_3d_batchpacked_head64",
                torch.randn(6, 33, 64),
                torch.randn(6, 29, 64),
                torch.randn(6, 29, 64),
            ),
            (
                "sdpa_4d_transformerish",
                torch.randn(1, 6, 33, 64),
                torch.randn(1, 6, 29, 64),
                torch.randn(1, 6, 29, 64),
            ),
        ]

        with torch.inference_mode():
            for name, query, key, value in cases:
                with self.subTest(case=name):
                    expected = F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        dropout_p=0.0,
                        scale=0.125)
                    actual = F.scaled_dot_product_attention(
                        query.to("vulkan"),
                        key.to("vulkan"),
                        value.to("vulkan"),
                        dropout_p=0.0,
                        scale=0.125).cpu()
                    self._assert_outputs_close(
                        expected,
                        actual,
                        atol=1e-4,
                        rtol=1e-4)

                    expected_math = torch.ops.aten._scaled_dot_product_attention_math(
                        query,
                        key,
                        value,
                        None,
                        0.0,
                        False,
                        None,
                        scale=0.125,
                        enable_gqa=False)[0]
                    actual_math = torch.ops.aten._scaled_dot_product_attention_math(
                        query.to("vulkan"),
                        key.to("vulkan"),
                        value.to("vulkan"),
                        None,
                        0.0,
                        False,
                        None,
                        scale=0.125,
                        enable_gqa=False)[0].cpu()
                    self._assert_outputs_close(
                        expected_math,
                        actual_math,
                        atol=1e-4,
                        rtol=1e-4)

    def test_depth_anything_v2_style_dpt_decoder(self):
        torch.manual_seed(0)
        module = DepthAnythingStyleMiniDPTHead(
            use_clstoken=False,
            scratch_bias=False).eval()
        features = self._make_depth_anything_style_features()

        self._assert_vulkan_matches_cpu(
            lambda feats: module(feats, 4, 4),
            features,
            atol=1e-4,
            rtol=1e-4)

    def test_depth_anything_v2_style_cls_readout(self):
        torch.manual_seed(0)
        module = DepthAnythingStyleReadoutProject(embed_dim=16).eval()
        tokens = torch.randn(1, 16, 16)
        cls_token = torch.randn(1, 16)

        self._assert_vulkan_matches_cpu(
            module,
            tokens,
            cls_token,
            atol=1e-3,
            rtol=1e-3)

    def test_depth_anything_v2_style_dpt_decoder_with_cls_readout(self):
        torch.manual_seed(0)
        module = DepthAnythingStyleMiniDPTHead(
            use_clstoken=True,
            scratch_bias=False).eval()
        features = self._make_depth_anything_style_features(use_clstoken=True)

        self._assert_vulkan_matches_cpu(
            lambda feats: module(feats, 4, 4),
            features,
            atol=1e-3,
            rtol=1e-3)

    def test_known_limitations(self):
        torch.manual_seed(0)
        x4 = torch.randn(2, 3, 8, 8)
        cases = [
            (
                "unsqueeze_4d",
                lambda t: t.unsqueeze(0),
                (x4,),
                RuntimeError,
                "Vulkan unsqueeze only supports up to 3d tensors as input",
            ),
            (
                "stack_4d",
                lambda a, b: torch.stack([a, b], dim=0),
                (x4, x4),
                RuntimeError,
                "Vulkan stack only supports up to 3d tensors as input",
            ),
            (
                "as_strided",
                lambda t: torch.as_strided(t, (2, 3, 4, 4), (96, 32, 8, 1)),
                (x4,),
                NotImplementedError,
                "Could not run 'aten::as_strided'",
            ),
        ]

        for name, fn, args, exc_type, message in cases:
            with self.subTest(case=name):
                self._assert_known_limitation(
                    fn,
                    *args,
                    exc_type=exc_type,
                    message=message)

if __name__ == "__main__":
    run_tests()
