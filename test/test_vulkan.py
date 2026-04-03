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
            ("fill_scalar", lambda t: t.clone().fill_(0.25), (x,)),
            ("relu", F.relu, (x,)),
            ("hardtanh", lambda t: F.hardtanh(t, -0.5, 0.5), (x,)),
            ("sigmoid", torch.sigmoid, (x,)),
            ("softplus_default", F.softplus, (x,)),
            ("softplus_custom", lambda t: F.softplus(t, beta=0.75, threshold=10.0), (x,)),
            ("exp", torch.exp, (x,)),
            ("sin", torch.sin, (x,)),
            ("cos", torch.cos, (x,)),
            ("neg", torch.neg, (x,)),
            ("sqrt", lambda t: torch.sqrt(t + 1e-3), (positive,)),
            ("rsqrt", lambda t: torch.rsqrt(t + 1e-3), (positive,)),
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

    def test_large_buffer_backed_binary_and_unary_ops(self):
        torch.manual_seed(0)
        x = torch.randn(2048, 1024)
        y = torch.randn(2048, 1024)

        with torch.inference_mode():
            x_vulkan = x.to("vulkan")
            y_vulkan = y.to("vulkan")

            self._assert_outputs_close(
                torch.exp(x),
                torch.exp(x_vulkan).cpu(),
                atol=1e-4,
                rtol=1e-4)
            self._assert_outputs_close(
                x + y,
                (x_vulkan + y_vulkan).cpu(),
                atol=1e-4,
                rtol=1e-4)
            self._assert_outputs_close(
                x * y,
                (x_vulkan * y_vulkan).cpu(),
                atol=1e-4,
                rtol=1e-4)

    def test_large_buffer_backed_full_reductions(self):
        torch.manual_seed(0)
        x = torch.randn(2048, 1024)

        with torch.inference_mode():
            x_vulkan = x.to("vulkan")
            self._assert_outputs_close(
                x.sum(),
                x_vulkan.sum().cpu(),
                atol=1e-4,
                rtol=1e-4)
            self._assert_outputs_close(
                x.mean(),
                x_vulkan.mean().cpu(),
                atol=1e-4,
                rtol=1e-4)

    def test_large_buffer_backed_dim_reductions(self):
        torch.manual_seed(0)
        x = torch.randn(2048, 1024)
        x_odd = torch.randn(1025, 1027)
        xb = torch.randn(512, 512, dtype=torch.bfloat16)
        xb_odd = torch.randn(257, 259, dtype=torch.bfloat16)

        with torch.inference_mode():
            x_vulkan = x.to("vulkan")
            x_odd_vulkan = x_odd.to("vulkan")
            xb_vulkan = xb.to("vulkan")
            xb_odd_vulkan = xb_odd.to("vulkan")

            self._assert_outputs_close(
                x.sum(dim=1),
                x_vulkan.sum(dim=1).cpu(),
                atol=1e-4,
                rtol=1e-4)
            self._assert_outputs_close(
                x.sum(dim=1, keepdim=True),
                x_vulkan.sum(dim=1, keepdim=True).cpu(),
                atol=1e-4,
                rtol=1e-4)
            self._assert_outputs_close(
                x.mean(dim=0),
                x_vulkan.mean(dim=0).cpu(),
                atol=1e-4,
                rtol=1e-4)
            self._assert_outputs_close(
                x.sum(dim=(0, 1)),
                x_vulkan.sum(dim=(0, 1)).cpu(),
                atol=1e-4,
                rtol=1e-4)
            self._assert_outputs_close(
                x_odd.sum(dim=1),
                x_odd_vulkan.sum(dim=1).cpu(),
                atol=1e-4,
                rtol=1e-4)
            self._assert_outputs_close(
                x_odd.mean(dim=0, keepdim=True),
                x_odd_vulkan.mean(dim=0, keepdim=True).cpu(),
                atol=1e-4,
                rtol=1e-4)
            self._assert_outputs_close(
                x_odd.sum(dim=(0, 1)),
                x_odd_vulkan.sum(dim=(0, 1)).cpu(),
                atol=1e-4,
                rtol=1e-4)
            self._assert_outputs_close(
                xb.mean(dim=1, dtype=torch.float32),
                xb_vulkan.mean(dim=1, dtype=torch.float32).cpu(),
                atol=1e-2,
                rtol=1e-2)
            self._assert_outputs_close(
                xb_odd.mean(dim=1, dtype=torch.float32),
                xb_odd_vulkan.mean(dim=1, dtype=torch.float32).cpu(),
                atol=1e-2,
                rtol=1e-2)

    def test_large_buffer_backed_metadata_views(self):
        torch.manual_seed(0)
        x = torch.randn(1025, 1027)

        with torch.inference_mode():
            x_vulkan = x.to("vulkan")

            expected_slice = x[3:1000:2, 5:1020:3]
            actual_slice = x_vulkan[3:1000:2, 5:1020:3]
            self._assert_outputs_close(
                expected_slice,
                actual_slice.cpu(),
                atol=1e-4,
                rtol=1e-4)
            self._assert_outputs_close(
                expected_slice.sum(dim=1),
                actual_slice.sum(dim=1).cpu(),
                atol=1e-4,
                rtol=1e-4)
            self._assert_outputs_close(
                expected_slice.mean(dim=0),
                actual_slice.mean(dim=0).cpu(),
                atol=1e-4,
                rtol=1e-4)

            expected_select = x.select(0, 17)
            actual_select = x_vulkan.select(0, 17)
            self._assert_outputs_close(
                expected_select,
                actual_select.cpu(),
                atol=1e-4,
                rtol=1e-4)
            self._assert_outputs_close(
                expected_select.exp(),
                actual_select.exp().cpu(),
                atol=1e-4,
                rtol=1e-4)

            expected_as_strided = torch.as_strided(
                x,
                (128, 96),
                (1027, 2),
                storage_offset=9)
            actual_as_strided = torch.as_strided(
                x_vulkan,
                (128, 96),
                (1027, 2),
                storage_offset=9)
            self._assert_outputs_close(
                expected_as_strided,
                actual_as_strided.cpu(),
                atol=1e-4,
                rtol=1e-4)
            self._assert_outputs_close(
                expected_as_strided.sum(dim=1),
                actual_as_strided.sum(dim=1).cpu(),
                atol=1e-4,
                rtol=1e-4)

    def test_reduction_dtype_resolution_and_buffer_cast(self):
        with torch.inference_mode():
            ints = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
            bf16 = torch.randn(512, 512, dtype=torch.bfloat16)

            self._assert_outputs_close(
                ints.sum(),
                ints.to("vulkan").sum().cpu())
            self._assert_outputs_close(
                ints.mean(dtype=torch.float32),
                ints.to("vulkan").mean(dtype=torch.float32).cpu(),
                atol=1e-5,
                rtol=1e-5)
            self._assert_outputs_close(
                bf16.mean(),
                bf16.to("vulkan").mean().cpu(),
                atol=1e-2,
                rtol=1e-2)

    def test_buffer_cast_matrix_core(self):
        torch.manual_seed(0)
        floats = (torch.randn(513, 257) * 8.0).clamp(-16.0, 16.0)
        ints = torch.randint(-32, 32, (513, 257), dtype=torch.int32)
        longs = torch.randint(-64, 64, (64, 64), dtype=torch.int64)
        bf16 = (torch.randn(513, 257) * 4.0).to(torch.bfloat16)
        large_floats = (torch.randn(2048, 1024) * 8.0).clamp(-16.0, 16.0)
        large_ints = torch.randint(-32, 32, (2048, 1024), dtype=torch.int32)

        with torch.inference_mode():
            floats_vulkan = floats.to("vulkan")
            ints_vulkan = ints.to("vulkan")
            longs_vulkan = longs.to("vulkan")
            bf16_vulkan = bf16.to("vulkan")
            large_floats_vulkan = large_floats.to("vulkan")
            large_ints_vulkan = large_ints.to("vulkan")

            floats_to_int = floats_vulkan.to(torch.int32)
            ints_to_float = ints_vulkan.to(torch.float32)
            longs_to_float = longs_vulkan.to(torch.float32)
            bf16_to_float = bf16_vulkan.to(torch.float32)
            large_floats_to_int = large_floats_vulkan.to(torch.int32)
            large_ints_to_float = large_ints_vulkan.to(torch.float32)
            floats_view_to_int = floats_vulkan[1:, 1:].to(torch.int32)
            bf16_view_to_float = bf16_vulkan[1:, 1:].to(torch.float32)

            self.assertEqual(floats_to_int.dtype, torch.int32)
            self.assertEqual(ints_to_float.dtype, torch.float32)
            self.assertEqual(longs_to_float.dtype, torch.float32)
            self.assertEqual(bf16_to_float.dtype, torch.float32)
            self.assertEqual(large_floats_to_int.dtype, torch.int32)
            self.assertEqual(large_ints_to_float.dtype, torch.float32)

            self._assert_outputs_close(
                ints,
                ints_vulkan.cpu())
            self._assert_outputs_close(
                floats.to(torch.int32),
                floats_to_int.cpu())
            self._assert_outputs_close(
                ints.to(torch.float32),
                ints_to_float.cpu(),
                atol=1e-5,
                rtol=1e-5)
            self._assert_outputs_close(
                longs.to(torch.float32),
                longs_to_float.cpu(),
                atol=1e-5,
                rtol=1e-5)
            self._assert_outputs_close(
                bf16.to(torch.float32),
                bf16_to_float.cpu(),
                atol=1e-2,
                rtol=1e-2)
            self._assert_outputs_close(
                large_floats.to(torch.int32),
                large_floats_to_int.cpu())
            self._assert_outputs_close(
                large_ints.to(torch.float32),
                large_ints_to_float.cpu(),
                atol=1e-5,
                rtol=1e-5)
            self._assert_outputs_close(
                floats[1:, 1:].to(torch.int32),
                floats_view_to_int.cpu())
            self._assert_outputs_close(
                bf16[1:, 1:].to(torch.float32),
                bf16_view_to_float.cpu(),
                atol=1e-2,
                rtol=1e-2)

            copy_dst_vulkan = torch.empty(
                floats.shape, device="vulkan", dtype=torch.int32)
            copy_dst_vulkan.copy_(floats_vulkan)
            self._assert_outputs_close(
                floats.to(torch.int32),
                copy_dst_vulkan.cpu())

            copy_dst_large_vulkan = torch.empty(
                large_floats.shape, device="vulkan", dtype=torch.int32)
            copy_dst_large_vulkan.copy_(large_floats_vulkan)
            self._assert_outputs_close(
                large_floats.to(torch.int32),
                copy_dst_large_vulkan.cpu())

            copy_dst_from_cpu = torch.empty(
                floats.shape, device="vulkan", dtype=torch.int32)
            copy_dst_from_cpu.copy_(floats)
            self._assert_outputs_close(
                floats.to(torch.int32),
                copy_dst_from_cpu.cpu())

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

    def test_fill_and_tril_factories(self):
        with torch.inference_mode():
            expected_ones = torch.ones(6, 6)
            actual_ones = torch.ones(6, 6, device="vulkan")
            self._assert_outputs_close(expected_ones, actual_ones)

        mat = torch.randn(6, 6)
        self._assert_vulkan_matches_cpu(lambda t: torch.tril(t, diagonal=-1), mat)

    def test_arange_factories(self):
        with torch.inference_mode():
            expected_default = torch.arange(7)
            actual_default = torch.arange(7, device="vulkan").cpu()
            self._assert_outputs_close(expected_default, actual_default)

            expected_step = torch.arange(1, 9, 2, dtype=torch.float32)
            actual_step = torch.arange(
                1, 9, 2, dtype=torch.float32, device="vulkan").cpu()
            self._assert_outputs_close(expected_step, actual_step)

            out = torch.empty(3, device="vulkan", dtype=torch.long)
            actual_out = torch.arange(2, 11, 3, out=out).cpu()
            expected_out = torch.arange(2, 11, 3, dtype=torch.long)
            self._assert_outputs_close(expected_out, actual_out)
            self._assert_outputs_close(expected_out, out.cpu())

    def test_linspace_factories(self):
        with torch.inference_mode():
            expected_default = torch.linspace(0.0, 1.0, 5)
            actual_default = torch.linspace(0.0, 1.0, 5, device="vulkan").cpu()
            self._assert_outputs_close(expected_default, actual_default)

            expected_typed = torch.linspace(-1.0, 1.0, 4, dtype=torch.float32)
            actual_typed = torch.linspace(
                -1.0, 1.0, 4, dtype=torch.float32, device="vulkan").cpu()
            self._assert_outputs_close(expected_typed, actual_typed)

            out = torch.empty(4, device="vulkan", dtype=torch.float32)
            actual_out = torch.linspace(-2.0, 2.0, 4, out=out).cpu()
            expected_out = torch.linspace(-2.0, 2.0, 4, dtype=torch.float32)
            self._assert_outputs_close(expected_out, actual_out)
            self._assert_outputs_close(expected_out, out.cpu())

    def test_argmax_matches_cpu(self):
        with torch.inference_mode():
            x = torch.randn(2, 3, 5)
            expected = torch.argmax(x, dim=-1)
            actual = torch.argmax(x.to("vulkan"), dim=-1).cpu()
            self._assert_outputs_close(expected, actual)

            out = torch.empty((2, 3), device="vulkan", dtype=torch.long)
            actual_out = torch.argmax(x.to("vulkan"), dim=-1, out=out).cpu()
            self._assert_outputs_close(expected, actual_out)
            self._assert_outputs_close(expected, out.cpu())

    def test_all_matches_cpu(self):
        with torch.inference_mode():
            x = torch.tensor([[True, True, True], [True, False, True]])
            expected = torch.all(x)
            actual = torch.all(x.to("vulkan")).cpu()
            self._assert_outputs_close(expected, actual)

            out = torch.empty((), device="vulkan", dtype=torch.bool)
            actual_out = torch.all(x.to("vulkan"), out=out).cpu()
            self._assert_outputs_close(expected, actual_out)
            self._assert_outputs_close(expected, out.cpu())

    def test_5d_tensor_roundtrip_uses_buffer_storage(self):
        with torch.inference_mode():
            expected = torch.randn(2, 3, 4, 5, 6)
            actual = expected.to("vulkan").cpu()
            self._assert_outputs_close(expected, actual)

            empty_vulkan = torch.empty(
                (2, 3, 4, 5, 6), device="vulkan", dtype=torch.float32)
            self.assertEqual(tuple(empty_vulkan.shape), (2, 3, 4, 5, 6))

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

    def test_as_strided_with_preexisting_vulkan_input_in_inference_mode(self):
        torch.manual_seed(0)
        x_cpu = torch.randn(2, 3, 8, 8)
        x_vulkan = x_cpu.to("vulkan")

        with torch.inference_mode():
            cases = [
                ("basic", (2, 3, 4, 4), (96, 32, 8, 1), None),
                ("storage_offset", (2, 3, 4, 4), (96, 32, 8, 1), 32),
            ]

            for name, size, stride, storage_offset in cases:
                with self.subTest(case=name):
                    expected = torch.as_strided(
                        x_cpu,
                        size,
                        stride,
                        storage_offset=storage_offset)
                    actual_vulkan = torch.as_strided(
                        x_vulkan,
                        size,
                        stride,
                        storage_offset=storage_offset)
                    self.assertFalse(actual_vulkan.is_inference())
                    actual = actual_vulkan.cpu()
                    self._assert_outputs_close(
                        expected,
                        actual,
                        atol=1e-4,
                        rtol=1e-4)

    def test_unsqueeze_4d_with_preexisting_vulkan_input_in_inference_mode(self):
        torch.manual_seed(0)
        x_cpu = torch.randn(2, 3, 8, 8)
        x_vulkan = x_cpu.to("vulkan")

        with torch.inference_mode():
            expected = x_cpu.unsqueeze(0)
            actual_vulkan = x_vulkan.unsqueeze(0)
            self.assertFalse(actual_vulkan.is_inference())
            actual = actual_vulkan.cpu()
            self._assert_outputs_close(
                expected,
                actual,
                atol=1e-4,
                rtol=1e-4)

    def test_unsqueeze_long_buffer_with_preexisting_vulkan_input_in_inference_mode(self):
        x_cpu = torch.arange(6, dtype=torch.long)
        x_vulkan = x_cpu.to("vulkan")

        with torch.inference_mode():
            expected = x_cpu.unsqueeze(0)
            actual_vulkan = x_vulkan.unsqueeze(0)
            self.assertFalse(actual_vulkan.is_inference())
            actual = actual_vulkan.cpu()
            self._assert_outputs_close(expected, actual)

    def test_long_buffer_to_float_after_unsqueeze_with_preexisting_vulkan_input(self):
        x_cpu = torch.arange(6, dtype=torch.long).unsqueeze(0)
        x_vulkan = x_cpu.to("vulkan")

        with torch.inference_mode():
            expected = x_cpu[:, None, :].float()
            actual = x_vulkan[:, None, :].float().cpu()
            self._assert_outputs_close(expected, actual)

    def test_rank3_vulkan_float_to_long_conversion(self):
        x_cpu = torch.zeros(1, 1, 2, dtype=torch.float32)
        x_vulkan = x_cpu.to("vulkan")

        with torch.inference_mode():
            expected = x_cpu.to(torch.long)
            actual = x_vulkan.to(torch.long).cpu()
            self._assert_outputs_close(expected, actual)

    def test_im2col_with_preexisting_vulkan_input_in_inference_mode(self):
        torch.manual_seed(0)
        x_cpu = torch.randn(1, 3, 8, 8)
        x_vulkan = x_cpu.to("vulkan")

        with torch.inference_mode():
            expected = F.unfold(
                x_cpu,
                kernel_size=(3, 3),
                dilation=(1, 1),
                padding=(1, 1),
                stride=(2, 2),
            )
            actual_vulkan = F.unfold(
                x_vulkan,
                kernel_size=(3, 3),
                dilation=(1, 1),
                padding=(1, 1),
                stride=(2, 2),
            )
            self.assertFalse(actual_vulkan.is_inference())
            actual = actual_vulkan.cpu()
            self._assert_outputs_close(
                expected,
                actual,
                atol=1e-4,
                rtol=1e-4)

    def test_view_then_scalar_mul_then_linear(self):
        torch.manual_seed(0)
        x = torch.randn(1, 4, 8)
        weight = torch.randn(6, 16)
        bias = torch.randn(6)

        def fn(t):
            t = t.view(2, 16)
            t = t * 0.5
            return F.linear(t, weight, bias)

        self._assert_vulkan_matches_cpu(fn, x, atol=1e-4, rtol=1e-4)

    def test_view_then_select_attention_style(self):
        torch.manual_seed(0)
        x = torch.randn(1, 17, 8)
        weight = torch.randn(24, 8)
        bias = torch.randn(24)

        def fn(t):
            qkv = F.linear(t, weight, bias).reshape(1, 17, 3, 8)
            q = qkv[:, :, 0].reshape(1, 17, 2, 4)
            q = q.permute(0, 2, 1, 3).reshape(2, 17, 4)
            return q

        self._assert_vulkan_matches_cpu(fn, x, atol=1e-4, rtol=1e-4)

    def test_index_select_dim0_with_vulkan_weight_and_cpu_indices(self):
        torch.manual_seed(0)
        weight_cpu = torch.randn(32, 16)
        weight_vulkan = weight_cpu.to("vulkan")
        indices = torch.tensor([0, 7, 31, 4, 12], dtype=torch.long)

        with torch.inference_mode():
            expected = weight_cpu.index_select(0, indices)
            actual = weight_vulkan.index_select(0, indices).cpu()
            self._assert_outputs_close(
                expected,
                actual,
                atol=1e-4,
                rtol=1e-4)

    def test_index_select_dim0_with_texture_derived_vulkan_weight_and_cpu_indices(self):
        torch.manual_seed(0)
        base_cpu = torch.randn(2212, 16)
        base_vulkan = base_cpu.to("vulkan")
        old_height = 47
        old_width = 47
        new_height = 47
        new_width = 71
        index_base = torch.arange(new_height * new_width + 3, dtype=torch.long)
        indices = index_base.repeat(25)

        with torch.inference_mode():
            old_sub_cpu = base_cpu[: old_height * old_width].reshape(
                1, old_width, old_height, -1).permute(0, 3, 1, 2)
            old_sub_vulkan = base_vulkan[: old_height * old_width].reshape(
                1, old_width, old_height, -1).permute(0, 3, 1, 2)

            new_sub_cpu = F.interpolate(
                old_sub_cpu,
                size=(new_height, new_width),
                mode="bilinear",
            )
            new_sub_vulkan = F.interpolate(
                old_sub_vulkan,
                size=(new_height, new_width),
                mode="bilinear",
            )

            weight_cpu = torch.cat(
                (
                    new_sub_cpu.permute(0, 2, 3, 1).reshape(new_height * new_width, -1),
                    base_cpu[old_height * old_width:],
                ),
                dim=0,
            )
            weight_vulkan = torch.cat(
                (
                    new_sub_vulkan.permute(0, 2, 3, 1).reshape(new_height * new_width, -1),
                    base_vulkan[old_height * old_width:],
                ),
                dim=0,
            )

            expected = weight_cpu.index_select(0, indices)
            actual = weight_vulkan.index_select(0, indices).cpu()
            self._assert_outputs_close(
                expected,
                actual,
                atol=1e-4,
                rtol=1e-4)

    def test_to_vulkan_labeled_roundtrip(self):
        torch.manual_seed(0)
        value = torch.randn(128, 64)

        with torch.inference_mode():
            value_vulkan = torch.ops.vulkan_prepack.to_vulkan_labeled(
                value,
                "test.weight",
            )
            self.assertTrue(value_vulkan.is_vulkan)
            self._assert_outputs_close(
                value,
                value_vulkan.cpu(),
                atol=1e-4,
                rtol=1e-4,
            )

    def test_index_select_dim0_with_large_buffer_backed_vulkan_weight_and_cpu_indices(self):
        torch.manual_seed(0)
        weight_cpu = torch.randn(17000, 256)
        weight_vulkan = weight_cpu.to("vulkan")
        indices = torch.tensor([0, 7, 31, 4, 12, 1024, 16000], dtype=torch.long)

        with torch.inference_mode():
            expected = weight_cpu.index_select(0, indices)
            actual = weight_vulkan.index_select(0, indices).cpu()
            self._assert_outputs_close(
                expected,
                actual,
                atol=1e-4,
                rtol=1e-4)

    def test_embedding_with_vulkan_weight_and_cpu_indices(self):
        torch.manual_seed(0)
        module_cpu = torch.nn.Embedding(64, 24).eval()
        module_vulkan = torch.nn.Embedding(64, 24).eval()
        module_vulkan.load_state_dict(module_cpu.state_dict())
        module_vulkan = module_vulkan.to("vulkan")
        indices = torch.tensor([[1, 5, 7, 2, 9, 4]], dtype=torch.long)

        with torch.inference_mode():
            expected = module_cpu(indices)
            actual = module_vulkan(indices).cpu()
            self._assert_outputs_close(
                expected,
                actual,
                atol=1e-4,
                rtol=1e-4)

            expected_functional = F.embedding(indices, module_cpu.weight)
            actual_functional = F.embedding(indices, module_vulkan.weight).cpu()
            self._assert_outputs_close(
                expected_functional,
                actual_functional,
                atol=1e-4,
                rtol=1e-4)

    def test_embedding_with_large_buffer_backed_vulkan_weight_and_cpu_indices(self):
        torch.manual_seed(0)
        module_cpu = torch.nn.Embedding(17000, 256).eval()
        module_vulkan = torch.nn.Embedding(17000, 256).eval()
        module_vulkan.load_state_dict(module_cpu.state_dict())
        module_vulkan = module_vulkan.to("vulkan")
        indices = torch.tensor([[1, 5, 7, 2, 9, 4, 1024, 16000]], dtype=torch.long)

        with torch.inference_mode():
            expected = module_cpu(indices)
            actual = module_vulkan(indices).cpu()
            self._assert_outputs_close(
                expected,
                actual,
                atol=1e-4,
                rtol=1e-4)

            expected_functional = F.embedding(indices, module_cpu.weight)
            actual_functional = F.embedding(indices, module_vulkan.weight).cpu()
            self._assert_outputs_close(
                expected_functional,
                actual_functional,
                atol=1e-4,
                rtol=1e-4)

    def test_long_tensor_roundtrip_and_zeros(self):
        src = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        vulkan = src.to("vulkan")

        self.assertEqual(vulkan.device.type, "vulkan")
        self.assertEqual(vulkan.cpu(), src)

        zeros = torch.zeros((2, 3), dtype=torch.long, device="vulkan")
        self.assertEqual(zeros.cpu(), torch.zeros((2, 3), dtype=torch.long))

        shifted = vulkan + 1
        self.assertEqual(shifted.cpu(), src + 1)

    def test_module_to_vulkan_with_long_buffer(self):
        class BufferModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("token_ids", torch.tensor([3, 1, 4, 1], dtype=torch.long))

        module = BufferModule().eval().to("vulkan")
        self.assertEqual(module.token_ids.device.type, "vulkan")
        self.assertEqual(
            module.token_ids.cpu(),
            torch.tensor([3, 1, 4, 1], dtype=torch.long),
        )

    def test_long_slice_and_select_with_vulkan_input(self):
        src = torch.arange(24, dtype=torch.long).reshape(2, 12)
        vulkan = src.to("vulkan")

        self.assertEqual(vulkan[:, 2:9:2].cpu(), src[:, 2:9:2])
        self.assertEqual(vulkan.select(1, 5).cpu(), src.select(1, 5))

    def test_bfloat16_tensor_roundtrip_and_zeros(self):
        src = torch.tensor([[1.0, -0.5, 3.25], [4.0, 5.5, -6.0]], dtype=torch.bfloat16)
        vulkan = src.to("vulkan")

        self.assertEqual(vulkan.device.type, "vulkan")
        self.assertEqual(vulkan.cpu(), src)

        zeros = torch.zeros((2, 3), dtype=torch.bfloat16, device="vulkan")
        self.assertEqual(zeros.cpu(), torch.zeros((2, 3), dtype=torch.bfloat16))

    def test_module_to_vulkan_with_bfloat16_buffer(self):
        class BufferModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "stats",
                    torch.tensor([1.0, -0.5, 0.25, 2.0], dtype=torch.bfloat16),
                )

        module = BufferModule().eval().to("vulkan")
        self.assertEqual(module.stats.device.type, "vulkan")
        self.assertEqual(
            module.stats.cpu(),
            torch.tensor([1.0, -0.5, 0.25, 2.0], dtype=torch.bfloat16),
        )

    def test_vulkan_autocast_context_is_available(self):
        self.assertTrue(torch.amp.is_autocast_available("vulkan"))
        with torch.autocast(device_type="vulkan", dtype=torch.float16):
            x = torch.randn(2, 3, device="vulkan")
            y = x + 1.0
        self.assertEqual(y.dtype, torch.float32)

    def test_vulkan_autocast_linear_runs_as_safe_noop(self):
        torch.manual_seed(0)
        module_cpu = nn.Linear(4, 3).eval()
        module_vulkan = nn.Linear(4, 3).eval()
        module_vulkan.load_state_dict(module_cpu.state_dict())
        module_vulkan = module_vulkan.to("vulkan")
        x_cpu = torch.randn(2, 4)
        x_vulkan = x_cpu.to("vulkan")

        with torch.inference_mode():
            expected = module_cpu(x_cpu)
            with torch.autocast(device_type="vulkan", dtype=torch.float16):
                actual = module_vulkan(x_vulkan)

        self.assertEqual(actual.dtype, torch.float32)
        self._assert_outputs_close(
            expected,
            actual,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_bfloat16_linear_widens_to_float_for_compute(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, dtype=torch.bfloat16)
        weight = torch.randn(3, 4, dtype=torch.bfloat16)
        bias = torch.randn(3, dtype=torch.bfloat16)
        x_vulkan = x.to("vulkan")
        weight_vulkan = weight.to("vulkan")
        bias_vulkan = bias.to("vulkan")

        with torch.inference_mode():
            expected = F.linear(x.float(), weight.float(), bias.float())
            actual = F.linear(
                x_vulkan,
                weight_vulkan,
                bias_vulkan,
            )

        self.assertEqual(actual.dtype, torch.float32)
        self._assert_outputs_close(expected, actual, atol=1e-4, rtol=1e-4)

    def test_bfloat16_linear_3d_native_buffer_compute(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
        weight = torch.randn(5, 4, dtype=torch.bfloat16)
        bias = torch.randn(5, dtype=torch.bfloat16)
        x_vulkan = x.to("vulkan")
        weight_vulkan = weight.to("vulkan")
        bias_vulkan = bias.to("vulkan")

        with torch.inference_mode():
            expected = F.linear(x.float(), weight.float(), bias.float())
            actual = F.linear(
                x_vulkan,
                weight_vulkan,
                bias_vulkan,
            )

        self.assertEqual(actual.dtype, torch.float32)
        self._assert_outputs_close(expected, actual, atol=1e-4, rtol=1e-4)

    def test_bfloat16_conv2d_widens_to_float_for_compute(self):
        torch.manual_seed(0)
        x = torch.randn(1, 3, 8, 8, dtype=torch.bfloat16)
        weight = torch.randn(4, 3, 3, 3, dtype=torch.bfloat16)
        bias = torch.randn(4, dtype=torch.bfloat16)
        x_vulkan = x.to("vulkan")
        weight_vulkan = weight.to("vulkan")
        bias_vulkan = bias.to("vulkan")

        with torch.inference_mode():
            expected = F.conv2d(x.float(), weight.float(), bias.float(), padding=1)
            actual = F.conv2d(
                x_vulkan,
                weight_vulkan,
                bias_vulkan,
                padding=1,
            )

        self.assertEqual(actual.dtype, torch.float32)
        self._assert_outputs_close(expected, actual, atol=1e-4, rtol=1e-4)

    def test_bfloat16_buffer_full_reductions(self):
        torch.manual_seed(0)
        x = torch.randn(513, 257, dtype=torch.bfloat16)
        x_vulkan = x.to("vulkan")

        with torch.inference_mode():
            expected_sum = torch.sum(x, dtype=torch.float32)
            actual_sum = torch.sum(x_vulkan, dtype=torch.float32).cpu()
            expected_mean = torch.mean(x, dtype=torch.float32)
            actual_mean = torch.mean(x_vulkan, dtype=torch.float32).cpu()

        self.assertEqual(actual_sum.dtype, torch.float32)
        self.assertEqual(actual_mean.dtype, torch.float32)
        self._assert_outputs_close(expected_sum, actual_sum, atol=1e-4, rtol=1e-4)
        self._assert_outputs_close(expected_mean, actual_mean, atol=1e-4, rtol=1e-4)

    def test_int32_buffer_binary_tensor_ops(self):
        torch.manual_seed(0)
        x = torch.randint(-16, 16, (513, 257), dtype=torch.int32)
        y = torch.randint(-8, 8, (513, 257), dtype=torch.int32)
        x_vulkan = x.to("vulkan")
        y_vulkan = y.to("vulkan")

        with torch.inference_mode():
            self._assert_outputs_close(x + y, (x_vulkan + y_vulkan).cpu())
            self._assert_outputs_close(x - y, (x_vulkan - y_vulkan).cpu())
            self._assert_outputs_close(x * y, (x_vulkan * y_vulkan).cpu())

    def test_int32_buffer_binary_scalar_ops(self):
        torch.manual_seed(0)
        x = torch.randint(-16, 16, (513, 257), dtype=torch.int32)
        x_vulkan = x.to("vulkan")

        with torch.inference_mode():
            self._assert_outputs_close(x + 3, (x_vulkan + 3).cpu())
            self._assert_outputs_close(x - 5, (x_vulkan - 5).cpu())
            self._assert_outputs_close(x * -2, (x_vulkan * -2).cpu())

    def test_int32_buffer_binary_ops_on_metadata_views(self):
        torch.manual_seed(0)
        x = torch.randint(-16, 16, (513, 257), dtype=torch.int32)
        y = torch.randint(-8, 8, (513, 257), dtype=torch.int32)
        x_view = x[:, 3:203]
        y_view = y[:, 5:205]
        x_vulkan = x.to("vulkan")[:, 3:203]
        y_vulkan = y.to("vulkan")[:, 5:205]

        with torch.inference_mode():
            self._assert_outputs_close(x_view + y_view, (x_vulkan + y_vulkan).cpu())
            self._assert_outputs_close(x_view - y_view, (x_vulkan - y_vulkan).cpu())
            self._assert_outputs_close(x_view * y_view, (x_vulkan * y_vulkan).cpu())
            self._assert_outputs_close(x_view + 3, (x_vulkan + 3).cpu())
            self._assert_outputs_close(x_view - 5, (x_vulkan - 5).cpu())
            self._assert_outputs_close(x_view * -2, (x_vulkan * -2).cpu())

    def test_int8_and_uint8_buffer_binary_ops(self):
        torch.manual_seed(0)
        cases = (
            (torch.int8, -64, 64, -7, 7),
            (torch.uint8, 0, 256, 0, 32),
        )
        with torch.inference_mode():
            for dtype, x_low, x_high, y_low, y_high in cases:
                x = torch.randint(x_low, x_high, (513, 200), dtype=dtype)
                y = torch.randint(y_low, y_high, (513, 200), dtype=dtype)
                x_vulkan = x.to("vulkan")
                y_vulkan = y.to("vulkan")

                self._assert_outputs_close(x + y, (x_vulkan + y_vulkan).cpu())
                self._assert_outputs_close(x - y, (x_vulkan - y_vulkan).cpu())
                self._assert_outputs_close(x * y, (x_vulkan * y_vulkan).cpu())
                self._assert_outputs_close(
                    torch.add(x, y, alpha=2),
                    torch.add(x_vulkan, y_vulkan, alpha=2).cpu(),
                )
                self._assert_outputs_close(
                    torch.sub(x, y, alpha=2),
                    torch.sub(x_vulkan, y_vulkan, alpha=2).cpu(),
                )
                self._assert_outputs_close(x + 3, (x_vulkan + 3).cpu())
                self._assert_outputs_close(x - 5, (x_vulkan - 5).cpu())
                self._assert_outputs_close(x * -2, (x_vulkan * -2).cpu())

    def test_int8_and_uint8_buffer_binary_ops_on_metadata_views(self):
        torch.manual_seed(0)
        cases = (
            (torch.int8, -64, 64, -7, 7),
            (torch.uint8, 0, 256, 0, 32),
        )
        with torch.inference_mode():
            for dtype, x_low, x_high, y_low, y_high in cases:
                x = torch.randint(x_low, x_high, (513, 257), dtype=dtype)
                y = torch.randint(y_low, y_high, (513, 257), dtype=dtype)
                x_view = x[:, 3:203]
                y_view = y[:, 5:205]
                x_vulkan = x.to("vulkan")[:, 3:203]
                y_vulkan = y.to("vulkan")[:, 5:205]

                self._assert_outputs_close(
                    x_view + y_view, (x_vulkan + y_vulkan).cpu()
                )
                self._assert_outputs_close(
                    x_view - y_view, (x_vulkan - y_vulkan).cpu()
                )
                self._assert_outputs_close(
                    x_view * y_view, (x_vulkan * y_vulkan).cpu()
                )
                self._assert_outputs_close(x_view + 3, (x_vulkan + 3).cpu())
                self._assert_outputs_close(x_view - 5, (x_vulkan - 5).cpu())
                self._assert_outputs_close(x_view * -2, (x_vulkan * -2).cpu())

    def test_bool_buffer_binary_ops(self):
        torch.manual_seed(0)
        x = torch.randint(0, 2, (513, 200), dtype=torch.int32).to(torch.bool)
        y = torch.randint(0, 2, (513, 200), dtype=torch.int32).to(torch.bool)
        x_vulkan = x.to("vulkan")
        y_vulkan = y.to("vulkan")

        with torch.inference_mode():
            self.assertTrue(torch.equal(x + y, (x_vulkan + y_vulkan).cpu()))
            self.assertTrue(torch.equal(x * y, (x_vulkan * y_vulkan).cpu()))
            self.assertTrue(
                torch.equal(
                    torch.add(x, y, alpha=0),
                    torch.add(x_vulkan, y_vulkan, alpha=0).cpu(),
                )
            )
            self.assertTrue(torch.equal(x + True, (x_vulkan + True).cpu()))
            self.assertTrue(torch.equal(x * False, (x_vulkan * False).cpu()))

    def test_bool_buffer_binary_ops_on_metadata_views(self):
        torch.manual_seed(0)
        x = torch.randint(0, 2, (513, 257), dtype=torch.int32).to(torch.bool)
        y = torch.randint(0, 2, (513, 257), dtype=torch.int32).to(torch.bool)
        x_view = x[:, 3:203]
        y_view = y[:, 5:205]
        x_vulkan = x.to("vulkan")[:, 3:203]
        y_vulkan = y.to("vulkan")[:, 5:205]

        with torch.inference_mode():
            self.assertTrue(torch.equal(x_view + y_view, (x_vulkan + y_vulkan).cpu()))
            self.assertTrue(torch.equal(x_view * y_view, (x_vulkan * y_vulkan).cpu()))
            self.assertTrue(torch.equal(x_view + True, (x_vulkan + True).cpu()))
            self.assertTrue(torch.equal(x_view * False, (x_vulkan * False).cpu()))

    def test_group_norm_with_vulkan_weights(self):
        torch.manual_seed(0)
        module_cpu = nn.GroupNorm(4, 8).eval()
        module_vulkan = nn.GroupNorm(4, 8).eval()
        module_vulkan.load_state_dict(module_cpu.state_dict())
        module_vulkan = module_vulkan.to("vulkan")
        x_cpu = torch.randn(2, 8, 5, 7)
        x_vulkan = x_cpu.to("vulkan")

        with torch.inference_mode():
            expected = module_cpu(x_cpu)
            actual = module_vulkan(x_vulkan).cpu()

        self._assert_outputs_close(expected, actual, atol=1e-4, rtol=1e-4)

    def test_permute_reshape_then_linear(self):
        torch.manual_seed(0)
        x = torch.randn(1, 2, 17, 8)
        weight = torch.randn(12, 16)
        bias = torch.randn(12)

        def fn(t):
            t = t.permute(0, 2, 1, 3).reshape(1, 17, 16)
            return F.linear(t, weight, bias)

        self._assert_vulkan_matches_cpu(fn, x, atol=1e-4, rtol=1e-4)

    def test_view_then_slice_tokens(self):
        torch.manual_seed(0)
        x = torch.randn(1, 19, 8)

        def fn(t):
            cls = t[:, :1]
            patches = t[:, 3:]
            return torch.cat([cls, patches], dim=1)

        self._assert_vulkan_matches_cpu(fn, x, atol=1e-4, rtol=1e-4)

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
                "upsample_bicubic2d_align_false",
                lambda t: F.interpolate(
                    t,
                    size=(10, 10),
                    mode="bicubic",
                    align_corners=False),
                (x,),
                1e-3,
                1e-3,
            ),
            (
                "upsample_bicubic2d_align_true",
                lambda t: F.interpolate(
                    t,
                    size=(10, 10),
                    mode="bicubic",
                    align_corners=True),
                (x,),
                1e-3,
                1e-3,
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

    def test_upsample_bicubic2d_out(self):
        torch.manual_seed(0)
        x_cpu = torch.randn(1, 3, 8, 8)
        x_vulkan = x_cpu.to("vulkan")

        with torch.inference_mode():
            expected = F.interpolate(
                x_cpu,
                size=(11, 13),
                mode="bicubic",
                align_corners=False)
            out_vulkan = torch.empty((1, 3, 11, 13), device="vulkan")
            actual = torch.ops.aten.upsample_bicubic2d.out(
                x_vulkan,
                [11, 13],
                False,
                None,
                None,
                out=out_vulkan).cpu()

        self._assert_outputs_close(expected, actual, atol=1e-3, rtol=1e-3)

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

    def test_permute_reshape_then_conv2d_module_with_vulkan_weights(self):
        torch.manual_seed(0)

        x_cpu = torch.randn(1, 37 * 56, 384)
        x_vulkan = x_cpu.to("vulkan")

        module_cpu = torch.nn.Conv2d(384, 48, kernel_size=1, bias=True).eval()
        module_vulkan = torch.nn.Conv2d(
            384,
            48,
            kernel_size=1,
            bias=True).eval()
        module_vulkan.load_state_dict(module_cpu.state_dict())
        module_vulkan = module_vulkan.to("vulkan")

        with torch.inference_mode():
            expected = module_cpu(x_cpu.permute(0, 2, 1).reshape(1, 384, 37, 56))
            actual = module_vulkan(
                x_vulkan.permute(0, 2, 1).reshape(1, 384, 37, 56)).cpu()
            self._assert_outputs_close(
                expected,
                actual,
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
                atol=3e-4,
                rtol=3e-3)

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

    def test_view_then_scaled_dot_product_attention(self):
        torch.manual_seed(0)
        query = torch.randn(1, 2, 9, 8)
        key = torch.randn(1, 2, 7, 8)
        value = torch.randn(1, 2, 7, 8)

        def fn(q, k, v):
            q = q.reshape(2, 9, 8)
            k = k.reshape(2, 7, 8)
            v = v.reshape(2, 7, 8)
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
                scale=0.125)

        self._assert_vulkan_matches_cpu(fn, query, key, value, atol=1e-4, rtol=1e-4)

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
                "stack_4d",
                lambda a, b: torch.stack([a, b], dim=0),
                (x4, x4),
                RuntimeError,
                "Vulkan stack only supports up to 3d tensors as input",
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
