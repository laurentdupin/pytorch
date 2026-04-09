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

    def _run_repo_python_subprocess(
            self,
            script,
            *,
            extra_env=None,
            timeout=120,
            error_prefix="Vulkan subprocess failed."):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            repo_root
            if not existing_pythonpath
            else repo_root + os.pathsep + existing_pythonpath
        )
        if extra_env:
            env.update(extra_env)

        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            env=env,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=(
                f"{error_prefix}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            ),
        )
        return repo_root, result

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
            ("silu", F.silu, (x,)),
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
                F.silu(x),
                F.silu(x_vulkan).cpu(),
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

    def test_pow_integer_scalar_negative_base(self):
        small = torch.linspace(-2.0, 2.0, steps=48, dtype=torch.float32).reshape(2, 3, 8)
        large = torch.linspace(-2.0, 2.0, steps=1025 * 1027, dtype=torch.float32).reshape(1025, 1027)

        with torch.inference_mode():
            for exponent in (2, 3, 4):
                with self.subTest(exponent=exponent, layout="texture_like"):
                    expected = small.pow(exponent)
                    actual = small.to("vulkan").pow(exponent).cpu()
                    self._assert_outputs_close(
                        expected,
                        actual,
                        atol=1e-4,
                        rtol=1e-4,
                    )

                    expected_inplace = small.clone()
                    expected_inplace.pow_(exponent)
                    actual_inplace = small.to("vulkan")
                    actual_inplace.pow_(exponent)
                    self._assert_outputs_close(
                        expected_inplace,
                        actual_inplace.cpu(),
                        atol=1e-4,
                        rtol=1e-4,
                    )

                with self.subTest(exponent=exponent, layout="buffer_like"):
                    expected = large.pow(exponent)
                    actual = large.to("vulkan").pow(exponent).cpu()
                    self._assert_outputs_close(
                        expected,
                        actual,
                        atol=1e-4,
                        rtol=1e-4,
                    )

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
            self._assert_outputs_close(
                expected_slice + expected_slice,
                (actual_slice + actual_slice).cpu(),
                atol=1e-4,
                rtol=1e-4)
            self._assert_outputs_close(
                expected_slice.sum(),
                actual_slice.sum().cpu(),
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
            self._assert_outputs_close(
                expected_as_strided * expected_as_strided,
                (actual_as_strided * actual_as_strided).cpu(),
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

    def test_fill_and_triangular_factories(self):
        with torch.inference_mode():
            expected_ones = torch.ones(6, 6)
            actual_ones = torch.ones(6, 6, device="vulkan")
            self._assert_outputs_close(expected_ones, actual_ones)

        mat = torch.randn(6, 6)
        self._assert_vulkan_matches_cpu(lambda t: torch.tril(t, diagonal=-1), mat)
        self._assert_vulkan_matches_cpu(lambda t: torch.triu(t, diagonal=1), mat)

        mask = torch.ones(6, 6, dtype=torch.bool)
        self._assert_vulkan_matches_cpu(lambda t: torch.triu(t, diagonal=0), mask)

    def test_eye_factory_and_out(self):
        with torch.inference_mode():
            expected_square = torch.eye(5, dtype=torch.float32)
            actual_square = torch.eye(5, dtype=torch.float32, device="vulkan")
            self._assert_outputs_close(expected_square, actual_square)

            expected_rect = torch.eye(4, 6, dtype=torch.int32)
            actual_rect = torch.eye(4, 6, dtype=torch.int32, device="vulkan")
            self._assert_outputs_close(expected_rect, actual_rect)

            out = torch.empty((1, 2), device="vulkan", dtype=torch.int32)
            actual_out = torch.eye(4, 6, out=out).cpu()
            self._assert_outputs_close(expected_rect, actual_out)
            self._assert_outputs_close(expected_rect, out.cpu())

    def test_5d_binary_ops_fallback_match_cpu(self):
        x = torch.randn(2, 3, 4, 5, 6)
        y = torch.randn(2, 3, 4, 5, 6)
        z = torch.randn(2, 3, 4, 5, 1)

        self._assert_vulkan_matches_cpu(lambda a, b: a - b, x, y)
        self._assert_vulkan_matches_cpu(lambda a, b: a + b, x, z)

    def test_arange_factories(self):
        with torch.inference_mode():
            expected_default = torch.arange(7)
            actual_default = torch.arange(7, device="vulkan").cpu()
            self._assert_outputs_close(expected_default, actual_default)

            expected_step = torch.arange(1, 9, 2, dtype=torch.float32)
            actual_step = torch.arange(
                1, 9, 2, dtype=torch.float32, device="vulkan").cpu()
            self._assert_outputs_close(expected_step, actual_step)

            expected_out = torch.arange(2, 11, 3, dtype=torch.long)
            out = torch.empty(1, device="vulkan", dtype=torch.long)
            actual_out = torch.arange(2, 11, 3, out=out).cpu()
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

            out = torch.empty(1, device="vulkan", dtype=torch.float32)
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

            out = torch.empty((1, 1), device="vulkan", dtype=torch.long)
            actual_out = torch.argmax(x.to("vulkan"), dim=-1, out=out).cpu()
            self._assert_outputs_close(expected, actual_out)
            self._assert_outputs_close(expected, out.cpu())

    def test_max_and_min_matches_cpu(self):
        with torch.inference_mode():
            x = torch.randn(2, 3, 5)
            self._assert_outputs_close(
                torch.max(x),
                torch.max(x.to("vulkan")).cpu(),
            )
            self._assert_outputs_close(
                torch.min(x),
                torch.min(x.to("vulkan")).cpu(),
            )

    def test_tan_and_atan_match_cpu(self):
        with torch.inference_mode():
            x = torch.randn(2, 3, 5) * 0.25
            self._assert_outputs_close(
                torch.tan(x),
                torch.tan(x.to("vulkan")).cpu(),
                atol=1e-4,
                rtol=1e-4,
            )
            self._assert_outputs_close(
                torch.atan(x),
                torch.atan(x.to("vulkan")).cpu(),
                atol=1e-4,
                rtol=1e-4,
            )

    def test_depth_anything_3_rope_matches_cpu_without_vulkan_branch(self):
        script = r"""
import os
import sys
import torch

workspace_root = os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(workspace_root, "temp", "Depth-Anything-3", "src"))

from depth_anything_3.model.dinov2.layers.rope import PositionGetter, RotaryPositionEmbedding2D

torch.manual_seed(0)
getter = PositionGetter()
rope = RotaryPositionEmbedding2D()
tokens_cpu = torch.randn(1, 4, 256, 64)
positions_cpu = getter(1, 16, 16, torch.device("cpu"))
positions_vulkan = getter(1, 16, 16, torch.device("vulkan"))
expected = rope(tokens_cpu, positions_cpu)
actual = rope(tokens_cpu.to("vulkan"), positions_vulkan).cpu()
torch.testing.assert_close(expected, actual, atol=1e-4, rtol=1e-4)
print("OK")
"""
        self._run_repo_python_subprocess(
            script,
            error_prefix="Depth Anything 3 RoPE Vulkan smoke failed.",
        )

    def test_depth_anything_3_pose_transform_backend_ops_match_cpu(self):
        script = r"""
import os
import sys
import torch

workspace_root = os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(workspace_root, "temp", "Depth-Anything-3", "src"))

from depth_anything_3.model.utils.transform import (
    extri_intri_to_pose_encoding,
    pose_encoding_to_extri_intri,
)

torch.manual_seed(0)
pose = torch.randn(2, 3, 9)
expected_extr, expected_intr = pose_encoding_to_extri_intri(pose, (518, 518))
actual_extr, actual_intr = pose_encoding_to_extri_intri(pose.to("vulkan"), (518, 518))
torch.testing.assert_close(expected_extr, actual_extr.cpu(), atol=1e-4, rtol=1e-4)
torch.testing.assert_close(expected_intr, actual_intr.cpu(), atol=1e-4, rtol=1e-4)

extr = torch.eye(4)[:3].reshape(1, 1, 3, 4).repeat(2, 3, 1, 1)
intr = torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 3, 1, 1)
expected_pose = extri_intri_to_pose_encoding(extr, intr, (518, 518))
actual_pose = extri_intri_to_pose_encoding(
    extr.to("vulkan"),
    intr.to("vulkan"),
    (518, 518),
)
torch.testing.assert_close(expected_pose, actual_pose.cpu(), atol=1e-4, rtol=1e-4)
print("OK")
"""
        self._run_repo_python_subprocess(
            script,
            error_prefix="Depth Anything 3 pose transform Vulkan smoke failed.",
        )

    def test_all_matches_cpu(self):
        with torch.inference_mode():
            x = torch.tensor([[True, True, True], [True, False, True]])
            expected = torch.all(x)
            actual = torch.all(x.to("vulkan")).cpu()
            self._assert_outputs_close(expected, actual)

            out = torch.empty((2,), device="vulkan", dtype=torch.bool)
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

    def test_index_select_dim0_with_vulkan_weight_and_vulkan_indices(self):
        torch.manual_seed(0)
        weight_cpu = torch.randn(32, 16)
        weight_vulkan = weight_cpu.to("vulkan")
        indices = torch.tensor([0, 7, 31, 4, 12], dtype=torch.long)

        with torch.inference_mode():
            expected = weight_cpu.index_select(0, indices)
            actual = weight_vulkan.index_select(0, indices.to("vulkan")).cpu()
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

    def test_vulkan_prepack_create_causal_attention_mask(self):
        prototype = torch.randn(2, 4, 8).to("vulkan")

        with torch.inference_mode():
            bool_mask = torch.ops.vulkan_prepack.create_causal_attention_mask(
                prototype,
                2,
                4,
                6,
                3,
                0,
                False,
            )
            self.assertTrue(bool_mask.is_vulkan)
            self.assertEqual(bool_mask.dtype, torch.bool)

            expected_keep = (
                (torch.arange(4) + 3).unsqueeze(1) >= torch.arange(6).unsqueeze(0)
            ).unsqueeze(0).unsqueeze(0).expand(2, 1, 4, 6)
            self.assertTrue(torch.equal(bool_mask.cpu(), expected_keep))

            float_mask = torch.ops.vulkan_prepack.create_causal_attention_mask(
                prototype,
                2,
                4,
                6,
                3,
                0,
                True,
            )
            self.assertTrue(float_mask.is_vulkan)
            self.assertEqual(float_mask.dtype, torch.float32)

            float_mask_cpu = float_mask.cpu()
            self.assertTrue(
                torch.equal(
                    torch.isneginf(float_mask_cpu),
                    expected_keep.logical_not(),
                )
            )
            self._assert_outputs_close(
                float_mask_cpu.masked_fill(expected_keep.logical_not(), 0.0),
                torch.zeros_like(float_mask_cpu),
                atol=1e-4,
                rtol=1e-4,
            )

    def test_vulkan_prepack_hidden_state_runtime_helpers(self):
        hidden_states = torch.randn(2, 6, 8).to("vulkan")
        positions = torch.tensor([1, 4], dtype=torch.long)

        with torch.inference_mode():
            sliced = torch.ops.vulkan_prepack.slice_hidden_states_for_logits(
                hidden_states,
                2,
            )
            self.assertTrue(sliced.is_vulkan)
            self._assert_outputs_close(
                hidden_states.cpu()[:, -2:, :],
                sliced.cpu(),
                atol=1e-4,
                rtol=1e-4,
            )

            selected = torch.ops.vulkan_prepack.index_select_hidden_states_for_logits(
                hidden_states,
                positions,
            )
            self.assertTrue(selected.is_vulkan)
            self._assert_outputs_close(
                hidden_states.cpu().index_select(1, positions),
                selected.cpu(),
                atol=1e-4,
                rtol=1e-4,
            )

            gathered = torch.ops.vulkan_prepack.gather_hidden_states_by_batch_positions(
                hidden_states,
                positions.to("vulkan"),
            )
            self.assertTrue(gathered.is_vulkan)
            expected_gathered = torch.stack(
                [
                    hidden_states.cpu()[0, positions[0]],
                    hidden_states.cpu()[1, positions[1]],
                ],
                dim=0,
            )
            self._assert_outputs_close(
                expected_gathered,
                gathered.cpu(),
                atol=1e-4,
                rtol=1e-4,
            )

    def test_vulkan_prepack_find_timestep_index(self):
        schedule = torch.tensor([9.0, 7.0, 7.0, 5.0], dtype=torch.float32).to("vulkan")
        timestep = torch.tensor(7.0, dtype=torch.float32).to("vulkan")

        with torch.inference_mode():
            index = torch.ops.vulkan_prepack.find_timestep_index(schedule, timestep)

        self.assertEqual(index, 2)

    def test_vulkan_prepack_compute_rotary_cos_sin(self):
        prototype = torch.randn(1, 2, 4, 8, dtype=torch.float16).to("vulkan")
        inv_freq = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)
        position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64).to("vulkan")
        scaling = 1.25

        with torch.inference_mode():
            cos, sin = torch.ops.vulkan_prepack.compute_rotary_cos_sin(
                prototype,
                inv_freq,
                position_ids,
                scaling,
            )

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.size(0), -1, 1)
        position_ids_expanded = position_ids.cpu()[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        expected_cos = (emb.cos() * scaling).to(dtype=prototype.dtype)
        expected_sin = (emb.sin() * scaling).to(dtype=prototype.dtype)

        self.assertTrue(cos.is_vulkan)
        self.assertTrue(sin.is_vulkan)
        self.assertEqual(cos.dtype, prototype.dtype)
        self.assertEqual(sin.dtype, prototype.dtype)
        self._assert_outputs_close(expected_cos, cos.cpu(), atol=2e-3, rtol=2e-3)
        self._assert_outputs_close(expected_sin, sin.cpu(), atol=2e-3, rtol=2e-3)

    def test_vulkan_prepack_moe_router_helpers(self):
        logits_cpu = torch.tensor(
            [
                [1.0, 4.0, 3.0],
                [2.0, 0.5, 1.5],
                [0.1, 0.9, 0.2],
            ],
            dtype=torch.float32,
        )
        logits = logits_cpu.to("vulkan")
        top_k = 2

        with torch.inference_mode():
            batch_index_cpu, batch_gates, expert_size = torch.ops.vulkan_prepack.compute_moe_router(
                logits,
                top_k,
                logits_cpu.size(1),
            )

        top_k_logits_cpu, top_k_indices_cpu = logits_cpu.topk(top_k, dim=1)
        top_k_gates_cpu = torch.softmax(top_k_logits_cpu, dim=1)
        gates_cpu = torch.zeros(
            (top_k_indices_cpu.size(0), logits_cpu.size(1)),
            dtype=top_k_gates_cpu.dtype,
        ).scatter(1, top_k_indices_cpu, 1)
        expected_expert_size = gates_cpu.long().sum(0)
        top_k_experts_cpu = top_k_indices_cpu.flatten()
        _, index_sorted_experts_cpu = top_k_experts_cpu.sort(0)
        expected_batch_index = index_sorted_experts_cpu.div(top_k, rounding_mode="trunc")
        expected_batch_gates = top_k_gates_cpu.flatten().index_select(0, index_sorted_experts_cpu)

        self.assertFalse(batch_index_cpu.is_vulkan)
        self.assertTrue(batch_gates.is_vulkan)
        self.assertFalse(expert_size.is_vulkan)
        self.assertEqual(batch_index_cpu.dtype, torch.int64)
        self.assertEqual(expert_size.dtype, torch.int64)
        self.assertTrue(torch.equal(batch_index_cpu.cpu(), expected_batch_index))
        self.assertTrue(torch.equal(expert_size.cpu(), expected_expert_size))
        self._assert_outputs_close(
            expected_batch_gates,
            batch_gates.cpu(),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_vulkan_prepack_accumulate_expert_outputs(self):
        expert_outputs_cpu = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=torch.float32,
        )
        batch_index_cpu = torch.tensor([0, 2, 0], dtype=torch.int64)
        expert_outputs = expert_outputs_cpu.to("vulkan")

        with torch.inference_mode():
            accumulated = torch.ops.vulkan_prepack.accumulate_expert_outputs(
                expert_outputs,
                batch_index_cpu,
                4,
            )

        expected = torch.zeros((4, 2), dtype=torch.float32).index_add(
            0,
            batch_index_cpu,
            expert_outputs_cpu,
        )
        self.assertTrue(accumulated.is_vulkan)
        self._assert_outputs_close(
            expected,
            accumulated.cpu(),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_transformers_legacy_causal_attention_mask_converter_on_vulkan(self):
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
            from transformers.modeling_attn_mask_utils import AttentionMaskConverter

            converter = AttentionMaskConverter(True, sliding_window=2047)
            mask = converter.to_causal_4d(
                batch_size=1,
                query_length=8,
                key_value_length=8,
                dtype=torch.float32,
                device=torch.device("vulkan"),
            )
            expected_keep = (
                torch.arange(8).unsqueeze(1) >= torch.arange(8).unsqueeze(0)
            ).unsqueeze(0).unsqueeze(0)
            mask_cpu = mask.cpu()
            assert mask_cpu.shape == (1, 1, 8, 8)
            assert torch.equal(torch.isneginf(mask_cpu), expected_keep.logical_not())
            print(float(mask_cpu[:, :, -1, -1].item()))
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
                "Transformers legacy causal-mask converter crashed on Vulkan.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            ),
        )

    def test_transformers_mistral_logits_to_keep_on_vulkan(self):
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
            import sys
            from pathlib import Path
            sys.path.insert(0, str((Path.cwd().parent / "scripts" / "benchmarks").resolve()))
            from transformers_runtime_compat import ensure_transformers_runtime_compat
            ensure_transformers_runtime_compat(torch)
            from transformers.models.mistral.configuration_mistral import MistralConfig
            from transformers.models.mistral.modeling_mistral import MistralForCausalLM

            config = MistralConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                max_position_embeddings=32,
            )
            model = MistralForCausalLM(config).eval().to("vulkan")
            input_ids = torch.randint(0, 128, (1, 8), dtype=torch.long)
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
            logits = outputs.logits
            assert logits.is_vulkan
            assert logits.shape == (1, 1, 128)
            print(float(logits.cpu().sum().item()))
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
                "Transformers Mistral logits_to_keep path crashed on Vulkan.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            ),
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

    def test_embedding_with_vulkan_weight_and_vulkan_indices(self):
        torch.manual_seed(0)
        module_cpu = torch.nn.Embedding(64, 24).eval()
        module_vulkan = torch.nn.Embedding(64, 24).eval()
        module_vulkan.load_state_dict(module_cpu.state_dict())
        module_vulkan = module_vulkan.to("vulkan")
        indices = torch.tensor([[1, 5, 7, 2, 9, 4]], dtype=torch.long)
        indices_vulkan = indices.to("vulkan")

        with torch.inference_mode():
            expected = module_cpu(indices)
            actual = module_vulkan(indices_vulkan).cpu()
            self._assert_outputs_close(
                expected,
                actual,
                atol=1e-4,
                rtol=1e-4)

            expected_functional = F.embedding(indices, module_cpu.weight)
            actual_functional = F.embedding(indices_vulkan, module_vulkan.weight).cpu()
            self._assert_outputs_close(
                expected_functional,
                actual_functional,
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
        script = textwrap.dedent(
            """
            import torch
            import torch.nn.functional as F

            indices = torch.tensor([[1, 5, 7, 2, 9, 4, 1024, 16000]], dtype=torch.long)

            def run_once():
                torch.manual_seed(0)
                module_cpu = torch.nn.Embedding(17000, 256).eval()
                module_vulkan = torch.nn.Embedding(17000, 256).eval()
                module_vulkan.load_state_dict(module_cpu.state_dict())
                module_vulkan = module_vulkan.to("vulkan")

                with torch.inference_mode():
                    expected = module_cpu(indices)
                    actual = module_vulkan(indices).cpu()
                    module_diff = (expected - actual).abs().max().item()

                    expected_functional = F.embedding(indices, module_cpu.weight)
                    actual_functional = F.embedding(indices, module_vulkan.weight).cpu()
                    functional_diff = (
                        expected_functional - actual_functional
                    ).abs().max().item()
                return module_diff, functional_diff

            module_diff, functional_diff = run_once()
            if module_diff > 1e-4 or functional_diff > 1e-4:
                module_diff, functional_diff = run_once()

            assert module_diff <= 1e-4, module_diff
            assert functional_diff <= 1e-4, functional_diff

            print("ok")
            """
        )

        _, result = self._run_repo_python_subprocess(
            script,
            error_prefix="Large buffer-backed embedding subprocess failed.",
        )
        self.assertIn("ok", result.stdout)

    def test_embedding_with_large_buffer_backed_half_vulkan_weight_and_cpu_indices(self):
        torch.manual_seed(0)
        weight_cpu = torch.randn(17000, 256, dtype=torch.float16)
        weight_vulkan = weight_cpu.to("vulkan")
        indices = torch.tensor([[1, 5, 7, 2, 9, 4, 1024, 16000]], dtype=torch.long)

        with torch.inference_mode():
            expected = F.embedding(indices, weight_cpu)
            actual = F.embedding(indices, weight_vulkan).cpu()
            self._assert_outputs_close(
                expected,
                actual,
                atol=2e-2,
                rtol=2e-2)

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

    def test_float_select_and_unbind_with_vulkan_input(self):
        src = torch.linspace(-1.0, 1.0, steps=8, dtype=torch.float32)
        vulkan = src.to("vulkan")

        self.assertEqual(vulkan.select(0, 3).cpu(), src.select(0, 3))

        actual_unbind = [item.cpu() for item in vulkan.unbind(0)]
        expected_unbind = [item for item in src.unbind(0)]
        self.assertEqual(actual_unbind, expected_unbind)

    def test_long_expand_position_ids_style_with_vulkan_input(self):
        with torch.inference_mode():
            src = torch.arange(8, dtype=torch.long)
            vulkan = src.to("vulkan")

            expected = src.view(1, 1, -1).expand(4, 1, -1)[1:]
            actual = vulkan.view(1, 1, -1).expand(4, 1, -1)[1:].cpu()

            self.assertEqual(actual, expected)

    def test_long_cat_and_stack_with_vulkan_input(self):
        with torch.inference_mode():
            a = torch.arange(4, dtype=torch.long)
            b = torch.arange(4, dtype=torch.long) + 10
            a_vulkan = a.to("vulkan")
            b_vulkan = b.to("vulkan")

            expected_cat = torch.cat((a.unsqueeze(0), b.unsqueeze(0)), dim=0)
            actual_cat = torch.cat((a_vulkan.unsqueeze(0), b_vulkan.unsqueeze(0)), dim=0).cpu()
            self.assertEqual(actual_cat, expected_cat)

            expected_stack = torch.stack((a, b), dim=0)
            actual_stack = torch.stack((a_vulkan, b_vulkan), dim=0).cpu()
            self.assertEqual(actual_stack, expected_stack)

    def test_5d_expand_fallback_match_cpu(self):
        src = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
        vulkan = src.to("vulkan")

        expected = src.unsqueeze(2).expand(2, 3, 7, 4, 5)
        actual = vulkan.unsqueeze(2).expand(2, 3, 7, 4, 5).cpu()

        self.assertEqual(actual, expected)

    def test_5d_transpose_and_permute_fallback_match_cpu(self):
        src = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(2, 3, 4, 5, 6)
        vulkan = src.to("vulkan")

        expected_transpose = src.transpose(-1, -2)
        actual_transpose = vulkan.transpose(-1, -2).cpu()
        self.assertEqual(actual_transpose, expected_transpose)

        expected_permute = src.permute(0, 2, 1, 4, 3)
        actual_permute = vulkan.permute(0, 2, 1, 4, 3).cpu()
        self.assertEqual(actual_permute, expected_permute)

    def test_5d_masked_fill_fallback_match_cpu(self):
        src = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(2, 3, 4, 5, 6)
        mask = (src.remainder(3) == 0)
        vulkan = src.to("vulkan")
        mask_vulkan = mask.to("vulkan")

        expected = src.masked_fill(mask, -1.25)
        actual = vulkan.masked_fill(mask_vulkan, -1.25).cpu()

        self.assertEqual(actual, expected)

        expected_inplace = src.clone()
        expected_inplace.masked_fill_(mask, -0.5)
        actual_inplace = src.to("vulkan")
        actual_inplace.masked_fill_(mask_vulkan, -0.5)
        self.assertEqual(actual_inplace.cpu(), expected_inplace)

    def test_5d_select_and_slice_fallback_match_cpu(self):
        src = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(2, 3, 4, 5, 6)
        vulkan = src.to("vulkan")

        expected_select = src.select(2, 1)
        actual_select = vulkan.select(2, 1).cpu()
        self.assertEqual(actual_select, expected_select)

        expected_slice = src[:, :, 1:3, :, :]
        actual_slice = vulkan[:, :, 1:3, :, :].cpu()
        self.assertEqual(actual_slice, expected_slice)

    def test_5d_sum_and_mean_dim_fallback_match_cpu(self):
        src = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(2, 3, 4, 5, 6)
        vulkan = src.to("vulkan")

        expected_sum = src.sum(dim=-2)
        actual_sum = vulkan.sum(dim=-2).cpu()
        self.assertEqual(actual_sum, expected_sum)

        expected_mean = src.mean(dim=-2)
        actual_mean = vulkan.mean(dim=-2).cpu()
        self.assertEqual(actual_mean, expected_mean)

    def test_5d_zeros_and_zero_fallback_match_cpu(self):
        with torch.inference_mode():
            expected = torch.zeros((2, 3, 4, 5, 6), dtype=torch.float32)
            actual = torch.zeros((2, 3, 4, 5, 6), dtype=torch.float32, device="vulkan")
            self._assert_outputs_close(expected, actual)

            src = torch.randn(2, 3, 4, 5, 6)
            vulkan = src.to("vulkan")
            vulkan.zero_()
            self._assert_outputs_close(torch.zeros_like(src), vulkan)

    def test_bfloat16_tensor_roundtrip_and_zeros(self):
        src = torch.tensor([[1.0, -0.5, 3.25], [4.0, 5.5, -6.0]], dtype=torch.bfloat16)
        vulkan = src.to("vulkan")

        self.assertEqual(vulkan.device.type, "vulkan")
        self.assertEqual(vulkan.cpu(), src)

        zeros = torch.zeros((2, 3), dtype=torch.bfloat16, device="vulkan")
        self.assertEqual(zeros.cpu(), torch.zeros((2, 3), dtype=torch.bfloat16))

    def test_half_tensor_roundtrip_and_labeled_roundtrip(self):
        src = torch.tensor([[1.0, -0.5, 3.25], [4.0, 5.5, -6.0]], dtype=torch.float16)
        vulkan = src.to("vulkan")

        self.assertEqual(vulkan.device.type, "vulkan")
        self.assertEqual(vulkan.cpu(), src)

        labeled = torch.ops.vulkan_prepack.to_vulkan_labeled(
            src,
            "test.half_weight",
        )
        self.assertTrue(labeled.is_vulkan)
        self.assertEqual(labeled.cpu(), src)

    def test_large_half_matrix_roundtrip(self):
        torch.manual_seed(0)
        src = torch.randn(2048, 1024, dtype=torch.float16)

        with torch.inference_mode():
            vulkan = src.to("vulkan")
            self.assertTrue(vulkan.is_vulkan)
            self.assertEqual(vulkan.cpu(), src)

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

    def test_module_to_vulkan_preserves_shared_parameters_and_buffers(self):
        class SharedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(12, 5)
                self.proj = nn.Linear(5, 12, bias=False)
                self.proj.weight = self.embed.weight
                shared_ids = torch.tensor([3, 1, 4, 1], dtype=torch.long)
                self.register_buffer("token_ids_a", shared_ids)
                self.register_buffer("token_ids_b", shared_ids)

            def forward(self, indices):
                embedded = self.embed(indices)
                pooled = embedded.sum(dim=1)
                return self.proj(pooled)

        torch.manual_seed(0)
        module_cpu = SharedModule().eval()
        module_vulkan = SharedModule().eval()
        module_vulkan.load_state_dict(module_cpu.state_dict())
        module_vulkan.proj.weight = module_vulkan.embed.weight

        module_vulkan = module_vulkan.to("vulkan")

        self.assertIs(module_vulkan.proj.weight, module_vulkan.embed.weight)
        self.assertIs(module_vulkan.token_ids_a, module_vulkan.token_ids_b)
        self.assertEqual(module_vulkan.embed.weight.device.type, "vulkan")
        self.assertEqual(module_vulkan.token_ids_a.device.type, "vulkan")

        indices = torch.tensor([[1, 5, 7], [2, 9, 4]], dtype=torch.long)
        with torch.inference_mode():
            expected = module_cpu(indices)
            actual = module_vulkan(indices).cpu()
            self._assert_outputs_close(
                expected,
                actual,
                atol=1e-4,
                rtol=1e-4,
            )

    def test_module_to_vulkan_can_keep_marked_submodule_on_cpu(self):
        class KeepCpuSubmodule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 3, bias=False)
                self.buffer_holder = nn.Module()
                self.buffer_holder.register_buffer("ids", torch.tensor([1, 2, 3], dtype=torch.long))
                self.buffer_holder._vulkan_keep_cpu = True

        module = KeepCpuSubmodule().eval().to("vulkan")

        self.assertEqual(module.linear.weight.device.type, "vulkan")
        self.assertEqual(module.buffer_holder.ids.device.type, "cpu")

    def test_vulkan_autocast_context_is_available(self):
        self.assertTrue(torch.amp.is_autocast_available("vulkan"))
        self.assertFalse(torch.is_autocast_enabled("vulkan"))
        self.assertEqual(torch.get_autocast_dtype("vulkan"), torch.float16)
        torch.set_autocast_enabled("vulkan", True)
        self.assertFalse(torch.is_autocast_enabled("vulkan"))
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

    def test_half_linear_3d_runs_on_vulkan(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 4, dtype=torch.float16)
        weight = torch.randn(5, 4, dtype=torch.float16)
        bias = torch.randn(5, dtype=torch.float16)
        x_vulkan = x.to("vulkan")
        weight_vulkan = weight.to("vulkan")
        bias_vulkan = bias.to("vulkan")

        with torch.inference_mode():
            expected = F.linear(x, weight, bias)
            actual = F.linear(
                x_vulkan,
                weight_vulkan,
                bias_vulkan,
            ).cpu()

        self.assertEqual(actual.dtype, torch.float16)
        self._assert_outputs_close(expected, actual, atol=2e-2, rtol=2e-2)

    def test_half_bmm_runs_on_vulkan(self):
        torch.manual_seed(0)
        lhs = torch.randn(2, 3, 4, dtype=torch.float16)
        rhs = torch.randn(2, 4, 5, dtype=torch.float16)
        lhs_vulkan = lhs.to("vulkan")
        rhs_vulkan = rhs.to("vulkan")

        with torch.inference_mode():
            expected = torch.bmm(lhs.float(), rhs.float())
            actual = torch.bmm(lhs_vulkan, rhs_vulkan).cpu()

        self.assertEqual(actual.dtype, torch.float32)
        self._assert_outputs_close(expected, actual, atol=2e-2, rtol=2e-2)

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

    def test_float16_conv2d_widens_to_float_for_compute(self):
        torch.manual_seed(0)
        x = torch.randn(1, 3, 8, 8, dtype=torch.float16)
        weight = torch.randn(4, 3, 3, 3, dtype=torch.float16)
        bias = torch.randn(4, dtype=torch.float16)
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
        self._assert_outputs_close(expected, actual, atol=2e-2, rtol=2e-2)

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
            (
                "rms_norm",
                lambda t: F.rms_norm(t, (8,), norm_weight, 1e-5),
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
            out_vulkan = torch.empty((1, 3, 5, 5), device="vulkan")
            actual = torch.ops.aten.upsample_bicubic2d.out(
                x_vulkan,
                [11, 13],
                False,
                None,
                None,
                out=out_vulkan).cpu()

        self._assert_outputs_close(expected, actual, atol=5e-3, rtol=5e-3)
        self._assert_outputs_close(expected, out_vulkan.cpu(), atol=1e-3, rtol=1e-3)

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

    def test_scaled_dot_product_attention_masks_and_causal(self):
        torch.manual_seed(0)
        q4 = torch.randn(1, 3, 9, 8)
        k4 = torch.randn(1, 3, 7, 8)
        v4 = torch.randn(1, 3, 7, 8)
        q3 = torch.randn(3, 9, 8)
        k3 = torch.randn(3, 7, 8)
        v3 = torch.randn(3, 7, 8)
        q_gqa = torch.randn(1, 6, 9, 8)
        k_gqa = torch.randn(1, 2, 7, 8)
        v_gqa = torch.randn(1, 2, 7, 8)

        float_mask = torch.zeros(1, 1, 9, 7)
        float_mask[..., :, -2:] = -1000.0
        cases = [
            ("sdpa_4d_causal", q4, k4, v4, None, True, False, True),
            ("sdpa_4d_float_mask", q4, k4, v4, float_mask, False, False, True),
            ("sdpa_3d_causal", q3, k3, v3, None, True, False, True),
            ("sdpa_4d_gqa", q_gqa, k_gqa, v_gqa, None, False, True, True),
        ]

        with torch.inference_mode():
            for name, query, key, value, attn_mask, is_causal, enable_gqa, check_functional in cases:
                with self.subTest(case=name):
                    if check_functional:
                        expected = F.scaled_dot_product_attention(
                            query,
                            key,
                            value,
                            attn_mask=attn_mask,
                            dropout_p=0.0,
                            is_causal=is_causal,
                            scale=0.125,
                            enable_gqa=enable_gqa,
                        )
                        actual = F.scaled_dot_product_attention(
                            query.to("vulkan"),
                            key.to("vulkan"),
                            value.to("vulkan"),
                            attn_mask=None if attn_mask is None else attn_mask.to("vulkan"),
                            dropout_p=0.0,
                            is_causal=is_causal,
                            scale=0.125,
                            enable_gqa=enable_gqa,
                        ).cpu()
                        self._assert_outputs_close(
                            expected,
                            actual,
                            atol=1e-4,
                            rtol=1e-4,
                        )

                    expected_math = torch.ops.aten._scaled_dot_product_attention_math(
                        query,
                        key,
                        value,
                        attn_mask,
                        0.0,
                        is_causal,
                        None,
                        scale=0.125,
                        enable_gqa=enable_gqa,
                    )[0]
                    actual_math = torch.ops.aten._scaled_dot_product_attention_math(
                        query.to("vulkan"),
                        key.to("vulkan"),
                        value.to("vulkan"),
                        None if attn_mask is None else attn_mask.to("vulkan"),
                        0.0,
                        is_causal,
                        None,
                        scale=0.125,
                        enable_gqa=enable_gqa,
                    )[0].cpu()
                    self._assert_outputs_close(
                        expected_math,
                        actual_math,
                        atol=1e-4,
                        rtol=1e-4,
                    )

    def test_vulkan_dispatch_tables_expose_backend_kernels(self):
        dispatch_expectations = {
            "aten::linear": ("Vulkan: registered at", "Mm.cpp"),
            "aten::mm": ("Vulkan: registered at", "Mm.cpp"),
            "aten::bmm": ("Vulkan: registered at", "Mm.cpp"),
            "aten::_softmax": ("Vulkan: registered at", "Softmax.cpp"),
            "aten::group_norm": ("Vulkan: registered at", "Mean.cpp"),
            "aten::_scaled_dot_product_attention_math": (
                "Vulkan: registered at",
                "Softmax.cpp",
            ),
            "vulkan_prepack::run_linear_context": (
                "Vulkan: registered at",
                "Register.cpp",
            ),
        }

        for opname, expected_substrings in dispatch_expectations.items():
            with self.subTest(opname=opname):
                table = torch._C._dispatch_dump_table(opname)
                for expected_substring in expected_substrings:
                    self.assertIn(expected_substring, table)

        public_sdpa_table = torch._C._dispatch_dump_table(
            "aten::scaled_dot_product_attention"
        )
        self.assertIn("CompositeImplicitAutograd", public_sdpa_table)
        self.assertFalse(
            hasattr(torch.ops.vulkan_prepack, "scaled_dot_product_attention")
        )

        layer_norm_table = torch._C._dispatch_dump_table("aten::layer_norm")
        self.assertIn("CompositeImplicitAutograd", layer_norm_table)
        self.assertNotIn("Layernorm.cpp", layer_norm_table)

        native_layer_norm_table = torch._C._dispatch_dump_table(
            "aten::native_layer_norm"
        )
        self.assertIn("CompositeExplicitAutograd", native_layer_norm_table)
        self.assertNotIn("NativeLayerNorm.cpp", native_layer_norm_table)

        rms_norm_table = torch._C._dispatch_dump_table("aten::rms_norm")
        self.assertIn("CompositeImplicitAutograd", rms_norm_table)
        self.assertNotIn("RMSNorm.cpp", rms_norm_table)

    def test_vulkan_runtime_op_hit_logging(self):
        log_name = "vulkan_op_hit_logging_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(repo_root, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)

        try:
            script = """
                import torch
                import torch.nn.functional as F

                torch.manual_seed(0)
                with torch.inference_mode():
                    x = torch.randn(1, 16, 32, dtype=torch.float32).to("vulkan")
                    w = torch.randn(64, 32, dtype=torch.float32).to("vulkan")
                    b = torch.randn(64, dtype=torch.float32).to("vulkan")
                    F.linear(x, w, b)

                    ctx = torch.ops.vulkan_prepack.create_linear_context(
                        w.clone().t(),
                        b,
                    )
                    torch.ops.vulkan_prepack.run_linear_context(x, ctx)

                    mm_a = torch.randn(8, 8, dtype=torch.float32).to("vulkan")
                    mm_b = torch.randn(8, 8, dtype=torch.float32).to("vulkan")
                    torch.mm(mm_a, mm_b)

                    bmm_a = torch.randn(2, 8, 8, dtype=torch.float32).to("vulkan")
                    bmm_b = torch.randn(2, 8, 8, dtype=torch.float32).to("vulkan")
                    torch.bmm(bmm_a, bmm_b)

                    s = torch.randn(2, 3, 4, dtype=torch.float32).to("vulkan")
                    F.softmax(s, dim=-1)

                    ln_x = torch.randn(1, 17, 32, dtype=torch.float32).to("vulkan")
                    ln_weight = torch.randn(32, dtype=torch.float32)
                    ln_bias = torch.randn(32, dtype=torch.float32)
                    F.layer_norm(
                        ln_x,
                        (32,),
                        ln_weight,
                        ln_bias,
                        1e-5,
                    )
                    F.rms_norm(
                        ln_x,
                        (32,),
                        ln_weight,
                        1e-5,
                    )
                    gn_x = torch.randn(2, 8, 7, 7, dtype=torch.float32).to("vulkan")
                    gn_weight = torch.randn(8, dtype=torch.float32)
                    gn_bias = torch.randn(8, dtype=torch.float32)
                    torch.nn.functional.group_norm(
                        gn_x,
                        4,
                        gn_weight,
                        gn_bias,
                        1e-5,
                    )

                    q = torch.randn(2, 9, 8, dtype=torch.float32).to("vulkan")
                    k = torch.randn(2, 7, 8, dtype=torch.float32).to("vulkan")
                    v = torch.randn(2, 7, 8, dtype=torch.float32).to("vulkan")
                    torch.ops.aten._scaled_dot_product_attention_math(
                        q,
                        k,
                        v,
                        None,
                        0.0,
                        False,
                        None,
                        scale=0.125,
                        enable_gqa=False,
                    )[0]
                print("ok")
            """

            _, result = self._run_repo_python_subprocess(
                script,
                extra_env={"PYTORCH_VULKAN_OP_HIT_LOG": log_name},
                error_prefix="Vulkan op-hit logging subprocess failed.",
            )
            self.assertIn("ok", result.stdout)

            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as log_file:
                log_text = log_file.read()

            for op_name in (
                "aten::linear",
                "vulkan_prepack::run_linear_context",
                "aten::mm",
                "aten::bmm",
                "aten::_softmax",
                "aten::layer_norm",
                "aten::layer_norm.fused_width",
                "aten::rms_norm",
                "aten::rms_norm.fused_width",
                "aten::_scaled_dot_product_attention_math",
            ):
                with self.subTest(op_name=op_name):
                    self.assertIn(f"op={op_name}", log_text)
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_vulkan_execution_object_logging_for_decode_like_linear_norm(self):
        log_name = "vulkan_execution_object_decode_like_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(repo_root, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)

        try:
            script = """
                import torch
                import torch.nn.functional as F

                torch.manual_seed(0)
                x = torch.randn(1, 1, 384, dtype=torch.float32).to("vulkan")
                ln_weight = torch.randn(384, dtype=torch.float32)
                ln_bias = torch.randn(384, dtype=torch.float32)
                linear_weight = torch.randn(384, 384, dtype=torch.float32).to("vulkan")
                linear_bias = torch.randn(384, dtype=torch.float32).to("vulkan")

                with torch.inference_mode():
                    for _ in range(2):
                        x = F.layer_norm(x, (384,), ln_weight, ln_bias, 1e-5)
                        x = F.linear(x, linear_weight, linear_bias)

                print(float(x.cpu().sum()))
            """

            self._run_repo_python_subprocess(
                script,
                extra_env={
                    "PYTORCH_VULKAN_EXECUTION_OBJECT_LOG": log_name,
                },
                error_prefix="Execution-object logging subprocess failed.",
            )

            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as log_file:
                log_text = log_file.read()

            self.assertIn("execution_object_event kind=ScratchArena event=store", log_text)
            self.assertIn("execution_object_event kind=ScratchArena event=hit", log_text)
            self.assertIn("execution_object_event kind=ScratchArena event=reserve", log_text)
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_vulkan_runtime_policy_prefill_like_linear_bmm_enables_scratch(self):
        policy_log_name = "vulkan_runtime_policy_prefill_like_test.log"
        object_log_name = "vulkan_execution_object_prefill_like_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        policy_log_path = os.path.join(repo_root, policy_log_name)
        object_log_path = os.path.join(repo_root, object_log_name)
        for path in (policy_log_path, object_log_path):
            if os.path.exists(path):
                os.remove(path)

        try:
            script = """
                import torch
                import torch.nn.functional as F

                torch.manual_seed(0)
                x = torch.randn(1, 8, 1024, dtype=torch.float32).to("vulkan")
                linear_weight = torch.randn(1024, 1024, dtype=torch.float32).to("vulkan")
                linear_bias = torch.randn(1024, dtype=torch.float32).to("vulkan")
                bmm_lhs = torch.randn(16, 8, 128, dtype=torch.float32).to("vulkan")
                bmm_rhs = torch.randn(16, 128, 128, dtype=torch.float32).to("vulkan")

                with torch.inference_mode():
                    for _ in range(2):
                        y = F.linear(x, linear_weight, linear_bias)
                        z = torch.bmm(bmm_lhs, bmm_rhs)

                print(float(y.cpu().sum() + z.cpu().sum()))
            """

            self._run_repo_python_subprocess(
                script,
                extra_env={
                    "PYTORCH_VULKAN_RUNTIME_POLICY_LOG": policy_log_name,
                    "PYTORCH_VULKAN_EXECUTION_OBJECT_LOG": object_log_name,
                },
                error_prefix="Prefill-like runtime-policy subprocess failed.",
            )

            self.assertTrue(os.path.exists(policy_log_path))
            with open(policy_log_path, "r", encoding="utf-8") as log_file:
                policy_log_text = log_file.read()

            self.assertRegex(
                policy_log_text,
                r"runtime_policy workload=LinearMatmul model_domain=LLM execution_phase=Prefill .* has_scratch_arena_plan=1 inferred_from_label=0",
            )

            self.assertTrue(os.path.exists(object_log_path))
            with open(object_log_path, "r", encoding="utf-8") as log_file:
                object_log_text = log_file.read()

            self.assertIn("execution_object_event kind=ScratchArena event=store", object_log_text)
            self.assertIn("execution_object_event kind=ScratchArena event=hit", object_log_text)
        finally:
            for path in (policy_log_path, object_log_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_vulkan_runtime_label_infers_vision_backbone_policy(self):
        policy_log_name = "vulkan_runtime_label_vision_policy_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        policy_log_path = os.path.join(repo_root, policy_log_name)
        if os.path.exists(policy_log_path):
            os.remove(policy_log_path)

        try:
            script = """
                import torch
                import torch.nn.functional as F

                assert hasattr(torch.ops.vulkan_prepack, "swap_runtime_label")

                torch.manual_seed(0)
                x = torch.randn(1, 8, 384, dtype=torch.float32).to("vulkan")
                weight = torch.randn(384, 384, dtype=torch.float32).to("vulkan")
                bias = torch.randn(384, dtype=torch.float32).to("vulkan")

                previous = torch.ops.vulkan_prepack.swap_runtime_label(
                    "depth.dino.backbone.block"
                )
                try:
                    with torch.inference_mode():
                        y = F.linear(x, weight, bias)
                finally:
                    torch.ops.vulkan_prepack.swap_runtime_label(previous)

                print(float(y.cpu().sum()))
            """

            self._run_repo_python_subprocess(
                script,
                extra_env={
                    "PYTORCH_VULKAN_RUNTIME_POLICY_LOG": policy_log_name,
                },
                error_prefix="Vision runtime-label subprocess failed.",
            )

            self.assertTrue(os.path.exists(policy_log_path))
            with open(policy_log_path, "r", encoding="utf-8") as log_file:
                policy_log_text = log_file.read()

            self.assertRegex(
                policy_log_text,
                r"runtime_policy workload=VisionBackbone model_domain=Vision execution_phase=Backbone .* inferred_from_label=1",
            )
        finally:
            if os.path.exists(policy_log_path):
                os.remove(policy_log_path)

    def test_vulkan_vision_backbone_block_context_matches_reference(self):
        torch.manual_seed(0)
        embed_dim = 32
        num_heads = 4
        hidden_dim = 64
        token_count = 17

        x = torch.randn(1, token_count, embed_dim, dtype=torch.float32)
        norm1_weight = torch.randn(embed_dim, dtype=torch.float32)
        norm1_bias = torch.randn(embed_dim, dtype=torch.float32)
        qkv_weight = torch.randn(embed_dim * 3, embed_dim, dtype=torch.float32)
        qkv_bias = torch.randn(embed_dim * 3, dtype=torch.float32)
        proj_weight = torch.randn(embed_dim, embed_dim, dtype=torch.float32)
        proj_bias = torch.randn(embed_dim, dtype=torch.float32)
        ls1_gamma = torch.randn(embed_dim, dtype=torch.float32)
        norm2_weight = torch.randn(embed_dim, dtype=torch.float32)
        norm2_bias = torch.randn(embed_dim, dtype=torch.float32)
        fc1_weight = torch.randn(hidden_dim, embed_dim, dtype=torch.float32)
        fc1_bias = torch.randn(hidden_dim, dtype=torch.float32)
        fc2_weight = torch.randn(embed_dim, hidden_dim, dtype=torch.float32)
        fc2_bias = torch.randn(embed_dim, dtype=torch.float32)
        ls2_gamma = torch.randn(embed_dim, dtype=torch.float32)

        norm_eps = 1.0e-6
        head_dim = embed_dim // num_heads

        def reference(inp):
            norm1 = F.layer_norm(inp, (embed_dim,), norm1_weight, norm1_bias, norm_eps)
            qkv = F.linear(norm1.reshape(token_count, embed_dim), qkv_weight, None)
            q, k, v = qkv.chunk(3, dim=1)
            q = (
                q + qkv_bias[:embed_dim]
            ).reshape(token_count, num_heads, head_dim).permute(1, 0, 2)
            k = (
                k + qkv_bias[embed_dim : 2 * embed_dim]
            ).reshape(token_count, num_heads, head_dim).permute(1, 0, 2)
            v = (
                v + qkv_bias[2 * embed_dim :]
            ).reshape(token_count, num_heads, head_dim).permute(1, 0, 2)
            q = q * (head_dim ** -0.5)
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0,
            )
            attn = attn.permute(1, 0, 2).reshape(token_count, embed_dim)
            attn = F.linear(attn, proj_weight, proj_bias).reshape(1, token_count, embed_dim)
            hidden = inp + attn * ls1_gamma
            norm2 = F.layer_norm(hidden, (embed_dim,), norm2_weight, norm2_bias, norm_eps)
            mlp = F.linear(norm2, fc1_weight, fc1_bias)
            mlp = F.gelu(mlp)
            mlp = F.linear(mlp, fc2_weight, fc2_bias)
            return hidden + mlp * ls2_gamma

        expected = reference(x)
        with torch.inference_mode():
            context = torch.ops.vulkan_prepack.create_vision_backbone_block_context(
                norm1_weight,
                norm1_bias,
                norm_eps,
                qkv_weight,
                qkv_bias,
                num_heads,
                proj_weight,
                proj_bias,
                ls1_gamma,
                norm2_weight,
                norm2_bias,
                norm_eps,
                fc1_weight,
                fc1_bias,
                fc2_weight,
                fc2_bias,
                ls2_gamma,
                "depth.dino.backbone.block.test",
            )
            actual = torch.ops.vulkan_prepack.run_vision_backbone_block_context(
                x.to("vulkan"),
                context,
            ).cpu()

        self._assert_outputs_close(expected, actual, atol=5e-3, rtol=5e-3)

    def test_vulkan_planning_runtime_ops_expose_scheduler_bridge(self):
        script = """
            import torch

            assert hasattr(torch.ops.vulkan_prepack, "query_runtime_policy")
            assert hasattr(torch.ops.vulkan_prepack, "create_kv_cache_storage_for_request")
            assert hasattr(torch.ops.vulkan_prepack, "create_scratch_arena_storage_for_request")
            assert hasattr(torch.ops.vulkan_prepack, "run_scheduled_gated_delta_rule_chunk")
            assert hasattr(torch.ops.vulkan_prepack, "run_scheduled_gated_delta_rule_recurrent")

            prototype = torch.randn(1, dtype=torch.float32).to("vulkan")
            decode_policy = list(torch.ops.vulkan_prepack.query_runtime_policy(
                prototype,
                11,  # LLMDecode
                2,   # LLM
                2,   # Decode
                0,   # Input
            ))
            cache_policy = list(torch.ops.vulkan_prepack.query_runtime_policy(
                prototype,
                3,   # AttentionCache
                2,   # LLM
                2,   # Decode
                3,   # Cache
            ))

            assert len(decode_policy) == 21
            assert decode_policy[0] == 2  # backend_route=Split
            assert decode_policy[6] == 1  # has_scratch_plan
            assert cache_policy[1] == 1  # has_kv_cache_plan
            assert decode_policy[11] == 1  # linear_kernel_family=UnifiedBufferView
            assert decode_policy[13] == 2  # attention_kernel_family=SplitCoordinator
            assert cache_policy[13] == 2  # attention_kernel_family=SplitCoordinator
            assert decode_policy[14] == 1  # has_boundary_plan
            assert decode_policy[15] == 1  # boundary_kind=LLMLinearAttentionSplit
            assert decode_policy[18] == 1  # boundary_backend_owned_execution
            assert decode_policy[19] == 1  # boundary_requires_scratch
            assert decode_policy[20] == 1  # boundary_preferred_cpu_threads

            kv_cache = torch.ops.vulkan_prepack.create_kv_cache_storage_for_request(
                prototype,
                [1, 8, 16, 128],
                2,
                3,  # AttentionCache
                2,  # LLM
                2,  # Decode
                3,  # Cache
            )
            scratch = torch.ops.vulkan_prepack.create_scratch_arena_storage_for_request(
                prototype,
                65536,
                256,
                11,  # LLMDecode
                2,   # LLM
                2,   # Decode
                4,   # Scratch
            )

            assert kv_cache.device.type == "vulkan"
            assert scratch.device.type == "vulkan"
            assert kv_cache.numel() > 0
            assert scratch.numel() >= 65536
            print("ok")
        """

        _, result = self._run_repo_python_subprocess(
            script,
            error_prefix="Planning runtime bridge subprocess failed.",
        )
        self.assertIn("ok", result.stdout)

    def test_vulkan_scheduled_gated_delta_runtime_ops_match_reference(self):
        script = """
            import torch
            import torch.nn.functional as F

            def l2norm(x, dim=-1, eps=1e-6):
                inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
                return x * inv_norm

            def ref_chunk(
                query,
                key,
                value,
                g,
                beta,
                chunk_size=64,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            ):
                initial_dtype = query.dtype
                if use_qk_l2norm_in_kernel:
                    query = l2norm(query, dim=-1, eps=1e-6)
                    key = l2norm(key, dim=-1, eps=1e-6)
                query, key, value, beta, g = [
                    x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
                ]

                batch_size, num_heads, sequence_length, k_head_dim = key.shape
                v_head_dim = value.shape[-1]
                pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
                query = F.pad(query, (0, 0, 0, pad_size))
                key = F.pad(key, (0, 0, 0, pad_size))
                value = F.pad(value, (0, 0, 0, pad_size))
                beta = F.pad(beta, (0, pad_size))
                g = F.pad(g, (0, pad_size))
                total_sequence_length = sequence_length + pad_size
                scale = 1 / (query.shape[-1] ** 0.5)
                query = query * scale

                v_beta = value * beta.unsqueeze(-1)
                k_beta = key * beta.unsqueeze(-1)
                query, key, value, k_beta, v_beta = [
                    x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                    for x in (query, key, value, k_beta, v_beta)
                ]
                g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
                mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

                g = g.cumsum(dim=-1)
                decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
                attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
                for i in range(1, chunk_size):
                    row = attn[..., i, :i].clone()
                    sub = attn[..., :i, :i].clone()
                    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
                attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
                value = attn @ v_beta
                k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
                last_recurrent_state = (
                    torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
                    if initial_state is None
                    else initial_state.to(value)
                )
                core_attn_out = torch.zeros_like(value)
                mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

                for i in range(0, total_sequence_length // chunk_size):
                    q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
                    attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
                    v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
                    v_new = v_i - v_prime
                    attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
                    core_attn_out[:, :, i] = attn_inter + attn @ v_new
                    last_recurrent_state = (
                        last_recurrent_state * g[:, :, i, -1, None, None].exp()
                        + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
                    )

                if not output_final_state:
                    last_recurrent_state = None
                core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
                core_attn_out = core_attn_out[:, :, :sequence_length]
                core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
                return core_attn_out, last_recurrent_state

            def ref_recurrent(
                query,
                key,
                value,
                g,
                beta,
                initial_state,
                output_final_state,
                use_qk_l2norm_in_kernel=False,
            ):
                initial_dtype = query.dtype
                if use_qk_l2norm_in_kernel:
                    query = l2norm(query, dim=-1, eps=1e-6)
                    key = l2norm(key, dim=-1, eps=1e-6)
                query, key, value, beta, g = [
                    x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
                ]

                batch_size, num_heads, sequence_length, k_head_dim = key.shape
                v_head_dim = value.shape[-1]
                scale = 1 / (query.shape[-1] ** 0.5)
                query = query * scale

                core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
                last_recurrent_state = (
                    torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
                    if initial_state is None
                    else initial_state.to(value)
                )

                for i in range(sequence_length):
                    q_t = query[:, :, i]
                    k_t = key[:, :, i]
                    v_t = value[:, :, i]
                    g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
                    beta_t = beta[:, :, i].unsqueeze(-1)

                    last_recurrent_state = last_recurrent_state * g_t
                    kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
                    delta = (v_t - kv_mem) * beta_t
                    last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
                    core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

                if not output_final_state:
                    last_recurrent_state = None
                core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
                return core_attn_out, last_recurrent_state

            torch.manual_seed(0)
            query = torch.randn(1, 5, 2, 8, dtype=torch.float32)
            key = torch.randn(1, 5, 2, 8, dtype=torch.float32)
            value = torch.randn(1, 5, 2, 6, dtype=torch.float32)
            g = torch.randn(1, 5, 2, dtype=torch.float32)
            beta = torch.sigmoid(torch.randn(1, 5, 2, dtype=torch.float32))
            initial_state = torch.randn(1, 2, 8, 6, dtype=torch.float32)

            chunk_ref = ref_chunk(
                query,
                key,
                value,
                g,
                beta,
                chunk_size=4,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            recurrent_ref = ref_recurrent(
                query,
                key,
                value,
                g,
                beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

            query_vk = query.to("vulkan")
            key_vk = key.to("vulkan")
            value_vk = value.to("vulkan")
            g_vk = g.to("vulkan")
            beta_vk = beta.to("vulkan")
            initial_state_vk = initial_state.to("vulkan")

            chunk_out = torch.ops.vulkan_prepack.run_scheduled_gated_delta_rule_chunk(
                query_vk,
                key_vk,
                value_vk,
                g_vk,
                beta_vk,
                4,
                initial_state_vk,
                True,
                True,
            )
            recurrent_out = torch.ops.vulkan_prepack.run_scheduled_gated_delta_rule_recurrent(
                query_vk,
                key_vk,
                value_vk,
                g_vk,
                beta_vk,
                initial_state_vk,
                True,
                True,
            )

            torch.testing.assert_close(chunk_out[0].cpu(), chunk_ref[0], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(chunk_out[1].cpu(), chunk_ref[1], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(recurrent_out[0].cpu(), recurrent_ref[0], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(recurrent_out[1].cpu(), recurrent_ref[1], rtol=1e-5, atol=1e-5)
            print("ok")
        """

        _, result = self._run_repo_python_subprocess(
            script,
            error_prefix="Scheduled gated-delta backend subprocess failed.",
        )
        self.assertIn("ok", result.stdout)

    def test_vulkan_scheduled_gated_delta_runtime_uses_scratch_plan(self):
        policy_log_name = "vulkan_gated_delta_runtime_policy_test.log"
        object_log_name = "vulkan_gated_delta_execution_object_test.log"
        program_log_name = "vulkan_gated_delta_execution_program_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        policy_log_path = os.path.join(repo_root, policy_log_name)
        object_log_path = os.path.join(repo_root, object_log_name)
        program_log_path = os.path.join(repo_root, program_log_name)
        for path in (policy_log_path, object_log_path, program_log_path):
            if os.path.exists(path):
                os.remove(path)

        try:
            script = """
                import torch

                torch.manual_seed(0)
                query = torch.randn(1, 5, 2, 8, dtype=torch.float32).to("vulkan")
                key = torch.randn(1, 5, 2, 8, dtype=torch.float32).to("vulkan")
                value = torch.randn(1, 5, 2, 6, dtype=torch.float32).to("vulkan")
                g = torch.randn(1, 5, 2, dtype=torch.float32).to("vulkan")
                beta = torch.sigmoid(torch.randn(1, 5, 2, dtype=torch.float32)).to("vulkan")
                initial_state = torch.randn(1, 2, 8, 6, dtype=torch.float32).to("vulkan")

                with torch.inference_mode():
                    for _ in range(2):
                        torch.ops.vulkan_prepack.run_scheduled_gated_delta_rule_chunk(
                            query,
                            key,
                            value,
                            g,
                            beta,
                            4,
                            initial_state,
                            True,
                            True,
                        )
                        torch.ops.vulkan_prepack.run_scheduled_gated_delta_rule_recurrent(
                            query,
                            key,
                            value,
                            g,
                            beta,
                            initial_state,
                            True,
                            True,
                        )
                print("ok")
            """

            _, result = self._run_repo_python_subprocess(
                script,
                extra_env={
                    "PYTORCH_VULKAN_RUNTIME_POLICY_LOG": policy_log_name,
                    "PYTORCH_VULKAN_EXECUTION_OBJECT_LOG": object_log_name,
                    "PYTORCH_VULKAN_EXECUTION_PROGRAM_LOG": program_log_name,
                },
                error_prefix="Scheduled gated-delta runtime subprocess failed.",
            )
            self.assertIn("ok", result.stdout)

            self.assertTrue(os.path.exists(policy_log_path))
            with open(policy_log_path, "r", encoding="utf-8") as log_file:
                policy_log_text = log_file.read()

            self.assertRegex(
                policy_log_text,
                r"runtime_policy workload=LLMDecode model_domain=LLM execution_phase=Prefill .* has_execution_program_plan=1 execution_program_kind=GatedDeltaSplit .* has_scratch_arena_plan=1 inferred_from_label=0",
            )
            self.assertRegex(
                policy_log_text,
                r"runtime_policy workload=LLMDecode model_domain=LLM execution_phase=Decode .* has_execution_program_plan=1 execution_program_kind=GatedDeltaSplit .* has_scratch_arena_plan=1 inferred_from_label=0",
            )

            self.assertTrue(os.path.exists(object_log_path))
            with open(object_log_path, "r", encoding="utf-8") as log_file:
                object_log_text = log_file.read()

            self.assertIn("execution_object_event kind=ScratchArena event=store", object_log_text)
            self.assertTrue(os.path.exists(program_log_path))
            with open(program_log_path, "r", encoding="utf-8") as log_file:
                program_log_text = log_file.read()

            self.assertIn(
                "execution_program event=store kind=GatedDeltaSplit",
                program_log_text,
            )
            self.assertIn(
                "execution_program event=hit kind=GatedDeltaSplit",
                program_log_text,
            )
        finally:
            for path in (policy_log_path, object_log_path, program_log_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_vulkan_scheduled_gated_delta_cpu_single_chunk_uses_recurrent_shortcut(self):
        script = """
            import torch

            torch.manual_seed(0)
            query = torch.randn(1, 5, 2, 8, dtype=torch.float32)
            key = torch.randn(1, 5, 2, 8, dtype=torch.float32)
            value = torch.randn(1, 5, 2, 6, dtype=torch.float32)
            g = torch.randn(1, 5, 2, dtype=torch.float32)
            beta = torch.sigmoid(torch.randn(1, 5, 2, dtype=torch.float32))
            initial_state = torch.randn(1, 2, 8, 6, dtype=torch.float32)

            with torch.inference_mode():
                chunk_out = torch.ops.vulkan_prepack.run_scheduled_gated_delta_rule_chunk(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    8,
                    initial_state,
                    True,
                    True,
                )
                recurrent_out = torch.ops.vulkan_prepack.run_scheduled_gated_delta_rule_recurrent(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    initial_state,
                    True,
                    True,
                )

            torch.testing.assert_close(chunk_out[0], recurrent_out[0], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(chunk_out[1], recurrent_out[1], rtol=1e-5, atol=1e-5)
            print("ok")
        """

        _, result = self._run_repo_python_subprocess(
            script,
            error_prefix="Scheduled gated-delta CPU single-chunk subprocess failed.",
        )
        self.assertIn("ok", result.stdout)

    def test_vulkan_scheduled_gated_delta_recurrent_native_buffer_op_hit(self):
        log_name = "vulkan_gated_delta_recurrent_native_buffer_op_hit_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(repo_root, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)

        try:
            script = """
                import torch

                torch.manual_seed(0)
                query = torch.randn(1, 5, 2, 8, dtype=torch.float32).to("vulkan")
                key = torch.randn(1, 5, 2, 8, dtype=torch.float32).to("vulkan")
                value = torch.randn(1, 5, 2, 6, dtype=torch.float32).to("vulkan")
                g = torch.randn(1, 5, 2, dtype=torch.float32).to("vulkan")
                beta = torch.sigmoid(torch.randn(1, 5, 2, dtype=torch.float32)).to("vulkan")
                initial_state = torch.randn(1, 2, 8, 6, dtype=torch.float32).to("vulkan")

                with torch.inference_mode():
                    torch.ops.vulkan_prepack.run_scheduled_gated_delta_rule_recurrent(
                        query,
                        key,
                        value,
                        g,
                        beta,
                        initial_state,
                        True,
                        True,
                    )
                print("ok")
            """

            _, result = self._run_repo_python_subprocess(
                script,
                extra_env={"PYTORCH_VULKAN_OP_HIT_LOG": log_name},
                error_prefix="Scheduled gated-delta recurrent native-buffer subprocess failed.",
            )
            self.assertIn("ok", result.stdout)

            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as log_file:
                log_text = log_file.read()

            self.assertIn(
                "vulkan_prepack::run_scheduled_gated_delta_rule_recurrent.native_buffer",
                log_text,
            )
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_vulkan_scheduled_gated_delta_chunk_native_full_sequence_op_hit(self):
        log_name = "vulkan_gated_delta_chunk_native_full_sequence_op_hit_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(repo_root, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)

        try:
            script = """
                import torch

                torch.manual_seed(0)
                query = torch.randn(1, 17, 2, 8, dtype=torch.float32).to("vulkan")
                key = torch.randn(1, 17, 2, 8, dtype=torch.float32).to("vulkan")
                value = torch.randn(1, 17, 2, 6, dtype=torch.float32).to("vulkan")
                g = torch.randn(1, 17, 2, dtype=torch.float32).to("vulkan")
                beta = torch.sigmoid(torch.randn(1, 17, 2, dtype=torch.float32)).to("vulkan")

                with torch.inference_mode():
                    torch.ops.vulkan_prepack.run_scheduled_gated_delta_rule_chunk(
                        query,
                        key,
                        value,
                        g,
                        beta,
                        8,
                        None,
                        True,
                        True,
                    )
                print("ok")
            """

            _, result = self._run_repo_python_subprocess(
                script,
                extra_env={"PYTORCH_VULKAN_OP_HIT_LOG": log_name},
                error_prefix="Scheduled gated-delta chunk native full-sequence subprocess failed.",
            )
            self.assertIn("ok", result.stdout)

            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as log_file:
                log_text = log_file.read()

            self.assertIn(
                "vulkan_prepack::run_scheduled_gated_delta_rule_chunk.native_full_sequence_recurrent",
                log_text,
            )
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_vulkan_qwen_linear_attention_prefill_context_matches_reference(self):
        script = """
            import torch
            import torch.nn.functional as F

            torch.manual_seed(0)
            batch_size, seq_len, hidden_size = 1, 17, 16
            num_k_heads, num_v_heads = 2, 4
            head_k_dim, head_v_dim = 4, 3
            key_dim = num_k_heads * head_k_dim
            value_dim = num_v_heads * head_v_dim
            qkv_out = key_dim * 2 + value_dim

            assert hasattr(torch.ops.vulkan_prepack, "create_qwen_linear_attention_prefill_context")
            assert hasattr(torch.ops.vulkan_prepack, "run_qwen_linear_attention_prefill_context")

            x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
            qkv_weight = torch.randn(qkv_out, hidden_size, dtype=torch.float32)
            z_weight = torch.randn(value_dim, hidden_size, dtype=torch.float32)
            a_weight = torch.randn(num_v_heads, hidden_size, dtype=torch.float32)
            b_weight = torch.randn(num_v_heads, hidden_size, dtype=torch.float32)
            out_weight = torch.randn(hidden_size, value_dim, dtype=torch.float32)
            conv_weight = torch.randn(qkv_out, 1, 4, dtype=torch.float32)
            norm_weight = torch.randn(head_v_dim, dtype=torch.float32)
            A_log = torch.randn(num_v_heads, dtype=torch.float32)
            dt_bias = torch.randn(num_v_heads, dtype=torch.float32)

            mixed_qkv = F.linear(x, qkv_weight, None).transpose(1, 2)
            mixed_qkv = F.conv1d(
                mixed_qkv,
                conv_weight,
                None,
                stride=1,
                padding=3,
                dilation=1,
                groups=qkv_out,
            )
            mixed_qkv = F.silu(mixed_qkv[:, :, :seq_len]).transpose(1, 2).contiguous()

            z = F.linear(x, z_weight, None).reshape(batch_size, seq_len, -1, head_v_dim)
            a = F.linear(x, a_weight, None)
            b = F.linear(x, b_weight, None)

            query, key, value = torch.split(
                mixed_qkv,
                [key_dim, key_dim, value_dim],
                dim=-1,
            )
            query = query.reshape(batch_size, seq_len, -1, head_k_dim)
            key = key.reshape(batch_size, seq_len, -1, head_k_dim)
            value = value.reshape(batch_size, seq_len, -1, head_v_dim)
            query = query.repeat_interleave(num_v_heads // num_k_heads, dim=2)
            key = key.repeat_interleave(num_v_heads // num_k_heads, dim=2)

            beta = b.sigmoid()
            g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
            core_attn_out, _ = torch.ops.vulkan_prepack.run_scheduled_gated_delta_rule_chunk(
                query.to("vulkan"),
                key.to("vulkan"),
                value.to("vulkan"),
                g.to("vulkan"),
                beta.to("vulkan"),
                8,
                None,
                False,
                True,
            )
            core_attn_out = core_attn_out.cpu().reshape(-1, head_v_dim)
            z = z.reshape(-1, head_v_dim)
            variance = (core_attn_out.float() * core_attn_out.float()).mean(-1, keepdim=True)
            core_attn_out = core_attn_out.float() * torch.rsqrt(variance + 1.0e-6)
            core_attn_out = norm_weight * core_attn_out
            core_attn_out = core_attn_out * F.silu(z.float())
            reference = F.linear(
                core_attn_out.reshape(batch_size, seq_len, value_dim),
                out_weight,
                None,
            )

            context = torch.ops.vulkan_prepack.create_qwen_linear_attention_prefill_context(
                qkv_weight,
                z_weight,
                a_weight,
                b_weight,
                out_weight,
                conv_weight,
                None,
                norm_weight,
                A_log,
                dt_bias,
                key_dim,
                value_dim,
                head_k_dim,
                head_v_dim,
                num_k_heads,
                num_v_heads,
                8,
                1.0e-6,
                "test_qwen_prefill",
            )
            actual = torch.ops.vulkan_prepack.run_qwen_linear_attention_prefill_context(
                x.to("vulkan"),
                context,
            ).cpu()

            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=2e-5)
            print("ok")
        """

        _, result = self._run_repo_python_subprocess(
            script,
            error_prefix="Qwen linear-attention prefill context subprocess failed.",
        )
        self.assertIn("ok", result.stdout)

    def test_vulkan_qwen_linear_attention_decode_context_matches_reference(self):
        script = """
            import torch
            import torch.nn.functional as F

            def l2norm(x, dim=-1, eps=1e-6):
                inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
                return x * inv_norm

            def ref_recurrent(query, key, value, g, beta, initial_state):
                query = l2norm(query, dim=-1, eps=1e-6)
                key = l2norm(key, dim=-1, eps=1e-6)
                query, key, value, beta, g = [
                    x.transpose(1, 2).contiguous().to(torch.float32)
                    for x in (query, key, value, beta, g)
                ]
                scale = 1 / (query.shape[-1] ** 0.5)
                query = query * scale
                q_t = query[:, :, 0]
                k_t = key[:, :, 0]
                v_t = value[:, :, 0]
                g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
                beta_t = beta[:, :, 0].unsqueeze(-1)

                next_state = initial_state.to(torch.float32) * g_t
                kv_mem = (next_state * k_t.unsqueeze(-1)).sum(dim=-2)
                delta = (v_t - kv_mem) * beta_t
                next_state = next_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
                out = (next_state * q_t.unsqueeze(-1)).sum(dim=-2)
                return out[:, :, None, :].transpose(1, 2).contiguous(), next_state

            torch.manual_seed(0)
            batch_size, hidden_size = 1, 16
            num_k_heads, num_v_heads = 2, 4
            head_k_dim, head_v_dim = 4, 3
            key_dim = num_k_heads * head_k_dim
            value_dim = num_v_heads * head_v_dim
            qkv_out = key_dim * 2 + value_dim
            conv_state_len = 4

            x = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32)
            conv_state = torch.randn(batch_size, qkv_out, conv_state_len, dtype=torch.float32)
            recurrent_state = torch.randn(batch_size, num_v_heads, head_k_dim, head_v_dim, dtype=torch.float32)
            qkv_weight = torch.randn(qkv_out, hidden_size, dtype=torch.float32)
            z_weight = torch.randn(value_dim, hidden_size, dtype=torch.float32)
            a_weight = torch.randn(num_v_heads, hidden_size, dtype=torch.float32)
            b_weight = torch.randn(num_v_heads, hidden_size, dtype=torch.float32)
            out_weight = torch.randn(hidden_size, value_dim, dtype=torch.float32)
            conv_weight = torch.randn(qkv_out, 1, conv_state_len, dtype=torch.float32)
            norm_weight = torch.randn(head_v_dim, dtype=torch.float32)
            A_log = torch.randn(num_v_heads, dtype=torch.float32)
            dt_bias = torch.randn(num_v_heads, dtype=torch.float32)

            mixed_qkv = F.linear(x, qkv_weight, None).transpose(1, 2)
            conv_input = torch.cat([conv_state, mixed_qkv], dim=-1).to(torch.float32)
            next_conv_state_ref = conv_input[:, :, -conv_state_len:].contiguous()
            mixed_qkv = F.conv1d(
                conv_input,
                conv_weight,
                None,
                stride=1,
                padding=0,
                dilation=1,
                groups=qkv_out,
            )
            mixed_qkv = F.silu(mixed_qkv[:, :, -1:]).transpose(1, 2).contiguous()

            z = F.linear(x, z_weight, None)
            a = F.linear(x, a_weight, None)
            b = F.linear(x, b_weight, None)

            query, key, value = torch.split(
                mixed_qkv,
                [key_dim, key_dim, value_dim],
                dim=-1,
            )
            query = query.reshape(batch_size, 1, -1, head_k_dim)
            key = key.reshape(batch_size, 1, -1, head_k_dim)
            value = value.reshape(batch_size, 1, -1, head_v_dim)
            query = query.repeat_interleave(num_v_heads // num_k_heads, dim=2)
            key = key.repeat_interleave(num_v_heads // num_k_heads, dim=2)

            beta = b.sigmoid()
            g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
            core_attn_out, next_recurrent_state_ref = ref_recurrent(
                query,
                key,
                value,
                g,
                beta,
                recurrent_state,
            )
            core_attn_out = core_attn_out.reshape(-1, head_v_dim)
            z = z.reshape(-1, head_v_dim)
            variance = (core_attn_out.float() * core_attn_out.float()).mean(-1, keepdim=True)
            core_attn_out = core_attn_out.float() * torch.rsqrt(variance + 1.0e-6)
            core_attn_out = norm_weight * core_attn_out
            core_attn_out = core_attn_out * F.silu(z.float())
            reference = F.linear(
                core_attn_out.reshape(batch_size, 1, value_dim),
                out_weight,
                None,
            )

            context = torch.ops.vulkan_prepack.create_qwen_linear_attention_prefill_context(
                qkv_weight,
                z_weight,
                a_weight,
                b_weight,
                out_weight,
                conv_weight,
                None,
                norm_weight,
                A_log,
                dt_bias,
                key_dim,
                value_dim,
                head_k_dim,
                head_v_dim,
                num_k_heads,
                num_v_heads,
                64,
                1.0e-6,
                "test_qwen_decode",
            )
            actual, next_conv_state, next_recurrent_state = (
                torch.ops.vulkan_prepack.run_qwen_linear_attention_decode_context(
                    x.to("vulkan"),
                    conv_state,
                    recurrent_state,
                    context,
                )
            )

            assert actual.device.type == "vulkan"
            assert next_conv_state.device.type == "vulkan"
            assert next_recurrent_state.device.type == "vulkan"

            torch.testing.assert_close(actual.cpu(), reference, rtol=1e-5, atol=2e-5)
            torch.testing.assert_close(
                next_conv_state.cpu(),
                next_conv_state_ref,
                rtol=1e-5,
                atol=2e-5,
            )
            torch.testing.assert_close(
                next_recurrent_state.cpu(),
                next_recurrent_state_ref,
                rtol=1e-5,
                atol=2e-5,
            )
            print("ok")
        """

        _, result = self._run_repo_python_subprocess(
            script,
            error_prefix="Qwen linear-attention decode context subprocess failed.",
        )
        self.assertIn("ok", result.stdout)

    def test_rms_norm_runtime_hits_fused_width_kernel(self):
        log_name = "vulkan_rms_norm_fused_width_op_hit_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(repo_root, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)

        try:
            script = """
                import torch
                import torch.nn.functional as F

                torch.manual_seed(0)
                x = torch.randn(1, 257, 384, dtype=torch.float32).to("vulkan")
                weight = torch.randn(384, dtype=torch.float32)
                with torch.inference_mode():
                    y = F.rms_norm(x, (384,), weight, 1e-6)
                    print(float(y.cpu().sum()))
            """

            self._run_repo_python_subprocess(
                script,
                extra_env={"PYTORCH_VULKAN_OP_HIT_LOG": log_name},
                error_prefix="RMSNorm fused-width subprocess failed.",
            )

            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as log_file:
                log_text = log_file.read()

            self.assertIn("op=aten::rms_norm", log_text)
            self.assertIn("op=aten::rms_norm.fused_width", log_text)
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_layer_norm_runtime_hits_fused_width_kernel(self):
        log_name = "vulkan_layer_norm_fused_width_op_hit_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(repo_root, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)

        try:
            script = """
                import torch
                import torch.nn.functional as F

                torch.manual_seed(0)
                x = torch.randn(1, 257, 384, dtype=torch.float32).to("vulkan")
                weight = torch.randn(384, dtype=torch.float32)
                bias = torch.randn(384, dtype=torch.float32)
                with torch.inference_mode():
                    y = F.layer_norm(x, (384,), weight, bias, 1e-5)
                    print(tuple(y.shape))
                    print(y.cpu()[0, 0, :4])
            """

            _, result = self._run_repo_python_subprocess(
                script,
                extra_env={"PYTORCH_VULKAN_OP_HIT_LOG": log_name},
                error_prefix="Vulkan layer_norm fused-width subprocess failed.",
            )
            self.assertIn("(1, 257, 384)", result.stdout)

            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as log_file:
                log_text = log_file.read()

            self.assertIn("op=aten::layer_norm", log_text)
            self.assertIn("op=aten::layer_norm.fused_width", log_text)
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_layer_norm_runtime_family_is_consumed(self):
        log_name = "vulkan_layer_norm_family_op_hit_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(repo_root, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)

        try:
            script = """
                import torch
                import torch.nn.functional as F

                torch.manual_seed(0)
                x = torch.randn(1, 8, 1024, dtype=torch.float32).to("vulkan")
                weight = torch.randn(1024, dtype=torch.float32)
                bias = torch.randn(1024, dtype=torch.float32)

                with torch.inference_mode():
                    y = F.layer_norm(x, (1024,), weight, bias, 1e-5)
                    print(float(y.cpu().sum()))
            """

            self._run_repo_python_subprocess(
                script,
                extra_env={"PYTORCH_VULKAN_OP_HIT_LOG": log_name},
                error_prefix="Vulkan layer_norm family subprocess failed.",
            )

            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as log_file:
                log_text = log_file.read()

            self.assertIn("op=aten::norm.family_texture_width", log_text)
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_native_layer_norm_runtime_hits_fused_width_kernel(self):
        log_name = "vulkan_native_layer_norm_fused_width_op_hit_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(repo_root, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)

        try:
            script = """
                import torch

                torch.manual_seed(0)
                x = torch.randn(1, 257, 384, dtype=torch.float32).to("vulkan")
                weight = torch.randn(384, dtype=torch.float32)
                bias = torch.randn(384, dtype=torch.float32)
                with torch.inference_mode():
                    y, mean, rstd = torch.native_layer_norm(
                        x,
                        (384,),
                        weight,
                        bias,
                        1e-6,
                    )
                    print(float(y.cpu().sum()))
            """

            self._run_repo_python_subprocess(
                script,
                extra_env={"PYTORCH_VULKAN_OP_HIT_LOG": log_name},
                error_prefix="Native-layer-norm fused-width subprocess failed.",
            )

            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as log_file:
                log_text = log_file.read()

            self.assertIn("op=aten::native_layer_norm", log_text)
            self.assertIn("op=aten::native_layer_norm.fused_width", log_text)
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_scaled_dot_product_attention_runtime_hits_vulkan_kernel(self):
        log_name = "vulkan_sdpa_op_hit_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(repo_root, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)

        try:
            script = """
                import torch
                import torch.nn.functional as F

                torch.manual_seed(0)
                q = torch.randn(2, 9, 8, dtype=torch.float32).to("vulkan")
                k = torch.randn(2, 7, 8, dtype=torch.float32).to("vulkan")
                v = torch.randn(2, 7, 8, dtype=torch.float32).to("vulkan")
                with torch.inference_mode():
                    out = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=0.0,
                        scale=0.125,
                    )
                    print(float(out.cpu().sum()))
            """

            self._run_repo_python_subprocess(
                script,
                extra_env={"PYTORCH_VULKAN_OP_HIT_LOG": log_name},
                error_prefix="Scaled-dot-product attention subprocess failed.",
            )

            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as log_file:
                log_text = log_file.read()

            self.assertTrue(
                (
                    "op=aten::scaled_dot_product_attention" in log_text
                    or "op=aten::_scaled_dot_product_attention_math" in log_text
                ),
                msg=(
                    "Expected the Vulkan SDPA runtime path to hit either the "
                    "public aten::scaled_dot_product_attention kernel or the "
                    "Vulkan aten::_scaled_dot_product_attention_math kernel."
                ),
            )
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_scaled_dot_product_attention_runtime_family_is_consumed(self):
        log_name = "vulkan_sdpa_family_op_hit_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(repo_root, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)

        try:
            script = """
                import torch
                import torch.nn.functional as F

                torch.manual_seed(0)
                q = torch.randn(1, 2, 9, 8, dtype=torch.float32).to("vulkan")
                k = torch.randn(1, 2, 7, 8, dtype=torch.float32).to("vulkan")
                v = torch.randn(1, 2, 7, 8, dtype=torch.float32).to("vulkan")

                with torch.inference_mode():
                    y = F.scaled_dot_product_attention(q, k, v)
                    print(float(y.cpu().sum()))
            """

            self._run_repo_python_subprocess(
                script,
                extra_env={"PYTORCH_VULKAN_OP_HIT_LOG": log_name},
                error_prefix="Vulkan SDPA family subprocess failed.",
            )

            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as log_file:
                log_text = log_file.read()

            self.assertIn(
                "op=aten::scaled_dot_product_attention.family_texture_math",
                log_text,
            )
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_execution_plan_logging(self):
        log_name = "execution_plan_logging_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(repo_root, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)

        try:
            script = """
                import torch
                import torch.nn.functional as F

                torch.manual_seed(0)
                x = torch.randn(1, 16, 32, dtype=torch.float32).to("vulkan")
                w = torch.randn(64, 32, dtype=torch.float32).to("vulkan")
                b = torch.randn(64, dtype=torch.float32).to("vulkan")

                with torch.inference_mode():
                    y = F.linear(x, w, b)
                    z = torch.exp(y)
                    r = z.sum(dim=-1)
                    print(float(r.cpu().sum()))
                """

            _, result = self._run_repo_python_subprocess(
                script,
                extra_env={"PYTORCH_VULKAN_EXECUTION_PLAN_LOG": log_name},
                error_prefix="Execution-plan logging subprocess failed.",
            )

            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as log_file:
                log_text = log_file.read()

            self.assertIn("execution_plan_summary", log_text)
            self.assertIn("kind=LinearInputSource", log_text)
            self.assertIn("kind=ElementwiseInput", log_text)
            self.assertIn("kind=ReductionDimInput", log_text)
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_linear_avoids_width_relayout_for_channels_packed_input(self):
        log_name = "linear_materialize_log_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(repo_root, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)

        try:
            script = """
                import torch
                import torch.nn.functional as F

                torch.manual_seed(0)
                x = torch.randn(257, 384, dtype=torch.float32).to("vulkan")
                w = torch.randn(1152, 384, dtype=torch.float32).to("vulkan")
                b = torch.randn(1152, dtype=torch.float32).to("vulkan")

                with torch.inference_mode():
                    y = F.linear(x, w, b)
                    print(float(y.cpu().sum()))
            """

            self._run_repo_python_subprocess(
                script,
                extra_env={"PYTORCH_VULKAN_MATERIALIZE_LOG": log_name},
                error_prefix="Linear materialization subprocess failed.",
            )

            log_text = ""
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as log_file:
                    log_text = log_file.read()

            self.assertNotIn(
                "caller=linear path=image_layout_convert_width",
                log_text,
            )
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_linear_runtime_prefill_family_is_consumed_in_mm(self):
        log_name = "linear_channel_packed_family_op_hit_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(repo_root, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)

        try:
            script = """
                import torch
                import torch.nn.functional as F

                torch.manual_seed(0)
                x = torch.randn(1, 8, 1024, dtype=torch.float32).to("vulkan")
                w = torch.randn(1024, 1024, dtype=torch.float32).to("vulkan")
                b = torch.randn(1024, dtype=torch.float32).to("vulkan")

                with torch.inference_mode():
                    y = F.linear(x, w, b)
                    print(float(y.cpu().sum()))
            """

            self._run_repo_python_subprocess(
                script,
                extra_env={"PYTORCH_VULKAN_OP_HIT_LOG": log_name},
                error_prefix="Linear channel-packed family subprocess failed.",
            )

            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as log_file:
                log_text = log_file.read()

            self.assertIn("op=aten::linear", log_text)
            self.assertIn("op=aten::linear.channel_packed_family", log_text)
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_texture_contiguous_reshape_hits_backend_fast_path(self):
        log_name = "texture_contiguous_reshape_op_hit_test.log"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(repo_root, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)

        try:
            script = """
                import torch
                import torch.nn.functional as F

                torch.manual_seed(0)
                x = torch.randn(1, 17, 8, dtype=torch.float32).to("vulkan")
                weight = torch.randn(24, 8, dtype=torch.float32).to("vulkan")
                bias = torch.randn(24, dtype=torch.float32).to("vulkan")

                with torch.inference_mode():
                    qkv = F.linear(x, weight, bias).reshape(1, 17, 3, 8)
                    q = qkv[:, :, 0].reshape(1, 17, 2, 4)
                    q = q.permute(0, 2, 1, 3).reshape(2, 17, 4)
                    print(float(q.cpu().sum()))
            """

            self._run_repo_python_subprocess(
                script,
                extra_env={"PYTORCH_VULKAN_OP_HIT_LOG": log_name},
                error_prefix="Texture contiguous reshape subprocess failed.",
            )

            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as log_file:
                log_text = log_file.read()

            self.assertIn(
                "op=aten::view.texture_contiguous_reshape",
                log_text,
            )
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

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
