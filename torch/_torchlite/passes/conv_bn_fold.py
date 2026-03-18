"""Conv-BN weight folding: fold batch_norm into preceding conv at compile time.

For eval-mode models, batch_norm parameters (gamma, beta, running_mean,
running_var) are fixed constants. When a batch_norm immediately follows a
convolution, we can algebraically absorb the BN transform into the conv's
weight and bias, eliminating the BN op entirely from the graph.

This pass runs BEFORE decompose so BN is still a single node
(_native_batch_norm_legit_no_training or native_batch_norm).
"""
import operator
from typing import List

import torch
from torch.fx import GraphModule

from torch._torchlite.passes.common import (
    _create_name,
    _deep_getattr,
    _deep_setattr,
    _set_phase,
    PassResult,
)


def conv_bn_fold(
    gm: GraphModule, example_inputs: List[torch.Tensor]
) -> PassResult:
    graph = gm.graph
    changed = False

    _BN_TARGETS = {
        torch.ops.aten.native_batch_norm.default,
        torch.ops.aten._native_batch_norm_legit_no_training.default,
    }

    for bn_node in list(graph.nodes):
        if bn_node.op != "call_function" or bn_node.target not in _BN_TARGETS:
            continue

        is_legit_no_training = (
            bn_node.target
            == torch.ops.aten._native_batch_norm_legit_no_training.default
        )

        # _native_batch_norm_legit_no_training signature:
        #   (input, weight, bias, running_mean, running_var, momentum, eps)
        # native_batch_norm signature:
        #   (input, weight, bias, running_mean, running_var, training, momentum, eps)
        if is_legit_no_training:
            if len(bn_node.args) < 7:
                continue
            conv_out = bn_node.args[0]
            bn_weight_node = bn_node.args[1]
            bn_bias_node = bn_node.args[2]
            rm_node = bn_node.args[3]
            rv_node = bn_node.args[4]
            eps = bn_node.args[6]
        else:
            if len(bn_node.args) < 8:
                continue
            training = bn_node.args[5]
            if training:
                continue
            conv_out = bn_node.args[0]
            bn_weight_node = bn_node.args[1]
            bn_bias_node = bn_node.args[2]
            rm_node = bn_node.args[3]
            rv_node = bn_node.args[4]
            eps = bn_node.args[7]

        if not isinstance(conv_out, torch.fx.Node):
            continue
        if conv_out.op != "call_function":
            continue
        if conv_out.target != torch.ops.aten.convolution.default:
            continue

        # All BN params must be get_attr (constant parameters on the module)
        param_nodes = [bn_weight_node, bn_bias_node, rm_node, rv_node]
        if not all(
            isinstance(n, torch.fx.Node) and n.op == "get_attr"
            for n in param_nodes
        ):
            continue

        conv_weight_node = conv_out.args[1]
        conv_bias_node = conv_out.args[2]
        if not isinstance(conv_weight_node, torch.fx.Node):
            continue
        if conv_weight_node.op != "get_attr":
            continue

        gamma = _deep_getattr(gm, bn_weight_node.target)
        beta = _deep_getattr(gm, bn_bias_node.target)
        running_mean = _deep_getattr(gm, rm_node.target)
        running_var = _deep_getattr(gm, rv_node.target)
        conv_weight = _deep_getattr(gm, conv_weight_node.target)

        has_conv_bias = (
            isinstance(conv_bias_node, torch.fx.Node)
            and conv_bias_node.op == "get_attr"
        )
        if has_conv_bias:
            conv_bias = _deep_getattr(gm, conv_bias_node.target)
        else:
            conv_bias = torch.zeros(gamma.shape[0], dtype=conv_weight.dtype,
                                    device=conv_weight.device)

        invstd = torch.rsqrt(running_var + eps)
        coeff = (gamma * invstd).reshape([-1] + [1] * (conv_weight.ndim - 1))
        new_weight = conv_weight * coeff
        new_bias = beta + (gamma * invstd) * (conv_bias - running_mean)

        w_attr = f"_folded_weight_{bn_node.name}"
        b_attr = f"_folded_bias_{bn_node.name}"
        _deep_setattr(gm, w_attr, torch.nn.Parameter(new_weight, requires_grad=False))
        _deep_setattr(gm, b_attr, torch.nn.Parameter(new_bias, requires_grad=False))

        graph.inserting_before(bn_node)
        new_w_node = graph.get_attr(w_attr)
        new_w_node.name = _create_name(graph, "folded_weight")
        new_w_node.meta["shape"] = list(new_weight.shape)
        new_w_node.meta["dtype"] = new_weight.dtype

        new_b_node = graph.get_attr(b_attr)
        new_b_node.name = _create_name(graph, "folded_bias")
        new_b_node.meta["shape"] = list(new_bias.shape)
        new_b_node.meta["dtype"] = new_bias.dtype

        new_conv_args = list(conv_out.args)
        new_conv_args[1] = new_w_node
        new_conv_args[2] = new_b_node
        new_conv_node = graph.call_function(
            torch.ops.aten.convolution.default, tuple(new_conv_args)
        )
        new_conv_node.name = _create_name(graph, "conv_bn_folded")
        new_conv_node.meta.update(conv_out.meta)
        _set_phase(new_conv_node, conv_out.meta.get("phase", "forward"))

        # BN returns (output, save_mean, save_rstd). Replace uses of
        # getitem(bn, 0) with the new conv, and remove getitem(bn, 1/2).
        for user in list(bn_node.users.keys()):
            if (
                user.op == "call_function"
                and user.target is operator.getitem
                and len(user.args) >= 2
            ):
                if user.args[1] == 0:
                    user.replace_all_uses_with(new_conv_node)
                if not user.users:
                    graph.erase_node(user)

        # If BN node itself was used directly (not through getitem),
        # replace those uses too
        if bn_node.users:
            bn_node.replace_all_uses_with(new_conv_node)

        if not bn_node.users:
            graph.erase_node(bn_node)
        if not conv_out.users:
            graph.erase_node(conv_out)

        changed = True

    if changed:
        graph.lint()
        gm.recompile()
    return PassResult(gm=gm, changed=changed)
