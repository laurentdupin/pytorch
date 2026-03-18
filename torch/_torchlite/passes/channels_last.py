"""Channels-last memory format annotation pass.

Marks convolution outputs and their pointwise consumers as channels-last
so downstream passes (fuse, triton codegen) can iterate in NHWC order,
avoiding expensive NCHW↔NHWC layout conversions at runtime.
"""
from typing import List

import torch
from torch.fx import GraphModule

from torch._torchlite.passes.common import (
    _aten_op_name,
    _create_name,
    _node_shape,
    PassResult,
)
from torch._torchlite.passes.fusion import _POINTWISE_OPS


def channels_last(
    gm: GraphModule, example_inputs: List[torch.Tensor]
) -> PassResult:
    graph = gm.graph

    # Only apply CL when there are enough convolutions to amortize the
    # boundary conversion cost (input NCHW→NHWC + output NHWC→NCHW).
    # With ≤2 convolutions the overhead exceeds the savings from avoiding
    # cuDNN's internal format conversions.
    n_convs = sum(
        1 for n in graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.aten.convolution.default
        and _node_shape(n) is not None
        and len(_node_shape(n)) == 4
    )
    if n_convs <= 2:
        return PassResult(gm=gm)

    cl_nodes: set = set()

    for node in graph.nodes:
        if node.op != "call_function":
            continue

        if node.target == torch.ops.aten.convolution.default:
            shape = _node_shape(node)
            if shape is not None and len(shape) == 4:
                node.meta["memory_format"] = "channels_last"
                cl_nodes.add(node.name)
            continue

        op_name = _aten_op_name(node.target)
        if op_name not in _POINTWISE_OPS:
            continue

        shape = _node_shape(node)
        if shape is None or len(shape) != 4:
            continue

        has_cl_input = False
        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and arg.name in cl_nodes:
                arg_shape = _node_shape(arg)
                if arg_shape is not None and len(arg_shape) == 4:
                    has_cl_input = True
                    break

        if has_cl_input:
            node.meta["memory_format"] = "channels_last"
            cl_nodes.add(node.name)

    def _insert_contiguous(out_arg, node):
        if isinstance(out_arg, torch.fx.Node) and out_arg.name in cl_nodes:
            graph.inserting_before(node)
            contig = graph.call_function(
                torch.ops.aten.contiguous.default, (out_arg,)
            )
            contig.name = _create_name(graph, "channels_last_to_contiguous")
            contig.meta["shape"] = out_arg.meta.get("shape")
            contig.meta["dtype"] = out_arg.meta.get("dtype", torch.float32)
            node.replace_input_with(out_arg, contig)

    for node in graph.nodes:
        if node.op != "output":
            continue
        out_arg = node.args[0]
        if isinstance(out_arg, (tuple, list)):
            for oa in out_arg:
                _insert_contiguous(oa, node)
        else:
            _insert_contiguous(out_arg, node)

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)
