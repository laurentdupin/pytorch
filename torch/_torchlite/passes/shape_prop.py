"""Shape propagation pass: populate shape/dtype metadata on FX graph nodes.

When torchlite receives a graph from torch.compile/dynamo, nodes may lack
shape and dtype metadata. This pass runs the graph on meta tensors to infer
shapes without performing actual computation or causing side effects.
Subsequent passes (decompose, fuse, triton_codegen) rely on this metadata.
"""
from typing import Dict, List

import torch
from torch.fx import GraphModule

from torch._torchlite.passes.common import _deep_getattr, _is_torch_op, PassResult


def _to_meta(t):
    if isinstance(t, torch.Tensor):
        return torch.empty(t.shape, dtype=t.dtype, device="meta")
    return t


def _needs_shape_prop(gm):
    """Check whether the graph has nodes that need shape metadata.

    Returns True if any call_function node that could benefit from
    decomposition is missing shape metadata.
    """
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if not _is_torch_op(node.target):
            continue
        if node.meta.get("shape") is None:
            return True
    return False


def shape_prop(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    if not _needs_shape_prop(gm):
        return PassResult(gm=gm)

    graph = gm.graph
    env: Dict[str, object] = {}

    input_idx = 0
    for node in graph.nodes:
        if node.op == "placeholder":
            if input_idx < len(example_inputs):
                val = _to_meta(example_inputs[input_idx])
            else:
                val = None
            env[node.name] = val
            input_idx += 1
            if isinstance(val, torch.Tensor) and "shape" not in node.meta:
                node.meta["shape"] = list(val.shape)
                node.meta["dtype"] = val.dtype

        elif node.op == "get_attr":
            val = _deep_getattr(gm, node.target)
            val = _to_meta(val)
            env[node.name] = val
            if isinstance(val, torch.Tensor) and "shape" not in node.meta:
                node.meta["shape"] = list(val.shape)
                node.meta["dtype"] = val.dtype

        elif node.op == "call_function":
            args = _resolve_args(node.args, env)
            kwargs = _resolve_kwargs(node.kwargs, env)
            try:
                result = node.target(*args, **kwargs)
            except Exception:
                result = None

            env[node.name] = result
            if node.meta.get("shape") is None and isinstance(result, torch.Tensor):
                node.meta["shape"] = list(result.shape)
                node.meta["dtype"] = result.dtype
            elif node.meta.get("shape") is None and isinstance(result, (tuple, list)):
                for r in result:
                    if isinstance(r, torch.Tensor):
                        node.meta.setdefault("shape", list(r.shape))
                        node.meta.setdefault("dtype", r.dtype)

        elif node.op == "call_module":
            mod = _deep_getattr(gm, node.target)
            args = _resolve_args(node.args, env)
            kwargs = _resolve_kwargs(node.kwargs, env)
            try:
                result = mod(*args, **kwargs)
            except Exception:
                result = None
            env[node.name] = result
            if node.meta.get("shape") is None and isinstance(result, torch.Tensor):
                node.meta["shape"] = list(result.shape)
                node.meta["dtype"] = result.dtype

        elif node.op == "output":
            pass

    return PassResult(gm=gm)


def _resolve_args(args, env):
    result = []
    for a in args:
        if isinstance(a, torch.fx.Node):
            result.append(env.get(a.name))
        elif isinstance(a, (list, tuple)):
            result.append(type(a)(_resolve_one(x, env) for x in a))
        else:
            result.append(a)
    return tuple(result)


def _resolve_kwargs(kwargs, env):
    return {k: _resolve_one(v, env) for k, v in kwargs.items()}


def _resolve_one(v, env):
    if isinstance(v, torch.fx.Node):
        return env.get(v.name)
    if isinstance(v, (list, tuple)):
        return type(v)(_resolve_one(x, env) for x in v)
    return v
