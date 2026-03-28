# mypy: allow-untyped-defs

import functools
import logging

import torch
from torch.fx._compatibility import compatibility


logger = logging.getLogger(__name__)

__all__ = ["regional_inductor"]


# standalone_inductor returns a callable class object - this does not sit well
# with Fx graph node op call_function which expects a function. So this is just
# a wrapper function to make Fx graph codegen happy.
def _dummy_wrapper(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)

    return inner


def _compile_submod(gm, prefix):
    from torch._inductor.standalone_compile import AOTCompiledArtifact

    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target.startswith(prefix):
            fake_inputs = []
            for inp_node in node.all_input_nodes:
                if hasattr(inp_node, "meta") and "val" in inp_node.meta:
                    fake_inputs.append(inp_node.meta["val"])
                else:
                    raise RuntimeError(
                        f"Partition is bad because non fake tensor value is seen {inp_node}"
                    )

            submod = getattr(gm, node.target)

            # Get inductor configs from annotation
            # TODO we should change partition when there are multiple differently
            # annotated regions.
            inductor_options = {}
            for sub_node in submod.graph.nodes:
                if hasattr(sub_node, "meta") and sub_node.meta.get("custom", None):
                    custom = sub_node.meta["custom"]
                    if isinstance(custom, dict) and "compile_with_inductor" in custom:
                        compile_value = custom["compile_with_inductor"]
                        if (
                            isinstance(compile_value, dict)
                            and "inductor_configs" in compile_value
                        ):
                            inductor_options = compile_value["inductor_configs"]
                            break

            # Log the options being used
            logger.info(
                "Compiling submodule %s with inductor options: %s",
                node.target,
                inductor_options,
            )

            # Apply config patches before compilation
            import torch._inductor.config as inductor_config

            # Validate that all config keys exist
            for key in inductor_options:
                if not hasattr(inductor_config, key):
                    raise ValueError(
                        f"Invalid inductor config key '{key}' in regional_inductor annotation. "
                        f"Available config keys can be found in torch._inductor.config"
                    )

            with inductor_config.patch(inductor_options):
                compiled_fn = torch._inductor.standalone_compile(
                    submod,
                    fake_inputs,
                    dynamic_shapes="from_tracing_context",
                    aot=True,
                )
            if not isinstance(compiled_fn, AOTCompiledArtifact):
                raise AssertionError(
                    f"Expected AOTCompiledArtifact, got {type(compiled_fn)}"
                )
            # _dummy_wrapper is to make call_function happy
            compiled_submod = _dummy_wrapper(compiled_fn)
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(
                    compiled_submod, args=node.args, kwargs=node.kwargs
                )
                new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
                del gm._modules[node.target]

    gm.recompile()
    return gm


def _needs_inductor_compile(node: torch.fx.Node):
    return (
        node.op not in ("placeholder", "output")
        and hasattr(node, "meta")
        and node.meta.get("custom", None)
        and "compile_with_inductor" in node.meta["custom"]
    )


def _relocate_graphmodule_get_attrs(gm, partitions):
    """Move get_attr nodes for GraphModules to the partition containing their consumer.

    standalone_compile cannot handle GraphModule outputs, so get_attr nodes
    that reference GraphModules must be co-located with their consumer.
    This is safe because get_attr nodes have no dependencies.
    """
    node_to_partition: dict[torch.fx.Node, int] = {}
    for i, partition in enumerate(partitions):
        for node in partition:
            node_to_partition[node] = i

    nodes_to_move: list[tuple[torch.fx.Node, int, int]] = []
    for i, partition in enumerate(partitions):
        for node in list(partition):
            if node.op != "get_attr":
                continue
            if not isinstance(getattr(gm, node.target, None), torch.fx.GraphModule):
                continue
            for user in node.users:
                consumer_idx = node_to_partition.get(user)
                if consumer_idx is not None and consumer_idx != i:
                    nodes_to_move.append((node, i, consumer_idx))
                    break

    for node, from_idx, to_idx in nodes_to_move:
        del partitions[from_idx][node]
        partitions[to_idx][node] = None

    return [p for p in partitions if p]


class _RegionScooper:
    """
    Scoops out the inductor marked regions. It does NOT compile them.
    """

    @staticmethod
    def scoop_regions(gm):
        from torch.fx.passes.utils.fuser_utils import fuse_by_partitions

        # Collect contiguous runs of marked nodes (no horizontal fusion).
        # Each partition maps nodes to None (no partition-id needed).
        partitions: list[dict[torch.fx.Node, int | None]] = []
        current_run: dict[torch.fx.Node, int | None] = {}
        for node in gm.graph.nodes:
            if _needs_inductor_compile(node):
                current_run[node] = None
            else:
                if current_run:
                    partitions.append(current_run)
                    current_run = {}
        if current_run:
            partitions.append(current_run)

        if not partitions:
            logger.info("No inductor marked nodes found")
            return gm

        partitions = _relocate_graphmodule_get_attrs(gm, partitions)

        if not partitions:
            logger.info("No inductor marked nodes found after relocation")
            return gm

        return fuse_by_partitions(
            gm,
            partitions,
            prefix="__marked_inductor_submod",
            always_return_tuple=True,
        )

    @staticmethod
    def recursively_scoop_regions(gm, _processed=None):
        if _processed is None:
            _processed = set()
        for node in gm.graph.find_nodes(op="get_attr"):
            if _needs_inductor_compile(node):
                # If the get_attr itself is marked for compile, the outer graph will
                # take care of it. If we dont do that, we end up with nested
                # regional inductor compiles that do not work well.
                continue
            submod = getattr(gm, node.target)
            # Track by id: multiple get_attr nodes may reference the same GraphModule
            if (
                isinstance(submod, torch.fx.GraphModule)
                and id(submod) not in _processed
            ):
                _processed.add(id(submod))
                _RegionScooper.recursively_scoop_regions(submod, _processed)

        return _RegionScooper.scoop_regions(gm)

    def __call__(self, gm):
        with torch.fx.traceback.preserve_node_meta(enable=False):
            return _RegionScooper.recursively_scoop_regions(gm)


class _RegionCompiler:
    """
    Compiles the scooped out regions.
    """

    @staticmethod
    def compile_region(gm):
        from torch.fx.graph import _BoxedCodeGen

        gm = _compile_submod(gm, "__marked_inductor_submod")
        gm.graph.set_codegen(_BoxedCodeGen())
        gm.recompile()
        return gm

    @staticmethod
    def recursively_compile_regions(gm):
        # Find if the graph module has a scooped out region
        found_region = False
        for node in gm.graph.find_nodes(op="call_module"):
            submod = getattr(gm, node.target)
            if isinstance(submod, torch.fx.GraphModule):
                if node.target.startswith("__marked_inductor_submod"):
                    found_region = True

        # Recurse through the subgraphs
        for node in gm.graph.find_nodes(op="get_attr"):
            submod = getattr(gm, node.target)
            if isinstance(submod, torch.fx.GraphModule):
                _RegionCompiler.recursively_compile_regions(submod)

        if found_region:
            return _RegionCompiler.compile_region(gm)
        return gm

    def __call__(self, gm):
        with torch.fx.traceback.preserve_node_meta(enable=False):
            return _RegionCompiler.recursively_compile_regions(gm)


def _create_inductor_marked_regions(gm):
    with torch.fx.traceback.preserve_node_meta(enable=False):
        return _RegionScooper()(gm)


def _compile_inductor_marked_regions(gm):
    with torch.fx.traceback.preserve_node_meta(enable=False):
        return _RegionCompiler()(gm)


@compatibility(is_backward_compatible=False)
def regional_inductor(gm, *example_args):
    """
    Scoops out inductor marked regions and compiles them with inductor.

    Inductor options should be provided via the annotation API:
    with fx_traceback.annotate({
        "compile_with_inductor": {
            "inductor_configs": {
                "max_autotune": True,
                "triton.cudagraphs": False
            }
        }
    }):
    """

    # fuser utils create new nodes using create_proxy which retains the seq_nr
    # metadata and cause issues

    with torch.fx.traceback.preserve_node_meta(enable=False):
        gm = _create_inductor_marked_regions(gm)
        gm = _compile_inductor_marked_regions(gm)
        if torch._functorch.config.force_autograd_cache:
            from torch._inductor.output_code import RegionalOutputCode

            gm = RegionalOutputCode(gm)
        return gm
