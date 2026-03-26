# mypy: allow-untyped-defs
import contextlib
import math
import warnings
from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING

import torch


__all__ = [
    "set_rng_state",
    "get_rng_state",
    "manual_seed",
    "seed",
    "initial_seed",
    "fork_rng",
    "split",
    "fold_in",
    "normal",
    "uniform",
    "thread_safe_generator",
]


if TYPE_CHECKING:
    from torch.utils.data._utils.worker import WorkerInfo

from torch._C import default_generator


def key(
    seed: int, impl: str = "philox4x32-10", device: torch.device | None = None
) -> torch.Tensor:
    if impl != "philox4x32-10":
        raise NotImplementedError(
            f"torch.random.key() does not support PRNG impl '{impl}'"
        )

    # (seed, offset)
    return torch.tensor([seed, 0], dtype=torch.uint64, device=device)


def split(key: torch.Tensor, num: int = 2) -> torch.Tensor:
    return torch.ops.aten._philox_key_split(key, num)


def grid_split(
    key: torch.Tensor,
    shape: tuple,
    splits: tuple,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Split a key into a grid of keys for tiled generation.

    For 1D, each returned key covers a contiguous block of the stream::

        keys = grid_split(key, (100,), (10,))
        full = uniform(key, (100,))
        tile_size = 100 // 10  # = 10
        tiled = torch.cat([uniform(keys[i], (tile_size,)) for i in range(10)])
        assert torch.equal(full, tiled)

    For N-D, each tile key is a batched key with per-row offsets into the flat
    stream. The tile shape is ``shape[i] // splits[i]`` along each dimension,
    and ``uniform(keys[t0, ..., t_{n-1}], tile_shape)`` reproduces the
    corresponding sub-block of the full generation.

    Args:
        key: A stateless RNG key tensor of shape ``(..., 2)`` with dtype uint64.
        shape: Shape of the full tensor to be generated.
        splits: Number of keys (tiles) along each dimension. Must evenly
            divide the corresponding element of *shape*.
        dtype: The dtype that will be generated. Needed because float64
            consumes 2 Philox outputs per element vs 1 for other types.

    Returns:
        Batched key tensor. For 1D: shape ``(*splits, 2)``.
        For N-D: shape ``(*splits, *tile_shape[:-1], 2)``, where each tile key
        carries one sub-key per row of the tile.
    """
    if len(shape) != len(splits):
        raise ValueError(
            f"shape and splits must have the same length, got {len(shape)} and {len(splits)}"
        )
    for i, (s, sp) in enumerate(zip(shape, splits)):
        if s % sp != 0:
            raise ValueError(
                f"splits[{i}]={sp} does not evenly divide shape[{i}]={s}"
            )
    outputs_per_elem = 2 if dtype is not None and dtype == torch.float64 else 1
    return _philox_grid_split(key, shape, splits, outputs_per_elem)


def _philox_grid_split(
    key: torch.Tensor, shape: tuple, splits: tuple, outputs_per_elem: int
) -> torch.Tensor:
    ndim = len(shape)
    tile_shape = tuple(s // sp for s, sp in zip(shape, splits))
    data = key.view(torch.int64)
    seed = data[..., 0]
    base_offset = data[..., 1]

    if ndim == 1:
        flat_indices = torch.arange(
            splits[0], dtype=torch.int64, device=key.device
        )
        offsets = base_offset + flat_indices * (tile_shape[0] * outputs_per_elem)
        seeds = seed.expand_as(offsets)
        return torch.stack([seeds, offsets], dim=-1).view(torch.uint64)

    # N-D: tiles are not contiguous in the flat stream. Each "row" (innermost
    # slice of size tile_shape[-1]) IS contiguous, so we emit one key per row
    # within each tile. Returned shape: (*splits, *tile_shape[:-1], 2).

    # Row-major strides of the full shape (in Philox outputs).
    strides = []
    s = outputs_per_elem
    for d in reversed(shape):
        strides.append(s)
        s *= d
    strides.reverse()

    # Build range tensors for tile indices and inner-tile row indices.
    ranges = []
    for j in range(ndim - 1):
        t = torch.arange(splits[j], dtype=torch.int64, device=key.device)
        i = torch.arange(tile_shape[j], dtype=torch.int64, device=key.device)
        global_j = (t * tile_shape[j]).unsqueeze(1) + i.unsqueeze(0)
        ranges.append(global_j)
    # Last dim: just tile index * tile_shape[-1]
    t_last = (
        torch.arange(splits[-1], dtype=torch.int64, device=key.device)
        * tile_shape[-1]
    )
    ranges.append(t_last.unsqueeze(1))

    # Broadcast all ranges to compute flat offsets.
    # Layout: (splits[0], tile_shape[0], ..., splits[n-2], tile_shape[n-2], splits[n-1], 1)
    total_dims = 2 * (ndim - 1) + 2
    offset = torch.zeros(1, dtype=torch.int64, device=key.device)
    for j in range(ndim - 1):
        view_shape = [1] * total_dims
        view_shape[2 * j] = splits[j]
        view_shape[2 * j + 1] = tile_shape[j]
        offset = offset + ranges[j].reshape(view_shape) * strides[j]
    view_shape = [1] * total_dims
    view_shape[2 * (ndim - 1)] = splits[-1]
    offset = offset + ranges[-1].reshape(view_shape)

    offset = offset + base_offset
    offset = offset.squeeze(-1)
    target_shape = []
    for j in range(ndim - 1):
        target_shape.extend([splits[j], tile_shape[j]])
    target_shape.append(splits[-1])
    offset = offset.reshape(target_shape)
    # Permute: (sp0, ts0, sp1, ts1, ..., sp_{n-1}) -> (*splits, *tile_shape[:-1])
    tile_perm = list(range(0, 2 * (ndim - 1), 2))
    tile_perm.append(2 * (ndim - 1))
    inner_perm = list(range(1, 2 * (ndim - 1), 2))
    offset = offset.permute(tile_perm + inner_perm).contiguous()

    seeds = seed.expand_as(offset)
    return torch.stack([seeds, offset], dim=-1).view(torch.uint64)


def fold_in(key: torch.Tensor, data: int) -> torch.Tensor:
    return torch.ops.aten._philox_key_fold_in(key, data)


def normal(
    key: torch.Tensor,
    *shape: tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    portable: bool = True,
) -> torch.Tensor:
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = tuple(shape[0])
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = key.device
    result = torch.empty(shape, dtype=dtype, device=device)
    return torch.ops.aten._philox_normal_(result, key, mean, std, portable)


def uniform(
    key: torch.Tensor,
    *shape: tuple[int, ...],
    low: float = 0.0,
    high: float = 1.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    portable: bool = True,
) -> torch.Tensor:
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = tuple(shape[0])
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = key.device
    result = torch.empty(shape, dtype=dtype, device=device)
    return torch.ops.aten._philox_uniform_(result, key, low, high, portable)


def set_rng_state(new_state: torch.Tensor) -> None:
    r"""Sets the random number generator state.

    .. note:: This function only works for CPU. For CUDA, please use
        :func:`torch.manual_seed`, which works for both CPU and CUDA.

    Args:
        new_state (torch.ByteTensor): The desired state
    """
    default_generator.set_state(new_state)


def get_rng_state() -> torch.Tensor:
    r"""Returns the random number generator state as a `torch.ByteTensor`.

    .. note:: The returned state is for the default generator on CPU only.

    See also: :func:`torch.random.fork_rng`.
    """
    return default_generator.get_state()


def manual_seed(seed) -> torch._C.Generator:
    r"""Sets the seed for generating random numbers on all devices. Returns a
    `torch.Generator` object.

    Args:
        seed (int): The desired seed. Value must be within the inclusive range
            `[-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]`. Otherwise, a RuntimeError
            is raised. Negative inputs are remapped to positive values with the formula
            `0xffff_ffff_ffff_ffff + seed`.
    """
    return _manual_seed_impl(seed)


def _manual_seed_impl(seed) -> torch._C.Generator:
    seed = int(seed)
    import torch.cuda

    if not torch.cuda._is_in_bad_fork():
        torch.cuda.manual_seed_all(seed)

    import torch.mps

    if not torch.mps._is_in_bad_fork():
        torch.mps.manual_seed(seed)

    import torch.xpu

    if not torch.xpu._is_in_bad_fork():
        torch.xpu.manual_seed_all(seed)

    import torch.mtia

    if not torch.mtia._is_in_bad_fork():
        torch.mtia.manual_seed_all(seed)

    _seed_custom_device(seed)

    return default_generator.manual_seed(seed)


def seed() -> int:
    r"""Sets the seed for generating random numbers to a non-deterministic
    random number on all devices. Returns a 64 bit number used to seed the RNG.
    """
    seed = default_generator.seed()
    import torch.cuda

    if not torch.cuda._is_in_bad_fork():
        torch.cuda.manual_seed_all(seed)

    import torch.mps

    if not torch.mps._is_in_bad_fork():
        torch.mps.manual_seed(seed)

    import torch.xpu

    if not torch.xpu._is_in_bad_fork():
        torch.xpu.manual_seed_all(seed)

    import torch.mtia

    if not torch.mtia._is_in_bad_fork():
        torch.mtia.manual_seed_all(seed)

    _seed_custom_device(seed)

    return seed


def _seed_custom_device(seed) -> None:
    r"""Sets the seed to generate random numbers for custom device.

    Args:
        seed (int): The desired seed.

    See [Note: support the custom device with privateuse1]
    """
    seed = int(seed)
    custom_backend_name = torch._C._get_privateuse1_backend_name()
    if hasattr(torch, custom_backend_name):
        custom_device_mod = getattr(torch, custom_backend_name)
        _bad_fork_name = "_is_in_bad_fork"
        _seed_all_name = "manual_seed_all"
        if hasattr(custom_device_mod, _bad_fork_name) and hasattr(
            custom_device_mod, _seed_all_name
        ):
            if not getattr(custom_device_mod, _bad_fork_name)():
                getattr(custom_device_mod, _seed_all_name)(seed)
        else:
            message = f"Set seed for `{custom_backend_name}` device does not take effect, please add API's "
            message += f"`{_bad_fork_name}` and `{_seed_all_name}` to `{custom_backend_name}` device module."
            warnings.warn(message, UserWarning, stacklevel=3)


def initial_seed() -> int:
    r"""Returns the initial seed for generating random numbers as a
    Python `long`.

    .. note:: The returned seed is for the default generator on CPU only.
    """
    return default_generator.initial_seed()


_fork_rng_warned_already = False


@contextlib.contextmanager
def fork_rng(
    devices=None,
    enabled=True,
    _caller="fork_rng",
    _devices_kw="devices",
    device_type="cuda",
) -> Generator:
    """
    Forks the RNG, so that when you return, the RNG is reset
    to the state that it was previously in.

    Args:
        devices (iterable of Device IDs): devices for which to fork
            the RNG. CPU RNG state is always forked. By default, :meth:`fork_rng` operates
            on all devices, but will emit a warning if your machine has a lot
            of devices, since this function will run very slowly in that case.
            If you explicitly specify devices, this warning will be suppressed
        enabled (bool): if ``False``, the RNG is not forked.  This is a convenience
            argument for easily disabling the context manager without having
            to delete it and unindent your Python code under it.
        device_type (str): device type str, default is `cuda`. As for supported device,
            see details in :ref:`accelerator<accelerators>`
    """

    if device_type == "meta":
        yield
        return

    device_type = torch.device(device_type).type
    device_mod = getattr(torch, device_type, None)
    if device_mod is None:
        raise RuntimeError(
            f"torch has no module of `{device_type}`, you should register "
            + "a module by `torch._register_device_module`."
        )
    global _fork_rng_warned_already

    # Internal arguments:
    #   _caller: the function which called fork_rng, which the user used
    #   _devices_kw: the devices keyword of _caller

    if not enabled:
        yield
        return

    if devices is None:
        num_devices = device_mod.device_count()
        if num_devices > 1 and not _fork_rng_warned_already:
            message = (
                f"{device_type.upper()} reports that you have {num_devices} available devices, and "
                f"you have used {_caller} without explicitly specifying which devices are being used. "
                f"For safety, we initialize *every* {device_type.upper()} device by default, which can "
                f"be quite slow if you have a lot of {device_type.upper()}s. If you know that you are only"
                f" making use of a few {device_type.upper()} devices, set the environment variable "
                f"{device_type.upper()}_VISIBLE_DEVICES or the '{_devices_kw}' keyword argument of {_caller} "
                "with the set of devices you are actually using. For example, if you are using CPU only, "
                "set device.upper()_VISIBLE_DEVICES= or devices=[]; if you are using device 0 only, "
                f"set {device_type.upper()}_VISIBLE_DEVICES=0 or devices=[0].  To initialize all devices "
                f"and suppress this warning, set the '{_devices_kw}' keyword argument to "
                f"`range(torch.{device_type}.device_count())`."
            )
            warnings.warn(message, stacklevel=2)
            _fork_rng_warned_already = True
        devices = list(range(num_devices))
    else:
        # Protect against user passing us a generator; we need to traverse this
        # multiple times but a generator will be exhausted upon first traversal
        devices = list(devices)

    cpu_rng_state = torch.get_rng_state()
    device_rng_states = [device_mod.get_rng_state(device) for device in devices]

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        for device, device_rng_state in zip(devices, device_rng_states):
            device_mod.set_rng_state(device_rng_state, device)


def thread_safe_generator() -> torch.Generator | None:
    """Returns a thread-safe random number generator for use in DataLoader workers.
    This function provides a convenient way for transforms and user code to use
    thread-safe random number generation without manually checking worker context.
    When called in a DataLoader thread worker, returns the worker's thread-local
    :class:`torch.Generator`. When called in the main process or process workers,
    returns ``None`` (which causes PyTorch functions to use the default global RNG).
    Returns:
        Optional[torch.Generator]: Thread-local generator in thread workers, None otherwise.
    Example::
        >>> from torch.random import thread_safe_generator
        >>> generator = thread_safe_generator()
        >>> torch.randint(0, 10, (5,), generator=generator)
    Example with transforms::
        >>> from torch.random import thread_safe_generator
        >>> class MyRandomTransform:
        ...     def __call__(self, img):
        ...         generator = thread_safe_generator()
        ...         offset = torch.randint(0, 10, (2,), generator=generator)
        ...         return img[..., offset[0]:, offset[1]:]
    """
    # Lazy import to avoid circular dependency during torch module initialization
    # torch.__init__ loads torch.random early, but torch.utils.data triggers
    # torch.distributed which needs torch to be fully initialized
    from torch.utils.data import get_worker_info

    worker_info: WorkerInfo | None = get_worker_info()
    if (
        worker_info is not None
        and worker_info.worker_method == "thread"
        and worker_info.rng is not None
    ):
        return worker_info.rng.torch_generator
    return None
