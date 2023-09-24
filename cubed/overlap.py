from typing import Tuple

import numpy as np

from cubed.core.ops import map_direct
from cubed.types import T_RectangularChunks
from cubed.utils import _cumsum
from cubed.vendor.dask.array.core import normalize_chunks
from cubed.vendor.dask.array.overlap import coerce_boundary, coerce_depth


def map_overlap(
    func,
    *args,
    dtype=None,
    chunks=None,
    depth=None,
    boundary=None,
    trim=False,
    **kwargs,
):
    if trim:
        raise ValueError("trim is not supported")

    chunks = normalize_chunks(chunks, dtype=dtype)
    shape = tuple(map(sum, chunks))

    # Coerce depth and boundary arguments to lists of individual
    # specifications for each array argument
    def coerce(xs, arg, fn):
        if not isinstance(arg, list):
            arg = [arg] * len(xs)
        return [fn(x.ndim, a) for x, a in zip(xs, arg)]

    depth = coerce(args, depth, coerce_depth)
    boundary = coerce(args, boundary, coerce_boundary)

    extra_projected_mem = 0  # TODO

    return map_direct(
        _overlap,
        *args,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        extra_projected_mem=extra_projected_mem,
        overlap_func=func,
        depth=depth,
        boundary=boundary,
        **kwargs,
    )


def _overlap(x, *arrays, overlap_func=None, depth=None, boundary=None, block_id=None):
    a = arrays[0]  # TODO: support multiple
    depth = depth[0]
    boundary = boundary[0]

    # First read the chunk with overlaps determined by depth, then pad boundaries second.
    # Do it this way round so we can do everything with one blockwise. The alternative,
    # which pads the entire array first (via concatenate), would result in at least one extra copy.
    out = a.zarray[get_item_with_depth(a.chunks, block_id, depth)]
    out = _pad_boundaries(out, depth, boundary, a.numblocks, block_id)
    return overlap_func(out)


def _clamp(minimum: int, x: int, maximum: int) -> int:
    return max(minimum, min(x, maximum))


def get_item_with_depth(
    chunks: T_RectangularChunks, idx: Tuple[int, ...], depth
) -> Tuple[slice, ...]:
    """Convert a chunk index to a tuple of slices with depth offsets."""
    # could use Dask's cached_cumsum here if it improves performance
    starts = tuple(_cumsum(c, initial_zero=True) for c in chunks)
    loc = tuple(
        (
            _clamp(0, start[i] - depth[ax], start[-1]),
            _clamp(0, start[i + 1] + depth[ax], start[-1]),
        )
        for ax, (i, start) in enumerate(zip(idx, starts))
    )
    return tuple(slice(*s, None) for s in loc)


def _pad_boundaries(x, depth, boundary, numblocks, block_id):
    for i in range(x.ndim):
        d = depth.get(i, 0)
        if d == 0 or block_id[i] not in (0, numblocks[i] - 1):
            continue
        pad_shape = list(x.shape)
        pad_shape[i] = d
        pad_shape = tuple(pad_shape)
        p = np.full_like(x, fill_value=boundary[i], shape=pad_shape)
        if block_id[i] == 0:  # first block on axis i
            x = np.concatenate([p, x], axis=i)
        elif block_id[i] == numblocks[i] - 1:  # last block on axis i
            x = np.concatenate([x, p], axis=i)
    return x
