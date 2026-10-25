"""Host to device transfer. The only ttnn-to-numpy path in engine/tensor."""

from __future__ import annotations

import numpy as np
import ttnn

_DEFAULT_DTYPE = ttnn.float32


def to_device(array: np.ndarray, device, dtype=_DEFAULT_DTYPE, *, layout=None) -> "ttnn.Tensor":
    """Upload a numpy array to the chip, in tile layout by default.

    `layout` exists for buffers whose only consumer wants row-major, notably
    bound observation buffers that ttnn.conv2d immediately untilizes. Tilizing
    on upload and untilizing again in the first op is a round trip that the
    row-major upload skips.
    """
    arr = np.ascontiguousarray(np.asarray(array).astype(np.float32))
    return ttnn.Tensor(arr, dtype).to(layout or ttnn.TILE_LAYOUT).to(device)


def restore_in_place(dst: "ttnn.Tensor", array) -> "ttnn.Tensor":
    """Write a host array into `dst`'s existing buffer, keeping its address.

    The checkpoint-restore counterpart of copy_in_place, and the only
    numpy-to-ttnn path in the engine. The host tensor is forced to TILE
    layout, which copy_host_to_device_tensor requires of a tiled destination.

    The host tensor may die on return. ttnn 0.80 either copies an unpinned
    source into the command queue's hugepage before returning, or, for a
    pinned one, issues the write blocking; the device never reads the source
    afterwards.
    """
    host = ttnn.Tensor(
        np.ascontiguousarray(np.asarray(array, dtype=np.float32)), dst.dtype
    ).to(ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(host, dst)
    return dst


def to_host(tensor: "ttnn.Tensor") -> np.ndarray:
    """Download a ttnn tensor to numpy float32. Never call this inside a step:
    trace capture forbids host reads.

    bfloat16 cannot cross into numpy directly, so the typecast happens
    on-device first.
    """
    t = tensor if tensor.dtype == ttnn.float32 else ttnn.typecast(tensor, ttnn.float32)
    host = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT)
    return np.asarray(host.to_numpy()).astype(np.float32)
