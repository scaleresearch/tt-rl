"""Uploading fresh data into bound input buffers without rebinding them."""

from __future__ import annotations

import numpy as np
import ttnn

__all__ = ["InputBinder"]

# Below this size the three extra device dispatches cost more than the host
# tilize they replace, so small inputs (rewards, dones, one-hots) go the plain
# host route.
_DEVICE_TILIZE_MIN_BYTES = 1 << 20


class InputBinder:
    """Owns the staging buffers behind `load`.

    One per TrainStep. The bound tensors themselves belong to the caller;
    this only ever writes into their existing buffers.
    """

    def __init__(self, device, inputs: dict):
        self.device = device
        self.inputs = inputs
        self._staging: dict[str, "ttnn.Tensor"] = {}

    def load(self, name: str, array) -> None:
        """Copy new data into a bound input tensor's existing device buffer.

        Address stability: the destination address is never touched, because a
        captured trace already references it. Any real failure raises, rather
        than being papered over by allocating a fresh tensor.
        """
        t = self.inputs[name]
        arr = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
        if self._device_tilize(name, t, arr):
            return
        host = ttnn.Tensor(arr, t.dtype).to(t.data.get_layout())
        ttnn.copy_host_to_device_tensor(host, t.data)

    def _device_tilize(self, name: str, t, arr: np.ndarray) -> bool:
        """Upload row-major into a reused staging buffer and tilize on device.

        The host tilize of a large observation buffer is mostly padding: a
        logically [1, 1, N*84*84, 4] tensor has its last dim padded 4 -> 32 by
        a 32x32 tile, so the host builds and DMAs an 8x inflated buffer with a
        single-threaded layout conversion. Returns False when the input is
        small enough that the host route is cheaper.
        """
        if arr.nbytes < _DEVICE_TILIZE_MIN_BYTES:
            return False

        # A row-major device tensor is paged by its last dimension, so staging
        # in the logical [..., 4] shape gives 16-byte DMA pages, measured at
        # ~0.65 GB/s, slower than the padded transfer this replaces. The bytes
        # are contiguous either way, so stage under a wide-row shape (~7x the
        # DMA rate) and restore the logical shape with a device reshape, which
        # is metadata only in row-major layout.
        shape = tuple(int(d) for d in arr.shape)
        stage_shape = _staging_shape(shape)
        staging = self._staging.get(name)
        if staging is None or tuple(staging.shape) != stage_shape:
            staging = (
                ttnn.Tensor(np.zeros(stage_shape, dtype=np.float32), t.dtype)
                .to(ttnn.ROW_MAJOR_LAYOUT)
                .to(self.device)
            )
            self._staging[name] = staging

        staged = arr.reshape(stage_shape)
        host = ttnn.Tensor(staged, t.dtype).to(ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, staging)
        reshaped = staging if stage_shape == shape else ttnn.reshape(staging, shape)
        src = reshaped
        if t.data.get_layout() == ttnn.TILE_LAYOUT:
            src = ttnn.to_layout(reshaped, ttnn.TILE_LAYOUT)
        # Address stability: ttnn.reshape and ttnn.to_layout both return their
        # own buffers, so the result is copied in place into the bound one.
        ttnn.copy(src, t.data)
        if src is not reshaped:
            ttnn.deallocate(src)
        if reshaped is not staging:
            ttnn.deallocate(reshaped)
        return True

    def close(self) -> None:
        for buf in self._staging.values():
            ttnn.deallocate(buf)
        self._staging.clear()


def _staging_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """A [1, 1, rows, width] view of the same contiguous elements with the
    widest row, and so the largest DMA page, that divides the count evenly."""
    numel = 1
    for d in shape:
        numel *= d
    for width in (512, 256, 128, 64, 32):
        if numel % width == 0:
            return (1, 1, numel // width, width)
    return shape
