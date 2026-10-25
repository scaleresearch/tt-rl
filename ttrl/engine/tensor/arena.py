"""Fixed-address device buffers and the in-place copy idiom trace capture needs.

Address stability is an engine invariant that fails silently when violated: a
captured trace replays recorded commands against fixed device buffer
addresses, so nothing a trace touches may ever be rebound to a fresh tensor.
"""

from __future__ import annotations

from typing import Iterable

import ttnn

from ttrl.engine.tensor.transfer import restore_in_place


def _raw(obj) -> "ttnn.Tensor":
    return obj if isinstance(obj, ttnn.Tensor) else obj.data


def copy_in_place(dst: "ttnn.Tensor", src: "ttnn.Tensor") -> "ttnn.Tensor":
    """Write src's contents into dst's existing buffer.

    Address stability: ttnn.add(src, 0.0, output_tensor=dst) writes into dst's
    buffer instead of allocating a new tensor for a name to be rebound to.
    Returns the same object, at the same device address.
    """
    # ttnn.is_device_tensor is still unbound in 0.80; this is the bound
    # spelling of the same question.
    if not ttnn.is_tensor_storage_on_device(dst):
        raise ValueError("copy_in_place destination must be a device tensor")
    ttnn.add(src, 0.0, output_tensor=dst)
    return dst


def copy_tensor_in_place(dst, src):
    """copy_in_place for autograd Tensor wrappers, used by target-network sync
    and checkpoint restore in place of `dst.data = ttnn.clone(src.data)`."""
    # Address stability: the write goes into dst.data's existing buffer and
    # dst.data is never rebound, so a captured trace keeps replaying against
    # the address it recorded.
    copy_in_place(dst.data, _raw(src))
    return dst


class StaticArena:
    """A named set of fixed-address device buffers, allocated once.

    Once a name is allocated, the ttnn.Tensor object behind it never changes
    identity; only its contents do. Everything a captured trace touches lives
    here.
    """

    def __init__(self, device):
        self.device = device
        self._buffers: dict[str, "ttnn.Tensor"] = {}

    def allocate(self, name: str, like) -> "ttnn.Tensor":
        """Allocate a zeroed buffer shaped and dtyped like `like`, once.

        A second allocate under the same name raises: silently reallocating is
        the address instability this class exists to prevent.
        """
        if name in self._buffers:
            raise ValueError(
                f"arena buffer {name!r} is already allocated; allocate() is "
                "one-shot. Use copy_into() to update its contents in place."
            )
        buf = ttnn.zeros_like(_raw(like))
        self._buffers[name] = buf
        return buf

    def __getitem__(self, name: str) -> "ttnn.Tensor":
        return self._buffers[name]

    def __contains__(self, name: str) -> bool:
        return name in self._buffers

    def names(self) -> Iterable[str]:
        return self._buffers.keys()

    def copy_into(self, name: str, src) -> "ttnn.Tensor":
        return copy_in_place(self._buffers[name], _raw(src))

    def restore_into_from_numpy(self, name: str, array) -> "ttnn.Tensor":
        """Checkpoint restore: upload a host array into buffer `name` in place."""
        return restore_in_place(self._buffers[name], array)
