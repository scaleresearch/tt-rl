"""Host<->device crossing interception, shared by the crossing-budget tests."""

from __future__ import annotations

import ttnn

from ttrl.engine.layers import Linear, ReLU, Sequential

__all__ = ["CrossingCounter", "make_mlp"]


class CrossingCounter:
    """Counts uploads and downloads across the four ttnn entry points while a `with` block runs."""

    def __init__(self):
        self.uploads = 0
        self.downloads = 0
        self._patches: list[tuple[object, str, object]] = []

    def _wrap(self, owner, name, counter_attr, only_device_arg=False):
        orig = getattr(owner, name)

        def patched(*a, **k):
            if not only_device_arg or isinstance(a[1], (ttnn.Device, ttnn.MeshDevice)):
                setattr(self, counter_attr, getattr(self, counter_attr) + 1)
            return orig(*a, **k)

        setattr(owner, name, patched)
        self._patches.append((owner, name, orig))

    def __enter__(self):
        self._wrap(ttnn.Tensor, "to", "uploads", only_device_arg=True)
        self._wrap(ttnn.Tensor, "cpu", "downloads")
        self._wrap(ttnn, "copy_host_to_device_tensor", "uploads")
        self._wrap(ttnn, "to_device", "uploads")
        return self

    def __exit__(self, *exc):
        for owner, name, orig in self._patches:
            setattr(owner, name, orig)
        self._patches.clear()
        return False


def make_mlp(dev):
    return Sequential(Linear(8, 16, dev), ReLU(), Linear(16, 4, dev))
