"""Host-only stand-ins for the pieces of `ttrl.engine.mesh` that need ttnn.

`ParamSync` is almost entirely plain Python sitting behind one seam, its
`reduce_fn`, so the averaging semantics are testable with an exact numpy sum
and no hardware at all. `restore_in_place` is the other seam: it is the one
ttnn primitive the write path uses, and patching it lets the whole average
and broadcast logic run over plain numpy buffers.
"""

from __future__ import annotations

import numpy as np

import ttrl.engine.mesh.sync as sync_mod

__all__ = ["patch_upload", "exact_sum_reduce"]


def patch_upload(monkeypatch):
    """Point `ParamSync`'s and `resume_replicas`' writes at `FakeTensor`.

    `restore_in_place` is a device copy. Replacing it keeps the address
    stability contract legible, since the fake writes through the existing
    `_array` rather than rebinding it.
    """

    def fake_restore(data, array):
        data._array = np.array(array, dtype=np.float32).reshape(data.shape)

    monkeypatch.setattr(sync_mod, "restore_in_place", fake_restore)


def exact_sum_reduce(arrays):
    """Exact fp32 elementwise sum, identical on every returned entry.

    Accumulates in float64 so the sum itself contributes no rounding and the
    only rounding left is the one `ParamSync` performs when it divides.
    """
    total = np.zeros_like(arrays[0], dtype=np.float64)
    for a in arrays:
        total += a.astype(np.float64)
    return [total.astype(np.float32) for _ in arrays]
