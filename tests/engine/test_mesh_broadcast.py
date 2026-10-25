"""`ParamSync.broadcast`."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.engine.mesh import ParamSync

from tests._fakes import FakeModule
from tests._fakes.mesh import exact_sum_reduce, patch_upload as _patch_upload


def test_broadcast_preserves_the_init_scale_that_average_would_shrink(monkeypatch):
    """The reason `_build_multi_chip` broadcasts instead of averaging."""
    rng = np.random.default_rng(0)
    n = 4
    inits = [rng.normal(0.0, 1.0, size=4096).astype(np.float32) for _ in range(n)]

    _patch_upload(monkeypatch)
    averaged_mods = [FakeModule([v]) for v in inits]
    ParamSync(modules=averaged_mods, reduce_fn=exact_sum_reduce).average()
    averaged_std = float(np.std(averaged_mods[0].parameters()[0].numpy()))
    # sqrt(4) == 2: averaging four inits halves the weight scale.
    assert averaged_std == pytest.approx(1.0 / np.sqrt(n), rel=0.1)

    broadcast_mods = [FakeModule([v]) for v in inits]
    ParamSync(modules=broadcast_mods, reduce_fn=exact_sum_reduce).broadcast()
    for m in broadcast_mods:
        # Not "approximately the right scale" -- the ACTUAL replica-0 init.
        np.testing.assert_array_equal(m.parameters()[0].numpy(), inits[0])

def test_broadcast_refreshes_targets_from_the_broadcast_online_net(monkeypatch):
    _patch_upload(monkeypatch)
    online0 = FakeModule([[1.0]])
    online1 = FakeModule([[3.0]])
    target0 = FakeModule([[999.0]])
    target1 = FakeModule([[999.0]])
    sync = ParamSync(modules=[online0, online1], reduce_fn=exact_sum_reduce)
    sync.broadcast(targets=[target0, target1])
    for t in (target0, target1):
        assert t.parameters()[0].numpy()[0] == 1.0
        assert t.copy_from_calls == 1

def test_broadcast_validates_targets_and_src_before_writing(monkeypatch):
    _patch_upload(monkeypatch)
    m0 = FakeModule([[1.0]])
    m1 = FakeModule([[3.0]])
    sync = ParamSync(modules=[m0, m1], reduce_fn=exact_sum_reduce)
    with pytest.raises(ValueError, match="expected 2 target"):
        sync.broadcast(targets=[FakeModule([[0.0]])])
    with pytest.raises(ValueError, match="src replica 2 out of range"):
        sync.broadcast(src=2)
    # All-or-nothing: nothing was written by either rejected call.
    assert m1.parameters()[0].numpy()[0] == 3.0

