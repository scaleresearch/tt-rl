"""`ParamSync.average`."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.engine.mesh import ParamSync, ParamSyncError

from tests._fakes import FakeModule, FakeTensor
from tests._fakes.mesh import exact_sum_reduce, patch_upload as _patch_upload


def test_param_order_mismatch_count_raises():
    m1 = FakeModule([[1.0, 2.0], [3.0]])
    m2 = FakeModule([[1.0, 2.0]])
    with pytest.raises(ParamSyncError, match="replica parameter schemas differ"):
        ParamSync(modules=[m1, m2], reduce_fn=exact_sum_reduce)


def test_average_zips_parameters_positionally(monkeypatch):
    _patch_upload(monkeypatch)
    m1 = FakeModule([[1.0, 0.0], [10.0]])
    m2 = FakeModule([[3.0, 0.0], [20.0]])
    sync = ParamSync(modules=[m1, m2], reduce_fn=exact_sum_reduce)
    sync.average()
    # param 0 averages (1,3)->2 per replica; param 1 averages (10,20)->15.
    np.testing.assert_allclose(m1.parameters()[0].numpy(), [2.0, 0.0])
    np.testing.assert_allclose(m2.parameters()[0].numpy(), [2.0, 0.0])
    np.testing.assert_allclose(m1.parameters()[1].numpy(), [15.0])


def test_target_refreshed_from_averaged_not_pre_average(monkeypatch):
    _patch_upload(monkeypatch)
    online1 = FakeModule([[1.0]])
    online2 = FakeModule([[3.0]])
    target1 = FakeModule([[999.0]])  # stale pre-average target
    target2 = FakeModule([[999.0]])
    sync = ParamSync(modules=[online1, online2], reduce_fn=exact_sum_reduce)
    sync.average(targets=[target1, target2])
    averaged = 2.0  # (1+3)/2
    assert target1.parameters()[0].numpy()[0] == averaged
    assert target2.parameters()[0].numpy()[0] == averaged
    assert target1.copy_from_calls == 1


def test_short_reduce_result_raises_instead_of_half_averaging(monkeypatch):
    """A reducer that returns fewer results than there are replicas would."""
    _patch_upload(monkeypatch)

    def short(arrays):
        return exact_sum_reduce(arrays)[:-1]

    m1 = FakeModule([[1.0]])
    m2 = FakeModule([[3.0]])
    m3 = FakeModule([[5.0]])
    sync = ParamSync(modules=[m1, m2, m3], reduce_fn=short)
    with pytest.raises(ParamSyncError, match="2 result.* for 3 replica"):
        sync.average()
    for m, original in ((m1, 1.0), (m2, 3.0), (m3, 5.0)):
        assert m.parameters()[0].numpy()[0] == original


def test_schema_change_after_construction_is_caught_before_any_write(monkeypatch):
    """`__post_init__` validates the schema once."""
    _patch_upload(monkeypatch)
    m1 = FakeModule([[1.0], [10.0]])
    m2 = FakeModule([[3.0], [20.0]])
    sync = ParamSync(modules=[m1, m2], reduce_fn=exact_sum_reduce)

    # Every replica grows the same head, so the replicas still agree with each
    # other -- only the layout `ParamSync` was built against is now stale.
    for m in (m1, m2):
        m._params.append(FakeTensor([99.0]))
        m._names.append("head")
    with pytest.raises(ParamSyncError, match="schema changed since ParamSync was built"):
        sync.average()
    # All-or-nothing: not even the first, still-matching parameter moved.
    assert m1.parameters()[0].numpy()[0] == 1.0
    assert m2.parameters()[0].numpy()[0] == 3.0


def test_same_shaped_parameters_swapped_between_replicas_are_rejected(monkeypatch):
    """Replica 1 built ``w0, w1``; replica 2 built ``w1, w0``."""
    _patch_upload(monkeypatch)
    m1 = FakeModule([[1.0], [10.0]], names=["w0", "w1"])
    m2 = FakeModule([[3.0], [20.0]], names=["w1", "w0"])
    with pytest.raises(ParamSyncError, match="replica parameter schemas differ"):
        ParamSync(modules=[m1, m2], reduce_fn=exact_sum_reduce)


def test_targets_are_not_refreshed_when_the_average_raises(monkeypatch):
    """A failed sync must leave the target networks stale rather than refreshing them from."""
    _patch_upload(monkeypatch)

    def boom(_arrays):
        raise RuntimeError("simulated fabric link failure")

    online = [FakeModule([[1.0]]), FakeModule([[3.0]])]
    targets = [FakeModule([[999.0]]), FakeModule([[999.0]])]
    sync = ParamSync(modules=online, reduce_fn=boom)
    with pytest.raises(ParamSyncError, match="the reduction failed on parameter"):
        sync.average(targets=targets)
    for t in targets:
        assert t.copy_from_calls == 0
        assert t.parameters()[0].numpy()[0] == 999.0


def test_maybe_average_fires_every_sync_every_calls(monkeypatch):
    _patch_upload(monkeypatch)
    m1 = FakeModule([[1.0]])
    m2 = FakeModule([[3.0]])
    sync = ParamSync(modules=[m1, m2], sync_every=3, reduce_fn=exact_sum_reduce)
    assert sync.maybe_average() is False
    assert sync.maybe_average() is False
    assert sync.maybe_average() is True
    assert m1.parameters()[0].numpy()[0] == 2.0

