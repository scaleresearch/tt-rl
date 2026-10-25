"""The open-by-discovery path's invariants, without opening a device."""

from __future__ import annotations

import pytest

from ttrl.engine.mesh import devices
from ttrl.engine.mesh.open import (
    ParamSyncError,
    _claim,
    _release,
    chips_of_open_mesh,
    replica_axis,
)


class FakeMesh:
    def __init__(self, shape):
        self.shape = shape


@pytest.fixture(autouse=True)
def _no_mesh_claimed():
    _release()
    yield
    _release()


def test_replica_axis_is_read_off_the_mesh_either_way_round():
    # The measured P300 discovery is [2, 1]; the installed p300 descriptor
    # says device_topology dims [1, 2]. A hardcoded axis is wrong under one.
    assert replica_axis(FakeMesh([2, 1])) == 0
    assert replica_axis(FakeMesh([1, 2])) == 1
    assert replica_axis(FakeMesh([1, 4])) == 1
    assert replica_axis(FakeMesh([1, 1])) == 0


def test_replica_axis_refuses_a_mesh_that_is_not_a_line():
    with pytest.raises(ParamSyncError, match="not a line"):
        replica_axis(FakeMesh([2, 2]))


def test_a_second_mesh_is_refused_and_the_first_is_named():
    _claim([2, 3])
    assert chips_of_open_mesh() == [2, 3]
    with pytest.raises(ParamSyncError) as excinfo:
        _claim([0, 1])
    message = str(excinfo.value)
    assert "[2, 3]" in message
    assert "[0, 1]" in message
    assert "scope_to_chips" in message


def test_independent_devices_are_refused_while_a_mesh_is_open():
    """The coexistence failure is a timed-out ethernet retrain and a degraded link."""

    _claim([2, 3])
    with pytest.raises(ParamSyncError, match="cannot also open independent devices"):
        with devices.open_independent_devices_for_chips([2, 3]):
            pass
