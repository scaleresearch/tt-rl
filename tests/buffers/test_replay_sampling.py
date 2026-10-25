"""Which transitions are sampleable, and the write-cursor seam."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.buffers import ReplayBuffer
from ttrl.buffers.indexing import sampleable
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec

OBS = ObsGroup.of(ObsSpec(shape=(1,), dtype="float32", kind="vector"))
ACT = ActionSpec.discrete(4)


def _obs(t):
    return np.full((1,), float(t), np.float32)


def _decode(a):
    return int(np.asarray(a).reshape(-1)[0])


def _drive(buf, n_writes, lane=0):
    for t in range(n_writes):
        buf.add(_obs(t), 0, float(t), False, False, lane=lane)


# ---------------------------------------------------------------------------
# the seam, by hand
# ---------------------------------------------------------------------------

def test_hand_computed_sampleable_set_at_the_wrap():
    """capacity 16, n_step 3, 26 writes, no episode boundaries."""
    buf = ReplayBuffer(OBS, ACT, capacity=16, n_step=3, gamma=0.5, seed=0)
    _drive(buf, 26)
    assert buf._full[0] and buf._pos[0] == 10

    start, count = sampleable(buf._pos, buf._full, 16, 3)
    assert (int(start[0]), int(count[0])) == (10, 13)
    assert buf.num_sampleable == 13

    batch = buf.sample(6000)
    assert sorted({_decode(o) for o in batch.obs["obs"]}) == list(range(10, 23))

    # The newest sampleable transition reads its successor from write 25, the
    # most recent write, and NOT from the column write 25 would be followed by.
    newest = batch.obs["obs"][:, 0] == 22
    assert newest.any()
    assert np.all(batch.next_obs["obs"][newest, 0] == 25.0)
    assert np.all(batch.rewards[newest] == np.float32(22 + 0.5 * 23 + 0.25 * 24))


def test_successor_is_always_exactly_n_step_ahead_without_boundaries():
    """With no episode boundaries the successor is state + n_step, always."""
    for capacity in (8, 9, 13, 16, 31):
        for n_step in (1, 2, 3, 4):
            if capacity <= n_step + 1:
                continue
            for n_writes in range(1, 3 * capacity):
                buf = ReplayBuffer(OBS, ACT, capacity=capacity, n_step=n_step,
                                   gamma=0.9, seed=0)
                _drive(buf, n_writes)
                if buf.num_sampleable == 0:
                    with pytest.raises(ValueError):
                        buf.sample(4)
                    continue
                batch = buf.sample(400)
                assert np.array_equal(batch.next_obs["obs"][:, 0],
                                      batch.obs["obs"][:, 0] + n_step)


# ---------------------------------------------------------------------------
# partial fill
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_step,capacity,writes,expect", [
    (1, 10, 1, 0), (1, 10, 2, 1), (3, 10, 4, 1), (3, 10, 57, 7),
])
def test_num_sampleable_while_filling_and_after_wrapping(n_step, capacity, writes, expect):
    """A transition needs n_step + 1 live columns, so a lane of `capacity`
    offers `min(writes, capacity) - n_step` of them."""
    buf = ReplayBuffer(OBS, ACT, capacity=capacity, n_step=n_step, gamma=0.9, seed=0)
    _drive(buf, writes)
    assert buf.num_sampleable == expect
    assert buf.num_sampleable == max(min(writes, capacity) - n_step, 0)


def test_empty_and_too_young_buffers_raise_rather_than_fabricate():
    buf = ReplayBuffer(OBS, ACT, capacity=10, n_step=3, gamma=0.9, seed=0)
    with pytest.raises(ValueError, match="no lane holds a complete"):
        buf.sample(1)
    _drive(buf, 3)
    assert buf.num_sampleable == 0
    with pytest.raises(ValueError, match="no lane holds a complete"):
        buf.sample(1)
    _drive(buf, 1)
    assert buf.num_sampleable == 1
    assert len(buf.sample(5)) == 5


# ---------------------------------------------------------------------------
# the sparse boundary table
# ---------------------------------------------------------------------------

def test_boundary_table_holds_only_live_boundaries():
    """It is evicted when its column is overwritten, so it cannot grow without
    bound and cannot serve a previous lap's observation as a live successor."""
    buf = ReplayBuffer(OBS, ACT, capacity=8, n_step=1, gamma=0.9, seed=0)
    for t in range(8):
        ends = t % 2 == 1
        buf.add(_obs(t), 0, 1.0, ends, False,
                final_obs=_obs(900 + t) if ends else None)
    assert buf.num_boundaries == 4
    # A second lap of non-ending writes evicts every one of them.
    for t in range(8, 16):
        buf.add(_obs(t), 0, 1.0, False, False)
    assert buf.num_boundaries == 0
    batch = buf.sample(500)
    assert np.array_equal(batch.next_obs["obs"][:, 0], batch.obs["obs"][:, 0] + 1)


def test_boundary_table_is_per_lane():
    buf = ReplayBuffer(OBS, ACT, capacity=40, n_envs=2, n_step=1, gamma=0.9, seed=0)
    for t in range(6):
        # Lane 0 ends on step 2; lane 1 never ends.
        buf.add_batch(
            np.stack([_obs(t), _obs(100 + t)]),
            np.zeros(2, np.int64), np.ones(2, np.float32),
            np.array([t == 2, False]),
            final_obs=[_obs(777) if t == 2 else None, None],
        )
    assert buf.num_boundaries == 1
    batch = buf.sample(2000)
    lane0 = batch.lanes == 0
    # Lane 0's transition at state 2 takes the recorded final observation;
    # lane 1 is untouched by it and still reads its own neighbour.
    at_two = lane0 & (batch.obs["obs"][:, 0] == 2.0)
    assert at_two.any()
    assert np.all(batch.next_obs["obs"][at_two, 0] == 777.0)
    lane1 = batch.lanes == 1
    assert np.all(batch.next_obs["obs"][lane1, 0] == batch.obs["obs"][lane1, 0] + 1)

