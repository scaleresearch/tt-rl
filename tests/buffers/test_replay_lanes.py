"""Lane isolation: no sampled transition may span two environment lanes."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.buffers import ReplayBuffer
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec

OBS = ObsGroup.of(ObsSpec(shape=(2,), dtype="float32", kind="vector"))
ACT = ActionSpec.discrete(8)

N_LANES = 6
GAMMA = 0.9
N_STEP = 3
#: sum_k gamma^k, the coefficient on a lane's stamp in a full n-step window
_WINDOW = sum(GAMMA ** k for k in range(N_STEP))


def _filled_buffer(n_writes=200, capacity=600):
    """Lane `e` writes observations and rewards stamped with `e`, so any
    mixture between lanes is visible in the sampled values."""
    buf = ReplayBuffer(OBS, ACT, capacity=capacity, n_envs=N_LANES, n_step=N_STEP,
                       gamma=GAMMA, seed=3)
    for t in range(n_writes):
        for e in range(N_LANES):
            buf.add(np.array([e, t], dtype=np.float32), e, float(e * 1000 + t),
                    False, False, lane=e)
    return buf


@pytest.mark.parametrize("n_writes, capacity", [(200, 600), (400, 360)])
def test_sampled_transitions_never_mix_lanes(n_writes, capacity):
    """The second case wraps every lane at least twice."""
    buf = _filled_buffer(n_writes, capacity)
    batch = buf.sample(20_000)
    lanes = batch.lanes

    # Every component of the transition carries its own lane's stamp.
    assert np.array_equal(batch.obs["obs"][:, 0], lanes.astype(np.float32))
    assert np.array_equal(batch.next_obs["obs"][:, 0], lanes.astype(np.float32))
    assert np.array_equal(batch.actions, lanes)

    # The successor is the NEXT step of the same lane, so its time stamp is
    # exactly n_step past the state's. This is the assertion the neighbour
    # derivation has to earn: reading the adjacent ring slot rather than the
    # adjacent slot OF THIS LANE would land in another lane entirely.
    assert np.array_equal(batch.next_obs["obs"][:, 1], batch.obs["obs"][:, 1] + N_STEP)

    # The reward is a discounted sum of three consecutive steps of ONE lane,
    # so dividing out the window coefficient recovers the lane it belongs to.
    lane_from_reward = np.round(batch.rewards / (1000 * _WINDOW))
    assert np.array_equal(lane_from_reward, lanes.astype(np.float64))


def test_add_batch_keeps_lanes_separate():
    """The vector-env entry point, which writes every lane in one call."""
    buf = ReplayBuffer(OBS, ACT, capacity=600, n_envs=N_LANES, n_step=N_STEP,
                       gamma=GAMMA, seed=5)
    for t in range(100):
        obs = np.stack([np.array([e, t], np.float32) for e in range(N_LANES)])
        buf.add_batch(obs, np.arange(N_LANES), np.arange(N_LANES) * 1000.0 + t,
                      np.zeros(N_LANES, bool))
    batch = buf.sample(8000)
    assert np.array_equal(batch.obs["obs"][:, 0], batch.lanes.astype(np.float32))
    assert np.array_equal(batch.next_obs["obs"][:, 1], batch.obs["obs"][:, 1] + N_STEP)

