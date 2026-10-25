"""What the two shared normalisers compute, and that the sharing is real."""

from __future__ import annotations

import numpy as np
import pytest

from tests.envs.test_policy_state import Ramp
from ttrl.envs import SyncVectorEnv
from ttrl.envs.base import FINAL_OBS_KEY
from ttrl.envs.vec_normalize import NormalizeObservation
from ttrl.envs.vector_async import AsyncVectorEnv
from ttrl.envs.wrappers import RecordEpisodeStatistics, TimeLimit
from ttrl.spaces import DEFAULT_LEAF


def make_lane():
    return TimeLimit(RecordEpisodeStatistics(Ramp()), 20)


def _drive(vec, steps=4):
    vec.reset(seed=0)
    for _ in range(steps):
        vec.step(np.zeros(vec.num_envs, dtype=np.int64))
    return vec


def sync(lanes):
    return SyncVectorEnv([make_lane] * lanes)


def asynchronous(lanes):
    # "fork" for the reason `tests/envs/test_vector_async.py` gives: nothing
    # here opens a device, so a forked child inherits no device handle.
    return AsyncVectorEnv([make_lane] * lanes, context="fork")


@pytest.mark.parametrize("vector", [sync, asynchronous])
def test_the_statistics_are_one_estimator_over_every_lane(vector):
    """THE defect, on both vector implementations."""
    lanes, steps = 4, 4
    with NormalizeObservation(vector(lanes)) as vec:
        _drive(vec, steps)
        (name, items), = vec.save_policy_state()
    assert name == "NormalizeObservation"
    assert items["count"] == lanes * (steps + 1)
    # Fitted on the batch, so the statistics do not describe a lane and a
    # restore into a narrower eval stack is the same transform.
    with NormalizeObservation(sync(1)) as fresh:
        fresh.load_policy_state([(name, items)])
        assert fresh.save_policy_state()[0][1]["count"] == items["count"]


def test_the_final_observation_is_whitened_like_the_one_beside_it():
    """A replay buffer stores `final_obs` as the transition's next_obs, so a
    raw one would be the single observation in the buffer on a scale the
    network never sees."""
    with NormalizeObservation(SyncVectorEnv([make_lane] * 2)) as vec:
        vec.reset(seed=0)
        for _ in range(5):
            obs, _r, term, trunc, info = vec.step(np.zeros(2, dtype=np.int64))
            for lane in np.flatnonzero(np.asarray(term) | np.asarray(trunc)):
                final = info[FINAL_OBS_KEY][int(lane)]
                assert final is not None
                np.testing.assert_array_equal(
                    final[DEFAULT_LEAF], obs[DEFAULT_LEAF][int(lane)])
