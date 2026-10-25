"""`AsyncVectorEnv`: the worker-failure and shutdown paths."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.envs import FINAL_OBS_KEY, check_vec_env
from ttrl.envs.vector_async import AsyncVectorEnv
from ttrl.spaces import ActionSpec, ObsSpec

SPEC = ObsSpec.vector(2)
ACTIONS = ActionSpec.discrete(3)


class ToyEnv:
    """Ends an episode every `period` steps. Observation encodes (episode, t)."""

    def __init__(self, period=3, *, spec=SPEC, fail_at=None, fail_on_init=False):
        if fail_on_init:
            raise RuntimeError("this lane cannot be built")
        self.obs_spec = spec
        self.action_spec = ACTIONS
        self.period = period
        self.fail_at = fail_at
        self.episode = 0
        self.t = 0

    def _obs(self):
        return np.array([self.episode, self.t], dtype=np.float32)

    def reset(self, *, seed=None):
        self.episode = 0 if seed is None else seed
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        if self.fail_at is not None and self.t >= self.fail_at:
            raise RuntimeError("this lane exploded")
        done = self.t % self.period == 0
        info = {"episode": {"r": float(self.period), "l": self.period}} if done else {}
        return self._obs(), 1.0, done, False, info

    def get_state(self):
        return {"episode": self.episode, "t": self.t}

    def set_state(self, state):
        self.episode, self.t = state["episode"], state["t"]

    def close(self):
        pass


def _async(n, **kwargs):
    return AsyncVectorEnv([lambda kw=kwargs: ToyEnv(**kw) for _ in range(n)],
                          context="fork")


def test_it_honours_the_vector_contract_across_subprocesses():
    with _async(2) as env:
        check_vec_env(env, steps=12)


def test_the_terminal_step_carries_the_final_observation():
    with _async(1, period=2) as env:
        env.reset(seed=0)
        env.step([0])
        obs, _r, term, _tr, info = env.step([0])
        assert term[0]
        assert np.array_equal(info[FINAL_OBS_KEY][0]["obs"], obs["obs"][0])

        obs, reward, term, _tr, info = env.step([0])
        assert reward[0] == 0.0 and not term[0]
        assert np.array_equal(obs["obs"][0], [0, 0])


def test_one_failing_lane_does_not_leave_a_siblings_reply_unread():
    """Every lane is read before anything is raised. Returning early would
    leave a reply in a sibling's pipe, and the next step would read that one
    as its own answer: silently off by one step, forever."""
    fns = [lambda: ToyEnv(), lambda: ToyEnv(fail_at=1), lambda: ToyEnv()]
    with AsyncVectorEnv(fns, context="fork") as env:
        env.reset(seed=0)
        with pytest.raises(ExceptionGroup) as caught:
            env.step([0, 0, 0])
    lanes = str(caught.value)
    assert "lanes [1] of 3" in lanes


def test_a_lane_whose_construction_fails_is_reported_not_hung():
    fns = [lambda: ToyEnv(), lambda: ToyEnv(fail_on_init=True)]
    with pytest.raises(ExceptionGroup) as caught:
        AsyncVectorEnv(fns, context="fork")
    assert "startup failed in lanes [1] of 2" in str(caught.value)


def test_close_leaves_no_subprocess_alive():
    env = _async(2)
    procs = list(env._lanes._procs)
    env.reset(seed=0)
    env.close()
    for proc in procs:
        proc.join(5.0)
        assert not proc.is_alive(), "a live child still holds its env"


def test_close_after_a_lane_has_already_died_is_not_an_error():
    """A child that has already exited cannot be asked anything, and that is
    not a failure during shutdown. What would be a failure is leaving a
    subprocess alive."""
    env = _async(2, fail_at=1)
    env.reset(seed=0)
    with pytest.raises(ExceptionGroup):
        env.step([0, 0])
    env.close()
    for proc in env._lanes._procs:
        proc.join(5.0)
        assert not proc.is_alive()


def test_state_round_trips_through_the_children():
    with _async(2, period=4) as env:
        env.reset(seed=0)
        env.step([0, 0])
        env.step([0, 0])
        saved = env.save_state()
        expected = env.step([0, 0])[0]["obs"]

        env.reset(seed=0)
        env.load_state(saved)
        assert np.array_equal(env.step([0, 0])[0]["obs"], expected)
