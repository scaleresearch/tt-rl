"""The parts of the env contract a custom environment gets wrong silently."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ttrl
from ttrl.envs import (
    EPISODE_KEY,
    FINAL_OBS_KEY,
    LANE_KEY,
    SyncVectorEnv,
    check_env,
    check_vec_env,
    supports_render,
)
from ttrl.envs import ENVS, register
from ttrl.envs.registry import EnvFactory
from ttrl.envs.wrappers import RecordEpisodeStatistics

SPEC = ttrl.ObsSpec.vector(2)


class Ticker:
    """Ends an episode every `period` steps, reporting no statistics."""

    obs_spec = SPEC
    action_spec = ttrl.ActionSpec.discrete(2)

    def __init__(self, period: int = 3, *, episode=None):
        self.episode_bound = period
        self.period = period
        self.episode = episode
        self.t = 0

    def reset(self, *, seed=None):
        self.t = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        done = self.t % self.period == 0
        info = {} if (not done or self.episode is None) else {EPISODE_KEY: self.episode}
        return np.zeros(2, dtype=np.float32), 1.0, done, False, info

    def close(self):
        pass


def test_the_library_computes_the_statistics_so_no_wrapper_is_required():
    """A bare environment, no wrapper, and the score is there."""
    check_env(Ticker(), steps=6)

    vec = SyncVectorEnv([Ticker] * 2)
    try:
        check_vec_env(vec, steps=8)
        vec.reset(seed=0)
        # Two episodes, so the autoreset step between them is crossed. Under
        # NEXT_STEP that step delivers a reset rather than a transition, and
        # counting it would put a spurious fourth step on every episode after
        # the first.
        seen = []
        for _ in range(7):
            _o, _r, term, trunc, info = vec.step(np.zeros(2, dtype=np.int64))
            if (np.asarray(term) | np.asarray(trunc)).any():
                seen.append([info[LANE_KEY][lane][EPISODE_KEY] for lane in (0, 1)])
        assert seen == [[{"r": 3.0, "l": 3}] * 2] * 2, seen
    finally:
        vec.close()


def test_a_vec_env_that_drops_the_statistics_is_still_refused():
    """The vector clause is enforced, which is what makes the line above a
    statement about this library rather than about the checker going quiet."""

    class NoStats(SyncVectorEnv):
        def step(self, actions):
            obs, r, term, trunc, info = super().step(actions)
            lanes = [{k: v for k, v in lane.items() if k != EPISODE_KEY}
                     for lane in info[LANE_KEY]]
            return obs, r, term, trunc, {**info, LANE_KEY: lanes}

    with pytest.raises(ValueError, match=EPISODE_KEY):
        check_vec_env(NoStats([Ticker] * 2), steps=8)


@pytest.mark.parametrize("episode", [{"r": 3.0}, {"r": float("nan"), "l": 3}, 3.0])
def test_statistics_must_carry_a_finite_return_and_length(episode):
    with pytest.raises(ValueError, match=EPISODE_KEY):
        check_env(Ticker(episode=episode), steps=6)


def test_a_vec_env_must_pass_each_lane_info_through():
    """Statistics are read per lane, so a vector env that drops `lane` is
    unusable even though every lane reports correctly."""

    class NoLanes(SyncVectorEnv):
        def step(self, actions):
            obs, r, term, trunc, info = super().step(actions)
            return obs, r, term, trunc, {FINAL_OBS_KEY: info[FINAL_OBS_KEY]}

    with pytest.raises(ValueError, match=LANE_KEY):
        check_vec_env(NoLanes([lambda: RecordEpisodeStatistics(Ticker())] * 2), steps=8)


class Dial:
    """A custom environment written against the public surface only."""

    obs_spec = ttrl.ObsSpec.vector(1)
    action_spec = ttrl.ActionSpec.discrete(3)

    def __init__(self, horizon: int = 8):
        self.episode_bound = horizon
        self.horizon = horizon
        self.rng = np.random.default_rng(0)
        self.x, self.t = 0.0, 0

    def reset(self, *, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x, self.t = float(self.rng.uniform(-1.0, 1.0)), 0
        return self._obs(), {}

    def _obs(self):
        return np.array([self.x], dtype=np.float32)

    def step(self, action):
        before = abs(self.x)
        self.x = float(np.clip(self.x + (int(action) - 1) * 0.25, -1.0, 1.0))
        self.t += 1
        return self._obs(), before - abs(self.x), False, self.t >= self.horizon, {}

    def can_render(self) -> bool:
        return True

    def render(self) -> np.ndarray:
        frame = np.zeros((16, 4, 3), dtype=np.uint8)
        frame[int((self.x + 1.0) * 7.5), :, 1] = 255
        return frame

    def close(self):
        pass


def make_dial(**kwargs):
    return RecordEpisodeStatistics(Dial(**kwargs))


def test_a_custom_env_satisfies_the_whole_contract_including_rendering():
    register("dial", make_dial)
    try:
        check_env(make_dial(), steps=20)
        with SyncVectorEnv([EnvFactory("dial")] * 2) as vec:
            check_vec_env(vec, steps=20)
        env = make_dial()
        env.reset(seed=0)
        assert supports_render(env)
        frames = []
        for action in (0, 0, 2, 2):
            env.step(action)
            frames.append(env.render())
        assert all(f.shape == (16, 4, 3) and f.dtype == np.uint8 for f in frames)
        assert any(not np.array_equal(frames[0], f) for f in frames)
        env.close()
    finally:
        ENVS.unregister("dial")


class Swapped(Ticker):
    """A correct environment with its two done flags exchanged.

    The easiest mistake an outside author makes, and the only one no return
    value is wrong enough to refuse: every flag has the right type and every
    episode ends. What it costs is downstream, where a learner bootstraps the
    value of a state that has no successor.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, truncated, terminated, info


class Limited(Ticker):
    """Only ever truncated, and only ever at its own declared bound.

    The legitimate always-truncating environment: a `TimeLimit` around
    something endless. It must not be called out.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, False, terminated, info


def test_a_swapped_terminated_and_truncated_is_called_out():
    env = Swapped(period=3)
    # Episodes of 3 under a bound of 5: they end with a future they never
    # had cut off, which is the pattern the warning reads.
    env.episode_bound = 5
    with pytest.warns(UserWarning, match="other order"):
        check_env(env, steps=9)


def test_an_environment_truncating_at_its_own_bound_is_not_called_out():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(Limited(period=3), steps=9)


def test_an_environment_that_ever_terminates_is_not_called_out():
    """One honest termination anywhere in the rollout proves the flag is
    reachable, so an env that also truncates early passes in silence."""
    class Mixed(Ticker):
        def step(self, action):
            obs, reward, terminated, _truncated, info = super().step(action)
            self.swap = not getattr(self, "swap", False)
            return obs, reward, terminated and self.swap, \
                terminated and not self.swap, info

    env = Mixed(period=3)
    env.episode_bound = 5
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(env, steps=9)
