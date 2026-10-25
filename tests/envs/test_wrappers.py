"""The observation and reward wrappers, against a scripted env."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.envs.wrappers import (
    FrameStack,
    TimeLimit,
    restore_env,
    snapshot_env,
)
from ttrl.spaces import ActionSpec, ObsSpec


class CountingEnv:
    """Yields frame `n` filled with the constant `n`, so stack order is legible."""

    def __init__(self, hw: int = 3, reward: float = 1.0):
        self.obs_spec = ObsSpec.image(1, hw, hw)
        self.action_spec = ActionSpec.discrete(2)
        self._hw = hw
        self._reward = reward
        self._n = 0

    def _frame(self, n: int) -> np.ndarray:
        return np.full((1, self._hw, self._hw), n % 256, dtype=np.uint8)

    def reset(self, *, seed=None):
        self._n = 0
        return self._frame(0), {}

    def step(self, action):
        self._n += 1
        return self._frame(self._n), self._reward, False, False, {}

    def close(self):
        pass

    def get_state(self):
        return {"n": self._n}

    def set_state(self, state):
        self._n = state["n"]


@pytest.mark.parametrize("k", [1, 4])
def test_in_place_shift_matches_an_np_roll_reference(k: int) -> None:
    """The in-place history shift is element-for-element `np.roll(..., -1)`."""
    env = FrameStack(CountingEnv(hw=3), k)
    obs, _ = env.reset()

    ref = obs.reshape(k, 1, 3, 3).copy()
    for n in range(1, 12):
        got, _r, _t, _tr, _i = env.step(0)
        ref = np.roll(ref, shift=-1, axis=0)
        ref[-1] = np.full((1, 3, 3), n, dtype=np.uint8)
        assert np.array_equal(got, ref.reshape(k, 3, 3)), f"k={k} step={n}"


def test_stack_is_oldest_to_newest() -> None:
    env = FrameStack(CountingEnv(hw=2), 4)
    env.reset()
    for _ in range(6):
        obs, *_ = env.step(0)
    # Frames 3, 4, 5, 6 came from the last four steps, oldest channel first.
    assert [int(obs[c, 0, 0]) for c in range(4)] == [3, 4, 5, 6]


def test_reset_fills_the_history_with_the_reset_frame() -> None:
    """Zero-padding the first frames of an episode puts them out of
    distribution for the whole of the agent's first k steps."""
    env = FrameStack(CountingEnv(hw=2), 4)
    obs, _ = env.reset()
    assert np.array_equal(obs, np.zeros((4, 2, 2), dtype=np.uint8))

    env._frames[:] = 7  # a dirty history from the previous episode
    obs, _ = env.reset()
    assert np.array_equal(obs, np.zeros((4, 2, 2), dtype=np.uint8))


def test_returned_stack_is_not_aliased_to_the_internal_buffer() -> None:
    env = FrameStack(CountingEnv(hw=2), 4)
    env.reset()
    first, *_ = env.step(0)
    snapshot = first.copy()
    for _ in range(3):
        env.step(0)
    assert np.array_equal(first, snapshot)
    assert first is not env._frames


def test_frame_stack_declares_the_shape_it_returns() -> None:
    env = FrameStack(CountingEnv(hw=5), 3)
    obs, _ = env.reset()
    assert env.obs_spec.shape == obs.shape == (3, 5, 5)
    assert env.obs_spec.kind == "image"
    assert env.obs_spec.scale == CountingEnv().obs_spec.scale


def test_frame_stack_round_trips_its_history() -> None:
    """A checkpoint that drops the stack resumes against a history of one
    frame repeated, which is a different observation from the one the agent
    was trained on."""
    env = FrameStack(CountingEnv(hw=2), 4)
    env.reset()
    for _ in range(5):
        env.step(0)
    state = snapshot_env(env)
    expected, *_ = env.step(0)

    env.reset()
    restore_env(env, state)
    assert np.array_equal(env.step(0)[0], expected)


def test_time_limit_truncates_without_terminating() -> None:
    """Truncation and termination differ in the bootstrap mask, so a limit
    reported as a termination trains the value of the final state to zero."""
    env = TimeLimit(CountingEnv(), 3)
    env.reset()
    flags = [env.step(0)[2:4] for _ in range(3)]
    assert flags == [(False, False), (False, False), (False, True)]
