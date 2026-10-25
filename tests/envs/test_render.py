"""Rendering as an optional capability: discovery, delegation, frame format."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.envs import Renderable, check_env, supports_render
from ttrl.envs.render import check_frame
from ttrl.envs.wrappers import RecordEpisodeStatistics, TimeLimit
from ttrl.spaces import ActionSpec, ObsSpec

SPEC = ObsSpec.vector(2)


class Blinker:
    """A two-pixel environment whose frame encodes the step counter."""

    episode_bound = 4
    obs_spec = SPEC
    action_spec = ActionSpec.discrete(2)

    def __init__(self, *, renders: bool = True, frame=None):
        self._renders = renders
        self._frame = frame
        self.t = 0

    def can_render(self) -> bool:
        return self._renders

    def render(self) -> np.ndarray:
        if self._frame is not None:
            return self._frame
        return np.full((4, 6, 3), self.t % 256, dtype=np.uint8)

    def reset(self, *, seed=None):
        self.t = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        done = self.t % 4 == 0
        info = {"episode": {"r": 4.0, "l": 4}} if done else {}
        return np.zeros(2, dtype=np.float32), 1.0, done, False, info

    def close(self):
        pass


class Blind:
    """An environment with no rendering at all, which is the common case."""

    obs_spec = SPEC
    action_spec = ActionSpec.discrete(2)

    def reset(self, *, seed=None):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(2, dtype=np.float32), 0.0, False, False, {}

    def close(self):
        pass


def test_an_env_without_rendering_is_not_renderable():
    """The common case must answer False rather than raise or need hasattr."""
    assert not supports_render(Blind())
    assert not isinstance(Blind(), Renderable)


def test_an_env_that_could_render_but_cannot_now_answers_false():
    """A Gymnasium env built without render_mode is exactly this case: the
    methods exist and the frames do not."""
    env = Blinker(renders=False)
    assert isinstance(env, Renderable)
    assert not supports_render(env)


def test_a_wrapper_stack_renders_without_reaching_past_it():
    """The bug this closes: the viewer had to call unwrapped.unwrapped."""
    env = TimeLimit(RecordEpisodeStatistics(Blinker()), 10)
    assert supports_render(env)
    env.reset()
    env.step(0)
    frame = env.render()
    assert frame.shape == (4, 6, 3) and frame[0, 0, 0] == 1


def test_a_wrapper_over_a_blind_env_says_what_to_do():
    env = TimeLimit(Blind(), 10)
    with pytest.raises(RuntimeError, match="supports_render"):
        env.render()


@pytest.mark.parametrize(
    "frame, match",
    [
        ([[[0, 0, 0]]], "numpy array"),
        (np.zeros((4, 6, 3), dtype=np.float32), "uint8"),
        (np.zeros((4, 6), dtype=np.uint8), "height, width, 3"),
        (np.zeros((4, 6, 4), dtype=np.uint8), "height, width, 3"),
        (np.zeros((0, 6, 3), dtype=np.uint8), "height, width, 3"),
    ],
)
def test_check_frame_refuses_anything_that_is_not_an_rgb_frame(frame, match):
    with pytest.raises((TypeError, ValueError), match=match):
        check_frame(frame)


def test_check_env_holds_a_renderable_env_to_the_frame_format():
    """A frame in the wrong format is caught by the checker rather than by
    whatever the caller feeds the frame to."""
    bad = Blinker(frame=np.zeros((4, 6, 3), dtype=np.float32))
    with pytest.raises(TypeError, match="render after reset"):
        check_env(bad, steps=4)


def test_check_env_passes_an_env_that_cannot_render():
    # `Blind` never ends an episode, which is the one case `check_env` takes
    # a declaration for rather than a bound: it can be trained on and must be
    # scored through a separate bounded TrainConfig.eval_env.
    check_env(RecordEpisodeStatistics(Blind()), steps=4, bounded=False)
