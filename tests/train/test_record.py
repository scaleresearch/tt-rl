"""Recording a policy: what is refused, and what reaches the file."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.envs import ENVS, SyncVectorEnv, make_atari, register
from ttrl.envs.render import check_frame
from ttrl.envs.wrappers import RecordEpisodeStatistics
from ttrl.train.config import EnvConfig
from ttrl.train.film import Film
from ttrl.train.record_env import render_env

from tests._fakes.train_layer import ToyEnv
from ttrl.envs.vec_normalize import NormalizeObservation
from ttrl.train.record import _frozen_transform, _roll


class Watchable(ToyEnv):
    """A toy environment that can hand over frames, as Gymnasium does it."""

    def __init__(self, *, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.render_mode = render_mode

    def can_render(self) -> bool:
        return self.render_mode == "rgb_array"

    def render(self) -> np.ndarray:
        return np.full((8, 6, 3), 7, dtype=np.uint8)


@pytest.fixture
def registered():
    names: list[str] = []
    yield names
    for name in names:
        ENVS.unregister(name)


def test_a_renderable_builder_is_built_with_the_render_mode(registered):
    register("watchable", lambda **kwargs: RecordEpisodeStatistics(Watchable(**kwargs)))
    registered.append("watchable")
    env = render_env(EnvConfig(name="watchable", kwargs={"episode_len": 3}))
    try:
        assert check_frame(env.render()).shape == (8, 6, 3)
    finally:
        env.close()


def test_a_builder_that_takes_no_render_mode_is_refused_by_name(registered):
    def make_blind(episode_len: int = 5):
        return RecordEpisodeStatistics(ToyEnv(episode_len=episode_len))

    register("blind", make_blind)
    registered.append("blind")
    with pytest.raises(ValueError) as excinfo:
        render_env(EnvConfig(name="blind"))
    message = str(excinfo.value)
    assert "render_mode" in message
    assert "make_blind" in message
    assert "Renderable" in message


def test_rendering_does_not_change_the_training_observation():
    """Asking Atari for frames must not move the 84x84 greyscale stack."""
    ale = pytest.importorskip("ale_py")  # noqa: F841

    def observations(**extra):
        env = make_atari(game="ALE/Pong-v5", episodic_life=False,
                         reward="none", **extra)
        try:
            obs, _info = env.reset(seed=7)
            out = [obs.copy()]
            for action in (0, 2, 3, 1, 2):
                obs, _r, _t, _tr, _i = env.step(action)
                out.append(obs.copy())
            return out
        finally:
            env.close()

    plain = observations()
    filmed = observations(render_mode="rgb_array")
    assert plain[0].shape == (4, 84, 84) and plain[0].dtype == np.uint8
    for before, after in zip(plain, filmed):
        np.testing.assert_array_equal(before, after)


class Recorder:
    """An agent that keeps what it was handed and acts on nothing."""

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def act(self, obs, *, deterministic=False, episode_starts=None):
        self.seen.append(np.asarray(obs).copy())
        return np.zeros(np.asarray(obs).shape[0], dtype=np.int64)

    def close(self) -> None:
        pass


def test_the_recorder_whitens_with_the_statistics_the_run_fitted(registered,
                                                                 tmp_path):
    """THE release blocker: a run that normalised observations must film."""
    pytest.importorskip("cv2")

    def lane():
        return RecordEpisodeStatistics(Watchable(episode_len=4))

    register("watchable", lambda **kwargs: RecordEpisodeStatistics(
        Watchable(**kwargs)))
    registered.append("watchable")

    with NormalizeObservation(SyncVectorEnv([lane] * 2), clip=0.5) as fitting:
        fitting.reset(seed=3)
        for _ in range(6):
            fitting.step(np.zeros(2, dtype=np.int64))
        snapshot = fitting.save_policy_state()

    env = render_env(EnvConfig(name="watchable", kwargs={"episode_len": 4}))
    agent = Recorder()
    try:
        whiten = _frozen_transform(env, snapshot, "fitted.npz")
        _roll(agent, env, str(tmp_path / "clip.mp4"), episodes=1, seed=5,
              fps=5.0, max_episode_steps=10, whiten=whiten,
              film=Film(title="toy"))
    finally:
        env.close()

    items = dict(snapshot[0][1])
    mean = np.asarray(items["mean"], dtype=np.float64)
    std = np.sqrt(items["m2"] / items["count"] + items["epsilon"])
    raw = np.stack([np.full(4, float(5 * 100 + t), dtype=np.float32)
                    for t in range(len(agent.seen))])
    expected = np.clip((raw - mean) / std, -items["clip"],
                       items["clip"]).astype(np.float32)
    assert agent.seen and not np.allclose(raw, expected), (
        "the fixture must make whitening visible"
    )
    np.testing.assert_allclose(np.concatenate(agent.seen), expected, rtol=1e-6)


@pytest.mark.parametrize("snapshot, match", [
    (None, "carries no 'env_policy_state'"),
    ([("ScaleObservation", {})], "ScaleObservation"),
])
def test_a_transform_this_recorder_cannot_apply_is_refused_by_name(
        registered, snapshot, match):
    """The two things a checkpoint can say that this recorder cannot honour."""

    register("watchable", lambda **kwargs: RecordEpisodeStatistics(
        Watchable(**kwargs)))
    registered.append("watchable")
    env = render_env(EnvConfig(name="watchable"))
    try:
        with pytest.raises(ValueError, match=match):
            _frozen_transform(env, snapshot, "old.npz")
    finally:
        env.close()
