"""The toy stacks and run helpers the policy-state tests share.

`NormalizeObservation` is the only layer in the tree that declares a
non-empty `POLICY_STATE`, so a test about what a checkpoint carries needs a
run with it and a run without one. It sits ABOVE the vector environment and
is switched on by `EnvConfig.normalize_obs`, so the difference between the
two is a config field rather than two registered builders: `config(...)`
takes `normalize_obs`, and the registered lane builders differ only in what
is inside the lane.

Lives here rather than in either test module because two modules need it:
`tests/train/test_env_state.py` asks what the file carries and
`tests/train/test_env_state_resume.py` asks what a resume does with it.
"""

from __future__ import annotations

import json
import os

import numpy as np

from tests._fakes.train_layer import RecordingLogger, ScriptedAgent, ToyEnv
from ttrl.envs import ENVS, register
from ttrl.envs.wrappers import FrameStack, RecordEpisodeStatistics
from ttrl.train.checkpoint import LATEST, read_meta
from ttrl.train.config import EnvConfig, TrainConfig
from ttrl.train.trainer import Trainer

__all__ = ["FakeDevice", "RecordingAgent", "config", "strip_policy_state",
           "reward_scale", "train", "welford"]


def make_toy_norm(**kwargs):
    """The lane the normalised runs below are built from."""
    return RecordEpisodeStatistics(ToyEnv(**kwargs))


def make_toy_stacked(**kwargs):
    """A lane holding a frame stack: trajectory state under the transform, so
    a checkpoint can be asked to carry one and not the other."""
    return FrameStack(RecordEpisodeStatistics(ToyEnv(**kwargs)), 2)


if "toy_norm" not in ENVS:
    register("toy_norm", make_toy_norm)
if "toy_stacked" not in ENVS:
    register("toy_stacked", make_toy_stacked)


class FakeDevice:
    pass


class RecordingAgent(ScriptedAgent):
    """Keeps the first leaf of every observation batch it acted on."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seen: list[np.ndarray] = []

    def act(self, obs, *, deterministic: bool = False, episode_starts=None):
        self.seen.append(np.array(obs[obs.leaves[0]], copy=True))
        return super().act(obs, deterministic=deterministic)


def config(tmp_path, *, name="toy_norm", normalize_obs=None,
           normalize_reward=False, **kwargs):
    """A run config. `normalize_obs` defaults to whether the name asks for it.

    The two registered builders differ in what is in the lane; whether the
    run whitens is `EnvConfig.normalize_obs`, because the estimator is one
    object above every lane. Defaulting it off the name keeps every caller
    that said "the normalised stack" or "the plain one" saying that.
    """
    whiten = name != "toy" if normalize_obs is None else normalize_obs
    base = dict(
        total_frames=12,
        eval_interval=10_000,
        eval_episodes=1,
        checkpoint_interval=12,
        log_dir=str(tmp_path / "run"),
        env=EnvConfig(name=name, kwargs={"episode_len": 3},
                      normalize_obs=whiten,
                      normalize_reward=normalize_reward,
                      # Stated, because these runs carry no algo_cfg to take
                      # it from and `TrainConfig` refuses to invent one.
                      normalize_reward_gamma=0.9 if normalize_reward else None),
    )
    return TrainConfig(**{**base, **kwargs})


def train(tmp_path, **kwargs):
    """Run to completion and return (config, latest checkpoint, live state)."""
    cfg = config(tmp_path, **kwargs)
    with Trainer(cfg, FakeDevice(), object(), logger=RecordingLogger()) as trainer:
        trainer.run()
        live = trainer.replicas.lead.env.save_policy_state()
    return cfg, os.path.join(cfg.log_dir, LATEST), live


def welford(snapshot):
    """The normaliser's statistics out of a policy snapshot."""
    name, fields = snapshot[0]
    assert name == "NormalizeObservation"
    return fields


def reward_scale(snapshot):
    """The reward normaliser's statistics out of a run-state snapshot."""
    name, fields = snapshot[0]
    assert name == "NormalizeReward"
    return fields


def strip_policy_state(latest: str) -> str:
    """Rewrite a checkpoint as one whose writer passed no policy snapshot.

    `save_checkpoint` does not require the keyword, so this is a file this
    build writes, not an older format.
    """

    real = os.path.realpath(latest)
    arrays, meta = read_meta(real)
    meta["env_policy_state"] = None
    np.savez(real, **arrays, __meta__=np.frombuffer(
        json.dumps(meta).encode(), dtype=np.uint8))
    return real
