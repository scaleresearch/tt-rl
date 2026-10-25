"""A run's checkpoint history: frame-keyed files, a stable latest, bounded."""

from __future__ import annotations

import os

import pytest

from tests._fakes.train_layer import RecordingLogger, ScriptedAgent
from ttrl.train import builders
from ttrl.train.checkpoint import (
    LATEST,
    history,
    load_checkpoint,
)
from ttrl.train.config import EnvConfig, TrainConfig
from ttrl.train.trainer import Trainer


class FakeDevice:
    pass


def config(tmp_path, **kwargs):
    base = dict(
        total_frames=8,
        eval_interval=1000,
        checkpoint_interval=2,
        log_dir=str(tmp_path / "run"),
        env=EnvConfig(name="toy", kwargs={"episode_len": 3}),
    )
    return TrainConfig(**{**base, **kwargs})


@pytest.fixture
def agents(monkeypatch):
    def build_agent(cfg, *, obs_spec, action_spec, device, profile=None, algo_cfg=None):
        return ScriptedAgent(obs_spec, action_spec)

    monkeypatch.setattr(builders, "build_agent", build_agent)


def _run(tmp_path, **kwargs):
    cfg = config(tmp_path, **kwargs)
    with Trainer(cfg, FakeDevice(), object(), logger=RecordingLogger()) as trainer:
        trainer.run()
    return cfg.log_dir


def test_the_cadence_keeps_every_checkpoint_it_writes(tmp_path, agents):
    log_dir = _run(tmp_path, checkpoint_keep=10)
    frames = [frame for frame, _ in history(log_dir)]
    assert frames == [2, 4, 6, 8], (
        "a run checkpointing every 2 frames to frame 8 must leave a "
        "progression, not one overwritten file"
    )
    for frame, path in history(log_dir):
        assert load_checkpoint(path)["frame"] == frame


def test_the_stable_path_resolves_to_the_newest(tmp_path, agents):
    log_dir = _run(tmp_path, checkpoint_keep=10)
    latest = os.path.join(log_dir, LATEST)
    assert os.path.islink(latest), "--resume stays simple because this path exists"
    assert os.readlink(latest) == "checkpoint-000000000008.npz"
    assert load_checkpoint(latest)["frame"] == 8


def test_retention_is_bounded_by_the_config(tmp_path, agents):
    log_dir = _run(tmp_path, checkpoint_keep=2)
    assert [frame for frame, _ in history(log_dir)] == [6, 8], (
        "an Atari checkpoint is ~27 MB, so an unbounded history fills a disk "
        "mid-run; the bound is the user's number and prunes oldest first"
    )
    assert load_checkpoint(os.path.join(log_dir, LATEST))["frame"] == 8

