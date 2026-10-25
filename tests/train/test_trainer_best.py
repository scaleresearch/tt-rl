"""The Trainer's best-checkpoint wiring, including across a warm resume."""

from __future__ import annotations

import json

import pytest

from tests._fakes.train_layer import RecordingLogger, ScriptedAgent
from ttrl.train import builders, periodic as periodic_mod
from ttrl.train.checkpoint import BEST, describe, history, read_meta
from ttrl.train.config import EnvConfig, TrainConfig
from ttrl.train.trainer import Trainer


def config(tmp_path, **kwargs):
    base = dict(
        total_frames=8,
        eval_interval=2,
        eval_episodes=1,
        checkpoint_interval=4,
        log_dir=str(tmp_path / "run"),
        env=EnvConfig(name="toy", kwargs={"episode_len": 3}),
    )
    return TrainConfig(**{**base, **kwargs})


@pytest.fixture
def agents(monkeypatch):
    built = []

    def build_agent(cfg, *, obs_spec, action_spec, device, profile=None,
                    algo_cfg=None):
        agent = ScriptedAgent(obs_spec, action_spec)
        built.append(agent)
        return agent

    monkeypatch.setattr(builders, "build_agent", build_agent)
    return built


@pytest.fixture
def scores(monkeypatch):
    """Hand the run a scripted eval score curve; record what it was asked.

    `curve.std` is the within-eval spread every eval reports, which is what
    the kept checkpoint's context is judged against.
    """

    class Curve(list):
        std = 2.0

    curve = Curve()
    asked: list[float] = []

    def evaluate(agent, env_config, *, n_episodes, seed, env_policy_state=None):
        score = curve[len(asked)] if len(asked) < len(curve) else curve[-1]
        asked.append(score)
        return {"n_episodes": 1, "scores": [score], "mean": score,
                "median": score, "min": score, "max": score,
                "std": curve.std}

    monkeypatch.setattr(periodic_mod, "evaluate", evaluate)
    return curve


def _best_extra(log_dir):
    _arrays, meta = read_meta(str(log_dir / "run" / BEST))
    return meta["extra"]


def test_the_best_eval_keeps_its_checkpoint(tmp_path, agents, scores):
    scores[:] = [5.0, 50.0, 7.0, 8.0]
    with Trainer(config(tmp_path), object(), object(),
                 logger=RecordingLogger()) as trainer:
        trainer.run()
    extra = _best_extra(tmp_path)
    assert extra["best_score"] == 50.0
    assert extra["best_frame"] == 4
    assert extra["best_metric"] == "eval/mean" and extra["best_mode"] == "max"


def test_the_kept_best_carries_what_its_neighbours_scored(
    tmp_path, agents, scores
):
    """Selecting the maximum picks a high draw, so the file says how high.

    The curve below is the shape Hopper seed 1 showed: a best that arrives
    straight after a far lower eval, against a small within-eval std.
    Selection must be untouched by the context, so this pins the kept frame.
    """
    scores[:] = [5.0, 50.0, 7.0, 8.0]
    with Trainer(config(tmp_path), object(), object(),
                 logger=RecordingLogger()) as trainer:
        trainer.run()
        stamp = trainer.best.stamp()
    extra = _best_extra(tmp_path)

    # Unchanged selection: the maximum, at the frame it happened.
    assert (extra["best_score"], extra["best_frame"]) == (50.0, 4)
    # The kept eval's own spread, and the two evals seen when it was kept.
    assert extra["best_eval_std"] == 2.0
    assert extra["best_evals"] == 2
    assert extra["best_evals_mean"] == pytest.approx(27.5)
    assert extra["best_prev_eval"] == 5.0
    # The run's outcome record and the file agree, because both are the stamp.
    assert json.loads((tmp_path / "run" / "outcome.json").read_text())[
        "best"] == stamp == {k: extra[k] for k in stamp}

    line = describe(extra)
    assert "best eval/mean 50 at frame 4" in line
    assert "within-eval std 2" in line
    assert "the final checkpoint is the honest one to publish" in line


def test_a_settled_run_is_not_warned_about(tmp_path, agents, scores):
    """A flat curve does not earn the warning. Hopper seed 0 is this shape."""
    scores[:] = [50.0, 50.5, 51.0, 51.5]
    with Trainer(config(tmp_path), object(), object(),
                 logger=RecordingLogger()) as trainer:
        trainer.run()
    assert "honest one to publish" not in describe(_best_extra(tmp_path))


def test_a_steadily_improving_run_is_not_warned_about(
    tmp_path, agents, scores
):
    """The regression that picked the local statistic over a whole-run spread.

    A run that climbs has a large spread across its evals BY CONSTRUCTION, so
    judging the kept eval against that spread warns about every healthy Atari
    run. These are the shipped Breakout PPO run's own evals and its
    within-eval spread, on which the whole-run statistic read 0.5x and the
    neighbour it actually uses reads 0.1x.
    """
    scores[:] = [10.03, 16.87, 21.43, 31.2, 32.47, 39.0, 49.67, 53.1, 58.27,
                 61.83, 72.8, 73.2, 86.73, 84.73, 89.3, 95.53, 94.7, 85.73,
                 107.33, 100.03, 120.33, 125.17]
    scores.std = 66.75
    cfg = config(tmp_path, total_frames=2 * len(scores))
    with Trainer(cfg, object(), object(), logger=RecordingLogger()) as trainer:
        trainer.run()
    extra = _best_extra(tmp_path)
    assert extra["best_score"] == 125.17
    assert "honest one to publish" not in describe(extra)


def test_the_best_checkpoint_outlives_the_retention_bound(
    tmp_path, agents, scores
):
    """The gap this closes: the good policy is the one pruning would delete."""

    scores[:] = [50.0, 1.0, 1.0, 1.0]
    cfg = config(tmp_path, total_frames=16, checkpoint_interval=2,
                 checkpoint_keep=1)
    with Trainer(cfg, object(), object(), logger=RecordingLogger()) as trainer:
        trainer.run()
    kept = history(str(tmp_path / "run"))
    assert [frame for frame, _ in kept] == [16]
    assert _best_extra(tmp_path)["best_frame"] == 2


def test_a_worse_score_after_a_resume_does_not_take_the_best_file(
    tmp_path, agents, scores
):
    """The one most likely to be got wrong, end to end through the Trainer."""
    scores[:] = [50.0, 3.0, 3.0, 3.0]
    cfg = config(tmp_path, total_frames=8)
    with Trainer(cfg, object(), object(), logger=RecordingLogger()) as trainer:
        trainer.run()
    assert _best_extra(tmp_path)["best_score"] == 50.0

    scores[:] = [4.0]
    resumed = config(tmp_path, total_frames=16)
    with Trainer(resumed, object(), object(),
                 resume_path=str(tmp_path / "run" / "checkpoint.npz"),
                 logger=RecordingLogger()) as trainer:
        assert trainer.frame_count == 8
        assert trainer.best.score == 50.0
        trainer.run()
    extra = _best_extra(tmp_path)
    assert extra["best_score"] == 50.0
    assert extra["best_frame"] == 2


def test_a_fresh_run_does_not_inherit_the_directorys_best_score(
    tmp_path, agents, scores
):
    """A fresh run has earned nothing, so it starts with no high-water mark."""
    scores[:] = [50.0]
    cfg = config(tmp_path, total_frames=8)
    with Trainer(cfg, object(), object(), logger=RecordingLogger()) as trainer:
        trainer.run()

    scores[:] = [1.0]
    with Trainer(config(tmp_path), object(), object(),
                 logger=RecordingLogger()) as trainer:
        assert trainer.best.score is None
        trainer.run()
    assert _best_extra(tmp_path)["best_score"] == 1.0
