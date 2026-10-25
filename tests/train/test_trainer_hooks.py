"""The caller's hook surface, the early stop, and the outcome record."""

from __future__ import annotations

import dataclasses
import json

import pytest

from tests._fakes.train_layer import RecordingLogger, ScriptedAgent
from ttrl.train import builders, periodic as periodic_mod
from ttrl.train.checkpoint import load_checkpoint
from ttrl.train.hooks import TrainHooks
from ttrl.train.config import EnvConfig, TrainConfig
from ttrl.train.provenance import OUTCOME_NAME, STATUSES
from ttrl.train.stop import StopOnRewardThreshold
from ttrl.train.trainer import Trainer


def config(tmp_path, **kwargs):
    base = dict(
        total_frames=8,
        eval_interval=2,
        eval_episodes=1,
        checkpoint_interval=1000,
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
    curve: list[float] = []
    asked: list[float] = []

    def evaluate(agent, env_config, *, n_episodes, seed, env_policy_state=None):
        score = curve[len(asked)] if len(asked) < len(curve) else curve[-1]
        asked.append(score)
        return {"n_episodes": 1, "scores": [score], "mean": score,
                "median": score, "min": score, "max": score}

    monkeypatch.setattr(periodic_mod, "evaluate", evaluate)
    return curve


def _outcome(tmp_path) -> dict:
    return json.loads((tmp_path / "run" / OUTCOME_NAME).read_text())


class Recorder:
    """One object with a bound method per hook, which is the intended shape."""

    def __init__(self) -> None:
        self.metrics: list[tuple[dict, int]] = []
        self.rounds: list[tuple[int, float]] = []
        self.updates = 0
        self.update_metrics: list[dict] = []
        self.states: list = []

    def on_metrics(self, metrics, step):
        self.metrics.append((dict(metrics), step))

    def on_round_end(self, frame, elapsed):
        self.rounds.append((frame, elapsed))

    def on_update(self, metrics):
        self.updates += 1
        self.update_metrics.append(dict(metrics))

    def stop(self, state):
        self.states.append(state)
        return None


def test_every_declared_hook_point_fires(tmp_path, agents, scores):
    """The set of hooks is derived from `TrainHooks`, never typed out here."""
    scores[:] = [1.0]
    recorder = Recorder()
    hooks = TrainHooks(**{
        field.name: getattr(recorder, field.name)
        for field in dataclasses.fields(TrainHooks)
    })
    with Trainer(config(tmp_path), object(), object(), hooks=hooks,
                 logger=RecordingLogger()) as trainer:
        trainer.run()
    fired = {
        "on_metrics": bool(recorder.metrics),
        "on_round_end": bool(recorder.rounds),
        "on_update": recorder.updates > 0,
        "stop": bool(recorder.states),
    }
    assert set(fired) == {f.name for f in dataclasses.fields(TrainHooks)}
    assert all(fired.values()), fired


def test_a_round_hook_sees_this_rounds_eval(tmp_path, agents, scores):
    """The library's own periodic work runs first, so a hook can react to it."""
    scores[:] = [1.0, 2.0, 3.0, 4.0]
    seen = []

    def on_round_end(frame, elapsed):
        seen.append((frame, trainer.best.evals))

    with Trainer(config(tmp_path), object(), object(),
                 hooks=TrainHooks(on_round_end=on_round_end),
                 logger=RecordingLogger()) as trainer:
        trainer.run()
    assert seen == [(1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3), (7, 3),
                    (8, 4)]


def test_a_reward_threshold_ends_the_run_early(tmp_path, agents, scores):
    scores[:] = [1.0, 99.0]
    hooks = TrainHooks(stop=StopOnRewardThreshold(50.0))
    with Trainer(config(tmp_path, total_frames=1000), object(), object(),
                 hooks=hooks, logger=RecordingLogger()) as trainer:
        result = trainer.run()
    assert result["frame_count"] == 4
    assert result["status"] == "stopped"
    assert "passes the max threshold 50" in result["stop_reason"]


def test_a_stop_leaves_by_the_normal_path(tmp_path, agents, scores):
    """A clean teardown: the final checkpoint is written and the parts close."""
    scores[:] = [99.0]
    hooks = TrainHooks(stop=StopOnRewardThreshold(50.0))
    logger = RecordingLogger()
    with Trainer(config(tmp_path, total_frames=1000), object(), object(),
                 hooks=hooks, logger=logger) as trainer:
        trainer.run()
    assert agents[0].closes == 1 and logger.closed == 1

    latest = load_checkpoint(str(tmp_path / "run" / "checkpoint.npz"))
    assert latest["frame"] == 2


def test_a_completed_run_says_so(tmp_path, agents, scores):
    scores[:] = [1.0]
    with Trainer(config(tmp_path), object(), object(),
                 logger=RecordingLogger()) as trainer:
        trainer.run()
    outcome = _outcome(tmp_path)
    assert outcome["status"] == "completed"
    assert outcome["reason"] is None and outcome["error"] is None
    assert outcome["frame_count"] == 8
    assert outcome["profile"]["frames"] == 8


def test_a_stopped_run_records_why(tmp_path, agents, scores):
    scores[:] = [99.0]
    with Trainer(config(tmp_path, total_frames=1000), object(), object(),
                 hooks=TrainHooks(stop=StopOnRewardThreshold(50.0)),
                 logger=RecordingLogger()) as trainer:
        trainer.run()
    outcome = _outcome(tmp_path)
    assert outcome["status"] == "stopped"
    assert "threshold" in outcome["reason"]


def test_a_crashed_run_is_neither_of_those(tmp_path, agents, scores):
    """An early stop, a completed run and a crash must not look alike."""
    scores[:] = [1.0]
    with pytest.raises(ZeroDivisionError, match="boom"):
        with Trainer(config(tmp_path), object(), object(),
                     logger=RecordingLogger()):
            raise ZeroDivisionError("boom")
    outcome = _outcome(tmp_path)
    assert outcome["status"] == "incomplete"
    assert "ZeroDivisionError('boom')" == outcome["error"]


def test_every_status_the_record_can_carry_is_reachable():
    """Derived from `provenance.STATUSES`, so a new status needs a test."""
    covered = {
        "completed": test_a_completed_run_says_so,
        "stopped": test_a_stopped_run_records_why,
        "incomplete": test_a_crashed_run_is_neither_of_those,
    }
    assert set(STATUSES) == set(covered)
