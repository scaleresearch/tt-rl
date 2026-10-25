"""The Trainer: lifetime, resume, device ownership and the teardown ladder."""

from __future__ import annotations


import os

import numpy as np
import pytest

from tests._fakes.train_layer import RecordingLogger, ScriptedAgent, ToyAlgoConfig
from ttrl.spaces import ActionSpec, DEFAULT_LEAF, ObsGroup, ObsSpec
from ttrl.train import builders, replicas as replicas_mod
from ttrl.train.checkpoint import L0, LATEST, load_checkpoint, save_checkpoint
from ttrl.train.config import DeviceConfig, EnvConfig, TrainConfig
from ttrl.train.trainer import Trainer


def config(tmp_path, **kwargs):
    base = dict(
        total_frames=8,
        eval_interval=1000,
        checkpoint_interval=4,
        log_dir=str(tmp_path / "run"),
        env=EnvConfig(name="toy", kwargs={"episode_len": 3}),
    )
    return TrainConfig(**{**base, **kwargs})


class FakeDevice:
    def __init__(self, index=0):
        self.index = index
        self.closed = False


@pytest.fixture
def agents(monkeypatch):
    """Substitute the agent builder; the env builder stays the real one."""
    built = []

    def build_agent(cfg, *, obs_spec, action_spec, device, profile=None, algo_cfg=None):
        agent = ScriptedAgent(obs_spec, action_spec)
        agent.device = device
        built.append(agent)
        return agent

    monkeypatch.setattr(builders, "build_agent", build_agent)
    return built


def test_trainer_is_a_context_manager_and_closes_its_parts(tmp_path, agents):
    logger = RecordingLogger()
    with Trainer(config(tmp_path), FakeDevice(), object(), logger=logger) as trainer:
        env = trainer.replicas.lead.env
    assert agents[0].closes == 1
    assert env._closed
    assert logger.closed == 1


def test_the_exception_path_still_closes_everything(tmp_path, agents):
    logger = RecordingLogger()
    with pytest.raises(ZeroDivisionError, match="boom"):
        with Trainer(config(tmp_path), FakeDevice(), object(), logger=logger):
            raise ZeroDivisionError("boom")
    assert agents[0].closes == 1
    assert logger.closed == 1


def test_resume_continues_from_the_checkpointed_frame(tmp_path, agents):
    cfg = config(tmp_path, total_frames=8)
    with Trainer(cfg, FakeDevice(), object(), logger=RecordingLogger()) as trainer:
        trainer.run()
        trained = agents[0].weight.copy()
        step = agents[0].step
    path = str(tmp_path / "run" / "checkpoint.npz")

    cfg2 = config(tmp_path, total_frames=12)
    with Trainer(cfg2, FakeDevice(), object(), resume_path=path,
                 logger=RecordingLogger()) as trainer:
        assert trainer.frame_count == 8
        assert np.array_equal(agents[1].weight, trained)
        assert agents[1].step == step
        result = trainer.run()
    assert result["frame_count"] == 12
    assert agents[1].updates == 4
    assert agents[1].step == step + 4


def test_resume_refuses_a_weights_only_checkpoint(tmp_path, agents):
    path = str(tmp_path / "w.npz")
    save_checkpoint(path, agent_state={"weight": np.zeros(4, dtype=np.float32)},
                    obs_spec=ObsSpec.vector(4), action_spec=ActionSpec.discrete(3),
                    level=L0, env_policy_state=[])
    with pytest.raises(ValueError, match="carries no frame count"):
        Trainer(config(tmp_path), FakeDevice(), object(), resume_path=path,
                logger=RecordingLogger())


def test_a_failed_construction_releases_what_it_opened(tmp_path, monkeypatch):
    envs = []
    real_build_env = builders.build_env

    def build_env(cfg):
        env = real_build_env(cfg)
        envs.append(env)
        return env

    def build_agent(cfg, **kwargs):
        raise RuntimeError("no room on the chip")

    monkeypatch.setattr(builders, "build_env", build_env)
    monkeypatch.setattr(builders, "build_agent", build_agent)
    logger = RecordingLogger()
    with pytest.raises(RuntimeError, match="no room"):
        Trainer(config(tmp_path), FakeDevice(), object(), logger=logger)
    assert envs and envs[0]._closed
    assert logger.closed == 1


def test_the_measured_frame_count_matches_the_run(tmp_path, agents):
    """A short last round must not be counted as a full one."""
    cfg = config(tmp_path, total_frames=7, env=EnvConfig(name="toy", n_envs=2,
                                                         kwargs={"episode_len": 3}))
    with Trainer(cfg, FakeDevice(), object(), logger=RecordingLogger()) as trainer:
        result = trainer.run()
    assert result["frame_count"] == 8
    assert result["profile"]["frames"] == 8


def test_per_replica_budgets_are_divided(tmp_path, monkeypatch):
    seen = []

    def build_agent(cfg, *, obs_spec, action_spec, device, profile=None, algo_cfg=None):
        seen.append(algo_cfg)
        return ScriptedAgent(obs_spec, action_spec)

    monkeypatch.setattr(builders, "build_agent", build_agent)
    monkeypatch.setattr(
        replicas_mod, "_devices",
        lambda cfg, device, profile, stack: (["d0", "d1"], [None, None]),
    )
    monkeypatch.setattr(replicas_mod, "_param_sync", lambda cfg, reps: None)
    cfg = config(tmp_path, device=DeviceConfig(chips=[0, 1]),
                 algo_cfg=ToyAlgoConfig(buffer_capacity=64))
    with Trainer(cfg, None, None, logger=RecordingLogger()):
        pass
    assert len(seen) == 2
    assert all(c.buffer_capacity == 32 for c in seen)


def test_a_budget_that_does_not_divide_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        replicas_mod, "_devices",
        lambda cfg, device, profile, stack: (["d0", "d1", "d2"], [None] * 3),
    )
    cfg = config(tmp_path, device=DeviceConfig(chips=[0, 1, 2]),
                 algo_cfg=ToyAlgoConfig(buffer_capacity=64))
    with pytest.raises(ValueError, match="does not divide by 3 replicas"):
        Trainer(cfg, None, None, logger=RecordingLogger())


class CountingCadence:
    """A cadence outside the agent, holding the run's credit in its own state."""

    def __init__(self, agent) -> None:
        self.agent = agent
        self.seen = self.owed = self.steps = 0

    def should_update(self, frame: int) -> bool:
        self.owed, self.seen = self.owed + max(0, frame - self.seen), frame
        return self.owed > 0

    def update(self) -> dict:
        owed, self.owed = self.owed, 0
        self.steps += owed
        self.agent.update()
        return {"gradient_steps": float(owed)}

    def state_dict(self) -> dict:
        return {"seen": self.seen, "owed": self.owed, "steps": self.steps}

    def load_state_dict(self, state: dict) -> None:
        self.seen, self.owed, self.steps = (
            int(state["seen"]), int(state["owed"]), int(state["steps"]))


class ListSource:
    """A StepSource that is not an environment. Carries no state, and says so."""

    def __init__(self, env_config, index: int) -> None:
        self._leaf = DEFAULT_LEAF
        self.obs_spec = ObsGroup.of(ObsSpec.vector(4))
        self.action_spec = ActionSpec.discrete(3)
        self.num_envs = 2
        self.index = index
        self.steps = 0
        self.closed = False

    def _obs(self):
        rows = np.full((self.num_envs, 4), float(self.index), np.float32)
        return self.obs_spec.observation({self._leaf: rows})

    def reset(self, *, seed=None):
        return self._obs(), {}

    def step(self, actions):
        self.steps += 1
        done = np.zeros(self.num_envs, bool)
        return (self._obs(), np.ones(self.num_envs, np.float32), done, done,
                {"lane": [{}] * self.num_envs})

    def close(self) -> None:
        self.closed = True

    # Neither kind of state, declared rather than left absent.
    def save_policy_state(self) -> list:
        return []

    def save_run_state(self) -> list:
        return []

    def load_policy_state(self, snapshot) -> None:
        assert not snapshot

    load_run_state = load_policy_state


def test_an_injected_cadence_and_source_drive_the_run(tmp_path, agents):
    env = EnvConfig(name="toy", n_envs=2, kwargs={"episode_len": 3})
    cfg = config(tmp_path, total_frames=8, env=env)
    with Trainer(cfg, FakeDevice(), object(), logger=RecordingLogger(),
                 cadence=CountingCadence, source=ListSource) as trainer:
        trainer.run()
        cadence = trainer.replicas.lead.cadence
        source = trainer.replicas.lead.env
    assert isinstance(source, ListSource) and source.steps == 4
    assert cadence.steps == 8, "the cadence, not the agent, owns the credit"
    assert source.closed, "the teardown ladder closes an injected source"


def test_an_injected_cadence_keeps_its_credit_across_a_resume(tmp_path, agents):
    env = EnvConfig(name="toy", n_envs=2, kwargs={"episode_len": 3})
    first = config(tmp_path / "a", total_frames=8, env=env)
    with Trainer(first, FakeDevice(), object(), logger=RecordingLogger(),
                 cadence=CountingCadence, source=ListSource) as trainer:
        trainer.run()
        before = dict(trainer.replicas.lead.cadence.state_dict())
    path = os.path.join(first.log_dir, LATEST)
    assert load_checkpoint(path)["cadence_state"] == before

    # Not zero: the credit is run state and rides in the file, or the resumed
    # run's replay ratio restarts while its frame count does not.
    second = config(tmp_path / "b", total_frames=16, env=env)
    with Trainer(second, FakeDevice(), object(), logger=RecordingLogger(),
                 resume_path=path, cadence=CountingCadence,
                 source=ListSource) as trainer:
        assert trainer.replicas.lead.cadence.state_dict() == before


def test_a_stateful_cadence_is_refused_a_file_that_carries_none(tmp_path, agents):
    env = EnvConfig(name="toy", n_envs=2, kwargs={"episode_len": 3})
    plain = config(tmp_path / "a", total_frames=8, env=env)
    with Trainer(plain, FakeDevice(), object(), logger=RecordingLogger()) as t:
        t.run()
    path = os.path.join(plain.log_dir, LATEST)
    with pytest.raises(ValueError, match="carries no 'cadence_state'"):
        Trainer(config(tmp_path / "b", total_frames=8, env=env), FakeDevice(),
                object(), logger=RecordingLogger(), resume_path=path,
                cadence=CountingCadence)
    # And the reverse: a file holding credit, resumed with no cadence passed.
    carried = config(tmp_path / "c", total_frames=8, env=env)
    with Trainer(carried, FakeDevice(), object(), logger=RecordingLogger(),
                 cadence=CountingCadence) as t:
        t.run()
    with pytest.raises(ValueError, match="carries 'cadence_state'"):
        Trainer(config(tmp_path / "d", total_frames=8, env=env), FakeDevice(),
                object(), logger=RecordingLogger(),
                resume_path=os.path.join(carried.log_dir, LATEST))


def test_a_stage_that_is_not_the_protocol_is_refused_by_name(tmp_path, agents):
    """Both refusals land in construction, before the run's first frame."""
    with pytest.raises(TypeError, match="is not an UpdateCadence"):
        Trainer(config(tmp_path), FakeDevice(), object(),
                logger=RecordingLogger(), cadence=lambda agent: object())
    with pytest.raises(TypeError, match="is not a ttrl.StepSource"):
        Trainer(config(tmp_path), FakeDevice(), object(),
                logger=RecordingLogger(), source=lambda cfg, index: object())
