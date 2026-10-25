"""`run(config)`: the one-call entry point, and the replica wiring it drives."""

from __future__ import annotations

import contextlib

import pytest

from tests._fakes.train_layer import RecordingLogger, ScriptedAgent
import ttrl.engine
from ttrl.engine.device import DEFAULT_WAIT_SECONDS
from ttrl.train import builders, run as run_mod, replicas as replicas_mod
from ttrl.train.config import DeviceConfig, EnvConfig, TrainConfig


def config(tmp_path, **kwargs):
    base = dict(
        total_frames=6,
        eval_interval=1000,
        checkpoint_interval=1000,
        log_dir=str(tmp_path / "run"),
        env=EnvConfig(name="toy", kwargs={"episode_len": 3}),
    )
    return TrainConfig(**{**base, **kwargs})


@pytest.fixture
def agents(monkeypatch):
    built = []

    def build_agent(cfg, *, obs_spec, action_spec, device, profile=None, algo_cfg=None):
        agent = ScriptedAgent(obs_spec, action_spec)
        agent.device = device
        built.append(agent)
        return agent

    monkeypatch.setattr(builders, "build_agent", build_agent)
    return built


def test_one_chip_scopes_then_acquires_and_releases(tmp_path, agents, monkeypatch):
    acquired = []
    scoped = []

    @contextlib.contextmanager
    def acquire(chip_id=None, *, cfg=None, **kwargs):
        acquired.append((chip_id, cfg))
        yield ("device", "profile")
        acquired.append("released")

    monkeypatch.setattr(run_mod, "acquire", acquire)
    monkeypatch.setattr(run_mod, "scope_for_run", lambda dc: scoped.append(dc.chip))
    result = run_mod.run(
        config(tmp_path, device=DeviceConfig(chip=2, l1_small_size=4096)),
        logger=RecordingLogger(),
    )
    assert result["frame_count"] == 6
    assert scoped == [2]
    assert acquired[0][0] is None
    assert acquired[0][1].l1_small_size == 4096
    assert acquired[-1] == "released"


def test_several_chips_let_the_trainer_open_its_own(tmp_path, agents, monkeypatch):
    """No handle is acquired here: it would hold a chip a replica wants."""

    def refuse(*args, **kwargs):
        raise AssertionError("run() acquired a device on the multi-chip path")

    monkeypatch.setattr(run_mod, "acquire", refuse)
    monkeypatch.setattr(
        replicas_mod, "_devices", lambda cfg, device, profile, stack: (["d0", "d1"], [None, None])
    )
    monkeypatch.setattr(replicas_mod, "_param_sync", lambda cfg, reps: None)
    result = run_mod.run(
        config(tmp_path, device=DeviceConfig(chips=[0, 1])), logger=RecordingLogger()
    )
    assert result["replicas"] == 2
    assert result["frame_count"] == 6


def test_averaging_runs_once_per_updating_round(tmp_path, agents, monkeypatch):
    calls = []

    class FakeSync:
        def maybe_average(self, targets=None):
            calls.append("maybe")
            return True

        def average(self, targets=None):
            calls.append("average")

    monkeypatch.setattr(
        replicas_mod, "_devices", lambda cfg, device, profile, stack: (["d0", "d1"], [None, None])
    )
    monkeypatch.setattr(replicas_mod, "_param_sync", lambda cfg, reps: FakeSync())
    result = run_mod.run(
        config(tmp_path, total_frames=6, device=DeviceConfig(chips=[0, 1])),
        logger=RecordingLogger(),
    )
    assert result["frame_count"] == 6
    assert calls == ["maybe", "maybe", "maybe", "average"]


def test_resume_path_in_the_config_is_honoured(tmp_path, agents, monkeypatch):
    @contextlib.contextmanager
    def acquire(chip_id=None, *, cfg=None, **kwargs):
        yield ("device", "profile")

    monkeypatch.setattr(run_mod, "acquire", acquire)
    cfg = config(tmp_path, checkpoint_interval=6)
    run_mod.run(cfg, logger=RecordingLogger())
    path = str(tmp_path / "run" / "checkpoint.npz")

    resumed = config(tmp_path, total_frames=12, resume_path=path)
    result = run_mod.run(resumed, logger=RecordingLogger())
    assert result["frame_count"] == 12
    assert agents[1].updates == 6


def test_an_incomplete_agent_is_refused_at_construction(tmp_path, monkeypatch):
    """`Agent` is a Protocol, so a missing method must be caught here."""
    class Incomplete:
        obs_spec = None
        action_spec = None

        def act(self, obs, *, deterministic=False):
            return None

    monkeypatch.setattr(
        builders, "resolve_and_build",
        lambda name, obs_spec, action_spec, device, **kwargs: Incomplete(),
    )
    with pytest.raises(TypeError, match="does not expose obs_spec"):
        builders.build_agent(config(tmp_path), obs_spec=None, action_spec=None,
                             device=None)


def test_the_acquire_timeout_comes_from_the_config(tmp_path, agents, monkeypatch):
    seen = {}

    @contextlib.contextmanager
    def acquire(chip_id=None, *, cfg=None, timeout=None, **kwargs):
        seen["timeout"] = timeout
        yield ("device", "profile")

    monkeypatch.setattr(run_mod, "acquire", acquire)
    monkeypatch.setattr(run_mod, "scope_for_run", lambda dc: None)
    run_mod.run(
        config(tmp_path, device=DeviceConfig(acquire_timeout=1800.0)),
        logger=RecordingLogger(),
    )
    assert seen["timeout"] == 1800.0


def test_the_default_wait_matches_the_engines_own(tmp_path):
    """`ttrl.train.config` must not import the engine, so the default is
    restated here. This is what stops the two drifting apart."""

    assert DeviceConfig().acquire_timeout == DEFAULT_WAIT_SECONDS



def _two_chips(monkeypatch):
    monkeypatch.setattr(
        replicas_mod, "_devices",
        lambda cfg, device, profile, stack: (["d0", "d1"], [None, None]),
    )
    monkeypatch.setattr(replicas_mod, "_param_sync", lambda cfg, reps: None)


def test_an_injected_agent_adapter_reaches_every_replica(tmp_path, agents,
                                                         monkeypatch):
    """The seam a config cannot carry: `PPO(boundaries=...)` from `run()`.

    The adapter is handed what the algorithm registry built, one per replica
    and with that replica's index, so it composes with `ALGOS` rather than
    standing in for it.
    """
    _two_chips(monkeypatch)
    seen = []

    def adapt(built, index):
        seen.append((built, index))
        built.boundaries = f"boundaries-{index}"
        return built

    result = run_mod.run(
        config(tmp_path, device=DeviceConfig(chips=[0, 1])),
        logger=RecordingLogger(), agent=adapt,
    )
    assert result["replicas"] == 2
    assert [index for _built, index in seen] == [0, 1]
    assert [built for built, _index in seen] == agents, (
        "the adapter must be handed the agent the registry built, not a "
        "second one built beside it"
    )
    assert [a.boundaries for a in agents] == ["boundaries-0", "boundaries-1"]


def test_an_agent_adapter_that_is_not_one_is_refused_by_name(tmp_path, agents,
                                                             monkeypatch):
    """Every refusal before the first frame, and none of them leaks an agent."""
    _two_chips(monkeypatch)
    cfg = lambda: config(tmp_path, device=DeviceConfig(chips=[0, 1]))

    with pytest.raises(TypeError, match="is not callable"):
        run_mod.run(cfg(), logger=RecordingLogger(), agent=object())
    with pytest.raises(TypeError, match="returned None for replica 0"):
        run_mod.run(cfg(), logger=RecordingLogger(),
                    agent=lambda built, index: None)
    with pytest.raises(TypeError, match="is not a ttrl.Agent"):
        run_mod.run(cfg(), logger=RecordingLogger(),
                    agent=lambda built, index: object())

    def raising(built, index):
        raise RuntimeError("my adapter is broken")

    with pytest.raises(RuntimeError, match="my adapter is broken"):
        run_mod.run(cfg(), logger=RecordingLogger(), agent=raising)
    # The built agent holds staging on an open chip, so a failed adaptation
    # has to release the agent it was handed.
    assert all(agent.closes >= 1 for agent in agents), agents
