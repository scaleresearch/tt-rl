"""Resource release on the failure paths, the ones no happy-path test walks."""

from __future__ import annotations

import fcntl
import importlib

import pytest
import ttnn

from tests._fakes.policy_state import FakeDevice
from tests._fakes.train_layer import RecordingLogger
import ttrl.engine.device as device_mod
import ttrl.engine.device.liveness as device_liveness_mod
import ttrl.engine.mesh as mesh_mod
import ttrl.engine.mesh.devices as mesh_devices_mod
from ttrl.engine.nn.discovery import serialises_own_state, slots_from
from ttrl.train import builders, replicas as replicas_mod
from ttrl.train.config import EnvConfig, TrainConfig
from ttrl.train.trainer import Trainer

# `ttrl.engine.device` re-exports `acquire`, so `import
# ttrl.engine.device.acquire as m` binds the function rather than the module
# it lives in. import_module reaches the module.
device_acquire_mod = importlib.import_module("ttrl.engine.device.acquire")


class _Boom(RuntimeError):
    """The failure under test, distinguishable from any incidental error."""


@pytest.fixture
def fake_chip(monkeypatch, tmp_path):
    """One always-free chip whose open/close is recorded, no hardware."""

    free = lambda cid: device_mod.ChipStatus(  # noqa: E731
        cid, f"/dev/tenstorrent/{cid}", (), None
    )
    monkeypatch.setenv("TTRL_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(device_liveness_mod, "enumerate_chips", lambda: [0])
    monkeypatch.setattr(device_liveness_mod, "physical_chip_id", lambda cid: cid)
    monkeypatch.setattr(device_liveness_mod, "status", free)
    # `acquire` does `from ...liveness import enumerate_chips, status`, so it
    # holds its OWN references and patching liveness alone leaves it calling
    # the real ones. These two tests then enumerated real chips and blocked
    # for the full acquire timeout waiting for one, which is why a host-only
    # file needed hardware.
    monkeypatch.setattr(device_acquire_mod, "enumerate_chips", lambda: [0])
    monkeypatch.setattr(device_acquire_mod, "status", free)

    class _Device:
        def enable_program_cache(self):
            pass

    closed: list = []
    device = _Device()
    monkeypatch.setattr(ttnn, "open_device", lambda **kw: device)
    monkeypatch.setattr(ttnn, "close_device", closed.append)
    monkeypatch.setattr(
        device_acquire_mod, "probe_profile", lambda dev, cfg: "profile-sentinel"
    )
    return device, closed


def test_acquire_closes_the_device_even_when_the_cache_purge_raises(
    monkeypatch, fake_chip
):
    device, closed = fake_chip
    monkeypatch.setattr(
        device_acquire_mod,
        "purge_device_caches",
        lambda dev: (_ for _ in ()).throw(_Boom("purge")),
    )

    with pytest.raises(_Boom):
        with device_mod.acquire(chip_id=0):
            pass

    assert closed == [device], "a failed purge must not strand the chip open"


def test_acquire_purge_failure_does_not_mask_the_bodys_exception(monkeypatch, fake_chip):
    device, closed = fake_chip
    monkeypatch.setattr(
        device_acquire_mod,
        "purge_device_caches",
        lambda dev: (_ for _ in ()).throw(RuntimeError("purge")),
    )

    with pytest.raises(_Boom):  # the body's error, not the purge's
        with device_mod.acquire(chip_id=0):
            raise _Boom("the real failure")

    assert closed == [device]


def test_hold_lock_releases_the_flock_even_when_erasing_the_pid_fails(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TTRL_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(device_liveness_mod, "physical_chip_id", lambda cid: cid)

    class _TruncateFails:
        """A real file handle (flock needs a live fd) whose *second* truncate fails."""

        def __init__(self, fh):
            self._fh = fh
            self._truncates = 0

        def __getattr__(self, name):
            return getattr(self._fh, name)

        def truncate(self, *a):
            self._truncates += 1
            if self._truncates > 1:
                raise _Boom("no space left on device")
            return self._fh.truncate(*a)

    real_open = device_liveness_mod.open if hasattr(device_liveness_mod, "open") else open
    monkeypatch.setattr(
        device_liveness_mod, "open", lambda *a, **kw: _TruncateFails(real_open(*a, **kw)),
        raising=False,
    )

    ops: list[int] = []
    real_flock = fcntl.flock
    monkeypatch.setattr(
        fcntl, "flock", lambda fh, op: (ops.append(op), real_flock(fh, op))[1]
    )

    with pytest.raises(_Boom):
        with device_liveness_mod.hold_lock(0) as got:
            assert got

    assert fcntl.LOCK_UN in ops, "the unlock must not be skipped by a bookkeeping error"


def test_multi_device_open_closes_every_device_when_a_purge_raises(monkeypatch):
    class _Device:
        def enable_program_cache(self):
            pass

    devices = {0: _Device(), 1: _Device()}
    closed: list = []
    monkeypatch.setattr(mesh_devices_mod, "require_exact_visible_scope", lambda chips: None)
    monkeypatch.setattr(mesh_devices_mod, "local_ids_for", lambda chips: [0, 1])
    monkeypatch.setattr(mesh_devices_mod, "lock_chips", lambda stack, ids: None)
    monkeypatch.setattr(ttnn, "CreateDevices", lambda ids, **kw: devices)
    monkeypatch.setattr(ttnn, "CloseDevices", closed.append)
    monkeypatch.setattr(
        mesh_devices_mod,
        "purge_device_caches",
        lambda dev: (_ for _ in ()).throw(_Boom("purge")),
    )

    with pytest.raises(_Boom):
        with mesh_mod.open_independent_devices_for_chips([0, 1]):
            pass

    assert closed == [devices], "one failed purge must not leak all N chips"


def test_the_parameter_walk_resolves_module_with_no_device_open():
    """`discovery` names `Module` through a deferred import, because `module`
    imports the walk at its own scope. A missing import here is invisible to
    every happy path: the two branches that need the class are only reached by
    a value that is not a Tensor, which is what a Polyak target sync passes."""

    assert serialises_own_state(object()) is False
    assert list(slots_from("x", object(), set())) == []


class _OpenThing:
    """An agent or an environment that records its own release."""

    obs_spec = None
    action_spec = None

    def __init__(self) -> None:
        self.closes = 0

    def close(self) -> None:
        self.closes += 1


def test_a_refused_trainer_releases_the_replicas_it_already_opened(monkeypatch,
                                                                   tmp_path):
    """A refusal raised AFTER `open_replicas` must not strand a chip.

    Construction opens agents, environment workers and, above one chip, the
    run's devices, and then keeps validating. Every refusal past that point
    used to close the logger alone, so the chip stayed held: ttnn leaves fds
    on /dev/tenstorrent/<n> that outlive close_device, and a leaked agent
    makes a chip look busy to every later process. `checkpoint_replay` against
    an agent that owns no replay buffer is the cheapest such refusal; the
    resume-without-transitions one in tests/train/test_replay_resume.py is the
    other.
    """

    agents: list[_OpenThing] = []
    envs: list[_OpenThing] = []
    released: list[bool] = []
    real_build_env = builders.build_env

    def build_env(cfg):
        built = real_build_env(cfg)
        env = _OpenThing()
        env.obs_spec, env.action_spec = built.obs_spec, built.action_spec
        env.save_policy_state = built.save_policy_state
        env.close = lambda: (built.close(), _bump(env))[0]
        envs.append(env)
        return env

    def build_agent(cfg, *, obs_spec, action_spec, device, profile=None,
                    algo_cfg=None):
        agent = _OpenThing()
        agent.obs_spec, agent.action_spec = obs_spec, action_spec
        agents.append(agent)
        return agent

    def open_replicas(cfg, device, profile, **kwargs):
        opened = replicas_mod.open_replicas(cfg, device, profile, **kwargs)
        release = opened.release_devices
        opened.release_devices = lambda: (released.append(True), release())[1]
        return opened

    monkeypatch.setattr(builders, "build_env", build_env)
    monkeypatch.setattr(builders, "build_agent", build_agent)
    monkeypatch.setattr("ttrl.train.trainer.open_replicas", open_replicas)

    logger = RecordingLogger()
    with pytest.raises(ValueError, match="checkpoint_replay"):
        Trainer(
            TrainConfig(
                total_frames=8,
                checkpoint_replay=True,
                log_dir=str(tmp_path / "run"),
                env=EnvConfig(name="toy", n_envs=1, kwargs={"episode_len": 3}),
            ),
            FakeDevice(), object(), logger=logger,
        )

    assert [a.closes for a in agents] == [1], "the agent holds a captured trace"
    assert [e.closes for e in envs] == [1], "the env may hold worker processes"
    assert released == [True], "the devices are the chip itself"
    assert logger.closed == 1


def _bump(thing) -> None:
    thing.closes += 1
