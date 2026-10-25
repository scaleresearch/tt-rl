"""`POLICY_STATE` and `RUN_STATE`: which of a layer's state travels, and where."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import numpy as np
import pytest

import ttrl.envs
from ttrl.envs.vec_normalize import NormalizeObservation, NormalizeReward
from ttrl.envs.vec_wrapper import VecWrapper
from ttrl.envs.vector_async import AsyncVectorEnv
from ttrl.envs.wrappers import (
    RecordEpisodeStatistics,
    TimeLimit,
    Wrapper,
    restore_policy_state,
    snapshot_env,
    snapshot_policy_state,
)
from ttrl.spaces import ActionSpec, ObsSpec


class Ramp:
    """Observations grow, so the running statistics are never the defaults."""

    obs_spec = ObsSpec.vector(3)
    action_spec = ActionSpec.discrete(2)

    def __init__(self):
        self.t = 0

    def reset(self, *, seed=None):
        self.t = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        obs = np.full(3, float(self.t), dtype=np.float32)
        done = self.t % 5 == 0
        info = {"episode": {"r": float(self.t), "l": self.t}} if done else {}
        return obs, 1.0, done, False, info

    def close(self):
        pass


class Whitener(Wrapper):
    """A per-lane bearer, standing in for the real one while the declaration
    machinery is tested. The real normaliser is one estimator above every
    lane, so a per-lane copy of it is the defect rather than the fixture."""

    POLICY_STATE = ("mean", "count")

    def __init__(self, env):
        super().__init__(env)
        self._mean = np.zeros(env.obs_spec.shape, dtype=np.float64)
        self._count = 0.0

    def _update(self, obs):
        self._count += 1.0
        self._mean += (obs - self._mean) / self._count
        return (obs - self._mean).astype(np.float32)

    def reset(self, *, seed=None):
        obs, info = self.env.reset(seed=seed)
        return self._update(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._update(obs), reward, terminated, truncated, info

    def get_state(self):
        return {"mean": self._mean.copy(), "count": self._count}

    def set_state(self, state):
        self._mean[:] = state["mean"]
        self._count = float(state["count"])


def make_stack():
    return Whitener(TimeLimit(RecordEpisodeStatistics(Ramp()), 20))


def _warm(env, steps=4):
    env.reset()
    for _ in range(steps):
        env.step(0)
    return env


def test_only_the_transform_layer_is_in_the_policy_snapshot():
    env = _warm(make_stack())
    assert [name for name, _ in snapshot_env(env)] == [
        "Whitener", "TimeLimit", "RecordEpisodeStatistics",
    ]
    assert [name for name, _ in snapshot_policy_state(env)] == ["Whitener"]


def test_it_restores_into_a_stack_that_differs_outside_the_transform():
    """An eval or export stack is rarely wrapped identically, and the
    transform is the part that has to travel."""
    trained = _warm(make_stack())
    fresh = Whitener(Ramp())
    restore_policy_state(fresh, snapshot_policy_state(trained))
    obs = np.full(3, 2.0, dtype=np.float32)
    assert np.allclose(trained._mean, fresh._mean)
    assert trained._count == fresh._count
    assert np.allclose(trained._update(obs.copy()), fresh._update(obs.copy()))


def test_restoring_leaves_the_layer_s_other_state_alone():
    """The saved keys are merged over current state, so a layer holding both
    kinds keeps its trajectory half."""

    class Mixed(Whitener):
        POLICY_STATE = ("mean", "count")

        def get_state(self):
            return {**super().get_state(), "steps": self.steps}

        def set_state(self, state):
            super().set_state(state)
            self.steps = int(state["steps"])

    env = Mixed(Ramp())
    env.steps = 0
    env.reset()
    env.step(0)
    env.steps = 7
    snap = snapshot_policy_state(env)
    assert [sorted(items) for _, items in snap] == [["count", "mean"]]
    other = Mixed(Ramp())
    other.steps = 99
    restore_policy_state(other, snap)
    assert other.steps == 99 and other._count == env._count


class Undeclared:
    """A stateful layer whose author never answered the question."""

    obs_spec = ObsSpec.vector(3)
    action_spec = ActionSpec.discrete(2)

    def reset(self, *, seed=None):
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(3, dtype=np.float32), 0.0, False, False, {}

    def get_state(self):
        return {"whatever": 1}

    def set_state(self, state):
        pass

    def close(self):
        pass


def test_a_stateful_layer_that_declares_nothing_is_refused():
    """Neither default is safe: dropping a transform silently kills an
    exported policy, and carrying everything puts an opaque blob in the file."""
    with pytest.raises(TypeError, match="POLICY_STATE"):
        snapshot_policy_state(Undeclared())
    with pytest.raises(TypeError, match="POLICY_STATE"):
        restore_policy_state(Undeclared(), [])


def shipped_layers():
    """Every STATEFUL env layer class this library ships, found by import
    rather than listed.
    """

    classes: dict[str, type] = {}
    for info in pkgutil.walk_packages(ttrl.envs.__path__, "ttrl.envs."):
        try:
            module = importlib.import_module(info.name)
        except ModuleNotFoundError as exc:
            raise AssertionError(
                f"{info.name} cannot be imported because its dependency "
                f"{exc.name!r} is not installed"
            ) from exc
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if not obj.__module__.startswith("ttrl.envs"):
                continue
            if obj in (Wrapper, VecWrapper) or not hasattr(obj, "get_state"):
                continue
            if not hasattr(obj, "step"):
                continue  # an estimator, not a layer in a stack
            if "get_state" not in obj.__dict__ and not issubclass(
                    obj, VecWrapper):
                continue  # inherits a pass-through, declares with its parent
            classes[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return classes


def test_every_shipped_layer_declares_what_its_state_is():
    """Enumerated from the package, so a layer added later is covered here on
    the day it is added rather than on the day someone edits a list.
    """
    for name, cls in sorted(shipped_layers().items()):
        declared = getattr(cls, "POLICY_STATE", None)
        assert declared is not None, f"{name} declares no POLICY_STATE"
        assert isinstance(declared, tuple), name
        if issubclass(cls, VecWrapper):
            run = getattr(cls, "RUN_STATE", None)
            assert run is not None, f"{name} declares no RUN_STATE"
            assert isinstance(run, tuple), name

    assert NormalizeObservation.POLICY_STATE == (
        "mean", "m2", "count", "clip", "epsilon")
    assert NormalizeObservation.RUN_STATE == ()
    assert NormalizeReward.POLICY_STATE == ()
    assert NormalizeReward.RUN_STATE == (
        "mean", "m2", "count", "gamma", "clip", "epsilon")


def test_an_async_vector_env_selects_in_the_child():
    """The declaration lives on the class, which only the child holds."""
    with AsyncVectorEnv([make_stack] * 2, context="fork") as vec:
        vec.reset(seed=0)
        for _ in range(4):
            vec.step(np.zeros(2, dtype=np.int64))
        snap = vec.save_policy_state()
        assert [name for name, _ in snap] == ["Whitener"]
        vec.load_policy_state(snap)
        assert [name for name, _ in vec.save_policy_state()] == ["Whitener"]
