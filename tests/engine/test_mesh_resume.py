"""Resuming a run at a different chip count."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from ttrl.engine.mesh import reset_optimizer_state, resume_replicas

from tests._fakes import FakeAdam, FakeModule, FakeTensor
import ttrl.engine.mesh.resume as resume_mod

STATE = {"p0": np.array([9.0], dtype=np.float32)}


class _StubAdam:
    """Adam's checkpointed counters."""

    def __init__(self, n_params, shape=(4,)):
        self.t = 5
        self.steps = [7 + i for i in range(n_params)]
        self.lr, self.b1, self.b2, self.eps, self.weight_decay = 1e-4, 0.9, 0.999, 1e-8, 0.0
        self.m = [FakeTensor(np.full(shape, 1.0)) for _ in range(n_params)]
        self.v = [FakeTensor(np.full(shape, 2.0)) for _ in range(n_params)]


@pytest.fixture(autouse=True)
def _fake_load(monkeypatch):
    def load_state_dict(self, state, device):
        self._params = [FakeTensor(state[f"p{i}"]) for i in range(len(state))]

    monkeypatch.setattr(FakeModule, "load_state_dict", load_state_dict, raising=False)


def _stub_ttnn(monkeypatch):
    mod = types.ModuleType("ttnn")
    mod.zeros_like = lambda t: FakeTensor(np.zeros_like(t.numpy()))

    def _add(a, scalar, output_tensor=None):
        output_tensor._array[:] = a.numpy() + scalar
        return output_tensor

    def _mul(a, scalar, output_tensor=None):
        output_tensor._array[:] = a.numpy() * scalar
        return output_tensor

    mod.add, mod.mul = _add, _mul
    monkeypatch.setitem(sys.modules, "ttnn", mod)


def _fake_reset(monkeypatch):
    def reset(optimizers):
        for opt in optimizers:
            opt.t = 0
            for buf in opt.m + opt.v:
                buf._array[:] = 0.0

    monkeypatch.setattr(resume_mod, "reset_optimizer_state", reset)


def test_resume_same_n_keeps_optimizer_state():
    m1, m2 = FakeModule([[1.0]]), FakeModule([[2.0]])
    opt1, opt2 = FakeAdam(1), FakeAdam(1)
    opt1.m[0]._array[:] = 42.0

    resume_replicas([m1, m2], [opt1, opt2], STATE, prev_n=2, new_n=2, devices=[None, None])

    assert m1.parameters()[0].numpy()[0] == 9.0
    assert m2.parameters()[0].numpy()[0] == 9.0
    assert np.all(opt1.m[0].numpy() == 42.0)


def test_resume_at_a_new_n_resets_every_replica_and_propagates_lr(monkeypatch):
    _fake_reset(monkeypatch)
    m1, m2 = FakeModule([[1.0]]), FakeModule([[1.0]])
    opt1, opt2 = FakeAdam(1, lr=5e-4), FakeAdam(1, lr=9e-3)
    opt1.t, opt2.t = 77, 88
    opt1.m[0]._array[:] = 42.0
    opt2.m[0]._array[:] = 43.0

    resume_replicas([m1, m2], [opt1, opt2], STATE, prev_n=4, new_n=2, devices=[None, None])

    for m in (m1, m2):
        assert m.parameters()[0].numpy()[0] == 9.0
    for opt in (opt1, opt2):
        assert opt.t == 0
        assert np.all(opt.m[0].numpy() == 0.0)
        assert opt.lr == pytest.approx(5e-4)


def test_resume_rejects_mismatched_hyperparameters_before_mutating(monkeypatch):
    _fake_reset(monkeypatch)
    m1, m2 = FakeModule([[1.0]]), FakeModule([[1.0]])
    opt1 = FakeAdam(1, lr=5e-4, weight_decay=0.0)
    opt2 = FakeAdam(1, lr=9e-3, weight_decay=1e-4)
    opt1.t, opt2.t = 77, 88

    with pytest.raises(ValueError, match="optimizer hyperparameters"):
        resume_replicas([m1, m2], [opt1, opt2], STATE, prev_n=4, new_n=2, devices=[None, None])

    for m in (m1, m2):
        assert m.parameters()[0].numpy()[0] == 1.0
    assert opt1.t == 77 and opt2.t == 88
    assert opt1.lr == pytest.approx(5e-4) and opt2.lr == pytest.approx(9e-3)


def test_reset_optimizer_state_zeroes_moments_and_both_step_counters(monkeypatch):
    """`steps` is not derivable from `t`: leaving it advanced skips the bias correction on resume."""
    _stub_ttnn(monkeypatch)
    opt = _StubAdam(3)
    reset_optimizer_state([opt])

    assert opt.t == 0
    assert opt.steps == [0, 0, 0]
    for buf in opt.m + opt.v:
        assert np.all(buf.numpy() == 0.0)


def test_resume_at_a_new_chip_count_resets_per_parameter_steps(monkeypatch):
    """End to end through `resume_replicas` with the real reset."""
    _stub_ttnn(monkeypatch)
    m1, opt1 = FakeModule([[1.0]]), _StubAdam(1)
    resume_replicas([m1], [opt1], STATE, prev_n=2, new_n=4, devices=[None])
    assert opt1.t == 0 and opt1.steps == [0]
