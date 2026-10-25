"""The sinks that need an optional dependency, and the whole set of sinks."""

from __future__ import annotations

import builtins
import sys

import pytest

import ttrl.train.logging as logging_mod
import ttrl.train.trackers as trackers_mod
from ttrl.train.logging import Logger
from ttrl.train.trackers import WandbLogger


class FakeRun:
    def __init__(self, **init_kwargs):
        self.init_kwargs = init_kwargs
        self.logged: list[tuple[dict, int]] = []
        self.finished = 0

    def log(self, metrics, step=None):
        self.logged.append((dict(metrics), step))

    def finish(self):
        self.finished += 1


class FakeWandb:
    def __init__(self):
        self.runs: list[FakeRun] = []

    def init(self, **kwargs):
        run = FakeRun(**kwargs)
        self.runs.append(run)
        return run


@pytest.fixture
def wandb(monkeypatch):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def _refuse(monkeypatch, prefix: str) -> None:
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith(prefix):
            raise ModuleNotFoundError(f"No module named {prefix!r}", name=prefix)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)


def test_scalars_are_logged_at_the_runs_frame_count(wandb):
    sink = WandbLogger("tt-rl")
    sink.log({"train/loss": 0.5, "eval/mean": 3}, 1280)
    sink.close()
    metrics, step = wandb.runs[0].logged[0]
    assert metrics == {"train/loss": 0.5, "eval/mean": 3.0}
    assert step == 1280


def test_non_scalar_and_boolean_values_are_left_out(wandb):
    """A tracker takes scalars; a reason string or a flag is not one."""
    sink = WandbLogger("tt-rl")
    sink.log({"run/stop_reason": "threshold passed", "ok": True}, 10)
    sink.log({"train/loss": 1.0}, 20)
    sink.close()
    assert [m for m, _ in wandb.runs[0].logged] == [{"train/loss": 1.0}]


def test_a_run_handed_in_is_logged_into_and_left_open(wandb):
    """Finishing a run this sink did not start would end it under its owner."""
    run = FakeRun()
    sink = WandbLogger("tt-rl", run=run)
    sink.log({"a": 1.0}, 1)
    sink.close()
    assert wandb.runs == []
    assert run.logged and run.finished == 0


#: Every sink name `ttrl.train.logging` exports, derived from `__all__` rather
#: than typed out: a sink added without a line here fails the first test
#: below. `Logger` itself is the protocol and not a sink.
EXPORTED_SINKS = {
    name for name in logging_mod.__all__
    if name != "Logger" and isinstance(getattr(logging_mod, name), type)
}

#: Which sinks need a package this library does not install, taken from the
#: declaration the sinks themselves build their refusal from. Nothing here
#: restates it, so a sink added there is a case added here.
OPTIONAL = trackers_mod.REQUIRES


def test_every_sink_satisfies_the_logger_protocol():
    for name in sorted(EXPORTED_SINKS):
        assert issubclass(getattr(logging_mod, name), Logger), name


def test_the_sinks_with_an_optional_dependency_are_exactly_the_trackers():
    assert set(trackers_mod.__all__) - {"REQUIRES"} == set(OPTIONAL)
    assert EXPORTED_SINKS >= set(OPTIONAL)


@pytest.mark.parametrize("name", sorted(OPTIONAL))
def test_every_optional_sink_names_its_extra_when_the_package_is_missing(
    name, monkeypatch, tmp_path
):
    """One refusal per sink that can be refused, derived from the module."""
    package, extra = OPTIONAL[name]
    _refuse(monkeypatch, package)
    with pytest.raises(ModuleNotFoundError, match=rf"tt-rl\[{extra}\]"):
        getattr(trackers_mod, name)(str(tmp_path))
