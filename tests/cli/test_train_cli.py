"""`ttrl train` gives a run its sinks, its config, and a quiet stack.

The command had been calling `ttrl.run(config)` with no logger, so a run got
the library default (a JSONL file) and printed one line in six thousand
frames, inside ttnn's own log stream. Both halves are asserted here because
both were invisible: a missing sink and a missing provenance file look
exactly like a working run until someone needs them. The provenance file
itself is now written by `ttrl.run` and asserted there; what is left here is
that this command hands over the config it says it does.

No device, no ttnn: `ttrl.run` is replaced, which is also what proves the
command passes what it says it passes.
"""

from __future__ import annotations

import os
import sys

from loguru import logger
import pytest

import ttrl
from ttrl.cli import build_parser, logs, train as train_cli
from ttrl.train.checkpoint import best_path
from ttrl.train.config import TrainConfig
from ttrl.train.logging import JsonlLogger, MultiLogger, StdoutLogger


@pytest.fixture
def quiet(monkeypatch):
    """Stop the command reconfiguring this process's loggers."""
    called = []
    # Imported by name into the command module, which is what it calls.
    monkeypatch.setattr(train_cli, "quiet_stack_logs", called.append)
    return called


@pytest.fixture
def captured_run(monkeypatch, tmp_path):
    """`ttrl.run` replaced by a recorder, with a config pointed at tmp_path."""
    config = TrainConfig(algo="dqn", total_frames=500, log_dir=str(tmp_path / "run"))
    seen = {}

    def fake_run(cfg, *, resume=None, logger=None, hooks=None):
        seen["config"] = cfg
        seen["resume"] = resume
        seen["logger"] = logger
        seen["hooks"] = hooks
        return {"frame_count": 500, "updates": 7, "replicas": 1,
                "status": "completed", "stop_reason": None,
                "best": {"best_metric": "eval/mean", "best_mode": "max",
                         "best_score": None, "best_frame": None}}

    monkeypatch.setattr(ttrl, "load_config", lambda path, overrides=None: config)
    monkeypatch.setattr(ttrl, "run", fake_run)
    seen["expected_config"] = config
    return seen


def _train(argv):
    args = build_parser().parse_args(["train", *argv])
    return args._run(args)


def test_a_run_gets_the_file_and_the_screen(quiet, captured_run, tmp_path, capsys):
    assert _train(["c.py"]) == 0
    logger = captured_run["logger"]
    assert isinstance(logger, MultiLogger)
    kinds = [type(sink) for sink in logger.sinks]
    assert kinds == [JsonlLogger, StdoutLogger]
    # The progress line carries the run's horizon, so it can show a percentage
    # and an implied finish time.
    assert logger.sinks[1].total_frames == 500
    assert (tmp_path / "run" / "metrics.jsonl").is_file()
    assert "training finished at frame 500" in capsys.readouterr().out


def test_the_quiet_level_reaches_the_log_setup(quiet, captured_run):
    _train(["c.py", "--ttnn-logs", "debug"])
    assert quiet == ["debug"]


def test_a_positional_that_is_not_an_override_is_refused(quiet, captured_run):
    with pytest.raises(SystemExit) as exc:
        _train(["c.py", "algo_cfg.lr"])
    assert "algo_cfg.lr=5e-5" in str(exc.value)


def test_quiet_stack_logs_sets_both_loggers(monkeypatch):
    """The C++ side reads its level from the environment once, at load; the
    Python side is loguru, which is reconfigured in place."""

    monkeypatch.delenv("TT_LOGGER_LEVEL", raising=False)
    monkeypatch.delenv("TT_METAL_LOGGER_LEVEL", raising=False)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    try:
        logs.quiet_stack_logs("warning")
        assert os.environ["TT_LOGGER_LEVEL"] == "warning"
        assert os.environ["TT_METAL_LOGGER_LEVEL"] == "warning"
    finally:
        logger.remove()
        logger.add(sys.stderr)


# -- the flags the library half just gained ---------------------------------


def test_both_stop_flags_are_asked_in_order(quiet, captured_run):
    """`AnyOf`, so the recorded reason names one condition and not a conjunction."""
    _train(["c.py", "--stop-reward", "18", "--stop-patience", "4"])
    stop = captured_run["hooks"].stop
    assert isinstance(stop, ttrl.AnyOf)
    assert [type(c) for c in stop.conditions] == [
        ttrl.StopOnRewardThreshold, ttrl.StopOnNoImprovement
    ]


def test_a_stopped_run_says_so_and_says_why(quiet, captured_run, capsys):
    captured_run["result"] = None
    train_cli.report_outcome({
        "status": "stopped",
        "stop_reason": "eval/mean is 18.2 at frame 400000, which passes the max "
                       "threshold 18",
        "best": {"best_metric": "eval/mean", "best_score": 18.2,
                 "best_frame": 400_000},
    })
    out = capsys.readouterr().out
    assert "stopped early" in out and "passes the max threshold 18" in out
    assert "checkpoint-best.npz" in out


def test_resume_best_resolves_to_the_run_directory_file(quiet, captured_run, tmp_path):
    path = best_path(str(tmp_path / "run"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()
    _train(["c.py", "--resume", "best"])
    assert captured_run["resume"] == path


def test_resume_best_with_no_best_file_names_the_path_it_wanted(quiet, captured_run):
    with pytest.raises(SystemExit) as exc:
        _train(["c.py", "--resume", "best"])
    assert "checkpoint-best.npz" in str(exc.value)
