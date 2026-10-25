"""Logging raises; it never degrades to silence."""

from __future__ import annotations

import builtins
import io
import json
from pathlib import Path

import numpy as np
import pytest

from ttrl.train import evaluate as evaluate_mod
from ttrl.train.config import EnvConfig, TrainConfig
from ttrl.train.config.schema import LOG_RECORDS_PER_RUN
import ttrl.train.logging
from ttrl.train.logging import (
    EVAL_PREFIX,
    JsonlLogger,
    MultiLogger,
    StdoutLogger,
    TensorBoardLogger,
)
from ttrl.train.trainer import Trainer


def test_a_missing_tensorboard_raises_and_names_the_extra(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("tensorboard"):
            raise ModuleNotFoundError("No module named 'tensorboard'", name="tensorboard")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(ModuleNotFoundError, match=r"tt-rl\[logging\]"):
        TensorBoardLogger(tmp_path)


def test_jsonl_writes_one_record_per_call(tmp_path):
    logger = JsonlLogger(tmp_path)
    logger.log({"train/loss": 1.5}, 10)
    logger.log({"train/loss": np.float32(2.5)}, 20)
    logger.close()
    lines = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in lines] == [10, 20]
    assert lines[1]["train/loss"] == 2.5


def test_every_record_is_on_disk_before_the_next_one(tmp_path):
    """A crashed run keeps its metric tail, and a mid-run read is not a race."""
    logger = JsonlLogger(tmp_path)
    path = tmp_path / "metrics.jsonl"
    for step in range(5):
        logger.log({"train/loss": float(step)}, step)
        assert len(path.read_text().splitlines()) == step + 1
    logger.close()


def test_multi_logger_fans_out_and_closes_every_sink(tmp_path):
    class Broken:
        def log(self, metrics, step):
            return None

        def flush(self):
            return None

        def close(self):
            raise OSError("broken sink")

    good = JsonlLogger(tmp_path)
    multi = MultiLogger(Broken(), good)
    multi.log({"a": 1}, 0)
    with pytest.raises(OSError, match="broken sink"):
        multi.close()
    assert good._fh.closed
    assert (tmp_path / "metrics.jsonl").read_text().strip()


class _Clock:
    """A monotonic clock the test advances, so throttling is tested not timed."""

    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_stdout_logger_throttles_by_wall_clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(ttrl.train.logging.time, "monotonic", clock)
    buf = io.StringIO()
    logger = StdoutLogger(stream=buf, min_interval_s=5.0)
    logger.log({"a": 1}, 1)  # the first is the sign of life, never throttled
    logger.log({"a": 2}, 2)
    clock.t += 5.0
    logger.log({"a": 3}, 3)
    lines = buf.getvalue().strip().splitlines()
    assert [ln.split("| a ")[-1] for ln in lines] == ["1", "3"]


def test_stdout_logger_refuses_a_stream_as_its_frame_count():
    """`StdoutLogger(sys.stdout)` used to construct and then divide a step by
    a file object about a thousand frames later, inside the sink."""
    buf = io.StringIO()
    with pytest.raises(TypeError, match=r"ttrl.build_logger\(log_dir, total_frames\)"):
        StdoutLogger(buf)
    StdoutLogger(1000, stream=buf).log({"a": 1}, 128)
    assert "frame 128/1000" in buf.getvalue()


def test_stdout_logger_never_throttles_a_halt(monkeypatch):
    """The last thing a halting run says must not be the thing that got dropped."""
    clock = _Clock()
    monkeypatch.setattr(ttrl.train.logging.time, "monotonic", clock)
    buf = io.StringIO()
    logger = StdoutLogger(stream=buf, min_interval_s=5.0)
    logger.log({"a": 1}, 1)
    logger.log({"error": "non-finite loss"}, 2)
    assert "non-finite loss" in buf.getvalue()


def test_a_watched_run_prints_its_eval_score(tmp_path, monkeypatch, agents):
    """What a user watching stdout most wants to see, at the default throttle."""

    result = {"n_episodes": 1, "scores": [7.0], "mean": 7.0, "median": 7.0,
              "min": 7.0, "max": 7.0}
    monkeypatch.setattr(
        evaluate_mod, "evaluate",
        lambda *args, **kwargs: dict(result),
    )
    buf = io.StringIO()
    config = TrainConfig(
        total_frames=8, eval_interval=2, eval_episodes=1,
        checkpoint_interval=1000, log_dir=str(tmp_path / "run"),
        env=EnvConfig(name="toy", kwargs={"episode_len": 3}),
    )
    with Trainer(config, object(), object(),
                 logger=StdoutLogger(config.total_frames, stream=buf)) as trainer:
        trainer.run()
    printed = buf.getvalue()
    expected = [f"{EVAL_PREFIX}{k}" for k in result if k != "scores"]
    assert expected
    missing = [key for key in expected if key not in printed]
    assert not missing, f"{missing} never reached stdout:\n{printed}"
    assert config.best_metric in printed


def test_a_derived_interval_bounds_a_long_run_and_still_shows_a_short_one(
    tmp_path, monkeypatch, agents
):
    """`TrainConfig.log_interval` thins the per-step records only.

    An eval score and an episode end are what the file is read for and there
    are hundreds of them; the update records are the 385 MB. Left unset the
    spacing comes off the run's length, so the long run is bounded and the
    short run still writes its loss; set by hand it is that spacing, with the
    run's first update record written either way.
    """

    monkeypatch.setattr(
        evaluate_mod, "evaluate",
        lambda *args, **kwargs: {"n_episodes": 1, "scores": [7.0], "mean": 7.0,
                                 "median": 7.0, "min": 7.0, "max": 7.0},
    )

    def config(name, **kwargs):
        return TrainConfig(
            eval_interval=4, eval_episodes=1, checkpoint_interval=1000,
            log_dir=str(tmp_path / name),
            env=EnvConfig(name="toy", kwargs={"episode_len": 3}),
            **kwargs,
        )

    def written(cfg):
        log_dir = Path(cfg.log_dir)
        with Trainer(cfg, object(), object(),
                     logger=JsonlLogger(log_dir)) as trainer:
            trainer.run()
        records = (json.loads(line) for line
                   in (log_dir / "metrics.jsonl").read_text().splitlines())
        return [set(r) - {"step", "wall_time"} for r in records]

    # Stored as declared, so an override that shortens the run re-derives
    # instead of carrying the old length's spacing into it.
    long_run = config("long", total_frames=1_000_000)
    assert long_run.log_interval is None
    assert long_run.log_interval_frames == 1_000_000 // LOG_RECORDS_PER_RUN
    assert config("tiny", total_frames=12).log_interval_frames == 1
    assert config("explicit", total_frames=12, log_interval=7).log_interval_frames == 7

    derived = written(config("derived", total_frames=12))
    updates = [k for k in derived if all(n.startswith("train/") for n in k)]
    assert len(updates) > 1, "a run shorter than the interval showed no loss"

    stated = written(config("stated", total_frames=12, log_interval=1_000_000))
    kept = [k for k in stated if all(n.startswith("train/") for n in k)]
    assert len(kept) == 1, "only the first update record survives the interval"
    for keys in (derived, stated):
        assert [k for k in keys if any(n.startswith("eval/") for n in k)]
        assert [k for k in keys if "rollout/episode_return" in k]
