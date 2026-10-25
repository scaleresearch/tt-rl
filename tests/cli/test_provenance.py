"""What a run records about itself, and the dirty-tree case in particular.

A run directory held `metrics.jsonl` and `checkpoint.npz`: no config outside
the 27 MB checkpoint, no SHA, no argv, no host. A run killed before its first
checkpoint left a curve nothing could be reproduced from.

The dirty case is the one with teeth. A SHA names code that was not what ran
whenever the tree has uncommitted changes, so the record says so in a line a
human reads, and a test pins that, because a provenance record believed
wrongly is worse than none.
"""

from __future__ import annotations

import json

from ttrl.train.config import TrainConfig
from ttrl.train.provenance import git_state, run_metadata, write_run_metadata


def _config(tmp_path) -> TrainConfig:
    return TrainConfig(algo="dqn", total_frames=100, log_dir=str(tmp_path / "run"))


def test_the_record_carries_the_resolved_config(tmp_path):
    record = run_metadata(_config(tmp_path), argv=["ttrl", "train", "c.py"])
    assert record["config"]["algo"] == "dqn"
    assert record["config"]["env"]["name"]
    assert record["argv"] == ["ttrl", "train", "c.py"]
    assert record["cwd"] and record["host"] and record["python"]


def test_the_record_lands_beside_the_metrics(tmp_path):
    config = _config(tmp_path)
    path = write_run_metadata(config.log_dir, config)
    assert path.name == "config.json"
    assert path.parent == (tmp_path / "run")
    assert json.loads(path.read_text())["config"]["total_frames"] == 100


def test_no_git_is_recorded_and_never_raised(tmp_path):
    state = git_state(tmp_path)
    assert state["available"] is False
    assert state["error"]
