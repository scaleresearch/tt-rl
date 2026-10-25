"""What a checkpoint reader is allowed to invent, and what it must refuse."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ttrl.spaces import ActionSpec, ObsSpec
from ttrl.train.checkpoint import L1, load_checkpoint, save_checkpoint
from ttrl.train.checkpoint.read import read_meta
from ttrl.train.checkpoint.tree import (
    CheckpointFieldError,
    defaulted_fields,
    join,
    split,
)
from ttrl.train.config import DeviceConfig, EnvConfig, TrainConfig

SCHEMA = "ttrl.train.config.schema"


def _write(path, config) -> str:
    save_checkpoint(
        str(path),
        agent_state={"w": np.zeros(3, dtype=np.float32)},
        obs_spec=ObsSpec.vector(3),
        action_spec=ActionSpec.discrete(2),
        level=L1,
        frame=128,
        config=config,
        env_policy_state=[],
    )
    return str(path)


def _config(**over) -> TrainConfig:
    base = dict(algo="dqn", total_frames=64, log_dir="runs/x",
                env=EnvConfig(name="atari", n_envs=1),
                device=DeviceConfig(chip=2))
    return TrainConfig(**{**base, **over})


def _stored(**fields) -> dict:
    """A config node as the file holds one, with only `fields` written."""
    return {"__dataclass__": f"{SCHEMA}:TrainConfig", "fields": fields}


def test_a_file_this_version_writes_defaults_nothing(tmp_path):
    """The enumeration this whole contract rests on: every declared field of
    every stored dataclass is written, so a reader invents nothing."""
    loaded = load_checkpoint(_write(tmp_path / "now.npz", _config()))
    assert loaded["defaulted_fields"] == []


def test_every_saved_field_of_the_config_is_in_the_file(tmp_path):
    _arrays, meta = read_meta(_write(tmp_path / "all.npz", _config()))
    written = set(meta["config"]["fields"])
    declared = {f.name for f in dataclasses.fields(TrainConfig)}
    unsaved = set(TrainConfig.UNSAVED_FIELDS)
    assert written == declared - unsaved, sorted(declared - unsaved - written)


def test_an_added_field_is_reported_rather_than_quietly_defaulted():
    node = _stored(algo="dqn", total_frames=64, log_dir="runs/x")
    absent = defaulted_fields(node)
    assert "TrainConfig.seed" in absent
    assert "TrainConfig.eval_episodes" in absent
    assert join(node, {}, file="f.npz").seed == 0


def test_a_nested_dataclass_field_the_class_rejects_is_named():
    """The coordinator's question: does the name check reach a nested class?"""
    node = _stored(algo="dqn", total_frames=64, log_dir="runs/x",
                   env={"__dataclass__": f"{SCHEMA}:EnvConfig",
                        "fields": {"name": "atari", "n_envs_renamed": 4}})
    with pytest.raises(CheckpointFieldError, match="EnvConfig field"):
        join(node, {}, file="f.npz")


def test_a_field_that_describes_the_invocation_does_not_block_a_resume():
    """The narrowing on this gate's own false-refusal subset."""
    assert {"resume_path", "profile_json"} <= set(TrainConfig.UNSAVED_FIELDS)
    node = split(_config(seed=3), "config", {})
    assert defaulted_fields(node) == []
