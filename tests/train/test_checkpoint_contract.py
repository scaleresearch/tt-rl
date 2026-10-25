"""The on-disk contract: field names and dropped fields."""

from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import pytest

from ttrl.algos.dqn.config import DQNConfig
from ttrl.spaces import ActionSpec, ObsSpec
from ttrl.train.checkpoint import (
    L1,
    CheckpointFieldError,
    load_checkpoint,
    read_meta,
    save_checkpoint,
)
from ttrl.train.checkpoint.tree import split
from ttrl.train.config import EnvConfig, TrainConfig

OBS = ObsSpec.vector(4)
ACTIONS = ActionSpec.discrete(3)
STATE = {"w": np.ones(3, dtype=np.float32)}
ENV_POLICY_STATE = []


def _save(path, **kwargs):
    save_checkpoint(
        str(path),
        **{
            "agent_state": STATE,
            "obs_spec": OBS,
            "action_spec": ACTIONS,
            "env_policy_state": ENV_POLICY_STATE,
            "level": L1,
            "config": TrainConfig(log_dir="runs/x"),
            **kwargs,
        },
    )


def _rename_field(monkeypatch, cls, old: str, new: str) -> None:
    """Rename one of `cls`'s dataclass fields, as a later release would."""
    fields = {k: copy.copy(v) for k, v in cls.__dataclass_fields__.items()}
    fields[new] = fields.pop(old)
    fields[new].name = new
    monkeypatch.setattr(cls, "__dataclass_fields__", fields)
    assert old not in {f.name for f in dataclasses.fields(cls)}


def _rewrite_meta(path, mutate) -> None:
    """Rewrite a checkpoint's JSON meta in place, arrays untouched."""
    arrays, meta = read_meta(str(path))
    mutate(meta)
    blob = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(str(path), **arrays, __meta__=blob)


def test_a_renamed_config_field_is_named_with_an_action(
    tmp_path, monkeypatch
):
    """The defect itself, on a field that is genuinely part of a run."""
    path = tmp_path / "c.npz"
    _save(path)
    _rename_field(monkeypatch, EnvConfig, "n_envs", "lanes")

    with pytest.raises(CheckpointFieldError) as excinfo:
        load_checkpoint(str(path))

    message = str(excinfo.value)
    assert "n_envs" in message, "the message must name the field"
    assert str(path) in message, "and the file it came from"
    assert "renamed or removed" in message
    assert "unsaved_fields" in message.lower()
    assert "UNSAVED_FIELDS" in message


def test_a_renamed_algo_config_field_is_caught_the_same_way(tmp_path, monkeypatch):
    """The agent's own state carries the algorithm's frozen config, so the
    same contract has to hold there and not only for the run config."""

    path = tmp_path / "c.npz"
    _save(path, agent_state={**STATE, "config": DQNConfig()})
    _rename_field(monkeypatch, DQNConfig, "gamma", "discount")

    with pytest.raises(CheckpointFieldError, match="gamma"):
        load_checkpoint(str(path))


def test_a_required_field_added_since_the_write_is_named(tmp_path, monkeypatch):
    path = tmp_path / "c.npz"
    _save(path)
    fields = {k: copy.copy(v) for k, v in EnvConfig.__dataclass_fields__.items()}
    added = copy.copy(fields["n_envs"])
    added.name = "lanes"
    added.default = dataclasses.MISSING
    added.default_factory = dataclasses.MISSING
    fields["lanes"] = added
    monkeypatch.setattr(EnvConfig, "__dataclass_fields__", fields)

    with pytest.raises(CheckpointFieldError) as excinfo:
        load_checkpoint(str(path))
    message = str(excinfo.value)
    assert "lanes" in message and "has no default" in message
    assert "Give the field a default" in message


def test_an_optional_field_added_since_the_write_takes_its_default(
    tmp_path, monkeypatch
):
    """The other direction must NOT raise: a new setting with a default does
    not change what the stored weights are."""
    path = tmp_path / "c.npz"
    _save(path)
    _rewrite_meta(
        path, lambda m: m["config"]["fields"]["env"]["fields"].pop("asynchronous")
    )
    loaded = load_checkpoint(str(path))
    assert loaded["config"].env.asynchronous is False


def test_an_unsaved_field_is_dropped_rather_than_reported_as_unknown(tmp_path):
    unsaved = TrainConfig.UNSAVED_FIELDS
    assert unsaved, "an empty declaration would make this test vacuous"
    declared = {f.name for f in dataclasses.fields(TrainConfig)}
    assert set(unsaved) <= declared, sorted(set(unsaved) - declared)
    assert "UNSAVED_FIELDS" not in declared

    arrays: dict = {}
    written = split(TrainConfig(total_frames=4, log_dir="runs/x"), "config", arrays)
    for name in unsaved:
        assert name not in written["fields"], name


#: Every dtype spelling an `ObsSpec` may be given, by the name a user writes,
#: mapped to the stored dtype it must normalise to or None if it must be
#: refused. `np.dtype("float32") == "float32"` is True and `in` compares with
#: `==`, so the np.dtype spelling was ACCEPTED by the closed-set check,
#: trained fine, and then crashed the first checkpoint write on
#: `TypeError: Object of type Float32DType is not JSON serializable`: a run
#: that died hours in, using the spelling this library's own error message
#: told the user to adopt. `__post_init__` normalises to the `str` name, and
#: the refused half of this table is what keeps that normalisation from
#: widening into a fallback that quietly narrows a float64 observation.
DTYPE_SPELLINGS = {
    "str": ("float32", "float32"),
    "scalar type": (np.float32, "float32"),
    "np.dtype": (np.dtype("float32"), "float32"),
    "defaulted": (None, "float32"),
    "str uint8": ("uint8", "uint8"),
    "str float64": ("float64", None),
    "builtin float": (float, None),
    "np.float64": (np.float64, None),
    "not a dtype": (7, None),
}


@pytest.mark.parametrize("spelling", sorted(DTYPE_SPELLINGS))
def test_every_dtype_spelling_is_refused_or_survives_a_checkpoint_write(
    tmp_path, spelling
):
    dtype, expected = DTYPE_SPELLINGS[spelling]
    kwargs = {} if spelling == "defaulted" else {"dtype": dtype}
    if expected is None:
        with pytest.raises(ValueError, match="uint8 or float32"):
            ObsSpec(shape=(4,), kind="vector", **kwargs)
        return

    spec = ObsSpec(shape=(4,), kind="vector", **kwargs)
    assert spec.dtype == expected
    assert isinstance(spec.dtype, str), "the stored dtype is what gets serialized"

    path = tmp_path / f"{spelling.replace(' ', '_')}.npz"
    _save(path, obs_spec=spec)
    group = load_checkpoint(str(path))["obs_spec"]
    assert group[group.names[0]] == spec
