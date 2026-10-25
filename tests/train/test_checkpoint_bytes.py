"""Bytes in a checkpoint, and how strong the serializer should be."""

from __future__ import annotations

import numpy as np

from ttrl.spaces import ActionSpec, ObsSpec
from ttrl.train.checkpoint import load_checkpoint, read_meta, save_checkpoint


def _save(tmp_path, **kwargs):
    path = tmp_path / "c.npz"
    save_checkpoint(
        str(path),
        **{
            "agent_state": {"w": np.zeros(2, dtype=np.float32)},
            "obs_spec": ObsSpec.vector(2),
            "action_spec": ActionSpec.discrete(2),
            "level": "L0",
            **kwargs,
        },
    )
    return path


def test_saving_without_env_state_is_allowed_by_the_serializer(tmp_path):
    path = tmp_path / "c.npz"
    save_checkpoint(
        str(path),
        agent_state={"w": np.zeros(2, dtype=np.float32)},
        obs_spec=ObsSpec.vector(2),
        action_spec=ActionSpec.discrete(2),
        level="L0",
    )
    assert load_checkpoint(str(path))["env_policy_state"] is None


def test_bytes_round_trip_as_bytes_and_not_as_an_array(tmp_path):
    """Bytes in an author-supplied state tree must come back as `bytes`."""
    blob = bytes(range(256)) * 7 + b"\x00\xff trailing"
    path = _save(tmp_path, agent_state={
        "w": np.zeros(2, dtype=np.float32), "rng": blob})

    back = load_checkpoint(str(path))["agent_state"]["rng"]
    assert type(back) is bytes, "a uint8 array here would corrupt a consumer"
    assert back == blob, "the round trip must be byte-exact"

    _arrays, meta = read_meta(str(path))
    assert meta["agent_state"]["rng"] == {"__bytes__": "state.rng"}

