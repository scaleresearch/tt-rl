"""Reading a checkpoint back: the format marker, the arrays, the members.

The read side of `ttrl.train.checkpoint.file`, which is the write side and
owns what each member MEANS. Split from it because the two are asked
different questions: that one is handed a run's state, this one is handed a
path that may not be a checkpoint at all. Both are re-exported from
`ttrl.train.checkpoint`.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ttrl.train.checkpoint.file import FORMAT, L1
from ttrl.train.checkpoint.specs import action_spec_from, obs_group_from
from ttrl.train.checkpoint.tree import defaulted_fields, join

__all__ = ["load_checkpoint", "read_meta"]


def read_meta(path: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The raw arrays and meta of a checkpoint, with the marker checked."""
    try:
        with np.load(path, allow_pickle=False) as npz:
            meta = json.loads(npz["__meta__"].tobytes().decode("utf-8"))
            arrays = {k: npz[k] for k in npz.files if k != "__meta__"}
    except Exception as exc:
        # A checkpoint is an npz holding a JSON `__meta__` array. Anything
        # else lands here: a pickle, a torch file, a truncated write, a file
        # that is not a checkpoint at all. The raw decoder error names a byte
        # offset and not the file, which is no use to whoever typed the path.
        raise ValueError(
            f"{path!r} is not a readable ttrl checkpoint: {type(exc).__name__}: "
            f"{exc}. Check the path, or train afresh."
        ) from exc
    if meta.get("format") != FORMAT:
        raise ValueError(
            f"{path!r} does not carry the checkpoint format marker {FORMAT!r} "
            f"this build writes (it carries {meta.get('format')!r}). The "
            f"checkpoint format is pre-release: it carries no migrations and "
            f"reads only what this build wrote. Train afresh."
        )
    return arrays, meta


def load_checkpoint(path: str) -> dict[str, Any]:
    """Read a checkpoint back. Needs no device, no environment and no agent.

    Returns `level`, `obs_spec`, `action_spec`, `agent_state`,
    `env_policy_state`, `extra`, `defaulted_fields`, and for an L1 file
    `frame`, `config` as a `TrainConfig`, `env_reward_state`,
    `replay_state` and `cadence_state`.

    `defaulted_fields` names every config field this reader supplied from a
    declared default because the file does not carry it. It is empty for a file
    whose writer and reader declare the same fields, and a resume refuses a
    file for which it is not: see `ttrl.train.checkpoint.tree`.
    """
    arrays, meta = read_meta(path)
    result: dict[str, Any] = {
        "level": meta["level"],
        "obs_spec": obs_group_from(meta["obs_spec"]),
        "action_spec": action_spec_from(meta["action_spec"]),
        "agent_state": join(meta["agent_state"], arrays, file=path),
        # None rather than an empty snapshot: a reader must be able to tell
        # "this file carries no transform state" from "its transform state is
        # empty", and only the first is a reason to refuse a resume.
        "env_policy_state": (
            join(meta["env_policy_state"], arrays, file=path)
            if meta.get("env_policy_state") is not None
            else None
        ),
        "extra": meta.get("extra") or {},
        "defaulted_fields": sorted(set(
            defaulted_fields(meta["config"]) if meta["level"] == L1 else []
        )),
    }
    if meta["level"] == L1:
        result["frame"] = int(meta["frame"])
        result["config"] = join(meta["config"], arrays, file=path)
        # None says the file carries no reward statistics, which is also the
        # truth about a run that normalised no reward.
        result["env_reward_state"] = (
            join(meta["env_reward_state"], arrays, file=path)
            if meta.get("env_reward_state") is not None
            else None
        )
        # Absent in any file whose run did not ask for it. None says the file
        # carries no transitions, which is what a warm resume into a
        # replay-owning agent is refused on.
        result["replay_state"] = (
            join(meta["replay_state"], arrays, file=path)
            if meta.get("replay_state") is not None
            else None
        )
        result["cadence_state"] = (
            join(meta["cadence_state"], arrays, file=path)
            if meta.get("cadence_state") is not None
            else None
        )
    return result
