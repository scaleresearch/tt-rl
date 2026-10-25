"""A checkpoint's stored specs, back as the frozen dataclasses they were.

`dataclasses.asdict` flattens an `ObsGroup`'s leaves, their specs and their
fields into plain dicts, and a spec holding dicts fails its own validation on
the first attribute access. This rebuilds them.

Split from `ttrl.train.checkpoint.file`, which owns the npz and the member
names, because this owns neither: it is the JSON form of `ttrl.spaces`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ttrl.spaces import ActionSpec, ObsField, ObsGroup, ObsLeaf, ObsSpec

__all__ = ["action_spec_from", "json_default", "obs_group_from"]


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"object of type {type(obj)!r} is not JSON serializable")


def obs_group_from(stored: dict[str, Any]) -> ObsGroup:
    """An `ObsGroup` from its JSON form, every nested dataclass rebuilt.

    `dataclasses.asdict` flattens the leaves, their specs and their fields
    into plain dicts, and a spec holding dicts fails its own validation on the
    first attribute access. A bare spec is promoted on the way in by
    `save_checkpoint`, so the stored form is always a group.
    """
    return ObsGroup(tuple(
        ObsLeaf(name=leaf["name"], spec=_obs_spec(leaf["spec"]),
                actor=bool(leaf["actor"]))
        for leaf in stored["leaves"]
    ))


def _obs_spec(stored: dict[str, Any]) -> ObsSpec:
    """One leaf's `ObsSpec` from its JSON form, its `ObsField`s rebuilt."""
    out = _tuples(stored, ("shape",))
    out["fields"] = tuple(ObsField(**f) for f in out.get("fields") or ())
    return ObsSpec(**out)


def _tuples(spec: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """JSON turns a spec's tuples into lists; the specs are frozen and hashable."""
    out = dict(spec)
    for key in keys:
        if out.get(key) is not None:
            out[key] = tuple(out[key])
    return out


def action_spec_from(stored: dict[str, Any]) -> ActionSpec:
    return ActionSpec(**_tuples(stored, ("low", "high", "nvec")))
