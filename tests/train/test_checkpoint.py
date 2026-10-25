"""The checkpoint format: two levels, self-describing, readable with nothing installed."""

from __future__ import annotations

import dataclasses
import json
import typing

import numpy as np
import pytest

import ttrl
from ttrl.algos.sac.config import SACConfig
from ttrl.spaces import ActionSpec, ObsField, ObsGroup, ObsSpec
from ttrl.train.checkpoint import (
    FORMAT,
    L0,
    L1,
    load_checkpoint,
    read_meta,
    save_checkpoint,
)
from ttrl.train.config import TrainConfig

OBS = ObsSpec.image(4, 84, 84)
ACTIONS = ActionSpec.discrete(6)
STATE = {
    "online": {"fc.W": np.arange(6, dtype=np.float32).reshape(2, 3)},
    "optimizer": {"m": [np.ones(3, dtype=np.float32)], "t": 12},
    "frame": 999,
    "mode": "traced",
}


#: The shape `SyncVectorEnv.save_policy_state()` returns: one entry per
#: transform-bearing wrapper, naming its `POLICY_STATE` keys. No per-lane
#: entries and no trajectory state.
ENV_POLICY_STATE = [
    ["NormalizeObservation", {
        "mean": np.array([1.5, -2.0, 0.25]),
        "m2": np.array([9.0, 4.0, 1.0]),
        "count": 512.0,
    }],
]


def _save(path, **kwargs):
    defaults = dict(agent_state=STATE, obs_spec=OBS, action_spec=ACTIONS,
                    env_policy_state=ENV_POLICY_STATE)
    save_checkpoint(str(path), **{**defaults, **kwargs})


def test_l1_round_trip(tmp_path):
    path = tmp_path / "ckpt.npz"
    config = TrainConfig(total_frames=500, log_dir="runs/t")
    _save(path, level=L1, frame=120, config=config)
    loaded = load_checkpoint(str(path))
    assert loaded["level"] == L1
    assert loaded["frame"] == 120
    assert loaded["obs_spec"] == ObsGroup.of(OBS)
    assert loaded["action_spec"] == ACTIONS
    assert loaded["config"].total_frames == 500
    state = loaded["agent_state"]
    assert np.array_equal(state["online"]["fc.W"], STATE["online"]["fc.W"])
    assert np.array_equal(state["optimizer"]["m"][0], STATE["optimizer"]["m"][0])
    assert state["optimizer"]["t"] == 12
    assert state["mode"] == "traced"


def test_tensors_live_once_and_meta_holds_no_arrays(tmp_path):
    path = tmp_path / "c.npz"
    _save(path, level=L1, config=TrainConfig())
    arrays, meta = read_meta(str(path))
    assert set(arrays) == {
        "state.online.fc.W",
        "state.optimizer.m.0",
        "env_policy_state.0.1.mean",
        "env_policy_state.0.1.m2",
    }
    json.dumps(meta)
    assert meta["format"] == FORMAT


def test_a_failed_write_leaves_the_previous_file_intact(tmp_path):
    path = tmp_path / "c.npz"
    _save(path, level=L0)
    before = path.read_bytes()
    with pytest.raises(TypeError):
        _save(path, level=L0, agent_state={"bad": object()})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".*tmp*"))


def test_a_composite_action_spec_comes_back_hashable(tmp_path):
    """JSON gives back a list, and a spec that must stay hashable no longer is."""
    composite = ActionSpec.multi_discrete([3, 5])
    path = tmp_path / "composite.npz"
    _save(path, level=L0, action_spec=composite, obs_spec=ObsSpec.vector(6))
    reloaded = load_checkpoint(str(path))["action_spec"]
    assert reloaded == composite and hash(reloaded) == hash(composite)


def test_the_actor_critic_division_survives_the_round_trip(tmp_path):
    """A checkpoint that forgot which fields were critic-only would rebuild an
    actor over them on resume, and nothing downstream could tell."""
    spec = ObsSpec.split_vector(ObsField("sensed", 11),
                                ObsField("sim_state", 12, actor=False))
    path = tmp_path / "split.npz"
    group = ObsGroup.of(spec)
    _save(path, level=L0, obs_spec=group)
    loaded = load_checkpoint(str(path))["obs_spec"]
    assert loaded == group
    assert loaded.privileged == ("obs.sim_state",)
    assert loaded.actor == group.actor


#: One or more instances per class, each tuple field set in at least one of
#: them. Compared against the classes derived from the annotations below, so a
#: new tuple field cannot be added without a value here to carry it.
TUPLE_FIELD_CASES = {
    ObsSpec: (ObsSpec.split_vector(ObsField("a", 2), ObsField("b", 3)),),
    ObsGroup: (ObsGroup.of(ObsSpec.split_vector(ObsField("a", 2),
                                                ObsField("b", 3))),),
    ActionSpec: (ActionSpec.continuous([-1.0, -2.0], [1.0, 2.0]),
                 ActionSpec.multi_discrete((2, 3))),
    SACConfig: (SACConfig(critic_hidden=(256, 128)),),
}


def _tuple_fields(cls) -> list[str]:
    """`cls`'s fields declared as a tuple, Optional ones included."""
    hints = typing.get_type_hints(cls)
    found = []
    for f in dataclasses.fields(cls):
        declared = [hints[f.name], *typing.get_args(hints[f.name])]
        if any(typing.get_origin(d) is tuple for d in declared):
            found.append(f.name)
    return found


def _classes_with_a_tuple_field() -> set[type]:
    """Every ttrl dataclass a checkpoint stores that declares a tuple field."""
    found: set[type] = set()
    seen: set[type] = set()

    def walk(cls):
        if cls in seen or not dataclasses.is_dataclass(cls):
            return
        seen.add(cls)
        if _tuple_fields(cls):
            found.add(cls)
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            nested = [hints[f.name], *typing.get_args(hints[f.name])]
            for declared in list(nested):
                nested += typing.get_args(declared)
            for declared in nested:
                walk(declared)

    roots = [TrainConfig, ObsGroup, ObsSpec, ActionSpec]
    roots += [getattr(ttrl, name) for name in ttrl.__all__
              if name.endswith("Config")]
    for root in roots:
        walk(root)
    return found


def test_every_stored_tuple_field_has_a_round_trip_case():
    assert set(TUPLE_FIELD_CASES) == _classes_with_a_tuple_field()
    for cls, cases in TUPLE_FIELD_CASES.items():
        covered = {name for case in cases for name in _tuple_fields(cls)
                   if getattr(case, name) not in (None, ())}
        assert covered == set(_tuple_fields(cls)), cls


@pytest.mark.parametrize("case", [case for cases in TUPLE_FIELD_CASES.values()
                                  for case in cases],
                         ids=lambda case: f"{type(case).__name__}{case!r:.0}")
def test_a_stored_tuple_field_comes_back_a_tuple(tmp_path, case):
    path = tmp_path / "tuples.npz"
    _save(path, level=L0,
          agent_state={"c": case, "w": np.zeros(2, np.float32)})
    restored = load_checkpoint(str(path))["agent_state"]["c"]
    assert restored == case
    assert hash(restored) == hash(case)


def test_a_dataclass_outside_the_library_is_refused(tmp_path):
    @dataclasses.dataclass
    class Foreign:
        x: int = 1

    with pytest.raises(TypeError, match="outside 'ttrl.'"):
        _save(tmp_path / "c.npz", level=L0, agent_state={"c": Foreign()})


def test_a_foreign_type_path_in_a_file_is_refused(tmp_path):
    path = tmp_path / "c.npz"
    _save(path, level=L0, agent_state={"w": np.zeros(1, np.float32)})
    arrays, meta = read_meta(str(path))
    meta["agent_state"] = {"c": {"__dataclass__": "os:system", "fields": {}}}
    blob = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(str(path), **arrays, __meta__=blob)
    with pytest.raises(ValueError, match="Refusing to import"):
        load_checkpoint(str(path))

