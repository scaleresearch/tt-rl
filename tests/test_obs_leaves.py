"""What a composite observation is, host side. No hardware.

The spec, the observation it validates, the actor's half of both, and the one
place a bare `ObsSpec` is promoted to a one-leaf group. The leaf set and the
kind set are derived from the code in every gate below, never typed here: a
test that walks a set it wrote down itself passes forever once the real set
moves.
"""

from __future__ import annotations

import ast
import pathlib
import typing

import numpy as np
import pytest

from ttrl.buffers import ReplayBuffer
from ttrl.envs.obs_batch import observation_group
from ttrl.nets.multi import DEFAULT_ENCODER_BY_ROLE
from ttrl.nets.registry import ENCODERS
from ttrl.spaces import (DEFAULT_LEAF, ActionSpec, ObsField, ObsGroup, ObsLeaf,
                         ObsSpec, Observation, group_mask_rows, require_group)
from ttrl.spaces.obs import ObsKind

CAMERA = ObsSpec.image(3, 8, 8)
STATE = ObsSpec.vector(5)
DISCRETE = ActionSpec.discrete(3)

#: The robotics case: a camera the actor reads, simulator state only the
#: critic may.
ROBOT = ObsGroup((ObsLeaf("camera", CAMERA), ObsLeaf("sim", STATE, actor=False)))


def _camera(lanes: int = 2) -> np.ndarray:
    return np.arange(lanes * 3 * 8 * 8, dtype=np.uint8).reshape(lanes, 3, 8, 8)


def _sim(lanes: int = 2) -> np.ndarray:
    return np.arange(lanes * 5, dtype=np.float32).reshape(lanes, 5)


def _robot_obs(lanes: int = 2) -> dict:
    return {"camera": _camera(lanes), "sim": _sim(lanes)}


# -- one leaf is the default ------------------------------------------------

def test_a_bare_spec_is_the_one_leaf_group_and_promotion_is_one_call():
    group = observation_group(STATE)
    assert group == ObsGroup.of(STATE)
    assert group.names == (DEFAULT_LEAF,)
    assert group[DEFAULT_LEAF] is STATE
    # The zero-cost default: nothing is critic-only, so the actor's group is
    # this very object and no second set of buffers or towers exists.
    assert group.actor is group
    assert group.privileged == ()


def test_each_layer_promotes_a_spec_in_one_place_and_no_class_does():
    """`ObsGroup.of` is called from module-level functions only, one per
    subpackage.

    Two layers promote: `envs` turns what an environment declares into a
    group, and `train/checkpoint` turns what a file recorded into one. They
    stay separate because the checkpoint layer reads a file with no ttnn and
    no environment imported, and routing it through the env layer would pull
    the whole `ttrl.envs` package into a checkpoint read.

    Derived by parsing the tree rather than by naming the callers, after a
    hand-typed caller list in this test went red when the call moved: a
    second promotion site inside one layer would leave every later reader
    asking which of the two forms it was handed, and a call from inside a
    class would mean a component promotes what it was given.
    """

    root = pathlib.Path(__file__).resolve().parents[1] / "ttrl"
    definition = root / "spaces" / "group.py"
    files = sorted(root.rglob("*.py"))
    assert files, "found no package source to read"

    sites = []
    for path in files:
        if path == definition:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for owner in ast.walk(tree):
            if not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                continue
            for call in ast.walk(owner):
                if (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "of"
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "ObsGroup"):
                    sites.append((path.relative_to(root), owner))
    methods = [
        f"{rel}:{owner.name}" for rel, owner in sites
        if isinstance(owner, ast.ClassDef)
    ]
    assert not methods, f"a class promotes a spec: {methods}"
    assert sites, "no function promotes a spec"

    by_layer: dict[str, set[str]] = {}
    for rel, owner in sites:
        by_layer.setdefault(rel.parts[0], set()).add(f"{rel}:{owner.name}")
    crowded = {k: sorted(v) for k, v in by_layer.items() if len(v) != 1}
    assert not crowded, crowded


def test_a_component_refuses_a_bare_spec_rather_than_promoting_it():
    with pytest.raises(TypeError, match="ObsGroup.of"):
        require_group(STATE, "somewhere")
    with pytest.raises(TypeError, match="takes an ObsGroup"):
        ReplayBuffer(STATE, DISCRETE, capacity=64)


# -- the spec ---------------------------------------------------------------

def test_a_group_needs_a_leaf_and_unique_names():
    with pytest.raises(ValueError, match="at least one leaf"):
        ObsGroup(())
    with pytest.raises(ValueError, match="unique"):
        ObsGroup((ObsLeaf("a", STATE), ObsLeaf("a", STATE)))
    with pytest.raises(TypeError, match="does not nest"):
        ObsLeaf("a", ROBOT)


def test_a_group_with_nothing_for_the_actor_is_refused():
    with pytest.raises(ValueError, match="critic-only"):
        ObsGroup((ObsLeaf("sim", STATE, actor=False),))


def test_the_mask_belongs_to_one_leaf_and_the_refusal_names_the_leaves():
    masked = ObsSpec.vector(6, action_mask=3)
    with pytest.raises(ValueError) as caught:
        ObsGroup((ObsLeaf("a", masked), ObsLeaf("b", masked)))
    message = str(caught.value)
    assert "['a', 'b']" in message
    assert "belongs on one leaf" in message
    group = ObsGroup((ObsLeaf("camera", CAMERA), ObsLeaf("moves", masked)))
    assert group.mask_leaf == "moves"
    assert group.action_mask == 3


def test_leaf_order_is_declaration_order_and_not_a_mappings_order():
    """The order the encoder concatenates in is a property of the spec."""
    assert ROBOT.names == ("camera", "sim")
    shuffled = {"sim": _sim(), "camera": _camera()}
    assert ROBOT.observation(shuffled).leaves == ROBOT.names
    assert tuple(ROBOT.observation(_robot_obs()).leaves) == ROBOT.names


# -- the observation --------------------------------------------------------

def test_an_observation_carries_every_leaf_and_its_lane_count():
    obs = ROBOT.observation(_robot_obs(4))
    assert obs.leaves == ("camera", "sim")
    assert obs.lanes == 4
    assert obs["camera"].shape == (4, 3, 8, 8)
    assert ROBOT.observation(
        {"camera": _camera(1)[0], "sim": _sim(1)[0]}).lanes is None
    assert obs.select(2).lanes is None
    assert obs.select(np.array([0, 1])).lanes == 2


def test_a_dtype_is_refused_rather_than_cast():
    """The silent cast this type exists to prevent."""
    wrong = dict(_robot_obs())
    wrong["camera"] = wrong["camera"].astype(np.float32)
    with pytest.raises(TypeError, match="declares uint8"):
        ROBOT.observation(wrong)


def test_a_missing_or_extra_leaf_is_refused_by_name():
    with pytest.raises(ValueError, match="missing=\\['sim'\\]"):
        ROBOT.observation({"camera": _camera()})
    with pytest.raises(ValueError, match="extra=\\['extra'\\]"):
        ROBOT.observation({**_robot_obs(), "extra": _sim()})
    with pytest.raises(TypeError, match="cannot hold them"):
        ROBOT.observation(_camera())


def test_leaves_must_cover_the_same_lanes():
    with pytest.raises(ValueError, match="covers the same lanes"):
        ROBOT.observation({"camera": _camera(2), "sim": _sim(3)})


def test_a_leaf_is_read_by_name_and_nothing_else():
    obs = ROBOT.observation(_robot_obs())
    #: Positional indexing is refused with migration guidance instead, which
    #: tests/test_obs_coercion.py gates along with the rest of that surface.
    with pytest.raises(TypeError, match="indexes an Observation by position"):
        obs[0]
    with pytest.raises(KeyError, match="no leaf 'depth'"):
        obs["depth"]
    assert isinstance(obs, Observation)


# -- the actor's half -------------------------------------------------------

def test_the_actor_sees_neither_a_privileged_leaf_nor_a_privileged_field():
    split = ObsSpec.split_vector(ObsField("sensed", 3),
                                 ObsField("hidden", 2, actor=False))
    group = ObsGroup((ObsLeaf("camera", CAMERA), ObsLeaf("state", split),
                      ObsLeaf("sim", STATE, actor=False)))
    assert group.privileged == ("state.hidden", "sim")
    actor = group.actor
    assert actor.names == ("camera", "state")
    assert actor["state"].shape == (3,)
    obs = group.observation({"camera": _camera(), "state": np.arange(10, dtype=np.float32).reshape(2, 5),
                             "sim": _sim()})
    cut = group.actor_observation(obs)
    assert cut.leaves == ("camera", "state")
    assert np.array_equal(cut["state"], obs["state"][:, :3])


def test_the_actors_group_is_the_group_itself_when_nothing_is_privileged():
    group = ObsGroup((ObsLeaf("camera", CAMERA), ObsLeaf("state", STATE)))
    assert group.actor is group
    obs = group.observation({"camera": _camera(), "state": _sim()})
    assert group.actor_observation(obs) is obs


# -- every kind, enumerated from the code ----------------------------------

def test_every_obs_kind_has_a_default_encoder_in_every_role():
    """Derived from the code on both sides, so a new kind cannot slip past.

    Every role's table is covered, not just the observation one: 'auto' is
    resolved per role, so a kind missing from the critic table would raise
    inside `build_encoder` on a chip rather than here.
    """

    kinds = set(typing.get_args(ObsKind))
    assert kinds, "ObsKind enumerates nothing"
    assert DEFAULT_ENCODER_BY_ROLE, "no roles declare a default encoder"
    names = set(ENCODERS.names())
    for role, table in DEFAULT_ENCODER_BY_ROLE.items():
        assert set(table) == kinds, role
        assert set(table.values()) <= names, role


def test_a_critic_fuses_its_action_at_the_input_on_a_vector_leaf_only():
    """The defaulted topology per kind, which is the whole point of the role.

    An image leaf must keep its conv tower: `identity_encoder` refuses an
    image spec by name, so defaulting one to 'identity' would make a pixel
    SAC config unbuildable.
    """

    critic = DEFAULT_ENCODER_BY_ROLE["critic"]
    observation = DEFAULT_ENCODER_BY_ROLE["observation"]
    assert critic["vector"] == "identity"
    assert critic["image"] == observation["image"] != "identity"


# -- the buffers ------------------------------------------------------------

def test_replay_stores_one_array_per_leaf_and_serves_them_together():
    buf = ReplayBuffer(ROBOT, DISCRETE, capacity=64, n_step=1, seed=0)
    assert set(buf.obs) == set(ROBOT.names)
    assert buf.obs["camera"].dtype == np.uint8
    assert buf.obs["sim"].dtype == np.float32
    for i in range(20):
        one = {"camera": np.full((3, 8, 8), i, np.uint8),
               "sim": np.full(5, float(i), np.float32)}
        end = i % 7 == 6
        final = {"camera": np.full((3, 8, 8), 200 + i, np.uint8),
                 "sim": np.full(5, 200.0 + i, np.float32)}
        buf.add(one, 1, 0.5, end, final_obs=final if end else None)
    batch = buf.sample(8)
    assert batch.obs.leaves == ROBOT.names
    # Both leaves of a sample come from the same transition, which is what
    # one (lane, column) per observation buys.
    camera = batch.obs["camera"][:, 0, 0, 0].astype(np.float32)
    assert np.array_equal(camera, batch.obs["sim"][:, 0])


# -- the lane store ---------------------------------------------------------


def test_the_group_level_mask_reads_the_leaf_that_carries_it():
    masked = ObsSpec.vector(4, action_mask=2)
    group = ObsGroup((ObsLeaf("camera", CAMERA), ObsLeaf("moves", masked)))
    obs = group.observation({
        "camera": _camera(),
        "moves": np.array([[1.0, 2.0, 1.0, 0.0], [3.0, 4.0, 0.0, 1.0]], np.float32),
    })
    assert np.array_equal(group_mask_rows(obs, group),
                          np.array([[1.0, 0.0], [0.0, 1.0]], np.float32))
