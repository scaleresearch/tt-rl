"""What an `Observation` refuses to an ndarray's surface. No hardware.

Every user of this library before `ObsGroup` wrote `np.asarray(obs)`,
`obs.shape`, `len(obs)` or `obs[0]` against a single array. None of those
mean anything on a group of named leaves, and `np.asarray` in particular
produced a 0-dimensional object array rather than raising, so the wrong
reading survived to fail somewhere unrelated. Each gate below asserts the
refusal names the replacement to write, not merely that something raised.

The refused surface is derived from numpy here, never typed out: every name
ndarray has and `object` does not is sorted into one of four buckets, and a
name that lands in none of them fails `test_every_ndarray_name_is_accounted`.
A coercion added to numpy, or an attribute passed through here later, cannot
slip past ungated.
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from ttrl.spaces import ObsGroup, ObsLeaf, ObsSpec
from ttrl.spaces.observation import _ARITH_OPS, _NDARRAY_ATTRS

#: The single-array case, which is the one every migrating caller has.
PLAIN = ObsGroup.of(ObsSpec.vector(4))

ROBOT = ObsGroup((ObsLeaf("camera", ObsSpec.image(3, 8, 8)),
                  ObsLeaf("sim", ObsSpec.vector(5))))

#: Names numpy carries that an Observation supports on purpose.
SUPPORTED = {
    "__getitem__",  # the leaf accessor, by name
    "__contains__",  # whether a leaf name is carried
    "__bool__",  # left as object's, so `if obs` stays usable
    "__class_getitem__", "__copy__", "__deepcopy__", "__setstate__",
    "__setitem__", "__delitem__",  # absent: an Observation is not written to
    "_set_dtype", "_set_shape",
}

#: Array protocols an Observation must not implement, so that no numpy path
#: can read it as an array behind the refusal in `__array__`.
PROTOCOLS = {
    "__array_ufunc__", "__array_function__", "__array_interface__",
    "__array_struct__", "__array_priority__", "__array_wrap__",
    "__array_finalize__", "__array_namespace__", "__array_prepare__",
    "__buffer__",
    "__dlpack__", "__dlpack_device__",
}

#: Refused by a gate of its own below, because each message is its own text.
NAMED = {"__array__", "__len__", "__iter__"}


def plain(lanes: int = 3):
    return PLAIN.observation(np.zeros((lanes, 4), dtype=np.float32))


def robot():
    return ROBOT.observation({
        "camera": np.zeros((3, 3, 8, 8), dtype=np.uint8),
        "sim": np.zeros((3, 5), dtype=np.float32),
    })


def says(message: str, *fragments: str) -> None:
    assert "named leaf arrays" in message, message
    for fragment in fragments:
        assert fragment in message, f"{fragment!r} missing from {message!r}"


def test_asarray_refuses_and_names_the_leaf_accessor():
    with pytest.raises(TypeError) as excinfo:
        np.asarray(plain())
    says(str(excinfo.value), "np.asarray", "obs[name]", "obs.leaves",
         "obs.lanes", "ttrl.DEFAULT_LEAF", "obs[ttrl.DEFAULT_LEAF]")


@pytest.mark.parametrize("call", [
    lambda obs: np.array(obs),
    lambda obs: np.asarray(obs),
    lambda obs: np.asarray(obs, dtype=np.float32),
    lambda obs: np.stack([obs]),
    lambda obs: np.concatenate([obs]),
    lambda obs: np.ascontiguousarray(obs),
])
def test_every_conversion_path_reaches_the_same_refusal(call):
    """No numpy entry point gets an array out of an Observation."""
    with pytest.raises(TypeError) as excinfo:
        call(plain())
    says(str(excinfo.value), "obs[name]", "obs.lanes")


def test_len_refuses_and_names_both_readings():
    with pytest.raises(TypeError) as excinfo:
        len(plain())
    says(str(excinfo.value), "len(obs)", "obs.lanes", "len(obs.leaves)",
         "len(obs[name])")


def test_iteration_refuses_and_names_both_readings():
    with pytest.raises(TypeError) as excinfo:
        list(plain())
    says(str(excinfo.value), "obs.leaves", "obs.select(i)")


@pytest.mark.parametrize("index", [0, slice(0, 2), np.array([0, 1])])
def test_positional_indexing_refuses_and_names_select(index):
    with pytest.raises(TypeError) as excinfo:
        plain()[index]
    says(str(excinfo.value), "indexes an Observation by position",
         "obs.select(", "obs[name]")


def test_leaf_name_indexing_still_works():
    obs = robot()
    assert obs["camera"].shape == (3, 3, 8, 8)
    assert "sim" in obs
    assert "depth" not in obs
    with pytest.raises(KeyError, match="no leaf 'depth'"):
        obs["depth"]


def test_missing_leaf_keeps_its_own_error():
    """A wrong leaf name is a KeyError, which is not an ndarray mistake."""
    with pytest.raises(KeyError):
        plain()["camera"]


@pytest.mark.parametrize("name", sorted(_NDARRAY_ATTRS))
def test_every_ndarray_attribute_is_refused(name):
    """Derived from `dir(np.ndarray)`, so numpy's surface drives the gate."""
    with pytest.raises(TypeError) as excinfo:
        getattr(plain(), name)
    message = str(excinfo.value)
    says(message, f"obs[name].{name}", "obs.leaves")
    assert name in message


@pytest.mark.parametrize("name", sorted(_ARITH_OPS))
def test_every_operator_is_refused(name):
    obs = plain()
    operands = () if name in ("__neg__", "__pos__", "__abs__", "__invert__",
                              "__float__", "__int__", "__complex__",
                              "__index__") else (2.0,)
    with pytest.raises(TypeError) as excinfo:
        getattr(obs, name)(*operands)
    says(str(excinfo.value), _ARITH_OPS[name], "obs[name]", "obs.leaves")


@pytest.mark.parametrize("name", sorted(PROTOCOLS))
def test_no_array_protocol_is_implemented(name):
    assert not hasattr(plain(), name), (
        f"Observation carries {name}, which lets a numpy path read it as an "
        f"array around the refusal in __array__"
    )


def test_every_ndarray_name_is_accounted():
    """The gate that keeps a later-added passthrough from going unrefused."""
    surface = set(dir(np.ndarray)) - set(dir(object))
    buckets = set(_NDARRAY_ATTRS) | set(_ARITH_OPS) | SUPPORTED | PROTOCOLS
    buckets |= NAMED
    buckets |= {f"__i{n[2:]}" for n in _ARITH_OPS}
    unaccounted = sorted(surface - buckets)
    assert not unaccounted, (
        f"ndarray carries {unaccounted}, which Observation neither refuses "
        f"nor deliberately supports. Refuse each one with a message naming "
        f"the replacement, or list it in SUPPORTED with the reason."
    )


def test_in_place_operators_are_covered_by_their_binary_form():
    """Python falls back to the binary dunder, so no `__i*__` is needed."""
    inplace = {n for n in dir(np.ndarray)
               if n.startswith("__i") and n.endswith("__")
               and n not in ("__init__", "__init_subclass__", "__index__",
                             "__int__", "__invert__", "__iter__")}
    for name in inplace:
        binary = f"__{name[3:]}"
        assert binary in _ARITH_OPS, (
            f"{name} falls back to {binary}, which is not refused"
        )


@pytest.mark.parametrize("roundtrip", [
    copy.copy,
    copy.deepcopy,
    lambda obs: pickle.loads(pickle.dumps(obs)),
])
def test_copy_and_pickle_survive_the_refusals(roundtrip):
    """Both rebuild an instance with its slots unset.

    `__getattr__` must not read a leaf out of a half-built instance, and an
    async vector environment pickles observations across the process edge.
    """
    obs = roundtrip(robot())
    assert obs.leaves == ("camera", "sim")
    assert obs.lanes == 3
    assert obs["sim"].shape == (3, 5)
