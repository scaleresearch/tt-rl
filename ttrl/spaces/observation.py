"""One observation, as the named leaf arrays it is made of.

`Observation` is what crosses every boundary an observation crosses: out of a
vector environment, through a `Transition`, into a buffer, back out as a
batch, and into the device buffers `ttrl.nets.obs` owns. It is a mapping from
leaf name to host array and nothing else, so a consumer reads the leaf it
wants by name and cannot reach for a layout the spec did not promise.

It is built in exactly one place, `ObsGroup.observation`, which is also where
a bare array is promoted to the one-leaf case. Nothing else constructs one, so
there is no second form of an observation for a later reader to test for.

Leaf order is the group's declaration order, never a dict's insertion order: a
multi-modal encoder concatenates its leaves' features in this order, and a
permuted order would train on scrambled features with nothing to report it.
`ObsGroup.observation` reorders whatever mapping it is given into declaration
order, so the order is a property of the spec and a test can assert it.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ttrl.spaces.refusals import (
    _ARITH_OPS,
    _BASICS,
    _NDARRAY_ATTRS,
    _arith,
    _refuse,
    _refuse_attr,
)

__all__ = ["Observation", "actor_observation", "build_observation"]


class Observation:
    """Named leaf arrays for one observation, or for a batch of them.

    `lanes` is the length of the leading lane axis, or None for a single
    observation whose leaves carry exactly their declared element shape. The
    two are told apart by rank, which is unambiguous: a batch of a leaf of
    rank r has rank r + 1.
    """

    __slots__ = ("_arrays", "_lanes")

    def __init__(self, arrays: dict[str, np.ndarray], lanes: int | None):
        #: Built only by `ObsGroup.observation`, which has already put the
        #: arrays in declaration order and checked every shape and dtype.
        self._arrays = arrays
        self._lanes = lanes

    @property
    def leaves(self) -> tuple[str, ...]:
        """Leaf names, in the group's declaration order."""
        return tuple(self._arrays)

    @property
    def lanes(self) -> int | None:
        """Lane count, or None for a single observation."""
        return self._lanes

    def __getitem__(self, name: str) -> np.ndarray:
        if not isinstance(name, str):
            _refuse(
                f"obs[{name!r}] indexes an Observation by position",
                f"obs[...] takes a leaf name. For that lane of every leaf "
                f"write obs.select({name!r}), which gives an Observation "
                f"again. {_BASICS}",
            )
        try:
            return self._arrays[name]
        except KeyError:
            raise KeyError(
                f"observation has no leaf {name!r}; its leaves are "
                f"{list(self._arrays)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._arrays

    def __getattr__(self, name: str):
        if name in _NDARRAY_ATTRS:
            _refuse_attr(name)
        #: `_arrays` is itself unset while copy and pickle rebuild an
        #: instance slot by slot, so reading it below would recurse.
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(
            f"Observation has no attribute {name!r}; it carries leaves "
            f"{list(self._arrays)}, read with obs[name], plus leaves, lanes "
            f"and select")

    def __array__(self, dtype=None, copy=None):
        _refuse(
            "np.asarray / np.array was called on an Observation",
            f"There is no single array to convert it to, and numpy would "
            f"otherwise wrap it in a 0-dimensional object array whose shape "
            f"is (), which is where a lane count read as "
            f"np.asarray(obs).shape[0] fails. For every leaf at once write "
            f"[obs[n] for n in obs.leaves]. {_BASICS}",
        )

    def __len__(self) -> int:
        _refuse(
            "len(obs) asks an Observation for one length",
            f"A length here is ambiguous between the lane count and the "
            f"number of leaves, so it is refused rather than guessed. Write "
            f"obs.lanes for the lane count, len(obs.leaves) for the number "
            f"of leaves, and len(obs[name]) for one leaf's leading axis. "
            f"{_BASICS}",
        )

    def __iter__(self):
        _refuse(
            "iterating an Observation",
            f"Iterating is ambiguous between the lanes and the leaves, so it "
            f"is refused rather than guessed. Write "
            f"for name in obs.leaves to walk the leaves and "
            f"obs.select(i) for one lane as an Observation. {_BASICS}",
        )

    #: Every operator in `_ARITH_OPS`, refused. An operator dunder is looked
    #: up on the type, so each is bound here rather than via `__getattr__`.
    for _op in _ARITH_OPS:
        locals()[_op] = _arith(_op)
    del _op

    def select(self, lanes) -> "Observation":
        """The same observation narrowed to `lanes` of its lane axis.

        An integer gives one observation, whose leaves then carry their
        element shape and whose `lanes` is None. Anything numpy can index a
        leading axis with gives a batch.
        """
        if self._lanes is None:
            raise ValueError(
                "select: this observation is a single observation with no lane "
                "axis, so there is nothing to select from it"
            )
        single = isinstance(lanes, (int, np.integer))
        arrays = {name: arr[lanes] for name, arr in self._arrays.items()}
        if single:
            return Observation(arrays, None)
        first = next(iter(arrays.values()))
        return Observation(arrays, int(first.shape[0]))

    def __repr__(self) -> str:
        shapes = ", ".join(f"{k}: {tuple(v.shape)} {v.dtype}"
                           for k, v in self._arrays.items())
        return f"Observation(lanes={self._lanes}, {shapes})"


def build_observation(group, obs) -> Observation:
    """Host arrays as an `Observation`. THE one place arrays are promoted.

    Accepts an `Observation` (returned as itself after a leaf check), a
    mapping of leaf name to array, or, for a one-leaf group only, a bare
    array. Shapes and dtypes are checked exactly: a leaf handed the wrong
    dtype is refused here rather than cast on the way into storage, which
    is how a uint8 image would quietly become something else.
    """
    if isinstance(obs, Observation):
        if obs.leaves != group.names:
            raise ValueError(
                f"observation carries leaves {list(obs.leaves)} and this "
                f"ObsGroup declares {list(group.names)}. Emit one array per "
                f"declared leaf, or declare the leaves the environment "
                f"actually emits in its obs_spec."
            )
        return obs
    arrays = _as_mapping(group, obs)
    out: dict[str, np.ndarray] = {}
    lanes: int | None = None
    for index, leaf in enumerate(group.leaves):
        arr = np.asarray(arrays[leaf.name])
        want = tuple(leaf.spec.shape)
        if arr.dtype != leaf.spec.numpy_dtype:
            raise TypeError(
                f"leaf {leaf.name!r} has dtype {arr.dtype}, its ObsSpec "
                f"declares {leaf.spec.dtype}. Convert at the source: a cast "
                f"here would silently change what was stored."
            )
        if arr.shape == want:
            leaf_lanes = None
        elif arr.ndim == len(want) + 1 and arr.shape[1:] == want:
            leaf_lanes = int(arr.shape[0])
        else:
            raise ValueError(
                f"leaf {leaf.name!r} has shape {arr.shape}, its ObsSpec "
                f"declares {want} per observation (a batch is "
                f"(lanes, *{want}))"
            )
        if index and leaf_lanes != lanes:
            raise ValueError(
                f"leaf {leaf.name!r} carries {leaf_lanes} lanes and leaf "
                f"{group.leaves[0].name!r} carries {lanes}; every leaf of one "
                f"observation covers the same lanes"
            )
        lanes = leaf_lanes
        out[leaf.name] = arr
    return Observation(out, lanes)


def _as_mapping(group, obs) -> Mapping:
    if isinstance(obs, Mapping):
        missing = [n for n in group.names if n not in obs]
        extra = [k for k in obs if k not in group.names]
        if missing or extra:
            raise ValueError(
                f"observation names leaves {sorted(obs)} and this ObsGroup "
                f"declares {list(group.names)} (missing={missing}, "
                f"extra={extra}). Emit one array per declared leaf, or declare "
                f"the leaves the environment actually emits in its obs_spec."
            )
        return obs
    if len(group.leaves) == 1:
        return {group.leaves[0].name: obs}
    raise TypeError(
        f"this ObsGroup has {len(group.leaves)} leaves "
        f"{list(group.names)}, which differ in shape and dtype, so one array "
        f"cannot hold them. Emit a mapping of leaf name to array."
    )


def actor_observation(group, obs) -> Observation:
    """`obs` cut down to what the actor may read.

    The one sanctioned way to build an actor's input, and the group-level
    twin of `ObsSpec.actor_obs`: critic-only leaves are dropped and each
    surviving leaf's critic-only fields are cut out of it.
    """
    whole = build_observation(group, obs)
    actor = group.actor
    if actor is group:
        return whole
    arrays = {leaf.name: group[leaf.name].actor_obs(whole[leaf.name])
              for leaf in actor.leaves}
    return build_observation(actor, arrays)
