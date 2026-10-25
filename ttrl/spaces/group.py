"""What an observation IS: a named tuple of leaves, one leaf being the default.

An environment emits one observation made of one or more LEAVES. A leaf is a
single array with its own shape, dtype and kind, which is exactly what
`ObsSpec` already describes, so `ObsSpec` is the leaf and `ObsGroup` is the
observation. A camera frame for the actor plus privileged simulator state for
the critic is a two-leaf group; the overwhelmingly common single-array
observation is a one-leaf group, and it walks the same code.

This is the same move `ObsSpec.fields` already makes one level down.
Asymmetric is the general shape there, with `fields = ()` spelling the
symmetric case; multi-leaf is the general shape here, with one leaf spelling
the single-array case. `ObsGroup.actor is group` whenever nothing is
critic-only, exactly as `ObsSpec.actor is spec` does, so a symmetric run
allocates one set of buffers and builds one tower.

The actor/critic split composes rather than duplicates. A whole leaf is the
critic's when `ObsLeaf.actor` is False (an image the real robot has no camera
for, or a state vector it cannot sense), and part of a leaf is the critic's
when that leaf's own `ObsSpec.fields` say so. `ObsGroup.actor` applies both.

LEAF ORDER IS DECLARATION ORDER and it is load-bearing: a multi-modal encoder
concatenates per-leaf features in this order, and the wrong order trains on
scrambled features with no error anywhere. The order is a property of this
spec, never of a dict's insertion history, so it is something a test asserts.

PROMOTION HAPPENS IN ONE PLACE. `ObsGroup.of` is the only code that turns a
bare `ObsSpec` into a one-leaf group, and `ObsGroup.observation` is the only
code that turns host arrays into an `Observation`. Everything above the vector
environment sees an `ObsGroup` and an `Observation` and never asks which form
it was handed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ttrl.spaces.obs import ObsSpec
from ttrl.spaces.observation import Observation, actor_observation, build_observation

__all__ = ["DEFAULT_LEAF", "ObsGroup", "ObsLeaf", "require_group"]

#: The name of the single leaf of a single-array observation. A one-leaf group
#: built by `ObsGroup.of` uses it, so the common case has one spelling.
DEFAULT_LEAF = "obs"


def require_group(spec, where: str) -> "ObsGroup":
    """`spec`, which must already be an `ObsGroup`, or a refusal saying why.

    Everything above the vector environment takes a group, because promotion
    happens once, at that boundary. This is the refusal for a caller that
    reached a component with a bare leaf spec, and it is deliberately not a
    promotion: a second promotion site would leave every later reader asking
    which of the two forms it holds.
    """
    if isinstance(spec, ObsGroup):
        return spec
    raise TypeError(
        f"{where} takes an ObsGroup describing the whole observation, got "
        f"{type(spec).__name__}. An environment's obs_spec is already a group; "
        f"outside one, spell the single-array case ObsGroup.of(spec)."
    )


@dataclass(frozen=True)
class ObsLeaf:
    """One named array of an observation, described by its own `ObsSpec`.

    `actor=False` marks the whole leaf as the critic's: privileged simulator
    state, or a view of the scene the deployed system has no sensor for. No
    actor network is ever built over such a leaf.
    """

    name: str
    spec: ObsSpec
    actor: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ObsLeaf needs a non-empty name")
        if not isinstance(self.spec, ObsSpec):
            raise TypeError(
                f"ObsLeaf {self.name!r} needs an ObsSpec describing one array, got "
                f"{type(self.spec).__name__}. A leaf is a single array; a tree of "
                f"leaves is an ObsGroup, which does not nest."
            )


@dataclass(frozen=True)
class ObsGroup:
    """One observation: its leaves, in declaration order."""

    leaves: tuple[ObsLeaf, ...]

    def __post_init__(self) -> None:
        if not self.leaves:
            raise ValueError(
                "ObsGroup needs at least one leaf; an observation with no arrays "
                "in it is not an observation"
            )
        names = [leaf.name for leaf in self.leaves]
        if len(set(names)) != len(names):
            raise ValueError(f"ObsGroup leaf names must be unique, got {names}")
        self._check_mask()
        if not any(leaf.actor for leaf in self.leaves):
            raise ValueError(
                f"every leaf of {names} is marked critic-only, so the actor would "
                f"have no input at all. Mark the leaves the deployed system can "
                f"sense with actor=True."
            )

    def _check_mask(self) -> None:
        masked = [leaf.name for leaf in self.leaves if leaf.spec.action_mask]
        if len(masked) > 1:
            raise ValueError(
                f"leaves {masked} each declare a legal-action mask. One observation "
                f"has one legal-action set, so the mask belongs on one leaf: keep "
                f"action_mask on the single leaf whose tail entries carry it and "
                f"pass action_mask=0 for the others."
            )

    # -- promotion ---------------------------------------------------------

    @classmethod
    def of(cls, spec) -> "ObsGroup":
        """The group `spec` describes. THE one place a bare spec is promoted.

        An `ObsGroup` is returned as itself and an `ObsSpec` becomes the
        one-leaf group named `DEFAULT_LEAF`. Every reader above this point has
        a group, so none of them tests which form it was given.
        """
        if isinstance(spec, ObsGroup):
            return spec
        if isinstance(spec, ObsSpec):
            return cls((ObsLeaf(DEFAULT_LEAF, spec),))
        raise TypeError(
            f"ObsGroup.of needs an ObsSpec or an ObsGroup, got "
            f"{type(spec).__name__}"
        )

    # -- the leaves --------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        """Leaf names in declaration order, which is concatenation order."""
        return tuple(leaf.name for leaf in self.leaves)

    def __getitem__(self, name: str) -> ObsSpec:
        for leaf in self.leaves:
            if leaf.name == name:
                return leaf.spec
        raise KeyError(f"no leaf {name!r} in ObsGroup{list(self.names)}")

    def __len__(self) -> int:
        return len(self.leaves)

    def specs(self) -> tuple[tuple[str, ObsSpec], ...]:
        """`(name, spec)` per leaf, in declaration order."""
        return tuple((leaf.name, leaf.spec) for leaf in self.leaves)

    # -- the actor's half --------------------------------------------------

    @property
    def privileged(self) -> tuple[str, ...]:
        """What the critic may read and the actor may not, named.

        A whole critic-only leaf is named by itself, and a critic-only field
        inside a leaf as `leaf.field`, so a refusal can say which it is.
        """
        out: list[str] = []
        for leaf in self.leaves:
            if not leaf.actor:
                out.append(leaf.name)
                continue
            out.extend(f"{leaf.name}.{field}" for field in leaf.spec.privileged)
        return tuple(out)

    @property
    def actor(self) -> "ObsGroup":
        """The group the actor sees. `is self` when nothing is critic-only."""
        keep = tuple(leaf for leaf in self.leaves if leaf.actor)
        if not keep:
            raise ValueError(
                f"every leaf of {list(self.names)} is critic-only"
            )  # pragma: no cover - __post_init__ refuses this group
        narrowed = tuple(replace(leaf, spec=leaf.spec.actor) for leaf in keep)
        if len(narrowed) == len(self.leaves) and all(
            new.spec is old.spec for new, old in zip(narrowed, self.leaves)
        ):
            return self
        return ObsGroup(narrowed)

    # -- the legal-action mask --------------------------------------------

    @property
    def mask_leaf(self) -> str | None:
        """The one leaf carrying a legal-action mask, or None."""
        for leaf in self.leaves:
            if leaf.spec.action_mask:
                return leaf.name
        return None

    @property
    def action_mask(self) -> int:
        """Width of the legal-action mask, 0 for an unmasked observation."""
        name = self.mask_leaf
        return 0 if name is None else int(self[name].action_mask)

    @property
    def mask_spec(self) -> ObsSpec:
        """The masked leaf's spec, for `ttrl.spaces.mask`."""
        name = self.mask_leaf
        if name is None:
            raise ValueError(
                f"ObsGroup{list(self.names)} declares no legal-action mask; build "
                f"the leaf that carries it with ObsSpec.vector(dim, action_mask=n)"
            )
        return self[name]

    # -- observations ------------------------------------------------------

    def observation(self, obs) -> Observation:
        """Host arrays as an `Observation`. THE one place arrays are promoted.

        Accepts an `Observation` (returned as itself after a leaf check), a
        mapping of leaf name to array, or, for a one-leaf group only, a bare
        array. The validation lives in `ttrl.spaces.observation`.
        """
        return build_observation(self, obs)

    def actor_observation(self, obs) -> Observation:
        """`obs` cut down to what the actor may read.

        The one sanctioned way to build an actor's input, and the group-level
        twin of `ObsSpec.actor_obs`: critic-only leaves are dropped and each
        surviving leaf's critic-only fields are cut out of it.
        """
        return actor_observation(self, obs)
