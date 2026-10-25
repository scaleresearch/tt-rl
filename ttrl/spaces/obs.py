"""What an observation is, and which half of one the actor is allowed to read.

An environment emits one observation. `ObsSpec` describes it, and `fields`
describes how it is divided between the two readers an actor-critic algorithm
has: a field with `actor=False` is simulator state the critic may condition on
and the actor may not. `ObsSpec.actor` is the narrower spec of what the actor
sees, and `ObsSpec.actor_obs` is the one sanctioned way to cut an observation
down to it.

Asymmetric is the general shape and symmetric is the case where every field is
the actor's, which `fields = ()` spells: one field covering the whole
observation. Every component downstream builds the actor from `obs_spec.actor`
and the critic from the spec itself, so a run where the two coincide walks the
same code as a run where they do not.

THE KIND SET IS CLOSED BY DESIGN. `ObsKind` is "image" or "vector" and
`__post_init__` rejects anything else, because the kind is what every encoder
and every head is written against: a third kind would be a case every one of
them has to grow, which no registration can do for them. A new representation
is served by registering an encoder (`ttrl.nets.ENCODERS`) over a vector or
image spec, which is where the architecture seam is. Adding a kind here is a
change to this library rather than an extension of it.

`scale` is the factor mapping stored values onto the network's input range; it
is folded into the first layer's weights rather than applied as a separate
device op.

`action_mask` is the width of the legal-action mask carried in the final
entries of every observation vector, or 0 for an environment where every
action is always legal. The mask itself is per-step and per-lane, so it is
environment output and the spec says only where to find it. Putting the value
in the observation is what makes it arrive everywhere it is needed: a
`Transition` already carries obs and next_obs, a replay buffer already stores
observations and already recovers an episode-final successor from `final_obs`,
and a checkpointed spec already round-trips, so the next-state mask a
Double-DQN bootstrap needs is a slice of the next_obs the buffer already
serves. The mask is the actor's to read, so the fields covering it must be
the actor's too.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from ttrl.spaces.fields import ObsField, check_fields

ObsKind = Literal["image", "vector"]

#: `ObsField` is re-exported: it is half of an `ObsSpec`'s signature, so a
#: reader of this module needs the name in hand. It is defined in
#: `ttrl.spaces.fields` with the validation of the division it declares.
__all__ = ["ObsField", "ObsSpec", "ObsKind"]


def _dtype_name(dtype) -> str:
    """Any spelling of a numpy dtype as its `str` name.

    Normalisation and not a fallback: `np.float32`, `np.dtype("float32")` and
    `"float32"` name one dtype, and `__post_init__` then holds the name to the
    closed set, so nothing is widened by being spelled differently. The stored field being always a `str` is a
    correctness requirement rather than tidiness: `dataclasses.asdict` of a
    spec goes straight into a checkpoint's JSON meta, and a `np.dtype` object
    there raised `TypeError: not JSON serializable` at the first checkpoint
    write of a run that had trained for hours.
    """
    if isinstance(dtype, str):
        name = dtype
    else:
        try:
            name = np.dtype(dtype).name
        except TypeError as exc:
            raise ValueError(
                f"obs dtype must be uint8 or float32, got {dtype!r}, which "
                f"names no numpy dtype at all ({exc})"
            ) from exc
    return name


@dataclass(frozen=True)
class ObsSpec:
    """Shape, dtype, scaling and field division of a single observation.

    `shape` excludes the batch axis. Image observations are channels-first
    (C, H, W) where C counts stacked frames times colour planes.

    REACH FOR A CONSTRUCTOR RATHER THAN THIS SIGNATURE. `ObsSpec.vector(dim)`
    is a float32 1-D observation; `ObsSpec.image(channels, height, width)` is
    a uint8 (C, H, W) one and also sets `scale=1/255`, which an image spec
    built by hand almost always wants and almost always forgets.
    `ObsSpec.split_vector(ObsField(...), ...)` declares a vector field by
    field and sums the widths.

    `dtype` defaults to float32, and is normalised the way
    `gymnasium.spaces.Box` normalises its own, so `"float32"`, `np.float32`
    and `np.dtype("float32")` are three spellings of one spec. The stored
    field is always the `str` name, which is what makes a spec JSON-round-trip
    into a checkpoint. The set stays closed at uint8 and float32: a dtype
    outside it, `float` and so `float64` included, is refused by the name it
    normalises to rather than quietly narrowed.

    THE LEGAL-ACTION MASK IS THE FINAL `action_mask` ENTRIES OF EVERY
    OBSERVATION VECTOR, written as {0.0, 1.0} indicators by the environment
    on every step and on every terminal observation. `action_mask` is the
    width of that tail, declared with `ObsSpec.vector(dim, action_mask=n)`
    where `dim` counts the features AND the mask. `ttrl.mask_rows` reads the
    tail back, `ttrl.draw_legal` draws a uniform legal action per row, and
    `ttrl.check_legal` refuses a transition whose action its own
    observation's mask forbade.
    """

    shape: tuple[int, ...]
    kind: ObsKind
    dtype: str = "float32"
    scale: float = 1.0
    action_mask: int = 0
    fields: tuple[ObsField, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", _dtype_name(self.dtype))
        if not self.shape or any(d < 1 for d in self.shape):
            raise ValueError(f"obs shape must be positive and non-empty, got {self.shape}")
        if self.dtype not in ("uint8", "float32"):
            raise ValueError(f"obs dtype must be uint8 or float32, got {self.dtype!r}")
        if self.kind == "image" and len(self.shape) != 3:
            raise ValueError(f"image obs must be (C, H, W), got {self.shape}")
        if self.kind == "vector" and len(self.shape) != 1:
            raise ValueError(f"vector obs must be 1-D, got {self.shape}")
        if self.kind not in ("image", "vector"):
            raise ValueError(f"obs kind must be image or vector, got {self.kind!r}")
        if not self.scale > 0.0:
            raise ValueError(f"obs scale must be positive, got {self.scale}")
        if self.action_mask < 0:
            raise ValueError(
                f"action_mask must be 0 or the number of actions, got "
                f"{self.action_mask}. Pass 0 for an environment where every action "
                f"is always legal."
            )
        if self.action_mask:
            if self.kind != "vector":
                raise ValueError(
                    f"action_mask={self.action_mask} needs a vector observation, got "
                    f"kind={self.kind!r}. The mask occupies the final entries of the "
                    f"observation vector, so flatten the observation and declare it "
                    f"with ObsSpec.vector(dim, action_mask=n)."
                )
            if self.action_mask >= self.shape[0]:
                raise ValueError(
                    f"action_mask={self.action_mask} leaves no features in an "
                    f"observation of dim {self.shape[0]}. Widen the observation so "
                    f"the mask is the final {self.action_mask} entries of more than "
                    f"{self.action_mask}."
                )
        check_fields(self)

    # -- the whole observation ---------------------------------------------

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    # -- the actor's half --------------------------------------------------

    @property
    def privileged(self) -> tuple[str, ...]:
        """Names of the fields the critic may read and the actor may not."""
        return tuple(f.name for f in self.fields if not f.actor)

    @property
    def actor(self) -> "ObsSpec":
        """The spec of what the actor sees, which is this spec's actor fields.

        Equal to this spec when nothing is critic-only, so an actor built from
        it in the symmetric case is the network that was always built.
        """
        if not self.privileged:
            return self
        keep = tuple(f for f in self.fields if f.actor)
        return replace(self, shape=(sum(f.size for f in keep),), fields=keep)

    @property
    def actor_spans(self) -> tuple[tuple[int, int], ...]:
        """`(start, stop)` column ranges of the actor's fields, merged."""
        spans: list[list[int]] = []
        start = 0
        for field in self.fields:
            stop = start + field.size
            if field.actor:
                if spans and spans[-1][1] == start:
                    spans[-1][1] = stop
                else:
                    spans.append([start, stop])
            start = stop
        return tuple((a, b) for a, b in spans)

    def actor_obs(self, obs) -> np.ndarray:
        """`(batch, *shape)` observations cut down to the actor's fields.

        The one sanctioned way to build an actor's input. A batch that is
        already the actor's width is refused rather than cut again: that is
        what a deployed policy fed the real robot's sensors looks like, and
        silently reading privileged columns out of sensor columns is the
        failure this whole split exists to prevent.
        """
        arr = np.asarray(obs)
        width = arr.shape[-1] if arr.ndim > 1 else None
        if not self.privileged:
            return arr
        if width != self.shape[0]:
            raise ValueError(
                f"actor_obs: observations are {width} wide and this ObsSpec "
                f"describes {self.shape[0]}, of which fields "
                f"{list(self.privileged)} are the critic's "
                f"({self.actor.shape[0]} entries reach the actor). Pass the whole "
                f"observation the environment emits and let this cut it, or build "
                f"the agent from the spec that matches what you are passing."
            )
        return np.concatenate([arr[..., a:b] for a, b in self.actor_spans], axis=-1)

    def require_actor(self, where: str) -> "ObsSpec":
        """This spec, or a refusal naming the fields the actor must not read.

        For a component whose single network both acts and learns, which has
        nowhere to put a critic-only field.
        """
        if self.privileged:
            raise ValueError(
                f"{where} builds one network that both chooses actions and is "
                f"trained, so field(s) {list(self.privileged)} marked critic-only "
                f"would reach the actor. Use an algorithm with a separate critic "
                f"('ppo' or 'sac'), or drop the privileged fields from the "
                f"environment's ObsSpec."
            )
        return self

    # -- constructors ------------------------------------------------------

    @classmethod
    def image(cls, channels: int, height: int, width: int, *, scale: float = 1.0 / 255.0) -> "ObsSpec":
        """A `uint8` frame stack of `channels` planes, scaled into [0, 1]."""
        return cls(shape=(channels, height, width), kind="image", dtype="uint8",
                   scale=scale)

    @classmethod
    def vector(cls, dim: int, *, action_mask: int = 0,
               fields: tuple[ObsField, ...] = ()) -> "ObsSpec":
        """A `float32` vector of `dim` entries.

        `action_mask` declares that the final n of those entries are the
        legal-action mask, and `fields` names the spans for an asymmetric
        actor-critic, which `split_vector` sums the width of for you.
        """
        return cls(shape=(dim,), kind="vector", dtype="float32",
                   action_mask=action_mask, fields=tuple(fields))

    @classmethod
    def split_vector(cls, *fields: ObsField, action_mask: int = 0) -> "ObsSpec":
        """A vector observation declared field by field, widths summed here."""
        return cls.vector(sum(f.size for f in fields), action_mask=action_mask,
                          fields=fields)
