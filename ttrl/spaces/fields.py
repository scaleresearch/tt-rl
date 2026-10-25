"""The actor/critic division of one vector observation, declared field by field.

`ObsField` is the declaration and the two functions here are its validation,
which is the whole of what the division costs outside `ObsSpec` itself: a
field is a named contiguous span of the observation vector, and `actor=False`
marks a span as privileged simulator state the critic may condition on and no
actor network is ever built over.

Split from `ttrl.spaces.obs` because the two are different subjects. That
module owns what one observation IS (shape, dtype, kind, scale, and where the
legal-action mask sits in it); this one owns who is allowed to read which part
of it. The checks are free functions taking the spec rather than methods on
it, so nothing here imports `ObsSpec` and the direction of the dependency
stays one way.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ObsField", "check_fields"]


@dataclass(frozen=True)
class ObsField:
    """One named contiguous span of a vector observation.

    `actor=False` marks the span as the critic's alone, which is what makes an
    asymmetric actor-critic expressible: privileged simulator state goes in
    such a field and no actor network is ever built over it.
    """

    name: str
    size: int
    actor: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ObsField needs a non-empty name")
        if self.size < 1:
            raise ValueError(
                f"ObsField {self.name!r} has size {self.size}; a field covers at "
                f"least one entry of the observation vector"
            )


def check_fields(spec) -> None:
    """Assert `spec.fields` covers `spec` and leaves the actor an input."""
    if not spec.fields:
        return
    if spec.kind != "vector":
        raise ValueError(
            f"fields describe spans of a vector observation, and this spec is "
            f"an image of shape {spec.shape}. An image is one field: leave "
            f"fields empty, and carry privileged state in a separate vector "
            f"observation."
        )
    names = [f.name for f in spec.fields]
    if len(set(names)) != len(names):
        raise ValueError(f"obs field names must be unique, got {names}")
    total = sum(f.size for f in spec.fields)
    if total != spec.shape[0]:
        raise ValueError(
            f"obs fields {names} cover {total} entries but the observation is "
            f"{spec.shape[0]} wide. Every entry belongs to exactly one field, "
            f"so make the sizes sum to {spec.shape[0]}."
        )
    if not any(f.actor for f in spec.fields):
        raise ValueError(
            f"every obs field in {names} is marked critic-only, so the actor "
            f"would have no input at all. Mark the fields the real system can "
            f"sense with actor=True."
        )
    _check_mask_fields(spec)


def _check_mask_fields(spec) -> None:
    """The mask lives in the actor's half, because the actor is what masks.

    The actor's observation is its fields in declaration order, so the mask is
    the final entries of that narrower vector only if every field the final
    `action_mask` entries fall in is the actor's.
    """
    covered = 0
    for field in reversed(spec.fields):
        if covered >= spec.action_mask:
            return
        covered += field.size
        if not field.actor:
            raise ValueError(
                f"the legal-action mask occupies the final {spec.action_mask} "
                f"entries of this observation, which fall in critic-only field "
                f"{field.name!r}. An actor chooses among legal actions, so mark "
                f"the mask's own field actor=True and put critic-only state "
                f"ahead of it."
            )
