"""Lane-major observation storage for a vector environment.

THE promotion site. This is the one place in the package where a bare
`ObsSpec` becomes a one-leaf `ObsGroup` and where host arrays become an
`Observation`. Both vector environments assemble their lanes here, and every
consumer above them (a transition, a buffer, an agent, the device buffers)
takes a group and an observation, so none of them asks which form it holds.

A single environment declares either an `ObsSpec`, which is the single-array
case, or an `ObsGroup` whose leaves it returns as a mapping. One array per
leaf, because leaves differ in dtype and kind and no single array holds them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ttrl.spaces import ObsGroup, Observation

__all__ = ["LaneStore", "observation_group"]


def observation_group(spec: Any) -> ObsGroup:
    """The `ObsGroup` an environment's declared spec describes.

    The single promotion site. `ObsGroup.of` is called here and nowhere else,
    so a bare `ObsSpec` becomes a one-leaf group exactly once, at the
    environment boundary, and every reader above it holds a group.
    """
    return ObsGroup.of(spec)


class LaneStore:
    """One lane-major array per leaf, written lane by lane.

    `snapshot` copies, because the arrays are reused on the next step and a
    consumer that stored a view of them would see its observation change
    underneath it.
    """

    def __init__(self, spec: Any, lanes: int):
        self.group: ObsGroup = observation_group(spec)
        self.lanes = int(lanes)
        self.arrays = {
            name: np.zeros((self.lanes, *leaf.shape), dtype=leaf.numpy_dtype)
            for name, leaf in self.group.specs()
        }

    def store(self, lane: int, obs) -> None:
        """Validate one lane's observation against the spec and copy it in."""
        try:
            one = self.group.observation(obs)
        except (TypeError, ValueError) as exc:
            raise type(exc)(
                f"lane {lane} returned an observation this obs_spec refuses: {exc}"
            ) from exc
        if one.lanes is not None:
            raise ValueError(
                f"lane {lane} returned {one.lanes} observations; one lane steps "
                f"once and returns one observation"
            )
        for name, array in self.arrays.items():
            array[lane] = one[name]

    def snapshot(self) -> Observation:
        """Every lane's observation, copied out of this store."""
        return self.group.observation(
            {name: array.copy() for name, array in self.arrays.items()})

    def lane(self, index: int) -> Observation:
        """One lane's observation, copied. This is an ending lane's final one."""
        return self.group.observation(
            {name: np.array(array[index], copy=True)
             for name, array in self.arrays.items()})
