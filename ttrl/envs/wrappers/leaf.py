"""Privileged simulator state as a critic-only LEAF beside the observation.

`PrivilegedObservation` appends privileged values to a vector observation as a
critic-only field of that same array, which is the right shape when what the
actor senses is itself a vector. It cannot be the right shape when the actor
senses a camera: an image and a state vector share neither dtype nor rank, so
they cannot be spans of one array.

`PrivilegedLeaf` is that case. It leaves the inner observation exactly as it
is, as its own leaf, and adds a second leaf marked `actor=False`. The actor's
group is then the inner leaf alone, so a policy trained with this wrapper in
the stack runs unchanged against the environment without it, which is the same
guarantee the field-level wrapper gives.

This is the image-plus-privileged robotics case: camera frames for the actor,
the simulator's own state for the critic.
"""

from __future__ import annotations

import numpy as np

from ttrl.envs.obs_batch import observation_group
from ttrl.envs.wrappers.core import Wrapper
from ttrl.spaces import ObsGroup, ObsLeaf, ObsSpec

__all__ = ["PrivilegedLeaf"]


class PrivilegedLeaf(Wrapper):
    """Add `extract(inner_env)` as a critic-only leaf of every observation.

    `extract` is called after each reset and step and must return `size`
    float32 values. `size` is declared rather than measured on the first call,
    so a stack whose widths disagree is refused before a network is built.

    An inner environment declaring a bare `ObsSpec` has it promoted to the
    one-leaf group by the same promotion site every environment goes through,
    so the inner leaf's name is the default one; an inner environment that
    declares a group of its own keeps its leaf names.
    """

    POLICY_STATE = ()

    def __init__(self, env, extract, *, name: str, size: int):
        super().__init__(env)
        self._inner = observation_group(env.obs_spec)
        if name in self._inner.names:
            raise ValueError(
                f"PrivilegedLeaf: the inner observation already has a leaf "
                f"named {name!r} ({list(self._inner.names)}); give the "
                f"privileged leaf its own name"
            )
        self._extract = extract
        self.name = str(name)
        self.size = int(size)
        self._spec = ObsGroup(
            self._inner.leaves
            + (ObsLeaf(self.name, ObsSpec.vector(self.size), actor=False),)
        )

    @property
    def obs_spec(self) -> ObsGroup:
        return self._spec

    def _with_leaf(self, obs) -> dict:
        extra = np.asarray(self._extract(self.env), dtype=np.float32).reshape(-1)
        if extra.shape[0] != self.size:
            raise ValueError(
                f"PrivilegedLeaf: extract returned {extra.shape[0]} values and "
                f"the leaf was declared {self.size} wide. Declare the width the "
                f"extractor actually produces."
            )
        inner = self._inner.observation(obs)
        out = {leaf: inner[leaf] for leaf in self._inner.names}
        out[self.name] = extra
        return out

    def reset(self, *, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        return self._with_leaf(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._with_leaf(obs), reward, terminated, truncated, info
