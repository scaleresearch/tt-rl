"""Privileged simulator state, appended as a field only the critic may read.

An asymmetric actor-critic trains a critic on what the simulator knows and a
policy on what the real system can sense. This wrapper is the environment half
of that: it appends values taken from the simulator to every observation and
declares them in the `ObsSpec` as a field with `actor=False`, which is what
stops any actor network ever being built over those columns.

The field goes after the inner observation's own entries, so the actor's half
stays the inner observation exactly as it was and a policy trained with this
wrapper in the stack is runnable without it.
"""

from __future__ import annotations

import numpy as np

from ttrl.envs.wrappers.core import Wrapper
from ttrl.spaces import ObsField, ObsSpec

__all__ = ["PrivilegedObservation"]


class PrivilegedObservation(Wrapper):
    """Append `extract(inner_env)` to every observation as a critic-only field.

    `extract` is called after each reset and step and must return `size`
    float32 values. `size` is declared rather than measured on the first call,
    so a stack whose widths disagree is refused before a network is built.
    """

    POLICY_STATE = ()

    def __init__(self, env, extract, *, name: str, size: int):
        super().__init__(env)
        inner = env.obs_spec
        if inner.kind != "vector" or inner.dtype != "float32":
            raise ValueError(
                f"PrivilegedObservation needs a float32 vector observation to "
                f"append to, got kind={inner.kind!r} dtype={inner.dtype!r}. "
                f"Privileged state is a vector field of the observation, so "
                f"flatten the inner observation first."
            )
        self._extract = extract
        self.size = int(size)
        self._spec = ObsSpec.vector(
            inner.shape[0] + self.size,
            action_mask=inner.action_mask,
            fields=(inner.fields or (ObsField("sensed", inner.shape[0]),))
                   + (ObsField(name, self.size, actor=False),),
        )

    @property
    def obs_spec(self) -> ObsSpec:
        return self._spec

    def _widen(self, obs: np.ndarray) -> np.ndarray:
        extra = np.asarray(self._extract(self.env), dtype=np.float32).reshape(-1)
        if extra.shape[0] != self.size:
            raise ValueError(
                f"PrivilegedObservation: extract returned {extra.shape[0]} values "
                f"and the field was declared {self.size} wide. Declare the width "
                f"the extractor actually produces."
            )
        return np.concatenate([np.asarray(obs, np.float32), extra])

    def reset(self, *, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        return self._widen(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._widen(obs), reward, terminated, truncated, info
