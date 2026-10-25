"""Observation wrappers: stacking and resizing.

Running-mean whitening is NOT here. It needs one estimator shared by every
lane, and a wrapper at this level is built inside each lane's factory, so it
lives above the vector environment instead: see
`ttrl.envs.vec_normalize.NormalizeObservation`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ttrl.envs.wrappers.core import Wrapper, require_module
from ttrl.spaces import ObsSpec

__all__ = ["FrameStack", "ResizeObservation"]


class FrameStack(Wrapper):
    """Stack the last `k` observations so velocity becomes observable.

    Images stack on the channel axis, (C, H, W) -> (k*C, H, W). Vectors
    concatenate, (D,) -> (k*D,). On reset the history is filled with the first
    observation rather than zero-padded, which keeps the first few frames of
    every episode in distribution.
    """

    POLICY_STATE = ()

    def __init__(self, env, k: int):
        super().__init__(env)
        if k < 1:
            raise ValueError(f"frame stack k must be >= 1, got {k}")
        inner = env.obs_spec
        self.k = k
        if inner.kind == "image":
            c, h, w = inner.shape
            shape: tuple[int, ...] = (k * c, h, w)
        else:
            shape = (k * inner.shape[0],)
        # A stacked observation repeats every declared field once per frame, so
        # the division between what the actor may read and what only the
        # critic may read is repeated with it. An inner spec that declares no
        # fields stays undivided, since the comprehension is then empty.
        fields = tuple(replace(f, name=f"{f.name}/{i}")
                       for i in range(k) for f in inner.fields)
        self._spec = replace(inner, shape=shape, fields=fields)
        self._frames = np.zeros((k, *inner.shape), dtype=inner.numpy_dtype)

    @property
    def obs_spec(self) -> ObsSpec:
        return self._spec

    def _stacked(self) -> np.ndarray:
        return self._frames.reshape(self._spec.shape).copy()

    def reset(self, *, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        self._frames[:] = obs
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Basic-slice self-assignment: numpy buffers the overlap, so this is
        # element-for-element np.roll(..., -1, axis=0) without the allocation.
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = obs
        return self._stacked(), reward, terminated, truncated, info

    def get_state(self) -> dict[str, Any]:
        return {"frames": self._frames.copy()}

    def set_state(self, state: dict[str, Any]) -> None:
        self._frames[:] = state["frames"]


class ResizeObservation(Wrapper):
    """Resize an image observation, optionally collapsing colour to grey.

    Channels-first in and out. INTER_AREA is the interpolation the DQN
    baselines use for downscaling.
    """

    def __init__(self, env, height: int, width: int, *, greyscale: bool = False):
        super().__init__(env)
        inner = env.obs_spec
        if inner.kind != "image":
            raise ValueError(
                f"ResizeObservation needs an image observation, got kind "
                f"{inner.kind!r} with shape {inner.shape}"
            )
        if height < 1 or width < 1:
            raise ValueError(f"resize target must be positive, got {height}x{width}")
        channels = inner.shape[0]
        if greyscale and channels != 3:
            raise ValueError(
                f"greyscale=True needs 3 colour channels to convert from, got {channels}"
            )
        self.height, self.width, self.greyscale = height, width, greyscale
        out_c = 1 if greyscale else channels
        self._spec = replace(inner, shape=(out_c, height, width))
        self._cv2 = None

    @property
    def obs_spec(self) -> ObsSpec:
        return self._spec

    def _convert(self, obs: np.ndarray) -> np.ndarray:
        if self._cv2 is None:
            self._cv2 = require_module("cv2", "atari")
        cv2 = self._cv2
        hwc = np.moveaxis(obs, 0, -1)
        if self.greyscale:
            hwc = cv2.cvtColor(hwc, cv2.COLOR_RGB2GRAY)[:, :, None]
        out = cv2.resize(
            hwc, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        if out.ndim == 2:
            out = out[:, :, None]
        return np.ascontiguousarray(
            np.moveaxis(out, -1, 0), dtype=self._spec.numpy_dtype
        )

    def reset(self, *, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        return self._convert(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._convert(obs), reward, terminated, truncated, info
