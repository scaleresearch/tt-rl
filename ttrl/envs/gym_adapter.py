"""Adapter from a Gymnasium environment to the `ttrl.envs.base.Env` contract.

Gymnasium spaces carry more freedom than the framework needs (object dtypes,
unbounded boxes, channel orders) so the translation is explicit and refuses
anything it cannot describe with an `ObsSpec`. This is the single place where
`gymnasium` is required, and it is imported at call time so a core install
never pays for it.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ttrl.envs.render import check_frame
from ttrl.envs.wrappers.core import require_module
from ttrl.spaces import ActionSpec, ObsSpec

__all__ = ["GymAdapter", "obs_spec_from_space", "action_spec_from_space"]

ImageLayout = Literal["hwc", "chw"]


def obs_spec_from_space(space, *, image_layout: ImageLayout | None) -> ObsSpec:
    gym = require_module("gymnasium", "atari")
    if not isinstance(space, gym.spaces.Box):
        raise TypeError(
            f"observation space {type(space).__name__} is not supported. "
            f"ObsSpec describes Box observations; adapt anything else yourself."
        )
    if space.dtype == np.uint8 and len(space.shape) == 3:
        if image_layout is None:
            raise ValueError(
                f"uint8 observation of shape {space.shape} looks like an image, "
                f"so GymAdapter needs image_layout='hwc' or 'chw'. Guessing from "
                f"the channel count would silently transpose a 3-frame stack."
            )
        if image_layout == "hwc":
            h, w, c = space.shape
        else:
            c, h, w = space.shape
        return ObsSpec(shape=(c, h, w), dtype="uint8", kind="image", scale=1.0 / 255.0)
    if len(space.shape) == 1:
        return ObsSpec(shape=(int(space.shape[0]),), dtype="float32", kind="vector")
    raise ValueError(
        f"observation space Box(shape={space.shape}, dtype={space.dtype}) is "
        f"neither a uint8 image nor a 1-D vector, so it has no ObsSpec. "
        f"Flatten or reshape it in a wrapper first."
    )


def action_spec_from_space(space) -> ActionSpec:
    gym = require_module("gymnasium", "atari")
    if isinstance(space, gym.spaces.Discrete):
        return ActionSpec.discrete(int(space.n))
    if isinstance(space, gym.spaces.Box):
        if len(space.shape) != 1:
            raise ValueError(
                f"continuous action space must be 1-D, got shape {space.shape}"
            )
        if not (np.isfinite(space.low).all() and np.isfinite(space.high).all()):
            raise ValueError(
                f"continuous action bounds must be finite, got low={space.low} "
                f"high={space.high}. An unbounded action space has no ActionSpec."
            )
        return ActionSpec.continuous(space.low, space.high)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return ActionSpec.multi_discrete(np.asarray(space.nvec).reshape(-1))
    if isinstance(space, gym.spaces.MultiBinary):
        if np.asarray(space.n).size != 1:
            raise ValueError(
                f"MultiBinary action space of shape {space.shape} must be 1-D, so "
                f"its switches are one flat vector; reshape it in a wrapper first."
            )
        return ActionSpec.multi_binary(int(np.asarray(space.n).reshape(-1)[0]))
    raise TypeError(
        f"action space {type(space).__name__} is not supported. ActionSpec "
        f"describes Discrete, MultiDiscrete, MultiBinary and bounded 1-D Box "
        f"actions."
    )


class GymAdapter:
    """Wrap one Gymnasium env, presenting the `Env` protocol.

    Observations are cast to the declared dtype and, for images, moved to
    channels-first, so every return matches `obs_spec` exactly.
    """

    def __init__(self, env, *, image_layout: ImageLayout | None = None):
        self.gym_env = env
        self._image_layout = image_layout
        self._obs_spec = obs_spec_from_space(
            env.observation_space, image_layout=image_layout
        )
        self._action_spec = action_spec_from_space(env.action_space)

    @property
    def obs_spec(self) -> ObsSpec:
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    @property
    def unwrapped(self):
        return self.gym_env.unwrapped

    def can_render(self) -> bool:
        return self.gym_env.render_mode == "rgb_array"

    def render(self) -> np.ndarray:
        if not self.can_render():
            raise RuntimeError(
                f"this environment was built with render_mode="
                f"{self.gym_env.render_mode!r}, so Gymnasium returns no frame. "
                f"Rebuild it with render_mode='rgb_array'."
            )
        return check_frame(np.asarray(self.gym_env.render()))

    def _convert(self, obs) -> np.ndarray:
        arr = np.asarray(obs)
        if self._obs_spec.kind == "image" and self._image_layout == "hwc":
            arr = np.moveaxis(arr, -1, 0)
        return np.ascontiguousarray(arr, dtype=self._obs_spec.numpy_dtype)

    def _action(self, action):
        if self._action_spec.kind == "discrete":
            return int(action)
        if self._action_spec.kind == "multi_discrete":
            return np.asarray(action, dtype=np.int64).reshape(
                len(self._action_spec.nvec))
        return np.asarray(action, dtype=np.float32).reshape(self._action_spec.dim)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.gym_env.reset(seed=seed)
        return self._convert(obs), dict(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.gym_env.step(
            self._action(action)
        )
        return (
            self._convert(obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info),
        )

    def close(self) -> None:
        self.gym_env.close()
