"""Reward wrappers. Scaling, interval clipping and sign clipping are three
different choices with three different fixed points, so each is its own
wrapper and the caller names the one it wants.
"""

from __future__ import annotations

import numpy as np

from ttrl.envs.wrappers.core import Wrapper

__all__ = ["ScaleReward", "ClipReward", "SignReward"]


class ScaleReward(Wrapper):
    """Multiply reward by a constant, leaving its sign and zeros alone."""

    def __init__(self, env, scale: float):
        super().__init__(env)
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError(f"reward scale must be finite and non-zero, got {scale}")
        self.scale = float(scale)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, float(reward) * self.scale, terminated, truncated, info


class ClipReward(Wrapper):
    """Clamp reward into [low, high], preserving magnitude inside the range."""

    def __init__(self, env, low: float, high: float):
        super().__init__(env)
        if not (np.isfinite(low) and np.isfinite(high)):
            raise ValueError(f"reward clip bounds must be finite, got [{low}, {high}]")
        if not low < high:
            raise ValueError(f"reward clip needs low < high, got [{low}, {high}]")
        self.low, self.high = float(low), float(high)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        clipped = min(max(float(reward), self.low), self.high)
        return obs, clipped, terminated, truncated, info


class SignReward(Wrapper):
    """Replace reward by its sign.

    This discards magnitude, which is the point on Atari (one Pong rally and
    one Breakout brick become equally valuable, so a single value scale works
    across games) and is wrong anywhere magnitude carries information. It is
    never applied implicitly.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, float(np.sign(reward)), terminated, truncated, info
