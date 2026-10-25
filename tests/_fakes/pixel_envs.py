"""A camera-plus-privileged-state environment, learnable by one small policy.

Not a test module. The robotics shape in miniature: the actor sees pixels, the
critic also sees the simulator's own state, and the optimal action is a
function of what the pixels encode, so a policy that learns from the camera
alone can solve it and one that silently read the privileged leaf would too.
That is why the leak is tested separately from the learning.
"""

from __future__ import annotations

import numpy as np

from ttrl.envs.wrappers import PrivilegedLeaf
from ttrl.spaces import ActionSpec, ObsSpec

__all__ = ["CAMERA", "PixelReachEnv", "STATE_SIZE", "pixel_reach"]

#: 68 rather than 84 because the strides must divide: 68 -> 16 -> 7 -> 5.
CAMERA = ObsSpec.image(3, 68, 68)
STATE_SIZE = 4
EPISODE_LENGTH = 8


class PixelReachEnv:
    """Emit a camera frame whose channel levels encode the target action.

    The target is two values in (-1, 1). Channel `i` of the frame is filled
    with the byte that encodes target `i`, so the optimal action is an affine
    function of a mean over pixels, which one convolution and one linear layer
    can represent.
    """

    obs_spec = CAMERA
    action_spec = ActionSpec.continuous([-1.0, -1.0], [1.0, 1.0])

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._return = 0.0
        self.target = np.zeros(2, np.float32)
        self._draw()

    def _draw(self) -> None:
        self.target = self._rng.uniform(-0.9, 0.9, 2).astype(np.float32)

    @property
    def state(self) -> np.ndarray:
        """What only the critic may read: the exact target and the clock."""
        return np.concatenate([
            self.target,
            np.array([self._t, self._return], np.float32),
        ]).astype(np.float32)

    def _frame(self) -> np.ndarray:
        levels = ((self.target + 1.0) * 127.0).astype(np.uint8)
        frame = np.zeros(CAMERA.shape, np.uint8)
        frame[0] = levels[0]
        frame[1] = levels[1]
        frame[2] = 7
        return frame

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._return = 0.0
        self._draw()
        return self._frame(), {}

    def step(self, action):
        act = np.asarray(action, np.float32).reshape(-1)
        reward = float(-np.sum((act - self.target) ** 2))
        self._t += 1
        self._return += reward
        truncated = self._t >= EPISODE_LENGTH
        info = {}
        if truncated:
            info["episode"] = {"r": self._return, "l": self._t}
        self._draw()
        return self._frame(), reward, False, truncated, info

    def close(self) -> None:
        pass


def pixel_reach(seed: int = 0):
    """`PixelReachEnv` with its simulator state as a critic-only leaf."""
    return PrivilegedLeaf(PixelReachEnv(seed), lambda env: env.state,
                          name="sim_state", size=STATE_SIZE)
