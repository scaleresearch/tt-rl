"""Wrappers that are only correct for the ALE.

Each one depends on something no other environment has: an action set with a
named FIRE button, a life counter, or flickering sprites that have to be
max-pooled away. They take the emulator explicitly rather than searching the
wrapper stack for it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ttrl.envs.atari.emulator import AtariEmulator
from ttrl.envs.wrappers.core import Wrapper

__all__ = ["NoopReset", "MaxAndSkip", "EpisodicLife", "FireReset"]


class NoopReset(Wrapper):
    """Start each episode after a random number of no-op actions.

    ALE games are deterministic from the first frame, so without this every
    episode begins from the same state and the agent can memorise an opening.
    The no-op action index is a parameter because "do nothing" is only action
    0 by ALE convention.
    """

    POLICY_STATE = ()

    def __init__(self, env, *, noop_max: int = 30, noop_action: int = 0):
        super().__init__(env)
        if noop_max < 1:
            raise ValueError(f"noop_max must be >= 1, got {noop_max}")
        spec = env.action_spec
        if spec.kind != "discrete" or not 0 <= noop_action < spec.n:
            raise ValueError(
                f"noop_action {noop_action} is not a valid action for {spec}"
            )
        self.noop_max, self.noop_action = noop_max, noop_action
        self._rng = np.random.default_rng()

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = self.env.reset(seed=seed)
        for _ in range(int(self._rng.integers(1, self.noop_max + 1))):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                # Re-seeding here would rewind the RNG to the state that just
                # produced the failure, so the recovery reproduces it.
                obs, info = self.env.reset()
        return obs, info

    def get_state(self) -> dict[str, Any]:
        return {"rng": self._rng.bit_generator.state}

    def set_state(self, state: dict[str, Any]) -> None:
        self._rng.bit_generator.state = state["rng"]


class MaxAndSkip(Wrapper):
    """Repeat an action `skip` frames and return the pixelwise max of the last two.

    Atari sprites flicker between frames, so a single frame can omit the ball.
    Generic `ActionRepeat` cannot do this: taking a maximum over observations
    is only meaningful for pixels.
    """

    POLICY_STATE = ()

    def __init__(self, env, *, skip: int = 4):
        super().__init__(env)
        if skip < 1:
            raise ValueError(f"skip must be >= 1, got {skip}")
        self._skip = skip
        self._buf = np.zeros((2, *env.obs_spec.shape), dtype=env.obs_spec.numpy_dtype)

    def step(self, action):
        total = 0.0
        produced = 0
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._buf[i % 2] = obs
            produced += 1
            total += float(reward)
            if terminated or truncated:
                break
        # Max only over frames this call actually produced. With one frame
        # there is no "last two" to pool, and pooling against the buffer would
        # leak a frame from the previous step into the observation.
        frame = obs if produced < 2 else self._buf.max(axis=0)
        return frame, total, terminated, truncated, info

    def reset(self, *, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        self._buf[:] = 0
        self._buf[1] = obs
        return obs, info

    def get_state(self) -> dict[str, Any]:
        return {"buf": self._buf.copy()}

    def set_state(self, state: dict[str, Any]) -> None:
        self._buf[:] = state["buf"]


class EpisodicLife(Wrapper):
    """End the learner's episode on life loss without resetting the game.

    This shortens credit assignment dramatically on games with lives. It
    changes where episode boundaries fall, so the score it implies is not the
    game score the ALE evaluation protocol asks for.
    """

    POLICY_STATE = ()

    def __init__(self, env, emulator: AtariEmulator):
        super().__init__(env)
        self._emulator = emulator
        self._lives = 0
        self._real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._real_done = terminated or truncated
        lives = self._emulator.ale.lives()
        if 0 < lives < self._lives:
            terminated = True
        self._lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None):
        if self._real_done:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset()
        self._lives = self._emulator.ale.lives()
        return obs, info

    def get_state(self) -> dict[str, Any]:
        return {"lives": self._lives, "real_done": self._real_done}

    def set_state(self, state: dict[str, Any]) -> None:
        self._lives = int(state["lives"])
        self._real_done = bool(state["real_done"])


class FireReset(Wrapper):
    """Press FIRE whenever play has to be started, at reset and after a life loss.

    The FIRE index comes from the emulator's own action meanings, so a game
    whose minimal action set orders its buttons differently still works, and a
    game with no FIRE button is left alone.

    Breakout-likes need FIRE to put the ball back in play, and losing a life
    takes it out of play again. Under episodic life a life loss ends the
    learner's episode and the next `reset` presses FIRE, but the ALE
    evaluation protocol runs whole games, where nothing resets mid-episode: a
    deterministic policy that never chooses FIRE then holds a ball-less game
    open forever. Pressing FIRE here makes the launch a property of the game
    rather than of the episode boundary.
    """

    POLICY_STATE = ()

    #: A game that dies on FIRE this many times running is broken, not unlucky.
    MAX_ATTEMPTS = 10

    def __init__(self, env, emulator: AtariEmulator):
        super().__init__(env)
        self._emulator = emulator
        meanings = emulator.action_meanings()
        self.fire_action = meanings.index("FIRE") if "FIRE" in meanings else None
        self._lives = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = self._emulator.ale.lives()
        relaunch = (
            self.fire_action is not None
            and 0 < lives < self._lives
            and not (terminated or truncated)
        )
        self._lives = lives
        if relaunch:
            obs, fire_reward, terminated, truncated, info = self.env.step(
                self.fire_action
            )
            reward += fire_reward
            self._lives = self._emulator.ale.lives()
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        self._lives = self._emulator.ale.lives()
        if self.fire_action is None:
            return obs, info
        # The observation this returns must be a started game. If FIRE ends the
        # episode, the fresh episode still needs its own FIRE, so press again.
        for _ in range(self.MAX_ATTEMPTS):
            obs, _, terminated, truncated, info = self.env.step(self.fire_action)
            if not (terminated or truncated):
                self._lives = self._emulator.ale.lives()
                return obs, info
            obs, info = self.env.reset()
            self._lives = self._emulator.ale.lives()
        raise RuntimeError(
            f"FireReset: the episode ended on the FIRE action "
            f"{self.MAX_ATTEMPTS} times in a row, so this game is not startable"
        )

    def get_state(self) -> dict[str, Any]:
        return {"lives": self._lives}

    def set_state(self, state: dict[str, Any]) -> None:
        self._lives = int(state["lives"])
