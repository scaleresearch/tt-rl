"""The ALE itself, behind the `Env` contract, plus its state snapshot."""

from __future__ import annotations

from typing import Any

from ttrl.envs.gym_adapter import GymAdapter
from ttrl.envs.wrappers.core import require_module

__all__ = ["AtariEmulator", "make_emulator"]


class AtariEmulator(GymAdapter):
    """One ALE game. Raw (3, 210, 160) uint8 frames, no preprocessing.

    Declares its state so `snapshot_env` can checkpoint a trajectory exactly:
    the emulator RAM and CPU state, the emulator RNG that drives sticky
    actions, and Gymnasium's own generator, which is a different RNG and is
    what the no-op reset draws from.
    """

    #: The ALE core snapshot continues one trajectory and nothing else.
    POLICY_STATE = ()

    @property
    def episode_bound(self) -> int:
        """Emulator frames one episode can take, which the ALE enforces itself.

        `ttrl.envs.bounds` reads this. The ALE truncates an episode at
        `max_num_frames_per_episode` and reports it through
        `game_truncated()`, so the bound is declared where it is applied
        rather than by a wrapper above it. `make_emulator` sets the number
        from `AtariConfig.max_frames`.
        """
        return int(self.ale.getInt("max_num_frames_per_episode"))

    @property
    def ale(self):
        return self.gym_env.unwrapped.ale

    def action_meanings(self) -> list[str]:
        return list(self.gym_env.unwrapped.get_action_meanings())

    def get_state(self) -> dict[str, Any]:
        # include_rng=True is load-bearing: the ALE default drops the emulator
        # RNG, which is exactly what drives sticky actions, so a restore would
        # replay a different trajectory under identical actions.
        return {
            "ale": self.ale.cloneState(include_rng=True).serialize(),
            "np_random": self.gym_env.unwrapped.np_random.bit_generator.state,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        ale_py = require_module("ale_py", "atari")
        self.ale.restoreState(ale_py._ale_py.ALEState(state["ale"]))
        self.gym_env.unwrapped.np_random.bit_generator.state = state["np_random"]


def make_emulator(
    game: str, *, sticky: float, render_mode: str | None, max_frames: int
) -> AtariEmulator:
    gym = require_module("gymnasium", "atari")
    ale_py = require_module("ale_py", "atari")
    # ALE/* ids only resolve once ale_py's envs are registered. Registration is
    # idempotent; a duplicate id warns rather than raising.
    gym.register_envs(ale_py)
    env = gym.make(
        game,
        frameskip=1,  # max-and-skip below does its own, with pixel max-pooling
        repeat_action_probability=sticky,
        render_mode=render_mode,
        # Stated rather than left to the id's registered default, so the
        # episode bound is this library's number and `AtariConfig.max_frames`
        # is the one place it is written.
        max_num_frames_per_episode=max_frames,
    )
    return AtariEmulator(env, image_layout="hwc")
