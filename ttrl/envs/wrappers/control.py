"""Control-flow wrappers: time limit, action repeat, episode statistics."""

from __future__ import annotations

from typing import Any

from ttrl.envs.wrappers.core import Wrapper

__all__ = ["TimeLimit", "ActionRepeat", "RecordEpisodeStatistics"]


class TimeLimit(Wrapper):
    """Truncate an episode after `max_steps` steps of this wrapper."""

    POLICY_STATE = ()

    def __init__(self, env, max_steps: int):
        super().__init__(env)
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        self.max_steps = max_steps
        self._elapsed = 0

    @property
    def episode_bound(self) -> int:
        """`ttrl.envs.bounds`: this is what the wrapper is for."""
        return self.max_steps

    def reset(self, *, seed: int | None = None):
        self._elapsed = 0
        return self.env.reset(seed=seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed += 1
        if self._elapsed >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info

    def get_state(self) -> dict[str, Any]:
        return {"elapsed": self._elapsed}

    def set_state(self, state: dict[str, Any]) -> None:
        self._elapsed = int(state["elapsed"])


class ActionRepeat(Wrapper):
    """Hold one action for `repeat` inner steps and sum the reward.

    Stops early on termination or truncation, so the returned observation is
    always the one that belongs with the returned done flags.
    """

    def __init__(self, env, repeat: int):
        super().__init__(env)
        if repeat < 1:
            raise ValueError(f"action repeat must be >= 1, got {repeat}")
        # An action legal in the state it was chosen for can be illegal in the
        # state the next inner step starts from, and this wrapper would send it
        # anyway. Choosing one action per state is the only honest repeat here.
        if env.obs_spec.action_mask:
            raise ValueError(
                f"ActionRepeat cannot hold an action over an environment that "
                f"declares a legal-action mask of width "
                f"{env.obs_spec.action_mask}: the repeat would replay an action "
                f"the next state may forbid. Step this environment once per "
                f"action."
            )
        self.repeat = repeat

    def step(self, action):
        total = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        return obs, total, terminated, truncated, info


class RecordEpisodeStatistics(Wrapper):
    """Accumulate undiscounted return and length, emitted at episode end.

    On the terminal step `info["episode"]` carries `{"r": return, "l": length}`
    for the episode that just ended.

    OPTIONAL. `SyncVectorEnv` and `AsyncVectorEnv` accumulate the same two
    numbers themselves (`ttrl.envs.episodes`), so no environment needs this
    to be scored. What it is still for is the one thing a vector-level
    accumulator cannot do: a lane that transforms its own reward
    (`ScaleReward`, `SignReward`, `ClipReward`) must have the return taken
    BELOW that transform, and only a wrapper inside the lane sits there. The
    number it reports is then the environment's own reward, which is the one a
    result table is allowed to quote. A lane that supplies the key wins over
    the vector-level accumulator, which never overwrites it.
    """

    POLICY_STATE = ()

    def __init__(self, env):
        super().__init__(env)
        self._return = 0.0
        self._length = 0

    @property
    def episode_return(self) -> float:
        """Return accumulated so far in the current episode.

        A caller that ends an episode itself, such as an evaluation step
        budget, never sees the terminal step and still needs this.
        """
        return self._return

    @property
    def episode_length(self) -> int:
        return self._length

    def reset(self, *, seed: int | None = None):
        self._return, self._length = 0.0, 0
        return self.env.reset(seed=seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._return += float(reward)
        self._length += 1
        if terminated or truncated:
            info = dict(info)
            info["episode"] = {"r": self._return, "l": self._length}
            self._return, self._length = 0.0, 0
        return obs, reward, terminated, truncated, info

    def get_state(self) -> dict[str, Any]:
        return {"return": self._return, "length": self._length}

    def set_state(self, state: dict[str, Any]) -> None:
        self._return = float(state["return"])
        self._length = int(state["length"])
