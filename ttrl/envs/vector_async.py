"""Vectorised environment with one subprocess per lane.

ALE emulation and the 84x84 preprocessing are host CPU work, and at 8 to 16
actors a Python loop over the lanes is the throughput limit of a run on this
hardware. This implementation sends every lane's command before reading any
lane's reply, so the lanes advance concurrently.

Autoreset is decided in the parent, exactly as in `SyncVectorEnv`: the parent
tracks which lanes owe a reset and sends `reset` rather than `step` to those.
The two implementations therefore produce identical trajectories from the
same seed, and the semantics `check_vec_env` asserts are implemented once.

The processes and the pipes live in `ttrl.envs._children.LaneGroup`. This file
owns the protocol and the autoreset bookkeeping and never touches a `Process`.

`SyncVectorEnv` stays the simple default. Reach for this one when host-side
env stepping is measurably the bottleneck.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ttrl.envs._children import LaneGroup
from ttrl.envs.base import FINAL_OBS_KEY
from ttrl.envs.episodes import LaneEpisodes, require_episode_state
from ttrl.envs.obs_batch import LaneStore
from ttrl.envs.wrappers.core import refuse_a_mismatched_stack
from ttrl.spaces import ActionSpec, ObsGroup

__all__ = ["AsyncVectorEnv"]


class AsyncVectorEnv:
    """`VecEnv` over one daemon subprocess per lane.

    See `LaneGroup` for what `env_fns` and `context` have to be: the children
    are spawned, so the factories are pickled.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Any]],
        *,
        context: str = "spawn",
    ):
        if not env_fns:
            raise ValueError("AsyncVectorEnv needs at least one env factory")
        self._lanes = LaneGroup(env_fns, context=context)
        n = self._lanes.num_envs

        specs = self._lanes.collect("startup")
        lane_zero = specs[0][0]
        self._action_spec: ActionSpec = specs[0][1]
        for i, (obs_spec, action_spec) in enumerate(specs[1:], start=1):
            if obs_spec != lane_zero:
                raise ValueError(
                    f"lane {i} declares obs_spec {obs_spec}, lane 0 declares "
                    f"{lane_zero}. All lanes must share one spec."
                )
            if action_spec != self._action_spec:
                raise ValueError(
                    f"lane {i} declares action_spec {action_spec}, lane 0 "
                    f"declares {self._action_spec}. All lanes must share one spec."
                )
        self._store = LaneStore(lane_zero, n)
        self._episodes = LaneEpisodes(n)
        self._rewards = np.zeros(n, dtype=np.float32)
        self._terminated = np.zeros(n, dtype=bool)
        self._truncated = np.zeros(n, dtype=bool)
        self._pending_reset = np.zeros(n, dtype=bool)

    @property
    def num_envs(self) -> int:
        return self._lanes.num_envs

    @property
    def obs_spec(self) -> ObsGroup:
        return self._store.group

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    @property
    def pending_reset(self) -> np.ndarray:
        return self._pending_reset.copy()

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._lanes.require_open()
        for lane in range(self.num_envs):
            self._lanes.send(lane, "reset", None if seed is None else int(seed) + lane)
        infos = []
        for lane, (obs, info) in enumerate(self._lanes.collect("reset")):
            self._store.store(lane, obs)
            infos.append(info)
        self._rewards[:] = 0.0
        self._terminated[:] = False
        self._truncated[:] = False
        self._pending_reset[:] = False
        self._episodes.reset()
        return self._store.snapshot(), {"lane": infos}

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        self._lanes.require_open()
        actions = np.asarray(actions)
        if actions.shape[0] != self.num_envs:
            raise ValueError(
                f"step got {actions.shape[0]} actions for {self.num_envs} lanes"
            )
        # Dispatch every lane before reading any reply; this is the whole point.
        for lane in range(self.num_envs):
            if self._pending_reset[lane]:
                self._lanes.send(lane, "reset", None)
            else:
                self._lanes.send(lane, "step", actions[lane])

        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        finals: list[Any] = [None] * self.num_envs
        for lane, reply in enumerate(self._lanes.collect("step")):
            if self._pending_reset[lane]:
                obs, info = reply
                reward, terminated, truncated = 0.0, False, False
            else:
                obs, reward, terminated, truncated, info = reply
            self._store.store(lane, obs)
            self._rewards[lane] = reward
            self._terminated[lane] = bool(terminated)
            self._truncated[lane] = bool(truncated)
            if terminated or truncated:
                finals[lane] = self._store.lane(lane)
            infos[lane] = info
        resetting = self._pending_reset.copy()
        self._pending_reset = self._terminated | self._truncated
        infos = self._episodes.observe(
            self._rewards, self._pending_reset, infos, resetting)
        return (
            self._store.snapshot(),
            self._rewards.copy(),
            self._terminated.copy(),
            self._truncated.copy(),
            {"lane": infos, FINAL_OBS_KEY: finals},
        )

    def save_state(self) -> dict[str, Any]:
        self._lanes.require_open()
        self._lanes.broadcast("get_state")
        return {
            "lanes": self._lanes.collect("save_state"),
            "pending_reset": self._pending_reset.copy(),
            "episodes": self._episodes.get_state(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._lanes.require_open()
        lanes = state["lanes"]
        if len(lanes) != self.num_envs:
            raise ValueError(
                f"checkpoint holds {len(lanes)} lanes, this env has "
                f"{self.num_envs}. A lane's state belongs to the lane that "
                f"produced it, so set env.n_envs={len(lanes)} to resume this "
                f"run, or start afresh at the lane count you want."
            )
        pending = np.asarray(state["pending_reset"], dtype=bool)
        if pending.shape != (self.num_envs,):
            raise ValueError(
                f"checkpoint pending_reset has shape {pending.shape}, "
                f"expected {(self.num_envs,)}, one flag per lane. The file was "
                f"not written by this code; start the run afresh."
            )
        self._episodes.set_state(require_episode_state(state))
        for lane in range(self.num_envs):
            self._lanes.send(lane, "set_state", lanes[lane])
        self._lanes.collect("load_state")
        self._pending_reset = pending

    def save_policy_state(self) -> list[tuple[str, dict[str, Any]]]:
        """Lane 0's policy state, which is every lane's.

        Every lane is asked, because a lane whose reply is left in the pipe
        desynchronises the next command. See `SyncVectorEnv.save_policy_state`.
        """
        self._lanes.require_open()
        self._lanes.broadcast("get_policy_state")
        return self._lanes.collect("save_policy_state")[0]

    def save_run_state(self) -> list[tuple[str, dict[str, Any]]]:
        """No lane holds run state. See `SyncVectorEnv.save_run_state`."""
        self._lanes.require_open()
        return []

    def load_run_state(self, snapshot) -> None:
        self._lanes.require_open()
        refuse_a_mismatched_stack([], [name for name, _ in snapshot], "run")

    def load_policy_state(self, snapshot: list[tuple[str, dict[str, Any]]]) -> None:
        self._lanes.require_open()
        self._lanes.broadcast("set_policy_state", snapshot)
        self._lanes.collect("load_policy_state")

    def close(self) -> None:
        self._lanes.close()

    def __enter__(self) -> AsyncVectorEnv:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
