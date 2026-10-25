"""Synchronous vectorised environment.

Autoreset is NEXT_STEP, matching Gymnasium 1.x: the step on which a lane ends
returns that lane's final observation with `terminated` or `truncated` set,
and the following step returns the new episode's first observation with both
flags False, ignoring the action passed for that lane. A replay buffer stores
(obs, action, reward, next_obs, done) straight off these returns, so a lane
that reported done twice, or that overwrote its final observation with a
reset observation, would corrupt every transition at an episode boundary.
`ttrl.envs.check_vec_env` asserts this directly.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ttrl.envs.base import FINAL_OBS_KEY
from ttrl.envs.episodes import LaneEpisodes, require_episode_state
from ttrl.envs.wrappers.core import (
    refuse_a_mismatched_stack,
    restore_env,
    restore_policy_state,
    snapshot_env,
    snapshot_policy_state,
)
from ttrl.envs.obs_batch import LaneStore
from ttrl.spaces import ActionSpec, ObsGroup

__all__ = ["SyncVectorEnv"]


class SyncVectorEnv:
    """Step a batch of environments in one process, one after another.

    Lanes are independent: the exception from a failing lane does not cancel
    the other lanes' steps, and it is never swallowed. Every lane is stepped,
    then the failures are raised together naming their lane indices.
    """

    def __init__(self, env_fns: Sequence[Callable[[], Any]]):
        if not env_fns:
            raise ValueError("SyncVectorEnv needs at least one env factory")
        self.envs = [fn() for fn in env_fns]
        self._action_spec: ActionSpec = self.envs[0].action_spec
        lane_zero = self.envs[0].obs_spec
        for i, env in enumerate(self.envs[1:], start=1):
            if env.obs_spec != lane_zero:
                raise ValueError(
                    f"lane {i} declares obs_spec {env.obs_spec}, lane 0 declares "
                    f"{lane_zero}. All lanes must share one spec."
                )
            if env.action_spec != self._action_spec:
                raise ValueError(
                    f"lane {i} declares action_spec {env.action_spec}, lane 0 "
                    f"declares {self._action_spec}. All lanes must share one spec."
                )
        n = len(self.envs)
        self._store = LaneStore(lane_zero, n)
        self._episodes = LaneEpisodes(n)
        self._rewards = np.zeros(n, dtype=np.float32)
        self._terminated = np.zeros(n, dtype=bool)
        self._truncated = np.zeros(n, dtype=bool)
        #: Lanes that ended on the previous step and owe a reset on this one.
        self._pending_reset = np.zeros(n, dtype=bool)
        self._closed = False

    @property
    def num_envs(self) -> int:
        return len(self.envs)

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
        """Reset every lane. Lane i is seeded `seed + i` so lanes decorrelate."""
        self._require_open()
        infos: list[dict[str, Any]] = []
        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=None if seed is None else int(seed) + i)
            self._store.store(i, obs)
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
        self._require_open()
        actions = np.asarray(actions)
        if actions.shape[0] != self.num_envs:
            raise ValueError(
                f"step got {actions.shape[0]} actions for {self.num_envs} lanes"
            )
        infos: list[dict[str, Any]] = [{} for _ in self.envs]
        finals: list[Any] = [None] * self.num_envs
        failures: list[tuple[int, BaseException]] = []
        for i, env in enumerate(self.envs):
            try:
                if self._pending_reset[i]:
                    obs, info = env.reset()
                    reward, terminated, truncated = 0.0, False, False
                else:
                    obs, reward, terminated, truncated, info = env.step(actions[i])
                self._store.store(i, obs)
                self._rewards[i] = reward
                self._terminated[i] = bool(terminated)
                self._truncated[i] = bool(truncated)
                if terminated or truncated:
                    finals[i] = self._store.lane(i)
                infos[i] = info
            except BaseException as exc:  # noqa: BLE001 - re-raised below, never dropped
                failures.append((i, exc))
        if failures:
            raise ExceptionGroup(
                f"SyncVectorEnv.step failed in lanes "
                f"{[i for i, _ in failures]} of {self.num_envs}",
                [exc for _, exc in failures],
            )
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

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("SyncVectorEnv is closed")

    def save_state(self) -> dict[str, Any]:
        """Snapshot every lane's declared wrapper state and autoreset phase."""
        self._require_open()
        return {
            "lanes": [snapshot_env(env) for env in self.envs],
            "pending_reset": self._pending_reset.copy(),
            "episodes": self._episodes.get_state(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
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
        for env, lane in zip(self.envs, lanes):
            restore_env(env, lane)
        self._pending_reset = pending

    def save_policy_state(self) -> list[tuple[str, dict[str, Any]]]:
        """Lane 0's policy state, which is every lane's.

        One config builds every lane, and what a lane may hold here is state
        no lane fits on its own: anything FITTED to the data, such as a
        running observation normaliser, is estimated once above this layer by
        `ttrl.envs.vec_normalize` precisely because N lanes would hold N
        different estimates of it. See `POLICY_STATE` in
        `ttrl.envs.wrappers.core`.
        """
        self._require_open()
        return snapshot_policy_state(self.envs[0])

    def save_run_state(self) -> list[tuple[str, dict[str, Any]]]:
        """No lane holds run state. The protocol ends here.

        Declared rather than absent, so `ttrl.envs.vec_wrapper.VecWrapper`
        forwards into one method that exists on both vector implementations
        instead of asking whether it does.
        """
        self._require_open()
        return []

    def load_run_state(self, snapshot) -> None:
        self._require_open()
        refuse_a_mismatched_stack([], [name for name, _ in snapshot], "run")

    def load_policy_state(self, snapshot: list[tuple[str, dict[str, Any]]]) -> None:
        self._require_open()
        for env in self.envs:
            restore_policy_state(env, snapshot)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for env in self.envs:
            env.close()

    def __enter__(self) -> SyncVectorEnv:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
