"""The agent contract.

A `Protocol`, so the trainer depends on a surface rather than on a base class,
and so an agent written outside this repo satisfies it without importing it.
`ttrl.algos.checks` turns the contract into executable refusals and is
re-exported from here, so the declaration and its enforcement are separate
modules behind one import site.

The trainer never learns which algorithm it drives. Off-policy and on-policy
differ in what `observe` does with a transition, in what `should_update`
answers, and in nothing else: a replay buffer, a rollout buffer, a warm-up and
an update cadence are all private to the agent that has them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol, runtime_checkable

import numpy as np

# The contract's refusals live in `ttrl.algos.checks` and are re-exported here:
# `ttrl.algos.base` is the one import site every algorithm and the train layer
# already name, and which half of the contract a name belongs to is not
# something a caller should have to know.
from ttrl.algos.checks import check_agent, require_continuous, require_discrete
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec, Observation

__all__ = [
    "Agent",
    "AutoresetTracker",
    "Transition",
    "check_agent",
    "require_continuous",
    "require_discrete",
]


@dataclass(frozen=True)
class Transition:
    """One vector-env step, lane-major on the leading axis.

    `obs` is the batch of states the actions were taken from and `next_obs` is
    what `step` returned, each an `Observation` over the environment's leaves. Under the NEXT_STEP autoreset contract
    (`ttrl.envs.base`) a lane that reports terminated or truncated returns its
    episode's final observation, so `next_obs.select(lane)` IS that lane's
    `info[FINAL_OBS_KEY]` entry and no separate field carries it.

    The step following a lane's end is that lane's autoreset padding step: its
    action was ignored and its reward and flags are fabricated. Nothing here
    marks those, because an agent sees every transition in order and
    `AutoresetTracker` derives them from the previous step's flags.
    """

    obs: Observation
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_obs: Observation

    def __post_init__(self) -> None:
        n = int(self.obs.lanes)
        for name in ("action", "reward", "terminated", "truncated"):
            got = int(np.asarray(getattr(self, name)).shape[0])
            if got != n:
                raise ValueError(
                    f"Transition: {name} has leading axis {got} but obs has {n}; "
                    f"every field is lane-major over the same vector env"
                )
        if self.next_obs.leaves != self.obs.leaves:
            raise ValueError(
                f"Transition: next_obs carries leaves "
                f"{list(self.next_obs.leaves)} and obs carries "
                f"{list(self.obs.leaves)}. One step of one environment "
                f"observes the same leaves before and after, so build both "
                f"from that environment's obs_spec."
            )
        for leaf in self.obs.leaves:
            if self.next_obs[leaf].shape != self.obs[leaf].shape:
                raise ValueError(
                    f"Transition: next_obs shape {self.next_obs[leaf].shape} "
                    f"differs from obs shape {self.obs[leaf].shape} for leaf "
                    f"{leaf!r}. Both are lane-major over the same vector env, "
                    f"so pass the successor observation for every lane the "
                    f"step covered, info['final_obs'] included."
                )

    @property
    def n_envs(self) -> int:
        return int(self.obs.lanes)

    @property
    def done(self) -> np.ndarray:
        return np.asarray(self.terminated, dtype=bool) | np.asarray(
            self.truncated, dtype=bool
        )

    @classmethod
    def from_step(cls, obs_group: ObsGroup, obs, action, step_result) -> "Transition":
        """Build one from `obs`, the action taken, and a `VecEnv.step` return.

        `obs_group` is the environment's own observation spec, which is what
        turns whatever the caller holds into the leaf mapping both halves
        carry; a vector env already returns one of those.
        """
        next_obs, reward, terminated, truncated, _info = step_result
        return cls(
            obs=obs_group.observation(obs), action=np.asarray(action),
            reward=np.asarray(reward, dtype=np.float32),
            terminated=np.asarray(terminated, dtype=bool),
            truncated=np.asarray(truncated, dtype=bool),
            next_obs=obs_group.observation(next_obs),
        )


@runtime_checkable
class Agent(Protocol):
    """What a trainer may do to an algorithm, and nothing more."""

    obs_spec: ObsGroup
    action_spec: ActionSpec

    def act(self, obs: np.ndarray, *, deterministic: bool = False,
            episode_starts: np.ndarray | None = None) -> np.ndarray:
        """One action per lane for a batch of observations.

        `episode_starts` is lane-major and 1.0 where that lane's PREVIOUS step
        ended an episode. A memoryless policy has nothing to reset by it. A
        policy with memory resets its hidden state there, and during
        evaluation it has no other way to learn of a boundary: it is handed
        observations and returns actions, and no transition reaches it. So the
        flags are part of the acting contract rather than an argument only one
        algorithm takes, and `ttrl.train.evaluate` passes them.
        """
        ...

    def named_parameters(self) -> Iterator[tuple[str, Any]]:
        """`(name, Tensor)` for every parameter kept in sync across replicas.

        The parameters an n-chip run averages, and nothing else: a lagged
        target network is derived from them rather than learned, and optimizer
        moments are local to a replica by construction. The names are what
        `ttrl.engine.mesh.ParamSync` compares across replicas before it
        averages anything, which is what catches two same-shaped parameters in
        different positions; averaging those mixes unrelated tensors and
        diverges the run with no error anywhere.
        """
        ...

    def parameters(self) -> Iterator[Any]:
        """The same parameters in the same order, which is the order the
        averaging zips positionally."""
        ...

    def observe(self, transition: Transition) -> None:
        """Record one env step into whatever storage this algorithm keeps."""
        ...

    def update(self) -> dict[str, float]:
        """One learn step over that storage. Returns scalar metrics."""
        ...

    def should_update(self, frame: int) -> bool:
        """Whether `update()` should run now. The cadence lives here.

        ASK EVERY AGENT. Being asked is how an agent learns the frame the run
        has reached, and an off-policy agent sizes the work of the next
        `update()` from that. An agent never asked cannot know a round
        happened and will do the smallest amount of work it can, so a caller
        that asks one agent and then calls `update()` on several starves all
        but the one it asked. By the same arithmetic that costs replica 1 of
        a two-chip run 87.5% of its gradient steps.
        """
        ...

    def state_dict(self, *, weights_only: bool = False) -> dict:
        """The agent's state. `weights_only=True` returns the parameters alone,
        which is what a snapshot tier can honestly promise."""
        ...

    def load_state_dict(self, state: dict, *, weights_only: bool = False) -> None:
        """Restore from either half. A weights-only state handed to a plain
        call raises rather than resuming with zeroed optimizer moments."""
        ...

    def close(self) -> None: ...


class AutoresetTracker:
    """Which lanes of the current step are real transitions.

    A lane that reported terminated or truncated on the previous step has its
    next step consumed by the autoreset, which ignores the action and
    fabricates the reward. Training on that step teaches the value function a
    transition the environment never made, so it is dropped. Derived from the
    flags an agent already sees, so no extra field crosses the agent contract.
    """

    def __init__(self, n_envs: int):
        if n_envs < 1:
            raise ValueError(f"AutoresetTracker: n_envs must be >= 1, got {n_envs}")
        self.n_envs = int(n_envs)
        self._pending = np.zeros(self.n_envs, dtype=bool)

    @property
    def pending(self) -> np.ndarray:
        return self._pending.copy()

    def valid(self, transition: Transition) -> np.ndarray:
        """Boolean mask of lanes whose step was a real transition."""
        if transition.n_envs != self.n_envs:
            raise ValueError(
                f"AutoresetTracker: transition has {transition.n_envs} lanes, "
                f"this agent was built for {self.n_envs}"
            )
        done = transition.done
        both = self._pending & done
        if both.any():
            raise ValueError(
                f"lanes {np.flatnonzero(both).tolist()} report done on the step "
                f"that delivers their autoreset observation. The contract is "
                f"NEXT_STEP: the step after a termination reports done False "
                f"for that lane (ttrl.envs.check_vec_env asserts this)."
            )
        real = ~self._pending
        self._pending = done
        return real

    def state_dict(self) -> dict:
        return {"pending": self._pending.copy()}

    def load_state_dict(self, state: dict) -> None:
        """Check the stored phase and then START FRESH, which is not the same.

        A checkpoint is only ever loaded into a run whose environments are
        about to be reset (`ttrl.train.loop.run_loop` resets every replica
        before its first round), so no lane owes an autoreset step however the
        run that wrote the file stood. Restoring the stored flags instead
        dropped the first REAL transition after the resume for every lane that
        happened to end its episode on the checkpointed step, which is every
        lane whenever the interval and the episode length share a factor. The
        phase is still written, because it describes the run that wrote it;
        the shape is still checked, because a file whose lane count is not
        this run's is the wrong file and says so here.
        """
        pending = np.asarray(state["pending"], dtype=bool)
        if pending.shape != (self.n_envs,):
            raise ValueError(
                f"AutoresetTracker: checkpoint pending has shape {pending.shape}, "
                f"expected {(self.n_envs,)}"
            )
        self._pending = np.zeros(self.n_envs, dtype=bool)
