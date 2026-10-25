"""Pass-through base for a wrapper that sits ABOVE a vector environment.

Why a layer exists here at all. A per-lane wrapper is built inside the lane's
own factory, so N lanes hold N of it, and under `AsyncVectorEnv` each one
lives in a different process. Anything that has to be SHARED across the lanes
therefore cannot be a per-lane wrapper: there is no object the lanes have in
common and, across a process boundary, no way to give them one. A running
estimator of the observation distribution is exactly such a thing, which is
what moved `ttrl.envs.vec_normalize.NormalizeObservation` up here.

It carries the two state protocols the layer below it carries, one level up
and with the same declarations:

  `save_policy_state` / `load_policy_state`   the transform the weights were
      fitted under, which travels into an exported artifact.
  `save_run_state` / `load_run_state`         cross-episode statistics a
      resume must continue that are NOT part of what the weights compute.

Each wrapper declares `POLICY_STATE` and `RUN_STATE` as the `get_state` keys
of each kind, and `ttrl.envs.wrappers.core` holds the one implementation of
the declaration, the selection and the refusal. A wrapper that declares
neither is refused by name rather than silently contributing nothing.

The bearer list a restore is checked against is read off the LIVE stack and
never off a literal here, so a layer added or removed is reported as the
mismatch it is.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ttrl.envs.wrappers.core import (
    declared_items,
    declared_keys,
    refuse_a_mismatched_stack,
)
from ttrl.spaces import ActionSpec, ObsGroup

__all__ = ["VecWrapper"]


class VecWrapper:
    """Wraps a `VecEnv`, forwarding everything a subclass does not change."""

    def __init__(self, env):
        self.env = env

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def obs_spec(self) -> ObsGroup:
        return self.env.obs_spec

    @property
    def action_spec(self) -> ActionSpec:
        return self.env.action_spec

    @property
    def pending_reset(self) -> np.ndarray:
        return self.env.pending_reset

    def reset(self, *, seed: int | None = None):
        return self.env.reset(seed=seed)

    def step(self, actions):
        return self.env.step(actions)

    def close(self) -> None:
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- state, in the two kinds the layer below also distinguishes ---------

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} declares state keys but has no set_state, "
            f"so a resume cannot put back what its get_state saved. Implement "
            f"set_state(state) to read the keys get_state writes, or return an "
            f"empty dict from get_state if this layer carries no state."
        )

    def save_state(self) -> dict[str, Any]:
        """This wrapper's whole state and the inner environment's.

        Nested rather than merged into the inner dict: the inner shape is
        `SyncVectorEnv`'s own and a wrapper adding keys to it would make the
        two indistinguishable on the way back in.
        """
        return {"self": (type(self).__name__, self.get_state()),
                "inner": self.env.save_state()}

    def load_state(self, state: dict[str, Any]) -> None:
        name, items = state["self"]
        if name != type(self).__name__:
            raise ValueError(
                f"vec env state restore: this layer is "
                f"{type(self).__name__} and the saved layer is {name!r}, so "
                f"the stack being restored into is not the one that was saved. "
                f"Build the env from the config the checkpoint stores "
                f"(ttrl.config_from_checkpoint prints it), or start afresh."
            )
        if items:
            self.set_state(items)
        self.env.load_state(state["inner"])

    def save_policy_state(self) -> list[tuple[str, dict[str, Any]]]:
        return self._own("POLICY_STATE") + self.env.save_policy_state()

    def load_policy_state(self, snapshot) -> None:
        rest = self._take(snapshot, "POLICY_STATE", "policy",
                          self.env.save_policy_state())
        self.env.load_policy_state(rest)

    def save_run_state(self) -> list[tuple[str, dict[str, Any]]]:
        return self._own("RUN_STATE") + self.env.save_run_state()

    def load_run_state(self, snapshot) -> None:
        rest = self._take(snapshot, "RUN_STATE", "run",
                          self.env.save_run_state())
        self.env.load_run_state(rest)

    def _own(self, attr: str) -> list[tuple[str, dict[str, Any]]]:
        items = declared_items(self, attr)
        return [(type(self).__name__, items)] if items else []

    def _take(self, snapshot, attr: str, kind: str, downstream) -> list:
        """Apply the entry that is this wrapper's and return what is below it.

        The live bearer list is this wrapper plus whatever the inner stack
        reports, so a refusal names the whole stack rather than one layer of
        it, and the names come from the objects rather than from a table.
        """
        snapshot = list(snapshot)
        if not declared_keys(self, attr):
            return snapshot
        live = [type(self).__name__] + [name for name, _ in downstream]
        refuse_a_mismatched_stack(live, [name for name, _ in snapshot], kind)
        self.set_state({**self.get_state(), **dict(snapshot[0][1])})
        return snapshot[1:]
