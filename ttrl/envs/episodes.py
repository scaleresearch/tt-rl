"""Per-lane episode return and length, accumulated by the vector env itself.

The library computes the score. An environment author writes `reset` and
`step` and nothing else: the vector env already sees every lane's reward and
every lane's `terminated`/`truncated` on every step, which is the same input
an episode accumulator needs, so it accumulates there and emits `EPISODE_KEY`
into the ending lane's own info.

WHERE THIS SITS, and why it is the only placement that works. Both
`SyncVectorEnv` and `AsyncVectorEnv` call this on the rewards their lanes
just returned, inside `step`, before `step` returns. That is BELOW
`ttrl.envs.vec_normalize.NormalizeReward`, which is a `VecWrapper` wrapping
the whole vector env: the reward this accumulates is the lane's own reward,
and the scaled one the loss is computed from never reaches `info`. A score
published off a normalised reward would be meaningless, and
`ttrl.train.evaluate` reads exactly this key.

PRECEDENCE. A lane whose own stack already supplied `EPISODE_KEY` keeps its
number. That is the `ttrl.envs.wrappers.RecordEpisodeStatistics` case, which
stays available because a lane that applies its own reward transform
(`ScaleReward`, `SignReward`) must take the return BELOW that transform, and
only a wrapper inside the lane can. The inner layer is the more specific
answer, so it wins; there is no recovery path here, only an ordering.

`RecordEpisodeStatistics` is therefore optional and nothing declares a class
attribute to say whether a stack can produce a score. `EPISODE_STATISTICS`
and the stack walk that read it are gone: they existed only so a builder
could answer that question without stepping, and the question no longer has
an answer other than yes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ttrl.envs.base import EPISODE_KEY

__all__ = ["LaneEpisodes", "require_episode_state"]


def require_episode_state(state: dict[str, Any]) -> dict[str, Any]:
    """A vector env `save_state` blob's accumulators, or a refusal naming them.

    Absent from a blob written by a build before the vector env accumulated
    the episode return itself. Restoring without them would resume mid-episode
    from a partial return of zero and then report a short first episode as the
    environment's own.
    """
    try:
        return state["episodes"]
    except KeyError:
        raise ValueError(
            "this env state carries no 'episodes' entry. It was saved before "
            "the vector env accumulated each lane's episode return itself, so "
            "the partial returns a resume would continue are not in the file. "
            "Start the run from scratch."
        ) from None


class LaneEpisodes:
    """Undiscounted return and length per lane, emitted at each lane's end."""

    def __init__(self, num_envs: int):
        if num_envs < 1:
            raise ValueError(f"num_envs must be at least 1, got {num_envs}")
        self._returns = np.zeros(num_envs, dtype=np.float64)
        self._lengths = np.zeros(num_envs, dtype=np.int64)

    def reset(self) -> None:
        """Begin a fresh episode in every lane."""
        self._returns[:] = 0.0
        self._lengths[:] = 0

    def observe(
        self,
        rewards: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        resetting: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Fold one step in and return the per-lane infos to publish.

        `resetting` names the lanes this step delivered an autoreset for
        instead of a real step. Those lanes contributed no reward and no
        length, so they are skipped: counting the autoreset step would put a
        spurious +1 on every episode's length after the first. Their
        accumulators were already restarted on the step that ended the
        episode, which is the one place a lane's counters go back to zero.

        A lane's info is copied rather than mutated, because it came from the
        environment and may be an object the environment still holds.
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        done = np.asarray(done, dtype=bool)
        resetting = np.asarray(resetting, dtype=bool)
        fresh = ~resetting
        self._returns[fresh] += rewards[fresh]
        self._lengths[fresh] += 1

        out = list(infos)
        for lane in np.flatnonzero(done & fresh).tolist():
            info = out[lane]
            info = dict(info) if isinstance(info, dict) else {}
            if EPISODE_KEY not in info:
                info[EPISODE_KEY] = {"r": float(self._returns[lane]),
                                     "l": int(self._lengths[lane])}
            out[lane] = info
            self._returns[lane] = 0.0
            self._lengths[lane] = 0
        return out

    def get_state(self) -> dict[str, Any]:
        return {"returns": self._returns.copy(), "lengths": self._lengths.copy()}

    def set_state(self, state: dict[str, Any]) -> None:
        for key, buf in (("returns", self._returns), ("lengths", self._lengths)):
            values = np.asarray(state[key])
            if values.shape != buf.shape:
                raise ValueError(
                    f"episode {key} has shape {values.shape}, expected "
                    f"{buf.shape} for this env's lane count"
                )
            buf[:] = values
