"""The refusals a replay write's arguments must pass.

Separated from `ttrl.buffers.replay` the way `ttrl.algos.checks` is separated
from `ttrl.algos.base`: the ring arithmetic is one subject and what an
argument has to look like before it reaches the ring is another, and each of
these refusals exists because the alternative is silently wrong training data
rather than a crash.
"""

from __future__ import annotations

import numpy as np

__all__ = ["check_add", "check_add_batch"]


def check_add(lane: int, n_envs: int, terminated, truncated, final_obs) -> bool:
    """Validate one `ReplayBuffer.add` call; return whether the step ended."""
    if not 0 <= lane < n_envs:
        raise ValueError(f"ReplayBuffer.add: lane must be in [0, {n_envs}), got {lane}")
    ended = bool(terminated) or bool(truncated)
    if ended and final_obs is None:
        raise ValueError(
            f"ReplayBuffer.add: lane {lane} reported terminated={bool(terminated)} "
            f"truncated={bool(truncated)} but final_obs is None. An ending step's "
            f"successor state cannot be the next slot's observation, which belongs "
            f"to the next episode. Pass info[{'final_obs'!r}][lane] from the env."
        )
    if not ended and final_obs is not None:
        raise ValueError(
            f"ReplayBuffer.add: lane {lane} did not end but final_obs was given; "
            f"the successor of a mid-episode transition is the next step's obs and "
            f"a boundary observation here would be read in its place"
        )
    return ended


def check_add_batch(obs, n_envs: int, terminated, truncated, final_obs):
    """Validate one `ReplayBuffer.add_batch` call.

    Returns `(term, ended)` as per-lane boolean arrays, since deriving them is
    the same work as checking them.
    """
    if obs.lanes != n_envs:
        raise ValueError(
            f"ReplayBuffer.add_batch: leading axis must be n_envs={n_envs}, "
            f"got {obs.lanes}"
        )
    term = np.asarray(terminated, dtype=bool).reshape(-1)
    trunc = (np.zeros(n_envs, bool) if truncated is None
             else np.asarray(truncated, dtype=bool).reshape(-1))
    ended = term | trunc
    if ended.any() and final_obs is None:
        raise ValueError(
            f"ReplayBuffer.add_batch: lanes {np.flatnonzero(ended).tolist()} ended "
            f"but final_obs is None. Pass info[{'final_obs'!r}] from the vector "
            f"env's step; an ending step's successor is not the next slot's obs."
        )
    if final_obs is not None and len(final_obs) != n_envs:
        raise ValueError(
            f"ReplayBuffer.add_batch: final_obs has length {len(final_obs)}, "
            f"expected n_envs={n_envs}"
        )
    for lane in range(n_envs):
        if ended[lane] and (final_obs is None or final_obs[lane] is None):
            raise ValueError(
                f"ReplayBuffer.add_batch: lane {lane} ended but "
                f"final_obs[{lane}] is None"
            )
    return term, ended
