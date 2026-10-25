"""The seam at a lane's newest column: what followed it, and closing it.

A replay buffer derives a transition's successor from the next column of its
lane, so the newest column of a lane that is mid-episode has no successor in
the ring until that lane's next write. This module holds the two halves of
that one job: recording the observation that followed the newest column, and
ending each lane's unfinished episode there as a truncation against the
observation recorded. `ttrl.buffers.persist` is the caller that needs both,
and says why a resume must not cross the seam instead.
"""

from __future__ import annotations

import numpy as np

__all__ = ["close_open_episodes", "note_open_successor"]


def note_open_successor(buffer, lane: int, obs) -> None:
    """Record the observation following `lane`'s newest column, or None.

    None says that column ends an episode, so its successor is already in
    the boundary table. Called on every write: the open successor belongs
    to the newest column alone.
    """
    if obs is None:
        buffer._open.pop(int(lane), None)
        return
    observation = buffer.obs_spec.observation(obs)
    buffer._open[int(lane)] = {
        name: np.array(observation[name], copy=True)
        for name in buffer.obs_spec.names
    }


def close_open_episodes(buffer) -> list[int]:
    """End every lane's unfinished episode at its newest column.

    As a TRUNCATION against the successor `note_open_successor` recorded:
    `ends` stops the n-step window there and the stored successor is the
    real one, so the transition keeps the value target it already had and
    nothing is fabricated. Returns the lanes closed.
    """
    closed = []
    for lane, obs in sorted(buffer._open.items()):
        col = int(buffer._pos[lane]) - 1
        if col < 0:
            if not buffer._full[lane]:
                continue
            col = buffer.per_lane_capacity - 1
        buffer.ends[lane, col] = True
        buffer._final_obs.store(lane, col, obs)
        closed.append(lane)
    buffer._open.clear()
    return closed
