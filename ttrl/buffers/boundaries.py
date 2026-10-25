"""The sparse table of episode-final observations, and the eviction invariant.

A replay buffer stores one observation per transition and recovers a
successor from the neighbouring column of the same lane. That derivation is
wrong at exactly one place, an episode boundary, where the next column holds
the first observation of the NEXT episode. This table holds the boundary cases
and nothing else, which is what keeps the buffer at one array per leaf instead
of two.

THE INVARIANT: an entry lives exactly as long as the column it belongs to.
Every write to a column evicts the entry under its key first, because that
entry belongs to the lap being overwritten and a stale boundary observation
left behind would be served as some live transition's successor state. An
entry missing when a boundary column asks for it is a bug in the ring
bookkeeping rather than a state to recover from, so it raises.
"""

from __future__ import annotations

import numpy as np

__all__ = ["BoundaryTable"]


class BoundaryTable:
    """`(lane, column) -> that episode's final observation`, for ending columns."""

    def __init__(self, obs_group):
        self.obs_group = obs_group
        self._entries: dict[tuple[int, int], dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def evict(self, lane: int, col: int) -> None:
        """Forget the entry at this column, whether or not it had one."""
        self._entries.pop((lane, col), None)

    def store(self, lane: int, col: int, final_obs) -> None:
        """Record one ending column's final observation, leaf by leaf."""
        final = self.obs_group.observation(final_obs)
        if final.lanes is not None:
            raise ValueError(
                f"final_obs for lane {lane} carries {final.lanes} lanes; one "
                f"ending lane has one final observation"
            )
        # A copy, because a vector env reuses one observation buffer and a
        # stored view of it would be mutated by the next step.
        self._entries[(lane, col)] = {
            name: np.array(final[name], copy=True) for name in self.obs_group.names
        }

    def read(self, lane: int, col: int) -> dict[str, np.ndarray]:
        entry = self._entries.get((lane, col))
        if entry is None:
            raise RuntimeError(
                f"column {col} of lane {lane} ends an episode but carries no "
                f"final observation. Every ending write records one, so this "
                f"means the entry was evicted while the column was still "
                f"sampleable, which is a bug in the ring bookkeeping."
            )
        return entry

    # -- checkpointing -----------------------------------------------------

    def state(self) -> dict:
        """The live entries as columns, which is what an npz can hold.

        A dict keyed by `(lane, column)` has no array form, so the keys go
        out as two integer arrays and each leaf as one stacked array. Entry
        `i` of every array is one entry, which is what keeps a leaf from
        being paired with another entry's key on the way back.

        `np.stack` copies, which is what the rest of a replay snapshot does
        for the reason `ttrl.buffers.persist.replay_state` gives.
        """
        keys = sorted(self._entries)
        return {
            "lanes": np.array([lane for lane, _ in keys], dtype=np.int64),
            "cols": np.array([col for _, col in keys], dtype=np.int64),
            "leaves": {
                name: np.stack([self._entries[key][name] for key in keys])
                if keys else np.zeros((0,), dtype=np.float32)
                for name in self.obs_group.names
            },
        }

    def load(self, state: dict) -> None:
        """Replace every entry with the stored ones. Not a merge: a table
        holding entries from two different rings would serve one ring's
        boundary observation as the other's successor state."""
        lanes = np.asarray(state["lanes"], dtype=np.int64)
        cols = np.asarray(state["cols"], dtype=np.int64)
        leaves = state["leaves"]
        missing = [n for n in self.obs_group.names if n not in leaves]
        if missing:
            raise ValueError(
                f"stored episode boundaries carry no observation leaf "
                f"{missing}; this run's observation has leaves "
                f"{list(self.obs_group.names)}"
            )
        self._entries = {
            (int(lane), int(col)): {name: np.array(leaves[name][i], copy=True)
                                    for name in self.obs_group.names}
            for i, (lane, col) in enumerate(zip(lanes, cols))
        }
