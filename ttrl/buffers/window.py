"""What a replay sampler is handed and what it may hand back.

WHY THE SEAM HANDS OVER A DECLARATION AND NOT THE RING. A sampler's whole job
is to pick which stored transitions a batch is built from, and the one way to
get that catastrophically wrong is to pick one outside the positional bound in
`ttrl.buffers.indexing.sampleable`: the write cursor's neighbourhood pairs a
transition with a successor observation it never had, and reading it is wrong
training data rather than an error. So the vocabulary a sampler returns cannot
express a ring column at all. It returns a POSITION inside a declared run,
`0 <= offset < window.count(lane)`, and `SampleWindow.locate` is the only code
that turns a position into a `(lane, column)` pair. A sampler that computes a
column of its own has nowhere to put it, and a position outside the run is
refused by name before a batch is assembled.

The samplers themselves, and the registry a name resolves through, are in
`ttrl.buffers.samplers`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

__all__ = ["Draw", "SampleWindow"]


class Draw(NamedTuple):
    """What `ReplaySampler.draw` returns.

    `lanes[i]` and `offsets[i]` name one sampled transition: lane `lanes[i]`
    of the ring, position `offsets[i]` inside that lane's sampleable run.
    `weights` is the per-sample importance-sampling correction a biased draw
    owes, or None from an unbiased one.
    """

    lanes: np.ndarray
    offsets: np.ndarray
    weights: np.ndarray | None = None


@dataclass(frozen=True)
class SampleWindow:
    """The sampleable runs a batch may be drawn from, one per lane.

    Handed to every `draw`. `counts[lane]` is how many transitions lane
    `lane` offers, and POSITION 0 IS THE OLDEST of them while position
    `counts[lane] - 1` is the newest. That ordering is the only fact about the
    storage a sampler may rely on, and it is the fact a recency-weighted
    sampler needs.

    `start` and `capacity` are the ring arithmetic `columns` is built from. A
    sampler has no reason to read either: `columns(lane)` is the supported way
    to reach a per-column table, and a draw is expressed in positions, which
    is what keeps it inside the bound.
    """

    counts: np.ndarray
    start: np.ndarray
    capacity: int

    @property
    def n_lanes(self) -> int:
        return int(self.counts.shape[0])

    @property
    def total(self) -> int:
        """Sampleable transitions across every lane."""
        return int(self.counts.sum())

    def count(self, lane: int) -> int:
        return int(self.counts[lane])

    def columns(self, lane: int) -> np.ndarray:
        """One lane's sampleable ring columns, oldest first.

        For a sampler holding a table keyed by column: position `p` of lane
        `l` reads `table[l, window.columns(l)[p]]`.
        """
        n = self.count(lane)
        if n <= 0:
            return np.zeros(0, dtype=np.int64)
        return (int(self.start[lane]) + np.arange(n, dtype=np.int64)) % self.capacity

    def locate(self, draw, *, batch_size: int, sampler) -> tuple:
        """`(lanes, cols, weights)` for a `Draw`, refusing anything outside the run.

        THE ENFORCEMENT, and it runs on every batch rather than under a debug
        flag, because every failure it names is silent in the data: a position
        outside the run reads a successor observation the write cursor has
        overwritten, and a short batch trains a fixed-size device buffer on
        whatever its tail held last.
        """
        who = type(sampler).__name__
        if not isinstance(draw, tuple) or len(draw) != 3:
            raise TypeError(
                f"{who}.draw must return (lanes, offsets, weights), got "
                f"{type(draw).__name__}. Return a ttrl.buffers.samplers.Draw, "
                f"whose weights default to None for an unbiased sampler."
            )
        lanes, offsets, weights = draw
        lanes = _integers(lanes, f"{who}.draw's lanes")
        offsets = _integers(offsets, f"{who}.draw's offsets")
        if lanes.shape != offsets.shape:
            raise ValueError(
                f"{who}.draw returned {lanes.shape[0]} lanes and "
                f"{offsets.shape[0]} offsets; they are read as pairs, so there "
                f"must be one offset per lane entry"
            )
        if lanes.shape[0] != batch_size:
            raise ValueError(
                f"{who}.draw returned {lanes.shape[0]} samples for a "
                f"batch_size of {batch_size}. A batch is loaded into device "
                f"buffers of exactly batch_size rows, so a short draw leaves "
                f"rows holding the previous batch; draw exactly batch_size "
                f"samples, with replacement if the run is smaller."
            )
        outside = (lanes < 0) | (lanes >= self.n_lanes)
        if outside.any():
            raise ValueError(
                f"{who}.draw returned lane {int(lanes[np.flatnonzero(outside)[0]])}, "
                f"and this buffer has lanes 0..{self.n_lanes - 1}. A lane is an "
                f"environment of the vector env and never an index into the ring."
            )
        limits = self.counts[lanes]
        bad = (offsets < 0) | (offsets >= limits)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"{who}.draw returned offset {int(offsets[i])} in lane "
                f"{int(lanes[i])}, whose sampleable run holds "
                f"{int(limits[i])} transitions. Every offset must satisfy "
                f"0 <= offset < window.count(lane): the transitions outside that "
                f"run have no stored successor observation, so drawing one "
                f"reads the column the write cursor has already overwritten "
                f"and trains on it silently."
            )
        cols = (self.start[lanes] + offsets) % self.capacity
        return lanes, cols.astype(np.int64), _weights(weights, batch_size, who)


def _integers(values, what: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{what} must be a 1-D array, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            f"{what} has dtype {array.dtype}, and an index into storage must "
            f"be an integer: a float offset rounds toward a neighbouring "
            f"transition wherever it is used as one. Cast it with "
            f".astype(np.int64)."
        )
    return array.astype(np.int64, copy=False)


def _weights(weights, batch_size: int, who: str):
    if weights is None:
        return None
    array = np.asarray(weights)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(
            f"{who}.draw's weights have dtype {array.dtype}; an "
            f"importance-sampling weight is a float and an integer one would "
            f"floor to zero for every sample. Return None if the draw needs "
            f"no correction."
        )
    if array.shape != (batch_size,):
        raise ValueError(
            f"{who}.draw returned {array.shape} weights for a batch of "
            f"{batch_size}; one weight per sampled transition, in the order "
            f"the draw returned them"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{who}.draw returned a non-finite importance-sampling weight, "
            f"which scales one transition's gradient by infinity"
        )
    return array
