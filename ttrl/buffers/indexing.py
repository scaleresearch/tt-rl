"""Lane-local ring index arithmetic and the n-step forward walk.

Pure numpy over arrays handed in by the buffer, so every rule here is testable
with no ttnn, no chip and no buffer object.

THE LANE INVARIANT: every index this module produces is a column index into
one lane's row of the ring, and the lane index is carried alongside it
unchanged. No arithmetic here combines two lanes, so no sampled transition can
span two environment lanes. A single interleaved ring with a lane id per slot
would need this invariant re-proved for the backward stack walk, the forward
n-step walk and the wrap seam; separate lanes make it structural.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sampleable", "draw_offsets", "n_step_forward"]


def sampleable(pos: np.ndarray, full: np.ndarray, capacity: int,
               n_step: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-lane `(start, count)` of the sampleable column range.

    The sampleable set of a lane is always one contiguous run in ring order,
    which is why no rejection sampling is needed anywhere below.

    A transition at column `i` reads `n_step + 1` columns of one lane: columns
    `i .. i + n_step - 1` for the rewards and the episode flags, and column
    `i + n_step` for `next_obs`, which is the neighbouring slot's observation
    (see `ttrl.buffers.replay`). So the run is `n_step + 1` columns shorter
    than the lane, not `n_step`.

    Filling (`not full`): writes are the contiguous prefix `[0, pos)`, and the
    read must not step past the cursor into never-written columns, so
    `i + n_step <= pos - 1`.

    Full: every column holds data, but `pos` is the seam where the oldest
    surviving lap sits immediately before the newest one in ring-index terms
    while being nowhere near adjacent in time. With
    `age(i) = (i - pos) % capacity` running 0 at the oldest column to
    `capacity - 1` at the newest, the read stays inside one lap exactly when
    `age(i) + n_step <= capacity - 1`. Nothing in the stored data marks that
    seam, and the neighbour read is the part that crosses it first: at
    `age(i) = capacity - n_step` the window itself still fits while
    `next_obs` would come from the column the write cursor has already
    overwritten. That is silent wrong data rather than a crash, which is why
    the bound is positional and tested directly.

    The neighbour column is only needed when the window ran its full length
    without hitting an episode boundary; a window that ended early takes
    `next_obs` from the boundary table instead. Excluding those transitions
    anyway keeps this bound POSITIONAL: it costs at most one transition per
    lane, and the alternative is a content-dependent sampleable set, which is
    exactly the distribution distortion that carrying `discounts` removed.
    """
    pos = np.asarray(pos, dtype=np.int64)
    full = np.asarray(full, dtype=bool)
    start = np.where(full, pos, 0)
    count = np.where(full, capacity, pos) - n_step
    return start, np.maximum(count, 0)


def draw_offsets(rng, count: np.ndarray, per_lane: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """`(lanes, offsets)` for a batch drawn uniformly inside each lane's run.

    `per_lane[e]` samples are taken from lane `e`, as POSITIONS in that lane's
    sampleable run rather than as ring columns: the column is the window's to
    compute (`ttrl.buffers.samplers.SampleWindow.locate`), which is what keeps
    a sampler unable to name a column outside the bound.

    Drawing uniformly on `[0, count)` is uniform over that lane's sampleable
    set exactly, with no acceptance test, so the distribution does not depend
    on the buffer's contents.
    """
    lanes_out = []
    offsets_out = []
    for lane, k in enumerate(int(v) for v in per_lane):
        if k == 0:
            continue
        if count[lane] <= 0:
            raise ValueError(
                f"replay lane {lane} has no sampleable transitions but was asked "
                f"for {k} samples"
            )
        offsets_out.append(rng.integers(0, int(count[lane]), size=k))
        lanes_out.append(np.full(k, lane, dtype=np.int64))
    if not offsets_out:
        raise ValueError("replay buffer has no sampleable transitions yet")
    return np.concatenate(lanes_out), np.concatenate(offsets_out).astype(np.int64)


def n_step_forward(rewards: np.ndarray, ends: np.ndarray, dones: np.ndarray,
                   lanes: np.ndarray, cols: np.ndarray, *, n_step: int,
                   gamma: float, capacity: int):
    """Discounted forward return, last column of the window, and the window's
    termination flag and length.

    Returns `(returns, end_cols, terminated, steps)`. The walk stops at the
    first episode boundary (`ends`, terminated or truncated); only a genuine
    termination (`dones`) zeroes the bootstrap. `steps` is how many rewards
    were actually summed, which the caller turns into `gamma ** steps` so a
    short window still carries the right bootstrap discount.

    The accumulation is float64 in the order `g += disc * r; disc *= gamma`, so
    it is reproducible against a scalar reference bit for bit.
    """
    lanes = np.asarray(lanes, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    g = np.zeros(cols.shape, dtype=np.float64)
    disc = np.ones(cols.shape, dtype=np.float64)
    end_cols = cols.copy()
    terminated = np.zeros(cols.shape, dtype=bool)
    steps = np.zeros(cols.shape, dtype=np.int64)
    running = np.ones(cols.shape, dtype=bool)
    for offset in range(n_step):
        j = (cols + offset) % capacity
        r = rewards[lanes, j]
        end_cols = np.where(running, j, end_cols)
        g += np.where(running, disc * r, 0.0)
        disc = np.where(running, disc * gamma, disc)
        steps += running
        boundary = ends[lanes, j] & running
        # `boundary` is true at most once per sample, so `|=` here is the
        # scalar walk's assign-then-break.
        terminated |= boundary & dones[lanes, j]
        running &= ~boundary
        if not running.any():
            break
    return g, end_cols, terminated, steps
