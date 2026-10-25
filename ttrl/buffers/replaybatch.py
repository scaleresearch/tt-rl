"""Turning drawn (lane, column) pairs into one `ReplayBatch`.

The read half of `ttrl.buffers.replay`, split out for the reason
`ttrl.buffers.seqbatch` is split from the rollout buffer: what ends up in one
training example is the part that goes silently wrong, and it is worth reading
on its own page. `ttrl.buffers.sampling` decides WHICH transitions a batch
holds; this reads what is stored at them.

It reads the buffer's own arrays and its boundary table directly, being one
of the two modules (with `ttrl.buffers.persist`) that is part of
`ReplayBuffer` rather than a consumer of it.
"""

from __future__ import annotations

import numpy as np

from ttrl.buffers.indexing import n_step_forward
from ttrl.buffers.types import ReplayBatch
from ttrl.spaces import group_mask_rows

__all__ = ["assemble", "successor_obs"]


def successor_obs(buffer, lanes: np.ndarray, end_cols: np.ndarray):
    """`next_obs` for each sample: the neighbouring slot, or the boundary table.

    `end_cols[b]` is the last column the n-step walk visited. If that column
    ends its episode the successor is that episode's final observation;
    otherwise it is the observation stored one column later in the SAME lane,
    which under the autoreset contract is the state that step led to.
    """
    out = {
        name: np.empty((lanes.shape[0], *array.shape[2:]), dtype=array.dtype)
        for name, array in buffer.obs.items()
    }
    neighbour = (end_cols + 1) % buffer.per_lane_capacity
    boundary = buffer.ends[lanes, end_cols]
    mid = ~boundary
    if mid.any():
        for name, array in out.items():
            array[mid] = buffer.obs[name][lanes[mid], neighbour[mid]]
    for b in np.flatnonzero(boundary):
        final = buffer._final_obs.read(int(lanes[b]), int(end_cols[b]))
        for name, array in out.items():
            array[b] = final[name]
    return buffer.obs_spec.observation(out)


def assemble(buffer, lanes: np.ndarray, cols: np.ndarray, weights) -> ReplayBatch:
    """The batch the transitions at `(lanes, cols)` make up.

    `cols` rides along in the batch beside `lanes`, because a prioritized
    run writes each sample's fresh TD error back to the column it came from
    and no other key identifies it.
    """
    returns, end_cols, terminated, steps = n_step_forward(
        buffer.rewards, buffer.ends, buffer.dones, lanes, cols,
        n_step=buffer.n_step, gamma=buffer.gamma,
        capacity=buffer.per_lane_capacity,
    )
    next_obs = successor_obs(buffer, lanes, end_cols)
    return ReplayBatch(
        obs=buffer.obs_spec.observation(
            {name: array[lanes, cols] for name, array in buffer.obs.items()}),
        actions=buffer.actions[lanes, cols],
        rewards=returns.astype(np.float32),
        next_obs=next_obs,
        next_mask=(group_mask_rows(next_obs, buffer.obs_spec)
                   if buffer.obs_spec.action_mask else None),
        dones=terminated.astype(np.float32),
        discounts=(buffer.gamma ** steps).astype(np.float32),
        lanes=lanes,
        cols=cols,
        weights=weights,
    )
