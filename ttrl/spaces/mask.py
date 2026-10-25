"""The legal-action mask as the tail of an observation vector.

Where the mask lives is `ObsSpec.action_mask`: the final n entries of every
observation. Reading it back, drawing from it and refusing a transition that
disagrees with it are the whole of what the mask needs outside the spec.
"""

from __future__ import annotations

import numpy as np

from ttrl.spaces.group import ObsGroup
from ttrl.spaces.obs import ObsSpec

__all__ = ["check_legal", "draw_legal", "group_mask_rows", "mask_rows"]


def group_mask_rows(obs, obs_group) -> np.ndarray:
    """The legal-action rows of a whole observation, from the leaf carrying them.

    `ObsGroup` admits the mask on one leaf, so this is `mask_rows` of that
    leaf and the group is the only thing that knows which leaf it is.

    Takes either form of declared spec. A single environment's `obs_spec` is
    often the bare `ObsSpec`, and promoting it here is what lets a caller pass
    whichever one it has instead of reaching into numpy and failing there.
    """
    obs_group = ObsGroup.of(obs_group)
    observation = obs_group.observation(obs)
    rows = observation[obs_group.mask_leaf]
    return mask_rows(np.atleast_2d(rows), obs_group.mask_spec)


def mask_rows(obs, obs_spec: ObsSpec) -> np.ndarray:
    """The legal-action rows carried in the tail of each observation.

    `(batch, obs_spec.action_mask)` float32 of 0.0 and 1.0. Raises for a spec
    that declares no mask, so nothing silently reads feature columns as one.
    """
    if not obs_spec.action_mask:
        raise ValueError(
            "mask_rows: this ObsSpec declares no legal-action mask. Build it with "
            "ObsSpec.vector(dim, action_mask=n) so the final n entries of every "
            "observation are read as the mask."
        )
    arr = np.asarray(obs)
    rows = arr.reshape(int(arr.shape[0]), -1)[:, -obs_spec.action_mask:]
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    if not np.all((rows == 0.0) | (rows == 1.0)):
        raise ValueError(
            f"mask_rows: the final {obs_spec.action_mask} entries of an observation "
            f"must each be 0.0 or 1.0, got values {np.unique(rows).tolist()[:8]}. "
            f"Write the legal-action mask there as indicators, and keep every other "
            f"feature ahead of it."
        )
    return rows


def draw_legal(rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    """One uniformly drawn legal action per row of a legal-action mask.

    The uniform with the largest value among a row's legal columns is a
    uniform choice among them, so the whole batch costs one generator call.
    """
    return np.argmax(np.where(np.asarray(mask) > 0.5, rng.random(np.shape(mask)), -1.0),
                     axis=-1).astype(np.int64)


def check_legal(obs, actions, obs_group, where: str) -> None:
    """Refuse a transition whose action its own observation's mask forbade.

    An environment handed an illegal action can only coerce it to a legal one,
    and the transition then stored carries an action that was never played.
    The library never sends an illegal action, so this firing means an action
    reached storage from somewhere other than a masking `act`.
    """
    mask = group_mask_rows(obs, obs_group)
    idx = np.asarray(actions).reshape(-1).astype(np.int64)
    taken = mask[np.arange(idx.shape[0]), idx]
    bad = np.flatnonzero(taken < 0.5)
    if bad.size:
        row = int(bad[0])
        raise ValueError(
            f"{where}: action {int(idx[row])} is illegal under the legal-action mask "
            f"its own observation carries (legal actions: "
            f"{np.flatnonzero(mask[row] > 0.5).tolist()}). Choose actions with the "
            f"agent's act(), which masks its greedy choice and its exploration "
            f"together."
        )
