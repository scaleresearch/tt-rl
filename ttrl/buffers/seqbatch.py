"""Turning a filled recurrent `RolloutBuffer` into sequence minibatches.

Separate from the storage for the same reason `ttrl.buffers.indexing` is
separate from `ttrl.buffers.replay`: what ends up in one training example is
the part that goes silently wrong, and the arithmetic behind it is in
`ttrl.buffers.sequences`, which holds no buffer and imports nothing but numpy.
This module is the thin layer that reads a buffer's arrays through it.

A MINIBATCH IS A SET OF WHOLE SEQUENCES. Sequence ids run over
`(chunk, lane)`, are shuffled as units, and `n_seq = minibatch_size / seq_len`
of them make one minibatch. Within a sequence, time is consecutive and the
lane is one lane, both structurally (see `sequences.gather`). So no minibatch
can mix two lanes' timesteps, which a flat shuffle over transitions does by
design and a recurrent update cannot survive.

ADVANTAGE NORMALIZATION IS OVER VALID SLOTS ONLY. The flat path gets that for
free by selecting valid rows before it normalizes. A sequence keeps its
autoreset padding slots, because dropping one would shorten the sequence and
so change the graph shape, so the mean and the standard deviation here are
taken over the valid slots and then applied to every slot. The padding slots'
normalized values are meaningless and never reach the loss, which reduces
against `valid`.
"""

from __future__ import annotations

import numpy as np

from ttrl.buffers.sequences import (
    chunks_per_minibatch,
    flatten_time_major,
    gather,
)
from ttrl.buffers.types import RolloutSequenceBatch
from ttrl.spaces import group_mask_rows

__all__ = ["SequenceBatcher"]


class SequenceBatcher:
    """Sequence minibatches out of one filled recurrent `RolloutBuffer`."""

    def __init__(self, buffer):
        if not buffer.recurrent:
            raise RuntimeError(
                "SequenceBatcher: this RolloutBuffer was built without a "
                "seq_len, so it stores no hidden states and cannot serve a "
                "recurrent update. Build it with seq_len and hidden_sizes."
            )
        if not buffer._gae_computed:
            raise RuntimeError(
                "SequenceBatcher: call compute_gae() on the buffer first")
        self.buf = buffer

    def minibatches(self, minibatch_size: int, *, epochs: int = 4,
                    normalize_adv: bool = True):
        buf = self.buf
        n_seq = chunks_per_minibatch(minibatch_size, buf.seq_len)
        total = buf.n_chunks * buf.n_envs
        n_full = total // n_seq
        if n_full == 0:
            raise ValueError(
                f"RolloutBuffer.sequence_minibatches: the rollout holds {total} "
                f"sequences of {buf.seq_len} steps (n_steps={buf.n_steps}, "
                f"n_envs={buf.n_envs}) and a minibatch of "
                f"{minibatch_size} rows wants {n_seq} of them, which would yield "
                f"zero minibatches. Lower minibatch_size or seq_len, or collect "
                f"a longer rollout."
            )
        advantages = self._normalized(normalize_adv)
        for _ in range(epochs):
            perm = buf._rng.permutation(total)
            for start in range(0, n_full * n_seq, n_seq):
                ids = perm[start:start + n_seq]
                # Sequence id -> (chunk, lane). Lane is the fast axis, so a
                # minibatch drawn from a contiguous run of ids still spreads
                # over lanes; the shuffle makes that moot and the mapping is
                # here so that a test can name a sequence by its (chunk, lane).
                yield self._batch(ids // buf.n_envs, ids % buf.n_envs, advantages)

    def _normalized(self, normalize_adv: bool) -> np.ndarray:
        buf = self.buf
        if not normalize_adv:
            return buf.advantages
        live = buf.valid > 0.0
        if not live.any():
            raise ValueError(
                "RolloutBuffer.sequence_minibatches: the rollout holds no valid "
                "transitions, so the advantage statistics have nothing to "
                "normalize against"
            )
        values = buf.advantages[live]
        return (buf.advantages - values.mean()) / (values.std() + 1e-8)

    def _batch(self, chunks: np.ndarray, lanes: np.ndarray,
               advantages: np.ndarray) -> RolloutSequenceBatch:
        buf = self.buf
        length = buf.seq_len

        def rows(array: np.ndarray) -> np.ndarray:
            return flatten_time_major(gather(array, chunks, lanes, length))

        obs = {name: rows(array) for name, array in buf.obs.items()}
        return RolloutSequenceBatch(
            obs=buf.obs_spec.observation(obs),
            actions=rows(buf.actions),
            old_logp=rows(buf.old_logp),
            values=rows(buf.values),
            advantages=rows(advantages),
            returns=rows(buf.returns),
            # (seq_len, n_seq, 1): the unroll's own layout, split on axis 0.
            starts=gather(buf.starts, chunks, lanes, length)[..., None],
            valid=rows(buf.valid),
            h0={name: array[chunks, lanes]
                for name, array in buf.hidden.items()},
            seq_len=length,
            lanes=np.asarray(lanes, dtype=np.int64),
            mask=(group_mask_rows(obs, buf.obs_spec)
                  if buf.obs_spec.action_mask else None),
        )
