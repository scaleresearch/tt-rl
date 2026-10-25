"""Cutting a `[T, n_envs, ...]` rollout into fixed-length sequences.

Pure numpy over arrays handed in by the buffer, so every rule here is testable
with no ttnn, no chip and no buffer object. This is to the recurrent rollout
what `ttrl.buffers.indexing` is to replay: the arithmetic that decides which
slots end up in one training example, kept away from the storage that holds
them.

THE CHUNK INVARIANT: a sequence is `seq_len` consecutive columns of ONE lane,
addressed by `(chunk, lane)`, and `gather` carries one lane index per sequence
held constant down the time axis. So no arithmetic here can build a sequence
that spans two environment lanes or two non-consecutive timesteps, the same
way `indexing`'s lane invariant is structural rather than asserted.

WHY FIXED-LENGTH CHUNKS AND NOT PADDED EPISODES. `ttrl.engine.recurrent.GRU`
unrolls the T it reads off its input, so T fixes the graph a trace captures.
Padding whole episodes to a common length makes T the longest episode in the
rollout: unbounded, different every rollout, so the captured trace is invalid
after the first one, and wasteful in proportion to the variance of episode
length. Fixed chunks give T = seq_len for the life of the run, one graph
shape, and no sequence padding at all. The price is the standard truncated
BPTT one, that gradient does not flow across a chunk boundary.

THE TIME-MAJOR FLATTEN. A sequence batch is held `(seq_len, n_seq, ...)` for
the unroll and flattened to rows for the encoder, which means row index
`t * n_seq + sequence`. That is plain C-order, so `reshape` turns the
encoder's `(seq_len * n_seq, features)` output back into the
`(seq_len, n_seq, features)` the unroll splits on axis 0 with no permutation
anywhere. Flattening sequence-major instead transposes time against sequence
and trains perfectly happily, which is why `flatten_time_major` exists as one
named function with one test rather than as a `reshape` at four call sites.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "chunk_count",
    "chunks_per_minibatch",
    "flatten_time_major",
    "gather",
    "shift_to_starts",
    "store_hidden",
]


def chunk_count(n_steps: int, seq_len: int) -> int:
    """How many sequences one lane of an `n_steps` rollout holds.

    `seq_len` must divide `n_steps`. A remainder would make the last sequence
    shorter, and a shorter sequence is a SECOND graph shape: the unroll is
    built for one T and a captured trace replays against that T alone. So the
    remainder is refused here rather than yielded as a ragged tail.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if n_steps % seq_len:
        raise ValueError(
            f"seq_len={seq_len} does not divide n_steps={n_steps}, which would "
            f"leave a final sequence of {n_steps % seq_len} steps. The unroll is "
            f"built for one fixed T, so a short tail is a second graph shape "
            f"rather than a smaller batch. Choose a seq_len dividing "
            f"{n_steps}."
        )
    return n_steps // seq_len


def chunks_per_minibatch(minibatch_size: int, seq_len: int) -> int:
    """How many whole sequences make up a minibatch of `minibatch_size` rows.

    A minibatch is a set of whole sequences, so its row count is a multiple of
    `seq_len`. Rounding instead would hand the network a different number of
    rows than it was built for, which a convolution refuses and a matmul
    silently accepts.
    """
    if minibatch_size % seq_len:
        raise ValueError(
            f"minibatch_size={minibatch_size} is not a multiple of "
            f"seq_len={seq_len}. A recurrent minibatch is a whole number of "
            f"sequences of seq_len steps, so its row count is "
            f"n_sequences * seq_len."
        )
    return minibatch_size // seq_len


def shift_to_starts(boundary: np.ndarray, carried: np.ndarray) -> np.ndarray:
    """Per-slot "this step begins an episode", from per-slot "this step ended
    one".

    THE SHIFT IS THE WHOLE POINT. `ttrl.engine.recurrent.GRU` applies its mask
    to the state ENTERING timestep `t`, so the mask at `t` means "the episode
    ended before `t`". A transition carries its own flag at its END. So
    `starts[t] = boundary[t - 1]`, and `starts[0]` comes from the flag carried
    over from the previous rollout, which is why `carried` is an argument and
    not a zero.

    Doing the shift once, here, is what lets every consumer hand the unroll
    `starts` verbatim. A consumer that redid it would get the chance to get it
    wrong again, and a reset placed one step off is a silent wrong answer: the
    forward pass still runs and the loss still falls.

    `boundary` is `terminated OR truncated`, which is NOT
    `RolloutBuffer.dones`. See `RolloutBuffer.add`.
    """
    boundary = np.asarray(boundary, dtype=np.float32)
    if boundary.ndim != 2:
        raise ValueError(
            f"shift_to_starts: boundary must be [T, n_envs], got shape "
            f"{boundary.shape}"
        )
    carried = np.asarray(carried, dtype=np.float32).reshape(1, -1)
    if carried.shape[1] != boundary.shape[1]:
        raise ValueError(
            f"shift_to_starts: carried covers {carried.shape[1]} lanes and "
            f"boundary covers {boundary.shape[1]}"
        )
    return np.concatenate([carried, boundary[:-1]], axis=0)


def gather(array: np.ndarray, chunks: np.ndarray, lanes: np.ndarray,
           seq_len: int) -> np.ndarray:
    """`(seq_len, n_seq, *element)` of a `[T, n_envs, *element]` rollout array.

    `chunks[i]` and `lanes[i]` address sequence `i`: timesteps
    `chunks[i] * seq_len ... + seq_len - 1` of lane `lanes[i]`. The lane index
    is broadcast down the time axis unchanged, which is what makes "one
    sequence, one lane" structural.
    """
    chunks = np.asarray(chunks, dtype=np.int64)
    lanes = np.asarray(lanes, dtype=np.int64)
    if chunks.shape != lanes.shape or chunks.ndim != 1:
        raise ValueError(
            f"gather: chunks {chunks.shape} and lanes {lanes.shape} must be "
            f"one matching 1-D index per sequence"
        )
    steps = chunks[None, :] * int(seq_len) + np.arange(seq_len, dtype=np.int64)[:, None]
    return array[steps, np.broadcast_to(lanes[None, :], steps.shape)]


def flatten_time_major(array: np.ndarray) -> np.ndarray:
    """`(seq_len, n_seq, *element)` -> `(seq_len * n_seq, *element)` rows.

    Row `t * n_seq + sequence`, which is plain C-order, so the inverse is a
    `reshape` back to `(seq_len, n_seq, ...)` and no consumer permutes
    anything. See THE TIME-MAJOR FLATTEN above for why this is a named
    function.
    """
    if array.ndim < 2:
        raise ValueError(
            f"flatten_time_major: expected at least (seq_len, n_seq), got shape "
            f"{array.shape}"
        )
    rows = int(array.shape[0]) * int(array.shape[1])
    return np.ascontiguousarray(array).reshape(rows, *array.shape[2:])


def store_hidden(arrays: dict, t: int, seq_len: int | None, hidden) -> None:
    """Write the state entering step `t` into each memory's chunk row.

    `arrays` is `{memory name: (n_chunks, n_envs, size)}`. A hidden state is
    stored at a sequence's first step and nowhere else, so a feedforward
    buffer (`seq_len` None) stores none and a mid-sequence step that was
    handed one is a caller that has lost track of where it is, which raises
    rather than being dropped.

    The state stored is the UNMASKED one entering the step. The reset lives in
    the mask the unroll applies on top of it, which is the same op in the same
    position as the acting step's, so the two cannot drift apart. Storing the
    masked state instead and also passing `starts` double-masks harmlessly,
    and storing it masked while not passing `starts` silently drops every
    reset that falls inside a sequence.
    """
    wanted = seq_len is not None and t % seq_len == 0
    if not wanted:
        if hidden is not None:
            raise ValueError(
                f"store_hidden: step {t} does not open a sequence "
                f"(seq_len={seq_len}), so it stores no hidden state"
            )
        return
    if hidden is None:
        raise ValueError(
            f"store_hidden: step {t} opens a sequence and needs the hidden "
            f"state entering it, one entry per memory {sorted(arrays)}. Without "
            f"it the update starts every sequence from zero and throws away the "
            f"memory this policy acted with."
        )
    if set(hidden) != set(arrays):
        raise ValueError(
            f"store_hidden: hidden carries memories {sorted(hidden)} and this "
            f"buffer stores {sorted(arrays)}"
        )
    chunk = t // seq_len
    for name, array in arrays.items():
        rows = np.asarray(hidden[name], dtype=np.float32)
        if rows.shape != array.shape[1:]:
            raise ValueError(
                f"store_hidden: hidden[{name!r}] has shape {rows.shape}, "
                f"expected {array.shape[1:]}"
            )
        array[chunk] = rows
