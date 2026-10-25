"""Batch containers and the spec-to-array mapping both buffers share.

Nothing here imports ttnn. Element shapes and dtypes come from each leaf's
`ObsSpec` and from `ActionSpec`, so a buffer never names an observation
geometry of its own. A batch's observations are an `Observation`, the leaf
mapping every consumer of a batch reads by name.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ttrl.spaces import ActionSpec, ObsSpec, Observation

__all__ = [
    "ReplayBatch",
    "RolloutMinibatch",
    "RolloutSequenceBatch",
    "action_dtype",
    "action_element_shape",
    "obs_element_shape",
]


def obs_element_shape(obs_spec: ObsSpec) -> tuple[int, ...]:
    return tuple(int(d) for d in obs_spec.shape)


def action_element_shape(action_spec: ActionSpec) -> tuple[int, ...]:
    """Per-transition action shape: scalar id for discrete, a vector otherwise.

    A multi_discrete action stores one index per factor, so its element is a
    vector of `len(nvec)` integers.
    """
    if action_spec.kind == "discrete":
        return ()
    if action_spec.kind == "multi_discrete":
        return (len(action_spec.nvec),)
    return (int(action_spec.dim),)


def action_dtype(action_spec: ActionSpec) -> np.dtype:
    integral = action_spec.kind in ("discrete", "multi_discrete")
    return np.dtype("int64" if integral else "float32")


@dataclass(frozen=True)
class ReplayBatch:
    """One sampled off-policy batch.

    `discounts` is gamma ** (number of rewards actually summed into
    `rewards`), which is not gamma ** n_step for a window cut short by an
    episode boundary. Carrying it per sample is what lets every window be
    sampleable: the alternative is excluding short windows from the sampleable
    set, which silently distorts the sampling distribution. A consumer forms
    the n-step target as `rewards + discounts * (1 - dones) * V(next_obs)`.
    """

    obs: Observation
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: Observation
    dones: np.ndarray
    discounts: np.ndarray
    lanes: np.ndarray
    #: Which ring column of its lane each sample came from. `(lane, col)` is
    #: the only key that identifies a stored transition, so a prioritized run
    #: writes its fresh TD error back by this pair.
    cols: np.ndarray
    #: `(B, n)` legal-action mask of the successor state, or None for an
    #: environment whose `ObsSpec` declares no mask. The bootstrap maxes over
    #: the NEXT state's legal set, so this is the mask a Double-DQN target
    #: needs and the mask of `obs` is not it.
    next_mask: np.ndarray | None = None
    #: `(B,)` importance-sampling weights in (0, 1], or None for a uniformly
    #: sampled batch, which needs no correction. A loss reduces per-sample
    #: terms against these INSTEAD of taking a plain mean: prioritized
    #: sampling is a biased estimator of the uniform expectation, and the
    #: weights are what make the gradient an estimate of the same quantity
    #: the uniform run estimates.
    weights: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.rewards.shape[0])


@dataclass(frozen=True)
class RolloutMinibatch:
    """One flattened on-policy minibatch. `actions` is (B,) int64 for discrete
    actions and (B, dim) float32 for continuous ones."""

    obs: Observation
    actions: np.ndarray
    old_logp: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    #: `(B, n)` legal-action mask of `obs`, or None when the spec declares no
    #: mask. On-policy, so the state being scored is the state acted in.
    mask: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.old_logp.shape[0])


@dataclass(frozen=True)
class RolloutSequenceBatch:
    """One on-policy minibatch of whole sequences, for a recurrent update.

    TWO LAYOUTS, DELIBERATELY. Everything the loss reduces over is rows, flat
    `(seq_len * n_seq, ...)` in the time-major order
    `ttrl.buffers.sequences.flatten_time_major` defines, so the per-row
    algebra of a feedforward update reads these fields unchanged. `starts` is
    `(seq_len, n_seq, 1)` because its consumer is the unroll, which splits it
    on the time axis and applies one slice per timestep.

    `h0` is one `(n_seq, hidden)` array PER MEMORY, keyed by the name the
    network gives that memory. An asymmetric observation gives the actor and
    the critic separate memories over different observations, so there is no
    single hidden state for a sequence to carry.

    `valid` is 0.0 for an autoreset padding slot. A padding slot CANNOT be
    dropped from a sequence, because dropping changes seq_len and so the graph
    shape, so every loss term is reduced against this mask instead. It is not
    merely tolerable to keep them: under the NEXT_STEP autoreset contract a
    padding slot's observation is the true first observation of the new
    episode and its `starts` flag is 1, so the state it produces is that
    episode's genuine first hidden state while its fabricated reward and
    action stay out of the loss.
    """

    obs: Observation
    actions: np.ndarray
    old_logp: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    #: `(seq_len, n_seq, 1)`: 1.0 where a timestep begins an episode, which is
    #: the mask the unroll applies to the state entering that timestep.
    starts: np.ndarray
    #: `(seq_len * n_seq,)` rows, 1.0 for a real transition.
    valid: np.ndarray
    #: memory name -> `(n_seq, hidden)` state entering each sequence's first
    #: step, UNMASKED. The unroll masks it with `starts[0]`, exactly as the
    #: acting step masks the carry, so the two cannot disagree.
    h0: dict
    seq_len: int
    #: Which lane each sequence came from, one per sequence. Carried so that
    #: "one sequence, one lane" is checkable by a consumer and not only by
    #: reading `ttrl.buffers.sequences`.
    lanes: np.ndarray
    mask: np.ndarray | None = None

    @property
    def n_seq(self) -> int:
        return int(self.lanes.shape[0])

    def __len__(self) -> int:
        return int(self.old_logp.shape[0])
