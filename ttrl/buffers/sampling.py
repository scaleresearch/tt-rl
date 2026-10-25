"""Prioritized replay: the one sampler that keeps a weight per stored column.

The seam it implements is declared in `ttrl.buffers.samplers`, which also
holds the uniform sampler. `ttrl.buffers.replay` owns the storage and the
n-step read and this owns the distribution, which is what lets prioritized
replay exist without a second buffer class, and it is where the lane quota
lives, because a quota is a statement about the distribution and not about
the ring.

THE DRAW STAYS INSIDE `indexing.sampleable`'s RUN BY CONSTRUCTION. A sampler
is handed a `SampleWindow` and returns POSITIONS in it, never ring columns, so
a priority cannot pull in the column the write cursor has already overwritten;
`ttrl.buffers.samplers` carries the whole of that contract. Priorities
ARE kept for every column of the ring, including the few the positional bound
excludes: a priority table whose live set tracked the moving window would
have to be re-derived on every write, and getting that wrong reads a
successor observation from the next lap, which is wrong training data and not
an error. The cost of keeping them all is that an excluded column's priority
sits unread until the window passes over it.

THE DRAW IS A CUMSUM OVER THE RUN, NOT A SUM TREE, AND THAT HAS A PRICE WORTH
KNOWING. A sum tree is O(log n) per sample against this O(run) per batch.
Indicative figures for the O(run) cumsum, per batch of 32: 0.07 ms uniform
and 0.71 ms prioritized at a 100k-transition buffer, 0.06 ms and 6.26 ms at
1M. No artifact in this repo records them, so read them for the shape of the
growth with capacity rather than as numbers for this machine. So at
`DQNConfig.buffer_capacity`'s default of 100k the draw is lost in a gradient
step, and at the 1M of `configs/pong_dqn.py` it is NOT: 6 ms is a real share
of one Atari update, and a run that wants prioritized replay at that capacity
should expect it.

The cumsum is here anyway rather than the tree because the tree needs the
EXCLUSION ZONE as state beside the write cursor, and the invariant above says
what getting that wrong does. The replacement, when someone needs 1M-capacity
PER: a sum tree over the whole ring with no exclusion at all, drawing and then
REJECTING any index outside `sampleable`'s run. The rejected set is at most
n_step + 1 columns of a lane, so the rejection rate is negligible, the draw
stays exactly proportional over the allowed set, and the bound is still read
from `indexing.sampleable` rather than cached beside the cursor.

Beta is NOT annealed here. It is a function of the update count, which the
agent already owns and already checkpoints, and a second clock inside the
buffer would be a second answer to "how far into the run is this".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ttrl.buffers.samplers import Draw, SampleWindow

__all__ = ["PriorityConfig", "PrioritySampler", "beta_at",
           "build_priority_sampler"]


@dataclass(frozen=True)
class PriorityConfig:
    """Prioritized replay's own settings. Opt-in: a run without one is uniform.

    `alpha` is how sharply a priority bends the distribution (0 is uniform,
    1 is fully proportional). `beta` is how much of the resulting bias the
    importance-sampling weights correct, annealed from `beta_start` to
    `beta_end` over `beta_anneal_updates` gradient steps, because an early
    correction of a sharply biased distribution is mostly variance.
    `epsilon` is the floor added to every priority so a transition whose TD
    error reached exactly zero is still eventually redrawn.
    """

    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_anneal_updates: int = 1_000_000
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                f"PriorityConfig: alpha must be in [0, 1], got {self.alpha}. "
                f"Set alpha=0 for uniform sampling with the weights still "
                f"computed, or leave priority unset for plain uniform replay."
            )
        for name in ("beta_start", "beta_end"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"PriorityConfig: {name} must be in [0, 1], got {value}. "
                    f"beta is the exponent of the importance-sampling "
                    f"correction, fully corrected at 1."
                )
        if self.beta_anneal_updates < 1:
            raise ValueError(
                f"PriorityConfig: beta_anneal_updates must be >= 1 gradient "
                f"steps, got {self.beta_anneal_updates}"
            )
        if self.epsilon <= 0.0:
            raise ValueError(
                f"PriorityConfig: epsilon must be > 0, got {self.epsilon}. A "
                f"zero floor lets a transition whose TD error hit exactly zero "
                f"fall out of the distribution permanently."
            )


def beta_at(cfg: PriorityConfig, update: int) -> float:
    """The annealed beta at a gradient-step count. Linear, then held."""
    frac = min(1.0, max(0, int(update)) / cfg.beta_anneal_updates)
    return cfg.beta_start + frac * (cfg.beta_end - cfg.beta_start)


class PrioritySampler:
    """Proportional sampling over the sampleable run, with IS weights.

    One priority per ring column, held as `(p + epsilon) ** alpha` so the draw
    needs no exponentiation. A column written for the first time, or
    overwritten, takes the largest raw priority seen so far, which is what
    gets a fresh transition drawn at least once before its own TD error is
    known.
    """

    def __init__(self, cfg: PriorityConfig, *, lanes: int, columns: int):
        self.cfg = cfg
        #: float64: a batch's weights divide by the total, and summing 1e5
        #: float32 priorities loses the small ones into the running sum.
        self._weight = np.zeros((lanes, columns), dtype=np.float64)
        self._max_priority = 1.0

    def _scaled(self, priority: np.ndarray) -> np.ndarray:
        return (priority + self.cfg.epsilon) ** self.cfg.alpha

    def on_write(self, lane: int, col: int) -> None:
        self._weight[lane, col] = self._scaled(np.float64(self._max_priority))

    def draw(self, rng, window: SampleWindow, *, batch_size: int,
             beta: float | None) -> Draw:
        """`batch_size` positions in `window`, drawn proportionally to priority.

        The lane quota is proportional to each lane's priority SUM rather than
        to its count, which is what makes the result proportional over the
        union of the runs instead of over each run separately.
        """
        if beta is None:
            raise ValueError(
                "ReplayBuffer.sample: this buffer samples by priority, so the "
                "batch's importance-sampling weights need a beta. Pass "
                "beta=ttrl.buffers.sampling.beta_at(cfg, update_count)."
            )
        runs = [self._run(window, lane) for lane in range(window.n_lanes)]
        sums = np.array([w.sum() for w in runs], dtype=np.float64)
        total = window.total
        if sums.sum() <= 0.0:
            raise RuntimeError(
                f"replay: {total} transitions are sampleable but every "
                f"priority in the sampleable runs is zero, so there is no "
                f"distribution to draw from. Every write stores a positive "
                f"priority, so this is a bug in the priority bookkeeping."
            )
        quota = rng.multinomial(batch_size, sums / sums.sum())
        lanes_out, offsets_out, probs = [], [], []
        for lane, k in enumerate(int(v) for v in quota):
            if k == 0:
                continue
            weight = runs[lane]
            picked = _searchsorted_draw(rng, weight, k)
            lanes_out.append(np.full(k, lane, dtype=np.int64))
            offsets_out.append(picked)
            probs.append(weight[picked] / sums.sum())
        lanes = np.concatenate(lanes_out)
        offsets = np.concatenate(offsets_out).astype(np.int64)
        return Draw(lanes, offsets,
                    _is_weights(np.concatenate(probs), total, float(beta)))

    def _run(self, window: SampleWindow, lane: int) -> np.ndarray:
        """One lane's scaled priorities, in run order, so position `p` of the
        run and `weight[p]` are the same transition."""
        columns = window.columns(lane)
        if columns.shape[0] == 0:
            return np.zeros(0, np.float64)
        return self._weight[lane, columns]

    def update(self, lanes: np.ndarray, cols: np.ndarray,
               priorities: np.ndarray) -> None:
        """Record fresh priorities for columns a batch was drawn from."""
        priority = np.abs(np.asarray(priorities, dtype=np.float64).reshape(-1))
        if priority.shape != np.asarray(lanes).reshape(-1).shape:
            raise ValueError(
                f"ReplayBuffer.update_priorities: {priority.shape[0]} "
                f"priorities for {np.asarray(lanes).reshape(-1).shape[0]} "
                f"sampled transitions; one per sampled transition, in the "
                f"order the batch carries them"
            )
        if not np.all(np.isfinite(priority)):
            raise ValueError(
                "ReplayBuffer.update_priorities: a priority is NaN or "
                "infinite, which would make one transition's sampling "
                "probability the whole distribution. The TD errors of a "
                "diverged update produce this; stop the run rather than "
                "clipping it here."
            )
        self._weight[lanes, cols] = self._scaled(priority)
        self._max_priority = max(self._max_priority, float(priority.max()))

    # -- checkpointing -----------------------------------------------------

    def state(self) -> dict:
        # A copy, for the reason `ttrl.buffers.persist.replay_state` copies.
        return {"weight": self._weight.copy(),
                "max_priority": float(self._max_priority)}

    def load(self, state: dict) -> None:
        stored = np.asarray(state["weight"], dtype=np.float64)
        if stored.shape != self._weight.shape:
            raise ValueError(
                f"replay priorities in the checkpoint are shaped "
                f"{stored.shape} and this buffer's table is {self._weight.shape}"
            )
        self._weight[...] = stored
        self._max_priority = float(state["max_priority"])


def _searchsorted_draw(rng, weight: np.ndarray, k: int) -> np.ndarray:
    """`k` indices into `weight`, drawn proportionally to it."""
    cdf = np.cumsum(weight)
    picked = np.searchsorted(cdf, rng.random(k) * cdf[-1], side="right")
    # A uniform draw of exactly cdf[-1], or float error in the cumsum, lands
    # one past the end; that row is the last column and not a wrapped one.
    return np.minimum(picked, weight.shape[0] - 1)


def _is_weights(probs: np.ndarray, total: int, beta: float) -> np.ndarray:
    """`(N * P_i) ** -beta`, normalised by the batch maximum.

    Normalised by the batch's own maximum rather than by the buffer-wide
    smallest probability: both remove the scale, and the batch maximum needs
    no second pass over the run. Every weight is then in (0, 1], so a
    prioritized update's effective learning rate is bounded by the uniform
    one's instead of being raised by whatever the rarest transition is.
    """
    weights = (total * probs) ** (-beta)
    return (weights / weights.max()).astype(np.float32)


def build_priority_sampler(*, lanes: int, columns: int,
                           config: PriorityConfig | None = None) -> PrioritySampler:
    """The `SAMPLERS["priority"]` entry. `config` is the run's `PriorityConfig`.

    Required rather than defaulted: alpha and the beta schedule change what a
    run converges to, so a run that asked for prioritized replay without
    saying how sharply says so here instead of being given a house default.
    """
    if config is None:
        raise ValueError(
            "the 'priority' sampler needs its settings: pass "
            "priority=PriorityConfig(...) alongside sampler='priority', or "
            "set the priority block in the algorithm's config, which selects "
            "this sampler on its own."
        )
    return PrioritySampler(config, lanes=lanes, columns=columns)
