"""DQN's configuration: every value that changes what the algorithm computes.

Frozen and explicit, so a run's physics is a value a caller constructed rather
than the state of the environment the process happened to start in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ttrl.algos.scaling import config_hash, divide_for_replicas
from ttrl.nets.choice import coerce_encoder
from ttrl.buffers.sampling import PriorityConfig

__all__ = ["DQNConfig"]


@dataclass(frozen=True)
class DQNConfig:
    """Everything that changes what DQN computes. No environment variables."""

    gamma: float = 0.99
    n_step: int = 3
    double: bool = True
    huber_delta: float = 1.0
    lr: float = 1e-4
    batch_size: int = 32
    buffer_capacity: int = 100_000
    train_freq: int = 4
    learning_starts: int = 1_000
    target_sync_interval: int = 8_000
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_frames: int = 1_000_000
    #: Mnih et al. 2015 evaluate at 0.05, not at 0. A policy that never learned
    #: to emit an action its training autoresets supplied for it stalls at zero
    #: epsilon and burns the whole episode budget.
    eval_epsilon: float = 0.05
    #: A registered name from `ttrl.ENCODERS`, a class or factory handed over
    #: directly, or a mapping from leaf name to either. A class is kept as an
    #: `UnnamedEncoder`, which says what it cannot promise on resume.
    encoder: Any = "auto"
    #: Keyword arguments for the encoder builder `encoder` names, e.g.
    #: `{"hidden": 96}` to size a vector tower. They reach that one builder and
    #: a key it does not take raises there; nothing here knows which builder
    #: was asked for. A width is a shape, so this is a `SHAPE_FIELDS` member
    #: and a checkpoint that disagrees says so.
    encoder_kwargs: dict = field(default_factory=dict)
    #: Prioritized experience replay, or None for uniform sampling. OFF by
    #: default and opt-in because it changes what a run converges to: the
    #: gradient is taken over a different distribution of transitions, which
    #: the importance-sampling weights correct in expectation and not sample
    #: by sample. See `ttrl.buffers.sampling`.
    priority: PriorityConfig | None = None
    #: Which transitions a batch is drawn from: a name registered in
    #: `ttrl.SAMPLERS`, or None for the declared default, which is 'priority'
    #: when `priority` is set and 'uniform' otherwise. A name rather than an
    #: object so that it round-trips through a checkpoint; a sampler being
    #: tried before it is worth naming is handed to `ReplayBuffer` directly.
    #: Hold your own to the contract with `ttrl.check_sampler` first.
    sampler: str | None = None
    #: 'traced_fwd_bwd' or 'eager'; see `ttrl.engine.step.TrainStep`. The two
    #: are gated bit-for-bit against each other, and traced is the faster.
    step_mode: str = "traced_fwd_bwd"

    #: The field naming the batch every network is built for, and so the most
    #: lanes an acting forward can be handed.
    NETWORK_BATCH_FIELD = "batch_size"

    #: Fields the networks' shapes are built from. A checkpoint that disagrees
    #: on one of these does not merely describe a different run: its saved
    #: parameters have different shapes, which `describe_config_diff` says out
    #: loud instead of leaving the reader to work out which mismatch matters.
    SHAPE_FIELDS = ("encoder", "encoder_kwargs", "batch_size")

    #: Fields that are budgets measured in frames of the whole run, so an
    #: n-chip run divides them: each replica sees 1/n of the frames.
    #: `train_freq` is deliberately absent, being a rate rather than a budget,
    #: and the update clock already measures it per replica.
    REPLICA_SCALED_FIELDS = ("buffer_capacity", "learning_starts",
                             "eps_decay_frames")

    @property
    def network_batch(self) -> int:
        return self.batch_size

    def scale_for_replicas(self, n: int) -> "DQNConfig":
        return divide_for_replicas(self, n, self.REPLICA_SCALED_FIELDS)


    #: One keyword mapping would make a frozen config unhashable; see
    #: `ttrl.algos.scaling.config_hash`.
    __hash__ = config_hash

    def __post_init__(self) -> None:
        if not isinstance(self.encoder_kwargs, dict):
            raise TypeError(
                f"DQNConfig: encoder_kwargs must be a dict of keyword arguments for "
                f"the encoder builder, got {type(self.encoder_kwargs).__name__} "
                f"({self.encoder_kwargs!r}). Write encoder_kwargs={{'hidden': 96}}."
            )
        if self.sampler is not None and not isinstance(self.sampler, str):
            raise TypeError(
                f"DQNConfig: sampler must be a name registered in ttrl.SAMPLERS, "
                f"got {self.sampler!r}. A config carries a NAME so that it "
                f"round-trips through a checkpoint; hand a sampler object to "
                f"ReplayBuffer(sampler=...) instead, or register yours: "
                f"ttrl.SAMPLERS.register('recency', build_recency)."
            )
        # `encoder` may be handed a class rather than a name;
        # `coerce_encoder` is the one place that becomes a config value.
        object.__setattr__(self, "encoder", coerce_encoder(self.encoder))
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"DQNConfig: gamma must be in [0, 1], got {self.gamma}")
        if self.n_step < 1:
            raise ValueError(f"DQNConfig: n_step must be >= 1, got {self.n_step}")
        if self.huber_delta <= 0.0:
            raise ValueError(f"DQNConfig: huber_delta must be > 0, got {self.huber_delta}")
        if self.lr <= 0.0:
            raise ValueError(f"DQNConfig: lr must be > 0, got {self.lr}")
        if self.batch_size < 1:
            raise ValueError(f"DQNConfig: batch_size must be >= 1, got {self.batch_size}")
        if self.train_freq < 1:
            raise ValueError(f"DQNConfig: train_freq must be >= 1, got {self.train_freq}")
        if self.learning_starts < self.batch_size:
            raise ValueError(
                f"DQNConfig: learning_starts={self.learning_starts} is below "
                f"batch_size={self.batch_size}, so the first update would be asked "
                f"for more transitions than the buffer can hold"
            )
        if self.target_sync_interval < 1:
            # update_count % interval raises ZeroDivisionError thousands of
            # updates into a run, and a negative silently freezes the target.
            raise ValueError(
                f"DQNConfig: target_sync_interval must be >= 1 updates, got "
                f"{self.target_sync_interval}"
            )
        if not 0.0 <= self.eps_end <= self.eps_start <= 1.0:
            raise ValueError(
                f"DQNConfig: the epsilon schedule must satisfy 0 <= eps_end <= "
                f"eps_start <= 1, got eps_start={self.eps_start}, eps_end={self.eps_end}"
            )
        if self.eps_decay_frames < 1:
            raise ValueError(
                f"DQNConfig: eps_decay_frames must be >= 1, got {self.eps_decay_frames}"
            )
        if not 0.0 <= self.eval_epsilon <= 1.0:
            raise ValueError(
                f"DQNConfig: eval_epsilon must be in [0, 1], got {self.eval_epsilon}"
            )
