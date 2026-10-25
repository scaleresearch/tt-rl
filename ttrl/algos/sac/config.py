"""SAC's configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ttrl.algos.scaling import config_hash, divide_for_replicas
from ttrl.nets.choice import coerce_encoder
from ttrl.buffers.sampling import PriorityConfig

__all__ = ["SACConfig"]


@dataclass(frozen=True)
class SACConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    n_step: int = 1
    batch_size: int = 256
    buffer_capacity: int = 100_000
    train_freq: int = 1
    learning_starts: int = 1_000
    #: Polyak coefficient: target <- tau * critic + (1 - tau) * target.
    tau: float = 0.005
    n_critics: int = 2
    #: Widths of the hidden layers between the state-action concatenation and
    #: the scalar Q output, one `Linear -> ReLU` per entry. It must hold at
    #: least one: a critic whose head is a single `Linear` makes Q affine in
    #: the action, so dQ/da is a state-independent constant and SAC's actor,
    #: which maximises exactly that derivative, drives the policy to one
    #: corner of the action cube whatever the state. Two is the default
    #: because with `critic_encoder` resolving to 'identity' on a vector leaf
    #: this tuple is the whole of Q, and two hidden layers of 256 is the
    #: critic Haarnoja et al. 2018 published the continuous-control numbers
    #: with. Both depths are gated against a float64 reference; see
    #: `ttrl.nets.encoders.identity_encoder` for the measurement.
    critic_hidden: tuple[int, ...] = (256, 256)
    learn_alpha: bool = True
    init_alpha: float = 1.0
    #: Entropy the temperature steers towards. None means -action_dim, the
    #: value Haarnoja et al. 2018 use for every continuous task.
    target_entropy: float | None = None
    #: A registered name from `ttrl.ENCODERS`, a class or factory handed over
    #: directly, or a mapping from leaf name to either. A class is kept as an
    #: `UnnamedEncoder`, which says what it cannot promise on resume.
    encoder: Any = "auto"
    #: The same, for the tower each critic runs BEFORE its action is
    #: concatenated on, which is where the action enters Q. 'auto' here is
    #: resolved against the CRITIC role, not the observation role: a vector
    #: leaf gives 'identity', a tower of no layers, so `concat(s, a)` is the
    #: critic's input and `critic_hidden` is the whole of Q, which is the
    #: published SAC critic. An image leaf gives its own conv tower and fuses
    #: the action after it, because an action cannot be concatenated onto a
    #: plane of pixels; 'identity' refuses an image leaf by name rather than
    #: guessing a fusion from the shape. The tables are
    #: `ttrl.nets.multi.DEFAULT_ENCODER_BY_ROLE`. Separate from `encoder`
    #: because the policy's tower and the critic's answer different questions:
    #: the policy has no action to fuse.
    critic_encoder: Any = "auto"
    #: Keyword arguments for the encoder builder `encoder` names, e.g.
    #: `{"hidden": 96}` to size a vector tower. They reach that one builder and
    #: a key it does not take raises there; nothing here knows which builder
    #: was asked for. A width is a shape, so this is a `SHAPE_FIELDS` member
    #: and a checkpoint that disagrees says so.
    encoder_kwargs: dict = field(default_factory=dict)
    #: The same, for the builder `critic_encoder` names. Separate from
    #: `encoder_kwargs` because the two towers are two independent choices:
    #: the critic's 'auto' is 'identity' on a vector leaf and takes no
    #: `hidden`, so one shared mapping would make sizing the policy's tower
    #: and fusing on the critic's input mutually exclusive.
    critic_encoder_kwargs: dict = field(default_factory=dict)
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

    NETWORK_BATCH_FIELD = "batch_size"

    #: Fields the networks' shapes are built from. A checkpoint that disagrees
    #: on one of these does not merely describe a different run: its saved
    #: parameters have different shapes, which `describe_config_diff` says out
    #: loud instead of leaving the reader to work out which mismatch matters.
    SHAPE_FIELDS = ("encoder", "encoder_kwargs", "critic_encoder",
                    "critic_encoder_kwargs", "batch_size", "n_critics",
                    "critic_hidden")

    #: See DQNConfig.REPLICA_SCALED_FIELDS. SAC has no exploration schedule to
    #: divide: its exploration is the policy's own entropy, which the
    #: temperature steers from the data rather than from a frame count.
    REPLICA_SCALED_FIELDS = ("buffer_capacity", "learning_starts")

    @property
    def network_batch(self) -> int:
        return self.batch_size

    def scale_for_replicas(self, n: int) -> "SACConfig":
        return divide_for_replicas(self, n, self.REPLICA_SCALED_FIELDS)


    #: One keyword mapping would make a frozen config unhashable; see
    #: `ttrl.algos.scaling.config_hash`.
    __hash__ = config_hash

    def __post_init__(self) -> None:
        if not isinstance(self.encoder_kwargs, dict):
            raise TypeError(
                f"SACConfig: encoder_kwargs must be a dict of keyword arguments for "
                f"the encoder builder, got {type(self.encoder_kwargs).__name__} "
                f"({self.encoder_kwargs!r}). Write encoder_kwargs={{'hidden': 96}}."
            )
        if not isinstance(self.critic_encoder_kwargs, dict):
            raise TypeError(
                f"SACConfig: critic_encoder_kwargs must be a dict of keyword arguments for "
                f"the encoder builder, got {type(self.critic_encoder_kwargs).__name__} "
                f"({self.critic_encoder_kwargs!r}). Write critic_encoder_kwargs={{'hidden': 96}}."
            )
        if self.sampler is not None and not isinstance(self.sampler, str):
            raise TypeError(
                f"SACConfig: sampler must be a name registered in ttrl.SAMPLERS, "
                f"got {self.sampler!r}. A config carries a NAME so that it "
                f"round-trips through a checkpoint; hand a sampler object to "
                f"ReplayBuffer(sampler=...) instead, or register yours: "
                f"ttrl.SAMPLERS.register('recency', build_recency)."
            )
        # `encoder` may be handed a class rather than a name;
        # `coerce_encoder` is the one place that becomes a config value.
        object.__setattr__(self, "encoder", coerce_encoder(self.encoder))
        object.__setattr__(self, "critic_encoder",
                           coerce_encoder(self.critic_encoder))
        if self.lr <= 0.0:
            raise ValueError(f"SACConfig: lr must be > 0, got {self.lr}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"SACConfig: gamma must be in [0, 1], got {self.gamma}")
        if self.n_step < 1:
            raise ValueError(f"SACConfig: n_step must be >= 1, got {self.n_step}")
        if self.batch_size < 1:
            raise ValueError(
                f"SACConfig: batch_size must be >= 1, got {self.batch_size}")
        if self.train_freq < 1:
            raise ValueError(
                f"SACConfig: train_freq must be >= 1, got {self.train_freq}")
        if self.learning_starts < self.batch_size:
            raise ValueError(
                f"SACConfig: learning_starts={self.learning_starts} is below "
                f"batch_size={self.batch_size}, so the first update would be "
                f"asked for more transitions than the buffer can hold"
            )
        if not 0.0 < self.tau <= 1.0:
            raise ValueError(
                f"SACConfig: tau must be in (0, 1], got {self.tau}. A zero tau "
                f"freezes the target network, which trains to garbage with no error."
            )
        if self.n_critics < 2:
            raise ValueError(
                f"SACConfig: n_critics must be >= 2, got {self.n_critics}. The "
                f"target is the minimum over the critics, which with one critic "
                f"is that critic and removes the overestimation correction."
            )
        if not self.critic_hidden or min(self.critic_hidden) < 1:
            raise ValueError(
                f"SACConfig: critic_hidden must hold at least one positive "
                f"width, got {self.critic_hidden!r}. An empty tuple leaves Q "
                f"affine in the action, which removes the state dependence "
                f"SAC's actor needs."
            )
        if self.init_alpha <= 0.0:
            raise ValueError(
                f"SACConfig: init_alpha must be > 0, got {self.init_alpha}; the "
                f"temperature is parameterised as log alpha"
            )
