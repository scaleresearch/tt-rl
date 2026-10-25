"""PPO's configuration.

`minibatch_size` is a field rather than a derived quantity because it is the
one hyper-parameter the hardware constrains: a convolution's input gradient
stops fitting in L1 above some batch for a given geometry, and the network is
built for exactly this batch, so an impossible value fails when the agent is
constructed rather than at the first backward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ttrl.algos.scaling import config_hash, divide_for_replicas
from ttrl.nets.choice import coerce_encoder

__all__ = ["PPOConfig", "RecurrentConfig"]


@dataclass(frozen=True)
class RecurrentConfig:
    """Memory for the policy: a GRU between trunk and heads.

    `None` on an algorithm's config means feedforward, so every existing
    config keeps its exact behaviour and a run becomes recurrent by naming
    this block and nothing else:

        ppo:
          recurrent:
            hidden_size: 128
            seq_len: 16

    `seq_len` is the truncated-BPTT window: the update runs over sequences of
    this many consecutive steps of one environment lane, each starting from
    the hidden state the acting policy had at that step. It must divide both
    `n_steps` and `minibatch_size`, for the reasons
    `ttrl.buffers.sequences.chunk_count` and `chunks_per_minibatch` set out:
    one fixed T is what the captured graph is built for, and a minibatch is a
    whole number of sequences.

    Longer sequences credit a reward to a more distant observation and cost
    proportionally more memory in the unrolled backward; shorter ones cut the
    credit path at the window edge. There is no setting that avoids the
    tradeoff, which is why this is a field and not a default buried in the
    agent.
    """

    hidden_size: int = 128
    seq_len: int = 16

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError(
                f"RecurrentConfig: hidden_size must be >= 1, got "
                f"{self.hidden_size}"
            )
        if self.seq_len < 2:
            raise ValueError(
                f"RecurrentConfig: seq_len must be >= 2, got {self.seq_len}. A "
                f"sequence of one step carries no gradient between timesteps, "
                f"so it is a feedforward update paying for a GRU. Leave "
                f"`recurrent` unset for that."
            )


@dataclass(frozen=True)
class PPOConfig:
    lr: float = 2.5e-4
    gamma: float = 0.99
    lam: float = 0.95
    n_steps: int = 128
    minibatch_size: int = 32
    epochs: int = 4
    clip_eps: float = 0.1
    value_clip_eps: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_adv: bool = True
    anneal_lr: bool = False
    #: Total frames the run will collect, the denominator of the LR schedule.
    #: Left unset by a config file and filled from `TrainConfig.total_frames`,
    #: which is the one source for the run's length. `anneal_lr` needs it by
    #: the time it anneals, which is why the refusal is `require_horizon` and
    #: not a check here: raising at construction makes the documented path
    #: impossible, since this object exists before any `TrainConfig` can fill
    #: it in.
    total_frames: int | None = None
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
    #: Memory for the policy, or None for a feedforward one.
    recurrent: RecurrentConfig | None = None

    #: PPO's networks are built for one minibatch, not for the rollout.
    NETWORK_BATCH_FIELD = "minibatch_size"

    #: Fields the networks' shapes are built from. A checkpoint that disagrees
    #: on one of these does not merely describe a different run: its saved
    #: parameters have different shapes, which `describe_config_diff` says out
    #: loud instead of leaving the reader to work out which mismatch matters.
    #: `recurrent` is here because a different hidden_size is not merely a
    #: different run: the GRU's saved parameters have different shapes, and a
    #: feedforward checkpoint has no GRU in it at all.
    SHAPE_FIELDS = ("encoder", "encoder_kwargs", "minibatch_size", "recurrent")

    #: Only the learning-rate schedule's horizon. `n_steps` is deliberately
    #: absent: it is the GAE horizon, so dividing it would shorten the
    #: advantage estimate's lookahead rather than split a budget, and the
    #: replicas' rollouts are already separate.
    REPLICA_SCALED_FIELDS = ("total_frames",)

    @property
    def network_batch(self) -> int:
        return self.minibatch_size

    def require_horizon(self) -> int | None:
        """The horizon the learning-rate schedule will anneal against.

        Checked when the agent is built, which is after `TrainConfig` has had
        its chance to fill the field and still before the first frame is
        collected. A schedule with no end is not a schedule, and annealing
        against an assumed horizon would make the learning rate a function of
        a number nobody chose.
        """
        if not self.anneal_lr:
            return None
        if not self.total_frames:
            raise ValueError(
                "PPOConfig: anneal_lr is set but total_frames is "
                f"{self.total_frames}, so the schedule has no end to decay to. "
                "Leave total_frames unset and let TrainConfig fill it from the "
                "run's own total_frames, or state it here."
            )
        return int(self.total_frames)

    def scale_for_replicas(self, n: int) -> "PPOConfig":
        return divide_for_replicas(self, n, self.REPLICA_SCALED_FIELDS)

    #: One keyword mapping would make a frozen config unhashable; see
    #: `ttrl.algos.scaling.config_hash`.
    __hash__ = config_hash

    def __post_init__(self) -> None:
        if not isinstance(self.encoder_kwargs, dict):
            raise TypeError(
                f"PPOConfig: encoder_kwargs must be a dict of keyword arguments for "
                f"the encoder builder, got {type(self.encoder_kwargs).__name__} "
                f"({self.encoder_kwargs!r}). Write encoder_kwargs={{'hidden': 96}}."
            )
        # `encoder` may be handed a class rather than a name;
        # `coerce_encoder` is the one place that becomes a config value.
        object.__setattr__(self, "encoder", coerce_encoder(self.encoder))
        if self.lr <= 0.0:
            raise ValueError(f"PPOConfig: lr must be > 0, got {self.lr}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"PPOConfig: gamma must be in [0, 1], got {self.gamma}")
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError(f"PPOConfig: lam must be in [0, 1], got {self.lam}")
        if self.n_steps < 1:
            raise ValueError(f"PPOConfig: n_steps must be >= 1, got {self.n_steps}")
        if self.minibatch_size < 1:
            raise ValueError(
                f"PPOConfig: minibatch_size must be >= 1, got {self.minibatch_size}"
            )
        if self.epochs < 1:
            raise ValueError(f"PPOConfig: epochs must be >= 1, got {self.epochs}")
        if self.clip_eps <= 0.0:
            raise ValueError(f"PPOConfig: clip_eps must be > 0, got {self.clip_eps}")
        if self.value_clip_eps <= 0.0:
            raise ValueError(
                f"PPOConfig: value_clip_eps must be > 0, got {self.value_clip_eps}"
            )
        if self.value_coef < 0.0 or self.entropy_coef < 0.0:
            raise ValueError(
                f"PPOConfig: value_coef and entropy_coef must be >= 0, got "
                f"{self.value_coef} and {self.entropy_coef}"
            )
        if self.max_grad_norm <= 0.0:
            raise ValueError(
                f"PPOConfig: max_grad_norm must be > 0, got {self.max_grad_norm}"
            )
        if self.recurrent is not None:
            length = self.recurrent.seq_len
            if self.n_steps % length:
                raise ValueError(
                    f"PPOConfig: recurrent.seq_len={length} does not divide "
                    f"n_steps={self.n_steps}, which would leave a final sequence "
                    f"of {self.n_steps % length} steps. The unroll is built for "
                    f"one fixed T, so a short tail is a second graph shape."
                )
            if self.minibatch_size % length:
                raise ValueError(
                    f"PPOConfig: recurrent.seq_len={length} does not divide "
                    f"minibatch_size={self.minibatch_size}. A recurrent "
                    f"minibatch is a whole number of sequences, so its row "
                    f"count is n_sequences * seq_len."
                )
        if self.total_frames is not None and self.total_frames < 1:
            raise ValueError(
                f"PPOConfig: total_frames must be >= 1 when set, got "
                f"{self.total_frames}"
            )
