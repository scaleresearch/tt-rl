"""The typed run configuration.

`TrainConfig` alone. The two halves it is built from live beside it:
`ttrl.train.config.device` for the chip and `ttrl.train.config.env` for the
environment, each with its own validation, and
`ttrl.train.config.horizon` for the one field a run shares with the
algorithm's own config. All are re-exported here, so every existing
`from ttrl.train.config.schema import ...` still reads.

A run is written by building a `TrainConfig` literal in a plain Python module
under `configs/`. Every validator raises `ValueError` or `TypeError`, never
`assert`: `python -O` strips asserts, and a config that validates itself only
in a non-optimised interpreter does not validate itself.

Nothing here names an environment suite or an algorithm. The environment is a
registered name plus its builder's keywords, described by `EnvConfig` in
`ttrl.train.config.env`; the algorithm is a registered
name plus that algorithm's OWN config object, which lives with the algorithm.
Restating an algorithm's hyperparameters here would be a second place for
every one of them to disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ttrl.algos.config import check_algo_config
from ttrl.algos.scaling import check_network_batch
from ttrl.train.checkpoint.best import MODES, refuse_metric
from ttrl.train.config.device import DeviceConfig
from ttrl.train.config.env import EnvConfig, align_reward_discount
from ttrl.train.config.horizon import RUN_HORIZON_FIELD, align_run_horizon

__all__ = ["DeviceConfig", "EnvConfig", "LOG_RECORDS_PER_RUN",
           "RUN_HORIZON_FIELD", "TrainConfig"]


#: About how many per-update records a run keeps when `log_interval` derives
#: itself. A property of the reader rather than of any run's length, which is
#: why it is one number for every run: 1000 points resolve a learning curve at
#: one point per 0.1% of the run, and 1000 records of about 195 bytes are
#: about 0.2 MB, which opens in anything.
LOG_RECORDS_PER_RUN = 1_000


@dataclass
class TrainConfig:
    """One run: what to train, on what, for how long, and where it is kept."""

    algo: str = "dqn"
    total_frames: int = 10_000_000
    eval_interval: int = 250_000
    eval_episodes: int = 30
    checkpoint_interval: int = 500_000
    #: How many of a run's periodic checkpoints to keep, newest first. See
    #: `ttrl.train.checkpoint.history` for why the bound is a number here
    #: rather than unbounded or a constant.
    checkpoint_keep: int = 10
    #: Whether a periodic checkpoint also carries the replay buffer, for an
    #: algorithm that has one. OFF by default and deliberately a choice: a
    #: 1M-transition Atari buffer of stacked uint8 frames is 28 GB, and a run
    #: checkpointing every 100k frames with this on would fill a disk in an
    #: afternoon. Off does NOT mean a resume quietly starts with no
    #: experience: `ttrl.train.resume` refuses a warm resume from a file that
    #: carries none, because the alternative is a run that continues its
    #: exploration schedule and its update clock over an empty buffer and
    #: converges somewhere worse while reporting itself as a continuation.
    checkpoint_replay: bool = False
    #: Which eval metric picks the run's best checkpoint, and which direction
    #: is better. The metric is one of the names an eval emits, such as
    #: `eval/mean`; `ttrl.train.evaluate.eval_metric_names` is the whole set it
    #: may take, and a name outside it is refused here rather than at the first
    #: eval, which is frames and a chip later. Configurable rather
    #: than fixed to the mean because the quantity a run is judged on is the
    #: user's to state: a success rate is still a mean, but a control cost or a
    #: tracking error is minimised, and a run selecting on it has no other way
    #: to say so. See `ttrl.train.checkpoint.best`.
    #:
    #: BEST MEANS THE HIGHEST-SCORING EVAL, which on a task whose policy
    #: moves between checkpoints is biased upward by roughly how far it moves
    #: from one eval to the next: two Hopper SAC seeds gave closing evals of
    #: 3260.5, 3283.7, 3298.5 and of 732.5, 197.6, 809.0, both at a within-eval
    #: std near 15, and only the second seed's best is mostly luck. The final
    #: checkpoint is the honest one to publish. The stamp records the eval
    #: before the kept one so the file says how exceptional its own score was;
    #: see `ttrl.train.checkpoint.best`.
    best_metric: str = "eval/mean"
    best_metric_mode: str = "max"
    #: Frames between the per-update records a run writes to its metrics
    #: sink, or None to derive the spacing from the run's own length.
    #:
    #: Derived is `max(1, total_frames // LOG_RECORDS_PER_RUN)`, so a run of
    #: any length keeps about `LOG_RECORDS_PER_RUN` update records: a
    #: 1M-frame run logs every 1000 frames and a 2000-frame debug run every
    #: 2. An update record costs about 195 bytes of JSONL and an off-policy
    #: run takes one gradient step per frame, so the 1M-frame run that left a
    #: 200 MB metrics.jsonl keeps about 0.2 MB of update records under this
    #: while the 2000-frame run still shows whether its loss moves. One flat
    #: number cannot do both: the 1000 that thins the long run to a point per
    #: 0.1% of it leaves the short run nothing to read at all.
    #:
    #: What is stored is what was declared, `None` included, and
    #: `log_interval_frames` below is what a reader uses. The alternative,
    #: overwriting the field with the resolved number, freezes a spacing
    #: derived from the length the config was written with: a checkpoint, or
    #: an override that shortens `total_frames`, would then carry a 10,000
    #: frame spacing into a 200-frame run and silence it. The derivation
    #: reads only `total_frames`, which the checkpoint records beside this,
    #: so a run that recorded `None` resolves to the same number when it is
    #: read back.
    #:
    #: An integer is exactly that spacing. `log_interval=1` keeps a record
    #: per update, which is what a reader comparing a whole loss sequence
    #: asks for.
    #:
    #: Eval records, episode-end records and the records a run writes once
    #: (its status, its stop reason, an error) are written whatever this says:
    #: there are hundreds of them over a run and they are the curve someone
    #: reads afterwards. So is a run's first update record; see
    #: `ttrl.train.trainer.Trainer._log`.
    log_interval: int | None = None
    log_dir: str = "runs/default"
    seed: int = 0
    # Where to write the machine-readable throughput report, if anywhere.
    # Empty by default: a run keeps no state unless asked.
    profile_json: str = ""
    # Checkpoint to resume from, or "" to start fresh.
    resume_path: str = ""

    device: DeviceConfig = field(default_factory=DeviceConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    #: The environment a score is taken on, or None to score on `env`. They
    #: differ whenever training applies shaping a published number must not
    #: include, such as reward clipping or a per-life episode boundary: the
    #: eval protocol is fixed and never tuned, so it is stated separately
    #: rather than derived from the training env by a flag.
    eval_env: EnvConfig | None = None
    #: The algorithm's own config object, or None for its defaults. Its type
    #: and its fields belong to the algorithm (`ttrl.algos.<name>`), which is
    #: the only place they are validated. The one field this layer reaches
    #: into is `total_frames`, because the run owns that number; see
    #: `ttrl.train.config.horizon`.
    algo_cfg: Any = None

    #: True when the algorithm's run horizon was left unset and so follows
    #: `total_frames`. Unannotated for the reason
    #: `EnvConfig._seed_follows_run_seed` is, and read where that one is.
    _run_horizon_follows_run = False

    #: Fields a checkpoint must not carry, because they describe the
    #: invocation rather than the agent.
    #:
    #: `device`: which chip a run sat on, how large its L1 scratch was and how
    #: long it waited for the chip. A stored chip id would send a scoring run
    #: at whichever chip the training run happened to get.
    #:
    #: `resume_path` and `profile_json`: where this invocation was told to
    #: resume from and where to write its throughput report. The first is
    #: consumed before the resume it names even begins, and neither touches
    #: what the run computes. They are also the reason they are listed here
    #: rather than merely tolerated: `ttrl.train.resume` refuses a warm
    #: resume from a file that did not record a config field, and a field
    #: nothing can be lost by defaulting has no business triggering that.
    #: `ttrl.train.provenance` is where a run's launch details are recorded.
    #:
    #: Read by `ttrl.train.checkpoint.tree`, which also drops these from an
    #: older file that carries them.
    #: NOT annotated: a class attribute, so it stays out of fields().
    UNSAVED_FIELDS = ("device", "resume_path", "profile_json")

    def __post_init__(self) -> None:
        if not self.algo:
            raise ValueError(
                "algo must name a registered algorithm; "
                "ttrl.algos.registry.ALGOS.names() lists them"
            )
        if self.total_frames < 1:
            raise ValueError(f"total_frames must be >= 1, got {self.total_frames}")
        if self.eval_interval < 1:
            raise ValueError(f"eval_interval must be >= 1, got {self.eval_interval}")
        if self.eval_episodes < 1:
            raise ValueError(f"eval_episodes must be >= 1, got {self.eval_episodes}")
        if self.checkpoint_interval < 1:
            raise ValueError(
                f"checkpoint_interval must be >= 1, got {self.checkpoint_interval}"
            )
        if self.checkpoint_keep < 1:
            raise ValueError(
                f"checkpoint_keep must be >= 1, got {self.checkpoint_keep}. A run "
                "keeps at least its newest checkpoint; raise it to keep more of "
                "the progression, at about 27 MB each for Atari."
            )
        if self.log_interval is not None and self.log_interval < 1:
            raise ValueError(
                f"log_interval must be >= 1, got {self.log_interval}. 1 writes "
                "a record for every update, which is what a reader comparing a "
                "whole loss sequence wants and what left a 1M-frame run a "
                "200 MB metrics file. Leave it None for a spacing derived from "
                "total_frames."
            )
        if not self.best_metric:
            raise ValueError(
                "best_metric must name an eval metric, such as 'eval/mean'. A run "
                "always keeps its best-scoring checkpoint; there is no setting "
                "that turns the selection off, because a run with no best file is "
                "a run whose good policy the retention bound eventually deletes."
            )
        # The directions are the selector's own, and this layer holds no second
        # copy of them. Imported here because `ttrl.train.evaluate` imports
        # this module.
        from ttrl.train.evaluate import eval_metric_names

        # Asked NOW rather than at the first eval, which on an Atari config is
        # 250k frames and a chip after the typo. The names are derived from
        # what an eval computes and the refusal is the selector's own, so the
        # early answer and the late one are one set and one sentence.
        known = eval_metric_names()
        if self.best_metric not in known:
            raise refuse_metric(self.best_metric, known)
        if self.best_metric_mode not in MODES:
            raise ValueError(
                f"best_metric_mode must be one of {MODES}, got "
                f"{self.best_metric_mode!r}"
            )
        if not self.log_dir:
            raise ValueError("log_dir must be a non-empty path")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        # Resolve the env's seed stream now, so every consumer reads a plain
        # int and `seed=N` alone really does move the whole run.
        if self.env._seed_follows_run_seed:
            self.env.seed = self.seed
        if self.eval_env is not None and self.eval_env._seed_follows_run_seed:
            self.eval_env.seed = self.seed
        self.algo_cfg, self._run_horizon_follows_run = align_run_horizon(
            self.algo_cfg, self.total_frames)
        for env in (self.env, self.eval_env):
            if env is not None:
                align_reward_discount(env, self.algo_cfg, self.algo)
        if self.algo_cfg is not None:
            # Before check_network_batch, which reads one of the members this
            # checks for: a config missing them is told what to add rather
            # than raising an AttributeError from inside a library file.

            check_algo_config(self.algo_cfg, self.algo)

            check_network_batch(self.algo_cfg, self.env.n_envs, self.algo)
            # Eval is the OTHER width the acting path is asked for, and it
            # comes from a different EnvConfig. Refused here, before a chip is
            # opened, rather than from inside the first eval: that one lands
            # after `eval_interval` frames of training, with the run's own
            # output as the only clue.
            #
            # `min` because `ttrl.train.evaluate` never runs more lanes than
            # episodes, so a 64-lane eval env asked for 2 episodes acts at 2
            # and refusing it would be wrong.
            check_network_batch(
                self.algo_cfg,
                min(self.eval_episodes, self.eval_env_config.n_envs),
                self.algo,
                field="min(eval_episodes, eval_env.n_envs)",
            )

    @property
    def log_interval_frames(self) -> int:
        """Frames between per-update records: `log_interval`, or derived.

        Always a positive integer, so every reader takes the spacing from
        here and none of them repeats the derivation.
        """
        if self.log_interval is not None:
            return self.log_interval
        return max(1, self.total_frames // LOG_RECORDS_PER_RUN)

    @property
    def eval_env_config(self) -> EnvConfig:
        """The environment to score on. `env` when no separate one was given."""
        return self.env if self.eval_env is None else self.eval_env
