"""`Trainer`: one run's resources, and the loop driven over them.

A Trainer owns whatever it opened and nothing else. At `chips == 1` the
device belongs to the caller; otherwise the Trainer opened the devices and
releases them. Either way its lifetime is the caller's to bound, with
`with Trainer(...) as trainer:` or an explicit `close()`.
"""

from __future__ import annotations

import contextlib
import sys
from typing import Any, Callable

from ttrl.train.builders import require_episode_bound
from ttrl.train.cadence import (
    Cadence, UpdateCadence, UpdateRecords, check_cadence_shape,
)
from ttrl.train.checkpoint import L1, BestScore
from ttrl.train.config import EnvConfig, TrainConfig
from ttrl.train.hooks import TrainHooks
from ttrl.train.logging import JsonlLogger, Logger
from ttrl.train.loop import StepSource, run_loop
from ttrl.train.periodic import run_eval, write_run_checkpoint
from ttrl.train.provenance import record_run_start, write_run_outcome
from ttrl.train.replicas import open_replicas
from ttrl.train.resume import resume_replicas
from ttrl.train.snapshot import check_replay_support
from ttrl.train.stop import run_state
from ttrl.train.teardown import run_ladder, run_steps
from ttrl.train.throughput import Throughput, write_report

__all__ = ["Trainer"]


class Trainer:
    """Build a run from a `TrainConfig` and run it.

    `cadence`, `source` and `agent` replace the run's swappable stages, and
    are FACTORIES because there is one of each per replica, wrapping something
    that does not exist at config time. `cadence(agent)` returns the
    `ttrl.train.cadence.UpdateCadence` deciding when that replica learns, left
    None the agent is its own; `source(env_config, index)` returns that
    replica's `ttrl.train.loop.StepSource`, the index second so a source can
    shard, left None `ttrl.build_env` builds the env the config names; and
    `agent(built, index)` adapts rather than constructs, taking the agent
    `config.algo` named and returning the one that replica runs, which is how
    a config-driven run reaches `PPO(boundaries=...)`. A cadence holding
    counters is run state; `ttrl.train.resume` says what it demands of one and
    `open_replicas` what the others must be.

    `device` and `profile` are the caller's open device at `chips == 1`, and
    must both be None otherwise. `resume_path` continues a checkpoint; the
    frame count, the config's schedules and the agent's state come out of the
    file, and the experience buffer refills by warm-up.

    Does NOT scope the process to a chip. At `chips == 1` the caller has
    already opened the device and so has already made that decision, and
    scoping once a device is open is refused because it cannot take effect.
    `ttrl.run` is the entry point that scopes, and `ttrl.train.scoping` says
    why it has to happen there and nowhere lower.
    """

    def __init__(
        self,
        config: TrainConfig,
        device=None,
        profile=None,
        *,
        resume_path: str | None = None,
        logger: Logger | None = None,
        hooks: TrainHooks | None = None,
        cadence: Callable[[Any], UpdateCadence] | None = None,
        source: Callable[[EnvConfig, int], StepSource] | None = None,
        agent: Callable[[Any, int], Any] | None = None,
    ) -> None:
        self.config = config
        # Above the unwind stack, which from `open_replicas` on owns a chip:
        # the chip a single-chip run is on is already decided by the time a
        # Trainer exists, so this needs nothing the stack holds and a refusal
        # here leaks nothing. See `ttrl.train.provenance.record_run_start`.
        record_run_start(config)
        self.frame_count = 0
        self.replicas = None
        self.throughput = Throughput(self.frame_count)
        self._closed = False
        self.hooks = TrainHooks() if hooks is None else hooks
        self._status = "incomplete"
        self._stop_reason: str | None = None
        self.best = BestScore(
            config.log_dir,
            metric=config.best_metric,
            mode=config.best_metric_mode,
        )
        # `log_dir` is a config field, so honouring it is the default. A
        # caller that wants a run to keep nothing passes NullLogger.
        self.logger: Logger = JsonlLogger(config.log_dir) if logger is None else logger
        # Under an unwind stack, popped only once initialization has fully
        # succeeded: past `open_replicas` this owns agents, environment
        # workers and, above one chip, the devices, so a refusal raised after
        # that point is a leaked chip.
        with contextlib.ExitStack() as unwind:
            unwind.callback(self._unwind)
            # Before the replicas, the chip and the first frame: an eval
            # environment whose episodes cannot end stalls the first eval, and
            # which one that is was decided when the config was written. Asked
            # of the EVAL env, which the builder path builds whatever `source`
            # is, because `source` replaces only where a run's TRAINING
            # transitions come from. See `require_episode_bound`.
            require_episode_bound(config.eval_env_config, where="eval_env")
            self.replicas = open_replicas(config, device, profile, agent=agent,
                                          cadence=cadence, source=source)
            check_replay_support(config, self.replicas.lead.agent)
            # Beside check_replay_support and inside the unwind, so a cadence
            # that is not one fails before the first frame. The SHAPE half
            # only: the invariant half of `ttrl.check_cadence` drives a
            # cadence through hundreds of frames and takes every gradient step
            # it owes, which against a run's live agent would train it on an
            # empty buffer before the run starts. That half is the user's to
            # run over a stand-in agent, which its diagnostics say.
            for replica in self.replicas.replicas:
                check_cadence_shape(replica.cadence)
            if resume_path:
                self.frame_count = resume_replicas(
                    self.replicas, resume_path, config)
                # The best score has to survive the resume, or the first eval
                # after one is the best this invocation has seen and a worse
                # policy takes the best checkpoint from the better one.
                self.best = BestScore.resumed(
                    config.log_dir,
                    metric=config.best_metric,
                    mode=config.best_metric_mode,
                )
            unwind.pop_all()
        # Both re-anchored to the frame the resume above may have moved to.
        self.throughput = Throughput(self.frame_count)
        self._records = UpdateRecords(
            config.log_interval_frames, frame=self.frame_count)

    # -- lifetime -----------------------------------------------------------

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _unwind(self) -> None:
        """The only release a construction that cannot finish ever gets.

        Through `close()` past `open_replicas`, rather than a second list of
        steps that can drift from the one a finished run unwinds.
        """
        if self.replicas is None:
            self.logger.close()
            return
        self.close()

    def close(self) -> None:
        """Release the run's resources. Safe to call more than once.

        `ttrl.train.teardown` owns the step list, the order and the rule that
        every step runs even when one raises.
        """
        if self._closed:
            return
        self._closed = True
        steps = run_steps(
            replicas=self.replicas.replicas,
            logger=self.logger,
            outcome=self._write_outcome,
            profile=self._write_profile_json,
        )
        try:
            run_ladder(steps)
        finally:
            self.replicas.release_devices()

    # -- running ------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Train to `total_frames` and return the run's result."""
        cfg = self.config
        evals = Cadence(cfg.eval_interval, frame=self.frame_count)
        checkpoints = Cadence(cfg.checkpoint_interval, frame=self.frame_count)

        def on_round_end(frame: int, elapsed: float) -> str | None:
            self.frame_count = frame
            self.throughput.round_ended(frame, elapsed)
            if evals.due(frame):
                self.evaluate()
            if checkpoints.due(frame):
                self.save_checkpoint()
            # The library's own work first, so a hook sees this round's eval
            # and its checkpoint; see `ttrl.train.hooks`.
            if self.hooks.on_round_end is not None:
                self.hooks.on_round_end(frame, elapsed)
            if self.hooks.stop is not None:
                self._stop_reason = self.hooks.stop(run_state(self.best, frame))
                return self._stop_reason
            return None

        def on_update(metrics: dict[str, Any]) -> None:
            self.throughput.updated()
            if self.replicas.param_sync is not None:
                # A collective failure on any chip stops the run: ParamSyncError
                # propagates out of the loop and close() releases every device.
                self.replicas.param_sync.maybe_average()
            if self.hooks.on_update is not None:
                self.hooks.on_update(metrics)

        self.frame_count = run_loop(
            self.replicas.replicas,
            total_frames=cfg.total_frames,
            frame=self.frame_count,
            on_metrics=self._log,
            on_round_end=on_round_end,
            on_update=on_update,
        )
        self.save_checkpoint()
        self._status = "stopped" if self._stop_reason else "completed"
        self._log({"run/status": self._status}, self.frame_count)
        return {
            "frame_count": self.frame_count,
            "replicas": self.replicas.n,
            "updates": self.throughput.updates,
            "status": self._status,
            "stop_reason": self._stop_reason,
            "best": self.best.stamp(),
            "profile": self.profile_report(),
        }

    def _log(self, metrics: dict[str, Any], step: int) -> None:
        """A metric record the run produces, to the sink and then to the hook.

        `UpdateRecords` decides whether a per-update record is kept; every
        other record is written at the frame it happened. A hook reading the
        whole update stream has `TrainHooks.on_update`, which the loop calls
        per update.
        """
        if not self._records.wants(metrics, step):
            return
        self.logger.log(metrics, step)
        if self.hooks.on_metrics is not None:
            self.hooks.on_metrics(metrics, step)

    # -- periodic work ------------------------------------------------------

    def evaluate(self) -> dict:
        """Score the model this run is training, on the eval cadence.

        `ttrl.train.periodic` holds the scoring. This owns what the run does
        with the number: the best checkpoint is written HERE, where the score
        exists, and at L1 so the best policy is one a run can continue from
        as well as score (`ttrl.train.checkpoint.best` says why).
        """
        result, scored = run_eval(self.replicas, self.config)
        if self.best.observe(scored, self.frame_count):
            self.save_checkpoint(self.best.path, extra=self.best.stamp())
        self._log({**scored, **self.best.metrics()}, self.frame_count)
        return result

    def save_checkpoint(self, path: str | None = None, *, level: str = L1,
                        extra: dict | None = None) -> str:
        """Write a checkpoint and return its path. See `ttrl.train.periodic`."""
        return write_run_checkpoint(self.replicas, self.config,
                                    frame=self.frame_count, path=path,
                                    level=level, extra=extra)

    # -- self-measurement ---------------------------------------------------

    def profile_report(self) -> dict[str, float] | None:
        """Wall-clock throughput, measured on every round.
        `ttrl.train.throughput` holds the arithmetic and why it is never
        conditional."""
        return self.throughput.report()

    def _write_outcome(self) -> None:
        """How the run ended, beside the config that started it.

        Written from the teardown ladder, so it is written on the failure path
        too. A run that never reached the end of `run()` is `incomplete`, and
        the exception propagating through teardown, when there is one, names
        what interrupted it.
        """
        exc = sys.exc_info()[1]
        write_run_outcome(
            self.config.log_dir,
            status=self._status,
            reason=self._stop_reason,
            error=None if exc is None else repr(exc),
            frame=self.frame_count,
            profile=self.profile_report(),
            best=self.best.stamp(),
        )

    def _write_profile_json(self) -> None:
        write_report(self.config.profile_json, self.profile_report(),
                     frame=self.frame_count)
