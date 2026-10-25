"""Releasing a run's resources without losing the reason it ended.

`run_steps` is the ladder one training run needs, which is the list of what a
run can own: its outcome record, its throughput report, each replica's agent
and environment, and its log sink. It lives here rather than in
`ttrl.train.trainer` because it is a statement about resources and their
release order, and because it is the one place the order can be read.

Teardown is a ladder of independent steps: a profile write, an agent's
captured trace, an environment's worker processes, a log file. One step
raising must not skip the rest, or a single bad env close strands every other
replica's workers and the log. Equally, a teardown error must never be
reported INSTEAD of the mid-run failure or the interrupt that got us here.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable

__all__ = ["Step", "run_ladder", "run_steps"]


@dataclass(frozen=True)
class Step:
    """One teardown step, and what to do if it fails.

    The action is required by the constructor, not by discipline: a report
    that says a thing broke without saying what to do about it is the failure
    mode this project's diagnostics standard exists to replace.
    """

    label: str
    action: str
    call: Callable[[], None]

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("a teardown step must have a label")
        if not self.action:
            raise ValueError(
                f"teardown step {self.label!r} carries no action. A diagnostic "
                "that reports a failure must say what to do about it."
            )


def run_ladder(steps: Iterable[Step]) -> None:
    """Run every step, then re-raise the first failure if nothing is unwinding.

    A `raise` inside a `finally` discards the exception that got us there, so
    re-raising a teardown error unconditionally reports a flaky `env.close()`
    in place of the mid-step failure, or the Ctrl-C, that actually ended the
    run. A teardown error is worth surfacing; it is never worth surfacing
    INSTEAD. `sys.exc_info()[1] is None` is that test.
    """
    first: tuple[Step, BaseException] | None = None
    for step in steps:
        try:
            step.call()
        except BaseException as exc:
            if first is None:
                first = (step, exc)
            else:
                _report(step, exc, "and was not the first failure")
    if first is None:
        return
    if sys.exc_info()[1] is None:
        raise first[1]
    _report(*first, "while an exception was already propagating")


def _report(step: Step, exc: BaseException, context: str) -> None:
    warnings.warn(
        f"teardown step {step.label!r} failed {context}: {exc!r}. {step.action}",
        stacklevel=3,
    )


def run_steps(*, replicas, logger, outcome: Callable[[], None],
              profile: Callable[[], None]) -> list[Step]:
    """The ladder for one training run, in release order.

    The outcome record goes FIRST, so the file saying how the run ended is
    written before anything that could wedge. The devices are not a step: they
    are released unconditionally by the caller after the ladder, including
    when a step here raised, or a failed `env.close()` would strand every chip
    the run opened.
    """
    steps = [
        Step(
            "outcome",
            "the run's outcome record could not be written, so outcome.json "
            "names an earlier run or is absent; the last 'run/status' record "
            "in metrics.jsonl says the same thing.",
            outcome,
        ),
        Step(
            "profile",
            "the throughput report could not be written; the run itself "
            "finished, so rerun with profile_json unset or fix the path.",
            profile,
        ),
    ]
    for replica in replicas:
        steps.append(
            Step(
                f"agent {replica.index}",
                "the agent holds a captured trace on its device. Check the "
                "chip with `ttrl doctor`; a wedged chip needs the process "
                "killed.",
                replica.agent.close,
            )
        )
        # Only for an injected cadence that declares one: the default cadence
        # IS the agent, closed above, and `UpdateCadence` requires no close.
        cadence_close = (
            None if replica.cadence is replica.agent
            else getattr(replica.cadence, "close", None)
        )
        if cadence_close is not None:
            steps.append(
                Step(
                    f"cadence {replica.index}",
                    "the update cadence passed as cadence= could not be "
                    "closed; whatever it holds beside the agent may still be "
                    "open.",
                    cadence_close,
                )
            )
        steps.append(
            Step(
                f"env {replica.index}",
                "an environment worker may still be running. Check for "
                "orphaned subprocesses before starting another run.",
                replica.env.close,
            )
        )
    steps.append(
        Step(
            "logger",
            "the tail of this run's metrics may be unflushed. Treat the "
            "metrics file as truncated.",
            logger.close,
        )
    )
    return steps
