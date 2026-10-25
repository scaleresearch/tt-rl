"""Ending a run before its frame budget, and saying why.

A run whose score stopped moving at frame 300k has no business holding a chip
until frame 10M. On a box whose chips are shared that is the difference
between one dead run and every other run waiting for it, so a stop condition
here is worth more than the convenience it is elsewhere.

A condition is a callable taking a `RunState` and returning either None or a
REASON: one sentence saying what fired, in the metric's own terms. A string
rather than a bool, because `ttrl.train.provenance.write_run_outcome` records
it and a user reading a run directory weeks later has to be able to tell an
early stop from a completed run from a crash. The stop itself is a `break` in
`ttrl.train.loop.run_loop`, so it leaves through the Trainer's teardown ladder
with the final checkpoint written, exactly as a completed run does.

The conditions read the run's score on the metric `TrainConfig.best_metric`
names, in the direction `best_metric_mode` gives. That is deliberate: a run
has one notion of better, the best-checkpoint selection already states it, and
a stop condition carrying its own would let a run stop on a quantity it did
not select on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

__all__ = [
    "AnyOf",
    "RunState",
    "StopOnNoImprovement",
    "StopOnRewardThreshold",
    "run_state",
]


@dataclass(frozen=True)
class RunState:
    """What a stop condition is allowed to see, and all of it.

    Narrow on purpose. Handing a condition the `Trainer` would make every
    private attribute of a run part of the hook contract.
    """

    frame: int
    metric: str
    mode: str
    #: Evals completed in THIS invocation of the run.
    evals: int
    #: The most recent score, None before the first eval.
    score: float | None
    #: The best score seen, including one recovered from a resume.
    best: float | None
    best_frame: int | None
    evals_since_best: int


def run_state(best, frame: int) -> RunState:
    """The state a condition sees, from a `BestScore` and the frame count."""
    return RunState(
        frame=frame,
        metric=best.metric,
        mode=best.mode,
        evals=best.evals,
        score=best.last,
        best=best.score,
        best_frame=best.frame,
        evals_since_best=best.evals_since_best,
    )


class StopOnRewardThreshold:
    """Stop once the run's score passes `threshold`.

    Which side counts as passing comes from the run's direction, so the same
    condition reads as "at least" for a return and "at most" for a cost.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)

    def __call__(self, state: RunState) -> str | None:
        if state.score is None:
            return None
        passed = (
            state.score >= self.threshold if state.mode == "max"
            else state.score <= self.threshold
        )
        if not passed:
            return None
        return (
            f"{state.metric} is {state.score:g} at frame {state.frame}, which "
            f"passes the {state.mode} threshold {self.threshold:g}"
        )


class StopOnNoImprovement:
    """Stop once `patience_evals` evals in a row have failed to improve.

    Counted in evals and not in frames, because that is the grid the score is
    measured on. The count restarts at a resume: how many evals the earlier
    invocation had already wasted is not in the checkpoint, and a patience
    measured from a number nothing recorded would be a guess.
    """

    def __init__(self, patience_evals: int) -> None:
        if patience_evals < 1:
            raise ValueError(
                f"patience_evals must be >= 1, got {patience_evals}. A patience of "
                "zero would stop the run at its first eval."
            )
        self.patience_evals = int(patience_evals)

    def __call__(self, state: RunState) -> str | None:
        if state.evals_since_best < self.patience_evals:
            return None
        return (
            f"{state.metric} has not improved on {state.best:g} from frame "
            f"{state.best_frame} for {state.evals_since_best} evals"
        )


class AnyOf:
    """The first of several conditions to fire, in the order given.

    The `MultiLogger` of stop conditions. Every condition is asked in order
    and the first reason wins, so the reason recorded names the one condition
    that ended the run rather than a conjunction of all of them.
    """

    def __init__(self, *conditions: Callable[[RunState], str | None]) -> None:
        if not conditions:
            raise ValueError(
                "AnyOf needs at least one condition; a run with none stops at its "
                "frame budget, which is what passing no stop hook already means."
            )
        self.conditions = list(conditions)

    def __call__(self, state: RunState) -> str | None:
        for condition in self.conditions:
            reason = condition(state)
            if reason:
                return reason
        return None
