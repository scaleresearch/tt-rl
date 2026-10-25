"""Stop conditions: what fires, what it says, and that every one is tested."""

from __future__ import annotations


from ttrl.train import stop as stop_mod
from ttrl.train.stop import (
    AnyOf,
    RunState,
    StopOnNoImprovement,
    StopOnRewardThreshold,
)


def _state(**kwargs) -> RunState:
    base = dict(frame=1000, metric="eval/mean", mode="max", evals=3, score=5.0,
                best=9.0, best_frame=500, evals_since_best=1)
    return RunState(**{**base, **kwargs})


def test_a_threshold_fires_once_the_score_passes_it():
    reason = StopOnRewardThreshold(5.0)(_state(score=5.0))
    assert reason and "eval/mean is 5" in reason and "frame 1000" in reason


def test_a_threshold_reads_the_runs_own_direction():
    """The same condition means "at most" for a run minimising a cost."""
    low = _state(score=4.0, mode="min")
    assert StopOnRewardThreshold(5.0)(low)
    assert StopOnRewardThreshold(5.0)(_state(score=6.0, mode="min")) is None


def test_no_improvement_fires_at_the_patience():
    condition = StopOnNoImprovement(3)
    assert condition(_state(evals_since_best=2)) is None
    reason = condition(_state(evals_since_best=3))
    assert reason and "has not improved on 9" in reason and "frame 500" in reason


def test_any_of_returns_the_first_reason_in_order():
    condition = AnyOf(StopOnRewardThreshold(100.0), StopOnNoImprovement(1))
    reason = condition(_state(score=5.0, evals_since_best=1))
    assert "has not improved" in reason
    assert condition(_state(score=5.0, evals_since_best=0)) is None


#: Every condition this module drives, keyed by the exported name. Checked
#: against `stop.__all__` below, so a condition added without a test here
#: fails rather than passing unexercised.
EXERCISED = {
    "StopOnRewardThreshold": StopOnRewardThreshold(1.0),
    "StopOnNoImprovement": StopOnNoImprovement(1),
    "AnyOf": AnyOf(StopOnRewardThreshold(1.0)),
}


def test_every_exported_condition_is_exercised_here():
    """Derived from `stop.__all__`, never from a list typed out by hand."""
    conditions = {
        name for name in stop_mod.__all__
        if name not in ("RunState", "run_state")
    }
    assert conditions == set(EXERCISED)

