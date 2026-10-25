"""When periodic work is due: the run's own eval and checkpoint schedule, and
the declared protocol for the one an agent decides for itself.

`Cadence` is the frame-keyed schedule the trainer runs eval and checkpointing
on. `UpdateCadence` is the other one: WHEN TO LEARN, which every agent already
owns as its `should_update`/`update` pair and which a user may therefore
replace without touching the loop. It is declared here, beside the schedule
the run owns, so the two answers to "is it time yet" are read together.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["Cadence", "UpdateCadence", "UpdateRecords", "WORK_FIELD",
           "check_cadence", "check_cadence_shape"]

from ttrl.train.logging import UPDATE_PREFIX

#: The metrics key an `update()` reports its gradient-step count under, and
#: the work count `check_cadence` measures. One call is one batch of steps,
#: not one step: `ttrl.algos.dqn.agent.DQN.update` takes every step the
#: frames so far are worth and reports the number here. A cadence that always
#: takes exactly one step per call may omit it, and each call then counts as
#: one step.
WORK_FIELD = "gradient_steps"


@runtime_checkable
class UpdateCadence(Protocol):
    """When to learn, and the learning. An agent is one of these already.

    ACCRUING AND ASKING ARE SEPARATE CALLS, and that is the whole invariant.
    `should_update(frame)` is a QUERY: it tells the cadence the frame the run
    has reached and reads back whether work is owed, and asking it twice at
    one frame answers twice the same. `update()` is the work.

    Collapsing the two is the bug this protocol exists to name. A predicate
    that accrues as a side effect of being asked cannot be asked on anyone
    else's behalf, and nothing in its signature says so: a caller that asks
    one agent and then calls `update()` on several leaves every other agent at
    zero credit, doing the least work it can however much data the round
    carried. On two chips the arithmetic costs replica 1 87.5% of its
    gradient steps.
    `ttrl.train.loop.run_loop` therefore asks every replica, and
    `check_cadence` refuses a cadence that cannot be asked that way.
    """

    def should_update(self, frame: int) -> bool: ...

    def update(self) -> dict: ...


def check_cadence(factory, *, total_frames: int = 256, coarse: int = 4,
                  fine: int = 2) -> None:
    """Hold an `UpdateCadence` to the accrue-and-ask invariant, or raise.

    Takes a zero-argument FACTORY rather than one cadence, because the
    invariant is a comparison between separate runs of the same cadence and
    one instance cannot answer it.

    THE MEASUREMENT. The same total frames are delivered twice, once in
    rounds of `coarse` frames and once in rounds of `fine`, so the second run
    asks more often for the same frames. A cadence that accrues from the
    frame number it is told owes the same total work either way. One that
    accrues from being asked owes more the more often it is asked, and that is
    the collapse: its credit is a function of the caller's loop shape, so a
    caller that asks it less often, or not at all on a round it still calls
    `update()` for, silently starves it. A third run asks twice at each
    coarse frame, which is the same fault with the frames held fixed.

    WHAT IS COUNTED IS GRADIENT STEPS, read from `WORK_FIELD` in the metrics
    `update()` returns, not calls to `update()`. One call is a batch of as
    many steps as the frames it was handed are worth, so a correct cadence
    owing one step per frame takes four steps per coarse call and two per
    fine call: equal work, different call counts. Counting calls rejected
    that, which is the batching DQN and SAC use.

    The repeated asks at one frame must also agree with each other, which is
    the same invariant read at a single frame: a query answers from the frame
    it is told and therefore answers twice the same.
    """
    probe = factory()
    check_cadence_shape(probe)
    who = type(probe).__name__
    reported: set[bool] = set()
    runs = {
        f"{coarse} frames a round":
            _drive(factory(), total_frames, coarse, 1, reported),
        f"{fine} frames a round":
            _drive(factory(), total_frames, fine, 1, reported),
        f"{coarse} frames a round, asked twice":
            _drive(factory(), total_frames, coarse, 2, reported),
    }
    if len(set(runs.values())) != 1:
        counted = (
            f"The counts are gradient steps, read from {WORK_FIELD!r} in the "
            f"metrics update() returns."
            if True in reported else
            f"The counts are calls to update(), because its metrics carry no "
            f"{WORK_FIELD!r}. If one call takes as many gradient steps as the "
            f"frames it was handed are worth, report that number under "
            f"{WORK_FIELD!r} and the work rather than the calls is compared."
        )
        raise AssertionError(
            f"{who} did a different number of gradient steps for the same "
            f"{total_frames} frames depending on how often it was asked: "
            f"{runs}. Being asked is how a cadence learns the frame the run "
            f"has reached, so the credit must come from the frame number the "
            f"ask carries and not from the ask itself. A cadence that accrues "
            f"per call owes whatever the caller's loop shape happens to be, so "
            f"a replica asked less often than it is updated does less work "
            f"than its data is worth and nothing reports it: that cost replica "
            f"1 of a two-chip run 87.5% of its gradient steps. {counted}"
        )
    if not any(runs.values()):
        raise AssertionError(
            f"{who}.should_update never said yes over {total_frames} frames, "
            f"so this check proved nothing about it. Raise total_frames past "
            f"its first update, or, if it triggers on data it collects rather "
            f"than on the frame count, drive those writes from the factory."
        )


def check_cadence_shape(cadence) -> None:
    """The members an `UpdateCadence` answers, with nothing driven.

    Separate from `check_cadence` because the measurement below DOES WORK: it
    drives a cadence through hundreds of frames and calls `update()` on every
    one it owes. That is exactly what a caller holding a cadence over a run's
    LIVE agent must not do before the run's first frame, so
    `ttrl.train.trainer` checks this half against the real cadence and leaves
    the invariant to the user, who can drive a stand-in.
    """
    who = type(cadence).__name__
    for method in ("should_update", "update"):
        if not callable(getattr(cadence, method, None)):
            raise TypeError(
                f"{who} is not an UpdateCadence: it has no callable "
                f"{method!r}. A cadence answers should_update(frame) and does "
                f"the work in update(). Add it, and hold the result to the "
                f"accrue-and-ask invariant with ttrl.check_cadence, which "
                f"takes a factory and drives it over a stand-in agent."
            )


def _drive(cadence, total_frames: int, frames_per_round: int, asks: int,
           reported: set[bool]) -> int:
    """How many gradient steps `cadence` took over `total_frames`, asked
    `asks` times at each round's frame.

    `reported` collects whether the metrics carried `WORK_FIELD`, so the
    caller's message can say which of the two counts it compared.
    """
    work = 0
    who = type(cadence).__name__
    for frame in range(frames_per_round, total_frames + 1, frames_per_round):
        answers = [bool(cadence.should_update(frame)) for _ in range(asks)]
        if len(set(answers)) > 1:
            raise AssertionError(
                f"{who}.should_update answered {answers} at frame {frame}: "
                f"the same frame, asked {asks} times, answered differently. "
                f"should_update is a query and the frame it is told is all it "
                f"may answer from, because the loop asks every replica and a "
                f"predicate whose answer depends on the asking cannot be "
                f"asked on anyone else's behalf. Move whatever the ask "
                f"changes into update()."
            )
        if answers[-1]:
            metrics = cadence.update()
            if not isinstance(metrics, dict):
                raise TypeError(
                    f"{who}.update returned {type(metrics).__name__}; it "
                    f"returns a dict of scalar metrics, which is what the "
                    f"loop logs and what a hook reads"
                )
            steps = metrics.get(WORK_FIELD)
            reported.add(steps is not None)
            if steps is not None and not isinstance(steps, (int, float)):
                raise TypeError(
                    f"{who}.update reported {WORK_FIELD}="
                    f"{steps!r} ({type(steps).__name__}); it is this call's "
                    f"gradient-step count and must be a number"
                )
            work += 1 if steps is None else int(steps)
    return work


class Cadence:
    """One periodic job, keyed on a run's frame count.

    Markers are anchored to the ABSOLUTE frame grid, the largest multiple of
    `interval` at or below the starting frame, not to the starting frame
    itself. On a fresh run the two are identical; on a resume they are not,
    and anchoring to the resume point re-phases the whole schedule off the
    round frame numbers the run's earlier work landed on (resume at 190 with
    interval 100 would fire at 290, 390, ... instead of 200, 300, ...), which
    makes the pre- and post-resume halves of one run's metrics un-joinable.
    """

    def __init__(self, interval: int, *, frame: int = 0) -> None:
        if interval < 1:
            raise ValueError(f"cadence interval must be >= 1, got {interval}")
        self.interval = interval
        self.last = (frame // interval) * interval

    def due(self, frame: int) -> bool:
        """Whether the job is due at `frame`, advancing the marker if it is.

        The marker advances by whole multiples of `interval`, never to
        `frame`. An iteration collects several frames at once, so `frame`
        generally overshoots the threshold, and snapping the marker to it
        throws the overshoot away: every period then costs `interval` plus one
        iteration's frames and the error compounds for the whole run. At
        n_envs=3 with interval=10 that is a firing every 12 frames rather than
        every 10, so a run configured for 40 evals gets 33 and the grid drifts
        off the round frame numbers a results table joins on.
        """
        elapsed = frame - self.last
        if elapsed < self.interval:
            return False
        self.last += self.interval * (elapsed // self.interval)
        return True


class UpdateRecords:
    """Which of a run's metric records reach its sink.

    The trainer asks this rather than reading key names itself: whether a
    record is one gradient step's metrics is a fact about the record's shape,
    and how often those are kept is the run's declared `log_interval`.

    A record counts as an update record only when EVERY key carries
    `UPDATE_PREFIX`, so a record that mixes update metrics with anything else
    is never thinned away.

    The first update record is kept whatever the interval says, so a watched
    run shows its loss as soon as it has one and an interval set by hand
    cannot make a run silent about whether it learns at all. The cadence
    marker is left alone by it, so the records after it still land on the
    absolute frame grid two runs join on.
    """

    def __init__(self, interval: int, *, frame: int = 0):
        self._due = Cadence(interval, frame=frame)
        self._kept_one = False

    @staticmethod
    def is_update(metrics: dict[str, Any]) -> bool:
        return bool(metrics) and all(
            key.startswith(UPDATE_PREFIX) for key in metrics
        )

    def wants(self, metrics: dict[str, Any], step: int) -> bool:
        if not self.is_update(metrics):
            return True
        first, self._kept_one = not self._kept_one, True
        return first or self._due.due(step)
