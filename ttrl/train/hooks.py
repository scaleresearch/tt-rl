"""The loop's hook points, as one object a caller passes to a `Trainer`.

`ttrl.train.loop.run_loop` has taken `on_metrics`, `on_round_end` and
`on_update` since it was written, and none of it was reachable from a
`TrainConfig` or a `Trainer`: the periodic work was eval, checkpoint and
logger, so curriculum, an external learning-rate override, a custom metric or
a stop meant forking the loop. These are those same three points plus a stop,
handed over unchanged.

THREE POINTS AND A STOP, not a callback class with a method per phase. The
loop has three places where anything happens; a five-method base class would
invent two of them and would make the set of hooks a class hierarchy to
subclass rather than a function to pass. A hook is any callable, so a user
whose state spans several hooks writes one object with three bound methods and
passes them.

ORDER: the library's own periodic work runs FIRST, then the matching hook. A
hook at `on_round_end` therefore sees the eval that round produced and the
checkpoint already on disk, which is what makes it possible to react to them.
`stop` is asked last of all, after the hooks, so a condition sees whatever
they did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ttrl.train.stop import RunState

__all__ = ["TrainHooks"]


@dataclass(frozen=True)
class TrainHooks:
    """What a caller wants called, and nothing the loop does not already call.

    `on_metrics(metrics, step)` sees every metric the run logs, which is the
    seam for a tracker of a user's own or for a derived metric.

    `on_round_end(frame, elapsed)` runs once per round, after the round's eval
    and checkpoint. Curriculum, an annealing override and a wall-clock budget
    live here.

    `on_update(metrics)` runs once per round in which the replicas learned,
    after parameter averaging, and is handed that update's own metrics under
    the `train/` keys the logger sees. It took no argument, which left a hook
    blind to the update it fires after and so useless for the three things
    hooks are for: a metric of a user's own, a curriculum that reacts to the
    loss, and an early stop on something no `RunState` field carries. The
    other points are not blind in the same way: `on_metrics` already sees
    every eval score and every rollout line, and `stop` reads the run's score
    through `RunState`, so neither is widened for symmetry.

    `stop(state)` returns a reason to end the run, or None.
    `ttrl.train.stop` holds the ready-made ones and the `RunState` they read.
    """

    on_metrics: Callable[[dict, int], None] | None = None
    on_round_end: Callable[[int, float], None] | None = None
    on_update: Callable[[dict], None] | None = None
    stop: Callable[[RunState], str | None] | None = None
