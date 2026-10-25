"""What an eval reports, and which of those fields a run may rank itself on.

The summary table lives apart from the driver because it is read by two
callers that never score anything: `TrainConfig` validates `best_metric`
against `eval_metric_names`, and `ttrl.train.checkpoint` selects on the name
it was given. `ttrl.train.evaluate` builds its result dict from the same
table, which is what keeps the two from drifting.
"""

from __future__ import annotations

import statistics

from ttrl.train.logging import EVAL_PREFIX

__all__ = ["EVAL_SUMMARY", "EVAL_UNRANKABLE", "eval_metric_names"]

#: Every summary field `evaluate` reports, and how each is computed. ONE
#: SOURCE: the result dict is built from this table and `eval_metric_names`
#: reads the metric names off the same keys, so what a run may select its
#: best checkpoint on cannot drift from what an eval produces. `scores` is
#: the per-episode evidence rather than a number to rank runs by.
#:
#: `min` and `max` bound the episodes of one eval, which since
#: `ttrl.train.evaluate` seeds each episode separately are starts named by
#: the eval seed rather than whatever the environment's generator had
#: reached. `std` is the sample standard deviation over those episodes, and
#: is what the uncertainty on a published `mean` is stated from; it is 0.0
#: for a one-episode eval, where no spread is measurable. None of the three
#: bounds the uncertainty on the RUN: a policy's score moves between
#: checkpoints by far more than its episodes move within one eval, measured
#: on Hopper at consecutive evals of 2970, 3063, 686 and 1142.
EVAL_SUMMARY = {
    "n_episodes": lambda scores, asked: asked,
    "mean": lambda scores, asked: float(statistics.fmean(scores)),
    "median": lambda scores, asked: float(statistics.median(scores)),
    "min": lambda scores, asked: float(min(scores)),
    "max": lambda scores, asked: float(max(scores)),
    "std": lambda scores, asked: (
        float(statistics.stdev(scores)) if len(scores) > 1 else 0.0),
}

#: Summary fields an eval reports that a run may NOT rank itself on.
#:
#: `n_episodes` holds the same value at every eval, so selecting on it keeps
#: the first checkpoint for the whole run, since nothing later can beat it.
#: `std` is a spread rather than a score: ranking checkpoints by it picks the
#: most erratic policy of the run under `mode="max"` and, under `mode="min"`,
#: the one whose episodes agree most, which a collapsed policy scoring the
#: same low return every episode wins outright. Reporting it is the point;
#: it says how certain `eval/mean` is, and the comparison between two of
#: those means is the reader's.
EVAL_UNRANKABLE = ("n_episodes", "std")


def eval_metric_names() -> tuple[str, ...]:
    """The metric names `TrainConfig.best_metric` may take.

    Derived from `EVAL_SUMMARY` and the prefix rather than typed out, less the
    fields in `EVAL_UNRANKABLE`, which an eval still reports.
    """

    return tuple(f"{EVAL_PREFIX}{name}" for name in EVAL_SUMMARY
                 if name not in EVAL_UNRANKABLE)
