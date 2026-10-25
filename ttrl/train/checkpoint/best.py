"""The best-scoring checkpoint a run has seen, kept beside its history.

A return curve is not monotonic, so a run's newest checkpoint is routinely a
worse policy than one from millions of frames earlier. The frame-keyed history
in `ttrl.train.checkpoint.history` is bounded and prunes oldest first, so the
good one is also the one that gets deleted. This keeps it under one stable
name, `checkpoint-best.npz`, which the history's twelve-digit pattern does not
match and `prune` therefore never considers.

L1, NOT L0. A score is not a resume, and the weights alone would be the
smaller file. But L1 is a superset of what L0 offers:
`ttrl.train.restore.agent_from_checkpoint` already reads weights out of a
resumable file with `weights_only=True`, and `ttrl.deploy.export_policy` takes
an agent whatever level the file carried, so an L0 best file would buy nothing
except being the one checkpoint in the run a user cannot continue from.
Continuing from the best policy is most of why anyone asks for it. The cost is
one file at full size, about 27 MB for Atari, and exactly one: this name is
overwritten and never accumulated, so a run's checkpoint footprint grows by one
file and not by one per improvement.

THE SCORE CARRIES ITS CONTEXT. Selecting the maximum over many evals picks a
high draw, so the stamp records the kept eval's own `eval/std`, the eval
immediately before it, and the mean of the evals seen so far, and `describe`
turns that into the line a user meets at the end of a run and on
`--resume best`. It changes nothing about WHICH checkpoint is kept.

THE MOVEMENT THAT MATTERS IS LOCAL, between the kept eval and the one before
it. Dispersion over the whole run is the learning curve plus any excursion
the run ever had, and both belong to a run that is working. Measured on three
real series, as a ratio of movement to the kept eval's within-eval std:

    statistic          Hopper s0   Hopper s1   Breakout PPO
    std over the run       103.5         9.7            0.5
    mean |consecutive|      51.0         7.5            0.1
    median |consecutive|    24.9         1.3            0.1
    |previous eval|          1.1        32.7            0.1

Hopper seed 1 kept 808.96 straight after an eval of 197.59 and must be
flagged; Hopper seed 0 kept 3298.47 after 3283.73 and must not, though it
collapsed to 1150.84 three evals earlier; Breakout PPO climbs 10.03 to 125.17
and must not. Only the last row separates all three, because it asks what
produced THIS checkpoint. One difference is one sample, which is the honest
weakness of it, and it costs nothing: `last` already holds the previous eval.

THE SCORE LIVES IN THE FILE, in `extra`, with the metric and the direction it
was judged on. The file and the high-water mark therefore cannot disagree, and
a user who deletes the file does not leave a mark behind that suppresses every
later save. A RESUME READS IT BACK, which is what keeps the first eval after a
resume from crowning a policy the run had already beaten. Recovery happens on
a resume and not on a fresh run: a fresh run has earned no best score, and
inheriting the previous occupant's would stop it from ever saving one.
"""

from __future__ import annotations

import os
from typing import Any

from ttrl.train.checkpoint.read import read_meta
from ttrl.train.logging import EVAL_PREFIX

__all__ = ["BEST", "MODES", "BestScore", "best_path", "describe",
           "refuse_metric"]

#: The stable name a run's best checkpoint is written under.
BEST = "checkpoint-best.npz"

#: Which direction counts as better. A return is maximised; a control cost or
#: a tracking error is minimised, and such a run has no other way to say so.
MODES = ("max", "min")

#: The context keys a stamp carries, with the value meaning "this file does
#: not say". `best_eval_std` is the kept eval's own episodes, `best_prev_eval`
#: the eval before it, and the rest the evals seen up to and including it.
NO_CONTEXT: dict[str, Any] = {
    "best_eval_std": None,
    "best_evals": None,
    "best_evals_mean": None,
    "best_prev_eval": None,
}

#: How far the jump from the previous eval must exceed the within-eval std
#: before the selection is worth warning about. A judgement call, placed with
#: room on both sides of the measured series in this module's docstring:
#: Breakout PPO sits at 0.1, Hopper seed 0 at 1.1, Hopper seed 1 at 32.7.
UNSTABLE_RATIO = 3.0


def describe(stamp: dict[str, Any]) -> str:
    """One line saying what the best checkpoint scored and how exceptional it was.

    ONE WORDING, several callers: the end-of-run report, `--resume best` and
    the examples all hold the same dict, from `outcome.json` or from the
    file's own `extra`.
    """
    if stamp.get("best_score") is None:
        return "no best checkpoint: no eval improved on anything"
    parts = [f"best {stamp.get('best_metric')} {stamp['best_score']:g} "
             f"at frame {stamp.get('best_frame')}"]
    std = stamp.get("best_eval_std")
    if std is not None:
        parts.append(f"within-eval std {std:g}")
    if stamp.get("best_evals"):
        parts.append(f"{stamp['best_evals']} evals averaging "
                     f"{stamp['best_evals_mean']:g}")
    prev = stamp.get("best_prev_eval")
    if prev is not None:
        jump = abs(stamp["best_score"] - prev)
        parts.append(f"previous eval {prev:g}")
        if std and jump > UNSTABLE_RATIO * std:
            parts.append(
                f"so this is the high draw of a policy that moved "
                f"{jump / std:.0f}x its measurement spread in one eval, and "
                f"the final checkpoint is the honest one to publish"
            )
    return ", ".join(parts)


def refuse_metric(metric: str, known) -> ValueError:
    """THE refusal for a `best_metric` no eval produces. One wording, two callers.

    `TrainConfig.__post_init__` raises it against the names an eval will
    produce, before a chip is opened, and `BestScore.observe` raises it again
    against the names one actually did. The second is not redundant: a run
    whose own hook adds metrics, or a config built by something other than
    `TrainConfig`, reaches `observe` without passing the first.
    """
    return ValueError(
        f"best_metric {metric!r} is not among the metrics a run may select "
        f"on ({sorted(known)}). Name one of those, or log the "
        f"metric you want from your own hook."
    )


def best_path(log_dir: str) -> str:
    """The file a run's best checkpoint belongs in."""
    return os.path.join(log_dir, BEST)


class BestScore:
    """The best score a run has seen on one metric, and when it saw it.

    Decides; it does not write. Only the `Trainer` can serialize an agent, so
    `observe` answers whether this eval earned the file and the Trainer writes
    it, stamped with `stamp()`.
    """

    def __init__(
        self,
        log_dir: str,
        *,
        metric: str,
        mode: str,
        score: float | None = None,
        frame: int | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"best_metric_mode must be one of {MODES}, got {mode!r}")
        self.log_dir = log_dir
        self.metric = metric
        self.mode = mode
        self.score = score
        self.frame = frame
        #: The most recent score, improvement or not. None before the first eval.
        self.last: float | None = None
        self.evals = 0
        #: Evals since the best last moved, which is what a patience is counted
        #: in. Zero on a resume: the pre-resume count is not in the file, and a
        #: patience measured from a number nothing recorded would be a guess.
        self.evals_since_best = 0
        #: Running mean of every eval score this run has seen, which is one
        #: float rather than the scores themselves.
        self._mean = 0.0
        #: What the eval that earned the current file looked like beside its
        #: neighbours. Replaced on every improvement, and recovered on a
        #: resume so it always describes the file that is actually there.
        self.context: dict[str, Any] = dict(NO_CONTEXT)

    @classmethod
    def resumed(cls, log_dir: str, *, metric: str, mode: str) -> BestScore:
        """Recover the best score `log_dir` already holds, if it holds one."""
        path = best_path(log_dir)
        if not os.path.exists(path):
            return cls(log_dir, metric=metric, mode=mode)

        _arrays, meta = read_meta(path)
        extra = meta.get("extra") or {}
        if "best_score" not in extra:
            raise ValueError(
                f"{path!r} carries no 'best_score', so it was not written as this "
                f"run's best checkpoint. A resume reads it to keep the first eval "
                f"from crowning a worse policy, and cannot do that from a file "
                f"that does not say what it scored. Move the file aside, or resume "
                f"into a different log_dir."
            )
        stored = (extra.get("best_metric"), extra.get("best_mode"))
        if stored != (metric, mode):
            raise ValueError(
                f"{path!r} holds the best {stored[0]!r} ({stored[1]}) of the run it "
                f"came from, and this run selects on {metric!r} ({mode}). The two "
                f"scores are not comparable, so continuing would either keep a "
                f"file this run never judged or overwrite one it cannot beat. "
                f"Resume with best_metric={stored[0]!r} and "
                f"best_metric_mode={stored[1]!r}, or resume into a different "
                f"log_dir."
            )
        best = cls(
            log_dir,
            metric=metric,
            mode=mode,
            score=float(extra["best_score"]),
            frame=int(extra["best_frame"]),
        )
        # `get`, because a file written before the stamp carried context is
        # still a file this build can resume from and select against.
        best.context = {k: extra.get(k) for k in NO_CONTEXT}
        return best

    @property
    def path(self) -> str:
        return best_path(self.log_dir)

    def observe(self, metrics: dict[str, Any], frame: int) -> bool:
        """Record this eval's score; answer whether it earned the best file.

        A tie does not displace the incumbent: two equal scores are one
        policy's worth of evidence apiece, and the earlier one reached it on
        fewer frames.
        """
        if self.metric not in metrics:
            raise refuse_metric(self.metric, metrics)
        score = float(metrics[self.metric])
        previous = self.last
        self.evals += 1
        self.last = score
        self._mean += (score - self._mean) / self.evals
        if self.score is None or self._better(score):
            self.score = score
            self.frame = int(frame)
            self.evals_since_best = 0
            self.context = self._context(metrics, previous)
            return True
        self.evals_since_best += 1
        return False

    def _context(self, metrics: dict[str, Any],
                 previous: float | None) -> dict[str, Any]:
        """How the eval just kept sat beside its own episodes and its neighbour.

        Backward-looking only, since the file is written the moment the eval
        wins it and a later eval cannot be in a number already on disk.
        """

        std = metrics.get(f"{EVAL_PREFIX}std")
        return {
            "best_eval_std": None if std is None else float(std),
            "best_evals": self.evals,
            "best_evals_mean": self._mean,
            "best_prev_eval": previous,
        }

    def _better(self, score: float) -> bool:
        return score > self.score if self.mode == "max" else score < self.score

    def stamp(self) -> dict[str, Any]:
        """What the best checkpoint's `extra` carries, so the file says what it is."""
        return {
            "best_metric": self.metric,
            "best_mode": self.mode,
            "best_score": self.score,
            "best_frame": self.frame,
            **self.context,
        }

    def metrics(self) -> dict[str, Any]:
        """The best score as metrics, so the curve carries it beside the eval."""
        if self.score is None:
            return {}
        return {"eval/best": self.score, "eval/best_frame": self.frame}
