"""One running mean and variance, fitted on batches.

THE shared estimator. Observation whitening and reward scaling are two uses
of one quantity, the running moments of a stream, so there is one
implementation of it and both normalisers hold an instance.

Fitted on BATCHES, which is the whole point. A vector environment advances
every lane on one step, and an estimator that saw one lane's rows saw 1/N of
the data and described a distribution the other lanes do not draw from.
`update` takes the whole lane batch and folds it in with Chan's parallel
form, which is Welford's algorithm for two groups and is exact in one pass:
the result does not depend on how the stream was cut into batches.

Host-side numpy, in the environment layer. Neither device invariant binds
here: nothing on this path holds a `ttnn` tensor, so there is no `.data` to
rebind and no address for a captured trace to replay against, and the
arithmetic runs on the host before any observation reaches a device buffer.
`ttrl.train.loop` writes observations into the device buffers after the
environment has returned them, so every number this touches is already on
the host.

float64 throughout. The device pipeline computes in fp32, and an estimator
accumulating a long run's second moment in the network's precision loses the
variance it exists to report.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["RunningMeanStd"]


class RunningMeanStd:
    """Running mean and population variance of a stream of rows of `shape`.

    `count` is the number of ROWS folded in, not the number of batches, so
    the statistics of an N-lane run and of an N-times-longer single-lane run
    over the same rows are the same numbers.
    """

    def __init__(self, shape: tuple[int, ...]):
        self.shape = tuple(int(d) for d in shape)
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.m2 = np.zeros(self.shape, dtype=np.float64)
        self.count = 0.0

    def update(self, batch: np.ndarray) -> None:
        """Fold a `(rows, *shape)` batch in. One call per environment step."""
        rows = np.asarray(batch, dtype=np.float64)
        if rows.ndim != len(self.shape) + 1 or rows.shape[1:] != self.shape:
            raise ValueError(
                f"RunningMeanStd over rows of shape {self.shape} was given a "
                f"batch of shape {rows.shape}; a batch is (rows, *{self.shape})"
            )
        n = float(rows.shape[0])
        if n == 0.0:
            raise ValueError(
                "RunningMeanStd.update was given an empty batch. A vector "
                "environment steps at least one lane, so an empty batch is a "
                "caller that lost its lanes rather than a step with no data."
            )
        batch_mean = rows.mean(axis=0)
        # Population second moment of the batch about its own mean, which is
        # what Chan's form combines; `np.var` with ddof=0 is the same thing
        # divided by n, so it is multiplied back rather than recomputed.
        batch_m2 = rows.var(axis=0) * n
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (n / total)
        self.m2 = self.m2 + batch_m2 + delta * delta * (self.count * n / total)
        self.count = total

    def variance(self) -> np.ndarray:
        """Population variance, or ones until a second row has been seen.

        One row has no variance, and the alternative to a placeholder is a
        division by zero in the first step of every run. The placeholder is
        ones rather than epsilon so the first observation passes through at
        its own scale instead of being multiplied by 1e8.
        """
        if self.count > 1.0:
            return self.m2 / self.count
        return np.ones_like(self.m2)

    def std(self, epsilon: float) -> np.ndarray:
        return np.sqrt(self.variance() + float(epsilon))

    def get_state(self) -> dict[str, Any]:
        return {"mean": self.mean.copy(), "m2": self.m2.copy(),
                "count": self.count}

    def set_state(self, state: dict[str, Any]) -> None:
        mean = np.asarray(state["mean"], dtype=np.float64)
        m2 = np.asarray(state["m2"], dtype=np.float64)
        if mean.shape != self.shape or m2.shape != self.shape:
            raise ValueError(
                f"running statistics of shape {mean.shape}/{m2.shape} loaded "
                f"into an estimator over rows of shape {self.shape}. The "
                f"observation or reward stream this was fitted on is not the "
                f"one this run produces. Resume with the env config the "
                f"checkpoint stores (ttrl.config_from_checkpoint prints it), "
                f"or start the run afresh."
            )
        self.mean, self.m2 = mean.copy(), m2.copy()
        self.count = float(state["count"])
