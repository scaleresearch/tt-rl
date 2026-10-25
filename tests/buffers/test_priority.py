"""Prioritized replay: the draw bends toward priority, the weights undo it.

Three claims, one test, because they are one mechanism and checking any of
them alone passes under a defect in the others. A draw that follows priority
with no correction is a biased gradient. A correction with no skew in the
draw is a slower uniform run. And a skew that reaches OUTSIDE
`indexing.sampleable`'s run reads a successor observation the write cursor has
already overwritten, which is the one failure here that raises nothing and
corrupts the training data instead.

The correction is checked as what it is, an importance-sampling estimator: at
beta = 1 the weight is exactly 1 / (N * P_i), so the weighted mean of any
function of the sampled transitions estimates its UNIFORM mean. A test that
only asserted "the weights are smaller for likely samples" would pass for
any decreasing function of the probability.
"""

from __future__ import annotations

import numpy as np

from ttrl.buffers import ReplayBuffer
from ttrl.buffers.indexing import sampleable
from ttrl.buffers.sampling import PriorityConfig
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec

OBS = ObsGroup.of(ObsSpec(shape=(1,), dtype="float32", kind="vector"))
ACT = ActionSpec.discrete(2)


def test_priority_bends_the_draw_and_the_weights_correct_for_it():
    # alpha=1 so the intended probabilities are the priorities themselves and
    # the arithmetic below is exact rather than approximately proportional.
    buf = ReplayBuffer(OBS, ACT, capacity=16, n_step=1, gamma=0.9, seed=0,
                       priority=PriorityConfig(alpha=1.0, epsilon=1e-12))
    for t in range(24):
        buf.add(np.full((1,), float(t), np.float32), 0, 0.0, False, False)

    # The ring holds writes 8..23 and n_step=1 costs the newest two, so the
    # sampleable writes are 8..22 (the cursor is at column 8).
    start, count = sampleable(buf._pos, buf._full, 16, 1)
    assert (int(start[0]), int(count[0])) == (8, 15)

    # Make one sampleable transition ten times as surprising as the rest, and
    # one EXCLUDED transition a thousand times as surprising: write 23 is in
    # the ring and carries a priority, and the positional bound must still
    # keep it out, because its successor observation is not stored.
    lanes = np.zeros(2, np.int64)
    buf.update_priorities(lanes, np.array([12 % 16, 23 % 16]),
                          np.array([10.0, 1000.0]))
    buf.update_priorities(np.zeros(14, np.int64),
                          np.array([c % 16 for c in range(8, 23) if c != 12]),
                          np.full(14, 1.0))

    batch = buf.sample(30_000, beta=1.0)
    drawn = batch.obs["obs"].reshape(-1)
    assert set(np.unique(drawn)) == set(float(w) for w in range(8, 23)), (
        "the draw must stay inside the sampleable run: write 23 carries a "
        "priority a thousand times the rest and is still not drawn, because "
        "its successor observation has been overwritten"
    )

    # The skew. Fourteen transitions at priority 1 and one at 10, so the
    # favoured one should take 10/24 of the batch.
    share = float(np.mean(drawn == 12.0))
    assert abs(share - 10.0 / 24.0) < 0.02, f"drawn {share:.3f} of the time"
    assert share > 5 * np.mean(drawn == 11.0), "priority must bend the draw"

    # The correction. At beta=1 the weighted mean of the sampled values is an
    # importance-sampling estimate of the mean over the sampleable run, which
    # is what the uniform run's gradient is an estimate over.
    weights = batch.weights.astype(np.float64)
    corrected = float(np.sum(weights * drawn) / np.sum(weights))
    uniform = float(np.mean(np.arange(8, 23)))
    skewed = float(np.mean(drawn))
    assert abs(corrected - uniform) < 0.1, (
        f"the importance weights must recover the uniform expectation: "
        f"corrected {corrected:.3f} against {uniform:.3f}"
    )
    assert abs(skewed - uniform) > 0.3, (
        "the unweighted mean must be biased, or there is nothing to correct "
        "and this test would pass with the weights ignored"
    )
    assert weights.max() <= 1.0 and weights.min() > 0.0
