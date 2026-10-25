"""`UpdateClock`: the replay-ratio arithmetic, and the ask/accrue separation.

Host-only and genuinely so: `ttrl.algos.clock` imports no ttnn, which
`tests/test_algos_no_ttnn.py` holds as its own case. That matters beyond
tidiness. The clock used to live in `ttrl.algos.offpolicy`, behind that
module's `ttrl.engine` imports, so any probe of the replay ratio reached ttnn
and therefore had to be scoped to a chip before it could run at all.

The defect these tests exist for: `due` was a method that accrued credit as a
side effect of being asked, so "ask" and "accrue" were one call. A trainer
that asked ONE replica on behalf of several left every other replica at zero
credit, and each of those took one gradient step a round however much data the
round carried. On two chips that was measured as a [15840, 1980] split.
"""

from __future__ import annotations

import pytest

from ttrl.algos.clock import UpdateClock


def test_credit_comes_from_frames_not_from_calls():
    """Four lanes at one step per frame is worth four gradient steps."""
    clock = UpdateClock(train_freq=1, warmup=0)
    clock.advance(4)
    assert clock.owed == 4
    assert clock.due is True


def test_advancing_to_the_same_frame_twice_accrues_once():
    """Idempotence in the frame number is what lets every replica be asked."""
    clock = UpdateClock(train_freq=1, warmup=0)
    clock.advance(4)
    clock.advance(4)
    clock.advance(3)
    assert clock.owed == 4


def test_asking_never_accrues():
    """The separation, as a test. Reading `due` and `owed` any number of times
    must leave the clock exactly where it was."""
    clock = UpdateClock(train_freq=2, warmup=0)
    clock.advance(10)
    before = clock.state_dict()
    for _ in range(5):
        assert clock.due is True
        assert clock.owed == 5
    assert clock.state_dict() == before


def test_the_warm_up_frames_earn_no_credit():
    """A run does not owe `warmup / train_freq` steps the moment it ends."""
    clock = UpdateClock(train_freq=1, warmup=100)
    clock.advance(100)
    assert clock.owed == 0
    assert clock.due is False
    clock.advance(103)
    assert clock.owed == 3


def test_skip_advances_without_credit():
    """Frames collected while the buffer cannot be sampled owe nothing, so
    the credit is not banked and spent in one unbounded burst."""
    clock = UpdateClock(train_freq=1, warmup=0)
    clock.skip(50)
    assert clock.owed == 0
    clock.advance(53)
    assert clock.owed == 3


def test_consume_keeps_the_fraction():
    """A run whose frames per round do not divide `train_freq` must not lose
    the remainder every round and drift below its configured ratio."""
    clock = UpdateClock(train_freq=3, warmup=0)
    clock.advance(4)
    assert clock.owed == 1
    clock.consume(1)
    assert clock.owed == 0
    clock.advance(8)
    # 8/3 total credit, one step already taken, so the third frame of the
    # second round completes the second step rather than being dropped.
    assert clock.owed == 1


@pytest.mark.parametrize("n", [1, 4])
def test_n_clocks_advanced_to_one_frame_hold_identical_credit(n):
    """The invariant a multi-replica run depends on.

    Every replica is advanced to the same run-wide frame, so every replica
    must owe the same work. The expected value is derived from `n` and the
    frame count, never from a hand-written list: a literal would stop being a
    check on `n` the moment the caller changed.
    """
    frames_per_round, rounds, train_freq = 2 * n, 10, 1
    clocks = [UpdateClock(train_freq, warmup=0) for _ in range(n)]
    taken = [0] * n
    for round_index in range(1, rounds + 1):
        frame = round_index * frames_per_round
        for i, clock in enumerate(clocks):
            clock.advance(frame)
            assert clock.due is True
            steps = clock.owed
            clock.consume(steps)
            taken[i] += steps

    want = rounds * frames_per_round // train_freq
    assert taken == [want] * n, f"n={n} accrued unevenly: {taken}"
    assert sum(taken) == want * n


def test_state_round_trips():
    clock = UpdateClock(train_freq=3, warmup=5)
    clock.advance(20)
    restored = UpdateClock(train_freq=3, warmup=5)
    restored.load_state_dict(clock.state_dict())
    assert restored.owed == clock.owed
    # And it carries the marker, so a resumed run does not re-earn credit for
    # every frame before the checkpoint.
    restored.advance(20)
    assert restored.owed == clock.owed


@pytest.mark.parametrize("train_freq,warmup", [(0, 0)])
def test_a_non_positive_train_freq_is_refused(train_freq, warmup):
    with pytest.raises(ValueError, match="train_freq must be >= 1"):
        UpdateClock(train_freq, warmup)
