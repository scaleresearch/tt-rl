"""Where a recurrent rollout says the hidden state resets."""

from __future__ import annotations

import numpy as np

from tests.buffers import _recurrent as R


def test_a_step_starts_an_episode_when_the_previous_step_ended_one():
    """The shift, over the whole rollout at once. A reset placed on the
    boundary step itself, or one step late, fails here."""
    buf = R.new_buffer()
    R.fill(buf)
    expected = np.zeros((R.N_STEPS, R.N_ENVS), dtype=np.float32)
    for step, lane, _terminated in R.BOUNDARIES:
        if step + 1 < R.N_STEPS:
            expected[step + 1, lane] = 1.0
    assert np.array_equal(buf.starts, expected)


def test_passing_dones_where_the_boundary_belongs_loses_a_reset():
    """The injection this file exists to make impossible to land unnoticed."""
    wrong = R.new_buffer()
    R.fill(wrong, boundary_source=lambda t: R.boundary_flags(
        t, terminated_only=True))
    right = R.new_buffer()
    R.fill(right)
    step, lane, _terminated = R.BOUNDARIES[0]
    assert right.starts[step + 1, lane] == 1.0
    assert wrong.starts[step + 1, lane] == 0.0
    assert not np.array_equal(wrong.starts, right.starts)


def test_the_flag_carries_across_the_rollout_seam():
    """A lane that ended its episode on the last stored step begins a fresh
    one on the first step of the NEXT rollout. `reset` therefore leaves the
    carried flag alone, which is the one piece of buffer state that outlives a
    rollout."""
    buf = R.new_buffer()
    R.fill(buf)
    last = [(lane, step) for step, lane, _t in R.BOUNDARIES
            if step == R.N_STEPS - 1]
    assert last
    buf.reset()
    R.fill(buf)
    for lane, _step in last:
        assert buf.starts[0, lane] == 1.0
    others = [lane for lane in range(R.N_ENVS)
              if lane not in {lane for lane, _s in last}]
    for lane in others:
        assert buf.starts[0, lane] == 0.0


def test_an_autoreset_padding_slot_is_also_an_episode_start():
    """Under the NEXT_STEP autoreset contract the padding step carries the new
    episode's first observation, so the same slot is invalid for the loss and
    a genuine episode start for the recurrence. Both at once, which is why a
    padding slot stays in its sequence instead of being dropped."""
    buf = R.new_buffer()
    R.fill(buf)
    padding = np.argwhere(buf.valid == 0.0)
    assert padding.size, "the fill stored no padding slots at all"
    for step, lane in padding:
        assert buf.starts[step, lane] == 1.0
