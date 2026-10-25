"""Assembling a recurrent rollout into sequence minibatches."""

from __future__ import annotations

import numpy as np
import pytest

from tests.buffers import _recurrent as R

MINIBATCH = R.SEQ_LEN * R.N_SEQ


def _decode(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(lane, t)` for each observation row `_recurrent.fill` wrote."""
    tags = np.asarray(rows)[:, 0].astype(np.float64)
    lane = np.floor(tags + 1e-6).astype(np.int64)
    step = np.rint((tags - lane) * 100.0).astype(np.int64)
    return lane, step


def _one_batch():
    buf = R.new_buffer()
    R.fill(buf)
    return buf, next(iter(buf.sequence_minibatches(MINIBATCH, epochs=1)))


def test_a_sequence_is_one_lane_and_consecutive_timesteps():
    """The chunk invariant, decoded from the data rather than assumed. A
    gather that mixed lanes, or one that strided time wrongly, fails here."""
    _buf, batch = _one_batch()
    lane, step = _decode(batch.obs["obs"])
    lane = lane.reshape(R.SEQ_LEN, R.N_SEQ)
    step = step.reshape(R.SEQ_LEN, R.N_SEQ)
    for column in range(R.N_SEQ):
        assert len(set(lane[:, column].tolist())) == 1
        assert lane[0, column] == batch.lanes[column]
        times = step[:, column]
        assert np.array_equal(times, np.arange(times[0], times[0] + R.SEQ_LEN))
        assert times[0] % R.SEQ_LEN == 0


def test_every_slots_fields_come_from_the_same_slot():
    """One sequence is assembled from one set of slots: the action, the value
    and the reset flag of a row are that row's own. A per-field gather that
    drifted by one would still produce a well-shaped batch."""
    buf, batch = _one_batch()
    lane, step = _decode(batch.obs["obs"])
    for row, (lane_i, step_i) in enumerate(zip(lane, step)):
        assert batch.old_logp[row] == pytest.approx(-R.slot_tag(step_i, lane_i))
        assert batch.values[row] == pytest.approx(R.slot_tag(step_i, lane_i) * 2.0)
        assert batch.valid[row] == buf.valid[step_i, lane_i]
    flat_starts = batch.starts.reshape(-1)
    for row, (lane_i, step_i) in enumerate(zip(lane, step)):
        assert flat_starts[row] == buf.starts[step_i, lane_i]


def test_each_sequence_starts_from_the_state_stored_entering_its_first_step():
    """`h0` is the acting policy's hidden state at the sequence's first step,
    not zero and not another sequence's. The fill encodes the chunk and the
    lane into it, so a row taken from the wrong chunk is visible."""
    buf, batch = _one_batch()
    lane, step = _decode(batch.obs["obs"])
    first_rows = {int(column): (int(lane.reshape(R.SEQ_LEN, R.N_SEQ)[0, column]),
                                int(step.reshape(R.SEQ_LEN, R.N_SEQ)[0, column]))
                  for column in range(R.N_SEQ)}
    for column, (lane_i, step_i) in first_rows.items():
        chunk = step_i // R.SEQ_LEN
        assert np.array_equal(batch.h0[R.MEMORY][column],
                              buf.hidden[R.MEMORY][chunk, lane_i])
    assert float(np.abs(batch.h0[R.MEMORY]).max()) > 0.0


def test_advantages_are_normalized_over_valid_slots_only():
    """The flat path selects valid rows before it normalizes. A sequence keeps
    its padding, so the statistics must exclude it explicitly: a padding
    slot's fabricated reward would otherwise move the mean every rollout."""
    buf = R.new_buffer()
    R.fill(buf)
    live = buf.valid > 0.0
    assert not live.all(), "the fill produced no padding, so this is vacuous"
    values = buf.advantages[live]
    expected = (buf.advantages - values.mean()) / (values.std() + 1e-8)
    naive = (buf.advantages - buf.advantages.mean()) / (buf.advantages.std() + 1e-8)
    assert not np.allclose(expected, naive)
    for batch in buf.sequence_minibatches(MINIBATCH, epochs=1):
        lane, step = _decode(batch.obs["obs"])
        for row, (lane_i, step_i) in enumerate(zip(lane, step)):
            assert batch.advantages[row] == pytest.approx(
                expected[step_i, lane_i], abs=1e-5)


def test_an_epoch_covers_the_whole_sequence_set_bar_the_dropped_tail():
    """The set is enumerated from the buffer's own geometry: every
    `(chunk, lane)` there is. A minibatch count that silently visited some
    sequences twice and others never would still yield the right number of
    batches."""
    buf = R.new_buffer()
    R.fill(buf)
    total = buf.n_chunks * buf.n_envs
    assert total == (R.N_STEPS // R.SEQ_LEN) * R.N_ENVS
    seen = []
    batches = list(buf.sequence_minibatches(MINIBATCH, epochs=1))
    assert len(batches) == total // R.N_SEQ
    for batch in batches:
        lane, step = _decode(batch.obs["obs"])
        grid_lane = lane.reshape(R.SEQ_LEN, R.N_SEQ)[0]
        grid_step = step.reshape(R.SEQ_LEN, R.N_SEQ)[0]
        seen += [(int(s) // R.SEQ_LEN, int(l)) for s, l in zip(grid_step, grid_lane)]
    assert len(seen) == len(set(seen))
    missing = {(chunk, lane) for chunk in range(buf.n_chunks)
               for lane in range(R.N_ENVS)} - set(seen)
    assert len(missing) == total % R.N_SEQ
