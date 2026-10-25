"""`ttrl.engine.carry.ActingCarry`: the two buffers a traced acting step needs."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.engine import acquire, capture
from ttrl.engine.autograd import Tensor, no_grad
from ttrl.engine.carry import ActingCarry
from ttrl.engine.ops import apply_done_mask
from ttrl.engine.recurrent import GRUCell

pytestmark = pytest.mark.tt

LANES = 3
FEATURES = 5
HIDDEN = 7


@pytest.fixture(scope="module")
def dev():
    with acquire(0) as (handle, _profile):
        yield handle


def _rows(rng, lanes=LANES):
    return rng.normal(size=(lanes, FEATURES)).astype(np.float32)


def test_the_carry_advances_only_when_it_is_committed(dev):
    """`commit` is the whole mechanism. Without it the step keeps reading the
    state it started from, so the policy would act from a frozen memory while
    every shape and every metric stayed plausible."""
    rng = np.random.default_rng(3)
    cell = GRUCell(FEATURES, HIDDEN, dev)
    carry = ActingCarry(HIDDEN, dev, lanes=LANES)
    x = Tensor.from_numpy(_rows(rng), dev)

    assert float(np.abs(carry.numpy()).max()) == 0.0, "a fresh carry is zeroed"
    with no_grad():
        first = cell(x, carry.prev)
    carry.bind(first)
    assert float(np.abs(carry.numpy()).max()) == 0.0, (
        "the carry advanced before commit, so the step wrote the buffer it "
        "also reads")
    carry.commit()
    assert np.allclose(carry.numpy(), first.numpy(), atol=1e-6)

    # The second step must start from the first step's output, so its own
    # output differs from the first. A missing commit makes these equal.
    with no_grad():
        second = cell(x, carry.prev)
    carry.bind(second)
    carry.commit()
    assert not np.allclose(second.numpy(), first.numpy(), atol=1e-5), (
        "two steps on the same input produced the same state, so the carry is "
        "not advancing between them")


def test_prev_keeps_its_address_and_a_rebind_is_refused(dev):
    """Address stability, which the acting carry lives or dies on."""
    rng = np.random.default_rng(5)
    cell = GRUCell(FEATURES, HIDDEN, dev)
    carry = ActingCarry(HIDDEN, dev, lanes=LANES)
    address = id(carry.prev.data)
    x = Tensor.from_numpy(_rows(rng), dev)
    for _ in range(3):
        with no_grad():
            carry.bind(cell(x, carry.prev))
        carry.commit()
        assert id(carry.prev.data) == address, (
            "commit rebound prev; a captured trace would still be reading the "
            "buffer at the old address")

    # What the invariant forbids, done deliberately: a name rebound to a fresh
    # tensor instead of written in place.
    carry.prev.data = Tensor.from_numpy(
        np.zeros((LANES, HIDDEN), dtype=np.float32), dev).data
    with pytest.raises(RuntimeError, match="was rebound"):
        carry.commit()


def test_the_start_flag_zeroes_the_state_entering_the_cell(dev):
    """The reset, in the position the unroll also uses."""
    rng = np.random.default_rng(7)
    cell = GRUCell(FEATURES, HIDDEN, dev)
    carry = ActingCarry(HIDDEN, dev, lanes=LANES)
    x = Tensor.from_numpy(_rows(rng), dev)

    # Put something in the carry to be reset.
    with no_grad():
        carry.bind(cell(x, carry.prev))
    carry.commit()
    assert float(np.abs(carry.numpy()).max()) > 0.0

    carry.load_start(np.ones(LANES, dtype=np.float32))
    with no_grad():
        reset_out = cell(x, apply_done_mask(carry.prev, carry.start))
    zero_state = Tensor.from_numpy(np.zeros((LANES, HIDDEN), np.float32), dev)
    with no_grad():
        from_zero = cell(x, zero_state)
    assert np.allclose(reset_out.numpy(), from_zero.numpy(), atol=1e-6), (
        "a step whose start flag is set did not behave like a step from a zero "
        "state, so the flag is not resetting the state entering the cell")
    assert float(np.abs(reset_out.numpy()).max()) > 0.0, (
        "the output of a reset step is identically zero, so the mask was "
        "applied to the state leaving the cell rather than entering it")

    carry.load_start(np.zeros(LANES, dtype=np.float32))
    with no_grad():
        kept = cell(x, apply_done_mask(carry.prev, carry.start))
    assert not np.allclose(kept.numpy(), from_zero.numpy(), atol=1e-5), (
        "a cleared start flag still reset the state, so the mask ignores its "
        "input")


def test_an_acting_step_with_a_carry_captures_and_replays_as_a_trace(dev):
    """Both engine invariants end to end, which is what the two buffers exist
    for. The step reads `prev` and writes its own node, the copy back happens
    outside the captured region, and no host read is dispatched inside it."""
    rng = np.random.default_rng(13)
    cell = GRUCell(FEATURES, HIDDEN, dev)
    carry = ActingCarry(HIDDEN, dev, lanes=LANES)
    obs = Tensor.from_numpy(_rows(rng), dev)
    addresses = (id(carry.prev.data), id(carry.start.data), id(obs.data))

    def step():
        with no_grad():
            return cell(obs, apply_done_mask(carry.prev, carry.start))

    # Warm the program cache: capture records commands, so a first-time kernel
    # compile inside it is not recorded.
    carry.bind(step())
    trace, h_next = capture(dev, step)
    try:
        carry.bind(h_next)
        states = []
        for _ in range(3):
            trace.replay(blocking=True)
            carry.commit()
            states.append(carry.numpy())
    finally:
        trace.close()

    assert (id(carry.prev.data), id(carry.start.data), id(obs.data)) == addresses, (
        "a buffer the trace recorded was rebound between replays")
    for earlier, later in zip(states, states[1:]):
        assert not np.allclose(earlier, later, atol=1e-6), (
            "two consecutive replays left the carry in the same state, so the "
            "commit outside the captured region is not advancing it")

    # The reset still reaches a replayed step, through the bound flag buffer.
    carry.load_start(np.ones(LANES, dtype=np.float32))
    zero_state = Tensor.from_numpy(np.zeros((LANES, HIDDEN), np.float32), dev)
    with no_grad():
        from_zero = cell(obs, zero_state)
    trace, h_next = capture(dev, step)
    try:
        carry.bind(h_next)
        trace.replay(blocking=True)
        carry.commit()
    finally:
        trace.close()
    assert np.allclose(carry.numpy(), from_zero.numpy(), atol=1e-6), (
        "the start flag loaded between replays did not reach the replayed "
        "step, so an episode boundary would not reset a traced acting loop")


