"""Engine contract for `ttrl.engine.recurrent`: the reset, the trace, the
refusals.
"""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.engine import TrainStep, acquire, backward
from ttrl.engine.autograd import Tensor
from ttrl.engine.ops import mean_all, split, square
from ttrl.engine.recurrent import GRU
from ttrl.engine.tensor import to_host

pytestmark = pytest.mark.tt

TIMESTEPS = 8
BATCH = 16
FEATURES = 12
HIDDEN = 24


@pytest.fixture(scope="module")
def dev():
    with acquire(0) as (handle, _profile):
        yield handle


def test_a_done_boundary_cuts_the_gradient_path_through_it(dev):
    """`apply_done_mask` zeroes the state entering a timestep, so nothing
    before a boundary can influence anything after it. Checked as a gradient,
    which is the property; a forward check would pass on a mask applied one
    timestep off."""
    rng = np.random.default_rng(13)
    net = GRU(FEATURES, HIDDEN, dev)
    seq_np = rng.normal(size=(TIMESTEPS, BATCH, FEATURES)).astype(np.float32)
    done_np = np.zeros((TIMESTEPS, BATCH, 1), dtype=np.float32)
    boundary = 5
    done_np[boundary] = 1.0

    seq = Tensor.from_numpy(seq_np, dev, requires_grad=True)
    h0 = Tensor.from_numpy(rng.normal(size=(BATCH, HIDDEN)).astype(np.float32),
                           dev, requires_grad=True)
    net.zero_grad()
    outputs, _final = net(seq, h0, Tensor.from_numpy(done_np, dev))
    # Only the post-boundary outputs, so any gradient reaching an earlier
    # input had to cross the reset.
    after = split(outputs, (boundary, TIMESTEPS - boundary), dim=0)[1]
    backward(mean_all(square(after)))

    seq_grad = np.asarray(to_host(seq.grad), dtype=np.float64)
    assert np.abs(seq_grad[:boundary]).max() == 0.0, (
        "gradient crossed a done boundary into an earlier timestep")
    assert np.abs(np.asarray(to_host(h0.grad), dtype=np.float64)).max() == 0.0, (
        "gradient reached h0 through a reset that should have zeroed it")

    # The boundary step itself. A done at t means the episode ended before t,
    # so step t starts from a zeroed state and still consumes x[t] and still
    # emits an output. Masking the state coming OUT of the cell instead would
    # zero this output and cut x[t] out of the graph entirely, while leaving
    # every assertion above true, so these two are what tell the two
    # placements apart.
    assert np.abs(outputs.numpy()[boundary]).max() > 0.0, (
        "the output at the boundary step is identically zero; the reset was "
        "applied to the state leaving the cell rather than entering it")
    assert np.abs(seq_grad[boundary]).max() > 0.0, (
        "no gradient reached the boundary step's own input, so the reset was "
        "applied after the cell rather than before it")
    assert np.abs(seq_grad[boundary + 1:]).max() > 0.0, (
        "no gradient reached the timesteps after the boundary either, so the "
        "zero checks above are vacuous")


def test_the_unrolled_step_captures_as_a_trace(dev):
    """Both engine invariants, end to end. T is fixed, `h0` is an ordinary
    bound input written in place, every timestep's state is its own node, and
    the unroll reads nothing to host, so `TrainStep` can capture it."""

    rng = np.random.default_rng(17)
    net = GRU(FEATURES, HIDDEN, dev)
    inputs = {
        "seq": Tensor.from_numpy(
            rng.normal(size=(TIMESTEPS, BATCH, FEATURES)).astype(np.float32), dev),
        "h0": Tensor.from_numpy(np.zeros((BATCH, HIDDEN), dtype=np.float32), dev),
    }
    addresses = {name: id(tensor.data) for name, tensor in inputs.items()}

    def step_fn(bound):
        net.zero_grad()
        outputs, _final = net(bound["seq"], bound["h0"])
        loss = mean_all(square(outputs))
        backward(loss)
        return {"loss": loss}

    with TrainStep(step_fn, inputs, dev, mode="traced_fwd_bwd") as step:
        first = step.run()
        step.load_input("seq", np.zeros((TIMESTEPS, BATCH, FEATURES),
                                        dtype=np.float32))
        second = step.run()

    assert first["loss"] > 0.0
    # A zero sequence still drives the gates from the biases, so the loss
    # changes rather than vanishing; what matters is that the replay read the
    # new contents of the same buffer.
    assert second["loss"] != first["loss"]
    for name, tensor in inputs.items():
        assert id(tensor.data) == addresses[name], (
            f"{name} was rebound; a captured trace replays against the address "
            "its buffer had at capture time")


