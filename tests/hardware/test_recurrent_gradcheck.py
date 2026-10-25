"""BPTT parity against torch for `ttrl.engine.recurrent`."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.engine import acquire, backward
from ttrl.engine.autograd import Tensor
from ttrl.engine.ops import add, mean_all, square
from ttrl.engine.recurrent import GRU
from ttrl.engine.tensor import to_host

pytestmark = pytest.mark.tt

#: Same gate as the vector-policy gradcheck, for the same reasons.
GRAD_PCC_FLOOR = 0.999
GRAD_FROBENIUS_CEILING = 5e-2

#: The depths the unrolled gradcheck runs. See WHY THESE T above.
UNROLL_DEPTHS = (2, 16)
BATCH = 16
FEATURES = 12
HIDDEN = 24


@pytest.fixture(scope="module")
def dev():
    with acquire(0) as (handle, _profile):
        yield handle


def _assert_matches(name, param, reference, *, transpose: bool = False):
    """`transpose` for the two weights, whose device layout is the transpose of
    torch's. Reshaping instead of transposing silently compares a (12, 72)
    gradient against a (72, 12) one element by element and reports a near-zero
    correlation, which looks like a broken backward."""
    want = reference.grad.detach().numpy()
    if transpose:
        want = want.T
    got = np.asarray(to_host(param.grad), dtype=np.float64).reshape(want.shape)
    frobenius = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30)
    pcc = np.corrcoef(got.reshape(-1), want.reshape(-1))[0, 1]
    assert pcc > GRAD_PCC_FLOOR, f"{name}: pcc {pcc}"
    assert frobenius < GRAD_FROBENIUS_CEILING, f"{name}: frobenius {frobenius}"


def _torch_cell(torch, cell):
    """A float64 `torch.nn.GRUCell` holding this cell's weights."""
    reference = torch.nn.GRUCell(cell.in_features, cell.hidden_size,
                                 bias=True).double()
    with torch.no_grad():
        # Stored transposed relative to torch's `weight_ih`, matching
        # `ttrl.engine.layers.Linear`.
        reference.weight_ih.copy_(torch.tensor(cell.weight_ih.numpy().T,
                                               dtype=torch.float64))
        reference.weight_hh.copy_(torch.tensor(cell.weight_hh.numpy().T,
                                               dtype=torch.float64))
        reference.bias_ih.copy_(torch.tensor(cell.bias_ih.numpy().reshape(-1),
                                             dtype=torch.float64))
        reference.bias_hh.copy_(torch.tensor(cell.bias_hh.numpy().reshape(-1),
                                             dtype=torch.float64))
    for param in reference.parameters():
        param.requires_grad_(True)
    return reference


def _pairs(cell, reference):
    """Every parameter of the cell, paired with torch's. The four are the whole
    set `GRUCell` registers, which `test_the_cell_registers_exactly_these`
    pins."""
    return [
        ("weight_ih", cell.weight_ih, reference.weight_ih, True),
        ("weight_hh", cell.weight_hh, reference.weight_hh, True),
        ("bias_ih", cell.bias_ih, reference.bias_ih, False),
        ("bias_hh", cell.bias_hh, reference.bias_hh, False),
    ]


@pytest.mark.parametrize("steps", UNROLL_DEPTHS)
def test_unrolled_gradient_accumulates_over_timesteps_like_torch(steps, dev):
    """The gate. Each weight gradient is a sum of `steps` contributions and
    `h0.grad` is the whole chain run back to the start of the sequence.
    """
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(11)

    net = GRU(FEATURES, HIDDEN, dev)
    reference = _torch_cell(torch, net.cell)
    seq_np = rng.normal(size=(steps, BATCH, FEATURES)).astype(np.float32)
    h0_np = rng.normal(size=(BATCH, HIDDEN)).astype(np.float32)

    seq = Tensor.from_numpy(seq_np, dev, requires_grad=True)
    h0 = Tensor.from_numpy(h0_np, dev, requires_grad=True)
    net.zero_grad()
    outputs, final = net(seq, h0)
    assert tuple(outputs.shape) == (steps, BATCH, HIDDEN)
    assert tuple(final.shape) == (BATCH, HIDDEN)

    torch_seq = torch.tensor(seq_np, dtype=torch.float64, requires_grad=True)
    torch_h0 = torch.tensor(h0_np, dtype=torch.float64, requires_grad=True)
    torch_h = torch_h0
    torch_outputs = []
    for step in range(steps):
        torch_h = reference(torch_seq[step], torch_h)
        torch_outputs.append(torch_h)
    torch_stacked = torch.stack(torch_outputs, dim=0)

    np.testing.assert_allclose(outputs.numpy(), torch_stacked.detach().numpy(),
                               atol=5e-2, rtol=5e-2)

    # Both halves of the output, so gradient reaches the last state through
    # two paths and the accumulation at the final step is exercised too.
    loss = add(mean_all(square(outputs)), mean_all(square(final)))
    backward(loss)
    torch_loss = (torch_stacked ** 2).mean() + (torch_h ** 2).mean()
    torch_loss.backward()

    for name, param, torch_param, transpose in _pairs(net.cell, reference):
        _assert_matches(name, param, torch_param, transpose=transpose)
    _assert_matches("h0", h0, torch_h0)
    _assert_matches("seq", seq, torch_seq)
