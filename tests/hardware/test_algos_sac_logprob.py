"""The tanh-squash log-prob across the whole action range, against torch."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests._fakes.torch_twin import t64
from ttrl.algos.gaussian import (
    squashed_logp,
    squashed_logp_numpy,
)
from ttrl.engine import Tensor, acquire, backward
from ttrl.engine.ops import sum_all
from ttrl.engine.tensor import to_host

pytestmark = pytest.mark.tt

#: Includes the bounds: tanh(20) differs from 1.0 by less than fp32 can hold,
#: so these rows ARE the +/-1 corners of the action cube.
PRE_SQUASH = np.array(
    [-20.0, -12.0, -8.0, -5.0, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 5.0,
     8.0, 12.0, 20.0], dtype=np.float32)


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


def _torch_squashed_logp(u, mean, log_std):
    normal = torch.distributions.Normal(mean, log_std.exp())
    transform = torch.distributions.TanhTransform()
    return (normal.log_prob(u).sum(-1)
            - transform.log_abs_det_jacobian(u, torch.tanh(u)).sum(-1))


def test_squashed_logp_matches_torch_over_the_action_range(device):
    torch = pytest.importorskip("torch")

    rows = PRE_SQUASH.shape[0]
    rng = np.random.default_rng(0)
    u = np.stack([PRE_SQUASH, PRE_SQUASH[::-1].copy()], axis=-1)
    mean = rng.normal(size=(rows, 2)).astype(np.float32)
    log_std = rng.uniform(-2.0, 1.0, size=(rows, 2)).astype(np.float32)

    got = squashed_logp(
        Tensor.from_numpy(u, device),
        Tensor.from_numpy(mean, device),
        Tensor.from_numpy(log_std, device),
    ).numpy().reshape(-1)
    want = _torch_squashed_logp(
        torch.tensor(u, dtype=torch.float64),
        torch.tensor(mean, dtype=torch.float64),
        torch.tensor(log_std, dtype=torch.float64),
    ).numpy()
    host = squashed_logp_numpy(u, mean, log_std)

    relative = np.abs(got - want) / np.maximum(np.abs(want), 1e-12)
    assert np.isfinite(got).all()
    assert relative.max() < 1e-4
    # The host spelling records the log-prob an importance ratio is later
    # measured against, so it must agree with the device one it is compared to.
    assert np.abs(host - want).max() < 1e-3


def test_squashed_logp_gradient_matches_torch(device):
    torch = pytest.importorskip("torch")

    rows = PRE_SQUASH.shape[0]
    rng = np.random.default_rng(1)
    u = np.stack([PRE_SQUASH, PRE_SQUASH[::-1].copy()], axis=-1)
    mean_np = rng.normal(size=(rows, 2)).astype(np.float32)
    log_std_np = rng.uniform(-4.0, 1.0, size=(rows, 2)).astype(np.float32)

    mean = Tensor.from_numpy(mean_np, device, requires_grad=True)
    log_std = Tensor.from_numpy(log_std_np, device, requires_grad=True)
    backward(sum_all(squashed_logp(Tensor.from_numpy(u, device), mean, log_std)))

    tmean, tlog_std = t64(mean_np), t64(log_std_np)
    _torch_squashed_logp(torch.tensor(u, dtype=torch.float64), tmean,
                         tlog_std).sum().backward()

    for name, param, reference in (("mean", mean, tmean),
                                   ("log_std", log_std, tlog_std)):
        got = np.asarray(to_host(param.grad), np.float64).reshape(reference.shape)
        want = reference.grad.detach().numpy()
        relative = np.linalg.norm(got - want) / np.linalg.norm(want)
        pcc = float(np.corrcoef(got.reshape(-1), want.reshape(-1))[0, 1])
        assert pcc > 0.999999
        assert relative < 1e-4


