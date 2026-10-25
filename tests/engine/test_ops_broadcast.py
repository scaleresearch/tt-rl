"""Gradcheck for broadcast-reduction in add/sub/mul/div backward."""

import numpy as np
import pytest

from ttrl.engine.layers import Linear
from ttrl.engine.ops import sum_all
from ttrl.engine.tensor import to_host

pytest.importorskip("ttnn")  # skip cleanly on CPU-only CI (no ttnn install)

from ttrl.engine import device
from ttrl.engine.autograd import Tensor

pytestmark = pytest.mark.tt

B, N = 4, 8
ATOL = 2e-2  # fp32 path, but TT-NN reductions carry some kernel-level slop


def _run(op_np, op_tt, device_, seed):
    rng = np.random.default_rng(seed)
    a_np = rng.uniform(0.5, 2.0, size=(B, N)).astype(np.float32)
    b_np = rng.uniform(0.5, 2.0, size=(1, N)).astype(np.float32)

    a = Tensor.from_numpy(a_np, device_, requires_grad=True)
    b = Tensor.from_numpy(b_np, device_, requires_grad=True)
    out = op_tt(a, b)
    out.backward()

    grad_a = a.grad
    grad_b = b.grad
    assert grad_a is not None
    assert grad_b is not None

    grad_a_host = to_host(grad_a)
    grad_b_host = to_host(grad_b)

    assert grad_a_host.shape == (B, N), grad_a_host.shape
    assert grad_b_host.shape == (1, N), grad_b_host.shape

    # Reference: numpy autograd via finite structure — since loss = sum(out),
    # d(out)/d(a) and d(out)/d(b) follow standard broadcast rules.
    upstream = np.ones((B, N), dtype=np.float32)
    ref_a, ref_b = op_np(a_np, b_np, upstream)

    np.testing.assert_allclose(grad_a_host, ref_a, atol=ATOL, rtol=ATOL)
    np.testing.assert_allclose(grad_b_host, ref_b, atol=ATOL, rtol=ATOL)


def _sum(t):
    return sum_all(t)


def test_sum_all_grad():
    """sum_all's backward broadcasts on device; a host read of out.grad is illegal inside trace capture."""

    with device.acquire() as (dev, _profile):
        a_np = np.random.default_rng(5).uniform(-2, 2, size=(B, N)).astype(np.float32)
        a = Tensor.from_numpy(a_np, dev, requires_grad=True)
        loss = sum_all(a)
        loss.backward()

        assert a.grad is not None
        grad_a = to_host(a.grad)
        assert grad_a.shape == (B, N), grad_a.shape
        np.testing.assert_allclose(grad_a, np.ones((B, N), dtype=np.float32), atol=ATOL, rtol=ATOL)

        loss_val = to_host(loss.data).reshape(-1)[0]
        np.testing.assert_allclose(loss_val, a_np.sum(), atol=1e-1, rtol=1e-2)


def test_linear_bias_grad_shape_and_value():
    """End-to-end: Linear.forward's add(out, bias) must not leak batch dim."""

    with device.acquire() as (dev, _profile):
        lin = Linear(3, 5, dev, bias=True)
        x_np = np.random.default_rng(4).uniform(-1, 1, size=(6, 3)).astype(np.float32)
        x = Tensor.from_numpy(x_np, dev, requires_grad=True)
        out = lin.forward(x)
        loss = _sum(out)
        loss.backward()

        assert lin.bias.grad is not None
        bias_grad = to_host(lin.bias.grad)
        assert bias_grad.shape == (1, 5), bias_grad.shape

        w_np = to_host(lin.weight.data)
        expected_bias_grad = np.ones((6, 5), dtype=np.float32).sum(axis=0, keepdims=True)
        np.testing.assert_allclose(bias_grad, expected_bias_grad, atol=ATOL, rtol=ATOL)
