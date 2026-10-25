"""Gradchecks for ttrl/engine/ops/ against numpy, on real hardware.

Each differentiable function gets: forward-value check against a numpy
reference, and a backward check against numpy's *analytic* gradient (these
functions are all closed-form, so an analytic numpy reference is exact and
cheaper/more reliable on-device than finite differences, which would need
many extra forward passes through the chip). Style follows
tests/engine/test_ops_broadcast.py.
"""

import numpy as np
import pytest

from ttrl.engine.autograd import Tensor as T
from ttrl.engine.ops import argmax_dim_detached

pytest.importorskip("ttnn")  # skip cleanly on CPU-only CI (no ttnn install)

from ttrl.engine import device
from ttrl.engine.autograd import Tensor
from ttrl.engine.ops import mean_all, sum_all
from ttrl.engine.tensor import to_host
from ttrl.engine import ops as F

pytestmark = pytest.mark.tt

ATOL = 3e-2
RTOL = 3e-2


def _np_softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _np_log_softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    shifted = x - m
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


# --------------------------------------------------------------------------
# clip
# --------------------------------------------------------------------------

def test_clip_forward_and_grad():
    with device.acquire() as (dev, _profile):
        rng = np.random.default_rng(0)
        x_np = rng.uniform(-3, 3, size=(4, 8)).astype(np.float32)
        x = Tensor.from_numpy(x_np, dev, requires_grad=True)

        out = F.clamp(x, -1.0, 1.5)
        out_np = to_host(out.data)
        ref = np.clip(x_np, -1.0, 1.5)
        np.testing.assert_allclose(out_np, ref, atol=ATOL, rtol=RTOL)

        loss = sum_all(out)
        loss.backward()
        grad = to_host(x.grad)
        ref_grad = ((x_np > -1.0) & (x_np < 1.5)).astype(np.float32)
        np.testing.assert_allclose(grad, ref_grad, atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------
# huber
# --------------------------------------------------------------------------

def test_huber_forward_and_grad():
    with device.acquire() as (dev, _profile):
        rng = np.random.default_rng(1)
        pred_np = rng.uniform(-3, 3, size=(4, 8)).astype(np.float32)
        target_np = rng.uniform(-3, 3, size=(4, 8)).astype(np.float32)
        delta = 1.0

        pred = Tensor.from_numpy(pred_np, dev, requires_grad=True)
        target = Tensor.from_numpy(target_np, dev, requires_grad=False)

        out = F.huber(pred, target, delta=delta)
        out_np = to_host(out.data)

        diff = pred_np - target_np
        absdiff = np.abs(diff)
        ref = np.where(absdiff <= delta, 0.5 * diff ** 2, delta * (absdiff - 0.5 * delta))
        np.testing.assert_allclose(out_np, ref, atol=ATOL, rtol=RTOL)

        loss = sum_all(out)
        loss.backward()
        grad = to_host(pred.grad)
        ref_grad = np.where(absdiff <= delta, diff, delta * np.sign(diff))
        np.testing.assert_allclose(grad, ref_grad, atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------
# one_hot_gather
# --------------------------------------------------------------------------

def test_one_hot_gather_forward_and_grad():
    with device.acquire() as (dev, _profile):
        rng = np.random.default_rng(2)
        B, C = 6, 5
        values_np = rng.uniform(-2, 2, size=(B, C)).astype(np.float32)
        idx_np = rng.integers(0, C, size=(B,))

        values = Tensor.from_numpy(values_np, dev, requires_grad=True)
        gathered = F.one_hot_gather(values, idx_np, num_classes=C)

        out_np = to_host(gathered.data).reshape(-1)
        ref = values_np[np.arange(B), idx_np]
        np.testing.assert_allclose(out_np, ref, atol=ATOL, rtol=RTOL)

        loss = sum_all(gathered)
        loss.backward()
        grad = to_host(values.grad)
        ref_grad = np.zeros((B, C), dtype=np.float32)
        ref_grad[np.arange(B), idx_np] = 1.0
        np.testing.assert_allclose(grad, ref_grad, atol=ATOL, rtol=RTOL)


def test_one_hot_gather_device_indices_match_host_indices():
    """The on-device index path must be indistinguishable from the host one.

    This is the gate for `ttrl/algos/dqn/losses.py`'s Double-DQN action selection
    moving on device (`_argmax_host` was recorded as a
    hard trace blocker): the replacement is only safe if it gathers the exact
    same elements, so equality is asserted exactly -- neither path does
    arithmetic on the gathered values, only masking.
    """

    with device.acquire() as (dev, _profile):
        rng = np.random.default_rng(11)
        B, C = 7, 5
        values_np = rng.uniform(-2, 2, size=(B, C)).astype(np.float32)
        # Distinct per-row scores so argmax has no ties to break.
        scores_np = rng.permutation(B * C).reshape(B, C).astype(np.float32)

        scores = Tensor.from_numpy(scores_np, dev, requires_grad=False)
        idx_dev = argmax_dim_detached(scores, dim=-1)
        np.testing.assert_array_equal(
            to_host(idx_dev.data).reshape(-1).astype(np.int64),
            scores_np.argmax(axis=-1),
        )

        values_a = Tensor.from_numpy(values_np, dev, requires_grad=True)
        values_b = Tensor.from_numpy(values_np, dev, requires_grad=True)
        from_device = F.one_hot_gather(values_a, idx_dev, num_classes=C)
        from_host = F.one_hot_gather(values_b, scores_np.argmax(axis=-1), num_classes=C)

        np.testing.assert_array_equal(
            to_host(from_device.data), to_host(from_host.data)
        )

        # ...and the gradients they route back must match too, since that is
        # what actually reaches the optimizer.
        sum_all(from_device).backward()
        sum_all(from_host).backward()
        np.testing.assert_array_equal(to_host(values_a.grad), to_host(values_b.grad))


def test_one_hot_device_is_a_valid_one_hot():
    """Every row exactly one 1.0, in the right column, no host read involved."""

    with device.acquire() as (dev, _profile):
        B, C = 5, 6
        idx_np = np.array([0, 5, 3, 3, 1], dtype=np.float32).reshape(B, 1)
        idx = T.from_numpy(idx_np, dev, requires_grad=False)
        mask = to_host(F.one_hot_device(idx, C).data).reshape(B, C)

        ref = np.zeros((B, C), dtype=np.float32)
        ref[np.arange(B), idx_np.reshape(-1).astype(int)] = 1.0
        np.testing.assert_array_equal(mask, ref)


# --------------------------------------------------------------------------
# log_softmax
# --------------------------------------------------------------------------

def test_log_softmax_forward_and_grad():
    with device.acquire() as (dev, _profile):
        rng = np.random.default_rng(3)
        B, C = 4, 6
        x_np = rng.uniform(-5, 5, size=(B, C)).astype(np.float32)
        x = Tensor.from_numpy(x_np, dev, requires_grad=True)

        out = F.log_softmax(x, dim=-1)
        out_np = to_host(out.data)
        ref = _np_log_softmax(x_np, axis=-1)
        np.testing.assert_allclose(out_np, ref, atol=ATOL, rtol=RTOL)

        loss = sum_all(out)
        loss.backward()
        grad = to_host(x.grad)
        # d/dx sum_c logsoftmax(x)_c = 1 - C * softmax(x), row-wise (upstream=1 everywhere).
        p = _np_softmax(x_np, axis=-1)
        ref_grad = 1.0 - C * p
        np.testing.assert_allclose(grad, ref_grad, atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------
# entropy
# --------------------------------------------------------------------------

def test_entropy_forward_and_grad():
    with device.acquire() as (dev, _profile):
        rng = np.random.default_rng(4)
        B, C = 4, 6
        x_np = rng.uniform(-3, 3, size=(B, C)).astype(np.float32)
        x = Tensor.from_numpy(x_np, dev, requires_grad=True)

        out = F.entropy(x, dim=-1)
        out_np = to_host(out.data).reshape(-1)

        p = _np_softmax(x_np, axis=-1)
        logp = _np_log_softmax(x_np, axis=-1)
        ref = -(p * logp).sum(axis=-1)
        np.testing.assert_allclose(out_np, ref, atol=ATOL, rtol=RTOL)

        loss = mean_all(out)
        loss.backward()
        grad = to_host(x.grad)

        # Analytic d(entropy_row)/dx_j = -p_j * (logp_j + H_row), from
        # d(softmax)_c/dx_j = p_c(delta_cj - p_j) and d(logsoftmax)_c/dx_j =
        # delta_cj - p_j; mean_all divides by B*1.
        row_entropy = ref
        ref_grad = -p * (logp + row_entropy[:, None]) / B
        np.testing.assert_allclose(grad, ref_grad, atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------
# masks / apply_done_mask
# --------------------------------------------------------------------------

def test_apply_done_mask():
    with device.acquire() as (dev, _profile):
        rng = np.random.default_rng(5)
        value_np = rng.uniform(-2, 2, size=(4, 1)).astype(np.float32)
        done_np = rng.integers(0, 2, size=(4, 1)).astype(np.float32)

        value = Tensor.from_numpy(value_np, dev, requires_grad=True)
        done = Tensor.from_numpy(done_np, dev, requires_grad=False)

        out = F.apply_done_mask(value, done)
        out_np = to_host(out.data)
        ref = value_np * (1.0 - done_np)
        np.testing.assert_allclose(out_np, ref, atol=ATOL, rtol=RTOL)

        loss = sum_all(out)
        loss.backward()
        grad = to_host(value.grad)
        ref_grad = 1.0 - done_np
        np.testing.assert_allclose(grad, ref_grad, atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------
# non_finite_check
# --------------------------------------------------------------------------

def test_non_finite_check():
    with device.acquire() as (dev, _profile):
        clean = Tensor.from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), dev
        )
        assert F.non_finite_check(clean) is False

        bad_np = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        bad = Tensor.from_numpy(bad_np, dev)
        assert F.non_finite_check(bad) is True
