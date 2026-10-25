"""Numerical correctness and address-stability tests for ttrl.engine.optim.Adam."""

import numpy as np
import pytest

from ttrl.engine.optim import Adam

pytest.importorskip("ttnn")  # skip cleanly on CPU-only CI (no ttnn install)
import ttnn

from ttrl.engine import device
from ttrl.engine.autograd import Tensor
from ttrl.engine.tensor import to_host

pytestmark = pytest.mark.tt

ATOL = 2e-2
RTOL = 2e-2

# ttrl.engine.optim.Adam now composes primitive elementwise ops (ttnn.add/mul/sub/sqrt/div)
# instead of the abandoned ttnn.moreh_adam kernel (see ttrl/engine/optim/adam.py's module
# docstring), so it operates on fp32 master
# weights and moments: the master dtype is fp32.
ADAM_DTYPE = ttnn.float32


class _NumpyAdam:
    """Reference Adam, unfused, matching torch.optim.Adam semantics."""

    def __init__(self, shape, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, seed=0):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)

    def step(self, p, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * (g * g)
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        return p - self.lr * mhat / (np.sqrt(vhat) + self.eps)


def test_adam_matches_numpy_reference_and_updates_in_place():
    shape = (32, 32)
    rng = np.random.default_rng(0)
    p0 = rng.uniform(-1, 1, size=shape).astype(np.float32)
    grads = [rng.uniform(-0.1, 0.1, size=shape).astype(np.float32) for _ in range(5)]

    ref = _NumpyAdam(shape, lr=1e-3)
    p_ref = p0.copy()

    with device.acquire() as (dev, _profile):
        p = Tensor.from_numpy(p0, dev, requires_grad=True, dtype=ADAM_DTYPE)
        opt = Adam([p], lr=1e-3)
        data_obj, m_obj, v_obj = p.data, opt.m[0], opt.v[0]

        for g_np in grads:
            p.grad = Tensor.from_numpy(g_np, dev, dtype=ADAM_DTYPE).data
            opt.step()
            p_ref = ref.step(p_ref, g_np)
            assert p.data is data_obj and opt.m[0] is m_obj and opt.v[0] is v_obj, (
                "an Adam buffer was rebound instead of updated in place"
            )

            got = to_host(p.data)
            np.testing.assert_allclose(got, p_ref, atol=ATOL, rtol=RTOL)


def test_state_dict_is_canonical_m_and_v_and_round_trips_in_place():
    """The on-disk format is canonical Adam `m` and `v`."""

    shape = (32, 64)
    rng = np.random.default_rng(11)
    p0 = rng.uniform(-1, 1, size=shape).astype(np.float32)
    grads = [rng.uniform(-0.1, 0.1, size=shape).astype(np.float32) for _ in range(5)]

    with device.acquire() as (dev, _profile):
        p = Tensor.from_numpy(p0, dev, requires_grad=True, dtype=ADAM_DTYPE)
        opt = Adam([p], lr=1e-3)
        for g_np in grads:
            p.grad = Tensor.from_numpy(g_np, dev, dtype=ADAM_DTYPE).data
            opt.step()
        saved = opt.state_dict()
        assert saved["t"] == len(grads)

        # The canonical m is the exponential moving average of the gradients,
        # computed here from the same sequence with no reference to how the
        # device composed it.
        m_ref = np.zeros(shape, dtype=np.float64)
        v_ref = np.zeros(shape, dtype=np.float64)
        for g_np in grads:
            m_ref = 0.9 * m_ref + 0.1 * g_np
            v_ref = 0.999 * v_ref + 0.001 * np.square(g_np, dtype=np.float64)
        np.testing.assert_allclose(saved["m"][0], m_ref, rtol=2e-3, atol=1e-7)
        np.testing.assert_allclose(saved["v"][0], v_ref, rtol=2e-3, atol=1e-10)

        # Round-trip into a fresh optimizer, in place.
        q = Tensor.from_numpy(p0, dev, requires_grad=True, dtype=ADAM_DTYPE)
        fresh = Adam([q], lr=1e-3)
        before = [id(buf) for buf in list(fresh.m) + list(fresh.v)]
        fresh.load_state_dict(saved, dev)
        assert [id(buf) for buf in list(fresh.m) + list(fresh.v)] == before, (
            "load_state_dict rebound a moment buffer; a captured trace replays "
            "against the old address"
        )
        restored = fresh.state_dict()

    assert restored["t"] == saved["t"]
    assert restored["steps"] == saved["steps"]
    for name in ("m", "v"):
        np.testing.assert_allclose(
            restored[name][0], saved[name][0], rtol=2e-4, atol=1e-10
        )
