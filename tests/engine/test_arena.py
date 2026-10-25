"""Address-stability and correctness tests for ttrl.engine.tensor.StaticArena."""

import numpy as np
import pytest

from ttrl.engine.layers import Linear
from ttrl.engine.tensor import StaticArena

pytest.importorskip("ttnn")  # skip cleanly on CPU-only CI (no ttnn install)

from ttrl.engine import device
from ttrl.engine.autograd import Tensor
from ttrl.engine.tensor import to_host

pytestmark = pytest.mark.tt

ATOL = 1e-5
RTOL = 1e-5


def test_copy_into_updates_values_and_preserves_identity():
    """Buffer identity (the same Python `ttnn.Tensor` object) must survive `copy_into`."""

    shape = (32, 32)
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, size=shape).astype(np.float32)
    b = rng.uniform(-1, 1, size=shape).astype(np.float32)

    with device.acquire() as (dev, _profile):
        arena = StaticArena(dev)
        src_a = Tensor.from_numpy(a, dev)
        src_b = Tensor.from_numpy(b, dev)
        buf = arena.allocate("target", src_a)

        buf_id_before = id(arena["target"])

        arena.copy_into("target", src_a)
        np.testing.assert_allclose(to_host(arena["target"]), a, atol=ATOL, rtol=RTOL)
        assert id(arena["target"]) == buf_id_before

        arena.copy_into("target", src_b)
        np.testing.assert_allclose(to_host(arena["target"]), b, atol=ATOL, rtol=RTOL)
        assert id(arena["target"]) == buf_id_before
        assert arena["target"] is buf


def test_copy_from_does_not_rebind_dst_data():
    """`Module.copy_from` (target-network sync) must update `dst.data`'s buffer in place."""

    with device.acquire() as (dev, _profile):
        online = Linear(32, 32, dev)
        target = Linear(32, 32, dev)

        target_param_data_ids_before = [id(p.data) for p in target.parameters()]

        target.copy_from(online)

        target_param_data_ids_after = [id(p.data) for p in target.parameters()]
        assert target_param_data_ids_before == target_param_data_ids_after

        for dst, src in zip(target.parameters(), online.parameters()):
            np.testing.assert_allclose(to_host(dst.data), to_host(src.data), atol=ATOL, rtol=RTOL)


