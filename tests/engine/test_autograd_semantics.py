"""Regression tests for autograd semantics bugs found in review."""

import numpy as np
import pytest

pytest.importorskip("ttnn")  # skip cleanly on CPU-only CI (no ttnn install)

from ttrl.engine import device
from ttrl.engine.autograd import Tensor, no_grad
from ttrl.engine.ops import mul, sum_all

pytestmark = pytest.mark.tt


def test_leaf_requires_grad_survives_no_grad_construction():
    """A leaf built inside no_grad() keeps the requires_grad it was given, once no_grad() exits."""
    with device.acquire() as (dev, _profile):
        with no_grad():
            leaf = Tensor.from_numpy(
                np.ones((2, 2), dtype=np.float32), dev, requires_grad=True
            )
        # no_grad() has exited; the leaf's own flag must reflect the caller's
        # request, not the transient _GRAD_ENABLED state at construction time.
        assert leaf.requires_grad is True

        # And it must actually participate in graph-building now.
        other = Tensor.from_numpy(
            np.full((2, 2), 2.0, dtype=np.float32), dev, requires_grad=True
        )
        out = sum_all(mul(leaf, other))
        out.backward()
        assert leaf.grad is not None


def test_backward_rejects_reused_shared_subgraph():
    """Two different downstream tensors sharing an interior node."""
    with device.acquire() as (dev, _profile):
        a = Tensor.from_numpy(
            np.ones((2, 2), dtype=np.float32), dev, requires_grad=True
        )
        b = Tensor.from_numpy(
            np.full((2, 2), 3.0, dtype=np.float32), dev, requires_grad=True
        )
        shared = mul(a, b)  # interior node used by two downstream losses
        loss1 = sum_all(shared)
        loss2 = sum_all(mul(shared, shared))

        loss1.backward()  # consumes `shared`'s tape (and a's, b's)

        with pytest.raises(RuntimeError):
            loss2.backward()
