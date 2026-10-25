"""Building a conv layer on a chip must never reach for torch."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import ttnn

from ttrl.engine.autograd import Tensor
from ttrl.engine.conv import Conv2d
from ttrl.engine.device import acquire
from ttrl.engine.layers import Flatten, Linear, ReLU, Sequential
from ttrl.engine.ops import sum_all
from ttrl.engine.tensor import to_host

pytestmark = pytest.mark.tt


class _RefuseTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError(f"{name} was imported")
        return None


@pytest.fixture
def torch_refused():
    """torch gone from sys.modules and refused on import, then put back."""
    saved = {k: v for k, v in sys.modules.items()
             if k == "torch" or k.startswith("torch.")}
    for name in saved:
        del sys.modules[name]
    hook = _RefuseTorch()
    sys.meta_path.insert(0, hook)
    try:
        yield hook
    finally:
        sys.meta_path.remove(hook)
        sys.modules.update(saved)


def test_a_conv_tower_builds_and_trains_with_torch_refused(torch_refused):
    n = 8
    with acquire(0) as (dev, _profile):
        c1 = Conv2d((4, 84, 84), 32, 8, dev, batch=n, stride=4, needs_input_grad=False)
        c2 = Conv2d(c1.out_shape, 64, 4, dev, batch=n, stride=2)
        c3 = Conv2d(c2.out_shape, 64, 3, dev, batch=n, stride=1)
        flat = Flatten(*c3.out_shape[1:], c3.out_shape[0])
        net = Sequential(c1, ReLU(), c2, ReLU(), c3, ReLU(), flat,
                         Linear(flat.out_features, 6, dev))
        # The layouts the solve settled on, which are the numbers the deleted
        # torch read used to produce.
        assert [c.plan.pspec.layout.cp for c in (c1, c2, c3)] == [8, 32, 64]
        assert [c.plan.pspec.layout.shape for c in (c1, c2, c3)] == [
            (1, 1, 512, 32), (1, 1, 512, 64), (1, 1, 576, 64)
        ]
        x = ttnn.Tensor(
            np.random.default_rng(0).standard_normal(
                (1, 1, n * 84 * 84, 4)).astype(np.float32),
            ttnn.float32,
        ).to(ttnn.ROW_MAJOR_LAYOUT).to(dev)
        y = net(Tensor(x, requires_grad=False))
        sum_all(y).backward()
        assert to_host(y.data).shape == (n, 6)
        # Checkpointing crosses the same boundary: it unpacks the prepared
        # buffer to OIHW on host.
        state = net.state_dict()
        assert state["layers.0.weight"].shape == (32, 4, 8, 8)
        net.load_state_dict(state)
    assert "torch" not in sys.modules


def test_the_hook_really_refuses_torch(torch_refused):
    """The control. Without it the test above passes vacuously."""
    with pytest.raises(AssertionError, match="torch was imported"):
        import torch  # noqa: F401
