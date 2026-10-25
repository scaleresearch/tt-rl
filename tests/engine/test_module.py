"""`ttrl.engine.nn.Module`'s checkpoint and target-sync plumbing."""

import numpy as np
import pytest

pytest.importorskip("ttnn")

from ttrl.engine import device
from ttrl.engine.layers import Linear
from ttrl.engine.nn import Module

pytestmark = pytest.mark.tt


def test_copy_from_raises_on_parameter_count_mismatch():
    with device.acquire() as (d, _profile):
        small = Linear(4, 4, d, bias=False)
        big = Linear(4, 4, d, bias=True)  # has an extra bias parameter
        with pytest.raises(ValueError, match="parameter count mismatch"):
            small.copy_from(big)


def test_load_state_dict_rejects_missing_or_extra_keys():
    with device.acquire() as (d, _profile):
        layer = Linear(3, 5, d, bias=True)
        state = layer.state_dict()

        missing = dict(state)
        del missing["bias"]
        with pytest.raises(ValueError, match="missing"):
            layer.load_state_dict(missing, d)

        extra = dict(state)
        extra["extra_param"] = np.zeros((1, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="extra"):
            layer.load_state_dict(extra, d)


def test_load_state_dict_rejects_shape_mismatch():
    with device.acquire() as (d, _profile):
        layer = Linear(3, 5, d, bias=True)
        state = layer.state_dict()
        bad = dict(state)
        bad["weight"] = np.zeros((3, 6), dtype=np.float32)  # wrong out_features
        with pytest.raises(ValueError, match="shape mismatch"):
            layer.load_state_dict(bad, d)


def test_load_state_dict_field_reorder_does_not_silently_swap_params():
    """A regression test for the exact bug the positional `p{i}` scheme had."""

    class AB(Module):
        def __init__(self, d):
            self.a = Linear(2, 2, d, bias=False).weight
            self.b = Linear(2, 2, d, bias=False).weight

    with device.acquire() as (d, _profile):
        m1 = AB(d)
        state = m1.state_dict()
        assert set(state.keys()) == {"a", "b"}
        a_val, b_val = m1.a.numpy(), m1.b.numpy()
        assert not np.array_equal(a_val, b_val)  # independent random inits

        m2 = AB(d)
        m2.load_state_dict(state, d)
        np.testing.assert_array_equal(m2.a.numpy(), a_val)
        np.testing.assert_array_equal(m2.b.numpy(), b_val)
