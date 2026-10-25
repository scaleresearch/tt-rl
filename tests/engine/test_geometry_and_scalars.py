"""Host-only tests for the pure arithmetic behind conv geometry and Adam."""

import math

import pytest

from ttrl.engine.conv import conv_out_size
from ttrl.engine.conv.geometry import (
    ConvDoesNotFit,
    ConvGeometry,
    check_dx_capacity,
)
from ttrl.engine.optim import adam_fused_scalars


@pytest.mark.parametrize(
    "size, kernel, stride, padding, expected",
    [
        (84, 8, 4, 0, 20),   # Nature-CNN conv1
        (5, 3, 1, 1, 5),     # "same" padding
    ],
)
def test_conv_out_size_matches_the_standard_formula(size, kernel, stride, padding, expected):
    assert conv_out_size(size, kernel, stride, padding, axis="h") == expected


def test_conv_out_size_rejects_ragged_strides():
    """A ragged stride leaves an input row that feeds no output, so dX would come back short."""
    with pytest.raises(ValueError, match="ragged geometry"):
        conv_out_size(6, 3, 2, 0, axis="h")


def _reference_update(lr, b1, b2, eps, t, m, v):
    """Textbook Adam, in canonical m/v: `lr * mhat / (sqrt(vhat) + eps)`."""
    mhat = m / (1 - b1**t)
    vhat = v / (1 - b2**t)
    return lr * mhat / (math.sqrt(vhat) + eps)


@pytest.mark.parametrize("t", [1, 10, 100_000])
def test_fused_scalars_reproduce_textbook_adam(t):
    """`_step_fused` stores `S = m/(1-b1)` and `V = v/(1-b2)` and folds the whole bias correction into two."""
    lr, b1, b2, eps = 1e-4, 0.9, 0.999, 1e-8
    m, v = 0.37, 0.0021
    S, V = m / (1 - b1), v / (1 - b2)

    alpha_t, eps_t = adam_fused_scalars(lr, b1, b2, eps, t)
    got = alpha_t * S / (math.sqrt(V) + eps_t)

    assert got == pytest.approx(_reference_update(lr, b1, b2, eps, t, m, v), rel=1e-12)


def test_the_dx_capacity_check_refuses_an_unmeasured_arch_before_measuring():
    """The refusal has to come from the arch, not from the size, and only a
    size refusal is a ConvDoesNotFit: PPO relabels exactly that one as its
    minibatch_size."""
    geom = ConvGeometry(batch=32, in_channels=4, in_h=84, in_w=84,
                        out_channels=32, kernel=8, stride=4, padding=0)
    check_dx_capacity(geom, 110, 1_531_904, "blackhole")
    with pytest.raises(ValueError, match="wormhole_b0") as caught:
        check_dx_capacity(geom, 110, 1_531_904, "wormhole_b0")
    assert not isinstance(caught.value, ConvDoesNotFit)
    huge = ConvGeometry(batch=4096, in_channels=4, in_h=84, in_w=84,
                        out_channels=32, kernel=8, stride=4, padding=0)
    with pytest.raises(ConvDoesNotFit, match="batch 4096"):
        check_dx_capacity(huge, 110, 1_531_904, "blackhole")
