"""Space-to-depth fold of a convolution whose stride equals kernel/fold."""

import numpy as np
import pytest

from ttrl.engine.conv import (
    fold_input_numpy,
    fold_weight_numpy,
    unfold_weight_numpy,
)

CASES = [
    pytest.param(32, 4, 8, 4, id="conv1_4x8x8_fold4"),
    pytest.param(5, 3, 6, 3, id="odd_3x6x6_fold3"),
]


@pytest.mark.parametrize("out_c,in_c,k,fold", CASES)
def test_weight_fold_unfold_round_trips_exactly(out_c, in_c, k, fold):
    """Bit-exact, not approximately."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((out_c, in_c, k, k)).astype(np.float32)
    folded = fold_weight_numpy(w, fold)
    assert folded.shape == (out_c, in_c * fold * fold, k // fold, k // fold)
    assert np.array_equal(unfold_weight_numpy(folded, fold, in_c, k), w)


@pytest.mark.parametrize("out_c,in_c,k,fold", CASES)
def test_folded_conv_is_the_same_arithmetic(out_c, in_c, k, fold):
    """The whole point."""
    rng = np.random.default_rng(1)
    n, h, w_in = 2, k * 3, k * 3
    x = rng.standard_normal((n, h, w_in, in_c))
    w = rng.standard_normal((out_c, in_c, k, k))

    out_h = (h - k) // fold + 1
    ref = np.zeros((n, out_h, out_h, out_c))
    for i in range(out_h):
        for j in range(out_h):
            patch = x[:, i * fold:i * fold + k, j * fold:j * fold + k, :]
            ref[:, i, j, :] = np.einsum("nklc,ockl->no", patch, w)

    xf = fold_input_numpy(x, fold)
    wf = fold_weight_numpy(w, fold)
    kf = k // fold
    got = np.zeros_like(ref)
    for i in range(out_h):
        for j in range(out_h):
            patch = xf[:, i:i + kf, j:j + kf, :]
            got[:, i, j, :] = np.einsum("nklf,ofkl->no", patch, wf)

    assert np.allclose(got, ref, rtol=0, atol=1e-10)


