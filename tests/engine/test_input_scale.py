"""`Conv2d(input_scale=s)` is the exact identity `conv(s*x, W) == conv(x, s*W)`."""

from __future__ import annotations

import numpy as np
import pytest

ttnn = pytest.importorskip("ttnn")

from ttrl.engine import device  # noqa: E402
from ttrl.engine.autograd import Tensor  # noqa: E402
from ttrl.engine.conv import Conv2d  # noqa: E402
from ttrl.engine.tensor import to_host  # noqa: E402

FP32 = ttnn.float32
SCALE = 1.0 / 255.0
BATCH = 8
IN_SHAPE = (4, 84, 84)
OUT_CHANNELS, KERNEL, STRIDE = 32, 8, 4
OUT_HW = (84 - KERNEL) // STRIDE + 1

#: The two spellings are not bit-identical: one multiplies 8192 weight
#: elements by 1/255, the other 225k activation elements, so they are compared
#: on a Frobenius-norm ratio rather than exactly.
REL_TOL = 0.01


def _frob(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _obs(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(BATCH, *IN_SHAPE), dtype=np.uint8)


def _layer(dev, weight, *, input_scale, fold):
    return Conv2d(IN_SHAPE, OUT_CHANNELS, KERNEL, dev, batch=BATCH, stride=STRIDE,
                  bias=False, fold=fold, input_scale=input_scale, weight=weight)


def _run(layer, x_np, dtype, dy):
    """One forward and one backward with a fixed upstream gradient."""
    x = Tensor.from_numpy(x_np.astype(np.float32), layer.device, requires_grad=False,
                          dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    out = layer(x)
    fwd = to_host(out.data).reshape(-1).copy()
    out._backward(dy)
    return fwd, layer.weight_numpy().copy()


def _inputs(obs, fold):
    """(raw-pixel array, pre-divided array) in the layer's flattened NHWC form."""
    raw = obs.astype(np.float32)
    if fold > 1:
        f = fold
        v = raw.reshape(BATCH, IN_SHAPE[0], 84 // f, f, 84 // f, f)
        raw = v.transpose(0, 2, 4, 3, 5, 1)
    else:
        raw = raw.transpose(0, 2, 3, 1)
    raw = np.ascontiguousarray(raw).reshape(1, 1, -1, IN_SHAPE[0] * fold * fold)
    return raw, raw / 255.0


def _upstream_grad(dev, seed):
    rng = np.random.default_rng(seed)
    dy_np = rng.standard_normal(
        (1, 1, BATCH * OUT_HW * OUT_HW, OUT_CHANNELS)
    ).astype(np.float32)
    return ttnn.Tensor(dy_np, FP32).to(ttnn.TILE_LAYOUT).to(dev)


@pytest.mark.tt
@pytest.mark.parametrize("fold", [1, 4])
def test_input_scale_matches_scaling_the_input(fold):
    """`conv(x, s*W)` and `conv(s*x, W)` agree in forward AND in dW."""
    raw, scaled = _inputs(_obs(), fold)
    rng = np.random.default_rng(1)
    weight = (rng.standard_normal((OUT_CHANNELS, IN_SHAPE[0], KERNEL, KERNEL))
              / 16.0).astype(np.float32)

    with device.acquire() as (dev, _profile):
        dy = _upstream_grad(dev, 1)

        folded = _layer(dev, weight, input_scale=SCALE, fold=fold)
        f_fwd, f_weight = _run(folded, raw, ttnn.bfloat16, dy)
        f_dw = to_host(folded.weight.grad)

        plain = _layer(dev, weight, input_scale=1.0, fold=fold)
        p_fwd, _p_weight = _run(plain, scaled, FP32, dy)
        p_dw = to_host(plain.weight.grad)

        assert _frob(f_fwd, p_fwd) < REL_TOL, (
            f"forward mismatch: {_frob(f_fwd, p_fwd):.3e}")
        # A fold applied to the forward but not to dW would land 255x out.
        assert _frob(f_dw, p_dw) < REL_TOL, (
            f"dW mismatch: {_frob(f_dw, p_dw):.3e}. If this is ~1.0 or ~255, "
            "the 1/255 fold is applied on only one side of the tape")

        # The stored parameter is untouched: the scaling happens at the point
        # of use, never to the master weight.
        assert np.allclose(f_weight, weight, atol=0, rtol=0)


