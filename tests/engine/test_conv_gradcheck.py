"""Conv2d forward, dX, dW and db against an analytic float64 numpy reference."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ttnn")

from ttrl.engine import device  # noqa: E402
from ttrl.engine.autograd import Tensor  # noqa: E402
from ttrl.engine.conv import Conv2d  # noqa: E402
from ttrl.engine.tensor import to_device, to_host  # noqa: E402

pytestmark = pytest.mark.tt

MIN_PCC = 0.999
MAX_FROB = 5e-2

#: The three Nature-CNN layer geometries, plus a padded case. Padding is a
#: separate case because the dX adjoint has to crop the transpose back to the
#: unpadded extent, and an off-by-one there is a shifted gradient rather than
#: an error.
CASES = [
    pytest.param(dict(cin=4, cout=32, k=8, s=4, p=0, h=84, w=84, n=2), id="L1_32x8x8s4"),
    pytest.param(dict(cin=8, cout=8, k=3, s=1, p=1, h=8, w=8, n=2), id="pad1"),
]


def _patches(x, k, s, p):
    """(N, OH, OW, C, KH, KW) windows of a padded NHWC input."""
    xp = np.pad(np.asarray(x, np.float64), ((0, 0), (p, p), (p, p), (0, 0)))
    return np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(1, 2))[:, ::s, ::s]


def _reference(x, w, b, dy, *, k, s, p):
    """Forward and all three gradients, from the definition, in float64."""
    x64, w64, dy64 = (np.asarray(a, np.float64) for a in (x, w, dy))
    patches = _patches(x64, k, s, p)
    y = np.einsum("nijckl,ockl->nijo", patches, w64) + np.asarray(b, np.float64)
    dw = np.einsum("nijckl,nijo->ockl", patches, dy64)
    db = dy64.sum(axis=(0, 1, 2))

    n, h, wd, _c = x64.shape
    dx = np.zeros((n, h + 2 * p, wd + 2 * p, x64.shape[3]))
    oh, ow = dy64.shape[1], dy64.shape[2]
    for i in range(oh):
        for j in range(ow):
            dx[:, i * s:i * s + k, j * s:j * s + k, :] += np.einsum(
                "no,ockl->nklc", dy64[:, i, j, :], w64
            )
    return y, dx[:, p:p + h, p:p + wd, :], dw, db


def _pcc(got, ref):
    a = np.asarray(got, np.float64).ravel()
    b = np.asarray(ref, np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def _frob(got, ref):
    a = np.asarray(got, np.float64).ravel()
    b = np.asarray(ref, np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _assert_matches(name, got, ref):
    pcc, frob = _pcc(got, ref), _frob(got, ref)
    assert pcc > MIN_PCC, f"{name}: pcc {pcc:.7f} <= {MIN_PCC}"
    assert frob < MAX_FROB, f"{name}: Frobenius rel err {frob:.3e} >= {MAX_FROB}"
    return pcc, frob


def _backward_from(out):
    """Walk the tape from `out`, whose `.grad` is already seeded."""
    topo, seen = [], set()
    stack = [(out, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            topo.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        for parent in node._parents:
            if id(parent) not in seen:
                stack.append((parent, False))
    for node in reversed(topo):
        if node.grad is not None:
            node._backward(node.grad)


@pytest.mark.parametrize("case", CASES)
def test_conv2d_forward_and_all_three_gradients(case):
    cin, cout, k, s, p, h, w, n = (
        case["cin"], case["cout"], case["k"], case["s"], case["p"],
        case["h"], case["w"], case["n"],
    )
    rng = np.random.default_rng(0)

    with device.acquire() as (dev, _profile):
        layer = Conv2d((cin, h, w), cout, k, dev, batch=n, stride=s, padding=p,
                       bias=True)
        w_np = layer.weight_numpy()
        b_np = layer.bias.numpy().reshape(cout)
        oh, ow = layer.geom.out_h, layer.geom.out_w
        assert (oh, ow) == ((h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1)

        x_np = (rng.standard_normal((n, h, w, cin)) * 0.1).astype(np.float32)
        dy_np = (rng.standard_normal((n, oh, ow, cout)) * 0.1).astype(np.float32)

        x = Tensor.from_numpy(x_np.reshape(1, 1, n * h * w, cin), dev,
                              requires_grad=True)
        out = layer(x)
        y = np.asarray(to_host(out.data)).reshape(n, oh, ow, cout)

        out.grad = to_device(dy_np.reshape(1, 1, n * oh * ow, cout), dev)
        _backward_from(out)

        dx = np.asarray(to_host(x.grad)).reshape(n, h, w, cin)
        dw = layer.weight_grad_numpy()
        db = to_host(layer.bias.grad).reshape(cout)

    y_ref, dx_ref, dw_ref, db_ref = _reference(x_np, w_np, b_np, dy_np,
                                               k=k, s=s, p=p)
    report = []
    for name, got, ref in (("forward", y, y_ref), ("dX", dx, dx_ref),
                           ("dW", dw, dw_ref), ("db", db, db_ref)):
        pcc, frob = _assert_matches(name, got, ref)
        report.append(f"{name} pcc={pcc:.7f} frob={frob:.2e}")
    print("\n" + "  ".join(report))


def test_conv2d_program_cache_stops_growing_after_warmup():
    """A conv whose dispatch re-plans every call rebuilds a program every step."""
    with device.acquire() as (dev, _profile):
        layer = Conv2d((64, 9, 9), 64, 3, dev, batch=2, bias=True)
        rng = np.random.default_rng(0)
        x_np = (rng.standard_normal((2, 9, 9, 64)) * 0.1).astype(np.float32)
        oh, ow = layer.geom.out_h, layer.geom.out_w

        def run_once():
            x = Tensor.from_numpy(x_np.reshape(1, 1, 2 * 81, 64), dev,
                                  requires_grad=True)
            out = layer(x)
            dy = (rng.standard_normal((2, oh, ow, 64)) * 0.1).astype(np.float32)
            out.grad = to_device(dy.reshape(1, 1, 2 * oh * ow, 64), dev)
            _backward_from(out)

        for _ in range(3):
            run_once()
        warm = dev.num_program_cache_entries()
        for _ in range(3):
            run_once()
        assert dev.num_program_cache_entries() == warm, (
            f"the program cache grew from {warm} to "
            f"{dev.num_program_cache_entries()} after warm-up"
        )
