"""Gradchecks for the ops a recurrent cell and a LayerNorm are built from."""

import numpy as np
import pytest

pytest.importorskip("ttnn")  # skip cleanly on CPU-only CI (no ttnn install)

from ttrl.engine import device
from ttrl.engine.autograd import Tensor, backward
from ttrl.engine.layers import LayerNorm
from ttrl.engine.ops import concat, mul, split, sum_all
from ttrl.engine.tensor import to_host

pytestmark = pytest.mark.tt

ATOL = 3e-2
RTOL = 3e-2


@pytest.fixture(scope="module")
def dev():
    with device.acquire() as (handle, _profile):
        yield handle


def _grad(tensor):
    return np.asarray(to_host(tensor.grad), dtype=np.float64)


#: Every piece of a symmetric and an asymmetric cut. The asymmetric sizes are
#: load-bearing: with three equal pieces the padding on either side of the
#: middle one has the same width, so a backward that swapped the two sides
#: would scatter to exactly the right place and the check would pass.
#: The unaligned cuts are the state-action fusion of a continuous critic:
#: `concat(s, a)` for an 11-dim observation and a 3-dim action puts the
#: action's gradient at offset 11, and the tile is 32x32, so neither the
#: offset nor either extent is a tile boundary. Every cut here was a multiple
#: of 32 before, so the padding the scatter builds had always been whole tiles.
SPLIT_PLACEMENTS = [((32, 32, 32), 1), ((32, 64), 1), ((64, 32), 1),
                    ((11, 85), 0), ((11, 85), 1), ((17, 4, 75), 1),
                    ((17, 4, 75), 2)]


@pytest.mark.parametrize("sizes, index", SPLIT_PLACEMENTS)
def test_split_grad_lands_only_under_the_piece_it_came_from(sizes, index, dev):
    """The scatter is what makes `split` usable in a cell."""
    x = Tensor.from_numpy(np.zeros((4, 96), dtype=np.float32), dev,
                          requires_grad=True)
    pieces = split(x, sizes)
    backward(sum_all(mul(pieces[index], 3.0)))
    offset = sum(sizes[:index])
    want = np.zeros((4, 96))
    want[:, offset:offset + sizes[index]] = 3.0
    np.testing.assert_allclose(_grad(x), want, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("extents", [(32, 32), (11, 3), (17, 4), (1, 95)])
def test_concat_backward_slices_each_operand_back_off_an_unaligned_offset(extents, dev):
    """`concat`'s backward slices the incoming gradient at a running offset.

    The state-action fusion of a continuous critic concatenates widths that
    are not tile multiples and its action operand requires grad on every actor
    step, so this slice runs unaligned in the hot path.
    """
    parts = [Tensor.from_numpy(np.zeros((4, e), dtype=np.float32), dev,
                               requires_grad=True) for e in extents]
    joined = concat(parts, dim=-1)
    weights = np.arange(1, sum(extents) + 1, dtype=np.float32).reshape(1, -1)
    backward(sum_all(mul(joined, Tensor.from_numpy(weights, dev))))
    offset = 0
    for part, extent in zip(parts, extents):
        want = np.tile(weights[:, offset:offset + extent], (4, 1))
        np.testing.assert_allclose(_grad(part), want, atol=ATOL, rtol=RTOL)
        offset += extent


def test_split_grads_from_several_pieces_accumulate(dev):
    x = Tensor.from_numpy(np.zeros((4, 96), dtype=np.float32), dev,
                          requires_grad=True)
    first, second, third = split(x, (32, 32, 32))
    total = sum_all(mul(first, 1.0))
    total = total + sum_all(mul(second, 2.0))
    total = total + sum_all(mul(third, 5.0))
    backward(total)
    got = _grad(x)
    want = np.concatenate([np.full((4, 32), 1.0), np.full((4, 32), 2.0),
                           np.full((4, 32), 5.0)], axis=1)
    np.testing.assert_allclose(got, want, atol=ATOL, rtol=RTOL)


def test_split_on_the_leading_axis_is_the_sequence_cut(dev):
    """The unroll in `ttrl.engine.recurrent.GRU` cuts timesteps off axis 0."""
    x_np = np.arange(5 * 4 * 8, dtype=np.float32).reshape(5, 4, 8)
    x = Tensor.from_numpy(x_np, dev, requires_grad=True)
    steps = split(x, (1,) * 5, dim=0)
    assert [tuple(s.shape) for s in steps] == [(1, 4, 8)] * 5
    np.testing.assert_allclose(steps[3].numpy(), x_np[3:4])

    backward(sum_all(mul(steps[3], 7.0)))
    want = np.zeros((5, 4, 8))
    want[3] = 7.0
    np.testing.assert_allclose(_grad(x), want, atol=ATOL, rtol=RTOL)


def _np_layer_norm(x, weight, bias, eps):
    centred = x - x.mean(axis=-1, keepdims=True)
    variance = (centred ** 2).mean(axis=-1, keepdims=True)
    return centred / np.sqrt(variance + eps) * weight + bias


def test_layer_norm_forward_normalises_each_row(dev):
    rng = np.random.default_rng(2)
    x_np = (rng.normal(size=(8, 64)) * 4.0 + 3.0).astype(np.float32)
    layer = LayerNorm(64, dev)
    out = layer(Tensor.from_numpy(x_np, dev)).numpy()

    want = _np_layer_norm(x_np.astype(np.float64), 1.0, 0.0, layer.eps)
    np.testing.assert_allclose(out, want, atol=ATOL, rtol=RTOL)
    # Checked per row: a reduction over the wrong axis still produces a
    # zero-mean tensor when measured over the whole thing.
    np.testing.assert_allclose(out.mean(axis=-1), np.zeros(8), atol=ATOL)
    np.testing.assert_allclose(out.std(axis=-1), np.ones(8), atol=ATOL)


def test_layer_norm_grad_flows_through_the_statistics(dev):
    """The derivative of the real function, which has a mean-and-variance correction term."""
    rng = np.random.default_rng(3)
    features = 64
    x_np = (rng.normal(size=(8, features)) * 2.0).astype(np.float32)
    weights = rng.normal(size=(8, features)).astype(np.float32)

    layer = LayerNorm(features, dev)
    x = Tensor.from_numpy(x_np, dev, requires_grad=True)
    out = layer(x)
    backward(sum_all(mul(out, Tensor.from_numpy(weights, dev))))

    x64 = x_np.astype(np.float64)
    gamma = np.asarray(to_host(layer.weight.data), dtype=np.float64)
    upstream = weights.astype(np.float64) * gamma
    centred = x64 - x64.mean(axis=-1, keepdims=True)
    variance = (centred ** 2).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(variance + layer.eps)
    normed = centred * inv
    want = inv * (upstream
                  - upstream.mean(axis=-1, keepdims=True)
                  - normed * (upstream * normed).mean(axis=-1, keepdims=True))
    np.testing.assert_allclose(_grad(x), want, atol=ATOL, rtol=RTOL)


