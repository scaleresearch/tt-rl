"""Reductions, and the detached max/argmax pair."""

from __future__ import annotations

import ttnn

from ttrl.engine.autograd import Tensor, _detached, _needs
from ttrl.engine.ops.broadcast import _broadcast_grad_to
from ttrl.engine.ops.elementwise import mul


def sum_all(a: Tensor) -> Tensor:
    """Sum every element to a scalar-shaped tensor."""
    y = ttnn.sum(a.data)
    if not _needs(a):
        return _detached(y)
    rank = len(a.shape)
    out = Tensor(y, True, (a,), "sum_all")

    def _backward(grad):
        if not a.requires_grad:
            return
        # The gradient of a sum broadcasts unchanged to every input element,
        # done entirely on device: a host read of the grad here is illegal
        # during trace capture and made every sum_all-based loss uncapturable.
        # Explicit rank fixup: the scalar grad has to be reshaped to a
        # keepdim shape before it can broadcast. The deleted zeros-and-add
        # spelling of `_broadcast_grad_to` was absorbing this mismatch.
        if len(tuple(grad.shape)) != rank:
            grad = ttnn.reshape(grad, (1,) * rank)
        a._accumulate(_broadcast_grad_to(grad, a.data))

    out._backward = _backward
    return out


def mean_all(a: Tensor) -> Tensor:
    n = 1
    for d in a.shape:
        n *= int(d)
    return mul(sum_all(a), 1.0 / n)


def sum_dim(a: Tensor, dim: int, keepdim: bool = True) -> Tensor:
    """Sum along one axis. Tile layout dislikes rank changes, so callers on the
    tile-native path pass `keepdim=True` and squeeze explicitly if they need
    rank-1."""
    y = ttnn.sum(a.data, dim=dim, keepdim=keepdim)
    if not _needs(a):
        return _detached(y)
    out = Tensor(y, True, (a,), "sum_dim")

    def _backward(grad):
        if not a.requires_grad:
            return
        if not keepdim:
            d = dim if dim >= 0 else dim + len(a.shape)
            grad = ttnn.reshape(grad, tuple(a.shape[:d]) + (1,) + tuple(a.shape[d + 1:]))
        a._accumulate(_broadcast_grad_to(grad, a.data))

    out._backward = _backward
    return out


def mean_dim(a: Tensor, dim: int, keepdim: bool = True) -> Tensor:
    d = dim if dim >= 0 else dim + len(a.shape)
    n = int(a.shape[d])
    return mul(sum_dim(a, dim=dim, keepdim=keepdim), 1.0 / n)


def max_dim_detached(a: Tensor, dim: int, keepdim: bool = True) -> Tensor:
    """Max along one axis, deliberately not differentiable.

    Used as the stability shift in `log_softmax`: subtracting a constant before
    exponentiating does not change the result, so the max's true gradient
    contribution is exactly zero. Treating it as a constant is exact, not an
    approximation.
    """
    return Tensor(ttnn.max(a.data, dim=dim, keepdim=keepdim), requires_grad=False)


def argmax_dim_detached(a: Tensor, dim: int = -1, keepdim: bool = True) -> Tensor:
    """Index of the max along one axis, as a float32 device tensor.

    Detached because an index is piecewise constant in its input, so there is
    nothing to route.

    float32 rather than `ttnn.argmax`'s native uint32 because the only consumer
    compares the index against a float32 lane-number constant, and uint32
    tensors do not participate in the fp32 elementwise ops used elsewhere. The
    typecast is exact for any index a head can produce. The whole chain stays
    device to device: a host read of the indices mid-step is fatal to
    `ttnn.begin_trace_capture`.

    Note the input dtype matters upstream: bf16 rounding can collapse two
    distinct values into one and flip which index wins.
    """
    idx = ttnn.argmax(a.data, dim=dim, keepdim=keepdim)
    return Tensor(ttnn.typecast(idx, ttnn.float32), requires_grad=False)
