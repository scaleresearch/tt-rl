"""Binary elementwise arithmetic: add, sub, mul, div.

Each takes a Tensor and either a Tensor or a python scalar. The scalar
spellings hand the constant to the kernel instead of materialising it in DRAM.
"""

from __future__ import annotations

import ttnn

from ttrl.engine.autograd import Tensor, _as_tensor, _detached, _is_scalar, _needs
from ttrl.engine.ops.broadcast import _reduce_broadcast_grad


def add(a: Tensor, b) -> Tensor:
    if _is_scalar(b):
        # x + c: the constant goes to the kernel, not to DRAM. Shape is
        # unchanged, so there is no broadcast reduction on the way back.
        y = ttnn.add(a.data, float(b))
        if not _needs(a):
            return _detached(y)
        out = Tensor(y, True, (a,), "add")

        def _backward_scalar(grad):
            if a.requires_grad:
                a._accumulate(grad)

        out._backward = _backward_scalar
        return out
    b = _as_tensor(b, a)
    y = ttnn.add(a.data, b.data)
    if not _needs(a, b):
        return _detached(y)
    out = Tensor(y, True, (a, b), "add")

    def _backward(grad):
        if a.requires_grad:
            a._accumulate(_reduce_broadcast_grad(grad, a.shape))
        if b.requires_grad:
            b._accumulate(_reduce_broadcast_grad(grad, b.shape))

    out._backward = _backward
    return out


def sub(a, b) -> Tensor:
    if isinstance(a, Tensor) and _is_scalar(b):
        y = ttnn.sub(a.data, float(b))
        if not _needs(a):
            return _detached(y)
        out = Tensor(y, True, (a,), "sub")

        def _backward_scalar(grad):
            if a.requires_grad:
                a._accumulate(grad)

        out._backward = _backward_scalar
        return out
    if isinstance(b, Tensor) and _is_scalar(a):
        # c - x. `ttnn.rsub` is exactly that and avoids a `full_like`.
        y = ttnn.rsub(b.data, float(a))
        if not _needs(b):
            return _detached(y)
        out = Tensor(y, True, (b,), "sub")

        def _backward_rscalar(grad):
            if b.requires_grad:
                b._accumulate(ttnn.neg(grad))

        out._backward = _backward_rscalar
        return out
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise TypeError(f"sub expects Tensor operands or one scalar, got {type(a)} and {type(b)}")
    y = ttnn.sub(a.data, b.data)
    if not _needs(a, b):
        return _detached(y)
    out = Tensor(y, True, (a, b), "sub")

    def _backward(grad):
        if a.requires_grad:
            a._accumulate(_reduce_broadcast_grad(grad, a.shape))
        if b.requires_grad:
            b._accumulate(_reduce_broadcast_grad(ttnn.neg(grad), b.shape))

    out._backward = _backward
    return out


def mul(a: Tensor, b) -> Tensor:
    if _is_scalar(b):
        s = float(b)
        y = ttnn.mul(a.data, s)
        if not _needs(a):
            return _detached(y)
        out = Tensor(y, True, (a,), "mul")

        def _backward_scalar(grad):
            if a.requires_grad:
                a._accumulate(ttnn.mul(grad, s))

        out._backward = _backward_scalar
        return out
    b = _as_tensor(b, a)
    y = ttnn.mul(a.data, b.data)
    if not _needs(a, b):
        return _detached(y)
    out = Tensor(y, True, (a, b), "mul")

    def _backward(grad):
        if a.requires_grad:
            a._accumulate(_reduce_broadcast_grad(ttnn.mul(grad, b.data), a.shape))
        if b.requires_grad:
            b._accumulate(_reduce_broadcast_grad(ttnn.mul(grad, a.data), b.shape))

    out._backward = _backward
    return out


def div(a: Tensor, b) -> Tensor:
    if _is_scalar(b):
        s = float(b)
        y = ttnn.div(a.data, s)
        if not _needs(a):
            return _detached(y)
        out = Tensor(y, True, (a,), "div")

        def _backward_scalar(grad):
            if a.requires_grad:
                a._accumulate(ttnn.div(grad, s))

        out._backward = _backward_scalar
        return out
    b = _as_tensor(b, a)
    y = ttnn.div(a.data, b.data)
    if not _needs(a, b):
        return _detached(y)
    out = Tensor(y, True, (a, b), "div")

    def _backward(grad):
        if a.requires_grad:
            a._accumulate(_reduce_broadcast_grad(ttnn.div(grad, b.data), a.shape))
        if b.requires_grad:
            # d/db (a/b) = -a/b^2
            grad_b = ttnn.mul(grad, ttnn.neg(ttnn.div(a.data, ttnn.mul(b.data, b.data))))
            b._accumulate(_reduce_broadcast_grad(grad_b, b.shape))

    out._backward = _backward
    return out
