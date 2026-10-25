"""Unary elementwise ops.

`exp`, `tanh` and `sigmoid` compose their backward from the saved forward
output; `log` composes as `grad / x`. The fused `*_bw` kernels are NOT
bfloat16-only on ttnn 0.80.0 despite their docstrings: measured in fp32 on
Blackhole they return FLOAT32 with max abs error 9.54e-07 (`exp_bw`),
2.09e-07 (`tanh_bw`) and 1.64e-07 (`sigmoid_bw`), so `relu` below calls
`ttnn.relu_bw` (exact) directly. The compositions are kept because each reuses
an output already saved on the tape, and because `log_bw` computes `G/X`, whose
conditioning near zero shows up as 1.22e-04 error where `ttnn.div` does not.
"""

from __future__ import annotations

import ttnn

from ttrl.engine.autograd import Tensor, _detached, _needs
from ttrl.engine.ops.elementwise import mul


def neg(a: Tensor) -> Tensor:
    return mul(a, -1.0)


def relu(a: Tensor) -> Tensor:
    y = ttnn.relu(a.data)
    if not _needs(a):
        return _detached(y)
    out = Tensor(y, True, (a,), "relu")

    def _backward(grad):
        if not a.requires_grad:
            return
        a._accumulate(ttnn.relu_bw(grad, a.data)[0])

    out._backward = _backward
    return out


def abs_(a: Tensor) -> Tensor:
    y = ttnn.abs(a.data)
    if not _needs(a):
        return _detached(y)
    out = Tensor(y, True, (a,), "abs")

    def _backward(grad):
        if not a.requires_grad:
            return
        a._accumulate(ttnn.mul(grad, ttnn.sign(a.data)))

    out._backward = _backward
    return out


def square(a: Tensor) -> Tensor:
    y = ttnn.mul(a.data, a.data)
    if not _needs(a):
        return _detached(y)
    out = Tensor(y, True, (a,), "square")

    def _backward(grad):
        if not a.requires_grad:
            return
        a._accumulate(ttnn.mul(grad, ttnn.mul(a.data, 2.0)))

    out._backward = _backward
    return out


def exp(a: Tensor) -> Tensor:
    y = ttnn.exp(a.data)
    if not _needs(a):
        return _detached(y)
    out = Tensor(y, True, (a,), "exp")

    def _backward(grad):
        if not a.requires_grad:
            return
        # grad * y, reusing the saved forward output rather than recomputing
        # it inside `ttnn.exp_bw`.
        a._accumulate(ttnn.mul(grad, y))

    out._backward = _backward
    return out


def log(a: Tensor) -> Tensor:
    """Natural log. Input must be > 0 (ttnn.log's documented domain)."""
    y = ttnn.log(a.data)
    if not _needs(a):
        return _detached(y)
    out = Tensor(y, True, (a,), "log")

    def _backward(grad):
        if not a.requires_grad:
            return
        a._accumulate(ttnn.div(grad, a.data))

    out._backward = _backward
    return out


def tanh(a: Tensor) -> Tensor:
    y = ttnn.tanh(a.data)
    if not _needs(a):
        return _detached(y)
    out = Tensor(y, True, (a,), "tanh")

    def _backward(grad):
        if not a.requires_grad:
            return
        # grad * (1 - y*y), reusing the saved forward output.
        a._accumulate(ttnn.mul(grad, ttnn.rsub(ttnn.mul(y, y), 1.0)))

    out._backward = _backward
    return out


def sigmoid(a: Tensor) -> Tensor:
    y = ttnn.sigmoid(a.data)
    if not _needs(a):
        return _detached(y)
    out = Tensor(y, True, (a,), "sigmoid")

    def _backward(grad):
        if not a.requires_grad:
            return
        # grad * y * (1 - y), reusing the saved forward output.
        a._accumulate(ttnn.mul(grad, ttnn.mul(y, ttnn.rsub(y, 1.0))))

    out._backward = _backward
    return out


def sqrt(a: Tensor) -> Tensor:
    """Elementwise square root. Input must be > 0: the derivative 1/(2*sqrt(x))
    is unbounded at zero, so a caller normalising by a variance adds its
    epsilon before calling this, not after."""
    y = ttnn.sqrt(a.data)
    if not _needs(a):
        return _detached(y)
    out = Tensor(y, True, (a,), "sqrt")

    def _backward(grad):
        if not a.requires_grad:
            return
        a._accumulate(ttnn.div(grad, ttnn.mul(y, 2.0)))

    out._backward = _backward
    return out
