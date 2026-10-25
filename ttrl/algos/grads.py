"""Gradient-norm measurement and clipping, shared by the three algorithms."""

from __future__ import annotations

import numpy as np
import ttnn

from ttrl.engine.tensor import to_host

__all__ = ["clip_grad_norm", "grad_norm", "grad_sq_device", "param_snapshot"]


def grad_norm(params) -> float:
    """Global L2 norm over every parameter that received a gradient.

    One host read per parameter, which is why this is the eager-mode
    measurement: inside a captured region `grad_sq_device` is the only legal
    spelling, because trace capture forbids host reads.
    """
    total = 0.0
    for param in params:
        if param.grad is None:
            continue
        arr = to_host(param.grad).astype(np.float64)
        total += float(np.sum(arr * arr))
    return float(np.sqrt(total))


def clip_grad_norm(params, max_norm: float) -> float:
    """Scale every gradient so their global L2 norm is at most `max_norm`.

    Returns the norm BEFORE clipping, which is the quantity worth logging.
    """
    if not max_norm > 0.0:
        raise ValueError(f"clip_grad_norm: max_norm must be > 0, got {max_norm}")
    params = list(params)
    total = grad_norm(params)
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for param in params:
            if param.grad is not None:
                param.grad = ttnn.mul(param.grad, scale)
    return total


def grad_sq_device(params):
    """Sum of squared gradients as a device tensor, or None if no gradients.

    Device-side so it can be computed inside a captured trace, where the host
    read `grad_norm` does would be fatal. The caller takes the square root
    after reading the one scalar back.
    """

    total = None
    for param in params:
        if param.grad is None:
            continue
        part = ttnn.sum(ttnn.mul(param.grad, param.grad))
        total = part if total is None else ttnn.add(total, part)
    return total


def param_snapshot(module) -> dict[str, np.ndarray]:
    """Host copy of every parameter, for measuring what an update moved."""
    return {name: param.numpy().copy() for name, param in module.named_parameters()}
