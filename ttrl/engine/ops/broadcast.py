"""Broadcast-gradient reduction and expansion.

Shared by every elementwise, matmul and reduction backward.
"""

from __future__ import annotations

import ttnn


def _reduce_broadcast_grad(grad, target_shape):
    """Sum `grad` back down to `target_shape` along any broadcast dim.

    For every dim where the operand is 1 and the co-broadcast grad is not, sum
    that dim with keepdim. Handles equal-rank broadcasting and a grad with
    extra leading dims.
    """
    grad_shape = tuple(grad.shape)
    target_shape = tuple(target_shape)
    pad = len(grad_shape) - len(target_shape)
    assert pad >= 0, "grad rank must be >= operand rank"
    padded_target = (1,) * pad + target_shape

    for dim, (g_sz, t_sz) in enumerate(zip(grad_shape, padded_target)):
        if t_sz == 1 and g_sz != 1:
            grad = ttnn.sum(grad, dim=dim, keepdim=True)

    for _ in range(pad):
        grad = ttnn.squeeze(grad, 0)
    return grad


def _broadcast_grad_to(grad, like) -> "ttnn.Tensor":
    """Expand a reduced gradient back to `like`'s shape, on device."""
    gshape = tuple(int(d) for d in grad.shape)
    tshape = tuple(int(d) for d in like.shape)
    if len(gshape) != len(tshape):
        raise ValueError(
            f"cannot broadcast gradient of rank {len(gshape)} back to rank "
            f"{len(tshape)}: reshape it to a keepdim shape first."
        )
    # Broadcast semantics, not any integer multiple: a dim may only expand
    # from 1. Repeating a size-2 gradient into a size-4 axis tiles it instead
    # of summing, which is not the gradient of any reduction.
    for g, t in zip(gshape, tshape):
        if g != t and g != 1:
            raise ValueError(
                f"cannot broadcast gradient of shape {gshape} back to {tshape}: "
                "axis sizes must be equal or 1."
            )
    reps = tuple(t // g for g, t in zip(gshape, tshape))
    if reps == (1,) * len(reps):
        return grad
    return ttnn.repeat(grad, reps)

