"""Matmul at HiFi4 with fp32 accumulation, and a folded-transpose backward."""

from __future__ import annotations

import ttnn

from ttrl.engine.autograd import Tensor, _detached, _needs
from ttrl.engine.ops.broadcast import _reduce_broadcast_grad

#: HiFi4 + `fp32_dest_acc_en` compute config per device, applied to every
#: matmul regardless of operand dtype, so the fidelity is stated rather than
#: inherited from a ttnn default that could change.
#:
#: HiFi4 is this hardware's accuracy ceiling, not full fp32: the FPU runs an
#: fp32 matmul as repeated bf16-granularity passes. Measured on Blackhole over
#: a (8,16)@(16,32) fp32 product whose entries reach 12, max abs error against
#: a float64 reference was LoFi 0.31, HiFi2 0.099, HiFi3 0.0166, HiFi4 0.0159,
#: so roughly 1.3e-3 relative at the best setting and no kernel option gets
#: below it. Anything comparing a matmul or its gradient against a host
#: reference must budget for that; it is not a bug to chase.
#:
#: Keyed by `id(device)`, so `purge_matmul_caches` must run before the device
#: closes: `id()` is reused once the device object is freed, and a stale
#: config bound to a dead device is silent wrong arithmetic.
_MATMUL_ACC_CFG: dict = {}


def _accurate_matmul_config(*operands):
    """The compute config every matmul over `operands` runs with."""
    device = operands[0].device()
    cfg = _MATMUL_ACC_CFG.get(id(device))
    if cfg is None:
        cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        )
        _MATMUL_ACC_CFG[id(device)] = cfg
    return cfg


def _matmul_raw(a, b, **kwargs):
    """`ttnn.matmul` at HiFi4 with fp32 accumulation."""
    kwargs["compute_kernel_config"] = _accurate_matmul_config(a, b)
    return ttnn.matmul(a, b, **kwargs)


def purge_matmul_caches(device) -> None:
    """Drop `_MATMUL_ACC_CFG` entries belonging to `device`, before it closes."""
    did = id(device)
    for key in [k for k in _MATMUL_ACC_CFG if k == did]:
        del _MATMUL_ACC_CFG[key]


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """(..., M, K) @ (..., K, N) -> (..., M, N).

    Both operands must be rank >= 2: the backward rules transpose the last two
    axes, and a rank-1 operand has no such pair. Inserting a temporary axis
    silently would return a gradient of a shape the caller never passed in.
    """
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError(
            f"matmul expects rank>=2 operands, got {a.shape} @ {b.shape}; "
            "reshape vectors to an explicit (1, K) / (K, 1) matrix first."
        )
    y = _matmul_raw(a.data, b.data)
    if not _needs(a, b):
        return _detached(y)
    out = Tensor(y, True, (a, b), "matmul")

    def _backward(grad):
        # Leading batch axes broadcast in the forward matmul exactly as in an
        # elementwise op, so the raw gradients carry the output's batch shape
        # and have to be summed back down to each operand's own shape. Without
        # this a (B, M, K) @ (K, N) matmul hands the weight a (B, K, N)
        # gradient and the optimizer silently updates on garbage.
        if a.requires_grad:
            a._accumulate(_reduce_broadcast_grad(
                _matmul_raw(grad, b.data, transpose_b=True), a.shape))
        if b.requires_grad:
            b._accumulate(_reduce_broadcast_grad(
                _matmul_raw(a.data, grad, transpose_a=True), b.shape))

    out._backward = _backward
    return out
