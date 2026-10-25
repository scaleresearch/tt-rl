"""dX and dW, both built from device ops only.

dX is ttnn.conv_transpose2d, which is the adjoint of conv2d and is what the
hand-derived dilate/pad/flip/conv2d identity this replaces was approximating.
Verified against a torch reference at pcc 0.9999998 and ~2e-3 relative error,
which is this hardware's fp32 matmul floor.

dW is the batch/channel-swap identity written as one matmul over k*k strided
im2col slices. The matmul emits its result directly in the prepared master's
row order, so the weight gradient needs no permutation to be accumulated into
the master.
"""

from __future__ import annotations

import ttnn

from ttrl.engine.conv.config import FP32, accurate_compute_config, rm, tile
from ttrl.engine.conv.geometry import ConvGeometry
from ttrl.engine.conv.packing import PreparedLayout
from ttrl.engine.conv.transform import (
    TransposeSpec, dw_to_prepared, transpose_dispatch, transpose_weight,
)
from ttrl.engine.conv.weights import PreparedSpec

__all__ = ["input_grad", "weight_grad"]


def _scale(t, scale: float):
    return t if scale == 1.0 else ttnn.multiply(t, scale)


def input_grad(dy_rm, master, pspec: PreparedSpec, tspec: TransposeSpec, device, *,
               input_scale: float = 1.0):
    """dX = conv_transpose2d(dY, W), with W taken from the prepared master.

    mirror_kernel=True both here and in the weight transform: the op does not
    mirror the kernel itself, and with an unmirrored weight it returns a
    gradient uncorrelated with the true one without raising.
    """
    weight = _scale(transpose_weight(master, pspec, tspec), input_scale)
    return transpose_dispatch(dy_rm, weight, tspec, device)


def weight_grad(x_rm, dy_rm, geom: ConvGeometry, layout: PreparedLayout, device, *,
                input_scale: float = 1.0):
    """dW[k, c, p, q] = sum over (n, i, j) of dY[n, i, j, k] * X[n, i*s+p, j*s+q, c].

    One strided slice of X per kernel tap, concatenated, then a single
    (M, k*k*C)^T @ (M, K) matmul. The conv2d spelling of this identity needs
    its "weight" (dY) in conv2d's prepared layout, which for a tensor that
    changes every step means a per-step host round trip and therefore no trace
    capture; it also rejects the space-to-depth folded shape outright
    (reader_indices_actual_page_size, no Conv2dConfig escapes it). The matmul
    spelling has neither problem and covers every geometry.
    """
    n, k, s, p = geom.batch, geom.kernel, geom.stride, geom.padding
    c, out_h, out_w = geom.in_channels, geom.out_h, geom.out_w
    xf = ttnn.reshape(x_rm, (n, geom.in_h, geom.in_w, c))
    if p:
        xf = ttnn.pad(xf, ((0, 0), (p, p), (p, p), (0, 0)), value=0.0)
    m = n * out_h * out_w
    taps = []
    for i in range(k):
        for j in range(k):
            window = ttnn.slice(
                xf, (0, i, j, 0),
                (n, i + s * (out_h - 1) + 1, j + s * (out_w - 1) + 1, c),
                (1, s, s, 1),
            )
            taps.append(tile(ttnn.reshape(window, (m, c))))
    xcat = taps[0] if len(taps) == 1 else ttnn.concat(taps, dim=1)
    dy_m = tile(ttnn.reshape(rm(dy_rm), (m, layout.out_channels)))
    mm = ttnn.matmul(
        xcat, dy_m, transpose_a=True,
        compute_kernel_config=accurate_compute_config(device), dtype=FP32,
    )
    return dw_to_prepared(_scale(mm, input_scale), layout)
