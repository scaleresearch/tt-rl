"""Device-only transforms between the prepared master weight and the two
layouts the backward pass needs.

Everything here is device op to device op, which is what lets a whole conv
backward live inside a captured trace, and every transform is verified
bit-exact at construction against the ttnn call it replaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ttnn

from ttrl.engine.conv.config import FP32, accurate_compute_config, conv_config, tile
from ttrl.engine.conv.packing import PreparedLayout, candidate_layouts
from ttrl.engine.conv.weights import PreparedSpec
from ttrl.engine.tensor import to_host

__all__ = [
    "TransposeSpec", "dw_to_prepared", "prepare_transpose", "transpose_dispatch",
    "transpose_weight", "verify_dw_layout",
]

#: "the forward's shard layout", distinct from an explicit None.
_SAME = object()


@dataclass(frozen=True)
class TransposeSpec:
    """The conv_transpose2d dX call and the weight layout it needs.

    Its own fields rather than a ConvGeometry: this op consumes dY and emits
    something larger, so ConvGeometry's forward output-size rules do not
    describe it.
    """

    batch: int
    in_channels: int
    out_channels: int
    in_h: int
    in_w: int
    kernel: int
    stride: int
    padding: int
    layout: PreparedLayout
    shard_layout: object
    conv_cfg: object


def _flip(t, axis: int, n: int):
    """Reverse `t` along `axis`, on device, in TILE layout.

    ttnn has no flip or reverse and ttnn.slice refuses a negative step, so the
    reversal is n unit-width forward slices concatenated in reverse order.
    Only ever applied to a kernel axis (8, 4 or 3 elements), so this is a
    handful of ops on a kernel-sized tensor.
    """
    if n <= 1:
        return t
    parts = []
    for i in reversed(range(n)):
        begins = [0] * len(t.shape)
        ends = [int(v) for v in t.shape]
        begins[axis], ends[axis] = i, i + 1
        parts.append(ttnn.slice(t, tuple(begins), tuple(ends)))
    return ttnn.concat(parts, dim=axis)


def transpose_weight(master, pspec: PreparedSpec, tspec: TransposeSpec):
    """The conv_transpose2d weight for dX, derived from the prepared master.

    conv_transpose2d with mirror_kernel=True wants the master spatially
    flipped and with its channel axes swapped, in the prepared matrix layout of
    the dX geometry. In the (R, S, cp, cols) view of the master that is: drop
    the padding, reverse both kernel axes, transpose the channel pair, pad back
    out to the dX layout. The permute runs in TILE because moving data along
    the innermost axis of a row-major fp32 tensor loses precision on this build.
    """
    src, dst = pspec.layout, tspec.layout
    r, s = src.kernel_h, src.kernel_w
    w = ttnn.reshape(master, (r, src.row_block, src.cols))
    if not src.is_tight:
        w = ttnn.slice(w, (0, 0, 0), (r, src.packed_rows, src.cols))
    w = ttnn.reshape(w, (r, s, src.cp, src.cols))
    if src.cp != src.in_channels or src.cols != src.out_channels:
        w = ttnn.slice(w, (0, 0, 0, 0), (r, s, src.in_channels, src.out_channels))
    w = _flip(_flip(w, 0, r), 1, s)
    w = ttnn.permute(w, (0, 1, 3, 2))
    if src.out_channels != dst.cp or src.in_channels != dst.cols:
        w = ttnn.pad(
            w,
            ((0, 0), (0, 0), (0, dst.cp - src.out_channels), (0, dst.cols - src.in_channels)),
            value=0.0,
        )
    w = ttnn.reshape(w, (r, dst.packed_rows, dst.cols))
    if not dst.is_tight:
        w = ttnn.pad(w, ((0, 0), (0, dst.row_block - dst.packed_rows), (0, 0)), value=0.0)
    return ttnn.reshape(w, dst.shape)


def transpose_dispatch(dy_rm, weight, tspec: "TransposeSpec", device):
    """One ttnn.conv_transpose2d dispatch. The single spelling of the dX call.

    mirror_kernel=True: the op does not mirror the kernel itself, and with an
    unmirrored weight it returns a gradient uncorrelated with the true one
    without raising.
    """
    return ttnn.conv_transpose2d(
        input_tensor=dy_rm,
        weight_tensor=weight,
        device=device,
        in_channels=tspec.in_channels,
        out_channels=tspec.out_channels,
        batch_size=tspec.batch,
        input_height=tspec.in_h,
        input_width=tspec.in_w,
        kernel_size=(tspec.kernel, tspec.kernel),
        stride=(tspec.stride, tspec.stride),
        padding=(tspec.padding, tspec.padding),
        output_padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        conv_config=tspec.conv_cfg,
        compute_config=accurate_compute_config(device),
        dtype=FP32,
        mirror_kernel=True,
    )


def _reference_transpose(w_host, call: dict, device, conv_cfg, pspec: PreparedSpec):
    return ttnn.prepare_conv_transpose2d_weights(
        weight_tensor=w_host,
        input_memory_config=pspec.input_memory_config,
        input_layout=pspec.input_layout,
        weights_format="IOHW",
        device=device,
        input_dtype=FP32,
        conv_config=conv_cfg,
        compute_config=accurate_compute_config(device),
        mirror_kernel=True,
        **call,
    )


def prepare_transpose(master, w_oihw: np.ndarray, pspec: PreparedSpec, device, *,
                      shard_layout=_SAME) -> TransposeSpec:
    """Derive the dX weight layout and verify the device transform against it.

    Runs once per layer at construction. The gate is that the weight this
    module derives on device and the weight ttnn prepares for itself produce
    bit-identical gradients, because mirror_kernel is the whole correctness of
    dX: with the weight unmirrored the op runs happily and returns a gradient
    that correlates with the true one at pcc ~0.0.

    The gate is an output comparison rather than a matrix comparison because
    the prepared transpose matrix's padded bytes are only readable through
    torch on ttnn 0.80 (see weights.prepare_master), and the channel stride
    that is invisible in the metadata is therefore found by running each
    layout the metadata permits. Exactly one must reproduce ttnn's own weight.
    """
    fwd = pspec.geom
    fields = dict(
        batch=fwd.batch, in_channels=fwd.out_channels, out_channels=fwd.in_channels,
        in_h=fwd.out_h, in_w=fwd.out_w, kernel=fwd.kernel, stride=fwd.stride,
        padding=fwd.padding,
    )
    call = dict(
        in_channels=fwd.out_channels, out_channels=fwd.in_channels, batch_size=fwd.batch,
        input_height=fwd.out_h, input_width=fwd.out_w,
        kernel_size=(fwd.kernel, fwd.kernel), stride=(fwd.stride, fwd.stride),
        padding=(fwd.padding, fwd.padding), output_padding=(0, 0), dilation=(1, 1),
        has_bias=False, groups=1,
    )
    if shard_layout is _SAME:
        shard_layout = pspec.shard_layout
    conv_cfg = conv_config(shard_layout, weights_dtype=FP32)
    k = fwd.kernel
    rng = np.random.default_rng(2)
    probe = rng.standard_normal(
        (fwd.out_channels, fwd.in_channels, k, k)).astype(np.float32)
    probe_master = ttnn.Tensor(pspec.layout.pack(probe), FP32).to(
        ttnn.TILE_LAYOUT).to(device)
    ref = _reference_transpose(
        ttnn.Tensor(probe, FP32).to(pspec.input_layout), call, device, conv_cfg, pspec)
    rows, cols = int(ref.padded_shape[2]), int(ref.padded_shape[3])
    dy = ttnn.Tensor(
        rng.standard_normal(
            (1, 1, fwd.batch * fwd.out_h * fwd.out_w, fwd.out_channels)
        ).astype(np.float32),
        FP32,
    ).to(ttnn.ROW_MAJOR_LAYOUT).to(device)

    accepted, tried, want = [], [], None
    for layout in candidate_layouts(rows, cols, fwd.in_channels, fwd.out_channels, k, k):
        tried.append(layout.cp)
        spec = TransposeSpec(layout=layout, shard_layout=shard_layout,
                             conv_cfg=conv_cfg, **fields)
        if want is None:
            out = transpose_dispatch(dy, ref, spec, device)
            want = to_host(out)
            ttnn.deallocate(out)
            ttnn.deallocate(ref)
        try:
            derived = transpose_weight(probe_master, pspec, spec)
        except (ValueError, RuntimeError):
            continue
        out = transpose_dispatch(dy, derived, spec, device)
        got = to_host(out)
        ttnn.deallocate(out)
        ttnn.deallocate(derived)
        if got.shape == want.shape and np.array_equal(got, want):
            accepted.append(spec)
    ttnn.deallocate(dy)
    ttnn.deallocate(probe_master)
    if len(accepted) != 1:
        raise ValueError(
            f"dX weight transform could not be matched to "
            f"ttnn.prepare_conv_transpose2d_weights at geometry {fields}. ttnn "
            f"reports a {rows}x{cols} prepared transpose weight; of the "
            f"candidate channel strides {tried}, "
            f"{[s.layout.cp for s in accepted]} reproduced ttnn's own weight's "
            f"gradient bit-for-bit. Exactly one must. The device transform in "
            f"engine/conv/transform.transpose_weight no longer matches this "
            f"ttnn build's prepared transpose layout, and conv_transpose2d "
            f"would return a gradient that is wrong without raising."
        )
    return accepted[0]


def dw_to_prepared(mm_out, layout: PreparedLayout):
    """`(R*S*C, K)` weight gradient -> the prepared master's matrix layout.

    The dW matmul already emits its rows in the prepared layout's
    (r, s, in_channel) order and its columns in out-channel order, so only the
    channel padding has to be reinstated. The padding rows and columns stay
    zero, which is what keeps the optimizer from drifting the master's padding
    away from zero.
    """
    r, s = layout.kernel_h, layout.kernel_w
    w = ttnn.reshape(tile(mm_out), (r, s, layout.in_channels, layout.out_channels))
    if layout.in_channels != layout.cp or layout.out_channels != layout.cols:
        w = ttnn.pad(
            w,
            ((0, 0), (0, 0), (0, layout.cp - layout.in_channels),
             (0, layout.cols - layout.out_channels)),
            value=0.0,
        )
    w = ttnn.reshape(w, (r, layout.packed_rows, layout.cols))
    if not layout.is_tight:
        w = ttnn.pad(w, ((0, 0), (0, layout.row_block - layout.packed_rows), (0, 0)),
                     value=0.0)
    return ttnn.reshape(w, layout.shape)


def verify_dw_layout(layout: PreparedLayout, device) -> None:
    """Probe `dw_to_prepared` against the host packing, bit-exact.

    Construction-time only. A silently mis-shaped weight gradient trains on
    permuted gradients, which looks like a bad hyperparameter rather than a bug.
    """
    k, c = layout.out_channels, layout.in_channels
    w = (np.arange(k * c * layout.kernel_h * layout.kernel_w, dtype=np.float32)
         + 1.0).reshape(k, c, layout.kernel_h, layout.kernel_w)
    want = layout.pack(w).reshape(layout.rows, layout.cols)
    # What the dW matmul emits: rows (r, s, in_channel), columns out_channel.
    probe = np.ascontiguousarray(
        layout.blocked(w)[:, :, :c, :k].reshape(-1, k))
    src = ttnn.Tensor(probe, FP32).to(ttnn.TILE_LAYOUT).to(device)
    got = to_host(dw_to_prepared(src, layout))
    if not np.array_equal(got.reshape(layout.rows, layout.cols), want):
        raise ValueError(
            f"dW layout transform is not bit-exact at prepared layout "
            f"{layout.shape}: engine/conv/transform.dw_to_prepared does not "
            f"invert PreparedLayout.pack on this ttnn build."
        )
