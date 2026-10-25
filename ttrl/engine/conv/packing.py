"""Prepared-weight packing and the space-to-depth fold, as index arithmetic.

Pure python and numpy: nothing here imports ttnn, so the layout the device
weights are built in is testable on a laptop with no wheel and no chip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PreparedLayout",
    "candidate_layouts",
    "fold_input_numpy",
    "fold_weight_numpy",
    "s2d_flat_permutation",
    "unfold_weight_numpy",
]


@dataclass(frozen=True)
class PreparedLayout:
    """ttnn.conv2d's prepared weight layout, as index arithmetic.

    The prepared weight is a `[1, 1, rows, cols]` matrix holding the
    `(K, C, R, S)` master as

        out[r * row_block + s * cp + c, k] = w[k, c, r, s]

    with every other element zero. `cp` is the padded input-channel count,
    `row_block` the padded extent of one kernel row's `(s, c)` axis, and
    `cols` the padded output-channel count. None of the three is derivable
    from the geometry (cp is 8 for 4 input channels and 64 for 64, and
    row_block is 32 for a 3x3 kernel over 4 channels where S*cp is 24), so all
    three are solved for once per layer from a probe weight whose elements are
    distinct, and the packing is then verified bit-exact against
    ttnn.prepare_conv_weights.
    """

    rows: int
    cols: int
    cp: int
    row_block: int
    out_channels: int
    in_channels: int
    kernel_h: int
    kernel_w: int

    def __post_init__(self) -> None:
        if self.rows != self.kernel_h * self.row_block:
            raise ValueError(
                f"prepared weight layout rows={self.rows} is not kernel height "
                f"{self.kernel_h} times row_block={self.row_block}"
            )
        if self.row_block < self.kernel_w * self.cp:
            raise ValueError(
                f"prepared weight layout row_block={self.row_block} cannot hold "
                f"kernel width {self.kernel_w} times cp={self.cp}"
            )
        if self.cp < self.in_channels or self.cols < self.out_channels:
            raise ValueError(
                f"prepared weight layout {self.rows}x{self.cols} (cp={self.cp}) is "
                f"smaller than the weight it must hold "
                f"({self.out_channels}x{self.in_channels}x{self.kernel_h}x{self.kernel_w})"
            )

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (1, 1, self.rows, self.cols)

    @property
    def packed_rows(self) -> int:
        """Rows of one kernel row actually holding weight, i.e. S * cp."""
        return self.kernel_w * self.cp

    @property
    def is_tight(self) -> bool:
        """True when the layout has no per-kernel-row padding."""
        return self.row_block == self.packed_rows

    def blocked(self, w: np.ndarray) -> np.ndarray:
        """`(K, C, R, S)` -> `(R, S, cp, cols)`, the channel-padded core."""
        k, c, r, s = w.shape
        if (k, c, r, s) != (self.out_channels, self.in_channels, self.kernel_h, self.kernel_w):
            raise ValueError(
                f"weight shape {w.shape} does not match this layout's "
                f"{(self.out_channels, self.in_channels, self.kernel_h, self.kernel_w)}"
            )
        padded = np.zeros((self.cols, self.cp, r, s), np.float32)
        padded[:k, :c] = w
        return np.ascontiguousarray(padded.transpose(2, 3, 1, 0))

    def pack(self, w: np.ndarray) -> np.ndarray:
        """`(K, C, R, S)` -> `[1, 1, rows, cols]`, the prepared matrix."""
        core = self.blocked(w).reshape(self.kernel_h, self.packed_rows, self.cols)
        out = np.zeros((self.kernel_h, self.row_block, self.cols), np.float32)
        out[:, : self.packed_rows] = core
        return np.ascontiguousarray(out.reshape(1, 1, self.rows, self.cols))

    def unpack(self, m: np.ndarray) -> np.ndarray:
        """Exact inverse of `pack`, dropping the padding."""
        v = np.asarray(m, np.float32).reshape(self.kernel_h, self.row_block, self.cols)
        core = v[:, : self.packed_rows].reshape(
            self.kernel_h, self.kernel_w, self.cp, self.cols)
        return np.ascontiguousarray(
            core.transpose(3, 2, 0, 1)[: self.out_channels, : self.in_channels])


def s2d_flat_permutation(in_channels: int, kernel: int, fold: int) -> np.ndarray:
    """Flat within-output-channel index map for the space-to-depth weight fold.

    With `W` in master `(O, C, KH, KW)` layout and `Wf` folded to
    `(O, C*fold*fold, KH/fold, KW/fold)`,
    `Wf.reshape(O, -1)[:, i] == W.reshape(O, -1)[:, idx[i]]`. The folded
    channel axis is ordered (row offset in block, column offset in block,
    original channel), which is the order `fold_input_numpy` produces.
    """
    if fold <= 0 or kernel % fold != 0:
        raise ValueError(
            f"s2d_flat_permutation: fold must divide kernel, got kernel={kernel} fold={fold}"
        )
    k = kernel // fold
    a, b, c, p, q = np.meshgrid(
        np.arange(fold), np.arange(fold), np.arange(in_channels),
        np.arange(k), np.arange(k), indexing="ij",
    )
    master = c * (kernel * kernel) + (fold * p + a) * kernel + (fold * q + b)
    return master.reshape(-1).astype(np.int64)


def fold_weight_numpy(w: np.ndarray, fold: int) -> np.ndarray:
    """`(O, C, KH, KW)` -> `(O, C*fold*fold, KH/fold, KW/fold)`."""
    o, c, kh, kw = w.shape
    idx = s2d_flat_permutation(c, kh, fold)
    return np.ascontiguousarray(
        w.reshape(o, -1)[:, idx].reshape(o, c * fold * fold, kh // fold, kw // fold)
    )


def unfold_weight_numpy(wf: np.ndarray, fold: int, in_channels: int, kernel: int) -> np.ndarray:
    """Exact inverse of `fold_weight_numpy`."""
    idx = s2d_flat_permutation(in_channels, kernel, fold)
    inv = np.empty_like(idx)
    inv[idx] = np.arange(idx.size)
    return np.ascontiguousarray(
        wf.reshape(wf.shape[0], -1)[:, inv].reshape(-1, in_channels, kernel, kernel)
    )


def fold_input_numpy(nhwc: np.ndarray, fold: int) -> np.ndarray:
    """`(N, H, W, C)` -> `(N, H/fold, W/fold, C*fold*fold)`, channel order
    matching `s2d_flat_permutation`."""
    n, h, w, c = nhwc.shape
    r = nhwc.reshape(n, h // fold, fold, w // fold, fold, c).transpose(0, 1, 3, 2, 4, 5)
    return np.ascontiguousarray(r.reshape(n, h // fold, w // fold, c * fold * fold))


def candidate_layouts(rows: int, cols: int, out_channels: int, in_channels: int,
                      kernel_h: int, kernel_w: int):
    """Every prepared layout consistent with a `(rows, cols)` padded extent.

    `rows` and `cols` are metadata, readable from ttnn's prepared tensor. `cp`,
    the padded input-channel stride inside each kernel row, is not: it is only
    visible in the padded bytes, which ttnn 0.80 exposes through torch and
    nothing else (see `weights.prepare_master`). So the layouts the metadata
    permits are enumerated here, as integer arithmetic, and the convolution
    itself decides which one is real.
    """
    if kernel_h < 1 or kernel_w < 1:
        raise ValueError(f"kernel extents must be >= 1, got {kernel_h}x{kernel_w}")
    if rows % kernel_h:
        raise ValueError(
            f"prepared weight rows={rows} is not a multiple of kernel height "
            f"{kernel_h}, so the layout is not a per-kernel-row block matrix"
        )
    row_block = rows // kernel_h
    for cp in range(in_channels, row_block // kernel_w + 1):
        try:
            yield PreparedLayout(
                rows=rows, cols=cols, cp=cp, row_block=row_block,
                out_channels=out_channels, in_channels=in_channels,
                kernel_h=kernel_h, kernel_w=kernel_w,
            )
        except ValueError:
            continue
