"""Conv output geometry and the dX capacity bound. Pure python.

Nothing here imports ttnn, so the output-geometry rules and the arithmetic that
refuses an unbuildable dX are testable on a laptop with no wheel and no chip.
The prepared-weight packing lives in `packing`, for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = [
    "MEASURED_DX_ROWS_PER_CORE",
    "TILE",
    "ConvDoesNotFit",
    "ConvGeometry",
    "check_dx_capacity",
    "conv_out_size",
    "dx_rows_per_core",
    "largest_dx_batch",
    "max_dx_rows_per_core",
    "transpose_input_extent",
]


#: Tile height. The sharded row count every ttnn activation is padded up to.
#:
#: Not a measurement and not Blackhole-specific: it is ttnn.TILE_SIZE, which is
#: 32 on every arch ttnn 0.80 supports. It is spelled out rather than imported
#: so this module stays ttnn-free; DeviceProfile probes `tile_shape` at runtime
#: and is the place to compare against if a part ever ships another tile.
TILE = 32

#: Largest tile-aligned per-core row count of the dX activation at which
#: ttnn 0.80's conv_transpose2d weight preparation still terminates, per arch.
#:
#: CALIBRATED, NOT DERIVED, and the reason the check below is approximate.
#: ttnn builds the transpose convolution's sliding-window metadata per core on
#: host, inside prepare_conv_transpose2d_weights, and above some per-core row
#: count that work stops being bounded: it either spins CPU indefinitely or
#: segfaults in generate_shard_boundaries.
#:
#: MEASURED on Blackhole (110-core grid), ttnn 0.80, dX of conv1 (4->32, k8,
#: s4, 84x84) and of conv2 (32->64, k4, s2, 20x20), which have different
#: kernels, strides and channel counts:
#:
#:   conv1 batch 160, 12064 rows/core: returns in seconds
#:   conv2 batch 1600, 12320 rows/core: SIGSEGV
#:   conv1 batch 176,  13264 rows/core: no return in 110 s, killed
#:   conv1 batch 192,  14464 rows/core: no return in 120 s, killed, under both
#:                                      height sharding and ttnn's own choice
#:
#: Two unrelated geometries agree to within 2%, so the per-core row count is
#: the quantity that governs it. The bound is the largest value measured to
#: terminate, so the unknown 2% band between 12064 and 12320 is refused. A
#: refusal is therefore occasionally wrong in the direction of a bad error
#: message, never in the direction of a hang.
#:
#: Keyed by arch and deliberately holding one entry. The quantity is a ttnn
#: host-side shard-boundary behaviour, not a capacity, so it does not scale
#: with core count or L1 size and there is no honest way to derive a Wormhole
#: or Quasar value from the Blackhole one. An arch that is not in this table is
#: refused rather than guessed.
MEASURED_DX_ROWS_PER_CORE = {"blackhole": 12064}


def max_dx_rows_per_core(arch: str) -> int:
    """The measured dX row ceiling for `arch`, or a refusal naming it."""
    try:
        return MEASURED_DX_ROWS_PER_CORE[arch]
    except KeyError:
        raise ValueError(
            f"conv2d: the input-gradient (dX) row ceiling has not been measured "
            f"on arch {arch!r}; it is calibrated on "
            f"{sorted(MEASURED_DX_ROWS_PER_CORE)} only. The ceiling is not a "
            f"capacity: it is the per-core row count above which ttnn's own "
            f"conv_transpose2d weight preparation stops terminating, so it "
            f"cannot be scaled from another part's core count or L1 size, and a "
            f"derived-looking number would hang or segfault this process "
            f"instead of refusing. To establish it on {arch!r}, repeat the "
            f"measurement that produced the Blackhole entry: build one conv "
            f"layer with needs_input_grad=True, sweep the batch size upward "
            f"under a per-probe timeout, record dx_rows_per_core() at each "
            f"batch, and enter the LARGEST value that still returns -- not the "
            f"smallest that fails -- so the unmeasured band stays refused. "
            f"Repeat at a second geometry with a different kernel, stride and "
            f"channel count and take the lower of the two. Until then, build "
            f"image-observation layers with needs_input_grad=False, which needs "
            f"no dX at all, or run on "
            f"{sorted(MEASURED_DX_ROWS_PER_CORE)}."
        ) from None


def conv_out_size(size: int, kernel: int, stride: int, padding: int, *, axis: str) -> int:
    """`(size + 2*padding - kernel) // stride + 1`, with the geometry checked."""
    if kernel <= 0 or stride <= 0 or padding < 0:
        raise ValueError(
            f"conv2d: kernel and stride must be > 0 and padding >= 0, got "
            f"kernel={kernel}, stride={stride}, padding={padding}"
        )
    span = size + 2 * padding - kernel
    if span < 0:
        raise ValueError(
            f"conv2d: input {axis}={size} with padding={padding} is smaller than "
            f"kernel={kernel}; output would be empty"
        )
    if span % stride != 0:
        # dX is conv_transpose2d(dY), whose output is (out-1)*stride + kernel
        # - 2*padding, equal to `size` only when the division is exact. For a
        # ragged geometry the trailing rows feed no output element and dX comes
        # back short, so refuse the geometry instead of training on a truncated
        # gradient.
        raise ValueError(
            f"conv2d: ragged geometry on {axis}: (size + 2*padding - kernel) = "
            f"{span} is not a multiple of stride={stride} (size={size}, "
            f"kernel={kernel}, padding={padding}). The input gradient is only "
            f"the true adjoint for exactly divisible strides. Pad the "
            f"observation to a divisible size or change stride."
        )
    return span // stride + 1


class ConvDoesNotFit(ValueError):
    """This layer does not fit on the chip at its batch size.

    Raised only by the capacity checks, so a caller that sized the batch can
    name its own field without relabelling unrelated refusals.
    """


@dataclass(frozen=True)
class ConvGeometry:
    """Everything ttnn.conv2d needs to describe one convolution, derived from
    the input shape rather than from any benchmark's constants.

    `fold > 1` is the space-to-depth spelling: the same arithmetic over an
    input whose spatial extent is divided by `fold` and whose channel count is
    multiplied by fold**2. The convolution computed is identical; see
    `s2d_flat_permutation`.
    """

    batch: int
    in_channels: int
    in_h: int
    in_w: int
    out_channels: int
    kernel: int
    stride: int = 1
    padding: int = 0
    fold: int = 1

    def __post_init__(self) -> None:
        if min(self.batch, self.in_channels, self.in_h, self.in_w, self.out_channels) <= 0:
            raise ValueError(f"conv2d: batch, channels and input extent must be > 0, got {self}")
        if self.fold < 1:
            raise ValueError(f"conv2d: fold must be >= 1, got {self.fold}")
        if self.fold > 1:
            if self.stride != self.fold or self.padding != 0:
                raise ValueError(
                    f"conv2d: space-to-depth needs stride == fold and padding == 0, got "
                    f"stride={self.stride} fold={self.fold} padding={self.padding}"
                )
            for name, size in (("kernel", self.kernel), ("in_h", self.in_h), ("in_w", self.in_w)):
                if size % self.fold:
                    raise ValueError(
                        f"conv2d: fold={self.fold} must divide {name}={size}"
                    )
        # Validates the geometry even when the caller never reads out_h/out_w.
        _ = (self.out_h, self.out_w)

    @property
    def out_h(self) -> int:
        return conv_out_size(self.in_h, self.kernel, self.stride, self.padding, axis="h")

    @property
    def out_w(self) -> int:
        return conv_out_size(self.in_w, self.kernel, self.stride, self.padding, axis="w")

    @property
    def out_shape(self) -> tuple[int, int, int]:
        """(C, H, W) of this layer's output, channels first, as ObsSpec spells it."""
        return (self.out_channels, self.out_h, self.out_w)

    @property
    def in_numel(self) -> int:
        return self.batch * self.in_channels * self.in_h * self.in_w

    # The convolution actually dispatched: identical arithmetic, folded shape.
    @property
    def dispatch(self) -> "ConvGeometry":
        if self.fold == 1:
            return self
        f = self.fold
        return ConvGeometry(
            batch=self.batch, in_channels=self.in_channels * f * f,
            in_h=self.in_h // f, in_w=self.in_w // f,
            out_channels=self.out_channels, kernel=self.kernel // f,
            stride=1, padding=0,
        )


def transpose_input_extent(geom: "ConvGeometry") -> tuple[int, int]:
    """(height, width) of the dilated, padded dY that dX convolves.

    conv_transpose2d is an ordinary convolution over dY zero-inserted by
    `stride` and padded by `kernel - 1 - padding`, and that is the activation
    whose sliding-window metadata ttnn builds, so it is the extent the capacity
    check is written against.
    """
    disp = geom.dispatch
    pad = disp.kernel - 1 - disp.padding
    return ((disp.out_h - 1) * disp.stride + 1 + 2 * pad,
            (disp.out_w - 1) * disp.stride + 1 + 2 * pad)


def dx_rows_per_core(geom: "ConvGeometry", cores: int) -> int:
    """Tile-aligned rows of the dX activation each core holds."""
    if cores < 1:
        raise ValueError(f"cores must be >= 1, got {cores}")
    height, width = transpose_input_extent(geom)
    rows = geom.dispatch.batch * height * width
    per_core = -(-rows // cores)
    return -(-per_core // TILE) * TILE


def largest_dx_batch(geom: "ConvGeometry", cores: int, arch: str) -> int:
    """The largest batch whose dX passes `check_dx_capacity` at this geometry."""
    ceiling = max_dx_rows_per_core(arch)
    height, width = transpose_input_extent(geom)
    batch = ceiling * cores // (height * width)
    # Tile alignment rounds the per-core count up, so step back to the first
    # batch that actually passes rather than trusting the division.
    while batch > 1 and dx_rows_per_core(
            replace(geom.dispatch, batch=batch), cores) > ceiling:
        batch -= 1
    return max(batch, 0)


def check_dx_capacity(geom: "ConvGeometry", cores: int, l1_per_core: int,
                      arch: str) -> None:
    """Refuse a dX geometry that ttnn 0.80 would not fail promptly on.

    Checked arithmetically, before the first ttnn call, because the failure it
    guards against happens INSIDE that call: above this size
    prepare_conv_transpose2d_weights spins or segfaults, and no probe-and-catch
    can recover from either. Geometries that merely do not fit in L1 are left
    to the placement search, which fails promptly with its own message.

    APPROXIMATE in two ways, both stated in the message it raises: the row
    ceiling is calibrated on measurement rather than derived from ttnn's
    allocator (see MEASURED_DX_ROWS_PER_CORE) and only on Blackhole, which is
    why `arch` is required and an unmeasured arch is refused, and the byte
    figure counts only the activation shard, which is a lower bound on the L1 a
    core needs.
    """
    ceiling = max_dx_rows_per_core(arch)
    rows = dx_rows_per_core(geom, cores)
    if rows <= ceiling:
        return
    disp = geom.dispatch
    height, width = transpose_input_extent(geom)
    activation_bytes = rows * disp.in_channels * 4
    raise ConvDoesNotFit(
        f"conv2d: the input gradient (dX) is too large to compute at batch "
        f"{geom.batch} for this layer ({disp.in_channels}->{disp.out_channels}, "
        f"kernel {disp.kernel}, stride {disp.stride}, padding {disp.padding}, "
        f"input {disp.in_h}x{disp.in_w}). dX convolves a {height}x{width} "
        f"dilated gradient, which is {rows} rows per core over {cores} cores "
        f"against a ceiling of {ceiling} measured on {arch}, and at least "
        f"{activation_bytes} bytes of the {l1_per_core} bytes of L1 a core has "
        f"(activation shard only, so a lower bound). The largest batch whose dX "
        f"fits at this geometry is {largest_dx_batch(geom, cores, arch)}. Above the "
        f"ceiling ttnn 0.80's conv_transpose2d weight preparation does not "
        f"return or segfaults, so this is refused rather than attempted. Build "
        f"this layer with needs_input_grad=False if nothing upstream of it "
        f"needs a gradient."
    )
