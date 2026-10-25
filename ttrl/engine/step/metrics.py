"""Device-to-host metric readback in one round trip."""

from __future__ import annotations

import ttnn

from ttrl.engine.autograd import Tensor
from ttrl.engine.tensor import to_host

__all__ = ["read_metrics"]

_SAFE_DTYPES = (ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b)


def _element_count(data) -> int:
    n = 1
    for dim in data.shape:
        n *= int(dim)
    return n


def read_metrics(metrics: dict | None) -> dict:
    """Download every tensor-valued metric, packing the scalars into one trip.

    Each SCALAR metric is typecast to fp32, flattened, and concatenated into
    one device tensor which is downloaded once, then split back into the
    original per-key shapes on the host. typecast, reshape and concat relocate
    values without recomputing any, so this is bit-identical to reading each
    metric with its own `.numpy()`. Non-tensor values never touched the device
    and pass through unchanged.

    A PER-SAMPLE metric is read on its own instead, and that is load-bearing.
    A metric with more than one element is tile-padded on device (a batch of 8
    occupies a 32x32 tile), and when the tensor is a captured trace's output
    buffer the packing reads the padding rather than the values: every element
    but the first came back zero on every replay after the first, while a
    direct read of the same tensor was correct, and a full device synchronize
    before the read did not change it. That is how a prioritized run's TD
    errors reached the priority table as zeros in `traced_fwd_bwd` while
    `eager` was correct, which silently changes which transitions the run
    samples. One extra round trip per step buys a metric that is the number it
    claims to be.
    """
    if not metrics:
        return {}
    tensor_items = [(k, v) for k, v in metrics.items() if isinstance(v, Tensor)]
    out = {k: v for k, v in metrics.items() if not isinstance(v, Tensor)}
    if not tensor_items:
        return out
    packable = []
    for k, v in tensor_items:
        if _element_count(v.data) == 1:
            packable.append((k, v))
        else:
            out[k] = v.numpy()
    tensor_items = packable
    if not tensor_items:
        return out
    if len(tensor_items) == 1:
        (k, v), = tensor_items
        out[k] = v.numpy()
        return out

    flat_parts = []
    shapes: list[tuple[int, ...]] = []
    sizes: list[int] = []
    for k, v in tensor_items:
        d = v.data
        if d.dtype not in _SAFE_DTYPES:
            raise RuntimeError(
                f"read_metrics: metric {k!r} has dtype {d.dtype!r}, which is outside "
                f"the float family {[str(t) for t in _SAFE_DTYPES]} this packing's "
                "fp32 typecast is exact for. An int32 value above 2**24 would be "
                "silently rounded, so add explicit handling for this metric instead."
            )
        if d.dtype != ttnn.float32:
            d = ttnn.typecast(d, ttnn.float32)
        shape = tuple(int(s) for s in d.shape)
        n = 1
        for s in shape:
            n *= s
        if n <= 0:
            raise ValueError(
                f"read_metrics: metric {k!r} has 0 elements (shape {shape}), and "
                "ttnn.concat/reshape do not support zero-sized device dims."
            )
        shapes.append(shape)
        sizes.append(n)
        flat_parts.append(ttnn.reshape(d, (1, n)))

    first = flat_parts[0]
    for (k, _), part in zip(tensor_items, flat_parts):
        if part.device() != first.device():
            raise RuntimeError(
                f"read_metrics: metric {k!r} lives on device {part.device()!r}, which "
                f"differs from {first.device()!r}; ttnn.concat requires every input on "
                "the same device."
            )
        if part.layout != first.layout or part.storage_type() != first.storage_type():
            raise RuntimeError(
                f"read_metrics: metric {k!r} has layout={part.layout!r} / "
                f"storage={part.storage_type()!r}, which differs from "
                f"layout={first.layout!r} / storage={first.storage_type()!r}; "
                "ttnn.concat requires compatible memory layouts."
            )

    packed = to_host(ttnn.concat(flat_parts, dim=-1))
    offset = 0
    for (k, _), shape, n in zip(tensor_items, shapes, sizes):
        out[k] = packed[0, offset : offset + n].reshape(shape)
        offset += n
    return out
