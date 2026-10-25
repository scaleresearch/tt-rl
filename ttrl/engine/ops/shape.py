"""Shape ops: concat, split and reshape."""

from __future__ import annotations

import ttnn

from ttrl.engine.autograd import Tensor, _detached, _needs


def concat(tensors, dim: int = -1) -> Tensor:
    """Concatenate along `dim`. Backward slices the incoming gradient back to
    each operand's own extent along that axis.
    """
    tensors = tuple(tensors)
    if not tensors:
        raise ValueError("concat needs at least one tensor")
    rank = len(tensors[0].shape)
    axis = dim if dim >= 0 else dim + rank
    for t in tensors:
        if len(t.shape) != rank:
            raise ValueError(
                f"concat operands must share a rank, got {[t.shape for t in tensors]}"
            )
    y = ttnn.concat([t.data for t in tensors], dim=axis)
    if not _needs(*tensors):
        return _detached(y)
    # Captured at build time: the backward closure must not reach back through
    # the output node to recover the operand extents.
    extents = [int(t.shape[axis]) for t in tensors]
    operands = tensors
    out = Tensor(y, True, operands, "concat")

    def _backward(grad):
        out_shape = tuple(int(d) for d in grad.shape)
        offset = 0
        for operand, extent in zip(operands, extents):
            if operand.requires_grad:
                begins = [0] * rank
                ends = list(out_shape)
                begins[axis] = offset
                ends[axis] = offset + extent
                operand._accumulate(ttnn.slice(grad, begins, ends, [1] * rank))
            offset += extent

    out._backward = _backward
    return out


def reshape(a: Tensor, shape) -> Tensor:
    shape = tuple(int(d) for d in shape)
    in_shape = tuple(int(d) for d in a.shape)
    y = ttnn.reshape(a.data, shape)
    if not _needs(a):
        return _detached(y)
    out = Tensor(y, True, (a,), "reshape")

    def _backward(grad):
        if not a.requires_grad:
            return
        a._accumulate(ttnn.reshape(grad, in_shape))

    out._backward = _backward
    return out


def split(a: Tensor, sizes, dim: int = -1) -> tuple[Tensor, ...]:
    """Cut `a` into consecutive pieces of `sizes` along `dim`.

    The inverse of `concat`, and the forward direction `concat` already needs
    in its own backward. A gated recurrent cell wants it to compute its three
    gates from one wide matmul instead of three narrow ones, which is the
    difference between two dispatches and six in a per-timestep loop.

    Each piece is its own tape node, so a caller that uses only some of them
    routes gradient from only those; the pieces it drops contribute zero, which
    is what `_accumulate` summing independent contributions already gives.
    """
    sizes = tuple(int(s) for s in sizes)
    rank = len(a.shape)
    axis = dim if dim >= 0 else dim + rank
    if not 0 <= axis < rank:
        raise ValueError(
            f"split got dim={dim} for a rank-{rank} tensor of shape "
            f"{tuple(a.shape)}; dim must index an existing axis."
        )
    extent = int(a.shape[axis])
    if any(s <= 0 for s in sizes) or sum(sizes) != extent:
        raise ValueError(
            f"split sizes {sizes} must be positive and sum to {extent}, the "
            f"extent of axis {axis} in shape {tuple(a.shape)}; got sum "
            f"{sum(sizes)}."
        )
    in_shape = tuple(int(d) for d in a.shape)
    pieces = []
    offset = 0
    for size in sizes:
        begins = [0] * rank
        ends = list(in_shape)
        begins[axis] = offset
        ends[axis] = offset + size
        y = ttnn.slice(a.data, begins, ends, [1] * rank)
        if not _needs(a):
            pieces.append(_detached(y))
            offset += size
            continue
        out = Tensor(y, True, (a,), "split")
        # Bound per piece: the closure must not read its own output node to
        # recover where in the input this piece came from.
        out._backward = _split_backward(a, in_shape, axis, offset, size)
        pieces.append(out)
        offset += size
    return tuple(pieces)


def _split_backward(a: Tensor, in_shape, axis: int, offset: int, size: int):
    """Scatter one piece's gradient back into the input's full extent.

    Zero-pad-and-concat rather than `ttnn.pad`, which refuses front padding on
    a device tile layout, so every piece but the first would be unroutable.

    The padding is built by zeroing a slice of the input rather than by
    `ttnn.zeros`, which is a host-to-device write and so forbidden inside a
    trace capture: with `ttnn.zeros` here, a recurrent unroll backward failed
    capture with "Writes are not supported during trace capture". Slicing and
    multiplying by zero is pure device work, and the complementary slices of
    the input are exactly the shapes the concat needs.
    """
    rank = len(in_shape)
    before = (0, offset)
    after = (offset + size, in_shape[axis])

    def _zeros(span, dtype):
        begins = [0] * rank
        ends = list(in_shape)
        begins[axis], ends[axis] = span
        zeroed = ttnn.mul(ttnn.slice(a.data, begins, ends, [1] * rank), 0.0)
        if zeroed.dtype != dtype:
            zeroed = ttnn.typecast(zeroed, dtype)
        return zeroed

    def _backward(grad):
        if not a.requires_grad:
            return
        full = grad
        if before[1] > before[0]:
            full = ttnn.concat([_zeros(before, grad.dtype), full], dim=axis)
        if after[1] > after[0]:
            full = ttnn.concat([full, _zeros(after, grad.dtype)], dim=axis)
        a._accumulate(full)

    return _backward
