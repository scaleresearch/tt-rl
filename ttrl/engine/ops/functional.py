"""Composed building blocks: huber, log_softmax, entropy, one-hot gather,
masks and the non-finite guard.

Everything differentiable here is composed from the primitive ops, each of
which already has a gradchecked backward. The compositions are deliberate:
`huber`'s `minimum` is the relu identity (`ttnn.min_bw` is fp32-correct on 0.80
but wants a device tensor where `huber` passes a float; see
`ttrl.engine.ops.select` for the measurement) and
`ttnn.moreh_logsoftmax_backward` sits outside the tape and needs the forward
output threaded by hand. The one-hot gather exists because our indices are
float32 (`argmax_dim_detached`), while
`ttnn.gather` and `ttnn.scatter_add` take UINT16 or UINT32 index tensors and
have no bound backward.
"""

from __future__ import annotations

import numpy as np
import ttnn

from ttrl.engine.autograd import Tensor
from ttrl.engine.ops.elementwise import add, mul, sub
from ttrl.engine.ops.reduce import max_dim_detached, sum_dim
from ttrl.engine.ops.select import minimum
from ttrl.engine.ops.unary import abs_, exp, log
from ttrl.engine.tensor import to_host

#: Lane-number constants for `one_hot_device`, keyed by (device, batch,
#: num_classes, dtype). Cached rather than rebuilt per call because a freshly
#: uploaded tensor is a new device address each call, which a replayed trace
#: does not see. Populated on the first (warm-up, pre-capture) call and only
#: read thereafter.
_LANE_CACHE: dict = {}


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------

def apply_done_mask(value: Tensor, done: Tensor) -> Tensor:
    """value * (1 - done): zero a bootstrap value at episode boundaries.

    `done` is {0, 1}-valued and shaped like `value`; broadcastable is fine,
    since `mul`'s backward already reduces over broadcast dims.
    """
    return mul(value, sub(1.0, done))


masks = apply_done_mask


# --------------------------------------------------------------------------
# huber
# --------------------------------------------------------------------------

def huber(pred: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
    """Elementwise Huber (smooth-L1) loss, differentiable, unreduced.

    huber(e) = 0.5 * e^2                    for |e| <= delta
             = delta * (|e| - 0.5 * delta)  otherwise

    Composed as the clipped-quadratic identity so it is one smooth expression
    rather than a branch. Callers reduce it themselves, which lets them apply
    a mask before reducing.
    """
    absdiff = abs_(sub(pred, target))
    quad = minimum(absdiff, delta)
    lin = sub(absdiff, quad)
    return add(mul(mul(quad, quad), 0.5), mul(lin, delta))


# --------------------------------------------------------------------------
# one-hot gather
# --------------------------------------------------------------------------


def _lane_numbers(batch: int, num_classes: int, device, dtype) -> Tensor:
    key = (id(device), int(batch), int(num_classes), dtype)
    lanes = _LANE_CACHE.get(key)
    if lanes is None:
        rows = np.tile(np.arange(num_classes, dtype=np.float32), (batch, 1))
        lanes = Tensor.from_numpy(rows, device, requires_grad=False, dtype=dtype)
        _LANE_CACHE[key] = lanes
    return lanes


def purge_lane_caches(device) -> None:
    """Drop `_LANE_CACHE` entries belonging to `device`, before it closes.

    The cache holds device-resident tensors under an `id(device)` key, and
    `id()` is reused once the device object is freed.
    """
    did = id(device)
    for key in [k for k in _LANE_CACHE if k[0] == did]:
        del _LANE_CACHE[key]


def one_hot_host(indices, num_classes: int, device, dtype=None) -> Tensor:
    """(B,) or (B, 1) host integer ids -> (B, num_classes) one-hot Tensor.

    Uploads a fresh tensor per call, so a caller inside a traced region must
    instead bind the one-hot once as a step input and refresh it in place.
    """
    idx = np.asarray(indices).reshape(-1).astype(np.int64)
    batch = idx.shape[0]
    onehot_np = np.zeros((batch, num_classes), dtype=np.float32)
    onehot_np[np.arange(batch), idx] = 1.0
    kwargs = {} if dtype is None else {"dtype": dtype}
    return Tensor.from_numpy(onehot_np, device, requires_grad=False, **kwargs)


def one_hot_device(indices: Tensor, num_classes: int) -> Tensor:
    """(B, 1) device-resident integer-valued float ids -> (B, num_classes)
    one-hot, with no host round trip.

        one_hot[b, c] = 1 iff (c - indices[b]) == 0

    which is `ttnn.eqz(lanes - indices)`. The comparison is exact: both
    operands are small non-negative integers held in fp32. Keeping this on
    device is what lets Double-DQN action selection run inside a captured step.
    """
    lanes = _lane_numbers(int(indices.shape[0]), num_classes, indices.device, indices.dtype)
    return Tensor(ttnn.eqz(ttnn.subtract(lanes.data, indices.data)), requires_grad=False)


def gather_onehot(values: Tensor, onehot: Tensor) -> Tensor:
    """sum(values * onehot, dim=-1, keepdim=True), differentiable in `values`.

    `onehot` carries no gradient, so `mul`'s backward routes gradient into
    `values` only, which is exactly Q(s,a)'s gather backward.
    """
    return sum_dim(mul(values, onehot), dim=-1, keepdim=True)


def one_hot_gather(values: Tensor, indices, num_classes: int) -> Tensor:
    """values[b, indices[b]] for each row b, differentiable in `values`.

    `indices` as a numpy array or list uploads a one-hot per call and is not
    trace-legal. `indices` as a (B, 1) Tensor of integer-valued floats stays
    entirely on device. Returns (B, 1).
    """
    if isinstance(indices, Tensor):
        batch = int(indices.shape[0])
        onehot = one_hot_device(indices, num_classes)
    else:
        idx = np.asarray(indices).reshape(-1)
        batch = int(idx.shape[0])
        onehot = one_hot_host(idx, num_classes, values.device, dtype=values.dtype)
    if int(values.shape[0]) != batch:
        raise ValueError(
            f"one_hot_gather: values has batch {values.shape[0]}, indices has {batch}"
        )
    return gather_onehot(values, onehot)


# --------------------------------------------------------------------------
# log_softmax / entropy
# --------------------------------------------------------------------------

def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable log-softmax, differentiable.

    log_softmax(x) = (x - m) - log(sum(exp(x - m)))   where m = max(x, dim)

    `m` is a detached constant: shifting by a constant does not change the
    output, so its gradient contribution is exactly zero.
    """
    shifted = sub(x, max_dim_detached(x, dim=dim, keepdim=True))
    sum_exp = sum_dim(exp(shifted), dim=dim, keepdim=True)
    return sub(shifted, log(sum_exp))


def entropy(logits: Tensor, dim: int = -1) -> Tensor:
    """Categorical entropy from logits, differentiable, one value per row
    (keepdim), matching `sum_dim`'s shape convention.
    """
    logp = log_softmax(logits, dim=dim)
    return mul(sum_dim(mul(exp(logp), logp), dim=dim, keepdim=True), -1.0)


# --------------------------------------------------------------------------
# non-finite guard
# --------------------------------------------------------------------------

def non_finite_check(tensor) -> bool:
    """True if `tensor` holds any NaN or Inf.

    The one host read is the final scalar sum, which is unavoidable for a
    caller wanting a Python bool. This is the eager per-step guard: calling it
    inside a captured trace is a hard failure, since reads are not supported
    during trace capture.
    """
    raw = tensor.data if isinstance(tensor, Tensor) else tensor
    non_finite = ttnn.typecast(ttnn.logical_not(ttnn.isfinite(raw)), ttnn.float32)
    return bool(to_host(ttnn.sum(non_finite)).reshape(-1)[0] > 0.0)
