"""Reverse-mode autograd tape over ttnn tensors.

ttnn ships forward ops and a set of fused `*_bw` kernels but no tape. This
module supplies `Tensor`, `backward` and `no_grad`; the ops themselves live in
`ttrl.engine.ops`.
"""

from __future__ import annotations

import numpy as np
import ttnn

from ttrl.engine.tensor import to_device, to_host

DEFAULT_DTYPE = ttnn.float32
GRAD_ACCUM_DTYPE = ttnn.float32


# --------------------------------------------------------------------------
# no-grad mode
# --------------------------------------------------------------------------

_GRAD_ENABLED = True


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


class no_grad:
    """Ops built inside record no tape, so forward-only calls (act, eval) do
    not pin saved activations in device memory until the nodes are collected.
    """

    def __enter__(self):
        global _GRAD_ENABLED
        self._prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, *exc):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev
        return False


# --------------------------------------------------------------------------
# the tape
# --------------------------------------------------------------------------

def _no_backward(grad):
    # Module-level so leaves do not each carry their own closure object.
    return None


class Tensor:
    """A node in the autograd graph, backed by device memory.

    A `_backward(grad)` closure takes the node's gradient as an argument and
    must never close over the node it is attached to: a closure that captures
    its own node makes the graph cyclic and leaked DRAM until collection.
    """

    __slots__ = ("data", "requires_grad", "grad", "_backward", "_parents", "_op",
                 "_backward_done")

    def __init__(self, data, requires_grad: bool = False, _parents=(), _op: str = ""):
        self.data = data                 # ttnn.Tensor
        self.requires_grad = requires_grad
        self.grad = None                 # ttnn.Tensor | None
        self._backward = _no_backward
        self._parents = tuple(_parents)
        self._op = _op
        self._backward_done = False

    # -- construction ------------------------------------------------------

    @staticmethod
    def from_numpy(array, device, *, requires_grad=False, dtype=DEFAULT_DTYPE,
                   layout=None) -> "Tensor":
        return Tensor(to_device(np.asarray(array), device, dtype, layout=layout),
                      requires_grad)

    def numpy(self) -> np.ndarray:
        return to_host(self.data)

    # -- introspection -----------------------------------------------------

    @property
    def shape(self):
        return tuple(self.data.shape)

    @property
    def device(self):
        return self.data.device()

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op or 'leaf'}, grad={self.requires_grad})"

    # -- gradient plumbing -------------------------------------------------

    def _accumulate(self, grad):
        # One accumulator dtype whatever the forward pass produced, else
        # `ttnn.add` silently narrows to whichever operand arrived second.
        if grad.dtype != GRAD_ACCUM_DTYPE:
            grad = ttnn.typecast(grad, GRAD_ACCUM_DTYPE)
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = ttnn.add(self.grad, grad)

    def zero_grad(self):
        self.grad = None

    def detach(self) -> "Tensor":
        return Tensor(self.data, requires_grad=False)

    def backward(self):
        if not self.requires_grad:
            raise RuntimeError(
                "backward() called on a Tensor that requires no gradient; it "
                "carries no tape (a leaf, a detached node, or built inside "
                "no_grad()), so this call would silently produce no parameter "
                "gradients at all."
            )
        if self._backward_done:
            raise RuntimeError(
                "backward() called a second time on the same graph; this tape "
                "does not keep saved activations alive for a second pass "
                "(no retain_graph). Build a fresh forward pass instead."
            )

        topo: list[Tensor] = []
        seen: set[int] = set()
        # Post-order DFS over an explicit stack. A recursive helper here held
        # the whole tape alive and leaked ~486 MB per update.
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if id(parent) not in seen:
                    stack.append((parent, False))

        numel = 1
        for dim in self.shape:
            numel *= dim
        if numel != 1:
            raise RuntimeError(
                f"backward() called on a non-scalar Tensor with shape {self.shape} "
                f"(numel={numel}); there is no `gradient=` kwarg to seed a "
                "non-scalar backward pass, so reduce to a scalar first."
            )

        # Seed in the accumulator dtype, matching `_accumulate`, else a bf16
        # loss starts backward at 8 mantissa bits.
        seed = ttnn.full_like(self.data, 1.0)
        if seed.dtype != GRAD_ACCUM_DTYPE:
            seed = ttnn.typecast(seed, GRAD_ACCUM_DTYPE)
        self.grad = seed
        try:
            for node in reversed(topo):
                grad = node.grad
                if grad is not None:
                    if node._backward_done and node is not self:
                        raise RuntimeError(
                            "backward() reached a node whose tape was already "
                            "consumed by a previous backward() call on a graph "
                            "sharing this subgraph; reusing a consumed subgraph "
                            "is not supported (no retain_graph)."
                        )
                    node._backward(grad)
        finally:
            # Release the tape in a finally, so saved activations and backward
            # closures go whether backward finished or raised. Leaves start the
            # next step's graph, so only non-leaf nodes are marked done:
            # marking a leaf would make the next backward() believe this step's
            # consumed subgraph was being reused.
            for node in topo:
                if node._op:
                    node._backward_done = True
                node._backward = _no_backward
                node._parents = ()

    # -- operators ---------------------------------------------------------

    def __add__(self, other):
        # Deferred: ttrl.engine.ops.elementwise imports this module.
        from ttrl.engine.ops.elementwise import add

        return add(self, other)

    __radd__ = __add__

    def __mul__(self, other):
        # Deferred: ttrl.engine.ops.elementwise imports this module.
        from ttrl.engine.ops.elementwise import mul

        return mul(self, other)

    __rmul__ = __mul__

    def __sub__(self, other):
        # Deferred: ttrl.engine.ops.elementwise imports this module.
        from ttrl.engine.ops.elementwise import sub

        return sub(self, other)

    def __rsub__(self, other):
        # Deferred: ttrl.engine.ops.elementwise imports this module.
        from ttrl.engine.ops.elementwise import sub

        return sub(other, self)

    def __neg__(self):
        # Deferred: ttrl.engine.ops.elementwise imports this module.
        from ttrl.engine.ops.elementwise import mul

        return mul(self, -1.0)

    def __truediv__(self, other):
        # Deferred: ttrl.engine.ops.elementwise imports this module.
        from ttrl.engine.ops.elementwise import div

        return div(self, other)


def backward(tensor: Tensor) -> None:
    tensor.backward()


# --------------------------------------------------------------------------
# shared op-building helpers
# --------------------------------------------------------------------------

def _is_scalar(x) -> bool:
    """A python/numpy scalar operand, i.e. one ttnn can take directly."""
    return not isinstance(x, Tensor) and not hasattr(x, "shape")


def _as_tensor(x, like: Tensor) -> Tensor:
    """Promote a python scalar to a device tensor shaped like `like`."""
    if isinstance(x, Tensor):
        return x
    return Tensor(ttnn.full_like(like.data, float(x)), requires_grad=False)


def _needs(*tensors) -> bool:
    return _GRAD_ENABLED and any(t.requires_grad for t in tensors)


def _detached(data) -> Tensor:
    """A forward-only result: no parents, no backward closure, no tape.

    Every op routes through this when `_needs(...)` is False. Returning a node
    that still carried `_parents` and a live `_backward` would keep saved
    activations alive and leave the op backward-reachable, so a later
    `backward()` would push gradient through work the caller asked not to
    record.
    """
    return Tensor(data, requires_grad=False)
