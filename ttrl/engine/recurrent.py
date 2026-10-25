"""`GRUCell` and `GRU`: one gated recurrent step, and a fixed-length unroll.

A recurrent cell is what makes a partially observed task expressible: the
hidden state carries what the current observation does not say. The gate on
this module is the unrolled backward pass. A single step verifies the gate
algebra, which rarely breaks; what breaks is one weight tensor accumulating
gradient over T timesteps and gradient reaching the initial state.

WHAT THIS DOES AND DOES NOT TRACE. The unrolled training step traces with no
new mechanism: T is fixed at construction, so the graph is fixed, `h_0` is an
ordinary bound input whose buffer `ttrl.engine.step.inputs.InputBinder` writes
in place, each timestep's state is a fresh node with its own buffer, and
nothing in the unroll reads to host. A single-step acting loop is the case that
needs care, because there the state is both an input and an output of the same
step and a captured program would read and write one aliased buffer. The
arrangement that keeps both invariants is two fixed buffers, one read by the
trace and one written by it, with a device-to-device copy back after the
replay returns; that copy stays outside the captured region and reads nothing
to host. The acting loop itself is not built here, it belongs with the
sequence-aware rollout storage.
"""

from __future__ import annotations

import math

import numpy as np

from ttrl.engine.autograd import Tensor
from ttrl.engine.nn import Module
from ttrl.engine.ops import (
    add, apply_done_mask, concat, matmul, mul, reshape, sigmoid, split, sub,
    tanh,
)

__all__ = ["GRU", "GRUCell"]

#: The three gates, in the order their columns sit in the stacked weights.
#: PyTorch's `nn.GRUCell` uses this order, and a parity test against torch
#: compares the stacked tensors directly, so reordering here silently reads a
#: reset gate's columns as an update gate's.
GATES = ("reset", "update", "candidate")


class GRUCell(Module):
    """One gated recurrent step: (x, h) -> h'.

    reset     r = sigmoid(x @ W_ih[r] + b_ih[r] + h @ W_hh[r] + b_hh[r])
    update    z = sigmoid(x @ W_ih[z] + b_ih[z] + h @ W_hh[z] + b_hh[z])
    candidate n = tanh(x @ W_ih[n] + b_ih[n] + r * (h @ W_hh[n] + b_hh[n]))
    output    h' = (1 - z) * n + z * h

    `r` multiplies the hidden half of the candidate only, which is PyTorch's
    `nn.GRUCell` and so lets torch serve as the gradcheck reference.

    The two weights are stored stacked over the three gates and cut apart with
    `ttrl.engine.ops.split`, so a step is two matmuls rather than six. Both are
    held (features, 3 * hidden), the TRANSPOSE of PyTorch's `weight_ih_l0`
    layout, matching `ttrl.engine.layers.Linear` so that `forward` needs no
    transpose op. Every checkpoint assumes it.
    """

    def __init__(self, in_features: int, hidden_size: int, device):
        if in_features <= 0 or hidden_size <= 0:
            raise ValueError(
                "GRUCell: in_features and hidden_size must be > 0, got "
                f"in_features={in_features}, hidden_size={hidden_size}"
            )
        self.in_features = int(in_features)
        self.hidden_size = int(hidden_size)
        gates = len(GATES) * self.hidden_size
        bound = 1.0 / math.sqrt(self.hidden_size)

        def uniform(shape):
            return np.random.uniform(-bound, bound, shape).astype(np.float32)

        self.weight_ih = Tensor.from_numpy(
            uniform((self.in_features, gates)), device, requires_grad=True)
        self.weight_hh = Tensor.from_numpy(
            uniform((self.hidden_size, gates)), device, requires_grad=True)
        self.bias_ih = Tensor.from_numpy(
            uniform((1, gates)), device, requires_grad=True)
        self.bias_hh = Tensor.from_numpy(
            uniform((1, gates)), device, requires_grad=True)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        self._check(x, h)
        widths = (self.hidden_size,) * len(GATES)
        gi = split(add(matmul(x, self.weight_ih), self.bias_ih), widths)
        gh = split(add(matmul(h, self.weight_hh), self.bias_hh), widths)
        reset = sigmoid(add(gi[0], gh[0]))
        update = sigmoid(add(gi[1], gh[1]))
        candidate = tanh(add(gi[2], mul(reset, gh[2])))
        return add(mul(sub(1.0, update), candidate), mul(update, h))

    def _check(self, x: Tensor, h: Tensor) -> None:
        x_shape = tuple(int(d) for d in x.shape)
        h_shape = tuple(int(d) for d in h.shape)
        if len(x_shape) != 2 or x_shape[1] != self.in_features:
            raise ValueError(
                f"GRUCell({self.in_features}, {self.hidden_size}): expected an "
                f"input of shape (batch, {self.in_features}), got {x_shape}. "
                "Reshape one timestep to a 2-D batch of rows first."
            )
        if len(h_shape) != 2 or h_shape[1] != self.hidden_size:
            raise ValueError(
                f"GRUCell({self.in_features}, {self.hidden_size}): expected a "
                f"state of shape (batch, {self.hidden_size}), got {h_shape}."
            )
        if x_shape[0] != h_shape[0]:
            raise ValueError(
                f"GRUCell: input batch {x_shape[0]} and state batch "
                f"{h_shape[0]} disagree; one state row belongs to one input "
                "row, so they must match."
            )


class GRU(Module):
    """A `GRUCell` unrolled over a sequence of timesteps.

    `forward(x, h0, done=None)` takes a (T, batch, features) sequence and a
    (batch, hidden) initial state and returns the (T, batch, hidden) stacked
    outputs together with the final state. T is read off the input rather than
    fixed at construction, and one call unrolls one T, so the graph a trace
    captures is fixed by the shape the caller binds.

    The weights are the cell's, applied at every timestep, so the tape
    accumulates each weight's gradient T times through `Tensor._accumulate`.
    That accumulation is the thing this module exists to get right.

    `done` is an optional (T, batch, 1) mask of episode boundaries, applied to
    the state entering each timestep through
    `ttrl.engine.ops.apply_done_mask`, which zeroes a state carried across a
    boundary. It is a device tensor, so the reset adds no host read and the
    whole unroll stays capturable.
    """

    def __init__(self, in_features: int, hidden_size: int, device):
        self.cell = GRUCell(in_features, hidden_size, device)
        self.in_features = self.cell.in_features
        self.hidden_size = self.cell.hidden_size

    def forward(self, x: Tensor, h0: Tensor, done: Tensor | None = None):
        shape = tuple(int(d) for d in x.shape)
        if len(shape) != 3 or shape[2] != self.in_features:
            raise ValueError(
                f"GRU({self.in_features}, {self.hidden_size}): expected a "
                f"sequence of shape (timesteps, batch, {self.in_features}), "
                f"got {shape}. Stack the timesteps on axis 0."
            )
        steps, batch, _ = shape
        if done is not None:
            done_shape = tuple(int(d) for d in done.shape)
            if done_shape != (steps, batch, 1):
                raise ValueError(
                    f"GRU: done must be shaped ({steps}, {batch}, 1) to match "
                    f"the sequence it masks, got {done_shape}."
                )
            done_steps = split(done, (1,) * steps, dim=0)
        rows = split(x, (1,) * steps, dim=0)
        h = h0
        outputs = []
        for index, row in enumerate(rows):
            if done is not None:
                h = apply_done_mask(h, reshape(done_steps[index], (batch, 1)))
            h = self.cell(reshape(row, (batch, self.in_features)), h)
            outputs.append(reshape(h, (1, batch, self.hidden_size)))
        return concat(outputs, dim=0), h
