"""`Linear`, `LayerNorm`, `ReLU`, `Tanh`, `Sequential`, `Flatten`.

Every layer here is single-Tensor-in, single-Tensor-out, which is what lets
`Sequential` hold any of them (a `Conv2d` included) without knowing what it is.
"""

from __future__ import annotations

import math

import numpy as np

from ttrl.engine.autograd import Tensor
from ttrl.engine.nn import Module, ModuleList
from ttrl.engine.ops import (
    add, div, matmul, mean_dim, mul, relu, reshape, sqrt, square, sub, tanh,
)

__all__ = ["Linear", "LayerNorm", "ReLU", "Tanh", "Sequential", "Flatten"]


class Linear(Module):
    """y = x @ W + b.

    Init matches PyTorch's `nn.Linear` default (`kaiming_uniform_(a=sqrt(5))`)
    so a parity test can compare against torch directly.

    Weight and bias are the fp32 master buffers and are fed to `matmul`
    unchanged: `ttrl.engine.ops.matmul` pins HiFi4 with fp32 accumulation, so
    the fp32 path is the accurate one (see `ttrl.engine.tensor.precision`).
    """

    def __init__(self, in_features: int, out_features: int, device, *,
                 bias: bool = True):
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                "Linear: in_features and out_features must be > 0, got "
                f"in_features={in_features}, out_features={out_features}"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        bound = 1.0 / math.sqrt(in_features)
        # Stored (in_features, out_features), the TRANSPOSE of PyTorch's
        # `nn.Linear.weight` layout, so `forward` is a plain `x @ W` with no
        # transpose op. Every checkpoint and call site assumes this; changing
        # it without transposing every save and load path is a silent
        # transpose bug that still trains.
        weight = np.random.uniform(
            -bound, bound, (self.in_features, self.out_features)
        ).astype(np.float32)
        self.weight = Tensor.from_numpy(weight, device, requires_grad=True)
        self.bias = None
        if bias:
            b = np.random.uniform(-bound, bound, (1, self.out_features)).astype(np.float32)
            self.bias = Tensor.from_numpy(b, device, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        out = matmul(x, self.weight)
        if self.bias is not None:
            out = add(out, self.bias)
        return out


class LayerNorm(Module):
    """Normalise each row over its last axis, then scale and shift.

    y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta, with the mean and
    the variance taken over the final `normalized_shape` features.

    Composed from the primitive ops rather than calling `ttnn.layer_norm`,
    which ships a fused forward and, on 0.80, no bound `layer_norm_bw` at all:
    only `ttnn.moreh_layer_norm_backward`, which sits outside the tape, the
    same reason `log_softmax` is composed. Composing also means the
    gradient flows through the mean and the variance, which is the derivative
    of this function; treating the statistics as constants is a different
    function whose difference shows up as a slow bias rather than as an error.

    `eps` is added before the square root, because d/dx sqrt(x) is unbounded
    at zero and a constant row has variance zero.
    """

    def __init__(self, normalized_shape: int, device, *, eps: float = 1e-5):
        if normalized_shape <= 0:
            raise ValueError(
                "LayerNorm: normalized_shape must be > 0, got "
                f"{normalized_shape}"
            )
        if eps <= 0.0:
            raise ValueError(
                f"LayerNorm: eps must be > 0, got {eps}; the variance of a "
                "constant row is zero and sqrt has no derivative there."
            )
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        # Shaped (1, N) so the broadcast against a (B, N) activation is the
        # one every elementwise backward here already reduces over.
        self.weight = Tensor.from_numpy(
            np.ones((1, self.normalized_shape), dtype=np.float32), device,
            requires_grad=True)
        self.bias = Tensor.from_numpy(
            np.zeros((1, self.normalized_shape), dtype=np.float32), device,
            requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        if int(x.shape[-1]) != self.normalized_shape:
            raise ValueError(
                f"LayerNorm({self.normalized_shape}): expected a last axis of "
                f"{self.normalized_shape} features, got shape "
                f"{tuple(int(d) for d in x.shape)}. Build the layer for the "
                "width the previous layer emits."
            )
        centred = sub(x, mean_dim(x, dim=-1))
        variance = mean_dim(square(centred), dim=-1)
        return add(mul(div(centred, sqrt(add(variance, self.eps))), self.weight),
                   self.bias)


class ReLU(Module):
    """Elementwise `max(x, 0)`."""

    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class Tanh(Module):
    """Elementwise `tanh(x)`. The squashing layer a continuous policy ends in."""

    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)


class Sequential(Module):
    """Chain layers in order, feeding each one's output to the next.

    Children live in a `ModuleList`, so their parameters are named
    `layers.<i>.<name>` and an `append` after construction is registered.
    """

    def __init__(self, *layers):
        self.layers = ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def append(self, layer) -> "Sequential":
        self.layers.append(layer)
        return self

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index):
        return self.layers[index]


class Flatten(Module):
    """`[1, 1, N*H*W, C]` (the flattened-NHWC layout a `Conv2d` emits) ->
    `(N, H*W*C)`, the shape a `Linear` expects.

    `H`, `W` and `C` are fixed at construction, because a conv tower's spatial
    geometry is a model constant and the input's element count alone does not
    pin its layout down.
    """

    def __init__(self, h: int, w: int, channels: int):
        if h <= 0 or w <= 0 or channels <= 0:
            raise ValueError(
                f"Flatten: h, w and channels must be > 0, got {h}, {w}, {channels}"
            )
        self.h = int(h)
        self.w = int(w)
        self.channels = int(channels)

    @property
    def out_features(self) -> int:
        """Width of the row this layer emits, i.e. the next `Linear`'s input."""
        return self.h * self.w * self.channels

    def forward(self, x: Tensor) -> Tensor:
        shape = tuple(int(d) for d in x.shape)
        rows = self.h * self.w
        # An NCHW tensor and a flattened-NHWC one can hold the same element
        # count while ordering it differently, which would feed permuted
        # features onward and still train. Check the layout, not the count.
        if (
            len(shape) != 4
            or shape[0] != 1
            or shape[1] != 1
            or shape[3] != self.channels
            or shape[2] % rows != 0
        ):
            raise ValueError(
                f"Flatten({self.h}, {self.w}, {self.channels}): expected "
                f"flattened-NHWC input of shape [1, 1, N*{self.h}*{self.w}, "
                f"{self.channels}], got {shape}"
            )
        batch = shape[2] // rows
        return reshape(x, (batch, self.out_features))
