"""Host-only stand-ins for `ttrl.engine.autograd.Tensor`, `Module` and `Adam`.

Pure numpy: importing this must not need ttnn.
"""

from __future__ import annotations

import numpy as np


class FakeTensor:
    """Stands in for `ttrl.engine.autograd.Tensor`: a named host-numpy buffer with a
    fixed identity (`id(self)`), like a fixed device address."""

    def __init__(self, array, dtype="float32"):
        self._array = np.array(array, dtype=np.float32)
        self.dtype = dtype
        self.data = self  # `_upload_in_place` writes through `.data`

    @property
    def shape(self):
        return self._array.shape

    def numpy(self):
        return self._array.copy()

    def to(self, *_a, **_k):  # ttnn.Tensor(...).to(TILE_LAYOUT) stand-in
        return self


class FakeModule:
    """A fixed-order, fixed-identity list of parameters, like
    `ttrl.engine.nn.Module`.

    `named_parameters` is part of that interface, not decoration:
    `ttrl.engine.mesh.sync.param_schema` compares names as well as shapes,
    which is what catches two same-shaped parameters swapped between replicas.
    A fake without it could not exercise that check at all.
    """

    def __init__(self, values, names=None):
        self._params = [FakeTensor(v) for v in values]
        self._names = list(names) if names is not None else [f"w{i}" for i in range(len(values))]
        self.copy_from_calls = 0
        self._source_values = None

    def parameters(self):
        return list(self._params)

    def named_parameters(self):
        return list(zip(self._names, self._params))

    def copy_from(self, other: "FakeModule"):
        self.copy_from_calls += 1
        self._source_values = [p.numpy().copy() for p in other.parameters()]
        for dst, src in zip(self._params, other.parameters()):
            dst._array = src.numpy().copy()


class FakeAdam:
    def __init__(self, n_params, shape=(4,), lr=1e-4, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
        self.t = 5
        self.lr = lr
        self.b1, self.b2 = b1, b2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [FakeTensor(np.full(shape, 1.0)) for _ in range(n_params)]
        self.v = [FakeTensor(np.full(shape, 2.0)) for _ in range(n_params)]


class TapeData:
    """The minimum of `ttnn.Tensor` that the tape itself touches: a shape."""

    def __init__(self, shape=(1, 1)):
        self.shape = shape
        self.dtype = None


class MatmulData:
    """The minimum `ttrl.engine.ops.matmul` reads off a `Tensor.data`.

    Separate from `TapeData` on purpose: matmul reads a real `dtype` and calls
    `device()` to pick an accumulation config, so a shapes-only stand-in
    cannot drive it and a device-carrying one would hide what the tape test
    is proving (that the tape needs nothing but a shape).
    """

    def __init__(self, dtype, device):
        self.shape = (32, 32)
        self.dtype = dtype
        self._device = device

    def device(self):
        return self._device
