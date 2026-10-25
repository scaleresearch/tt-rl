"""Shared helpers for the hardware net tests. Not a test module.

`assert_every_parameter_trains` is here rather than duplicated in each test
file because it is the assertion those files exist to make, and two copies
would drift. Imports ttnn by way of `ttrl.engine`, so only `tt`-marked tests
may use it.
"""

from __future__ import annotations

import numpy as np

from ttrl.engine import Adam, Tensor, backward
from ttrl.engine.tensor import to_host
from ttrl.nets.obs import ObsPair

__all__ = ["upload_obs", "assert_every_parameter_trains"]


def upload_obs(obs_spec, batch: int, device, seed: int = 0):
    """A batch of observations in the layout the chosen encoder consumes.

    `obs_spec` is a one-leaf `ObsGroup`, which is what every net in
    `ttrl.nets.models` is built over here. Image observations go up as ttnn's
    flattened NHWC, `[1, 1, N*H*W, C]`, which is what `Conv2d` emits and
    consumes; vector observations go up as `(N, features)`. Returns
    `(ObsPair, host array)`, the pair being what every net is called with, so
    a torch twin can be built from the same numbers. Both halves are the one
    leaf mapping, which is what a spec with nothing critic-only describes.
    """

    leaf_name = obs_spec.names[0]
    leaf = obs_spec[leaf_name]
    rng = np.random.default_rng(seed)
    if leaf.kind == "image":
        channels, height, width = leaf.shape
        raw = rng.integers(0, 256, (batch, channels, height, width)).astype(np.float32)
        nhwc = raw.transpose(0, 2, 3, 1).reshape(1, 1, batch * height * width, channels)
        return ObsPair.shared({leaf_name: Tensor.from_numpy(
            np.ascontiguousarray(nhwc), device, requires_grad=False)}), raw
    raw = rng.normal(size=(batch, leaf.size)).astype(np.float32)
    return ObsPair.shared(
        {leaf_name: Tensor.from_numpy(raw, device, requires_grad=False)}), raw


def assert_every_parameter_trains(module, loss_fn, *, lr: float = 1e-3) -> list[str]:
    """Backward once, Adam once, and require every parameter to move.

    A name-only check is not enough: a net whose encoder is detached from its
    head still trains, still reports a falling loss, and still exposes every
    parameter name. Only the per-parameter delta catches it. Returns the
    parameter names so a caller can assert on the structure as well.
    """

    named = list(module.named_parameters())
    assert named, "module exposes no parameters"
    before = {name: p.numpy().copy() for name, p in named}

    module.zero_grad()
    backward(loss_fn())

    missing = [name for name, p in named if p.grad is None]
    assert not missing, f"no gradient reached {missing}"
    for name, p in named:
        assert np.isfinite(np.asarray(to_host(p.grad))).all(), \
            f"non-finite gradient in {name}"

    Adam([p for _, p in named], lr=lr).step()

    unmoved = [name for name, p in named
               if np.abs(p.numpy() - before[name]).max() == 0.0]
    assert not unmoved, f"one Adam step did not move {unmoved}"
    return [name for name, _ in named]
