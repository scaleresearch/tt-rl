"""The conv2d tape node, and the per-layer plan it runs against.

The plan is built once, on device, at layer construction: it decides the shard
placement, builds the prepared master weight and verifies both backward
transforms. Nothing in the call path derives, prepares or validates anything
that could read back to host, so forward and backward are both capturable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ttnn

from ttrl.engine.autograd import Tensor, is_grad_enabled
from ttrl.engine.conv.backward import input_grad, weight_grad
from ttrl.engine.conv.config import (
    FP32,
    HEIGHT_SHARDED,
    is_shard_too_small,
    rm,
    tile,
)
from ttrl.engine.conv.geometry import ConvDoesNotFit, ConvGeometry, check_dx_capacity
from ttrl.engine.conv.transform import (
    TransposeSpec,
    prepare_transpose,
    verify_dw_layout,
)
from ttrl.engine.conv.weights import (
    PreparedSpec, conv_forward, prepare_master, validate_prepared,
)

__all__ = ["ConvPlan", "build_plan", "conv2d"]


@dataclass(frozen=True)
class ConvPlan:
    """Immutable per-layer compute plan. Holds no weight.

    `tspec` is None for a layer built with needs_input_grad=False, which has no
    dX machinery at all.
    """

    geom: ConvGeometry
    pspec: PreparedSpec
    tspec: TransposeSpec | None
    device: object
    input_scale: float = 1.0


def _dispatch(x_rm, weight, pspec: PreparedSpec, device):
    return conv_forward(x_rm, weight, pspec.geom, pspec.conv_cfg, device)


def _zeros_activation(geom: ConvGeometry, device):
    rows = geom.batch * geom.in_h * geom.in_w
    return ttnn.zeros((1, 1, rows, geom.in_channels), dtype=FP32,
                      layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _place(candidates, build, run, what: str):
    """Pick the first shard placement that fits, or raise naming the search.

    A placement search that raises when its space is exhausted is not a
    fallback: every candidate computes the same arithmetic, and the one that
    does not fit in L1 does not run at all.

    `build` is inside the try because building is itself dispatching: the
    layout solve runs convolutions to find out which prepared layout ttnn is
    using, so a placement that does not fit announces itself there and not only
    in `run`.
    """
    tried = []
    for shard_layout in candidates:
        try:
            built = build(shard_layout)
            run(built)
            return built
        except RuntimeError as exc:
            if not is_shard_too_small(exc):
                raise
            tried.append(f"{shard_layout}: {str(exc).splitlines()[0]}")
    detail = "; ".join(tried)
    raise ConvDoesNotFit(
        f"conv2d: no shard placement fits for {what}. Tried {len(tried)} "
        f"placements and every one exceeded L1 ({detail}). Reduce the batch "
        f"size, the channel count or the spatial extent of this layer."
    )


def build_plan(w_oihw: np.ndarray, geom: ConvGeometry, device, *,
               input_scale: float = 1.0, needs_input_grad: bool = True,
               profile=None):
    """Prepare the master weight and the plan, probing the shard placement.

    Height sharding is a clear win at these shapes but splits only along the
    flattened N*H*W axis, so at a large enough batch the per-core activation
    slice stops fitting in L1 and ttnn's own layout choice has to be used
    instead. That is decided HERE, once, rather than retried per call: the
    prepared weight layout is derived from the activation's sharding, so a call
    that fell back to another placement would convolve with a weight prepared
    for the one it abandoned, and upstream's validator does not look at the
    shard config.

    `profile` is the DeviceProfile of `device`, which is where the arch name
    comes from: the dX row ceiling is calibrated per arch and refuses one it
    was not measured on. Without a profile the arch is read with the same ttnn
    call the profile itself uses, so the answer is identical; passing the
    profile is preferred because it keeps that call in one place.

    `needs_input_grad=False` builds no dX machinery at all. That is not an
    optimisation: a first conv layer takes an environment observation, which is
    a leaf, so its dX is never computed, and at this geometry the dX of conv1
    does not fit in L1 from batch 64 up. Building it eagerly would refuse a
    batch the layer can train at perfectly well.
    """
    disp = geom.dispatch
    probe = _zeros_activation(disp, device)
    master = None

    def build_forward(shard_layout):
        nonlocal master
        master, pspec = prepare_master(w_oihw, geom, device, shard_layout=shard_layout)
        return pspec

    pspec = _place(
        (HEIGHT_SHARDED, None), build_forward,
        lambda spec: _dispatch(probe, master, spec, device),
        f"the forward convolution at {disp}",
    )
    tspec = None
    if needs_input_grad:
        # Arithmetically, BEFORE the first ttnn call for dX. Above the measured
        # ceiling prepare_conv_transpose2d_weights spins or segfaults, and
        # neither is something the placement probe below could catch.
        grid = device.compute_with_storage_grid_size()
        arch = profile.arch if profile is not None else ttnn.get_arch_name()
        check_dx_capacity(geom, grid.x * grid.y,
                          ttnn.get_max_worker_l1_unreserved_size(), arch)
        dy_probe = ttnn.zeros(
            (1, 1, disp.batch * disp.out_h * disp.out_w, disp.out_channels),
            dtype=FP32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
        )
        dx_candidates = ((None,) if pspec.shard_layout is None
                         else (pspec.shard_layout, None))
        tspec = _place(
            dx_candidates,
            lambda shard_layout: prepare_transpose(master, w_oihw, pspec, device,
                                                   shard_layout=shard_layout),
            lambda spec: input_grad(dy_probe, master, pspec, spec, device),
            f"the dX conv_transpose2d at {disp}",
        )
    verify_dw_layout(pspec.layout, device)
    return master, ConvPlan(geom=geom, pspec=pspec, tspec=tspec, device=device,
                            input_scale=float(input_scale))


def conv2d(x: Tensor, weight: Tensor, plan: ConvPlan) -> Tensor:
    """Convolve flattened-NHWC activations with a prepared master weight.

    `plan.input_scale != 1.0` means the logical input is `x * input_scale`
    while `x` arrives unscaled. Convolution is linear in its input, so
    conv(s*x, W) == conv(x, s*W) and dL/dW == s * dW(x, dY) exactly; it exists
    so a 1/255 pixel normalisation costs a multiply over the weight instead of
    a pass over every observation. The master weight is deliberately left
    unscaled, which keeps p.data, p.grad and every optimizer update identical
    to the unscaled path.
    """
    device, pspec = plan.device, plan.pspec
    x_rm = rm(x.data)
    if int(np.prod(tuple(int(d) for d in x.shape))) != plan.geom.in_numel:
        raise ValueError(
            f"conv2d: activation has {int(np.prod(tuple(int(d) for d in x.shape)))} "
            f"elements but this layer's geometry {plan.geom} requires "
            f"{plan.geom.in_numel}; the layer was built for a different input."
        )
    validate_prepared(weight.data, x_rm, pspec)
    scaled = weight.data if plan.input_scale == 1.0 else ttnn.multiply(weight.data,
                                                                      plan.input_scale)
    y = _dispatch(x_rm, scaled, pspec, device)
    needs_grad = is_grad_enabled() and (x.requires_grad or weight.requires_grad)
    out = Tensor(tile(y), needs_grad, (x, weight), "conv2d")

    # Takes the gradient as an argument and never closes over `out`: a closure
    # that captures its own node makes the tape cyclic and leaks device DRAM.
    def _backward(grad):
        dy_rm = rm(grad)
        if x.requires_grad:
            if plan.tspec is None:
                raise RuntimeError(
                    "conv2d: this layer was built with needs_input_grad=False, "
                    "so it has no dX path, but its input requires a gradient. "
                    "Rebuild the layer with needs_input_grad=True, or detach "
                    "the activation it is given."
                )
            x._accumulate(tile(input_grad(dy_rm, weight.data, pspec, plan.tspec, device,
                                          input_scale=plan.input_scale)))
        if weight.requires_grad:
            weight._accumulate(weight_grad(x_rm, dy_rm, pspec.geom, pspec.layout, device,
                                           input_scale=plan.input_scale))

    out._backward = _backward
    return out
