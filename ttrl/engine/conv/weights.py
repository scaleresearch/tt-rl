"""The prepared-layout master weight.

ttnn.conv2d accepts a weight that is already in its own prepared TILE layout
and uses it with no host read and no re-preparation, so the prepared tensor IS
the master weight here: the optimizer updates it in place and nothing ever
re-derives it. That is what makes a conv forward fully trace-capturable.

Two things come with that and both are load-bearing:

  * conv_config.weights_dtype is pinned to the master's dtype, because conv2d
    reads it to decide whether the weight needs converting.
  * ttrl validates the prepared weight itself, because conv2d's own validator
    ignores in_channels and the shard config, and on a validation failure
    conv2d silently calls from_device() (a host read, fatal inside trace
    capture) with nothing but a log_warning. A weight prepared under a
    different parallel config passes upstream's check and computes silently
    wrong output.

The layout is derived and verified WITHOUT ever reading a padded buffer back to
host, because ttnn 0.80 exposes padded bytes through torch and through nothing
else, and torch is not a dependency of this library. `to_numpy`, `cpu()`,
`to_list`, `reshape`, `pad`, `pad_to_tile` and `host_buffer` were each checked
on 0.80: every one of them yields the logical view or refuses. So `cp`, the one
layout parameter that is not metadata, is found by enumerating the layouts the
metadata permits and asking the convolution which one is real, which also makes
the check stronger than a matrix comparison: it proves the weight ttrl builds
and the weight ttnn prepares are interchangeable to the op that consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ttnn

from ttrl.engine.conv.config import FP32, HEIGHT_SHARDED, accurate_compute_config, conv_config
from ttrl.engine.conv.geometry import ConvGeometry
from ttrl.engine.conv.packing import PreparedLayout, candidate_layouts
from ttrl.engine.tensor import to_host

__all__ = ["PreparedSpec", "conv_forward", "prepare_master", "validate_prepared"]


@dataclass(frozen=True)
class PreparedSpec:
    """Everything the master weight was prepared for. Fixed at construction.

    The shard layout is decided once, here, and never retried per call: the
    prepared layout is derived from the activation's sharding, so a conv that
    silently fell back to another placement would be convolving with a weight
    prepared for a different one.
    """

    geom: ConvGeometry
    layout: PreparedLayout
    shard_layout: object
    conv_cfg: object
    input_memory_config: object
    input_layout: object
    weights_dtype: object = FP32


def _reference_prepared(w_host, geom: ConvGeometry, device, conv_cfg,
                        input_memory_config, input_layout):
    return ttnn.prepare_conv_weights(
        weight_tensor=w_host,
        input_memory_config=input_memory_config,
        input_layout=input_layout,
        weights_format="OIHW",
        in_channels=geom.in_channels,
        out_channels=geom.out_channels,
        batch_size=geom.batch,
        input_height=geom.in_h,
        input_width=geom.in_w,
        kernel_size=(geom.kernel, geom.kernel),
        stride=(geom.stride, geom.stride),
        padding=(geom.padding, geom.padding),
        dilation=(1, 1),
        has_bias=False,
        groups=1,
        device=device,
        input_dtype=FP32,
        conv_config=conv_cfg,
        compute_config=accurate_compute_config(device),
    )


def conv_forward(x_rm, weight, geom: ConvGeometry, conv_cfg, device):
    """One ttnn.conv2d dispatch. The single spelling of the forward call."""
    return ttnn.conv2d(
        input_tensor=x_rm,
        weight_tensor=weight,
        device=device,
        in_channels=geom.in_channels,
        out_channels=geom.out_channels,
        batch_size=geom.batch,
        input_height=geom.in_h,
        input_width=geom.in_w,
        kernel_size=(geom.kernel, geom.kernel),
        stride=(geom.stride, geom.stride),
        padding=(geom.padding, geom.padding),
        dilation=(1, 1),
        groups=1,
        conv_config=conv_cfg,
        compute_config=accurate_compute_config(device),
        dtype=FP32,
    )


def _probe_weight(geom: ConvGeometry) -> np.ndarray:
    """A deterministic, non-degenerate weight for the layout equivalence test.

    Not the caller's weight: an all-zero or constant weight makes two different
    layouts produce identical output, and the test would then accept the wrong
    one.
    """
    rng = np.random.default_rng(0)
    return rng.standard_normal(
        (geom.out_channels, geom.in_channels, geom.kernel, geom.kernel)
    ).astype(np.float32)


def _probe_activation(geom: ConvGeometry, device):
    rng = np.random.default_rng(1)
    rows = geom.batch * geom.in_h * geom.in_w
    arr = rng.standard_normal((1, 1, rows, geom.in_channels)).astype(np.float32)
    return ttnn.Tensor(arr, FP32).to(ttnn.ROW_MAJOR_LAYOUT).to(device)


def prepare_master(w_oihw: np.ndarray, geom: ConvGeometry, device, *,
                   shard_layout=HEIGHT_SHARDED) -> tuple["ttnn.Tensor", PreparedSpec]:
    """Build the master weight in conv2d's prepared layout, and prove it.

    Runs once per layer at construction, never inside a step. The proof is
    interchangeability: a probe convolution is run with ttnn's own prepared
    weight and again with the weight ttrl packs, and the two outputs must be
    bit-identical. Exactly one candidate layout may pass; zero means the
    packing no longer describes this ttnn build, and more than one means the
    geometry cannot distinguish them and nothing should be guessed.
    """
    disp = geom.dispatch
    input_memory_config = ttnn.DRAM_MEMORY_CONFIG
    input_layout = ttnn.ROW_MAJOR_LAYOUT
    conv_cfg = conv_config(shard_layout, weights_dtype=FP32)
    probe = _probe_weight(disp)
    reference = _reference_prepared(
        ttnn.Tensor(probe, FP32).to(input_layout), disp, device, conv_cfg,
        input_memory_config, input_layout,
    )
    rows, cols = int(reference.padded_shape[2]), int(reference.padded_shape[3])
    activation = _probe_activation(disp, device)
    out = conv_forward(activation, reference, disp, conv_cfg, device)
    want = to_host(out)
    ttnn.deallocate(out)
    ttnn.deallocate(reference)

    accepted, tried = [], []
    for layout in candidate_layouts(rows, cols, disp.out_channels, disp.in_channels,
                                    disp.kernel, disp.kernel):
        tried.append(layout.cp)
        packed = ttnn.Tensor(layout.pack(probe), FP32).to(ttnn.TILE_LAYOUT).to(device)
        out = conv_forward(activation, packed, disp, conv_cfg, device)
        got = to_host(out)
        ttnn.deallocate(out)
        ttnn.deallocate(packed)
        if got.shape == want.shape and np.array_equal(got, want):
            accepted.append(layout)
    ttnn.deallocate(activation)
    if len(accepted) != 1:
        raise ValueError(
            f"conv2d: could not determine the prepared weight layout at geometry "
            f"{disp}. ttnn reports a {rows}x{cols} prepared weight; of the "
            f"candidate channel strides {tried}, {len(accepted)} "
            f"({[l.cp for l in accepted]}) reproduced ttnn's own prepared weight "
            f"bit-for-bit through the convolution. Exactly one must. "
            f"engine/conv/packing.PreparedLayout no longer describes this ttnn "
            f"build's prepared layout."
        )
    layout = accepted[0]
    master = ttnn.Tensor(layout.pack(np.asarray(w_oihw, np.float32)), FP32).to(
        ttnn.TILE_LAYOUT).to(device)
    spec = PreparedSpec(
        geom=disp, layout=layout, shard_layout=shard_layout, conv_cfg=conv_cfg,
        input_memory_config=input_memory_config, input_layout=input_layout,
    )
    return master, spec


def validate_prepared(master, x_rm, spec: PreparedSpec) -> None:
    """Precondition check on every conv call. Host-side attribute reads only.

    conv2d's own validation failure path is a silent from_device() with a
    log_warning, which inside trace capture is the exact failure this design
    deletes, so every property conv2d would have checked (and the two it would
    not) is checked here and raises.
    """
    if not ttnn.is_tensor_storage_on_device(master):
        raise ValueError(
            "conv2d: the master weight is not on the device. The prepared "
            "master weight must stay device-resident; a host weight would make "
            "conv2d re-prepare it on every call."
        )
    if master.dtype != spec.weights_dtype:
        raise ValueError(
            f"conv2d: master weight dtype {master.dtype} does not match the "
            f"conv_config weights_dtype {spec.weights_dtype} it was prepared "
            f"under; conv2d would convert it on host."
        )
    if master.get_layout() != ttnn.TILE_LAYOUT:
        raise ValueError(
            f"conv2d: master weight layout is {master.get_layout()}, but a "
            f"prepared weight is TILE. A row-major weight is treated as "
            f"unprepared and re-derived on host."
        )
    shape = tuple(int(v) for v in master.padded_shape)
    if shape != spec.layout.shape:
        raise ValueError(
            f"conv2d: master weight has padded shape {shape} but the layout it "
            f"was prepared for is {spec.layout.shape} "
            f"(in_channels={spec.geom.in_channels}, "
            f"out_channels={spec.geom.out_channels}, kernel={spec.geom.kernel}). "
            f"Upstream's validator ignores in_channels, so this would compute "
            f"silently wrong output."
        )
    if x_rm.memory_config() != spec.input_memory_config:
        raise ValueError(
            f"conv2d: activation memory config {x_rm.memory_config()} differs "
            f"from the {spec.input_memory_config} the weight was prepared for. "
            f"The prepared layout is derived from the activation's sharding, so "
            f"a weight prepared under another shard config computes silently "
            f"wrong output."
        )
    if x_rm.get_layout() != spec.input_layout:
        raise ValueError(
            f"conv2d: activation layout {x_rm.get_layout()} differs from the "
            f"{spec.input_layout} the weight was prepared for."
        )
