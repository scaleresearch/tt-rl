"""An image policy's conv tower, in numpy and as ONNX nodes.

Imports `numpy` and nothing else, as `ttrl.deploy.artifact` does, because the
deployed forward runs it.

The artifact stores each stage's weight in logical OIHW with `ObsSpec.scale`
folded into the first, its bias as `(out_channels,)`, and its stride and
padding in the metadata. The tower ends in the device's flatten order, which
is height, width, then channel: `ttrl.engine.layers.Flatten` reads the
flattened-NHWC layout a `Conv2d` emits, and the first `Linear` after it was
trained on rows in that order. Flattening NCHW instead would feed it permuted
features, and every action would still be a valid index.
"""

from __future__ import annotations

import numpy as np

__all__ = ["conv_features", "conv_stages", "emit_conv"]


def conv_stages(meta: dict, arrays: dict) -> list[tuple]:
    """`(weight, bias, stride, padding)` per stage; empty for a vector policy."""
    return [
        (arrays[f"conv.{i}.weight"], arrays[f"conv.{i}.bias"],
         int(stage["stride"]), int(stage["padding"]))
        for i, stage in enumerate(meta["conv"])
    ]


def conv_features(rows: np.ndarray, stages: list[tuple]) -> np.ndarray:
    """`(batch, C, H, W)` float32 -> the `(batch, features)` row the trunk reads.

    Each stage is ReLU(conv) computed as one matmul over strided windows, in
    NHWC so the final reshape is already the device's flatten order.
    """
    hidden = np.ascontiguousarray(rows.transpose(0, 2, 3, 1))
    for weight, bias, stride, padding in stages:
        out_channels, _, kernel, _ = weight.shape
        if padding:
            hidden = np.pad(hidden, ((0, 0), (padding, padding),
                                     (padding, padding), (0, 0)))
        windows = np.lib.stride_tricks.sliding_window_view(
            hidden, (kernel, kernel), axis=(1, 2))[:, ::stride, ::stride]
        batch, height, width = windows.shape[:3]
        columns = windows.reshape(batch * height * width, -1)
        out = columns @ weight.reshape(out_channels, -1).T + bias
        hidden = np.maximum(out, np.float32(0.0)).reshape(
            batch, height, width, out_channels)
    return hidden.reshape(hidden.shape[0], -1)


def emit_conv(builder, meta: dict, arrays: dict, source: str) -> str:
    """The same tower as ONNX nodes on `builder`, from NCHW `source`.

    ONNX's `Conv` is NCHW, so the tower runs in that layout and transposes to
    NHWC once, before the flatten.
    """
    hidden = source
    for i, (weight, bias, stride, padding) in enumerate(conv_stages(meta, arrays)):
        conv = builder.node(
            "Conv",
            [hidden, builder.constant(f"conv_{i}_w", weight),
             builder.constant(f"conv_{i}_b", bias)],
            f"conv_{i}_out", strides=[stride, stride],
            pads=[padding] * 4, kernel_shape=list(weight.shape[2:]))
        hidden = builder.node("Relu", [conv], f"conv_{i}_relu")
    nhwc = builder.node("Transpose", [hidden], "conv_nhwc", perm=[0, 2, 3, 1])
    return builder.node("Flatten", [nhwc], "conv_flat", axis=1)
