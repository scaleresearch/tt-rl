"""conv2d forward and backward over prepared-layout master weights.

ttnn ships conv2d and conv_transpose2d but no conv2d_backward. Here the
forward is ttnn.conv2d against a weight that is already in conv2d's own
prepared layout, so there is nothing to prepare per call; dX is
ttnn.conv_transpose2d; and dW is a single matmul over strided im2col slices
whose result already lands in the prepared layout. Every op in the call path
is device to device, which is what makes a conv model fully trace-capturable.

Activations move through the network in ttnn's flattened NHWC form,
[1, 1, N*H*W, C], because that is what conv2d emits and consumes. The logical
geometry is carried by ConvGeometry, derived from the input shape the layer was
constructed with.

No parameter-write hook is registered here, deliberately. The prepared tensor
IS the master weight, so an in-place write into it is seen by the next
convolution and by every trace replay, and nothing derived from the weight
contents is left to go stale.

Importing this package must not import ttnn, so that `geometry` (pure index
arithmetic, testable with no wheel and no chip) stays reachable on a host
without ttnn installed.
"""

from __future__ import annotations

import importlib
from typing import Any

from ttrl.engine.conv.geometry import (
    ConvDoesNotFit,
    ConvGeometry,
    check_dx_capacity,
    conv_out_size,
    dx_rows_per_core,
    largest_dx_batch,
    max_dx_rows_per_core,
    transpose_input_extent,
)
from ttrl.engine.conv.packing import (
    PreparedLayout,
    fold_input_numpy,
    fold_weight_numpy,
    s2d_flat_permutation,
    unfold_weight_numpy,
)

_LAZY = {"Conv2d": "module", "conv2d": "forward", "build_plan": "forward"}

__all__ = [
    "ConvDoesNotFit",
    "ConvGeometry",
    "PreparedLayout",
    "check_dx_capacity",
    "conv_out_size",
    "dx_rows_per_core",
    "largest_dx_batch",
    "max_dx_rows_per_core",
    "transpose_input_extent",
    "fold_input_numpy",
    "fold_weight_numpy",
    "s2d_flat_permutation",
    "unfold_weight_numpy",
    *sorted(_LAZY),
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return getattr(importlib.import_module(f"ttrl.engine.conv.{module}"), name)


def __dir__() -> list[str]:
    return sorted(__all__)
