"""Host transfer, fixed-address device buffers, in-place copy, precision policy.

Two engine invariants live here and both fail silently when violated:
  1. address stability: never rebind `.data`; copy in place via
     ttnn.add(src, 0.0, output_tensor=dst). Captured traces replay against
     fixed device buffer addresses.
  2. no host reads inside a step, because trace capture forbids them.
"""

from ttrl.engine.tensor.arena import (
    StaticArena,
    copy_in_place,
    copy_tensor_in_place,
)
from ttrl.engine.tensor.precision import MASTER_DTYPE
from ttrl.engine.tensor.transfer import restore_in_place, to_device, to_host

__all__ = [
    "MASTER_DTYPE",
    "StaticArena",
    "copy_in_place",
    "copy_tensor_in_place",
    "restore_in_place",
    "to_device",
    "to_host",
]
