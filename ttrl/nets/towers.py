"""Conv tower geometry, importable without ttnn.

`ttrl.nets.encoders` builds these towers on a device; `ttrl.deploy` reads
them to rebuild a checkpoint's forward on a host, and that path must not
import ttnn. So the stages, and which registered encoder name builds which
tower, live here.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DEFAULT_ENCODER_BY_KIND", "ConvStage", "NATURE_CNN", "NATURE_CNN_S2D",
           "TOWERS"]


@dataclass(frozen=True)
class ConvStage:
    """One convolution of a tower. Geometry of the input is not stated here.

    `fold > 1` is the engine's space-to-depth spelling: the identical
    convolution dispatched over an input whose spatial extent is divided by
    `fold` and whose channel count is multiplied by `fold ** 2`. It is valid
    only when `fold == stride`, `padding == 0` and `fold` divides the kernel
    and both input extents, which is why it is a per-stage option rather than
    something derived: a fold applied wherever it happened to be legal would
    silently dispatch different kernels for two different observation shapes.
    """

    out_channels: int
    kernel: int
    stride: int
    padding: int = 0
    fold: int = 1


#: Mnih et al. 2015. The architecture, not the geometry: the 84x84 input that
#: paper used gives 20x20 -> 9x9 -> 7x7, but the stages are the same stages at
#: any input size whose strides divide exactly.
NATURE_CNN = (ConvStage(32, 8, 4), ConvStage(64, 4, 2), ConvStage(64, 3, 1))

#: The same architecture with conv1 dispatched space-to-depth, which is the
#: measured-best spelling of that layer. Requires an input whose extents divide
#: by 4, so it is a separate named tower rather than the default.
NATURE_CNN_S2D = (ConvStage(32, 8, 4, fold=4),) + NATURE_CNN[1:]

#: The tower each registered image encoder builds by default. An image
#: encoder absent here has a geometry the exporter cannot read off its name.
TOWERS = {"nature_cnn": NATURE_CNN, "nature_cnn_s2d": NATURE_CNN_S2D}

#: Encoder name implied by a leaf for an OBSERVATION tower when the caller
#: does not name one: a policy's, a value function's, a Q-over-discrete-
#: actions'. Where kind maps to an architecture for a critic, which has an
#: action to fuse and therefore a different answer, is
#: `ttrl.nets.fusion.DEFAULT_CRITIC_ENCODER_BY_KIND`;
#: `ttrl.nets.multi.DEFAULT_ENCODER_BY_ROLE` joins the two and
#: `build_encoder` is the only reader of a leaf's kind. A gate derives the set
#: of kinds from these tables rather than from a typed list, so a new ObsKind
#: without an encoder cannot pass unnoticed.
DEFAULT_ENCODER_BY_KIND = {"image": "nature_cnn", "vector": "mlp"}
