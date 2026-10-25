"""Observation encoders, one per leaf of an observation.

An encoder is chosen by a leaf's `ObsSpec.kind` and its geometry is derived
from that leaf's `ObsSpec.shape`. Choosing and joining the per-leaf towers is
`ttrl.nets.multi`, which holds `build_encoder` and joins the
kind-to-architecture mapping below with the critic role's own. Nothing in this
file has an action: joining one onto a tower's features, and the mapping a
critic resolves 'auto' through, are `ttrl.nets.fusion`.

Nothing here branches on which environment suite produced the spec. There is
no 84, no 4 and no 3136: the Nature CNN's architecture (Mnih et al. 2015) is
kept as a tower of (out_channels, kernel, stride, padding) stages and the
spatial geometry falls out of the input shape, so (3, 64, 64) or (1, 60, 60) is
an ordinary input rather than a special case.

`ObsSpec.scale` is folded into the first layer's weight on both paths, so a
1/255 pixel normalisation costs one multiply over a small weight instead of a
pass over every observation. Convolution and matmul are both linear in their
input, so `f(s*x, W) == f(x, s*W)` and `dL/dW == s * dW(x, dY)` exactly; the
master weight stays unscaled, which keeps `p.data`, `p.grad` and every
optimizer update identical to the unscaled path.
"""

from __future__ import annotations

from ttrl.engine.conv import Conv2d
from ttrl.engine.layers import Flatten, Linear, ReLU, Sequential
from ttrl.engine.ops import add, matmul, mul
from ttrl.nets.towers import (
    DEFAULT_ENCODER_BY_KIND,
    NATURE_CNN,
    NATURE_CNN_S2D,
    ConvStage,
)
from ttrl.spaces import ObsSpec

__all__ = [
    "DEFAULT_ENCODER_BY_KIND",
    "ConvStage",
    "Encoder",
    "NATURE_CNN",
    "NATURE_CNN_S2D",
    "ScaledLinear",
    "identity_encoder",
    "image_encoder",
    "vector_encoder",
]


class ScaledLinear(Linear):
    """`y = (scale * x) @ W + b`, with `scale` folded into the weight.

    Exists so the vector path folds `ObsSpec.scale` the same way `Conv2d`'s
    `input_scale` does on the image path, instead of multiplying every
    observation elementwise. `scale == 1.0` is the plain `Linear` call with no
    extra op at all.
    """

    def __init__(self, in_features: int, out_features: int, device, *, scale: float = 1.0,
                 bias: bool = True):
        super().__init__(in_features, out_features, device, bias=bias)
        if not scale > 0.0:
            raise ValueError(f"ScaledLinear: scale must be positive, got {scale}")
        self.scale = float(scale)

    def forward(self, x):
        weight = self.weight if self.scale == 1.0 else mul(self.weight, self.scale)
        out = matmul(x, weight)
        if self.bias is not None:
            out = add(out, self.bias)
        return out


class Encoder(Sequential):
    """A `Sequential` that also reports the width of the row it emits.

    Heads are sized from `out_features`, so the number is derived from the
    tower's geometry once rather than restated at every call site.
    """

    def __init__(self, layers, out_features: int):
        super().__init__(*layers)
        self.out_features = int(out_features)


def image_encoder(obs_spec: ObsSpec, device, *, batch: int, hidden: int = 512,
                  tower=NATURE_CNN, input_needs_grad: bool = False) -> Encoder:
    """Conv tower -> flatten -> `Linear(hidden)` -> ReLU, for an image ObsSpec.

    Raises on a vector spec rather than quietly reshaping it: a 1-D
    observation has no spatial geometry for a stride to divide.

    `input_needs_grad=False` (the default) tells the FIRST stage not to build
    its dX machinery. The tower's input is the environment observation, which
    is a leaf, so nobody ever asks for its gradient, and dX is the part of a
    convolution that stops fitting in L1 first: building conv1's was the only
    reason this tower refused to construct at batch 64 and above. Dropping
    work that was never needed is what raises the usable minibatch to 160.

    The first stage's WEIGHT gradient is unaffected and still required; only
    the gradient with respect to its input is dropped, and a backward that
    nevertheless reaches for it raises naming the flag rather than returning
    nothing. Set it to True only for something that genuinely differentiates
    the observation itself, such as a saliency map, and expect the minibatch
    ceiling to drop back.
    """
    if obs_spec.kind != "image":
        raise ValueError(
            f"image encoder needs an image ObsSpec, got kind={obs_spec.kind!r} with "
            f"shape {obs_spec.shape}. A vector observation has no spatial geometry; "
            f"use the 'mlp' encoder."
        )
    if hidden < 1:
        raise ValueError(f"image encoder: hidden must be >= 1, got {hidden}")
    if not tower:
        raise ValueError("image encoder: tower must hold at least one ConvStage")
    in_shape = obs_spec.shape
    layers = []
    for index, stage in enumerate(tower):
        first = index == 0
        conv = Conv2d(
            in_shape, stage.out_channels, stage.kernel, device, batch=batch,
            stride=stage.stride, padding=stage.padding, fold=stage.fold,
            # Both of these are properties of being the stage the observation
            # enters, not of a position in a particular architecture: the scale
            # folds into the layer that sees unscaled pixels, and dX is
            # pointless for the layer whose input is a leaf. Every later stage
            # needs its dX to carry the gradient back to this one's weight.
            input_scale=obs_spec.scale if first else 1.0,
            needs_input_grad=input_needs_grad if first else True,
        )
        layers += [conv, ReLU()]
        in_shape = conv.out_shape
    channels, height, width = in_shape
    flatten = Flatten(height, width, channels)
    layers += [flatten, Linear(flatten.out_features, hidden, device), ReLU()]
    return Encoder(layers, hidden)


def image_encoder_s2d(obs_spec: ObsSpec, device, *, batch: int, hidden: int = 512,
                      input_needs_grad: bool = False) -> Encoder:
    """`nature_cnn` with conv1 space-to-depth folded. See `NATURE_CNN_S2D`."""
    return image_encoder(obs_spec, device, batch=batch, hidden=hidden,
                         tower=NATURE_CNN_S2D, input_needs_grad=input_needs_grad)


def vector_encoder(obs_spec: ObsSpec, device, *, batch: int, hidden=(256, 256)) -> Encoder:
    """`Linear -> ReLU` per entry of `hidden`, for a vector ObsSpec.

    `batch` is accepted for the one encoder signature every registry entry
    shares and is unused: a matmul's shapes do not depend on the batch size the
    way a prepared conv weight's layout does.
    """
    if obs_spec.kind != "vector":
        raise ValueError(
            f"mlp encoder needs a vector ObsSpec, got kind={obs_spec.kind!r} with "
            f"shape {obs_spec.shape}. Use the 'nature_cnn' encoder for images."
        )
    widths = tuple(int(w) for w in hidden)
    if not widths or min(widths) < 1:
        raise ValueError(f"mlp encoder: hidden must be positive widths, got {hidden}")
    layers = []
    in_features = obs_spec.size
    for index, width in enumerate(widths):
        first = index == 0
        layers += [
            ScaledLinear(in_features, width, device,
                         scale=obs_spec.scale if first else 1.0),
            ReLU(),
        ]
        in_features = width
    return Encoder(layers, widths[-1])


def identity_encoder(obs_spec: ObsSpec, device, *, batch: int) -> Encoder:
    """The leaf's own row, with no layer at all. Vector observations only.

    A tower of zero layers, which exists so that the fusion point of a
    state-action critic is a choice made from the specs rather than a fixed
    position in `build_sac_nets`. Naming it as the critic's encoder puts the
    raw observation next to the action, and `critic_hidden` then becomes the
    whole of Q: `concat(s, a) -> Linear -> ReLU -> ... -> Q`, which is how
    SAC's critic is written (Haarnoja et al. 2018) and what the published
    continuous-control numbers were measured with.

    Gated against a float64 reference at every depth it is shipped at. The
    comparison takes its relu gate pattern from the device wherever fp32
    cannot resolve the sign of a pre-activation, because it cannot: 4 to 7 of
    a 256-wide head's 32768 pre-activations per batch fall on the opposite
    side of the kink between the fp32 and the float64 forward, and each flip
    moves a whole rank-1 `joint[row] (x) dh[row, unit]` term into or out of
    the first weight's gradient. With the gates agreeing, the first weight of
    this critic measures Frobenius 2.3e-3 at one hidden layer, 3.3e-3 at two
    and 4.5e-3 at three over 8 seeds, which is the fp32 matmul floor rather
    than a depth effect. `tests/hardware/test_algos_sac_parity.py` gates one
    and two hidden layers here; `tests/_fakes/torch_twin.py` holds why the
    device deciding a gate inside that band is not the device grading its own
    homework.

    The unaligned fusion offset is exact on its own: `concat` reproduces a
    non-tile-multiple cut bit for bit and its backward slice lands each
    operand's gradient to about 5e-8 relative, which
    `tests/engine/test_recurrent_ops.py` gates.

    This is what 'auto' resolves to for a VECTOR leaf of a critic, by way of
    `ttrl.nets.fusion.DEFAULT_CRITIC_ENCODER_BY_KIND`, now that the gate
    measures the depth the topology wants. An image leaf of a critic still
    resolves to its own conv tower and fuses the action after it, which is the
    only place a pixel critic can fuse at all.

    Refuses an image spec rather than flattening one: a conv tower is the
    reason a pixel critic has to fuse its action after the encoder, so a
    pixel observation reaching here means the caller asked for a fusion this
    cannot give them, and the name of the encoder they want is in the message.

    Refuses `scale != 1.0` for a reason that is specific to having no layer:
    on every other path the scale folds into the first weight, and here the
    first weight is shared with the action's columns, so folding it would
    scale the action too. Normalise the leaf in the environment, or give the
    critic a tower of its own.
    """
    if obs_spec.kind != "vector":
        raise ValueError(
            f"identity encoder needs a vector ObsSpec, got kind={obs_spec.kind!r} "
            f"with shape {obs_spec.shape}. It exists so a state-action critic can "
            f"fuse its action onto the raw observation, which an image cannot do: "
            f"use the 'nature_cnn' encoder and fuse after it."
        )
    if obs_spec.scale != 1.0:
        raise ValueError(
            f"identity encoder: this leaf declares scale={obs_spec.scale}, and a "
            f"tower of no layers has no weight to fold it into. The next weight "
            f"is shared with whatever is concatenated onto this row, so folding "
            f"the scale there would scale that too. Scale the leaf in the "
            f"environment, or use the 'mlp' encoder."
        )
    return Encoder((), obs_spec.size)
