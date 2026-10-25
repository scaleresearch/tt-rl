"""Spec-driven network construction.

An observation is a group of named leaves. One encoder is built per leaf,
chosen by that leaf's `ObsSpec.kind` and sized from its `ObsSpec.shape`, and
their features are concatenated; a head is chosen by `ActionSpec.kind` and
sized from `ActionSpec.width`. Nothing in
this package branches on which environment suite produced the specs, and no
file in it names an observation geometry.

Lazy, like `ttrl.engine`: importing this package imports no ttnn and opens no
device, so a launcher can still scope the visible chips before ttnn
initialises.
"""

from __future__ import annotations

import importlib
from typing import Any

from ttrl.nets.registry import ENCODERS, Registry

_LAZY = {
    "ConcatEncoder": "fusion",
    "ConvStage": "encoders",
    "Encoder": "encoders",
    "NATURE_CNN": "encoders",
    "NATURE_CNN_S2D": "encoders",
    "image_encoder_s2d": "encoders",
    "ScaledLinear": "encoders",
    "build_encoder": "multi",
    "MultiEncoder": "multi",
    "DEFAULT_ENCODER_BY_KIND": "encoders",
    "DEFAULT_CRITIC_ENCODER_BY_KIND": "fusion",
    "DEFAULT_ENCODER_BY_ROLE": "multi",
    "image_encoder": "encoders",
    "vector_encoder": "encoders",
    "CategoricalHead": "heads",
    "GaussianHead": "heads",
    "QHead": "heads",
    "ValueHead": "heads",
    "build_policy_head": "heads",
    "GroupBuffer": "obs",
    "ObsBuffer": "obs",
    "ObsPair": "obs",
    "ObsViews": "obs",
    "obs_rows": "obs",
    "obs_tensor_shape": "obs",
    "UnnamedEncoder": "choice",
    "coerce_encoder": "choice",
    "resolve_encoder": "choice",
    "ActingNet": "acting",
    "check_encoder_layout": "acting",
    "net_at_batch": "acting",
    "ActorCritic": "models",
    "ContinuousCritic": "models",
    "EncoderHead": "models",
    "SacNets": "models",
    "RecurrentActorCritic": "recurrent",
    "build_recurrent_actor_critic": "recurrent",
    "build_actor_critic": "build",
    "build_q_net": "build",
    "build_sac_nets": "build",
}

#: The layers an encoder of one's own is written out of, reachable from the
#: package whose seam that encoder plugs into. `Encoder` and `ScaledLinear`
#: were here with no `Module` to subclass and no activation to put between two
#: `Linear`s, so a hand-written tower had to name `ttrl.engine.layers`, a
#: module nothing advertises. They stay owned by `ttrl.engine`: these are
#: lazy aliases, and `ttrl.engine.nn.Module` is still the one definition.
#: Not re-exported at the top level, where `ttrl.engine`'s names deliberately
#: do not appear (see `MOVED_TO_ENGINE` in `tests/test_public_api.py`).
_ENGINE_LAZY = {
    "Module": "ttrl.engine.nn",
    "ModuleDict": "ttrl.engine.nn",
    "ModuleList": "ttrl.engine.nn",
    "Linear": "ttrl.engine.layers",
    "LayerNorm": "ttrl.engine.layers",
    "ReLU": "ttrl.engine.layers",
    "Tanh": "ttrl.engine.layers",
    "Sequential": "ttrl.engine.layers",
    "Flatten": "ttrl.engine.layers",
    "Conv2d": "ttrl.engine.conv",
}

__all__ = ["ENCODERS", "Registry", *sorted(set(_LAZY) | set(_ENGINE_LAZY))]


def __getattr__(name: str) -> Any:
    engine_module = _ENGINE_LAZY.get(name)
    if engine_module is not None:
        return getattr(importlib.import_module(engine_module), name)
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"ttrl.nets.{module}"), name)


def __dir__() -> list[str]:
    return sorted(__all__)
