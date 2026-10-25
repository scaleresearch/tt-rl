"""Which conv tower a checkpoint's image actor was built with, and its weights.

A checkpoint stores each stage's weight and bias but not its stride or
padding, which are properties of the tower the run's encoder named. They are
read off that name through `ttrl.nets.towers.TOWERS`, which imports no ttnn,
and every stage's weight is then checked against the stage it claims to be,
so a tower this module misreads is refused rather than exported with the
wrong strides.
"""

from __future__ import annotations

import numpy as np

from ttrl.nets.towers import DEFAULT_ENCODER_BY_KIND, TOWERS

__all__ = ["split_tower"]


def split_tower(trunk: list, obs_spec, leaf: str, algo_config, path: str,
                require) -> tuple[list, list, list]:
    """`(stages, geometry, linears)` for an actor trunk read by name.

    `stages` holds each conv's `(weight, bias)` with `ObsSpec.scale` folded
    into the first weight, `geometry` each conv's stride and padding, and
    `linears` the `Linear` trunk after the flatten. A vector actor has no
    conv, so its trunk comes back whole.
    """
    convs = [layer for layer in trunk if layer[0].ndim == 4]
    linears = trunk[len(convs):]
    require(all(layer[0].ndim == 2 for layer in linears), path,
            "has an actor trunk whose convolutions do not all precede its "
            "Linear layers, which is not a tower this format describes")
    if obs_spec.kind != "image":
        require(not convs, path,
                f"has convolution weights in an actor reading {obs_spec.kind} "
                f"observations")
        return [], [], linears
    name, tower = _tower(algo_config, leaf, path, require)
    require(len(convs) == len(tower), path,
            f"has {len(convs)} conv stages where its {name!r} tower has "
            f"{len(tower)}")
    channels, height, width = obs_spec.shape
    stages, geometry = [], []
    for index, ((weight, bias), stage) in enumerate(zip(convs, tower)):
        expected = (stage.out_channels, channels, stage.kernel, stage.kernel)
        require(weight.shape == expected, path,
                f"has conv stage {index} weight of shape {weight.shape} where "
                f"its {name!r} tower builds {expected}")
        height = (height + 2 * stage.padding - stage.kernel) // stage.stride + 1
        width = (width + 2 * stage.padding - stage.kernel) // stage.stride + 1
        channels = stage.out_channels
        scale = np.float32(obs_spec.scale if index == 0 else 1.0)
        stages.append((weight * scale, bias.reshape(-1)))
        geometry.append({"stride": stage.stride, "padding": stage.padding})
    features = channels * height * width
    require(bool(linears) and linears[0][0].shape[0] == features, path,
            f"has a conv tower ending in {channels}x{height}x{width} = "
            f"{features} features, and no Linear reading that many after it")
    return stages, geometry, linears


def _tower(algo_config, leaf: str, path: str, require) -> tuple[str, tuple]:
    encoder = getattr(algo_config, "encoder", "auto")
    if isinstance(encoder, dict):
        encoder = encoder.get(leaf, "auto")
    name = DEFAULT_ENCODER_BY_KIND["image"] if encoder == "auto" else encoder
    require(isinstance(name, str) and name in TOWERS, path,
            f"was trained with the image encoder {name!r}, and this exporter "
            f"reads the strides of {sorted(TOWERS)}. A custom tower's geometry "
            f"is not in the checkpoint, so it cannot be exported by guessing")
    kwargs = getattr(algo_config, "encoder_kwargs", None) or {}
    return name, tuple(kwargs.get("tower", TOWERS[name]))
