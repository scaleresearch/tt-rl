"""The algorithm-config contract.

`ttrl.algos.base` says what an agent is. This says what the object in
`TrainConfig.algo_cfg` is, which until now was written down nowhere: the train
layer read `network_batch`, `NETWORK_BATCH_FIELD` and `scale_for_replicas` off
it, and an author who had seen only the `Agent` protocol and the `ALGOS`
registry discovered each one from an `AttributeError` thrown out of a file
they did not write.

A `Protocol` plus a checker, the same pair as `Agent`/`check_agent` and
`ReplaySampler`/`check_sampler`: the trainer depends on a surface rather than
on a base class, and a config written outside this repo satisfies it without
importing anything.

`check_algo_config` runs from `TrainConfig.__post_init__`, so every one of
these is a construction-time refusal that names what to add, before a chip is
opened and before a checkpoint is written.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol, runtime_checkable

from ttrl.train.checkpoint.types import resolve, type_path

__all__ = ["AlgoConfig", "check_algo_config"]


@runtime_checkable
class AlgoConfig(Protocol):
    """What the train layer may read off an algorithm's config, and nothing more.

    A frozen dataclass. Frozen because a run's config is compared against the
    one a checkpoint carries; a dataclass because that comparison and the
    checkpoint round trip are both by field name (see
    `ttrl.train.checkpoint.tree`).

    Two optional class attributes are read with a default and need not be
    declared: `SHAPE_FIELDS`, the fields a parameter's shape is built from,
    which only sharpens the message `ttrl.algos.checkpoint` prints for a
    mismatched resume, and `UNSAVED_FIELDS`, fields a checkpoint must not
    carry. A `gamma` field is read if present, to give a reward normaliser the
    algorithm's discount.
    """

    #: The name of the field `network_batch` returns, for a refusal to quote.
    NETWORK_BATCH_FIELD: str

    @property
    def network_batch(self) -> int:
        """The batch every network is built for, and so the widest acting call.

        A convolution fixes its batch at construction, so this bounds
        `env.n_envs` and the eval width; see
        `ttrl.algos.scaling.check_network_batch`.
        """
        ...

    def scale_for_replicas(self, n: int) -> "AlgoConfig":
        """This config as each of `n` replicas should see it.

        An n-chip run gives each replica 1/n of the frames, so every field
        that is a budget measured in frames is divided.
        `ttrl.divide_for_replicas` does it, given the field names. Return `self` at n == 1.
        """
        ...


def check_algo_config(cfg: Any, algo: str) -> None:
    """Assert that `cfg` carries the whole `AlgoConfig` surface."""
    if cfg is None:
        return
    name = type(cfg).__name__
    if not dataclasses.is_dataclass(cfg) or isinstance(cfg, type):
        raise TypeError(
            f"algo_cfg for {algo!r} is {name}, which is not a dataclass "
            f"INSTANCE. A run compares its config against the one its "
            f"checkpoint carries, and stores it, both by field name. Declare "
            f"it with @dataclasses.dataclass(frozen=True) and pass an "
            f"instance."
        )
    for attr, add in (
        ("NETWORK_BATCH_FIELD",
         "NETWORK_BATCH_FIELD = \"batch_size\", naming the field below"),
        ("network_batch",
         "a `network_batch` property returning that field"),
        ("scale_for_replicas",
         "a `scale_for_replicas(n)` returning this config with every "
         "frame-budget field divided by n, which "
         "ttrl.divide_for_replicas does given their names"),
    ):
        if getattr(cfg, attr, None) is None:
            raise TypeError(
                f"{name} is missing {attr!r}, which ttrl.AlgoConfig requires "
                f"of every algo_cfg: the train layer reads the acting width "
                f"and the per-replica budgets off it without knowing which "
                f"algorithm it drives. Add {add}. "
                f"`ttrl.check_algo_config(cfg, 'myalgo')` checks all of them."
            )
    if not callable(cfg.scale_for_replicas):
        raise TypeError(
            f"{name}.scale_for_replicas is not callable. ttrl.AlgoConfig "
            f"declares it as a method taking the replica count; make it "
            f"`def scale_for_replicas(self, n): ...`."
        )

    path = type_path(type(cfg))
    if not path.startswith("ttrl.") and resolve(path) is not type(cfg):
        raise TypeError(
            f"{name} is defined outside ttrl and is not registered as a "
            f"checkpoint type, so this run would train and then fail at its "
            f"first checkpoint. A checkpoint names a type by its path and must "
            f"never import a module the file chose, so a type outside this "
            f"library is stored only once its own package has registered it. "
            f"In {type(cfg).__module__}, decorate {name} with "
            f"@ttrl.register_checkpoint_type above the @dataclass line."
        )
