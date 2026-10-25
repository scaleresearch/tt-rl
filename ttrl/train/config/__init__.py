"""Typed configuration for a training run.

`schema.py` holds `TrainConfig`, `device.py` and `env.py` the two halves it is
built from, `coerce.py` the dict/dataclass conversion, `overrides.py`
dotted-path overrides and `loader.py` the config module loader. An algorithm's hyperparameters are not
here: they are the algorithm's own config object, carried by
`TrainConfig.algo_cfg`.
"""

from __future__ import annotations

from ttrl.train.config.coerce import config_to_dict
from ttrl.train.config.loader import load_config
from ttrl.train.config.overrides import apply_overrides
from ttrl.train.config.schema import DeviceConfig, EnvConfig, TrainConfig

__all__ = [
    "DeviceConfig",
    "EnvConfig",
    "TrainConfig",
    "apply_overrides",
    "config_to_dict",
    "load_config",
]
