"""The training layer: configuration, the loop, checkpoints, and evaluation.

Lazy, like the rest of the library. Importing this package imports no ttnn,
opens no device, reads no environment and configures no logging, so it costs
nothing on a machine with no hardware.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY = {
    "DeviceConfig": "config",
    "EnvConfig": "config",
    "TrainConfig": "config",
    "apply_overrides": "config",
    "config_to_dict": "config",
    "load_config": "config",
    "L0": "checkpoint",
    "L1": "checkpoint",
    "load_checkpoint": "checkpoint",
    "save_checkpoint": "checkpoint",
    "agent_from_checkpoint": "restore",
    "config_from_checkpoint": "restore",
    "evaluate": "evaluate",
    "evaluate_checkpoint": "evaluate",
    "record_video": "record",
    "JsonlLogger": "logging",
    "Logger": "logging",
    "MultiLogger": "logging",
    "build_logger": "logging",
    "NullLogger": "logging",
    "TensorBoardLogger": "logging",
    "WandbLogger": "logging",
    "TrainHooks": "hooks",
    "AnyOf": "stop",
    "RunState": "stop",
    "StopOnNoImprovement": "stop",
    "StopOnRewardThreshold": "stop",
    "BestScore": "checkpoint",
    "write_run_outcome": "provenance",
    "Trainer": "trainer",
    "run": "run",
    "StdoutLogger": "logging",
    "scope_for_run": "scoping",
    "run_metadata": "provenance",
    "write_run_metadata": "provenance",
    "build_agent": "builders",
    "build_env": "builders",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"ttrl.train.{module}"), name)


def __dir__() -> list[str]:
    return sorted(__all__)
