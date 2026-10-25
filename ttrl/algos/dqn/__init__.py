"""Double DQN.

Lazy, like `ttrl.engine` and `ttrl.envs`: `DQNConfig` is ttnn-free and imports
eagerly, because a run's config file constructs one at import time and
`ttrl.load_config` runs before any chip is scoped. The agent pulls ttnn, so it
resolves on first attribute access. A process that touches ttnn before
TT_VISIBLE_DEVICES has been applied does not open one chip: it maps hugepages
and takes the driver mutex for every chip on the board.
"""

from __future__ import annotations

import importlib
from typing import Any

from ttrl.algos.dqn.config import DQNConfig

_LAZY = {"DQN": "agent"}

__all__ = ["DQN", "DQNConfig"]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"ttrl.algos.dqn.{module}"), name)


def __dir__() -> list[str]:
    return sorted(__all__)
