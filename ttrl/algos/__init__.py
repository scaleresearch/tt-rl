"""The algorithms, and the registry that resolves one by name.

Lazy, like `ttrl.engine` and `ttrl.nets`: importing this package imports no
ttnn and opens no device, so a launcher can scope the visible chips before ttnn
initialises.

The three configs resolve from their own modules rather than from their
packages, so reaching one costs no ttnn either. `ttrl.load_config` runs before
any chip is scoped and a run's config file constructs a config at import time,
which is exactly the path that must stay clear.
"""

from __future__ import annotations

import importlib
from typing import Any

from ttrl.algos.base import Agent, AutoresetTracker, Transition, check_agent
from ttrl.algos.config import AlgoConfig, check_algo_config
from ttrl.algos.registry import ALGOS, resolve_and_build

_LAZY = {
    "DQN": "dqn.agent",
    "DQNConfig": "dqn.config",
    "PPO": "ppo.agent",
    "PPOConfig": "ppo.config",
    "SAC": "sac.agent",
    "SACConfig": "sac.config",
}

__all__ = [
    "ALGOS",
    "Agent",
    "AlgoConfig",
    "AutoresetTracker",
    "Transition",
    "check_agent",
    "check_algo_config",
    "resolve_and_build",
    *sorted(_LAZY),
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"ttrl.algos.{module}"), name)


def __dir__() -> list[str]:
    return sorted(__all__)
