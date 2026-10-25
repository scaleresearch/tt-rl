"""Soft Actor-Critic, for continuous control.

Lazy for the reason in `ttrl.algos.dqn`: the config is ttnn-free and eager,
everything that pulls ttnn resolves on first access.
"""

from __future__ import annotations

import importlib
from typing import Any

from ttrl.algos.sac.config import SACConfig

_LAZY = {
    "SAC": "agent",
    "SquashedGaussian": "policy",
}

__all__ = ["SACConfig", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"ttrl.algos.sac.{module}"), name)


def __dir__() -> list[str]:
    return sorted(__all__)
