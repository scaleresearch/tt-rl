"""Optimizers. Every update runs on device; nothing round-trips to host."""

from ttrl.engine.optim.adam import Adam
from ttrl.engine.optim.base import Optimizer
from ttrl.engine.optim.scalars import adam_fused_scalars

__all__ = ["Adam", "Optimizer", "adam_fused_scalars"]
