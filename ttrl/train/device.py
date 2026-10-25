"""The one public way to get an open chip: `ttrl.device_for(config.device)`.

Opening a chip is two steps that must happen in this order: scope the process
to one chip (`TT_VISIBLE_DEVICES`, read by ttnn at initialisation and never
again), then open it. A caller doing it by hand has to know that the first
step is only effective before the first `import ttnn`, and no signature can
say so. Here the engine is imported inside the body, after the scoping call,
so the order is a property of this function rather than a rule a caller has
to have read.

ONE CHIP. This entry point scopes to a single chip and opens that chip, and
it refuses a `DeviceConfig` asking for more rather than scoping wider than it
opens. The refusal is this function's own and does not rest on
`scope_for_run` returning early for a multi-chip config: a process scoped to
N chips with one chip opened holds N and uses one, which no error anywhere
would report.

`ttrl.engine.acquire` and `ttrl.engine.scope_to_chip` stay where they are and
stay out of the top-level surface: this is the blessed spelling, and the
engine's own surface remains free to move.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

from ttrl.engine.device import acquire
from ttrl.train.config.device import DeviceConfig
from ttrl.train.scoping import scope_for_run

__all__ = ["device_for"]


@contextlib.contextmanager
def device_for(device_config: DeviceConfig) -> Iterator[tuple[Any, Any]]:
    """Scope this process to the chip `device_config` asks for, and open it.

    Yields what `ttrl.engine.acquire` yields, `(device, profile)`, for exactly
    one chip, and closes and releases it on the way out. A multi-chip config
    is refused here; a run over replicas opens its own devices.

    This is what `ttrl.run` does for a training run; a script that scores,
    plays or exports a checkpoint calls it directly and passes the same
    `DeviceConfig`, including the chip id, the L1 scratch size and how long to
    wait for a busy chip.
    """
    if not isinstance(device_config, DeviceConfig):
        raise TypeError(
            f"device_for takes a DeviceConfig, got {type(device_config).__name__}. "
            f"A TrainConfig carries one as its `device` field, so pass "
            f"`config.device`; a script that only names a chip builds one with "
            f"`ttrl.DeviceConfig(chip=N)`."
        )
    if device_config.chips != 1:
        raise ValueError(
            f"device_for opens exactly one chip, and this DeviceConfig asks for "
            f"{device_config.chips!r}. A run over several chips opens them "
            f"itself, as replicas: pass the config to ttrl.run, or build a "
            f"ttrl.Trainer with it."
        )
    scope_for_run(device_config)
    # After the scoping above, never before: this import reaches ttnn, which
    # reads TT_VISIBLE_DEVICES once, at initialisation.

    with acquire(cfg=device_config.engine_config(),
                 timeout=device_config.acquire_timeout) as opened:
        yield opened
