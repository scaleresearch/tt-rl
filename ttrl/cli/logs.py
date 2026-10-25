"""Quietening the stack's own log streams, for the one process that owns them.

ttnn logs twice: through loguru on the Python side and through tt-metal's own
logger on the C++ side, both verbose by default. Importing ttnn prints a
`Config{...}` dump before a run has done anything, and opening a device adds
UMD cluster chatter, an "Unknown motherboard" warning and a teardown trace.
A run's own progress is then a handful of lines inside thousands, which is
how a 6,000-frame run ends up looking like it printed nothing.

Replacing loguru's handler and setting `TT_LOGGER_LEVEL` are process-wide
acts, so they live under `ttrl.cli`, the only part of the tree that owns a
process. The library never calls this and `import ttrl` never reaches it.

Both halves have to happen before ttnn is imported: tt-metal reads its level
once at load, and loguru's default handler has to be replaced before ttnn's
import-time records are emitted.
"""

from __future__ import annotations

import os
import sys

from loguru import logger

__all__ = ["LEVELS", "quiet_stack_logs"]

#: Accepted levels, quietest last. The same four names reach both loggers.
LEVELS = ("debug", "info", "warning", "error")


def quiet_stack_logs(level: str) -> None:
    """Hold ttnn's Python and C++ log streams at `level`.

    Call before anything imports ttnn. `error` is what a training run wants:
    its own progress line is the output, and ttnn's warnings about this box
    ("Unknown motherboard") say nothing about the run. `debug` is how a
    hardware problem gets diagnosed, and `ttrl train --ttnn-logs debug` is
    the one way to ask for it.
    """
    if level not in LEVELS:
        raise ValueError(
            f"unknown ttnn log level {level!r}. Pass one of {list(LEVELS)}."
        )
    os.environ["TT_LOGGER_LEVEL"] = level
    os.environ["TT_METAL_LOGGER_LEVEL"] = level

    logger.remove()
    logger.add(sys.stderr, level=level.upper())
