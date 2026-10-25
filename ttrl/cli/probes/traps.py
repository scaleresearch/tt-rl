"""The two environment traps that cost the most debugging time on this stack.

Both are things the stack does not tell you. They are reported every run,
whether or not they currently bite, because the point is that nobody should
have to rediscover them.
"""

from __future__ import annotations

import os
from typing import Iterator

from ttrl.cli.probes._ttnn import unavailable
from ttrl.cli.report import FAIL, NOTE, OK, Line
from ttrl.engine import device as dev

WEDGE_NOTE = (
    "an exception thrown between ttnn.begin_trace_capture and "
    "end_trace_capture leaves the device in the recording state and WEDGES the "
    "chip: it stays held until the process is SIGKILLed."
)


def _located() -> str | None:
    """Where the single-chip descriptor is, or None if it cannot be looked up.

    The lookup goes through ttnn's own install location, so with ttnn
    unavailable there is no path to name and the trap is reported without one.
    """

    return None if unavailable() else dev.mesh_graph_descriptor_path()


def descriptor_required() -> bool:
    """Is this process scoped to exactly one chip, where the trap fires?"""

    visible = dev.visible_chip_ids()
    return visible is not None and len(visible) == 1


def descriptor_line() -> Line:
    """Whether TT_MESH_GRAPH_DESC_PATH is set where ttnn requires it.

    With one visible chip on a paired board, the cluster shape matches no
    built-in fabric descriptor and ttnn.open_device dies with "Custom fabric
    mesh graph descriptor path must be specified".
    """

    name = dev.MESH_GRAPH_DESC_ENV
    current = os.environ.get(name, "")
    located = _located()
    if not descriptor_required():
        return Line(
            NOTE, name,
            f"not required while unscoped; set it to {located or dev.MESH_GRAPH_DESC_NAME} "
            f"whenever {dev.VISIBLE_DEVICES_ENV} names exactly one chip",
        )
    if current:
        exists = os.path.exists(current)
        return Line(
            OK if exists else FAIL, name,
            current if exists else f"{current} does not exist",
            action="" if exists else
            f"point it at a real file: export {name}={located}" if located else
            f"{dev.MESH_GRAPH_DESC_NAME} was not found in the installed ttnn "
            "package either; reinstall ttnn.",
        )
    return Line(
        FAIL, name,
        f"unset, but {dev.VISIBLE_DEVICES_ENV} names one chip",
        action=(f"export {name}={located}" if located else
                f"{dev.MESH_GRAPH_DESC_NAME} was not found in the installed ttnn "
                "package; reinstall ttnn or set the variable by hand") +
               "\nwithout it ttnn.open_device dies with \"Custom fabric mesh "
               "graph descriptor path must be specified\".",
    )


def probe() -> Iterator[Line]:
    yield descriptor_line()
    yield Line(NOTE, "trace capture wedge", WEDGE_NOTE)
