"""Which chips exist, which are free, and which process holds the rest.

Everything here comes from ttrl.engine.device's own liveness logic, the same
code acquire() gates on, so doctor and a training run cannot disagree about
whether a chip is busy.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from ttrl.cli.report import FAIL, NOTE, OK, WARN, Line
from ttrl.engine import device as dev


def _holder(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return f"PID {pid} (already gone)"
    argv = [tok for tok in raw.decode("utf-8", "replace").split("\0") if tok]
    return f"PID {pid} ({' '.join(argv)[:120] or 'unknown command'})"


def probe() -> Iterator[Line]:
    physical = dev.physical_chips()
    if not physical:
        yield Line(
            FAIL, "driver nodes", "no /dev/tenstorrent/<n> found",
            action="no chip is visible to this machine. Check the module is "
                   "loaded (`lsmod | grep tenstorrent`) and that you are in the "
                   "group that owns /dev/tenstorrent.",
        )
        return
    yield Line(OK, "driver nodes", ", ".join(f"/dev/tenstorrent/{c}" for c in physical))

    visible = dev.visible_chip_ids()
    if visible is None:
        yield Line(
            NOTE, "visibility", f"{dev.VISIBLE_DEVICES_ENV} unset, all chips visible",
        )
    else:
        local = ", ".join(f"{i} -> physical {p}" for i, p in enumerate(visible))
        yield Line(OK, "visibility", f"{dev.VISIBLE_DEVICES_ENV}="
                                     f"{os.environ[dev.VISIBLE_DEVICES_ENV]}, "
                                     f"ttnn device_id {local}")

    free = 0
    for chip_id in dev.enumerate_chips():
        line, is_free = _chip_line(dev, chip_id)
        free += is_free
        yield line

    if free == 0:
        yield Line(
            FAIL, "free chips", "none",
            action="every visible chip is held. `ttrl train` waits for one by "
                   "default; to run now, stop one of the processes listed above, "
                   "or scope to another chip with "
                   f"{dev.VISIBLE_DEVICES_ENV}=<n>.",
        )
    else:
        yield Line(OK, "free chips", str(free))


def _chip_line(dev, chip_id: int):
    label = f"chip {chip_id} (/dev/tenstorrent/{dev.physical_chip_id(chip_id)})"
    try:
        status = dev.status(chip_id)
    except dev.ChipLivenessCheckFailed as exc:
        # Liveness refuses to guess, and so does doctor: unconfirmed is not free.
        return Line(
            FAIL, label, f"liveness unconfirmed: {exc.problem}", action=exc.fix,
        ), False
    if status.foreign_pids:
        held = "; ".join(_holder(pid) for pid in status.foreign_pids)
        return Line(
            WARN, label, f"held by {held}", essential=False,
            action="wait for it, or stop that process. A chip is exclusive: "
                   "opening it anyway fails, and sometimes wedges.",
        ), False
    if status.locked_by is not None:
        who = "an unreadable PID" if status.locked_by < 0 else f"PID {status.locked_by}"
        return Line(
            WARN, label, f"ttrl advisory lock held by {who}", essential=False,
            action="another ttrl run has claimed this chip but has not opened it "
                   "yet. Wait, or pick another chip.",
        ), False
    return Line(OK, label, "free"), True
