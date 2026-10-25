"""What only a real chip can answer: arch, the Tensix grid, and trace capture.

The parent process here opens NOTHING. It picks a free chip from the liveness
survey and hands it to a scoped child (ttrl.cli.probes.chip_report), because
ttnn.open_device on an unscoped process makes UMD map hugepages and take the
driver mutexes for every local chip on the board. Doctor is the first command a
new user runs and the one that warns about exactly that hazard, so it must not
be the thing that does it.

Two consequences of the fd trap, both load-bearing. A process holding a device
node leaves fds on it, so if the parent had opened a chip the child's own
liveness probe would see the parent as a foreign holder and refuse the chip it
was just handed; the parent therefore checks that it holds nothing. And the
child takes the advisory lock itself, so the parent must not take it either.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator

from ttrl.cli.probes._ttnn import unavailable
from ttrl.cli.report import FAIL, OK, Line
from ttrl.engine import device as dev

#: A cold JIT cache compiles kernels on the first capture, which is slow but
#: bounded. Past this the child is not working, it is stuck.
PROBE_TIMEOUT_S = 300.0

_CHILD = "ttrl.cli.probes.chip_report"


def _skipped(reason: str, action: str) -> Line:
    return Line(FAIL, "chip probe", f"not run: {reason}", action=action)


def probe() -> Iterator[Line]:
    reason = unavailable()
    if reason is not None:
        yield _skipped(f"ttnn is unavailable ({reason}), so no chip can be opened",
                       "install ttnn; see the versions section above.")
        return

    if dev.device_already_open():
        yield _skipped(
            "this process already holds a /dev/tenstorrent node open, so the "
            "scoped child would see us as the chip's holder and refuse it",
            "run `ttrl doctor` on its own rather than from inside a process "
            "that has already opened a chip.",
        )
        return

    free = [c for c in dev.enumerate_chips() if dev.status(c).free]
    if not free:
        yield _skipped(
            "no free chip",
            "wait for one of the chips above to be released, then rerun. "
            "Nothing below this line can be checked without silicon.",
        )
        return

    physical = dev.physical_chip_id(free[0])
    untouched = [dev.physical_chip_id(c) for c in dev.enumerate_chips()
                 if dev.physical_chip_id(c) != physical]
    yield Line(
        OK, "probed chip", f"{physical} (/dev/tenstorrent/{physical}), opened in a "
        f"process scoped to it alone" + (
            f", leaving {', '.join(str(c) for c in untouched)} untouched"
            if untouched else ""),
    )
    yield from _child_lines(physical)


def _child_lines(physical: int) -> Iterator[Line]:
    try:
        env = dev.chip_env(physical, require_descriptor=True)
    except dev.ChipScopeError as exc:
        yield _skipped(
            f"cannot build a one-chip environment: {exc}",
            f"set {dev.MESH_GRAPH_DESC_ENV} to "
            f"{dev.MESH_GRAPH_DESC_NAME} inside the installed ttnn package, or "
            "reinstall ttnn if it is missing.",
        )
        return

    with tempfile.TemporaryDirectory(prefix="ttrl-doctor-") as scratch:
        out = Path(scratch) / "chip.json"
        argv = [sys.executable, "-m", _CHILD, str(physical), str(out)]
        try:
            done = subprocess.run(
                argv, env=env, capture_output=True, text=True,
                timeout=PROBE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            yield _skipped(
                f"the scoped probe did not finish in {PROBE_TIMEOUT_S:.0f}s",
                "a probe that hangs this long is usually a wedged chip. Check "
                f"`fuser /dev/tenstorrent/{physical}` and SIGKILL any process "
                "that holds it.",
            )
            return
        if not out.exists():
            yield _skipped(
                f"the scoped probe exited {done.returncode} without reporting",
                "its last words were:\n"
                + (_tail(done.stderr) or "(nothing on stderr)"),
            )
            return
        result = json.loads(out.read_text())

    if not result["ok"]:
        yield Line(
            FAIL, f"chip {physical}", f"could not be opened: {result['error']}",
            action="ttrl cannot open this chip, so it cannot train on it. If the "
                   "error came from inside trace capture the chip may now be "
                   "wedged: check `fuser /dev/tenstorrent/" + str(physical)
                   + "` and SIGKILL the holder.",
        )
        return
    yield from _found_lines(physical, result)


def _found_lines(physical: int, result: dict) -> Iterator[Line]:
    yield Line(OK, f"chip {physical} arch", result["arch"])
    grid, dram, tile = result["grid"], result["dram_grid"], result["tile"]
    yield Line(
        OK, f"chip {physical} tensix grid",
        f"{grid[0]}x{grid[1]} compute cores, {dram[0]}x{dram[1]} DRAM banks, "
        f"{tile[0]}x{tile[1]} tile, {result['num_devices']} device(s) visible",
    )
    trace = result["trace"]
    if trace["ok"]:
        yield Line(OK, "trace capture",
                   f"captured {trace['ops']} ops on chip {physical}, replayed, "
                   "result exact")
        return
    yield Line(
        FAIL, "trace capture", f"failed at {trace['stage']}: {trace['error']}",
        action="training runs traced steps, so this box cannot train until this "
               "works. The chip may now be wedged: check `fuser "
               f"/dev/tenstorrent/{physical}` and SIGKILL any process holding it.",
    )


def _tail(text: str, lines: int = 6) -> str:
    kept = [ln for ln in (text or "").splitlines() if ln.strip()][-lines:]
    return "\n".join(kept)
