"""The child half of doctor's chip probe: open ONE chip and write what it finds.

Run as `python -m ttrl.cli.probes.chip_report <chip> <out.json>` by
ttrl.cli.probes.hardware, never imported by the parent. It exists as a separate
process because TT_VISIBLE_DEVICES is read once, when ttnn initialises: an
in-process probe cannot scope itself, and ttnn.open_device on an unscoped
process makes UMD map hugepages and take the driver mutexes for EVERY local
chip, so doctor would disturb the whole board while warning the user not to.

The result goes to a file rather than stdout because ttnn and UMD write freely
to both streams, and a diagnostic must not have to guess which line is its own.
"""

from __future__ import annotations

import json
import sys
import traceback

import numpy as np

from ttrl.engine import Tensor, acquire, capture, ops
from ttrl.engine.tensor import to_device, to_host

#: 32x32 is one tile, the smallest shape that exercises a real kernel.
_PROBE_SHAPE = (32, 32)


def probe_chip() -> dict:
    """Open the one visible chip, describe it, and capture and replay one op."""

    # Scoped, the only device id ttnn accepts is local index 0.
    with acquire(chip_id=0, wait=False) as (device, profile):
        return {
            "arch": profile.arch,
            "grid": list(profile.grid),
            "dram_grid": list(profile.dram_grid),
            "tile": list(profile.tile_shape),
            "num_devices": profile.num_devices,
            "trace": _trace_result(device),
        }


def _trace_result(device) -> dict:
    """Capture and replay one real op, and check the replay actually wrote.

    Trace capture is the most environment-sensitive thing ttrl does and the one
    whose failure surfaces furthest from its cause, so doctor measures it
    instead of asserting it.
    """

    x = Tensor(to_device(np.full(_PROBE_SHAPE, 2.0, dtype=np.float32), device))

    def record():
        return ops.add(ops.mul(x, 3.0), 1.0)

    # Warm the program cache first: capture records commands, so a first-time
    # kernel compile inside the region is not recorded.
    to_host(record().data)
    try:
        trace, out = capture(device, record)
    except Exception as exc:
        return {"ok": False, "stage": "capture",
                "error": f"{type(exc).__name__}: {exc}"}
    with trace:
        trace.replay(blocking=True)
        got = to_host(out.data)
    if not np.allclose(got, 7.0):
        return {"ok": False, "stage": "replay",
                "error": f"replay wrote {got.min()}..{got.max()}, expected 7.0"}
    return {"ok": True, "ops": 2}


def main(argv: list[str]) -> int:
    chip, out_path = argv[0], argv[1]
    try:
        result = {"ok": True, "chip": int(chip), **probe_chip()}
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        result = {"ok": False, "chip": int(chip),
                  "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()}
    with open(out_path, "w") as fh:
        json.dump(result, fh)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
