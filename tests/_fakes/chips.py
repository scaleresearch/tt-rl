"""The two-idle-chip gate the multi-chip hardware tests share.

Import this only from a `tt`-marked module: the chip enumeration and the
`fuser` probe run at import time, which is the only moment at which a
module-level `skipif` can still be decided.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from ttrl.engine.device import physical_chips, visible_chip_ids


def candidate_chips(n: int = 1) -> list[int]:
    """The first `n` PHYSICAL chip ids these tests may use.

    `physical_chips()` reports every node on the box regardless of
    `TT_VISIBLE_DEVICES`, so taking its first two picked chips 0,1 even when
    the process was scoped to 2,3 -- and if anything else held 0,1 (the normal
    case while another training job runs) every gated test SKIPPED instead of
    running on the chips it was actually given. Silent skips on the multi-chip
    path are worse than useless: they read as green. Prefer the visible set.

    These are physical ids, which is what `DeviceConfig.chips` takes. A
    `device_id` handed to ttnn is a LOCAL index into the visible set instead;
    `ttrl.engine.device.local_index` converts.
    """

    return (visible_chip_ids() or physical_chips())[:n]


def two_candidate_chips() -> list[int]:
    """The two chips the multi-chip tests may use."""
    return candidate_chips(2)


def fuser_free(chip: int) -> bool:
    try:
        out = subprocess.run(
            ["fuser", f"/dev/tenstorrent/{chip}"], capture_output=True, text=True, timeout=10
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True
    me = os.getpid()
    return not any(int(t) != me for t in out.split() if t.isdigit())


TWO_CHIPS = two_candidate_chips()

TWO_IDLE_CHIPS_AVAILABLE = len(TWO_CHIPS) >= 2 and all(fuser_free(c) for c in TWO_CHIPS)


def two_idle_chips_gate(reason="needs 2 idle Tenstorrent chips"):
    return pytest.mark.skipif(not TWO_IDLE_CHIPS_AVAILABLE, reason=reason)


requires_two_idle_chips = two_idle_chips_gate()
