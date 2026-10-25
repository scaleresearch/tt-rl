"""Per-process and per-test preconditions for the test suite.

Chip scoping happens at conftest import time, before anything else can
`import ttnn`, which is the only moment at which it can still have an effect.
`ttrl/engine/device/scope.py` spells out the UMD behaviour: a process that
touches ttnn with no `TT_VISIBLE_DEVICES` set does not open one chip, it maps
hugepages and takes the `CHIP_IN_USE_<n>_PCIe` driver mutex for *every* local
chip on the board. That is true of `pytest -m tt` and, because several test
modules import ttnn at module scope, equally true of `pytest -m "not tt"`.

So the suite scopes itself to one chip, chosen from the ones nobody else is
holding. An explicit `TT_VISIBLE_DEVICES` from the caller always wins.

There is no flag-pinning fixture any more. The process-wide mode flags it
existed for (an active precision, a latched Adam kernel, an in-trace-capture
marker) are gone: behaviour is carried by values a caller constructs, so
nothing on the training path writes a module global for a later test to
inherit.
"""

from __future__ import annotations

import gc
import os
import subprocess

import pytest

from ttrl.engine.device.scope import (  # deliberately ttnn-free: see the docstring
    VISIBLE_DEVICES_ENV,
    scope_to_chip,
)


def _present_chips() -> list[int]:
    try:
        return sorted(
            int(name) for name in os.listdir("/dev/tenstorrent") if name.isdigit()
        )
    except OSError:
        return []


def _held_by_others(chip: int) -> bool:
    """Is another process holding `/dev/tenstorrent/<chip>` open?

    `fuser` rather than anything ttnn-shaped, for the same reason this whole
    module is ttnn-free: the answer is needed before ttnn may be imported.
    A missing or hanging `fuser` answers "no" -- this picks a chip, it is not
    a lock (`ttrl.engine.acquire` still takes the real one).
    """
    try:
        out = subprocess.run(
            ["fuser", f"/dev/tenstorrent/{chip}"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    me = os.getpid()
    return any(int(t) != me for t in out.split() if t.isdigit())


def _scope_this_process() -> None:
    """Pin this pytest process to a single chip, if the caller has not already.

    Falls back to the first present chip when every chip is busy: scoping to a
    contended chip still takes exactly one driver mutex instead of all of them,
    so it is strictly better for everyone else on the box than leaving the
    process unscoped. `ttrl.engine.acquire()` does the actual waiting.
    """
    if os.environ.get(VISIBLE_DEVICES_ENV, "").strip():
        return
    chips = _present_chips()
    if not chips:
        return
    chosen = next((c for c in chips if not _held_by_others(c)), chips[0])
    scope_to_chip(chosen)


_scope_this_process()


def _open_chip_nodes() -> set[str]:
    """Which `/dev/tenstorrent/<n>` nodes this process currently has open.

    Read from `/proc/self/fd`, so the probe costs nothing and answers about the
    driver rather than about ttnn's opinion of itself. An unreadable `/proc`
    answers "none", which turns the guard below into a no-op rather than a
    false accusation.
    """
    open_nodes: set[str] = set()
    try:
        entries = list(os.scandir("/proc/self/fd"))
    except OSError:
        return open_nodes
    for entry in entries:
        try:
            target = os.readlink(entry.path)
        except OSError:
            continue
        if target.startswith("/dev/tenstorrent/"):
            open_nodes.add(target)
    return open_nodes


@pytest.fixture(autouse=True)
def _host_only_tests_stay_host_only(request):
    """`-m "not tt"` must not touch the driver. Name the test that does.

    The `tt` marker is the suite's contract with everyone else on the box:
    unmarked means "no hardware", and the fleet relies on that to run CPU-only
    checks while other agents hold chips. A driver mutex on a chip that belongs
    to somebody else's run is exactly the cross-process contamination that
    makes a hardware test pass alone and fail in a fleet.

    The check is transition-based rather than absolute because once anything
    initialises the UMD cluster the nodes stay open for the life of the
    process: only the first offender is worth reporting, and it is the one that
    gets reported.
    """
    if request.node.get_closest_marker("tt") is not None:
        yield
        return
    before = _open_chip_nodes()
    yield
    opened = _open_chip_nodes() - before
    assert not opened, (
        f"{request.node.nodeid} is not marked `tt` but opened {sorted(opened)}. "
        "An unmarked test must not touch the driver: `-m \"not tt\"` is what "
        "the rest of the fleet runs while other agents hold chips, and every "
        "chip this process opens is a UMD mutex taken out from under one of "
        "them. Mark it `@pytest.mark.tt`, or keep the device out of it."
    )


@pytest.fixture(autouse=True)
def _settled_heap_before_hardware_tests(request):
    """Give every hardware test a collected heap to start from.

    `tests/_memory.py`'s `measure_dram` samples the device allocator after
    every iteration and `assert_no_growth` allows a slope of exactly zero, so
    the series has to be measured against a settled heap. Tapes left
    unreferenced by *earlier* tests are freed whenever CPython's generational
    collector next runs, which inside a full suite can land in the middle of a
    measurement rather than before it.
    """
    if request.node.get_closest_marker("tt") is None:
        yield
        return
    gc.collect()
    yield
