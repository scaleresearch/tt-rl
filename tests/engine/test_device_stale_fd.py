"""A blocking acquire must be bounded, because the usual reason a chip never
frees is a descriptor no amount of waiting releases."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import textwrap

import pytest

from ttrl.engine.device import (
    DEFAULT_WAIT_SECONDS,
    MESH_GRAPH_DESC_ENV,
    STALE_FD_NOTE,
    ChipStatus,
    acquire,
    describe_holders,
    mesh_graph_descriptor_path,
)

# The child inherits TT_VISIBLE_DEVICES and the descriptor path, so it must
# not re-scope: it would renumber the visible set out from under acquire(0).
_CHILD = textwrap.dedent(
    """
    from ttrl.engine.device import NoDeviceAvailable, acquire
    try:
        with acquire(0, wait=True, timeout=4.0, poll=1.0):
            print("ACQUIRED")
    except NoDeviceAvailable as exc:
        print("REFUSED", exc)
    """
)


def test_blocking_acquire_has_a_bounded_default_timeout():
    default = inspect.signature(acquire.__wrapped__).parameters["timeout"].default
    assert default == DEFAULT_WAIT_SECONDS
    assert 0 < default < float("inf")


def test_holder_description_names_the_pid_and_its_command_line():
    me = os.getpid()
    line = describe_holders(
        [ChipStatus(chip_id=0, path="/dev/tenstorrent/0", foreign_pids=(me,), locked_by=None)]
    )
    assert f"pid {me}" in line
    assert sys.argv[0].split("/")[-1] in line or "python" in line


@pytest.mark.tt
def test_child_of_a_process_that_closed_a_device_is_refused_not_hung():
    """The parent (this process) opens and closes a chip, keeping ttnn's fds.
    A child then sees those fds and must be refused with an explanation rather
    than waiting forever."""
    # conftest scopes this process only when it was not scoped already, and
    # the descriptor is what a single visible chip on a paired board needs.
    descriptor = mesh_graph_descriptor_path()
    assert descriptor is not None, "no single-chip mesh graph descriptor in ttnn"
    os.environ.setdefault(MESH_GRAPH_DESC_ENV, descriptor)

    with acquire(0):
        pass

    held = [
        fd
        for fd in os.listdir("/proc/self/fd")
        if os.path.realpath(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")
    ]
    assert held, "ttnn released its device fds; this test and STALE_FD_NOTE are stale"

    done = subprocess.run(
        [sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=120
    )
    assert "REFUSED" in done.stdout, done.stdout + done.stderr
    assert str(os.getpid()) in done.stdout
    assert STALE_FD_NOTE.split(",")[0] in done.stdout
