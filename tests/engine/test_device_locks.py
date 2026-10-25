"""The multi-process hazards chip liveness must survive."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time

import pytest

from ttrl.engine.device import liveness as device_mod


class _Probe:
    """Stand-in for `subprocess.run(["fuser", dev])`."""

    def __init__(self, stdout: str, returncode: int = 0, stderr: str = ""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _fake_fuser(monkeypatch, probe: _Probe) -> None:
    monkeypatch.setattr(
        device_mod.subprocess, "run", lambda *a, **kw: probe, raising=True
    )


def test_descendant_pids_finds_a_grandchild():
    """`/proc` ppid walk reaches transitively, not just direct children."""
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess,sys,time;"
         "g=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
         "print(g.pid,flush=True);time.sleep(30)"],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        grandchild = int(child.stdout.readline().strip())
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            kin = device_mod._descendant_pids(os.getpid())
            if child.pid in kin and grandchild in kin:
                break
            time.sleep(0.05)
        assert child.pid in kin
        assert grandchild in kin, "ppid walk stopped at the direct child"
        assert os.getpid() not in kin
    finally:
        child.kill()
        child.wait()


def test_own_env_workers_are_not_foreign_when_we_hold_the_chip(monkeypatch, tmp_path):
    """A forked child holding our inherited device fd must not make our own chip look busy to our own `status()`."""
    dev = "/dev/tenstorrent/0"
    worker = 424242
    _fake_fuser(monkeypatch, _Probe(f"{os.getpid()} {worker}\n"))
    monkeypatch.setattr(device_mod, "physical_chip_id", lambda cid: 0)
    monkeypatch.setattr(device_mod, "_holds_open", lambda path: path == dev)
    monkeypatch.setattr(device_mod, "_descendant_pids", lambda root: frozenset({worker}))

    assert device_mod._foreign_pids(0) == ()


def test_another_jobs_pid_is_still_foreign_when_we_hold_the_chip(monkeypatch):
    """Only *our* descendants are excused; a stranger still marks the chip busy."""
    monkeypatch.setattr(device_mod, "physical_chip_id", lambda cid: 0)
    _fake_fuser(monkeypatch, _Probe(f"{os.getpid()} 424242 999999\n"))
    monkeypatch.setattr(device_mod, "_holds_open", lambda path: True)
    monkeypatch.setattr(device_mod, "_descendant_pids", lambda root: frozenset({424242}))

    assert device_mod._foreign_pids(0) == (999999,)


def test_a_child_holding_a_chip_we_do_not_hold_is_foreign(monkeypatch):
    """A launcher's training subprocesses are descendants but own their chips independently."""
    monkeypatch.setattr(device_mod, "physical_chip_id", lambda cid: 0)
    _fake_fuser(monkeypatch, _Probe("424242\n"))
    monkeypatch.setattr(device_mod, "_holds_open", lambda path: False)
    monkeypatch.setattr(device_mod, "_descendant_pids", lambda root: frozenset({424242}))

    assert device_mod._foreign_pids(0) == (424242,)


@pytest.mark.parametrize("code", [2, 127])
def test_fuser_failing_to_look_is_not_free(monkeypatch, code):
    """Any exit code other than 0/1 means `fuser` could not inspect the node."""
    monkeypatch.setattr(device_mod, "physical_chip_id", lambda cid: 0)
    _fake_fuser(monkeypatch, _Probe("", returncode=code, stderr="cannot stat"))
    with pytest.raises(device_mod.ChipLivenessCheckFailed,
                       match="liveness is unconfirmed.*Install psmisc"):
        device_mod._foreign_pids(0)


def test_lock_is_exclusive_and_reports_its_holder(monkeypatch, tmp_path):
    monkeypatch.setenv("TTRL_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(device_mod, "physical_chip_id", lambda cid: 7)
    assert device_mod._lock_holder(7) is None
    with device_mod.hold_lock(7) as got:
        assert got
        # A second acquisition in *any* process, ours included, is refused:
        # flock conflicts across open file descriptions, not just processes.
        with device_mod.hold_lock(7) as again:
            assert not again
    assert device_mod._lock_holder(7) is None, "lock file left looking held"


def test_concurrent_probes_do_not_see_each_other_as_holders(monkeypatch, tmp_path):
    """`_lock_holder` must probe with a SHARED lock, not an exclusive one."""
    monkeypatch.setenv("TTRL_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(device_mod, "physical_chip_id", lambda cid: 7)
    lock = tmp_path / "chip7.lock"
    lock.write_text("")  # a released lock: the file exists, the pid is erased

    with open(lock, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            assert device_mod._lock_holder(7) is None, (
                "a concurrent shared probe was mistaken for a lock holder"
            )
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

    # ...and a genuine holder is still detected, shared probe or not.
    with device_mod.hold_lock(7) as got:
        assert got
        assert device_mod._lock_holder(7) == os.getpid()


def test_lock_probe_needs_no_write_permission(monkeypatch, tmp_path):
    """A read probe opens the lock file read-only."""
    monkeypatch.setenv("TTRL_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(device_mod, "physical_chip_id", lambda cid: 7)
    lock = tmp_path / "chip7.lock"
    lock.write_text("4242")
    lock.chmod(0o444)
    if os.access(lock, os.W_OK):  # running as root: the mode means nothing
        pytest.skip("cannot make a file read-only as this user")
    try:
        assert device_mod._lock_holder(7) is None
    finally:
        lock.chmod(0o644)


def test_unreadable_lock_is_not_a_free_chip(monkeypatch, tmp_path):
    monkeypatch.setenv("TTRL_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(device_mod, "physical_chip_id", lambda cid: 7)
    lock = tmp_path / "chip7.lock"
    lock.write_text("123")
    lock.chmod(0o000)
    if os.access(lock, os.W_OK):  # running as root: the mode means nothing
        pytest.skip("cannot make a file unreadable as this user")
    try:
        with pytest.raises(device_mod.ChipLivenessCheckFailed, match="TTRL_LOCK_DIR"):
            device_mod._lock_holder(7)
    finally:
        lock.chmod(0o644)
