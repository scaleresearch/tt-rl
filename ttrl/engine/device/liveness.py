"""Chip discovery, liveness, and the advisory lock concurrent runs cooperate on.

TT devices are exclusive: opening a busy one fails, sometimes loudly, sometimes
by wedging. So acquire() looks before it leaps. The fuser probe sees other
processes that already hold the device node; the PID-tagged flock covers what
fuser cannot see coming, our own concurrent runs, which have not opened the
node yet.
"""

from __future__ import annotations

import contextlib
import fcntl
import glob
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ttrl.engine.device.scope import physical_chip_id, visible_chip_ids

POLL_SECONDS = 5.0
#: Default ceiling on a blocking acquire. A chip is normally handed over in
#: seconds (device close plus cluster teardown), so two minutes covers a real
#: handover, and waiting past it means the holder is not going to let go. It is
#: bounded because the common cause is unrecoverable: see STALE_FD_NOTE.
DEFAULT_WAIT_SECONDS = 120.0
_DEFAULT_LOCK_DIR = "/tmp/ttrl-locks"
_LOCK_DIR_ENV = "TTRL_LOCK_DIR"
_FUSER_FIX = (
    "`fuser` must be installed and /proc readable for ttrl to tell a free "
    "chip from a busy one. Install psmisc."
)
_UNCONFIRMED = (
    "chip liveness is unconfirmed and treating that as free risks two "
    "processes opening one chip"
)

#: ttnn's cluster singleton keeps its /dev/tenstorrent/<n> descriptors for the
#: life of the process, so a process that has opened and closed a device still
#: looks like a holder to everyone else. Measured on Blackhole, ttnn 0.80: six
#: fds, still open after ttnn.close_device, and no ttnn call releases them.
STALE_FD_NOTE = (
    "a process that has opened and closed a device keeps its "
    "/dev/tenstorrent fds until it exits, so a child of such a process sees "
    "its parent as a holder even though the parent is no longer using the "
    "chip. Scope the child to a different chip with TT_VISIBLE_DEVICES, or "
    "open the device in the child rather than in the parent."
)


def lock_dir() -> Path:
    """Advisory lock directory. Read per call: the library must not read the
    environment at import time."""
    return Path(os.environ.get(_LOCK_DIR_ENV, _DEFAULT_LOCK_DIR))


class NoDeviceAvailable(RuntimeError):
    pass


class ChipLivenessCheckFailed(RuntimeError):
    """The liveness probe could not be run at all.

    A chip wrongly reported free lets two processes collide on one piece of
    silicon, so "cannot check" is never treated as "free". `fix` is the one
    action that lets the probe run.
    """

    def __init__(self, problem: str, fix: str):
        super().__init__(f"{problem}; {_UNCONFIRMED}. {fix}")
        self.problem = problem
        self.fix = fix


@dataclass(frozen=True)
class ChipStatus:
    chip_id: int
    path: str
    foreign_pids: tuple[int, ...]
    locked_by: int | None

    @property
    def free(self) -> bool:
        return not self.foreign_pids and self.locked_by is None


def physical_chips() -> list[int]:
    """Physical chip ids exposed by the driver, ignoring any scoping."""
    ids = []
    for path in glob.glob("/dev/tenstorrent/*"):
        name = os.path.basename(path)
        if name.isdigit():
            ids.append(int(name))
    return sorted(ids)


def enumerate_chips() -> list[int]:
    """Chip ids this process may pass to ttnn.open_device(device_id=...).

    Unscoped these are the driver node numbers. Scoped, ttnn accepts only
    local indices into the visible set, so this returns [0, 1, ...] over the
    visible chips that actually have a device node.
    """
    present = set(physical_chips())
    visible = visible_chip_ids()
    if visible is None:
        return sorted(present)
    return [i for i, phys in enumerate(visible) if phys in present]


def _parent_pid(pid: int) -> int | None:
    """pid's parent from /proc/<pid>/stat, or None if it is already gone.

    The comm field is parenthesised and may itself contain spaces and parens,
    so fields are read from after the last ")". Splitting the whole line on
    whitespace is the classic way to land on a nonsense ppid.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = stat.rpartition(")")[2].split()
    return int(fields[1]) if len(fields) > 1 else None


def _descendant_pids(root: int) -> frozenset[int]:
    children: dict[int, list[int]] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        ppid = _parent_pid(int(name))
        if ppid is not None:
            children.setdefault(ppid, []).append(int(name))
    kin: set[int] = set()
    queue = list(children.get(root, ()))
    while queue:
        pid = queue.pop()
        if pid in kin:
            continue
        kin.add(pid)
        queue.extend(children.get(pid, ()))
    return frozenset(kin)


def _holds_open(path: str) -> bool:
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{fd}") == path:
                return True
        except OSError:
            # Raced with one of our own fds closing; not the path in question.
            continue
    return False


def _foreign_pids(chip_id: int) -> tuple[int, ...]:
    """PIDs other than ours or our children's holding the chip's node open."""
    dev = f"/dev/tenstorrent/{physical_chip_id(chip_id)}"
    try:
        probe = subprocess.run(["fuser", dev], capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise ChipLivenessCheckFailed(
            f"`fuser {dev}` could not run ({exc})", _FUSER_FIX) from exc
    # fuser exits 0 when someone holds the file open and 1 when nobody does.
    # Anything else is fuser failing to look (a hidden /proc, a node it cannot
    # stat), and its stdout is then empty for the same reason a free chip's is.
    if probe.returncode not in (0, 1):
        raise ChipLivenessCheckFailed(
            f"`fuser {dev}` exited {probe.returncode} "
            f"({probe.stderr.strip() or 'no stderr'})", _FUSER_FIX)
    me = os.getpid()
    pids = tuple(int(t) for t in probe.stdout.split() if t.isdigit() and int(t) != me)
    if not pids or not _holds_open(dev):
        return pids
    # We hold this node open and TT devices are exclusive, so nobody else can
    # have opened it: every pid here is one of our own forked children (env
    # workers forked after the device was opened carry an inherited fd).
    # Calling those foreign makes a process's own chip look busy to its own
    # later status() call.
    return tuple(p for p in pids if p not in _descendant_pids(me))


def _lock_holder(chip_id: int) -> int | None:
    """PID holding our advisory lock for this chip, if any.

    The probe takes a shared lock. flock has no read-only "who holds this?"
    call, and LOCK_EX conflicts with other readers as well as with the holder,
    so two concurrent status() calls each saw the other's momentary probe as a
    holder and reported a free chip as locked. LOCK_SH conflicts only with the
    LOCK_EX a real holder takes. The file is opened read-only so a read probe
    does not need write permission on another user's lock file.
    """
    lock_path = lock_dir() / f"chip{physical_chip_id(chip_id)}.lock"
    if not lock_path.exists():
        return None
    try:
        with open(lock_path, "r") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                content = fh.read().strip()
                return int(content) if content.isdigit() else -1
            # We got the lock, so nobody held it: stale file.
            fcntl.flock(fh, fcntl.LOCK_UN)
            return None
    except OSError as exc:
        # None here would mean "nobody holds this chip", which acquire() acts
        # on. Reporting that because the lock file could not be read hands the
        # chip to a second trainer.
        raise ChipLivenessCheckFailed(
            f"cannot read chip lock {lock_path} ({exc})",
            f"Set {_LOCK_DIR_ENV} to a directory you own.") from exc


def status(chip_id: int) -> ChipStatus:
    """Liveness of one chip, named by the id ttnn would take for it.

    `path` is the physical node, which under TT_VISIBLE_DEVICES is not
    /dev/tenstorrent/<chip_id>.
    """
    return ChipStatus(
        chip_id=chip_id,
        path=f"/dev/tenstorrent/{physical_chip_id(chip_id)}",
        foreign_pids=_foreign_pids(chip_id),
        locked_by=_lock_holder(chip_id),
    )


def survey() -> list[ChipStatus]:
    return [status(c) for c in enumerate_chips()]


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return "exited"
    return " ".join(raw.decode("utf-8", "replace").split("\0")).strip() or "unknown"


def describe_holders(statuses: list[ChipStatus]) -> str:
    """One line per chip naming who holds it, with each holder's command line.

    An fd lingering in a process that has closed the device is
    indistinguishable from one in a process actively computing on it: the
    descriptor is the same, and ttnn exposes nothing that separates them. So
    both are reported as holders, because calling a real holder stale would
    put two processes on one chip.
    """
    lines = []
    for st in statuses:
        who = [f"pid {p} ({_cmdline(p)})" for p in st.foreign_pids]
        if st.locked_by is not None:
            who.append(f"lock held by pid {st.locked_by}")
        lines.append(f"  chip {st.chip_id} ({st.path}): {', '.join(who) or 'free'}")
    return "\n".join(lines)


@contextlib.contextmanager
def hold_lock(chip_id: int) -> Iterator[bool]:
    """Try to take the advisory lock for a chip. Yields whether we got it.

    The lock is PID-tagged and dies with the process.
    """
    directory = lock_dir()
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / f"chip{physical_chip_id(chip_id)}.lock"
    try:
        fh = open(lock_path, "a+")
    except PermissionError as exc:
        raise PermissionError(
            f"cannot open the chip lock {lock_path}, which another user owns. "
            f"Set {_LOCK_DIR_ENV} to a directory you own.") from exc
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        try:
            yield True
        finally:
            # Erasing the PID is bookkeeping, releasing the lock is not, so
            # the unlock is nested: a full disk must not turn a cosmetic
            # truncate failure into the exception the caller sees.
            try:
                fh.seek(0)
                fh.truncate()
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
    finally:
        fh.close()
