"""A run's checkpoints over time: frame-keyed files, one stable latest, bounded.

A periodic checkpoint used to be written to `<log_dir>/checkpoint.npz` every
time, so a run kept exactly one and a progression of behaviour across a long
run could not be shown at all without a user polling the directory and
copying the file aside between writes. A run now writes
`checkpoint-<frame>.npz`, keyed on the frame count the file itself carries,
and points `checkpoint.npz` at the newest as a relative symlink, so
`--resume <log_dir>/checkpoint.npz` and every published path still mean the
latest checkpoint.

Frames are zero-padded to twelve digits so a lexical listing is already in
frame order.

RETENTION IS BOUNDED, by `TrainConfig.checkpoint_keep`. An Atari checkpoint
is about 27 MB, so a 1.5M-frame run checkpointing every 50k would leave 30
files and 810 MB behind, and a sweep of ten such runs 8 GB; a run that fills
a disk fails in its last hour with its work half kept. An unconditional
built-in bound is no better, because the oldest checkpoints are exactly the
ones a learning curve needs. So the bound is one number the user sets,
pruning oldest first, defaulting to a size a run can leave behind unattended.
"""

from __future__ import annotations

import os
import re
import uuid

__all__ = ["LATEST", "history", "history_path", "point_at_latest", "prune"]

#: The stable path to a run's newest checkpoint, a symlink to one of the
#: frame-keyed files beside it.
LATEST = "checkpoint.npz"

_PREFIX = "checkpoint-"
_PATTERN = re.compile(r"^checkpoint-(\d{12})\.npz$")


def history_path(log_dir: str, frame: int) -> str:
    """The file a checkpoint taken at `frame` belongs in."""
    if frame < 0:
        raise ValueError(f"checkpoint frame must be >= 0, got {frame}")
    if frame >= 10**12:
        raise ValueError(
            f"checkpoint frame {frame} does not fit the twelve-digit name a "
            "checkpoint history sorts on. Split the run."
        )
    return os.path.join(log_dir, f"{_PREFIX}{frame:012d}.npz")


def history(log_dir: str) -> list[tuple[int, str]]:
    """Every frame-keyed checkpoint in `log_dir`, oldest frame first."""
    try:
        names = os.listdir(log_dir)
    except OSError:
        return []
    found = [(int(m.group(1)), os.path.join(log_dir, m.group(0)))
             for m in (_PATTERN.match(n) for n in names) if m]
    return sorted(found)


def point_at_latest(log_dir: str, target: str) -> str:
    """Point `<log_dir>/checkpoint.npz` at `target`, atomically.

    A relative symlink, so a whole run directory stays valid when it is moved
    or copied. Written to a temp name and `os.replace`d, so a reader of the
    stable path never sees it absent, and so it replaces whatever is there
    including the plain file older runs left.
    """
    link = os.path.join(log_dir, LATEST)
    tmp = os.path.join(log_dir, f".{LATEST}.link-{uuid.uuid4().hex}")
    os.symlink(os.path.basename(target), tmp)
    try:
        os.replace(tmp, link)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return link


def prune(log_dir: str, keep: int) -> list[str]:
    """Delete all but the newest `keep` frame-keyed checkpoints.

    Returns what it removed. Oldest first, and the newest is never a
    candidate, so the file `checkpoint.npz` resolves to survives any `keep`.
    """
    if keep < 1:
        raise ValueError(f"checkpoint_keep must be >= 1, got {keep}")
    found = history(log_dir)
    removed = []
    for _, path in found[: max(len(found) - keep, 0)]:
        try:
            os.remove(path)
        except OSError:
            # Someone else moved or removed it between the listing and here,
            # which is the state this call was asking for anyway.
            continue
        removed.append(path)
    return removed
