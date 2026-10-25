"""Which code a run ran from: the tree's identity, read or asserted.

Split out of `ttrl.train.provenance`, which writes a run's record. Reading a
checkout and writing `config.json` are two jobs: this one shells out to git
and parses it, and nothing here knows what a run is. `ttrl.train.provenance`
re-exports the names its own callers reach for, so an importer of either
module gets the same function and the two can never disagree.

`code_line` renders the one-line human form, because a SHA alone is a lie
whenever the working tree is dirty.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

__all__ = ["COMMIT_ENV_VAR", "code_line", "dirty_paths", "git_state",
           "repo_root"]

#: Where a caller states the commit a run was started from, when the tree it
#: runs is an export rather than a checkout. An environment variable rather
#: than a config field because the config files inside that export are part
#: of what was pinned: editing one to carry the commit changes the thing
#: being pinned, while the launcher script already knows the SHA and can
#: export it without touching the tree.
COMMIT_ENV_VAR = "TTRL_COMMIT"

#: How many changed paths a record lists before it stops naming them.
DIRTY_FILES_LISTED = 40


def repo_root() -> Path:
    """The directory the running `ttrl` package sits in."""
    return Path(__file__).resolve().parent.parent.parent


def _asserted_commit() -> str | None:
    """The commit the caller states this tree is, or None. A value that
    cannot be a commit id is refused rather than written into a record, where
    it would read as a SHA for as long as the run is kept."""
    stated = os.environ.get(COMMIT_ENV_VAR, "").strip()
    if not stated:
        return None
    lowered = stated.lower()
    if not 7 <= len(lowered) <= 40 or lowered.strip("0123456789abcdef"):
        raise ValueError(
            f"{COMMIT_ENV_VAR}={stated!r} is not a commit id. Set it to the "
            f"full 40-character SHA the tree was exported from, as "
            f"`git rev-parse <ref>` prints it, or unset it."
        )
    return lowered


def dirty_paths(status: str) -> list[str]:
    """The changed paths in `git status --porcelain -z` output.

    `-z` for the parsing, not for the speed: without it a path holding a
    space, a quote or a non-ASCII byte comes back quoted and C-escaped, and
    a record naming `"ttrl/\\303\\251.py"` names no file anyone can open. With
    it every path is literal and NUL-terminated, so there is nothing to
    unescape.

    Each record is two status characters, a space, then the path. A rename
    or a copy (`R`/`C` in either column) is followed by a second record
    holding the name it came FROM; the NEW name is what this keeps, because
    the record describes the tree the run read its code out of, and the old
    name is not in that tree.
    """
    records = iter([r for r in status.split("\0") if r])
    paths = []
    for record in records:
        if "R" in record[:2] or "C" in record[:2]:
            next(records, None)
        paths.append(record[3:] if len(record) > 3 else record)
    return paths


def git_state(root: Path | None = None) -> dict[str, Any]:
    """HEAD, branch, and whether the tree was dirty when this was taken.

    `source` says how this was obtained: `"checkout"` was read from a
    repository, and `"asserted"` came from the caller through
    `COMMIT_ENV_VAR` for a tree with no repository to read, which nothing
    here can verify, so `dirty` is None. A readable checkout answers for
    itself and the assertion is ignored.

    Never raises on a missing git, which is recorded as unavailable with the
    reason. A malformed asserted commit does raise: it is a typo in the pin.
    """
    asserted = _asserted_commit()
    root = Path(root) if root is not None else repo_root()

    def _git(*args: str) -> str:
        return subprocess.run(
            ("git", "-C", str(root)) + args,
            capture_output=True, text=True, timeout=10,
        ).stdout

    try:
        head = _git("rev-parse", "HEAD").strip()
        branch = _git("rev-parse", "--abbrev-ref", "HEAD").strip()
        # NOT stripped: the first status character is a space for a path
        # changed in the worktree only, and stripping the output ate it,
        # which cost the first listed path its first character.
        status = _git("status", "--porcelain", "-z")
    except Exception as exc:
        head, unavailable = "", f"{exc.__class__.__name__}: {exc}"
    else:
        unavailable = "" if head else "not a git repository"
    if not head:
        if asserted is not None:
            return {"available": True, "source": "asserted",
                    "asserted_by": COMMIT_ENV_VAR, "head": asserted,
                    "branch": None, "dirty": None}
        return {"available": False, "source": "none", "error": unavailable}
    files = dirty_paths(status)
    return {
        "available": True, "source": "checkout", "head": head, "branch": branch,
        "dirty": bool(files), "dirty_file_count": len(files),
        "dirty_files": files[:DIRTY_FILES_LISTED],
    }


def code_line(git: dict[str, Any], version: str) -> str:
    """One line a human can act on, which a bare SHA is not."""
    if not git.get("available"):
        return f"ttrl {version}, no git checkout ({git.get('error')})"
    head = str(git["head"])[:12]
    if git.get("source") == "asserted":
        return (f"ttrl {version}, {head} asserted by {git['asserted_by']}, no "
                "git checkout here, so the tree was not checked against it")
    if not git.get("dirty"):
        return f"ttrl {version}, {head} on {git['branch']}, tree clean"
    return (
        f"ttrl {version}, {head} on {git['branch']} plus "
        f"{git['dirty_file_count']} uncommitted file(s), so this SHA alone "
        "does not identify the code that ran"
    )
