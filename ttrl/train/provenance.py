"""What ran: the resolved config, the code it ran from, and the machine.

`write_run_metadata` puts a `config.json` beside `metrics.jsonl` at run
start. Before this, the resolved config existed only inside the checkpoint:
a run killed before its first checkpoint left a curve with no config at all,
and no run of any length recorded which code produced it.

Written at START, before the first frame, for that reason, and a missing git
is recorded rather than raised: the record is worth having on a box with no
repository, and a run must not fail because provenance could not be taken.
A tree with no `.git` at all, which is what `git archive` produces and what a
long measurement should run from, is the one case where the commit can only
come from the caller: `COMMIT_ENV_VAR` is that channel, and `git.source` keeps
a claim and an observation apart.

A SHA alone is a lie whenever the working tree is dirty, so `git` carries
`dirty`, the number of changed files and the first of their names, and
`code` renders the honest one-line version of all of it.

`write_run_outcome` is the other half, written at the END by the Trainer's
teardown ladder: how the run finished. Without it an early stop, a completed
run and a crash all look alike to anyone reading a run directory weeks later.
It is written from the ladder rather than from `run()` because the ladder
runs on the failure path too.

Reading the tree is `ttrl.train.gitstate`, whose names this module
re-exports: anything else that asks the same question about the same tree
reads it from there, and two implementations of it would eventually disagree.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ttrl import __version__
from ttrl.train.config import config_to_dict
from ttrl.train.gitstate import (COMMIT_ENV_VAR, code_line, git_state,
                                 repo_root)
from ttrl.train.scoping import resolved_chip

__all__ = ["COMMIT_ENV_VAR", "code_line", "git_state", "record_run_start",
           "repo_root", "run_metadata",
           "write_run_metadata", "write_run_outcome", "STATUSES"]

#: The name `write_run_metadata` writes, next to `metrics.jsonl`.
METADATA_NAME = "config.json"

#: The name `write_run_outcome` writes, beside it.
OUTCOME_NAME = "outcome.json"

#: How a run can end, and the whole set.
#:
#: `completed`  reached `total_frames`.
#: `stopped`    a stop condition fired; `reason` says which.
#: `incomplete` the run never finished: it raised, the process was
#:              interrupted, or the caller closed the Trainer early. `error`
#:              carries the exception propagating through teardown, if any.
STATUSES = ("completed", "stopped", "incomplete")


def _recorded(value: Any) -> Any:
    """`value` as this file can hold it.

    A config value may be an object rather than a number or a name: an
    encoder handed over as a class is an `UnnamedEncoder`, which carries the
    callable itself. Such a value says how it is recorded through its own
    `provenance()`, and what it returns is what the file says: the qualified
    name under a key of its own, so nothing here can write a handed-over
    class into the file as though a registered name had been used. An object
    with no `provenance()` is refused rather than rendered as its repr, which
    would put a memory address in a provenance record.
    """
    if isinstance(value, dict):
        return {k: _recorded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_recorded(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    render = getattr(value, "provenance", None)
    if callable(render):
        return _recorded(render())
    raise TypeError(
        f"run metadata cannot record {type(value).__name__} ({value!r}): a "
        f"config value is written to config.json as JSON. Give the type a "
        f"`provenance()` returning what the file should say, or put a plain "
        f"value in the config field."
    )


def run_metadata(config, *, argv: "list[str] | None" = None,
                 resolved_chip: "int | None" = None) -> dict[str, Any]:
    """The record `write_run_metadata` writes, as a plain dict.

    `resolved_chip` is the physical chip the run actually got, which is not
    knowable from `config` alone: `device.chip` is a REQUEST, and None there
    means "take whatever is free". Recording only the request would leave a
    reader unable to tell which chip produced the curve, so the resolved
    value is carried beside it rather than overwriting it. It is None in the
    record until scoping has happened.
    """

    git = git_state()
    now = time.time()
    return {
        "config": _recorded(config_to_dict(config)),
        "resolved_chip": resolved_chip,
        "ttrl_version": __version__,
        "code": code_line(git, __version__),
        "git": git,
        "started_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(
            timespec="seconds"
        ),
        "unix_time": now,
        "host": platform.node(),
        "python": platform.python_version(),
        "cwd": os.getcwd(),
        "argv": list(sys.argv if argv is None else argv),
    }


def write_run_metadata(log_dir, config, *, argv: "list[str] | None" = None,
                       resolved_chip: "int | None" = None) -> Path:
    """Write `<log_dir>/config.json` and return its path.

    Overwrites: a resumed or restarted run in the same directory describes
    itself as it is now, over the same curve. That is also what lets
    `ttrl.run` write once before scoping, so a run killed while waiting for a
    chip still leaves a record that `record_run_start` later refines.
    """
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / METADATA_NAME
    path.write_text(
        json.dumps(
            run_metadata(config, argv=argv, resolved_chip=resolved_chip),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def record_run_start(config) -> Path:
    """Write the start record for a run that is about to open its replicas.

    Called by `Trainer.__init__`, so a run composed by constructing a Trainer
    records what produced it as fully as one started through `ttrl.run`. It
    REFINES the pre-flight record `ttrl.run` writes before scoping by filling
    in the chip the request resolved to; every other field is taken the same
    way, and the file is overwritten, so a reader gets the later record.

    Raises with the fix named when `log_dir` cannot be written, which is the
    same refusal the run's logger and checkpoints would hit later.
    """

    try:
        return write_run_metadata(config.log_dir, config,
                                  resolved_chip=resolved_chip(config.device))
    except OSError as exc:
        raise OSError(
            f"cannot write {METADATA_NAME} under log_dir "
            f"{config.log_dir!r}: {exc}. Point config.log_dir at a writable "
            f"directory."
        ) from exc


def write_run_outcome(
    log_dir,
    *,
    status: str,
    frame: int,
    reason: str | None = None,
    error: str | None = None,
    profile: "dict | None" = None,
    best: "dict | None" = None,
) -> Path:
    """Write `<log_dir>/outcome.json` and return its path.

    Overwrites, like `write_run_metadata`: a resumed run in the same directory
    ends once, and how it ended this time is the answer.

    `best` is `BestScore.stamp()`, the same dict `checkpoint-best.npz` carries
    in its own `extra`. It is here because the user choosing a policy to ship
    reads this file, and the score alone does not say how far above its
    neighbours the kept eval sat.
    """
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / OUTCOME_NAME
    path.write_text(
        json.dumps(
            {
                "status": status,
                "best": best,
                "reason": reason,
                "error": error,
                "frame_count": int(frame),
                "profile": profile,
                "ended_utc": datetime.fromtimestamp(
                    time.time(), timezone.utc
                ).isoformat(timespec="seconds"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path
