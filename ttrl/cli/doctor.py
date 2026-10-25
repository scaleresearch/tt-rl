"""`ttrl doctor`: one screen saying whether this box can train, and what to fix.

Most first-run failures on this stack are environmental and surface as a ttnn
TT_FATAL deep inside a conv. Every check here answers one of the questions
that failure could have been, in the order in which an answer stops being
meaningful: versions, then chips, then the environment traps, then what only a
real chip can tell us, then the optional extras.

Exits non-zero if anything essential is broken. It never raises: a probe that
cannot run reports that it cannot run, and why.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from ttrl.cli.probes import chips, extras, hardware, traps, versions
from ttrl.cli.report import Section, failures, render, section

_SECTIONS = (
    ("versions", versions.probe),
    ("chips", chips.probe),
    ("environment traps", traps.probe),
    ("hardware probe", hardware.probe),
    ("optional extras", extras.probe),
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--traceback", action="store_true",
        help="re-raise instead of reporting a probe that fails to run "
             "(for debugging doctor itself)",
    )


def run(args: argparse.Namespace) -> int:
    sections = diagnose(traceback=args.traceback)
    print(render(sections), end="")
    broken = failures(sections)
    if broken:
        print(f"{len(broken)} essential check(s) failed: "
              + ", ".join(line.label for line in broken))
        return 1
    print("all essential checks passed")
    return 0


def diagnose(*, traceback: bool = False) -> Sequence[Section]:
    """Run every probe and return the sections, without printing anything."""
    return [section(title, probe, traceback=traceback) for title, probe in _SECTIONS]
