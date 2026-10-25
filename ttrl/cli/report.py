"""The status lines `ttrl doctor` prints, and how a probe failure becomes one.

A line carries what was found and, when something is wrong, what to do about
it. The constructor enforces the second half: a report that says a thing is
broken without saying what to do about it is the failure mode this command
exists to replace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

OK = "ok"
FAIL = "fail"
WARN = "warn"
NOTE = "note"

_MARK = {OK: "  ok  ", FAIL: " FAIL ", WARN: " warn ", NOTE: " note "}


@dataclass(frozen=True)
class Line:
    """One finding. `essential` decides whether a FAIL makes doctor exit non-zero."""

    status: str
    label: str
    detail: str
    action: str = ""
    essential: bool = True

    def __post_init__(self) -> None:
        if self.status not in _MARK:
            raise ValueError(f"unknown status {self.status!r} for line {self.label!r}")
        if self.status in (FAIL, WARN) and not self.action:
            raise ValueError(
                f"line {self.label!r} is a {self.status} with no action. Every "
                "failing doctor line must say what to do about it."
            )


@dataclass(frozen=True)
class Section:
    title: str
    lines: tuple[Line, ...]


def section(
    title: str, probe: Callable[[], Iterable[Line]], *, traceback: bool = False
) -> Section:
    """Run `probe()`, turning an unexpected exception into a reportable line.

    This is not a fallback: nothing else is attempted, nothing degrades, and
    the finding is reported as a failure. It exists because doctor's contract
    is that it always prints a report, and a probe that cannot run is itself
    the most interesting thing on the screen. `--traceback` re-raises instead,
    for debugging doctor itself.
    """
    try:
        return Section(title, tuple(probe()))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        if traceback:
            raise
        return Section(
            title,
            (
                Line(
                    FAIL,
                    title,
                    f"could not run: {type(exc).__name__}: {exc}",
                    action=(
                        "this probe raised where it should have reported. Rerun "
                        "with `ttrl doctor --traceback` and treat the traceback "
                        "as a ttrl bug or a broken install."
                    ),
                ),
            ),
        )


def render(sections: Sequence[Section]) -> str:
    out: list[str] = []
    for sec in sections:
        out.append(f"{sec.title}")
        for line in sec.lines:
            out.append(f"[{_MARK[line.status]}] {line.label}: {line.detail}")
            if line.action:
                for part in line.action.splitlines():
                    out.append(f"           -> {part}")
        out.append("")
    return "\n".join(out)


def failures(sections: Sequence[Section]) -> list[Line]:
    """Essential FAIL lines. Non-empty means doctor exits non-zero."""
    return [
        line
        for sec in sections
        for line in sec.lines
        if line.status == FAIL and line.essential
    ]
