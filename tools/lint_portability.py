"""Portability lint: no arch-specific ttnn names outside the device layer.

Portability policy: no `ttnn.*ComputeKernelConfig` class name, `ttnn.Arch`
member or arch string literal outside the device layer. Everywhere else asks
the probed `DeviceProfile` what this part can do. No numeric-literal bans:
shape rules come from the profile and the adapters, and grep cannot tell a
shape constant from an arithmetic one.

AN ARCH-KEYED CALIBRATION TABLE IS ALLOWED ANYWHERE. An arch literal used as
a dict key with a value after it is a measurement that refuses the arches it
has not measured, which is the opposite of guessing; `MEASURED_DX_ROWS_PER_CORE`
in `ttrl/engine/conv/geometry.py` is the case. Branching on an arch is banned
everywhere above the device layer and has its own rule, so permitting the key
shape does not open the door the policy exists to hold shut.

THE DEVICE LAYER IS ENUMERATED, never listed by hand: it is every module
under `ttrl/engine/device/`, whatever that directory comes to hold. A
hand-maintained set of filenames drifts from the package it stands for, and
the drift is silent in the safe direction for the lint and the unsafe one for
the policy.

This is a grep-based check, not an AST-based one, deliberately: the forbidden
patterns are textual (a class name, an enum member, a literal string), and a
regex over the source is simpler to read, extend, and trust than a visitor.

Run as a script (``python -m tools.lint_portability``) or import
``check(root)`` / ``violations(root)`` from a test. With no argument it lints
the ``ttrl`` package next to this repository's ``tools/`` directory.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

#: The directory whose modules may name an arch, relative to the linted root.
#: Every file under it is allowed; nothing outside it is.
DEVICE_LAYER = "engine/device"

_ARCH_STRINGS = ("wormhole_b0", "blackhole", "quasar", "wormhole", "grayskull")

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ttnn.*ComputeKernelConfig class name",
        re.compile(r"\bttnn\.\w*ComputeKernelConfig\b"),
    ),
    (
        "ttnn.Arch member",
        re.compile(r"\bttnn\.Arch\.\w+"),
    ),
    (
        "arch string comparison",
        re.compile(
            r"""(?:[=!]=\s*|\bin\s*[\[(]?\s*)["'](?:"""
            + "|".join(_ARCH_STRINGS)
            + r""")["']|["'](?:"""
            + "|".join(_ARCH_STRINGS)
            + r""")["']\s*[=!]=""",
            re.IGNORECASE,
        ),
    ),
    (
        "arch string literal",
        re.compile(
            r"""["'](?:""" + "|".join(_ARCH_STRINGS) + r""")["'](?!\s*:\s*\S)""",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class Violation:
    path: Path
    lineno: int
    line: str
    rule: str

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno}: [{self.rule}] {self.line.strip()}"


def _in_device_layer(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return relative.as_posix().startswith(DEVICE_LAYER + "/")


def violations(root: Path) -> list[Violation]:
    found: list[Violation] = []
    for path in sorted(root.rglob("*.py")):
        if _in_device_layer(root, path):
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for rule, pattern in _PATTERNS:
                if pattern.search(line):
                    found.append(Violation(path, lineno, line, rule))
    return found


def check(root: Path) -> bool:
    """Return True iff no violations were found (also prints them)."""
    vs = violations(root)
    for v in vs:
        print(v)
    return not vs


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else Path(__file__).resolve().parent.parent / "ttrl"
    ok = check(root)
    if ok:
        print("portability lint: clean")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
