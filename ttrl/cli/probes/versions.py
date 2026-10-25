"""Versions: ttnn against the range ttrl declares, the tt-metal binary, the driver."""

from __future__ import annotations

import os
import re
from importlib import metadata
from pathlib import Path
from typing import Iterator

from ttrl.cli.probes._ttnn import unavailable
from ttrl.cli.report import FAIL, NOTE, OK, WARN, Line

_DRIVER_VERSION_PATH = "/sys/module/tenstorrent/version"
_REQ = re.compile(r"(==|!=|>=|<=|>|<)\s*([0-9][0-9a-zA-Z.\-+]*)")

#: The import package whose providing distribution carries the ttnn pin.
_PACKAGE = "ttrl"


def _parts(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", version)[:3])


def _satisfies(version: str, op: str, bound: str) -> bool:
    got, want = _parts(version), _parts(bound)
    # Compare at equal length so 0.80 and 0.80.0 do not differ by padding.
    width = max(len(got), len(want))
    got += (0,) * (width - len(got))
    want += (0,) * (width - len(want))
    return {
        "==": got == want, "!=": got != want, ">=": got >= want,
        "<=": got <= want, ">": got > want, "<": got < want,
    }[op]


def providing_distributions() -> list[str]:
    """Distribution names that provide the `ttrl` import package.

    Derived, never written down. The distribution is named tt-rl while the
    import package is ttrl, and a hardcoded "ttrl" here resolved only against a
    stale ttrl.egg-info left over from the rename: the version check then
    passed in the checkout and raised PackageNotFoundError for everyone who
    installed the project properly. A name that can be looked up cannot drift
    from pyproject again.

    More than one name means two distributions claim the package, which is how
    that bug stayed hidden, so the caller reports it rather than picking one.
    """

    seen: list[str] = []
    for name in metadata.packages_distributions().get(_PACKAGE, ()):
        if name not in seen:
            seen.append(name)
    return seen


def declared_ttnn_spec(distribution: str) -> str:
    """The ttnn requirement `distribution` declares.

    The pin is a correctness statement (see tests/test_ttnn_version.py), so it
    is read from the installed metadata rather than restated here, where the
    two copies could disagree.
    """

    for raw in metadata.requires(distribution) or []:
        head = raw.split(";")[0]
        if re.match(r"^ttnn\b", head):
            return head.strip()
    raise LookupError(
        f"{distribution}'s metadata declares no ttnn dependency, so the "
        f"version range this build is calibrated against cannot be read. "
        f"{distribution}'s packaging is broken; reinstall it."
    )


def probe() -> Iterator[Line]:
    providers = providing_distributions()
    if not providers:
        yield Line(
            FAIL, "supported ttnn range", "unknown: no installed distribution "
            f"provides the {_PACKAGE!r} package, so its ttnn pin cannot be read",
            action=f"ttrl is on sys.path without being installed. Install it "
                   "(`uv pip install tt-rl`, or `uv pip install -e .` from a "
                   "checkout) so the version it is calibrated against is "
                   "recorded. Until then nothing below checks a range.",
        )
        yield from _uncheckable_ttnn_lines()
        yield from _driver_lines()
        return

    specs = {name: declared_ttnn_spec(name) for name in providers}
    if len(providers) > 1:
        yield from _ambiguous_provider_lines(specs)
        if len(set(specs.values())) > 1:
            yield from _driver_lines()
            return
    spec = specs[providers[0]]
    bounds = _REQ.findall(spec)
    if not bounds:
        yield Line(FAIL, "supported ttnn range", f"cannot parse {spec!r} from "
                   f"{providers[0]}'s metadata",
                   action=f"{providers[0]}'s packaging is broken; reinstall it.")
        return

    reason = unavailable()
    if reason is not None:
        yield Line(FAIL, "ttnn", f"unavailable: {reason}",
                   action=f"ttrl needs it for everything. Install a supported "
                          f"version: uv pip install 'ttnn{spec[4:]}'")
    else:
        yield from _ttnn_lines(spec, bounds)

    yield from _driver_lines()


def _ambiguous_provider_lines(specs: dict[str, str]) -> Iterator[Line]:
    """Two distributions claiming one import package give wrong answers."""
    listed = ", ".join(f"{name} declares {spec!r}" for name, spec in specs.items())
    if len(set(specs.values())) > 1:
        yield Line(
            FAIL, "supported ttnn range", f"ambiguous: {listed}",
            action="two installed distributions provide the same import package "
                   "and disagree about the ttnn pin, so which one you get "
                   "depends on sys.path order. Remove the stale one (a leftover "
                   "ttrl.egg-info/ from the tt-rl rename is the usual cause).",
        )
        return
    yield Line(
        WARN, "distributions providing ttrl", listed, essential=False,
        action="two installed distributions provide the same import package. "
               "They agree on the ttnn pin today, so this is not yet wrong, but "
               "remove the stale one (usually a leftover ttrl.egg-info/ from "
               "the tt-rl rename).",
    )


def _uncheckable_ttnn_lines() -> Iterator[Line]:
    """Report the installed ttnn when there is no declared range to check it against."""
    reason = unavailable()
    if reason is not None:
        yield Line(FAIL, "ttnn", f"unavailable: {reason}",
                   action="ttrl needs it for everything. Install ttnn.")
        return

    try:
        yield Line(NOTE, "ttnn version", f"{metadata.version('ttnn')}, unchecked")
    except metadata.PackageNotFoundError:
        yield Line(NOTE, "ttnn version", "source build, unchecked")


def _ttnn_lines(spec: str, bounds) -> Iterator[Line]:
    try:
        installed = metadata.version("ttnn")
    except metadata.PackageNotFoundError:
        # A tt-metal source build on PYTHONPATH carries no distribution
        # metadata, which is how the Tracy profiler has to be run.
        yield Line(
            NOTE, "ttnn version",
            "a tt-metal source build is on sys.path and carries no distribution "
            f"metadata, so it cannot be checked against {spec}",
        )
        return
    bad = [f"{op}{bound}" for op, bound in bounds if not _satisfies(installed, op, bound)]
    if bad:
        yield Line(FAIL, "ttnn version", f"{installed} violates {', '.join(bad)}",
                   action=f"ttrl is calibrated to {spec}. Install a version in "
                          f"range: uv pip install 'ttnn{spec[4:]}'")
    else:
        yield Line(OK, "ttnn version", f"{installed}, inside the supported {spec}")

    try:
        # Deferred: the import is the probe, and its failure is the report.
        import ttnn
    except Exception as exc:
        yield Line(
            FAIL, "ttnn import", f"{installed} is installed but will not import: "
            f"{type(exc).__name__}: {exc}",
            action="the distribution is present and the import fails, so this is "
                   "a broken install or a missing system library. Reinstall it.",
        )
        return

    path = Path(ttnn.__file__).resolve().parent
    kind = "wheel" if "site-packages" in str(path) else f"source build at {path}"
    metal = path / "tt_metal"
    sfpi = metal / "sfpi-version"
    note = ""
    if sfpi.exists():
        match = re.search(r"sfpi_version='([^']+)'", sfpi.read_text())
        note = f", sfpi {match.group(1)}" if match else ""
    yield Line(
        OK if metal.is_dir() else FAIL, "tt-metal",
        f"bundled with ttnn ({kind}){note}" if metal.is_dir()
        else f"no tt_metal directory under {path}",
        action="" if metal.is_dir() else
        "this ttnn install is incomplete; reinstall the wheel.",
    )


def _driver_lines() -> Iterator[Line]:
    if not os.path.exists(_DRIVER_VERSION_PATH):
        yield Line(FAIL, "driver (KMD)", f"{_DRIVER_VERSION_PATH} does not exist",
                   action="the tenstorrent kernel module is not loaded. "
                          "`lsmod | grep tenstorrent`, then `sudo modprobe tenstorrent`.")
        return
    yield Line(OK, "driver (KMD)", Path(_DRIVER_VERSION_PATH).read_text().strip())
