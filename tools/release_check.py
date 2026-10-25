"""The release gate: everything a release must not skip, as one exit code.

Run it before tagging: `python -m tools.release_check`.

These checks are about the distribution rather than the code, so the test
suite cannot see them. A wheel with no long description renders an empty PyPI
page. A wheel that quietly grows a `tools/` directory ships developer tooling
to strangers. An `import ttrl` that pulls in ttnn breaks every chip-scoped
subprocess, which is the one thing `ttrl.engine.device.scope` exists to
prevent. Each was verified by hand once; the point of this file is that nobody
has to know to do that again.

Host-only: nothing here opens a device or imports ttnn in this process, so it
is safe to run while the chips are busy.
"""

from __future__ import annotations

import argparse
import email
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import zipfile

#: The only top-level wheel entry besides the `.dist-info`. Anything else is
#: developer tooling that has leaked.
ALLOWED_WHEEL_TOPS = {"ttrl"}

#: Modules `import ttrl` must not pull in. ttnn because scoping has to happen
#: before the first import of it in a process; the rest because a bare install
#: has to work with no environment suite and no torch.
FORBIDDEN_ON_IMPORT = ("ttnn", "numpy", "gymnasium", "torch")

#: Run with `-I` in the throwaway venv: the import is clean and opened no
#: device node on the way.
IMPORT_PURITY = '''
import os, sys
import ttrl
forbidden = [m for m in {forbidden!r} if m in sys.modules]
assert not forbidden, f"import ttrl pulled in {{forbidden}}"
opened = []
for entry in os.scandir("/proc/self/fd"):
    try:
        target = os.readlink(entry.path)
    except OSError:
        continue
    if target.startswith("/dev/tenstorrent/"):
        opened.append(target)
assert not opened, f"import ttrl opened {{opened}}"
print(f"import ttrl {{ttrl.__version__}} clean, no device node opened")
'''

#: Also run in the throwaway venv. Every path under /dev/tenstorrent raises,
#: which is what a chipless machine looks like to the probes in
#: `ttrl.engine.device`, and then the whole public surface is resolved. The
#: lazy config exports matter most: reading and validating a run config is
#: what a stranger does first, and it must not need a chip.
ZERO_CHIP = '''
import builtins, os, sys

_listdir, _scandir, _open = os.listdir, os.scandir, builtins.open

def deny(real):
    def wrapper(path=".", *a, **k):
        if str(path).startswith("/dev/tenstorrent"):
            raise FileNotFoundError(2, "No such file or directory", str(path))
        return real(path, *a, **k)
    return wrapper

os.listdir = deny(_listdir)
os.scandir = deny(_scandir)
builtins.open = deny(_open)

import ttrl
for name in ttrl.__all__:
    getattr(ttrl, name)
assert "ttnn" not in sys.modules, "resolving the public surface imported ttnn"
print(f"ttrl {{ttrl.__version__}} imports and resolves {{len(ttrl.__all__)}} names with zero chips")
'''


class CheckFailed(Exception):
    """A release check said no. The message is what the operator has to fix."""


def run(argv: list[str], cwd: Path | None = None) -> str:
    """Run a command, returning stdout. Non-zero becomes CheckFailed.

    Output is folded into the exception, so a passing check stays one line and
    a failing one shows everything.
    """
    done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    if done.returncode != 0:
        raise CheckFailed(
            f"`{' '.join(argv)}` exited {done.returncode}\n"
            f"{done.stdout}{done.stderr}".rstrip()
        )
    return done.stdout


def check_lock(root: Path) -> str:
    """A stale lock means CI installs something other than what the release
    declares, and the hardware gate then measures a tree nobody is shipping.
    """
    run(["uv", "lock", "--check"], cwd=root)
    return "uv.lock resolves the current pyproject"


def build(root: Path, out: Path) -> tuple[Path, Path]:
    run(["uv", "build", "--out-dir", str(out)], cwd=root)
    wheels = sorted(out.glob("*.whl"))
    sdists = sorted(out.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise CheckFailed(
            f"expected one wheel and one sdist in {out}, got "
            f"{[p.name for p in wheels + sdists]}"
        )
    return wheels[0], sdists[0]


def check_wheel_contents(wheel: Path) -> str:
    """The wheel carries the package, its metadata, and nothing else."""

    names = zipfile.ZipFile(wheel).namelist()
    tops = {name.split("/")[0] for name in names}
    dist_info = {top for top in tops if top.endswith(".dist-info")}
    leaked = tops - ALLOWED_WHEEL_TOPS - dist_info
    if leaked:
        raise CheckFailed(
            f"{wheel.name} carries {sorted(leaked)} beside the package. "
            "Check [tool.setuptools.packages.find] in pyproject.toml."
        )
    cached = [name for name in names if "__pycache__" in name or name.endswith(".pyc")]
    if cached:
        raise CheckFailed(f"{wheel.name} ships bytecode: {cached[:5]}")
    return f"{wheel.name} carries only ttrl/ and {sorted(dist_info)[0]} ({len(names)} files)"


def check_metadata(wheel: Path) -> str:
    """The wheel's METADATA has to carry what an installer and a PyPI sidebar
    need. A missing field here is invisible to every test and to the build.
    """

    archive = zipfile.ZipFile(wheel)
    path = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
    message = email.message_from_bytes(archive.read(path))
    if not message.get("Requires-Python"):
        raise CheckFailed(f"{path} is missing Requires-Python")
    urls = message.get_all("Project-URL") or []
    if not urls:
        raise CheckFailed(
            f"{path} declares no Project-URL, so the PyPI sidebar has no link "
            "to the repository. Set [project.urls] in pyproject.toml."
        )
    return (f"METADATA complete: Requires-Python "
            f"{message.get('Requires-Python')}, {len(urls)} URLs")


def check_install(dist: Path, work: Path) -> str:
    """Install into a throwaway venv and verify what a stranger gets.

    With its real dependencies: resolving the public surface needs numpy, and
    a release whose declared dependencies do not resolve is no release.
    """
    venv = work / f"venv-{dist.name.split('-')[0]}-{dist.suffix.lstrip('.')}"
    run(["uv", "venv", "--python", "3.12", str(venv)])
    python = venv / "bin" / "python"
    run(["uv", "pip", "install", "--python", str(python), str(dist)])

    script = work / f"{venv.name}-checks.py"
    out = []
    for name, source in (("purity", IMPORT_PURITY), ("zero-chip", ZERO_CHIP)):
        script.write_text(source.format(forbidden=FORBIDDEN_ON_IMPORT))
        out.append(run([str(python), "-I", str(script)]).strip())

    console = venv / "bin" / "ttrl"
    if not console.is_file():
        raise CheckFailed(f"the `ttrl` console script is missing from {venv / 'bin'}")
    version = run([str(console), "--version"]).strip()
    out.append(f"`{console.name}` on PATH reports {version!r}")
    return f"{dist.name}: " + "; ".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.release_check",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the build output and venvs",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    work = Path(tempfile.mkdtemp(prefix="ttrl-release-"))
    failures = 0

    def report(label, fn):
        nonlocal failures
        try:
            print(f"ok   {label}: {fn()}")
        except CheckFailed as exc:
            failures += 1
            print(f"FAIL {label}: {exc}")

    print(f"tt-rl release check, repo {root}, work {work}\n")
    report("lock", lambda: check_lock(root))
    try:
        wheel, sdist = build(root, work / "dist")
        print(f"ok   build: {wheel.name}, {sdist.name}")
    except CheckFailed as exc:
        print(f"FAIL build: {exc}")
        print("\n1 check failed; nothing downstream of the build could run.")
        return 1
    report("wheel contents", lambda: check_wheel_contents(wheel))
    report("metadata", lambda: check_metadata(wheel))
    report("install wheel", lambda: check_install(wheel, work))
    report("install sdist", lambda: check_install(sdist, work))

    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    print()
    if failures:
        print(f"{failures} check(s) failed. Do not release.")
        return 1
    print("all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
