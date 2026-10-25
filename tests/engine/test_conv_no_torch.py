"""The conv package must work with torch absent."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent

_HOOK = """
import sys


class RefuseTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError(f"{name} was imported")
        return None


sys.meta_path.insert(0, RefuseTorch())
"""

_CHECK = """
assert "torch" not in sys.modules, "torch reached sys.modules"
print("OK")
"""

CASES = {
    "the package imports": "import ttrl.engine.conv",
    # The layout search itself: the candidate set a prepared weight's metadata
    # permits, and the packing each candidate implies.
    "a layout can be solved": (
        "import numpy as np\n"
        "from ttrl.engine.conv.packing import candidate_layouts\n"
        "cands = list(candidate_layouts(512, 32, 32, 4, 8, 8))\n"
        "assert [c.cp for c in cands] == [4, 5, 6, 7, 8]\n"
        "assert all(c.row_block == 64 and c.shape == (1, 1, 512, 32) for c in cands)\n"
        "w = np.arange(32 * 4 * 8 * 8, dtype=np.float32).reshape(32, 4, 8, 8)\n"
        "for c in cands:\n"
        "    packed = c.pack(w)\n"
        "    assert packed.shape == (1, 1, 512, 32)\n"
        "    assert np.array_equal(c.unpack(packed), w)"
    ),
}


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _HOOK + body + "\n" + _CHECK],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )


@pytest.mark.parametrize("name", list(CASES))
def test_conv_arithmetic_needs_no_torch(name):
    done = _run(CASES[name])
    assert done.returncode == 0, (
        f"{name!r} imported torch or failed:\n{done.stdout}\n{done.stderr}"
    )
    assert "OK" in done.stdout


def test_the_hook_really_refuses_torch():
    """The control."""
    done = _run("import torch")
    assert done.returncode != 0
    assert "torch was imported" in done.stderr


def test_no_module_under_ttrl_calls_torch():
    """The regression guard that needs neither a chip nor a subprocess."""
    offenders = []
    for path in sorted((REPO / "ttrl").rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if "to_torch" in code or "import torch" in code:
                offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "torch is not a dependency of this library, but these call sites need "
        "it:\n" + "\n".join(offenders)
    )
