"""No arch-specific ttnn names outside `ttrl/engine/device/`."""

from pathlib import Path

from tools.lint_portability import violations

PACKAGE = Path(__file__).resolve().parents[2] / "ttrl"


def test_no_arch_specific_ttnn_names_outside_device_module():
    assert len(sorted(PACKAGE.rglob("*.py"))) >= 50, "the lint root matches no package"
    vs = violations(PACKAGE)
    assert not vs, "portability violations:\n" + "\n".join(str(v) for v in vs)
