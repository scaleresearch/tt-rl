"""Version-compat test for the pinned `ttnn` range.

The pin is deliberately tight (currently `>=0.80.0,<0.81`) because an unusual
amount of this codebase is calibrated to ttnn behaviours that are bugs or
quirks rather than API contract, and which a minor bump could change in either
direction without changing a signature. The three that shaped the design were
all characterised against 0.77, and none has been re-characterised against the
0.80 pin:

* `ttnn.conv2d` used to do an unconditional host read to re-derive its
  prepared weight layout, and `prepare_conv_weights` TT_FATALs on a
  device-storage weight. The prepared tensor is the master weight now, so
  there is no read and a conv model captures whole;
  `tests/engine/test_step.py` is where that is asserted. A ttnn that puts the
  read back takes trace capture with it.
* Row-major fp32 ops that move data along the innermost axis silently truncate
  to ~8 mantissa bits, which is why `ttrl/engine/conv/transform.py`'s `_flip` appends a
  trailing unit axis.
* The fused `*_bw` kernels were taken to be bf16/bfloat8_b + TILE only. On
  0.80 they accept and return fp32, so the engine runs fp32 end to end; the
  0.80 docstrings still say BFLOAT16, so a bump needs re-measuring rather than
  re-reading them. See `ttrl/engine/tensor/precision.py`.

None of those are things a caller can feature-detect cheaply, so the honest
mechanism is: pin narrowly, and make the bump a deliberate act that re-runs
the spikes. This test is what turns "someone edited the pin" into a failure
rather than a silent behaviour change -- it asserts the installed version is
inside the declared range AND that the specific API surface ttrl calls still
exists, so a bump that keeps the version legal but drops a symbol still fails
here rather than mid-training-run.

Deliberately does not open a device: it runs on CPU CI, and on a box whose
chips are all busy.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ttnn = pytest.importorskip("ttnn")

from importlib.metadata import version as _installed_version  # noqa: E402

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _declared_ttnn_spec() -> str:
    data = tomllib.loads(PYPROJECT.read_text())
    for dep in data["project"]["dependencies"]:
        if re.match(r"^ttnn\b", dep):
            return dep
    raise AssertionError("pyproject declares no ttnn dependency")


def _parse(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", v)[:3])


def test_ttnn_dependency_declares_both_a_floor_and_a_ceiling():
    spec = _declared_ttnn_spec()
    assert ">=" in spec, f"ttnn pin has no floor: {spec!r}"
    assert "<" in spec.replace("<=", ""), (
        f"ttnn pin has no upper bound: {spec!r}. An open upper bound is not a pin -- "
        "see this module's docstring for the version-specific behaviours ttrl is "
        "calibrated to. Raise the ceiling only after re-running the conv spike."
    )


def test_installed_ttnn_is_inside_the_declared_range():
    spec = _declared_ttnn_spec()
    floor = _parse(re.search(r">=\s*([\d.]+)", spec).group(1))
    ceiling = _parse(re.search(r"(?<![<>=])<\s*([\d.]+)", spec).group(1))
    installed = _parse(_installed_version("ttnn"))
    assert floor <= installed < ceiling, (
        f"installed ttnn {installed} is outside the declared range {spec!r}. "
        "Either the environment drifted or the pin was raised without validating: "
        "re-run the hardware suite and the conv/precision spikes."
    )


@pytest.mark.parametrize(
    "name",
    [
        # Core tensor / layout surface.
        "Tensor", "float32", "bfloat16", "TILE_LAYOUT", "ROW_MAJOR_LAYOUT",
        "from_torch" if hasattr(ttnn, "from_torch") else "to_layout",
        "to_layout", "typecast", "reshape", "permute", "concat", "slice",
        # Ops the model and autograd tape call.
        "conv2d", "conv_transpose2d", "matmul", "relu", "add", "subtract",
        "multiply", "zeros",
        # Config/enum surface `ttrl.engine.conv` and the compute-config helpers need.
        # Not ops, but a rename here breaks import, not just a call site.
        "Conv2dConfig", "StorageType", "TensorMemoryLayout", "MathFidelity",
        "init_device_compute_kernel_config",
        # The pieces that make training possible at all.
        "open_device", "close_device",
    ],
)
def test_required_ttnn_symbols_exist(name):
    assert hasattr(ttnn, name), (
        f"ttnn {_installed_version('ttnn')} has no attribute {name!r}, which ttrl calls. "
        "The pin's range says this version should be compatible; it is not."
    )


def test_prepare_conv_weights_still_exists():
    """The symbol whose *behaviour* decided the trace gate.

    `ttrl.engine.conv` prepares its weights through it once and keeps the
    prepared tensor as the master. If a future ttnn removes or renames it, the
    conv spike's conclusion must be re-derived rather than inherited.
    """
    assert hasattr(ttnn, "prepare_conv_weights")
