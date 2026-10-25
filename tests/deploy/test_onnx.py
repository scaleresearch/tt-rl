"""What `ttrl.export_onnx` writes: one input, one output, and nothing implied.

The arithmetic is `tests/deploy/test_onnx_parity.py`'s job. This file holds
the contract a consumer reads the file under, because every one of these is a
promise somebody's deployment makes without asking: that the graph takes the
RAW observation, that it declares an opset and an IR version a current runtime
accepts, and that a discrete policy's graph carries no invented bounds.
"""

from __future__ import annotations

import sys

import pytest

from tests.deploy import _onnx
from tests.deploy.test_export_algos import export
import ttrl
from ttrl.deploy.graph import OPSET, REQUIRES


@pytest.fixture
def onnx_module():
    return _onnx.require_onnx()[0]


def graph_for(tmp_path, algo, kind):
    artifact = export(tmp_path, algo, kind)
    return ttrl.export_onnx(artifact, str(tmp_path / "policy.onnx"))


def test_the_writer_refuses_without_its_extra_and_names_it(monkeypatch):
    """The `ttrl.train.trackers` shape: the import is inside the call, so
    `import ttrl` costs nothing, and the refusal names the extra. A caller who
    asked for ONNX and silently got the npz has no way to find out."""

    package, extra = REQUIRES
    monkeypatch.setitem(sys.modules, package, None)
    monkeypatch.delitem(sys.modules, f"{package}.helper", raising=False)
    with pytest.raises(ModuleNotFoundError) as caught:
        ttrl.export_onnx("unread.npz", "unwritten.onnx")
    message = str(caught.value)
    assert package in message and f"tt-rl[{extra}]" in message


@pytest.mark.parametrize("algo,kind", [("sac", "continuous"), ("ppo", "discrete")])
def test_the_graph_pins_its_opset_and_the_oldest_ir_that_carries_it(
        tmp_path, onnx_module, algo, kind):
    """Left unpinned, `make_model` stamps the installed library's newest IR
    and a runtime one release behind refuses the file whatever the opset says.
    onnxruntime 1.31 rejects the IR 14 that onnx 1.23 writes by default."""
    model = onnx_module.load(graph_for(tmp_path, algo, kind))
    assert [(o.domain, o.version) for o in model.opset_import] == [("", OPSET)]
    assert model.ir_version == onnx_module.helper.find_min_ir_version_for(
        [onnx_module.helper.make_operatorsetid("", OPSET)])
