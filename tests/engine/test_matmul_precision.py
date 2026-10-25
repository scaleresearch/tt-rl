"""Every matmul runs at HiFi4 with fp32 destination accumulation, both dtypes."""

from __future__ import annotations

import pytest

pytest.importorskip("ttnn")
import ttnn

from ttrl.engine.autograd import Tensor
# `ttrl.engine.ops` re-exports the op under its module's own name, so
# `import ttrl.engine.ops.matmul as m` binds the function; reach the module
# through sys.modules instead. The cache is mutated in place, so importing the
# dict object is equivalent to reaching it through the module.
from ttrl.engine.ops.matmul import _MATMUL_ACC_CFG, matmul, purge_matmul_caches

from tests._fakes import MatmulData as _FakeData


class _FakeDevice:
    def arch(self):
        return "ARCH"


@pytest.fixture
def recorder(monkeypatch):
    """Record every `ttnn.matmul` call the op makes."""
    calls = []
    device = _FakeDevice()

    def fake_matmul(a, b, **kwargs):
        calls.append(kwargs)
        return _FakeData(ttnn.float32, device)

    def fake_config(arch, **kwargs):
        return ("cfg", arch, tuple(sorted(kwargs.items())))

    monkeypatch.setattr(ttnn, "matmul", fake_matmul)
    monkeypatch.setattr(ttnn, "init_device_compute_kernel_config", fake_config)
    _MATMUL_ACC_CFG.clear()
    yield calls, device
    _MATMUL_ACC_CFG.clear()


def _operand(dtype, device, *, requires_grad=False):
    return Tensor(_FakeData(dtype, device), requires_grad=requires_grad)


def _assert_accurate(kwargs):
    cfg = kwargs.get("compute_kernel_config")
    assert cfg is not None, (
        "the matmul named no compute config, so ttnn accumulates the "
        "K-dimension sum in bf16 destination registers"
    )
    settings = dict(cfg[2])
    assert settings["fp32_dest_acc_en"] is True
    assert settings["math_fidelity"] == ttnn.MathFidelity.HiFi4


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_the_forward_states_hifi4_and_fp32_accumulation(recorder, dtype):
    calls, device = recorder
    dt = getattr(ttnn, dtype)
    matmul(_operand(dt, device), _operand(dt, device))
    assert len(calls) == 1
    _assert_accurate(calls[0])


def test_both_backward_matmuls_get_the_same_treatment(recorder):
    """dA and dW are reductions over the same K."""
    calls, device = recorder
    dt = ttnn.float32
    a = _operand(dt, device, requires_grad=True)
    b = _operand(dt, device, requires_grad=True)
    out = matmul(a, b)
    calls.clear()
    out._backward(_FakeData(dt, device))

    assert len(calls) == 2
    for kwargs in calls:
        _assert_accurate(kwargs)


def test_purging_drops_the_entry_before_the_device_dies(recorder):
    """`id()` is reused once the device object is freed."""
    _calls, device = recorder
    matmul(_operand(ttnn.float32, device), _operand(ttnn.float32, device))
    assert _MATMUL_ACC_CFG

    purge_matmul_caches(device)
    assert _MATMUL_ACC_CFG == {}
