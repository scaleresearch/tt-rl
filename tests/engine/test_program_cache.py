"""Program-cache misses."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from tests._fakes import FakeModule
from ttrl.engine.mesh import ParamSync, open_mesh_for_chips
import ttrl.engine.mesh.sync as dmod

pytest.importorskip("ttnn")
import ttnn

import ttrl.engine.autograd as autograd
from ttrl.engine import device
from ttrl.engine.autograd import Tensor
from ttrl.engine.layers import Linear, ReLU, Sequential

pytestmark = pytest.mark.tt


from tests._fakes.chips import TWO_CHIPS as _TWO_CHIPS, two_idle_chips_gate  # noqa: E402

requires_two_idle_chips = two_idle_chips_gate(
    "needs 2 idle Tenstorrent chips for a real ttnn.all_reduce "
    f"(candidates {_TWO_CHIPS}); scope the process with TT_VISIBLE_DEVICES"
)


@contextlib.contextmanager
def _misses_disallowed(dev):
    dev.set_program_cache_misses_allowed(False)
    try:
        yield
    finally:
        dev.set_program_cache_misses_allowed(True)


def _make_mlp(dev):
    return Sequential(Linear(8, 16, dev), ReLU(), Linear(16, 4, dev))


def test_act_zero_misses_after_warmup():
    with device.acquire() as (dev, _profile):
        model = _make_mlp(dev)
        obs_np = np.random.default_rng(0).uniform(-1, 1, size=(1, 8)).astype(np.float32)

        def act_once():
            with autograd.no_grad():
                obs = Tensor.from_numpy(obs_np, dev)
                q = model(obs)
                action = ttnn.argmax(q.data, dim=-1).cpu().to_numpy()
            return action

        act_once()  # warm-up: fills the program cache

        with _misses_disallowed(dev):
            act_once()  # must build nothing new


@requires_two_idle_chips
def test_sync_zero_misses_after_warmup():
    """`ParamSync.average`'s all-reduce chokepoint builds nothing new once warm."""

    def fake_write_back(param, device_shard, original_shape):
        array = np.ascontiguousarray(
            device_shard.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_numpy()
        ).reshape(original_shape)
        param._array = np.array(array, dtype=np.float32).reshape(param.shape)

    with open_mesh_for_chips(_TWO_CHIPS) as mesh:
        m1 = FakeModule([np.full((32, 32), 2.0, dtype=np.float32)])
        m2 = FakeModule([np.full((32, 32), 4.0, dtype=np.float32)])
        sync = ParamSync(mesh=mesh, modules=[m1, m2])

        orig = dmod._write_device_result_in_place
        dmod._write_device_result_in_place = fake_write_back
        try:
            sync.average()  # warm-up: fills the program cache
            # The warm-up must have really reduced; otherwise the guarded run
            # below could be green because nothing ran at all.
            np.testing.assert_allclose(m1.parameters()[0].numpy(), 3.0)
            np.testing.assert_allclose(m2.parameters()[0].numpy(), 3.0)

            with _misses_disallowed(mesh):
                sync.average()  # must build nothing new

            np.testing.assert_allclose(m1.parameters()[0].numpy(), 3.0)
        finally:
            dmod._write_device_result_in_place = orig
