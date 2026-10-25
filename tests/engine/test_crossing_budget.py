"""The host<->device crossing budget for an act and for an update."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ttnn")
import ttnn

from ttrl.engine import device
from ttrl.engine.autograd import Tensor, no_grad
from ttrl.engine.ops import mean_all, square, sub
from ttrl.engine.optim import Adam
from ttrl.engine.step import TrainStep

from tests.engine._crossings import CrossingCounter, make_mlp

pytestmark = pytest.mark.tt


def test_act_crossing_budget():
    """One observation up, one action down."""
    with device.acquire() as (dev, _profile):
        model = make_mlp(dev)
        obs_np = np.random.default_rng(0).uniform(-1, 1, size=(1, 8)).astype(np.float32)

        with CrossingCounter() as counter:
            with no_grad():
                obs = Tensor.from_numpy(obs_np, dev)
                q = model(obs)
                action_dev = ttnn.argmax(q.data, dim=-1)
                _action = action_dev.cpu()

        assert counter.uploads == 1, f"act uploaded {counter.uploads} times, budget is 1"
        assert counter.downloads == 1, f"act downloaded {counter.downloads} times, budget is 1"


def _make_multi_metric_update(dev):
    """A `TrainStep` whose `step_fn` returns three `Tensor` metrics (loss, mean_pred, grad_sq) so the packed download path runs."""
    model = make_mlp(dev)
    opt = Adam(model.parameters(), lr=1e-2)

    x0 = np.zeros((4, 8), dtype=np.float32)
    y0 = np.zeros((4, 4), dtype=np.float32)
    inputs = {
        "x": Tensor.from_numpy(x0, dev),
        "y": Tensor.from_numpy(y0, dev),
    }

    def step_fn(ins):
        pred = model(ins["x"])
        diff = sub(pred, ins["y"])
        loss = mean_all(square(diff))
        loss.backward()
        grad_sq = None
        for p in model.parameters():
            if p.grad is None:
                continue
            part = ttnn.sum(ttnn.multiply(p.grad, p.grad))
            grad_sq = part if grad_sq is None else ttnn.add(grad_sq, part)
        metrics = {"loss": loss, "mean_pred": mean_all(pred)}
        if grad_sq is not None:
            metrics["grad_sq"] = Tensor(grad_sq, requires_grad=False)
        return metrics

    step = TrainStep(step_fn, inputs, dev, optimizer=opt, mode="eager")
    return step, inputs


def test_update_metrics_download_budget():
    """update (download side)."""
    with device.acquire() as (dev, _profile):
        step, inputs = _make_multi_metric_update(dev)

        x_np = np.random.default_rng(1).uniform(-1, 1, size=(4, 8)).astype(np.float32)
        y_np = np.random.default_rng(2).uniform(-1, 1, size=(4, 4)).astype(np.float32)

        # warm-up (program cache, staging buffers) outside the measured window
        step.load_input("x", x_np)
        step.load_input("y", y_np)
        step.run()

        with CrossingCounter() as counter:
            step.load_input("x", x_np)
            step.load_input("y", y_np)
            out = step.run()

        assert set(out) == {"loss", "mean_pred", "grad_sq"}
        assert counter.downloads == 1, (
            f"update downloaded {counter.downloads} times for a 3-key metrics "
            "dict; budget is 1 packed download -- see "
            "ttrl.engine.step.TrainStep._read_metrics."
        )


