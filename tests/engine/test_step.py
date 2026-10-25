"""`TrainStep` in both of its modes, on real hardware.

`eager` is the correctness reference and `traced_fwd_bwd` must match it
numerically. A model containing a `Conv2d` is now fully capturable, which it
was not: conv2d used to pull a device-resident weight back to host on every
call to re-derive its prepared layout, and that read is what capture forbids.
The prepared tensor IS the master weight now, so there is no read and no
eager fallback: `allow_eager_fallback` and the silent downgrade with it are
deleted, because a run that quietly spends itself in eager mode is a
performance cliff nothing forces a caller to notice.

`traced_full`, which put the optimizer inside the trace, is also gone. It
measured +1.3% to +1.7% against a 1.0% noise floor, because those dispatches
were never on the critical path.

Chip-scoped by the caller:

    TT_VISIBLE_DEVICES=0 pytest tests/engine/test_step.py -m tt
"""

import warnings

import numpy as np
import pytest

pytest.importorskip("ttnn")  # skip cleanly on CPU-only CI (no ttnn install)

from ttrl.engine import device
from ttrl.engine.autograd import Tensor
from ttrl.engine.ops import mean_all, square, sub
from ttrl.engine.conv import Conv2d
from ttrl.engine.layers import Linear, ReLU, Sequential
from ttrl.engine.optim import Adam
from ttrl.engine.step import TrainStep

pytestmark = pytest.mark.tt

ATOL = 5e-2
RTOL = 5e-2


def _toy_regression_data(n=64, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n, 1)).astype(np.float32)
    y = (2.0 * x + 1.0).astype(np.float32)
    return x, y


def _make_mlp(dev):
    return Sequential(Linear(1, 16, dev), ReLU(), Linear(16, 1, dev))


def _make_step_fn(model):
    def step_fn(inputs):
        pred = model(inputs["x"])
        loss = mean_all(square(sub(pred, inputs["y"])))
        loss.backward()
        return {"loss": loss}

    return step_fn


def test_eager_mode_trains_toy_regression():
    x_np, y_np = _toy_regression_data()

    with device.acquire() as (dev, _profile):
        np.random.seed(0)
        model = _make_mlp(dev)
        opt = Adam(model.parameters(), lr=1e-2)
        x = Tensor.from_numpy(x_np, dev)
        y = Tensor.from_numpy(y_np, dev)

        step = TrainStep(
            _make_step_fn(model),
            {"x": x, "y": y},
            dev,
            optimizer=opt,
            mode="eager",
        )
        assert step.mode == "eager"

        losses = [step.run()["loss"] for _ in range(20)]

        assert float(np.asarray(losses[0])) > float(np.asarray(losses[-1]))
        assert float(np.asarray(losses[-1])) < float(np.asarray(losses[0])) * 0.9


def test_traced_fwd_bwd_matches_eager():
    x_np, y_np = _toy_regression_data(seed=1)
    n_steps = 8

    with device.acquire() as (dev, _profile):
        np.random.seed(42)
        model_e = _make_mlp(dev)
        init_state = [p.numpy().copy() for p in model_e.parameters()]
        opt_e = Adam(model_e.parameters(), lr=1e-2)
        x_e = Tensor.from_numpy(x_np, dev)
        y_e = Tensor.from_numpy(y_np, dev)
        step_e = TrainStep(
            _make_step_fn(model_e), {"x": x_e, "y": y_e}, dev,
            optimizer=opt_e, mode="eager",
        )
        eager_losses = [step_e.run()["loss"] for _ in range(n_steps)]
        eager_final = [p.numpy() for p in model_e.parameters()]

        # Same init, same data, traced_fwd_bwd this time.
        np.random.seed(42)
        model_t = _make_mlp(dev)
        for p, init in zip(model_t.parameters(), init_state):
            np.testing.assert_allclose(p.numpy(), init, atol=1e-6)
        opt_t = Adam(model_t.parameters(), lr=1e-2)
        x_t = Tensor.from_numpy(x_np, dev)
        y_t = Tensor.from_numpy(y_np, dev)
        step_t = TrainStep(
            _make_step_fn(model_t), {"x": x_t, "y": y_t}, dev,
            optimizer=opt_t, mode="traced_fwd_bwd",
        )
        try:
            traced_losses = [step_t.run()["loss"] for _ in range(n_steps)]
        finally:
            step_t.close()
        traced_final = [p.numpy() for p in model_t.parameters()]

        assert step_t.mode == "traced_fwd_bwd"  # TrainStep has no eager fallback; traced means traced

        for i, (le, lt) in enumerate(zip(eager_losses, traced_losses)):
            np.testing.assert_allclose(
                np.asarray(le), np.asarray(lt), atol=ATOL, rtol=RTOL,
                err_msg=f"loss mismatch at step {i}",
            )
        for pe, pt in zip(eager_final, traced_final):
            np.testing.assert_allclose(pe, pt, atol=ATOL, rtol=RTOL)


def test_traced_fwd_bwd_captures_a_conv_model_and_matches_eager():
    """The acceptance criterion for the prepared-master-weight change.

    A conv model used to be uncapturable, and `TrainStep` raised
    `TraceCaptureError` naming the host read rather than silently running the
    whole run in eager mode. conv2d accepts an already-prepared device weight
    with no host read and no re-prep, and in-place mutation of that weight
    outside the trace is seen by the replay, so forward and backward capture
    together. Channel counts are conv2's, which is what a real DQN uses: a
    1 -> 2 channel toy conv behaved differently and made the old version of
    this test agree with a conclusion that does not hold at production shapes.

    The eager twin is the point. A replay that is merely stale returns finite
    losses forever, so matching eager step for step is the only check that
    separates a live replay from a frozen one.
    """
    batch, h, w = 2, 9, 9
    rng = np.random.default_rng(0)
    x_np = rng.uniform(-1, 1, size=(1, 1, batch * h * w, 32)).astype(np.float32)
    y_np = rng.uniform(-1, 1, size=(1, 1, batch * h * w, 64)).astype(np.float32)
    n_steps = 6

    def _run(dev, mode):
        np.random.seed(0)
        conv = Conv2d((32, h, w), 64, 3, dev, batch=batch, stride=1, padding=1)
        opt = Adam(list(conv.parameters()), lr=1e-2)
        x = Tensor.from_numpy(x_np, dev)
        y = Tensor.from_numpy(y_np, dev)

        def step_fn(inputs):
            out = conv(inputs["x"])
            loss = mean_all(square(sub(out, inputs["y"])))
            loss.backward()
            return {"loss": loss}

        step = TrainStep(step_fn, {"x": x, "y": y}, dev, optimizer=opt, mode=mode)
        try:
            losses = [step.run()["loss"] for _ in range(n_steps)]
            assert step.mode == mode, "the mode must not be downgraded silently"
            return losses, conv.weight_numpy().copy()
        finally:
            step.close()

    with device.acquire() as (dev, _profile):
        eager_losses, eager_weight = _run(dev, "eager")
        traced_losses, traced_weight = _run(dev, "traced_fwd_bwd")

    # Not stale: the loss has to actually move over the run.
    assert abs(traced_losses[-1] - traced_losses[0]) > 1e-4, (
        f"the traced loss did not move ({traced_losses}), so the replay may be "
        "returning a frozen result rather than training"
    )
    for i, (traced, eager) in enumerate(zip(traced_losses, eager_losses)):
        assert traced == pytest.approx(eager, rel=RTOL, abs=ATOL), (
            f"step {i}: traced loss {traced} vs eager {eager}"
        )
    assert np.allclose(traced_weight, eager_weight, rtol=RTOL, atol=ATOL), (
        "the traced run's weights diverged from the eager twin's"
    )


# ---------------------------------------------------------------------------
# Program-cache stability per named mode
# ---------------------------------------------------------------------------
#
# §7 promises "zero program-cache misses per named mode after warm-up
# (asserted)". Until now the only cache assertion in the repo covered
# `conv2d` alone (`tests/engine/test_conv_gradcheck.py`), which is a single op, not a mode --
# a `TrainStep` that recompiled something every iteration would have passed
# it. A cache entry count is a monotone counter, so the correct assertion is
# the strict one: after warm-up the count must not move AT ALL. That is both
# stronger and simpler than any tolerance, and it is the kind of gate that
# can actually go red (a green gate that cannot fail is
# worse than no gate) -- it fails the moment any op in the step reaches a
# shape or dtype combination it has not compiled for.


def _cache_entries_after_warmup(dev, step, warmup=3, iters=8):
    """Run `warmup` iterations to compile everything, then `iters` more and
    report (entries_after_warmup, entries_after_iters)."""
    for _ in range(warmup):
        step.run()
    before = dev.num_program_cache_entries()
    for _ in range(iters):
        step.run()
    return before, dev.num_program_cache_entries()


@pytest.mark.parametrize("mode", ["eager", "traced_fwd_bwd"])
def test_program_cache_is_stable_after_warmup_in_each_mode(mode):
    x_np, y_np = _toy_regression_data()

    with device.acquire() as (dev, _profile):
        np.random.seed(0)
        model = _make_mlp(dev)
        opt = Adam(model.parameters(), lr=1e-2)
        x = Tensor.from_numpy(x_np, dev)
        y = Tensor.from_numpy(y_np, dev)
        step = TrainStep(
            _make_step_fn(model), {"x": x, "y": y}, dev,
            optimizer=opt, mode=mode,
        )
        try:
            before, after = _cache_entries_after_warmup(dev, step)
        finally:
            step.close()

    assert before > 0, (
        f"mode={mode!r}: program cache is empty after warm-up, so this test is "
        "asserting nothing -- the cache is probably disabled on this device."
    )
    assert after == before, (
        f"mode={mode!r}: program cache grew from {before} to {after} entries over "
        f"{after - before} misses across 8 steady-state iterations. Something in the "
        "step is hitting a shape/dtype combination it did not compile during warm-up, "
        "which means per-iteration recompilation in production."
    )


def test_program_cache_is_stable_for_a_conv_model():
    """The conv model is the one that matters: it is what actually trains.

    A step that recompiles every iteration is a throughput cliff that no
    numeric check sees, and only a cache-entry count separates it from a step
    that compiled everything during warm-up.
    """
    rng = np.random.default_rng(0)
    x_np = rng.uniform(0, 1, size=(4, 8, 8, 3)).astype(np.float32)

    with device.acquire() as (dev, _profile):
        np.random.seed(0)
        conv = Conv2d((3, 8, 8), 4, 3, dev, batch=4, padding=1)
        opt = Adam(conv.parameters(), lr=1e-3)
        x = Tensor.from_numpy(x_np, dev)

        def step_fn(inputs):
            out = conv(inputs["x"])
            loss = mean_all(square(out))
            loss.backward()
            return {"loss": loss}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            step = TrainStep(step_fn, {"x": x}, dev, optimizer=opt, mode="eager")
            try:
                before, after = _cache_entries_after_warmup(dev, step, warmup=4, iters=6)
            finally:
                step.close()

    assert before > 0
    assert after == before, (
        f"conv model: program cache grew {before} -> {after} in steady state "
        f"({after - before} misses over 6 iterations)"
    )
