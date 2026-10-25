"""The gate that lets `act_block_h_override` stay deleted."""

from __future__ import annotations

import numpy as np
import pytest
import ttnn

from ttrl.engine import EngineConfig, acquire
from ttrl.engine.autograd import Tensor
from ttrl.engine.conv import Conv2d
from ttrl.engine.layers import Flatten, Linear, ReLU, Sequential
from ttrl.engine.ops import (add, apply_done_mask, gather_onehot, huber, max_dim_detached,
                             mean_all, mul)
from ttrl.engine.optim import Adam
from ttrl.engine.tensor import to_host
from ttrl.engine.trace import capture

pytestmark = pytest.mark.tt

BATCH = 32
ACTIONS = 6
SEED = 12345
#: Replays per capture. The hand measurement used 128 and saw 0 divergences;
#: this is the suite-affordable version of the same check.
REPLAYS = 32
UPDATES = 8

CFG = EngineConfig(trace_region_size=192 * 1024 * 1024)


@pytest.fixture(scope="module")
def device():
    with acquire(0, cfg=CFG) as (dev, _profile):
        yield dev


def _tower(dev):
    """A seeded Nature CNN. Identical in every call, which is the point."""
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    def w(shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    c1 = Conv2d((4, 84, 84), 32, 8, dev, batch=BATCH, stride=4, weight=w((32, 4, 8, 8)),
                needs_input_grad=False)
    c2 = Conv2d(c1.out_shape, 64, 4, dev, batch=BATCH, stride=2, weight=w((64, 32, 4, 4)))
    c3 = Conv2d(c2.out_shape, 64, 3, dev, batch=BATCH, stride=1, weight=w((64, 64, 3, 3)))
    flat = Flatten(*c3.out_shape[1:], c3.out_shape[0])
    return Sequential(c1, ReLU(), c2, ReLU(), c3, ReLU(), flat,
                      Linear(flat.out_features, 512, dev), ReLU(),
                      Linear(512, ACTIONS, dev))


def _batch(dev):
    rng = np.random.default_rng(0)
    rows = BATCH * 84 * 84
    onehot = np.zeros((BATCH, ACTIONS), np.float32)
    onehot[np.arange(BATCH), rng.integers(0, ACTIONS, BATCH)] = 1.0

    def obs():
        return Tensor.from_numpy((rng.standard_normal((1, 1, rows, 4)) * 0.3).astype(np.float32),
                                 dev, layout=ttnn.ROW_MAJOR_LAYOUT)

    return dict(
        s=obs(), s2=obs(),
        a_onehot=Tensor.from_numpy(onehot, dev),
        r=Tensor.from_numpy(rng.standard_normal((BATCH, 1)).astype(np.float32), dev),
        done=Tensor.from_numpy((rng.random((BATCH, 1)) < 0.1).astype(np.float32), dev),
    )


def _dqn_loss(online, target, data, gamma=0.99):
    """One DQN update's loss: Q(s,a) against r + gamma * (1-done) * max Q'(s')."""
    q = gather_onehot(online(data["s"]), data["a_onehot"])
    boot = apply_done_mask(max_dim_detached(target(data["s2"]), dim=-1), data["done"])
    return mean_all(huber(q, add(data["r"], mul(boot, gamma))))


def _capture_update(dev, online, target, data):
    """Capture a whole update's forward and backward, warming the cache first."""
    _dqn_loss(online, target, data).backward()
    for param in online.parameters():
        param.zero_grad()

    def record():
        loss = _dqn_loss(online, target, data)
        loss.backward()
        return loss

    trace, loss_buf = capture(dev, record)
    grads = {name: p.grad for name, p in online.named_parameters()}
    assert not [n for n, g in grads.items() if g is None], "capture produced no gradients"
    return trace, loss_buf, grads


def _fingerprint(trace, loss_buf, grads, watch):
    trace.replay(blocking=True)
    return (to_host(loss_buf.data).copy(),
            tuple(to_host(grads[name]).copy() for name in watch))


def _same(a, b):
    return (np.array_equal(a[0], b[0])
            and all(np.array_equal(x, y) for x, y in zip(a[1], b[1])))


def test_captured_dqn_update_replays_bit_identically(device):
    """Every replay of one capture reproduces the first, bit for bit."""
    online, target = _tower(device), _tower(device)
    target.load_state_dict(online.state_dict())
    data = _batch(device)
    trace, loss_buf, grads = _capture_update(device, online, target, data)
    watch = [n for n, _ in online.named_parameters() if "weight" in n][:3]
    try:
        base = _fingerprint(trace, loss_buf, grads, watch)
        divergent = [i for i in range(2, REPLAYS + 1)
                     if not _same(_fingerprint(trace, loss_buf, grads, watch), base)]
    finally:
        trace.close()
    assert not divergent, (
        f"{len(divergent)} of {REPLAYS} replays of one captured DQN update diverged "
        f"(first at replay {divergent[0] if divergent else None}). This is the "
        f"nondeterminism act_block_h_override was pinned for; see the record in "
        f"ttrl/engine/conv/config.py before changing anything else."
    )


def test_two_captures_in_one_process_agree(device):
    """Capture-to-capture, which replaying one capture cannot see."""
    online, target = _tower(device), _tower(device)
    target.load_state_dict(online.state_dict())
    data = _batch(device)
    prints = []
    for _ in range(2):
        trace, loss_buf, grads = _capture_update(device, online, target, data)
        watch = [n for n, _ in online.named_parameters() if "weight" in n][:3]
        try:
            prints.append(_fingerprint(trace, loss_buf, grads, watch))
        finally:
            trace.close()
        for param in online.parameters():
            param.zero_grad()
    assert _same(prints[0], prints[1]), (
        "two captures of the same DQN update, same weights and same inputs, "
        "replayed to different results"
    )


def test_traced_updates_match_an_eager_twin_step_for_step(device):
    """The traced path and the eager path stay bit-identical while Adam steps."""
    online, target = _tower(device), _tower(device)
    eager, eager_target = _tower(device), _tower(device)
    for dst in (target, eager, eager_target):
        dst.load_state_dict(online.state_dict())
    data = _batch(device)
    opt_traced = Adam(list(online.parameters()), lr=1e-4)
    opt_eager = Adam(list(eager.parameters()), lr=1e-4)
    trace, loss_buf, grads = _capture_update(device, online, target, data)
    worst, divergent = 0.0, []
    try:
        for step in range(1, UPDATES + 1):
            trace.replay(blocking=True)
            traced_loss = float(to_host(loss_buf.data).reshape(-1)[0])
            for name, param in online.named_parameters():
                param.grad = grads[name]
            opt_traced.step()
            loss = _dqn_loss(eager, eager_target, data)
            eager_loss = float(to_host(loss.data).reshape(-1)[0])
            loss.backward()
            opt_eager.step()
            for param in eager.parameters():
                param.zero_grad()
            delta = abs(traced_loss - eager_loss)
            for traced_p, eager_p in zip(online.parameters(), eager.parameters()):
                delta = max(delta, float(np.abs(to_host(traced_p.data)
                                                - to_host(eager_p.data)).max()))
            worst = max(worst, delta)
            if delta != 0.0:
                divergent.append(step)
    finally:
        trace.close()
    assert not divergent, (
        f"{len(divergent)} of {UPDATES} traced updates diverged from the eager twin "
        f"(worst absolute difference {worst:.3e}, first at update {divergent[0]})"
    )
