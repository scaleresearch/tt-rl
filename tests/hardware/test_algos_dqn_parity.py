"""DQN's TD target, Huber loss and gradients against a float64 torch twin."""

from __future__ import annotations

import numpy as np
import pytest

from tests._fakes.torch_twin import (DeviceGates, assert_close,
                                     assert_grads_match, t64, torch_stack)
from ttrl.algos.dqn import DQN, DQNConfig
from ttrl.algos.dqn.losses import leaf_key
from ttrl.engine import acquire
from ttrl.nets.obs import ObsPair
from ttrl.spaces import DEFAULT_LEAF, ActionSpec, ObsGroup, ObsSpec

pytestmark = pytest.mark.tt

#: 17 features, 6 actions, batch 64. The shape matters: the layer-0 bias
#: gradient is a signed sum over the batch, so at batch 32 and 6 features it is
#: a near-total cancellation and lands at pcc 0.9989, just under the gate, on
#: arithmetic that is otherwise identical. Measured, both shapes, same code.
OBS = ObsGroup.of(ObsSpec.vector(17))
ACTIONS = ActionSpec.discrete(6)
BATCH = 64


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


def _inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "obs": rng.normal(size=(BATCH, OBS[DEFAULT_LEAF].size)).astype(np.float32),
        "next_obs": rng.normal(size=(BATCH, OBS[DEFAULT_LEAF].size)).astype(np.float32),
        "rewards": rng.normal(size=(BATCH, 1)).astype(np.float32),
        "dones": (rng.random((BATCH, 1)) < 0.25).astype(np.float32),
        "discounts": (0.99 ** rng.integers(1, 4, (BATCH, 1))).astype(np.float32),
        "actions": rng.integers(0, ACTIONS.n, BATCH),
    }


def _load_step(agent, data: dict) -> None:
    """Fill every bound buffer. An observation occupies one buffer per leaf."""

    for which in ("obs", "next_obs"):
        agent._step.load_input(leaf_key(which, DEFAULT_LEAF), data[which])
    for name in ("rewards", "dones", "discounts"):
        agent._step.load_input(name, data[name])


def _device_gates(agent):
    """The relu gate pattern of the three forwards one DQN update runs.

    The update itself happens inside a `TrainStep`, which hands back metrics
    rather than a tape, and its target half runs under `no_grad`, which
    records none. So the gates are read off forwards of the same nets over the
    same bound input buffers, run before the step moves the weights. The
    arithmetic is deterministic, so these are the gates the step computes; a
    step that computed others would miss the tightened ceiling by the size of
    a flipped gate.
    """

    inputs = agent._step.inputs

    def pair(which):
        return ObsPair.shared(
            {DEFAULT_LEAF: inputs[leaf_key(which, DEFAULT_LEAF)]})

    return DeviceGates(agent.online(pair("obs")),
                       agent.online(pair("next_obs")),
                       agent.target(pair("next_obs")))


def _torch_q(net, x, prefix, gates=None):
    """float64 twin of an encoder plus its Q head. -> (q, {name: param})."""
    features, named = torch_stack(
        net.encoder[DEFAULT_LEAF], x, gates=gates,
        prefix=f"{prefix}encoder.towers.{DEFAULT_LEAF}.layers.")
    weight = t64(net.head.q.weight.numpy())
    bias = t64(net.head.q.bias.numpy())
    named[f"{prefix}head.q.weight"] = weight
    named[f"{prefix}head.q.bias"] = bias
    return features @ weight + bias, named


@pytest.mark.parametrize("double", [True, False])
def test_td_target_huber_loss_and_gradients_match_torch(double, device):
    torch = pytest.importorskip("torch")

    cfg = DQNConfig(batch_size=BATCH, learning_starts=BATCH, double=double,
                    n_step=3, huber_delta=1.0)
    agent = DQN(OBS, ACTIONS, device, cfg=cfg, n_envs=1, seed=11)
    try:
        # The target is born as a copy of the online net, which makes Double
        # and vanilla DQN the same arithmetic, so the parametrisation would
        # prove nothing. Jitter it apart first.
        jitter = np.random.default_rng(3)
        agent.target.load_state_dict(
            {name: array + jitter.normal(scale=0.2, size=array.shape).astype(np.float32)
             for name, array in agent.target.state_dict().items()}, device)
        data = _inputs()
        onehot = np.zeros((BATCH, ACTIONS.n), np.float32)
        onehot[np.arange(BATCH), data["actions"]] = 1.0
        _load_step(agent, data)
        agent._step.load_input("actions_onehot", onehot)

        gates = _device_gates(agent)
        obs64 = torch.tensor(data["obs"], dtype=torch.float64)
        next64 = torch.tensor(data["next_obs"], dtype=torch.float64)
        q_online, online_params = _torch_q(agent.online, obs64, "", gates)
        q_next_online, _ = _torch_q(agent.online, next64, "x.", gates)
        q_next_target, _ = _torch_q(agent.target, next64, "t.", gates)
        gates.assert_every_site_used()

        rows = torch.arange(BATCH)
        taken = torch.tensor(data["actions"], dtype=torch.long)
        pred = q_online[rows, taken]
        selector = q_next_online if double else q_next_target
        next_actions = selector.detach().argmax(dim=-1)
        bootstrap = q_next_target.detach()[rows, next_actions]
        rewards = torch.tensor(data["rewards"].reshape(-1), dtype=torch.float64)
        mask = torch.tensor(1.0 - data["dones"].reshape(-1), dtype=torch.float64)
        discounts = torch.tensor(data["discounts"].reshape(-1), dtype=torch.float64)
        target = rewards + discounts * mask * bootstrap
        want_loss = torch.nn.functional.huber_loss(pred, target, delta=cfg.huber_delta)
        # The control: the other selection rule on the same inputs must give a
        # different loss, or this test cannot tell the two apart.
        other = q_next_target if double else q_next_online
        other_target = rewards + discounts * mask * q_next_target.detach()[
            rows, other.detach().argmax(dim=-1)]
        other_loss = float(torch.nn.functional.huber_loss(
            pred, other_target, delta=cfg.huber_delta))
        assert abs(float(want_loss.detach()) - other_loss) > 1e-3, (
            "double and vanilla DQN agree on this batch, so the parametrisation "
            "is degenerate"
        )
        want_loss.backward()

        out = agent._step.run()
        got_loss = float(np.asarray(out["loss"]).reshape(-1)[0])
        assert_close(got_loss, float(want_loss.detach()),
                                  tolerance=2e-2, what="dqn loss")

        pairs = [(name, param, online_params[name])
                 for name, param in agent.online.named_parameters()]
        assert len(pairs) == len(online_params)
        report = assert_grads_match(pairs)
        for name, (pcc, frobenius) in report.items():
            print(f"  {name}: pcc {pcc:.8f} frobenius {frobenius:.3e}")
    finally:
        agent.close()


def test_n_step_target_uses_the_batch_discounts_not_a_fixed_power(device):
    """A window an episode boundary cut short must bootstrap with its own
    discount. A fixed gamma ** n_step is silently wrong for those samples."""
    torch = pytest.importorskip("torch")

    cfg = DQNConfig(batch_size=BATCH, learning_starts=BATCH, n_step=3, gamma=0.5)
    agent = DQN(OBS, ACTIONS, device, cfg=cfg, n_envs=1, seed=5)
    try:
        data = _inputs(seed=2)
        data["dones"][:] = 0.0
        # Half the batch summed one reward, half summed three.
        data["discounts"][: BATCH // 2] = 0.5
        data["discounts"][BATCH // 2:] = 0.125
        onehot = np.zeros((BATCH, ACTIONS.n), np.float32)
        onehot[np.arange(BATCH), data["actions"]] = 1.0
        _load_step(agent, data)
        agent._step.load_input("actions_onehot", onehot)

        # The reference is built BEFORE the step runs: `TrainStep.run` steps
        # the optimizer, so a twin built after it would be a twin of the
        # updated weights and would disagree by the size of one Adam step.
        next64 = torch.tensor(data["next_obs"], dtype=torch.float64)
        obs64 = torch.tensor(data["obs"], dtype=torch.float64)
        q_online, _ = _torch_q(agent.online, obs64, "")
        q_next_online, _ = _torch_q(agent.online, next64, "x.")
        q_next_target, _ = _torch_q(agent.target, next64, "t.")
        rows = torch.arange(BATCH)
        pred = q_online[rows, torch.tensor(data["actions"], dtype=torch.long)]
        bootstrap = q_next_target.detach()[rows, q_next_online.detach().argmax(-1)]
        rewards = torch.tensor(data["rewards"].reshape(-1), dtype=torch.float64)
        per_sample = torch.tensor(data["discounts"].reshape(-1), dtype=torch.float64)
        want = float(torch.nn.functional.huber_loss(
            pred, rewards + per_sample * bootstrap, delta=1.0))
        fixed = float(torch.nn.functional.huber_loss(
            pred, rewards + (0.5 ** 3) * bootstrap, delta=1.0))

        got = float(np.asarray(agent._step.run()["loss"]).reshape(-1)[0])

        assert_close(got, want, tolerance=1e-3, what="per-sample discount loss")
        assert abs(want - fixed) > 1e-3, (
            "the control is degenerate: this batch does not distinguish the two"
        )
    finally:
        agent.close()
