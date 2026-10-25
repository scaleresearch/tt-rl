"""SAC's critic, actor and temperature losses and their gradients against a
float64 torch twin.
"""

from __future__ import annotations

from functools import reduce

import numpy as np
import pytest
import torch

from tests._fakes.torch_twin import (DeviceGates, assert_close,
                                     assert_grads_match, t64, torch_clamp,
                                     torch_minimum, torch_relu, torch_stack)
from ttrl.algos.sac import SAC, SACConfig
from ttrl.engine import acquire, backward
from ttrl.spaces import DEFAULT_LEAF, ActionSpec, ObsGroup, ObsSpec

pytestmark = pytest.mark.tt

OBS = ObsGroup.of(ObsSpec.vector(17))
ACTIONS = ActionSpec.continuous([-1.0] * 4, [1.0] * 4)
BATCH = 64

#: The soft target is computed on the device under `no_grad`, so it carries
#: fp32 arithmetic error AND whatever relu gates the fp32 forward chose, with
#: no tape to read those gates off. It is gated as a value at this relative
#: Frobenius ceiling and then used as a constant, which keeps the critic's
#: gradient comparison free of it. Measured 1.6e-4 to 6.7e-4 over 8 seeds and
#: four critic topologies.
TARGET_CEILING = 2e-3


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


#: Both fusion points of Q, which are the two critic topologies the builder
#: can be asked for: the action on the critic's input, and the action after
#: the observation's own tower. The reference in `_torch_critic` is written
#: against the module tree rather than against a fixed depth, so one twin
#: covers both and a change to either topology is a parity failure here.
#:
#: The depth is no longer limited by the comparison: the twin takes its relu
#: gate pattern from the device, so a flipped fp32 gate is no longer the
#: dominant term in the first weight of a deep head. `("identity", (256, 256))`
#: is the SHIPPED default for a vector observation and is carried here for
#: that reason; the other two are the topologies a config can still name.
FUSIONS = (("identity", (256,)), ("identity", (256, 256)), ("mlp", (256,)))


@pytest.fixture(params=FUSIONS,
                ids=[f"{name}{len(widths)}" for name, widths in FUSIONS])
def agent(device, request):
    critic_encoder, critic_hidden = request.param
    cfg = SACConfig(batch_size=BATCH, learning_starts=BATCH, n_critics=2,
                    init_alpha=0.3, tau=0.01,
                    critic_encoder=critic_encoder, critic_hidden=critic_hidden)
    built = SAC(OBS, ACTIONS, device, cfg=cfg, n_envs=1, seed=13)
    try:
        yield built
    finally:
        built.close()


def _load(agent, seed: int = 0) -> dict:
    """Fill every bound buffer with known values and return the host copies."""
    rng = np.random.default_rng(seed)
    data = {
        "obs": rng.normal(size=(BATCH, OBS[DEFAULT_LEAF].size)).astype(np.float32),
        "next_obs": rng.normal(size=(BATCH, OBS[DEFAULT_LEAF].size)).astype(np.float32),
        "actions": rng.uniform(-1.0, 1.0, (BATCH, ACTIONS.dim)).astype(np.float32),
        "rewards": rng.normal(size=(BATCH, 1)).astype(np.float32),
        "dones": (rng.random((BATCH, 1)) < 0.25).astype(np.float32),
        "discounts": np.full((BATCH, 1), 0.99, np.float32),
        "noise": rng.standard_normal((BATCH, ACTIONS.dim)).astype(np.float32),
        "next_noise": rng.standard_normal((BATCH, ACTIONS.dim)).astype(np.float32),
    }
    for name, array in data.items():
        # Observations go in through the views, which hold the actor's half of
        # the batch and the critic's; everything else is one named buffer.
        if name in ("obs", "next_obs"):
            agent._batch.load_obs(name, array)
        else:
            agent._batch.load(name, array)
    return data


def _torch_policy(policy, x, gates=None):
    features, named = torch_stack(
        policy.encoder[DEFAULT_LEAF], x, gates=gates,
        prefix=f"encoder.towers.{DEFAULT_LEAF}.layers.")
    head = policy.head
    mw, mb = t64(head.heads["mean"].weight.numpy()), t64(head.heads["mean"].bias.numpy())
    sw, sb = (t64(head.heads["log_std"].weight.numpy()),
              t64(head.heads["log_std"].bias.numpy()))
    named.update({"head.heads.mean.weight": mw, "head.heads.mean.bias": mb,
                  "head.heads.log_std.weight": sw, "head.heads.log_std.bias": sb})
    log_std = torch_clamp(features @ sw + sb, head.log_std_min,
                          head.log_std_max, gates)
    return features @ mw + mb, log_std, named


def _torch_sample(mean, log_std, noise, gates=None):
    """The reparameterised sample and its log-density.

    The squash correction is spelled the way `ttrl.algos.gaussian` spells it,
    `2 * (log 2 - u - softplus(-2u))`, which is what
    `TanhTransform.log_abs_det_jacobian` computes as well. Spelling it out is
    what puts the softplus relu where the device put it, so that site takes
    the device's gate pattern instead of leaving one activation of the
    log-density to a float64 gate of its own.
    """

    pre_squash = mean + log_std.exp() * noise
    normal = torch.distributions.Normal(mean, log_std.exp())
    softplus = (torch_relu(-2.0 * pre_squash, gates)
                + torch.log1p(torch.exp(-torch.abs(2.0 * pre_squash))))
    correction = (2.0 * (np.log(2.0) - pre_squash - softplus)).sum(-1, keepdim=True)
    logp = normal.log_prob(pre_squash).sum(-1, keepdim=True) - correction
    return torch.tanh(pre_squash), logp


def _torch_critic(critic, obs, action, prefix, gates=None):
    features, named = torch_stack(
        critic.encoder.encoder[DEFAULT_LEAF], obs, gates=gates,
        prefix=f"{prefix}encoder.encoder.towers.{DEFAULT_LEAF}.layers.")
    joint = torch.cat([features, action], dim=-1)
    value, named = torch_stack(critic.head, joint, named=named, gates=gates,
                               prefix=f"{prefix}head.layers.")
    return value, named


def test_critic_loss_and_gradients_match_torch(agent, device):
    torch = pytest.importorskip("torch")

    data = _load(agent)
    alpha = agent.alpha()
    alpha64 = torch.tensor(float(alpha.numpy().reshape(-1)[0]), dtype=torch.float64)

    # The device forward runs FIRST, because the twin's relu gates are read
    # off its tape. Harvested before `backward`, which consumes the graph.
    got, device_target, _errors = agent._critic_loss(alpha)
    gates = DeviceGates(got)

    next64 = torch.tensor(data["next_obs"], dtype=torch.float64)
    obs64 = torch.tensor(data["obs"], dtype=torch.float64)
    mean, log_std, _ = _torch_policy(agent.nets.policy, next64)
    next_action, next_logp = _torch_sample(
        mean, log_std, torch.tensor(data["next_noise"], dtype=torch.float64))
    target_qs = [_torch_critic(net, next64, next_action, f"t{i}.")[0]
                 for i, net in enumerate(agent.nets.targets)]
    soft = torch.minimum(*target_qs).detach() - alpha64 * next_logp.detach()
    rewards = torch.tensor(data["rewards"], dtype=torch.float64)
    mask = torch.tensor(1.0 - data["dones"], dtype=torch.float64)
    discounts = torch.tensor(data["discounts"], dtype=torch.float64)
    reference_target = rewards + discounts * mask * soft

    # The target half is a constant of the step and the device computes it
    # under `no_grad`, so it leaves no tape and no gate pattern to borrow. It
    # is gated here as the value it is, against the float64 twin at
    # TARGET_CEILING, and then fed to the loss as that constant, which keeps
    # the gradient comparison below about the critic's own backward.
    target = torch.tensor(np.asarray(device_target.numpy(), np.float64),
                          dtype=torch.float64).reshape(reference_target.shape)
    drift = float(torch.linalg.norm(target - reference_target)
                  / torch.linalg.norm(reference_target))
    assert drift < TARGET_CEILING, f"soft target drift {drift}"

    action64 = torch.tensor(data["actions"], dtype=torch.float64)
    named: dict = {}
    want = torch.zeros((), dtype=torch.float64)
    for index, critic in enumerate(agent.nets.critics):
        value, params = _torch_critic(critic, obs64, action64, f"{index}.",
                                      gates=gates)
        named.update(params)
        want = want + ((value - target) ** 2).mean()
    want = 0.5 * want
    gates.assert_every_site_used()
    want.backward()

    backward(got)
    got_value = float(got.numpy().reshape(-1)[0])
    assert_close(got_value, float(want.detach()), tolerance=2e-2,
                            what="sac critic loss")
    report = assert_grads_match(
        [(name, param, named[name])
         for name, param in agent.nets.critics.named_parameters()])
    for name, (pcc, frobenius) in report.items():
        print(f"  {name}: pcc {pcc:.8f} frobenius {frobenius:.3e}")
    unmoved = [name for name, param in agent.nets.targets.named_parameters()
               if param.grad is not None]
    assert not unmoved, f"the critic loss put gradients on target parameters: {unmoved}"


def test_actor_and_temperature_losses_and_gradients_match_torch(agent, device):
    torch = pytest.importorskip("torch")

    data = _load(agent, seed=5)
    alpha = agent.alpha()
    alpha_value = float(alpha.numpy().reshape(-1)[0])
    alpha64 = torch.tensor(alpha_value, dtype=torch.float64)

    # Device first: the whole actor objective is on tape here, policy tower
    # and both critics, so every relu gate the twin needs is harvestable.
    got_actor, got_logp = agent._actor_loss(alpha)
    gates = DeviceGates(got_actor)

    obs64 = torch.tensor(data["obs"], dtype=torch.float64)
    mean, log_std, named = _torch_policy(agent.nets.policy, obs64, gates=gates)
    action, logp = _torch_sample(
        mean, log_std, torch.tensor(data["noise"], dtype=torch.float64),
        gates=gates)
    critic_qs = [_torch_critic(net, obs64, action, f"c{i}.", gates=gates)[0]
                 for i, net in enumerate(agent.nets.critics)]
    want_actor = (alpha64 * logp
                  - reduce(lambda a, b: torch_minimum(a, b, gates),
                           critic_qs)).mean()
    gates.assert_every_site_used()
    want_actor.backward(retain_graph=True)

    log_alpha = t64(agent.log_alpha.numpy())
    want_alpha = -(log_alpha * (logp.detach() + agent.target_entropy)).mean()
    want_alpha.backward()

    backward(got_actor)
    actor_value = float(got_actor.numpy().reshape(-1)[0])
    assert_close(actor_value, float(want_actor.detach()),
                             tolerance=2e-2, what="sac actor loss")
    report = assert_grads_match(
        [(name, param, named[name])
         for name, param in agent.nets.policy.named_parameters()])

    got_alpha_loss = agent._alpha_loss(got_logp)
    agent.alpha_opt.zero_grad()
    backward(got_alpha_loss)
    alpha_loss_value = float(got_alpha_loss.numpy().reshape(-1)[0])
    assert_close(alpha_loss_value, float(want_alpha.detach()),
                             tolerance=2e-2, what="sac temperature loss")
    assert_grads_match(
        [("log_alpha", agent.log_alpha, log_alpha)])

    for name, (pcc, frobenius) in report.items():
        print(f"  {name}: pcc {pcc:.8f} frobenius {frobenius:.3e}")
