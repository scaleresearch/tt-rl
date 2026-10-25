"""PPO's clipped surrogate, value clipping, entropy and gradients against a
float64 torch twin, on both action kinds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests._fakes.torch_twin import (DeviceGates, assert_close,
                                     assert_grads_match, t64, torch_clamp,
                                     torch_stack)
from ttrl.algos.ppo import PPO, PPOConfig
from ttrl.buffers import RolloutMinibatch
from ttrl.engine import acquire
from ttrl.spaces import DEFAULT_LEAF, ActionSpec, ObsGroup, ObsSpec

pytestmark = pytest.mark.tt

OBS = ObsGroup.of(ObsSpec.vector(17))
DISCRETE = ActionSpec.discrete(6)
CONTINUOUS = ActionSpec.continuous([-1.0] * 4, [1.0] * 4)
BATCH = 64


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


def _minibatch(action_spec, seed: int = 0) -> RolloutMinibatch:
    rng = np.random.default_rng(seed)
    if action_spec.kind == "discrete":
        actions = rng.integers(0, action_spec.n, BATCH).astype(np.int64)
    else:
        actions = rng.normal(size=(BATCH, action_spec.dim)).astype(np.float32)
    return RolloutMinibatch(
        obs=OBS.observation(
            rng.normal(size=(BATCH, OBS[DEFAULT_LEAF].size)).astype(np.float32)),
        actions=actions,
        # Deliberately spread so that a good share of the ratios land outside
        # the clip band; a batch whose ratios are all 1 cannot tell the clipped
        # surrogate from the unclipped one.
        old_logp=rng.normal(scale=0.5, size=BATCH).astype(np.float32) - 2.0,
        values=rng.normal(size=BATCH).astype(np.float32),
        advantages=rng.normal(size=BATCH).astype(np.float32),
        returns=rng.normal(size=BATCH).astype(np.float32),
    )


def _torch_heads(net, x, gates=None):
    """float64 twin of the shared trunk and both heads."""

    # One trunk in both slots: this spec leaves every field to the actor, so
    # the actor's encoder IS the critic's and its parameters are named once.
    features, named = torch_stack(
        net.encoders["actor"][DEFAULT_LEAF], x, gates=gates,
        prefix=f"encoders.actor.towers.{DEFAULT_LEAF}.layers.")
    value_head = net.heads["value"].value
    vw, vb = t64(value_head.weight.numpy()), t64(value_head.bias.numpy())
    named["heads.value.value.weight"] = vw
    named["heads.value.value.bias"] = vb
    value = features @ vw + vb
    policy = net.heads["policy"]
    if hasattr(policy, "logits"):
        pw, pb = t64(policy.logits.weight.numpy()), t64(policy.logits.bias.numpy())
        named["heads.policy.logits.weight"] = pw
        named["heads.policy.logits.bias"] = pb
        return (features @ pw + pb), value, named
    mw, mb = t64(policy.heads["mean"].weight.numpy()), t64(policy.heads["mean"].bias.numpy())
    sw, sb = (t64(policy.heads["log_std"].weight.numpy()),
              t64(policy.heads["log_std"].bias.numpy()))
    named["heads.policy.heads.mean.weight"] = mw
    named["heads.policy.heads.mean.bias"] = mb
    named["heads.policy.heads.log_std.weight"] = sw
    named["heads.policy.heads.log_std.bias"] = sb
    log_std = torch_clamp(features @ sw + sb, policy.log_std_min,
                          policy.log_std_max, gates)
    return (features @ mw + mb, log_std), value, named


def _torch_logp_entropy(policy_out, batch, action_spec):
    if action_spec.kind == "discrete":
        logp_all = torch.log_softmax(policy_out, dim=-1)
        taken = torch.tensor(batch.actions, dtype=torch.long)
        logp = logp_all[torch.arange(BATCH), taken]
        entropy = -(logp_all.exp() * logp_all).sum(dim=-1)
        return logp, entropy
    mean, log_std = policy_out
    normal = torch.distributions.Normal(mean, log_std.exp())
    taken = torch.tensor(batch.actions, dtype=torch.float64)
    return normal.log_prob(taken).sum(-1), normal.entropy().sum(-1)


def _torch_loss(net, batch, action_spec, cfg, gates=None):
    obs64 = torch.tensor(batch.obs[DEFAULT_LEAF], dtype=torch.float64)
    policy_out, value, named = _torch_heads(net, obs64, gates)
    new_logp, entropy = _torch_logp_entropy(policy_out, batch, action_spec)
    old_logp = torch.tensor(batch.old_logp, dtype=torch.float64)
    advantages = torch.tensor(batch.advantages, dtype=torch.float64)
    returns = torch.tensor(batch.returns, dtype=torch.float64)
    old_value = torch.tensor(batch.values, dtype=torch.float64)

    ratio = (new_logp - old_logp).exp()
    surrogate = torch.minimum(
        ratio * advantages,
        ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages)
    policy_loss = -surrogate.mean()
    value = value.reshape(-1)
    clipped = old_value + (value - old_value).clamp(
        -cfg.value_clip_eps, cfg.value_clip_eps)
    value_loss = 0.5 * torch.maximum((value - returns) ** 2,
                                     (clipped - returns) ** 2).mean()
    entropy_bonus = entropy.mean()
    total = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus
    return {
        "loss": total, "policy_loss": policy_loss, "value_loss": value_loss,
        "entropy": entropy_bonus, "ratio": ratio,
    }, named


@pytest.mark.parametrize("action_spec", [DISCRETE, CONTINUOUS],
                         ids=["discrete", "continuous"])
def test_ppo_losses_and_gradients_match_torch(action_spec, device):
    torch = pytest.importorskip("torch")

    cfg = PPOConfig(minibatch_size=BATCH, n_steps=BATCH, entropy_coef=0.01,
                    clip_eps=0.2, value_clip_eps=0.2, max_grad_norm=1e9)
    agent = PPO(OBS, action_spec, device, cfg=cfg, n_envs=1, seed=7)
    try:
        batch = _minibatch(action_spec)
        # The relu gate pattern is read off a forward of the same net on the
        # same observations, run before the step that will move the weights.
        # The arithmetic is deterministic, so these are the gates the step
        # itself computes; a step that computed different ones would miss the
        # tightened ceiling below by the size of a flipped gate.
        gates = DeviceGates(*agent._obs(batch.obs))
        want, named = _torch_loss(agent.net, batch, action_spec, cfg, gates)
        gates.assert_every_site_used()
        outside = float(((want["ratio"] < 1.0 - cfg.clip_eps)
                         | (want["ratio"] > 1.0 + cfg.clip_eps)).double().mean())
        assert outside > 0.2, (
            f"only {outside:.0%} of ratios are outside the clip band, so this "
            f"batch barely exercises the clipping"
        )
        want["loss"].backward()

        got = agent._minibatch_step(batch)
        report = assert_grads_match(
            [(name, param, named[name])
             for name, param in agent.net.named_parameters()])
        for key in ("loss", "policy_loss", "value_loss", "entropy"):
            assert_close(got[key], float(want[key].detach()),
                                    tolerance=2e-2, what=f"ppo {key}")
        for name, (pcc, frobenius) in report.items():
            print(f"  {name}: pcc {pcc:.8f} frobenius {frobenius:.3e}")
    finally:
        agent.close()
