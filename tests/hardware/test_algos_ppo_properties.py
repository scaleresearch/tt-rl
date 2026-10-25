"""Properties of the clipped surrogate that a loss value alone cannot show."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.algos.ppo import PPO, PPOConfig
from ttrl.buffers import RolloutMinibatch
from ttrl.engine import acquire
from ttrl.engine.autograd import no_grad
from ttrl.engine.tensor import to_host
from ttrl.spaces import DEFAULT_LEAF, ActionSpec, ObsGroup, ObsSpec

pytestmark = pytest.mark.tt

OBS = ObsGroup.of(ObsSpec.vector(8))
DISCRETE = ActionSpec.discrete(5)
CONTINUOUS = ActionSpec.continuous([-1.0] * 3, [1.0] * 3)
BATCH = 32


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


def _agent(device, action_spec, **over):
    cfg = PPOConfig(minibatch_size=BATCH, n_steps=BATCH, max_grad_norm=1e9, **over)
    return PPO(OBS, action_spec, device, cfg=cfg, n_envs=1, seed=4)


def _batch(action_spec, *, advantages, old_logp=None, seed=0) -> RolloutMinibatch:
    rng = np.random.default_rng(seed)
    actions = (rng.integers(0, action_spec.n, BATCH).astype(np.int64)
               if action_spec.kind == "discrete"
               else rng.normal(size=(BATCH, action_spec.dim)).astype(np.float32))
    return RolloutMinibatch(
        obs=OBS.observation(
            rng.normal(size=(BATCH, OBS[DEFAULT_LEAF].size)).astype(np.float32)),
        actions=actions,
        old_logp=(np.zeros(BATCH, np.float32) if old_logp is None
                  else np.asarray(old_logp, np.float32)),
        values=np.zeros(BATCH, np.float32),
        advantages=np.asarray(advantages, np.float32),
        returns=np.zeros(BATCH, np.float32),
    )


def _current_logp(agent, batch) -> np.ndarray:
    """The policy's log-prob of `batch.actions` right now, as acting records it."""

    with no_grad():
        policy_out, _value = agent._obs(batch.obs)
        logp, _entropy = agent.policy.logp_entropy(policy_out, batch.actions)
    return logp.numpy().reshape(-1)


def _max_abs_grad(module) -> float:
    return max(0.0 if param.grad is None
               else float(np.abs(to_host(param.grad)).max())
               for _name, param in module.named_parameters())


@pytest.mark.parametrize("action_spec", [DISCRETE, CONTINUOUS],
                         ids=["discrete", "continuous"])
def test_ratio_is_one_when_old_logp_is_the_current_policy(action_spec, device):
    agent = _agent(device, action_spec, entropy_coef=0.0, value_coef=0.0)
    try:
        probe = _batch(action_spec, advantages=np.full(BATCH, 2.0))
        batch = _batch(action_spec, advantages=np.full(BATCH, 2.0),
                       old_logp=_current_logp(agent, probe))
        got = agent._minibatch_step(batch)
        assert abs(got["policy_loss"] + 2.0) < 2e-3
        assert abs(got["approx_kl"]) < 1e-5
    finally:
        agent.close()


