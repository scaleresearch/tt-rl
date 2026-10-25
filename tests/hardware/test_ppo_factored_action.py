"""PPO over a composite (multi_discrete) action: the product-distribution maths.

The policy head emits one row of `sum(nvec)` logits, cut into one segment per
factor. What has to hold is that the thing sampled and the thing differentiated
are the same product distribution: the joint log-prob is the sum of the
per-factor log-probs, the joint entropy is the sum of the per-factor entropies,
and the log-prob recorded while acting is the one the update measures its ratio
against. Each of those is checked against a host recomputation from the head's
own output, so a wrong cut (one log_softmax over the concatenation, say) fails
here rather than training quietly.
"""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.algos.ppo import PPO, PPOConfig
from ttrl.buffers import RolloutMinibatch
from ttrl.engine import acquire
from ttrl.engine.autograd import no_grad
from ttrl.spaces import DEFAULT_LEAF, ActionSpec, ObsGroup, ObsSpec

pytestmark = pytest.mark.tt

OBS = ObsGroup.of(ObsSpec.vector(8))
NVEC = (3, 4, 2)
MULTI = ActionSpec.multi_discrete(NVEC)
BATCH = 32


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


def _agent(device, **over):
    cfg = PPOConfig(minibatch_size=BATCH, n_steps=BATCH, max_grad_norm=1e9, **over)
    return PPO(OBS, MULTI, device, cfg=cfg, n_envs=BATCH, seed=4)


def _obs_rows(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(BATCH, OBS[DEFAULT_LEAF].size)).astype(np.float32)


def _segments(logits: np.ndarray):
    """The head's row cut into one segment per factor, as log-probs."""
    out, offset = [], 0
    for size in NVEC:
        piece = logits[:, offset:offset + size].astype(np.float64)
        shifted = piece - piece.max(axis=-1, keepdims=True)
        out.append(shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True)))
        offset += size
    return out


def _head_logits(agent, obs_rows) -> np.ndarray:
    with no_grad():
        policy_out, _value = agent._obs(OBS.observation(obs_rows))
    return policy_out.numpy()[:BATCH]


def test_the_head_is_one_row_of_logits_per_factor_concatenated(device):
    agent = _agent(device)
    try:
        assert MULTI.width == sum(NVEC)
        assert _head_logits(agent, _obs_rows()).shape == (BATCH, sum(NVEC))
    finally:
        agent.close()


def test_acting_records_the_sum_of_the_per_factor_log_probs(device):
    """Also that each column is drawn from its own factor's range: a shared
    sampler over the concatenation would put an out-of-range index on the wire.
    """
    agent = _agent(device)
    try:
        obs_rows = _obs_rows(1)
        actions = agent.act(obs_rows)
        assert actions.shape == (BATCH, len(NVEC))
        assert actions.dtype == np.int64
        for index, size in enumerate(NVEC):
            assert actions[:, index].min() >= 0
            assert actions[:, index].max() < size

        logp_all = _segments(_head_logits(agent, obs_rows))
        rows = np.arange(BATCH)
        expected = sum(logp[rows, actions[:, i]] for i, logp in enumerate(logp_all))
        recorded = agent._pending["logp"]
        print(f"\nworst |recorded - sum of factors| = "
              f"{np.abs(recorded - expected).max():.3e}")
        assert np.abs(recorded - expected).max() < 2e-5
    finally:
        agent.close()


def test_the_update_measures_the_same_joint_and_a_product_entropy(device):
    agent = _agent(device)
    try:
        obs_rows = _obs_rows(2)
        rng = np.random.default_rng(3)
        actions = np.stack([rng.integers(0, size, BATCH) for size in NVEC],
                           axis=-1).astype(np.int64)
        with no_grad():
            policy_out, _value = agent._obs(OBS.observation(obs_rows))
            logp, entropy = agent.policy.logp_entropy(policy_out, actions)
        logp = logp.numpy().reshape(-1)[:BATCH]
        entropy = entropy.numpy().reshape(-1)[:BATCH]

        logp_all = _segments(_head_logits(agent, obs_rows))
        rows = np.arange(BATCH)
        expected_logp = sum(lp[rows, actions[:, i]] for i, lp in enumerate(logp_all))
        expected_entropy = sum(-(np.exp(lp) * lp).sum(axis=-1) for lp in logp_all)
        bound = float(np.log(NVEC).sum())
        print(f"\nlogp err {np.abs(logp - expected_logp).max():.3e}, entropy err "
              f"{np.abs(entropy - expected_entropy).max():.3e}, bound {bound:.4f}")
        assert np.abs(logp - expected_logp).max() < 2e-5
        assert np.abs(entropy - expected_entropy).max() < 2e-5
        assert entropy.max() <= bound + 1e-4
    finally:
        agent.close()


def test_the_ratio_is_one_at_the_log_prob_acting_recorded(device):
    """The end-to-end statement: one update step whose importance ratio is
    exactly 1, so the clipped surrogate is -mean(advantage)."""
    agent = _agent(device, entropy_coef=0.0, value_coef=0.0)
    try:
        obs_rows = _obs_rows(4)
        actions = agent.act(obs_rows)
        batch = RolloutMinibatch(
            obs=OBS.observation(obs_rows),
            actions=actions,
            old_logp=agent._pending["logp"],
            values=np.zeros(BATCH, np.float32),
            advantages=np.full(BATCH, 2.0, np.float32),
            returns=np.zeros(BATCH, np.float32),
        )
        got = agent._minibatch_step(batch)
        print(f"\npolicy loss {got['policy_loss']:.8f} against -2.0, approx_kl "
              f"{got['approx_kl']:.3e}")
        assert abs(got["policy_loss"] + 2.0) < 2e-3
        assert abs(got["approx_kl"]) < 1e-5
    finally:
        agent.close()
