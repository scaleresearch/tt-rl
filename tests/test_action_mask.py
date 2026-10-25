"""The legal-action mask: where it lives, what reads it, and what it refuses.

No hardware. Everything asserted here is reachable without a chip, because
every failure mode worth naming is a spec or a host-side selection error: an
illegal action that reaches a device step has already been chosen.
"""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.algos.dqn import DQN, DQNConfig
from ttrl.algos.mask import PENALTY, epsilon_greedy_masked, mask_width, obs_mask
from ttrl.algos.sac import SAC
from ttrl.buffers import ReplayBuffer, RolloutBuffer
from ttrl.spaces import (ActionSpec, ObsGroup, ObsSpec, check_legal,
                         group_mask_rows, mask_rows)

MASKED = ObsSpec.vector(6, action_mask=3)
PLAIN = ObsSpec.vector(6)
MASKED_GROUP = ObsGroup.of(MASKED)
PLAIN_GROUP = ObsGroup.of(PLAIN)
DISCRETE = ActionSpec.discrete(3)
CONTINUOUS = ActionSpec.continuous([-1.0], [1.0])
MULTI_DISCRETE = ActionSpec.multi_discrete([3, 4])


def row(features, legal):
    return np.concatenate([np.asarray(features, np.float32),
                           np.asarray(legal, np.float32)])


# -- where the mask lives ---------------------------------------------------

def test_a_plain_spec_declares_no_mask_and_costs_nothing():
    assert PLAIN.action_mask == 0
    assert mask_width(PLAIN, DISCRETE, "dqn") == 0


def test_the_mask_is_the_tail_of_the_observation_vector():
    obs = np.stack([row([1, 0, 0], [1, 1, 0]), row([0, 1, 0], [0, 1, 1])])
    assert np.array_equal(mask_rows(obs, MASKED),
                          np.array([[1, 1, 0], [0, 1, 1]], np.float32))


def test_a_spec_with_no_mask_refuses_to_have_one_read_off_it():
    with pytest.raises(ValueError, match="declares no legal-action mask"):
        mask_rows(np.zeros((2, 6), np.float32), PLAIN)


def test_an_image_observation_cannot_carry_a_mask():
    with pytest.raises(ValueError, match="needs a vector observation"):
        ObsSpec(shape=(4, 84, 84), dtype="uint8", kind="image", action_mask=3)


def test_a_mask_may_not_consume_the_whole_observation():
    with pytest.raises(ValueError, match="leaves no features"):
        ObsSpec.vector(3, action_mask=3)


def test_mask_entries_must_be_indicators():
    obs = np.stack([row([1, 0, 0], [1.0, 0.5, 0.0])])
    with pytest.raises(ValueError, match="must each be 0.0 or 1.0"):
        mask_rows(obs, MASKED)


# -- the one switch ---------------------------------------------------------

def test_the_mask_width_must_match_the_action_count():
    with pytest.raises(ValueError, match="build the ObsSpec with action_mask=4"):
        mask_width(MASKED_GROUP, ActionSpec.discrete(4), "dqn")


def test_a_continuous_action_spec_refuses_a_mask_and_names_why():
    with pytest.raises(ValueError) as caught:
        mask_width(MASKED_GROUP, CONTINUOUS, "ppo")
    message = str(caught.value)
    assert "no members to name" in message
    assert "low and high bounds" in message


def test_a_composite_action_refuses_a_mask_and_names_the_enumeration():
    """A mask over the FACTORS cannot name an illegal combination of legal
    values, which is the normal case (a bet size legal only for a raise), so
    the only honest mask over a factorized action is over its joint actions."""
    with pytest.raises(ValueError) as caught:
        mask_width(ObsGroup.of(ObsSpec.vector(16, action_mask=12)),
                   MULTI_DISCRETE, "ppo")
    message = str(caught.value)
    assert "multi_discrete" in message
    assert "ActionSpec.discrete(12)" in message and "action_mask=12" in message


def test_sac_refuses_a_masked_obs_spec():
    with pytest.raises(ValueError, match="no members to name"):
        SAC(MASKED_GROUP, CONTINUOUS, device=None)


def test_dqn_reads_the_switch_once_and_both_halves_hang_off_it():
    # No device, so construction stops at the first build. The refusal of a
    # mismatched mask must come before that, which is what makes it the one
    # place the two specs are compared.
    with pytest.raises(ValueError, match="action_mask=3"):
        DQN(ObsGroup.of(ObsSpec.vector(6, action_mask=2)), DISCRETE, device=None,
            cfg=DQNConfig())


# -- selection --------------------------------------------------------------

def test_greedy_selection_ignores_the_illegal_columns():
    q = np.array([[9.0, 1.0, 2.0], [0.0, 5.0, 7.0]], np.float32)
    mask = np.array([[0, 1, 1], [1, 1, 0]], np.float32)
    rng = np.random.default_rng(0)
    assert np.array_equal(epsilon_greedy_masked(q, mask, 0.0, rng),
                          np.array([2, 1]))


def test_exploration_only_ever_draws_a_legal_action():
    mask = np.array([[1, 0, 1]] * 64, np.float32)
    q = np.zeros((64, 3), np.float32)
    rng = np.random.default_rng(1)
    drawn = set()
    for _ in range(50):
        actions = epsilon_greedy_masked(q, mask, 1.0, rng)
        drawn.update(actions.tolist())
    assert drawn == {0, 2}


def test_a_state_with_no_legal_action_is_refused_for_every_algorithm():
    # One site, so a value-based and a policy-based agent give the same
    # message. An argmax over uniformly penalised Q values returns column 0
    # and a softmax over uniformly penalised logits is uniform over illegal
    # actions, and neither failure names the real mistake.
    obs = np.concatenate([np.ones(3, np.float32), np.zeros(3, np.float32)])
    with pytest.raises(ValueError, match="declares every action illegal"):
        obs_mask(obs[None], MASKED_GROUP, 3)


def test_an_empty_successor_mask_is_allowed_because_a_terminal_state_has_no_move():
    # The bootstrap reads the successor's mask and zeroes its value at a
    # terminal anyway, so refusing an empty row there would refuse every
    # environment whose terminal state offers nothing.
    buf = ReplayBuffer(MASKED_GROUP, DISCRETE, capacity=64, n_step=1, seed=0)
    for i in range(40):
        end = i % 5 == 4
        buf.add(row([i, 0, 0], [1, 1, 0]), 1, 0.5, end,
                final_obs=row([i, 0, 0], [0, 0, 0]) if end else None)
    batch = buf.sample(32)
    assert batch.next_mask.shape == (32, 3)
    assert np.array_equal(batch.next_mask, group_mask_rows(batch.next_obs, MASKED_GROUP))


def test_the_penalty_is_finite_so_the_entropy_term_is_zero_not_nan():
    # 0 * -inf is NaN; 0 * -PENALTY is 0. The entropy bonus multiplies the
    # probability of an illegal action by its log-probability, so this is the
    # whole reason the penalty is a large number and not an infinite one.
    assert np.isfinite(PENALTY)
    logp = -PENALTY
    assert float(np.exp(logp)) == 0.0
    assert float(np.exp(logp) * logp) == 0.0


# -- the buffers ------------------------------------------------------------

def test_replay_serves_the_successor_state_mask():
    buf = ReplayBuffer(MASKED_GROUP, DISCRETE, capacity=64, n_step=1, seed=0)
    for i in range(40):
        legal = [1, 1, 0] if i % 2 else [0, 1, 1]
        buf.add(row([i, 0, 0], legal), 1, 0.5, False)
    batch = buf.sample(16)
    assert batch.next_mask.shape == (16, 3)
    assert np.array_equal(batch.next_mask, group_mask_rows(batch.next_obs, MASKED_GROUP))


def test_replay_serves_no_mask_for_an_unmasked_spec():
    buf = ReplayBuffer(PLAIN_GROUP, DISCRETE, capacity=64, n_step=1, seed=0)
    for i in range(40):
        buf.add(np.full(6, float(i), np.float32), 1, 0.5, False)
    assert buf.sample(16).next_mask is None


def test_rollout_serves_the_acting_state_mask():
    buf = RolloutBuffer(MASKED_GROUP, DISCRETE, n_envs=2, n_steps=8, seed=0)
    for _ in range(8):
        obs = np.stack([row([1, 0, 0], [1, 1, 0]), row([0, 1, 0], [0, 1, 1])])
        buf.add(obs, np.array([1, 1]), np.zeros(2, np.float32),
                np.zeros(2, np.float32), np.zeros(2, np.float32),
                np.zeros(2, np.float32))
    buf.compute_gae(0.99, 0.95, np.zeros(2, np.float32))
    batch = next(buf.minibatches(8, epochs=1))
    assert np.array_equal(batch.mask, group_mask_rows(batch.obs, MASKED_GROUP))


def test_rollout_serves_no_mask_for_an_unmasked_spec():
    buf = RolloutBuffer(PLAIN_GROUP, DISCRETE, n_envs=2, n_steps=8, seed=0)
    for _ in range(8):
        buf.add(np.zeros((2, 6), np.float32), np.array([1, 1]),
                np.zeros(2, np.float32), np.zeros(2, np.float32),
                np.zeros(2, np.float32), np.zeros(2, np.float32))
    buf.compute_gae(0.99, 0.95, np.zeros(2, np.float32))
    assert next(buf.minibatches(8, epochs=1)).mask is None


# -- coercion ---------------------------------------------------------------

def test_an_illegal_action_never_reaches_replay():
    buf = ReplayBuffer(MASKED_GROUP, DISCRETE, capacity=64, n_step=1, seed=0)
    with pytest.raises(ValueError, match="illegal under the legal-action mask"):
        buf.add(row([1, 0, 0], [1, 1, 0]), 2, 0.0, False)


def test_an_illegal_action_never_reaches_a_rollout():
    buf = RolloutBuffer(MASKED_GROUP, DISCRETE, n_envs=1, n_steps=4, seed=0)
    with pytest.raises(ValueError, match="illegal under the legal-action mask"):
        buf.add(row([1, 0, 0], [0, 1, 1])[None], np.array([0]),
                np.zeros(1, np.float32), np.zeros(1, np.float32),
                np.zeros(1, np.float32), np.zeros(1, np.float32))


def test_the_mask_family_takes_either_form_of_declared_spec():
    """`Env.obs_spec` is often the bare `ObsSpec`, and the group is what the
    vector layer promotes it to. Both read the same mask here, so a caller
    passing the one it holds does not land inside numpy."""
    obs = np.stack([row([1, 0, 0], [1, 1, 0])])
    assert np.array_equal(group_mask_rows(obs, MASKED),
                          group_mask_rows(obs, MASKED_GROUP))
    check_legal(obs, [1], MASKED, "where")


def test_the_coercion_refusal_names_the_legal_actions():
    with pytest.raises(ValueError) as caught:
        check_legal(row([1, 0, 0], [0, 1, 1]), [0], MASKED_GROUP, "where")
    message = str(caught.value)
    assert "legal actions: [1, 2]" in message
    assert "act()" in message
