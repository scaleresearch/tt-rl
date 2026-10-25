"""What PPO's recurrent update must exclude, and which flag it must pass on."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ttrl.algos.base import Transition
from ttrl.algos.ppo import PPO, PPOConfig
from ttrl.algos.ppo.config import RecurrentConfig
from ttrl.algos.ppo.loss import objective
from ttrl.engine import acquire
from ttrl.engine.ops import mean_all
from ttrl.nets.recurrent import ACTOR
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec

pytestmark = pytest.mark.tt

SEQ_LEN = 4
N_ENVS = 3
HIDDEN = 7
OBS_DIM = 5
N_ACTIONS = 6
#: Inside a sequence, so the padding slot it creates is not on a sequence edge.
BOUNDARY_AT = 2

GROUP = ObsGroup.of(ObsSpec.vector(OBS_DIM))
DISCRETE = ActionSpec.discrete(N_ACTIONS)


@pytest.fixture(scope="module")
def dev():
    with acquire(0) as (handle, _profile):
        yield handle


def _cfg() -> PPOConfig:
    return PPOConfig(
        n_steps=SEQ_LEN * 3, minibatch_size=SEQ_LEN * 2, epochs=1,
        recurrent=RecurrentConfig(hidden_size=HIDDEN, seq_len=SEQ_LEN))


def _collect(agent, cfg, *, truncate_at=(), terminate_at=()):
    """A whole rollout, with the named steps ending lane 0's episode."""
    rng = np.random.default_rng(31)
    for t in range(cfg.n_steps):
        obs = {"obs": rng.normal(size=(N_ENVS, OBS_DIM)).astype(np.float32)}
        agent.act(obs)
        terminated = np.zeros(N_ENVS, bool)
        truncated = np.zeros(N_ENVS, bool)
        if t in terminate_at:
            terminated[0] = True
        if t in truncate_at:
            truncated[0] = True
        agent.observe(Transition(
            obs=GROUP.observation(obs),
            action=rng.integers(0, N_ACTIONS, N_ENVS),
            reward=rng.normal(size=N_ENVS).astype(np.float32),
            terminated=terminated, truncated=truncated,
            next_obs=GROUP.observation(obs)))


def test_a_termination_and_a_truncation_both_reset_the_state(dev):
    """Both kinds of boundary, so the test cannot pass by treating every step
    as a start."""
    cfg = _cfg()
    agent = PPO(GROUP, DISCRETE, dev, cfg=cfg, n_envs=N_ENVS, seed=0)
    try:
        _collect(agent, cfg, truncate_at=(1,), terminate_at=(5,))
        assert agent.buffer.dones[1, 0] == 0.0, (
            "a truncation was recorded as a termination, so the value function "
            "would stop bootstrapping through a time limit")
        assert float(np.abs(agent.buffer.starts[:, 1:]).max()) == 0.0, (
            "a lane that ended no episode was marked as starting one")
        starts = agent.buffer.starts[:, 0]
        expected = np.zeros(cfg.n_steps, dtype=np.float32)
        expected[[2, 6]] = 1.0
        assert np.array_equal(starts, expected), (
            f"lane 0's starts are {starts.tolist()}, expected {expected.tolist()}")
    finally:
        agent.close()


def _one_batch_with_padding(agent, cfg):
    agent.buffer.compute_gae(cfg.gamma, cfg.lam, agent._last_values())
    for batch in agent.buffer.sequence_minibatches(
            cfg.minibatch_size, epochs=1, normalize_adv=False):
        if float(batch.valid.min()) == 0.0:
            return batch
    raise AssertionError(
        "no sequence minibatch held an autoreset padding slot, so the mask "
        "below would have nothing to exclude")


def test_the_loss_ignores_what_an_autoreset_padding_slot_holds(dev):
    """The reduction, measured rather than read."""
    cfg = _cfg()
    agent = PPO(GROUP, DISCRETE, dev, cfg=cfg, n_envs=N_ENVS, seed=0)
    try:
        _collect(agent, cfg, terminate_at=(BOUNDARY_AT,))
        batch = _one_batch_with_padding(agent, cfg)
        padding = batch.valid == 0.0

        def loss_of(sample) -> float:
            policy_out, value, reduce, sample = agent._recurrent.forward(
                agent._obs, sample, dev)
            loss, _parts, _logp = objective(
                agent.policy, policy_out, value, sample, cfg, dev,
                reduce=reduce)
            return float(loss.numpy().reshape(-1)[0])

        first = loss_of(batch)
        poisoned = dataclasses.replace(
            batch,
            advantages=np.where(padding, 1e3, batch.advantages).astype(np.float32),
            returns=np.where(padding, 1e3, batch.returns).astype(np.float32),
            old_logp=np.where(padding, -1e2, batch.old_logp).astype(np.float32),
        )
        second = loss_of(poisoned)
        # Finite as well as equal: an excluded slot holding a huge log-prob
        # would otherwise reach the reduction as `0.0 * inf`, which is NaN,
        # and a NaN loss is not an ignored slot.
        assert np.isfinite(first) and np.isfinite(second), (
            f"losses {first} and {second}; a masked-out slot took the whole "
            f"reduction with it")
        assert second == pytest.approx(first, abs=1e-4), (
            f"the loss moved from {first} to {second} when only invalid slots "
            f"changed, so the reduction is averaging over autoreset padding")

        # An absolute check, because every comparison in this test is between
        # two reductions of the same kind and a reduction that ignored `valid`
        # entirely would scale both sides alike. Reducing this batch with
        # `mean_all`, which is what the feedforward path uses because its rows
        # are all valid, must give a DIFFERENT number: a padding row's squared
        # value error and its entropy are both real numbers that the masked
        # reduction leaves out.
        policy_out, value, _reduce, sanitized = agent._recurrent.forward(
            agent._obs, batch, dev)
        unmasked, _parts, _logp = objective(
            agent.policy, policy_out, value, sanitized, cfg, dev,
            reduce=mean_all)
        assert not np.isclose(float(unmasked.numpy().reshape(-1)[0]), first,
                              atol=1e-4), (
            "reducing over every row gives the same loss as reducing over the "
            "valid ones, so the mask is not being applied")

        # The denominator tracks `valid` too. Marking the padding slots valid,
        # with the same numbers in them, must move the loss: it adds rows to
        # the mean and lets a fabricated step's entropy into the bonus. Without
        # this, a reduction that divided by the row count instead of the valid
        # count would satisfy every assertion above, because both sides of the
        # comparison would be scaled by the same wrong denominator.
        all_valid = loss_of(dataclasses.replace(
            batch, valid=np.ones_like(batch.valid)))
        assert not np.isclose(all_valid, first, atol=1e-4), (
            f"calling the padding slots valid left the loss at {first}, so the "
            f"reduction is not dividing by the number of valid rows")

        # And the mask is not vacuous: the same poisoning of a VALID slot must
        # move the loss, or the comparison above would hold for any reduction.
        valid_row = int(np.flatnonzero(batch.valid > 0.0)[0])
        shifted = batch.advantages.copy()
        shifted[valid_row] = 1e3
        moved = loss_of(dataclasses.replace(batch, advantages=shifted))
        assert not np.isclose(moved, first, atol=1e-4), (
            "poisoning a valid slot did not move the loss either, so this test "
            "is measuring nothing")
    finally:
        agent.close()


def test_every_slot_of_a_sequence_still_runs_through_the_memory(dev):
    """A padding slot is excluded from the loss and NOT from the recurrence."""
    cfg = _cfg()
    agent = PPO(GROUP, DISCRETE, dev, cfg=cfg, n_envs=N_ENVS, seed=0)
    try:
        _collect(agent, cfg, terminate_at=(BOUNDARY_AT,))
        batch = _one_batch_with_padding(agent, cfg)
        assert len(batch) == SEQ_LEN * (cfg.minibatch_size // SEQ_LEN), (
            "the minibatch is not a whole number of full-length sequences, so "
            "a padding slot was dropped and the unrolled T is no longer fixed")
        flat_starts = batch.starts.reshape(-1)
        for row in np.flatnonzero(batch.valid == 0.0):
            assert flat_starts[row] == 1.0, (
                f"padding row {row} does not begin an episode, so the state "
                f"entering it is the previous episode's")
        assert float(np.abs(batch.h0[ACTOR]).max()) > 0.0
    finally:
        agent.close()


def test_the_bootstrap_reads_the_ending_episodes_own_memory(dev):
    """The value after a truncating last step is valued from the state that
    episode built, not from a blank one.
    """
    cfg = _cfg()
    agent = PPO(GROUP, DISCRETE, dev, cfg=cfg, n_envs=N_ENVS, seed=0)
    try:
        _collect(agent, cfg, truncate_at=(cfg.n_steps - 1,))
        carried = agent._recurrent.carries(N_ENVS, deterministic=False)
        state = carried[ACTOR].numpy().copy()
        assert float(np.abs(state).max()) > 0.0, (
            "the carry is zero at the end of the rollout, so a reset here "
            "would be indistinguishable from no reset")

        observation = GROUP.observation(agent._last_next_obs)
        with_memory = agent._recurrent.bootstrap(
            agent._obs, observation, lanes=N_ENVS)
        kept = with_memory.numpy().reshape(-1)[:N_ENVS].copy()

        # The same call with the reset applied, which is what the flags of the
        # last step would have done.
        for carry in carried.values():
            carry.load_start(np.ones(N_ENVS, dtype=np.float32))
        _p, reset_value, _h = agent._obs.step(
            observation, {ACTOR: carried[ACTOR].prev},
            {ACTOR: carried[ACTOR].start})
        reset = reset_value.numpy().reshape(-1)[:N_ENVS]

        assert not np.allclose(kept[0], reset[0], atol=1e-5), (
            "the bootstrap value is the same with and without the reset, so "
            "this test cannot tell which one was used")
        assert np.allclose(agent._last_values()[0], kept[0], atol=1e-5), (
            "the rollout's bootstrap value was computed from a reset state, so "
            "a truncated episode's last advantage uses a blank memory")
    finally:
        agent.close()
