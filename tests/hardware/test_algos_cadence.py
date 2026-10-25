"""`should_update` is where update cadence lives, so the trainer has no branch."""

from __future__ import annotations

import numpy as np
import pytest

from tests._fakes.toy_envs import REACH_GROUP, SIGN_GROUP, ReachEnv, SignEnv
from ttrl.algos.base import Transition
from ttrl.algos.dqn import DQN, DQNConfig
from ttrl.algos.ppo import PPO, PPOConfig
from ttrl.engine import acquire
from ttrl.spaces import DEFAULT_LEAF

pytestmark = pytest.mark.tt

LANES = 2


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


_RNG = {}


def _feed(agent, env_class, steps: int, *, lanes: int = LANES) -> None:
    """Hand the agent `steps` synthetic vector steps, no environment needed."""
    # The agent's own group: what it was built over, leaves and all.
    spec = agent.obs_spec
    action_spec = env_class.action_spec
    rng = _RNG.setdefault(id(agent), np.random.default_rng(0))
    for _ in range(steps):
        leaf = spec[DEFAULT_LEAF]
        obs = rng.normal(size=(lanes, *leaf.shape)).astype(leaf.numpy_dtype)
        if action_spec.kind == "discrete":
            action = rng.integers(0, action_spec.n, lanes)
        else:
            action = rng.uniform(-1, 1, (lanes, action_spec.dim)).astype(np.float32)
        agent.act(obs)
        nxt = rng.normal(size=(lanes, *leaf.shape)).astype(leaf.numpy_dtype)
        agent.observe(Transition(
            obs=spec.observation(obs), action=action,
            reward=rng.normal(size=lanes).astype(np.float32),
            terminated=np.zeros(lanes, bool), truncated=np.zeros(lanes, bool),
            next_obs=spec.observation(nxt),
        ))


def _off_policy(algo, device, *, n_envs, train_freq, learning_starts=64):
    cfg = DQNConfig(batch_size=16, learning_starts=learning_starts,
                    train_freq=train_freq, buffer_capacity=4000, n_step=1)
    return DQN(SIGN_GROUP, SignEnv.action_spec, device, cfg=cfg,
               n_envs=n_envs, seed=0), SignEnv


def test_off_policy_takes_nothing_before_the_warmup(device):
    agent, env_class = _off_policy("dqn", device, n_envs=LANES, train_freq=4)
    warmup, freq = agent.cfg.learning_starts, agent.cfg.train_freq
    try:
        _feed(agent, env_class, 100, lanes=LANES)
        assert agent.buffer.num_sampleable >= agent.cfg.batch_size
        assert not agent.should_update(0)
        # Credit accrues only past the warm-up, so the run does not owe
        # warmup / train_freq steps the moment the warm-up ends.
        assert not agent.should_update(warmup)
        assert agent.should_update(warmup + freq)
        assert agent._clock.owed == 1
    finally:
        agent.close()


def test_off_policy_also_waits_for_a_sampleable_batch(device):
    """learning_starts alone is not enough: with several lanes the buffer can
    hold the frames and still have no complete n-step window to draw."""
    agent, env_class = _off_policy("dqn", device, n_envs=LANES, train_freq=4)
    try:
        # Well past the warm-up, but the buffer is empty.
        assert not agent.should_update(1000)
        _feed(agent, env_class, 40, lanes=LANES)
        assert agent.should_update(1000 + agent.cfg.train_freq)
        # Those 1000 frames owe nothing: they were not frames whose gradient
        # step was missed, so the backlog is one step rather than 242.
        assert agent._clock.owed == 1
    finally:
        agent.close()


@pytest.mark.parametrize("n_envs", [1, 3])
def test_realised_replay_ratio_matches_the_configured_one(n_envs, device):
    """One gradient step per `train_freq` frames, whatever a vector step
    advances.
    """
    train_freq, warmup, frames = 4, 32, 480
    agent, env_class = _off_policy("dqn", device, n_envs=n_envs,
                                   train_freq=train_freq, learning_starts=warmup)
    try:
        steps = 0
        frame = 0
        while frame < frames:
            _feed(agent, env_class, 1, lanes=n_envs)
            frame += n_envs
            if agent.should_update(frame):
                steps += float(agent.update()["gradient_steps"])
        want = (frames - warmup) / train_freq
        # Within one step: the last partial credit is carried, not taken.
        assert abs(steps - want) <= 1.0
        assert agent.update_count == steps
    finally:
        agent.close()


def test_on_policy_cadence_is_the_rollout_filling(device):
    cfg = PPOConfig(n_steps=8, minibatch_size=8, epochs=2)
    agent = PPO(REACH_GROUP, ReachEnv.action_spec, device, cfg=cfg,
                n_envs=LANES, seed=0)
    try:
        answers = []
        for step in range(cfg.n_steps):
            answers.append(agent.should_update(step * LANES))
            _feed(agent, ReachEnv, 1)
        answers.append(agent.should_update(cfg.n_steps * LANES))
        assert answers == [False] * cfg.n_steps + [True], answers
        # The frame number is irrelevant to an on-policy trigger, and the
        # rollout resets after the update, so the answer goes back to False.
        assert agent.should_update(0)
        agent.update()
        assert not agent.should_update(10 ** 9)
    finally:
        agent.close()


def test_ppo_refuses_a_transition_no_act_produced(device):
    """The log-prob an importance ratio is measured against comes from the
    forward pass that chose the action, so act and observe are a pair."""
    agent = PPO(REACH_GROUP, ReachEnv.action_spec, device,
                cfg=PPOConfig(n_steps=4, minibatch_size=4), n_envs=LANES, seed=0)
    try:
        obs = np.zeros((LANES, 3), np.float32)
        group = agent.obs_spec
        transition = Transition(
            obs=group.observation(obs), action=np.zeros((LANES, 2), np.float32),
            reward=np.zeros(LANES, np.float32),
            terminated=np.zeros(LANES, bool), truncated=np.zeros(LANES, bool),
            next_obs=group.observation(obs))
        with pytest.raises(RuntimeError, match="no act"):
            agent.observe(transition)
        agent.act(obs, deterministic=True)
        with pytest.raises(RuntimeError, match="no act"):
            agent.observe(transition)
        agent.act(obs)
        agent.observe(transition)
    finally:
        agent.close()
