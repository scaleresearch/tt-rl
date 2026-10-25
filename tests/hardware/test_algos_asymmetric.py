"""Asymmetric actor-critic on real hardware: the critic sees privileged state
and the actor provably cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.algos.base import Transition
from ttrl.algos.ppo.config import PPOConfig
from ttrl.algos.sac.config import SACConfig
from ttrl.algos.dqn.agent import DQN
from ttrl.algos.dqn.config import DQNConfig
from ttrl.algos.ppo.agent import PPO
from ttrl.algos.sac.agent import SAC
from ttrl.engine import acquire
from ttrl.nets.build import build_actor_critic
from ttrl.nets.obs import ObsViews
from ttrl.spaces import (DEFAULT_LEAF, ActionSpec, ObsField, ObsGroup,
                         ObsSpec)

pytestmark = pytest.mark.tt

SENSED, PRIV = 6, 5
SPLIT = ObsGroup.of(ObsSpec.split_vector(
    ObsField("sensed", SENSED), ObsField("sim_state", PRIV, actor=False)))
PLAIN = ObsGroup.of(ObsSpec.vector(SENSED + PRIV))
CONTINUOUS = ActionSpec.continuous([-1.0, -1.0], [1.0, 1.0])
DISCRETE = ActionSpec.discrete(3)
LANES = 4


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


def _obs(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(LANES, SPLIT[DEFAULT_LEAF].size)).astype(np.float32)


def _repriv(obs: np.ndarray, value: float = 9.0) -> np.ndarray:
    """The same observation with every privileged entry replaced."""
    other = obs.copy()
    other[:, SENSED:] = value
    return other


# ---------------------------------------------------------------------------
# what the builders build
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# what reaches each network at run time
# ---------------------------------------------------------------------------

def test_the_actor_critic_value_moves_with_privileged_state_and_the_policy_does_not(device):
    """One forward pass, two privileged halves. The value head must notice and
    the policy head must not: a policy that notices is reading state the robot
    cannot sense."""
    net = build_actor_critic(SPLIT, CONTINUOUS, device, batch=LANES)
    views = ObsViews(SPLIT, device, batch=LANES)
    obs = _obs()

    (mean, log_std), value = net(views.load(obs))
    first = (mean.numpy().copy(), log_std.numpy().copy(), value.numpy().copy())
    (mean, log_std), value = net(views.load(_repriv(obs)))

    np.testing.assert_array_equal(mean.numpy(), first[0])
    np.testing.assert_array_equal(log_std.numpy(), first[1])
    assert not np.allclose(value.numpy(), first[2]), (
        "the critic's value is unchanged by replacing every privileged entry, "
        "so it is not reading them and this test cannot detect a leak"
    )


@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_the_agents_action_does_not_depend_on_privileged_state(algo, device):
    """Through the public surface: the same two observations handed to `act`."""
    if algo == "ppo":
        agent = PPO(SPLIT, CONTINUOUS, device,
                    cfg=PPOConfig(n_steps=8, minibatch_size=LANES,
                                  total_frames=1000),
                    n_envs=LANES, seed=0)
    else:
        agent = SAC(SPLIT, CONTINUOUS, device,
                    cfg=SACConfig(batch_size=LANES, learning_starts=8,
                                  buffer_capacity=256),
                    n_envs=LANES, seed=0)
    try:
        obs = _obs(2)
        first = agent.act(obs, deterministic=True)
        again = agent.act(_repriv(obs), deterministic=True)
        np.testing.assert_array_equal(first, again)
        # Not a constant policy: the sensed half still moves the action.
        sensed = obs.copy()
        sensed[:, :SENSED] *= -1.0
        assert not np.allclose(agent.act(sensed, deterministic=True), first)
    finally:
        agent.close()


# ---------------------------------------------------------------------------
# learning, and the algorithm that has nowhere to put the privileged half
# ---------------------------------------------------------------------------

def test_sac_learns_from_a_split_observation(device):
    """Both trunks get gradient, so the split costs nothing downstream."""

    agent = SAC(SPLIT, CONTINUOUS, device,
                cfg=SACConfig(batch_size=8, learning_starts=8,
                              buffer_capacity=256, train_freq=1),
                n_envs=LANES, seed=0)
    try:
        rng = np.random.default_rng(0)
        for step in range(8):
            obs = _obs(step)
            agent.act(obs)
            agent.observe(Transition(
                obs=SPLIT.observation(obs),
                action=rng.uniform(-1, 1, (LANES, CONTINUOUS.dim)).astype(np.float32),
                reward=rng.normal(size=LANES).astype(np.float32),
                terminated=np.zeros(LANES, bool), truncated=np.zeros(LANES, bool),
                next_obs=SPLIT.observation(_obs(step + 100))))
        assert agent.should_update(LANES * 8)
        before = [p.numpy().copy() for p in agent.nets.critics[0].parameters()]
        metrics = agent.update()
        assert np.isfinite(metrics["critic_loss"])
        assert any(not np.array_equal(a, b.numpy()) for a, b
                   in zip(before, agent.nets.critics[0].parameters()))
    finally:
        agent.close()


def test_dqn_refuses_a_split_observation_by_name(device):
    with pytest.raises(ValueError, match="sim_state"):
        DQN(SPLIT, DISCRETE, device, cfg=DQNConfig(batch_size=8), n_envs=LANES)
