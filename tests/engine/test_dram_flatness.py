"""Device-DRAM flatness across many training updates."""

import numpy as np
import pytest

from tests._memory import assert_no_growth, measure_dram
from ttrl.algos.base import Transition
from ttrl.algos.dqn import DQN, DQNConfig
from ttrl.algos.ppo import PPO, PPOConfig
from ttrl.engine import acquire
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec

MB = 1_000_000


def test_monotonic_leak_is_rejected():
    """The PPO signature: ~485 MB per update, straight up until it OOMs."""
    series = [int((626 + 485 * i) * MB) for i in range(9)]
    with pytest.raises(AssertionError, match="grows across the run"):
        assert_no_growth(series)


def test_sawtooth_leak_is_rejected():
    """The DQN signature, and the reason a two-sample check is not enough."""
    series = []
    level = 300 * MB
    for i in range(60):
        level = 300 * MB if i % 20 == 19 else level + 30 * MB
        series.append(int(level))
    assert series[-1] < series[0], "fixture must defeat a two-sample check"
    with pytest.raises(AssertionError, match="grows across the run"):
        assert_no_growth(series)


#
# One chip, scoped by the caller:
#   TT_VISIBLE_DEVICES=0 pytest tests/engine/test_dram_flatness.py -m tt

ttnn = pytest.importorskip("ttnn")


def _specs():
    return ObsGroup.of(ObsSpec.image(4, 68, 68)), ActionSpec.discrete(6)


def _fill(agent, spec, actions, lanes, steps, rng):
    """Act then observe, the order the Agent contract requires."""

    for _ in range(steps):
        obs = rng.integers(0, 256, (lanes, *spec["obs"].shape), dtype=np.uint8)
        action = agent.act(obs)
        nxt = rng.integers(0, 256, (lanes, *spec["obs"].shape), dtype=np.uint8)
        agent.observe(Transition(
            obs=spec.observation(obs), action=action,
            reward=rng.standard_normal(lanes).astype(np.float32),
            terminated=np.zeros(lanes, bool),
            truncated=np.zeros(lanes, bool),
            next_obs=spec.observation(nxt),
        ))


@pytest.mark.tt
def test_dqn_update_does_not_grow_device_memory():
    """300 DQN updates, measured against the leak rather than a round number."""

    spec, actions = _specs()
    cfg = DQNConfig(batch_size=16, learning_starts=16, train_freq=1, n_step=1,
                    buffer_capacity=512, lr=1e-4)
    rng = np.random.default_rng(0)
    with acquire(0) as (dev, _profile):
        agent = DQN(spec, actions, dev, cfg=cfg, n_envs=2, seed=0)
        try:
            _fill(agent, spec, actions, 2, 40, rng)
            series = measure_dram(dev, lambda _i: agent.update(), steps=300, warmup=10)
        finally:
            agent.close()
    assert_no_growth(series, label="DQN update DRAM")


@pytest.mark.tt
def test_ppo_update_does_not_grow_device_memory():
    """The PPO signature."""

    spec, actions = _specs()
    cfg = PPOConfig(n_steps=8, minibatch_size=16, epochs=1, lr=1e-4)
    rng = np.random.default_rng(0)
    with acquire(0) as (dev, _profile):
        agent = PPO(spec, actions, dev, cfg=cfg, n_envs=2, seed=0)

        def step(_i):
            _fill(agent, spec, actions, 2, cfg.n_steps, rng)
            return agent.update()

        try:
            series = measure_dram(dev, step, steps=100, warmup=10)
        finally:
            agent.close()
    assert_no_growth(series, label="PPO update DRAM")


@pytest.mark.tt
def test_forward_only_act_does_not_grow_device_memory():
    """`act` builds a tape it never calls `backward()` on."""

    spec, actions = _specs()
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 256, (2, *spec["obs"].shape), dtype=np.uint8)
    with acquire(0) as (dev, _profile):
        agent = PPO(spec, actions, dev, cfg=PPOConfig(n_steps=8, minibatch_size=16),
                    n_envs=2, seed=0)
        try:
            series = measure_dram(dev, lambda _i: agent.act(obs), steps=200, warmup=10)
        finally:
            agent.close()
    assert_no_growth(series, label="PPO act DRAM")
