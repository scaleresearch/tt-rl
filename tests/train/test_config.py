"""The config's rejection surface, and the two seams it now owns."""

from __future__ import annotations

import dataclasses


import pytest

from ttrl.algos.dqn.config import DQNConfig
from ttrl.algos.ppo.config import PPOConfig
from ttrl.algos.sac.config import SACConfig
from ttrl.train.config import (
    DeviceConfig,
    EnvConfig,
    TrainConfig,
    load_config,
)
from ttrl.train.evaluate import eval_metric_names

@pytest.mark.parametrize("kwargs,message", [
    ({"chip": 0, "chips": [0, 1]}, "scopes this process to one chip"),
    ({"chips": "al"}, "chips must be"),
    ({"chips": []}, "non-empty"),
    ({"chips": [0, 0]}, "repeat"),
    ({"chips": 0}, "chips count"),
    ({"chips": True}, "chips must be"),
])
def test_device_config_rejections(kwargs, message):
    with pytest.raises((ValueError, TypeError), match=message):
        DeviceConfig(**kwargs)


def test_an_algo_config_override_rebuilds_the_algorithms_own_type(tmp_path):
    path = tmp_path / "cfg.py"
    path.write_text(
        "from ttrl.train.config import TrainConfig\n"
        "from ttrl.algos.sac.config import SACConfig\n"
        "config = TrainConfig(algo='sac', total_frames=100, log_dir='runs/t',\n"
        "    algo_cfg=SACConfig(lr=1e-4, critic_hidden=(256, 128)))\n")
    cfg = load_config(str(path), overrides=["algo_cfg.lr=5e-5"])
    assert cfg.algo_cfg == SACConfig(lr=5e-5, critic_hidden=(256, 128))


def test_the_run_horizon_is_filled_from_the_run(tmp_path):
    """One source for the number a learning-rate schedule decays over."""

    cfg = TrainConfig(algo="ppo", total_frames=5_000, algo_cfg=PPOConfig())
    assert cfg.algo_cfg.total_frames == 5_000

    normalised = TrainConfig(
        algo="ppo", total_frames=5_000, algo_cfg=PPOConfig(),
        env=EnvConfig(name="atari", normalize_reward=True))
    assert normalised.env.normalize_reward_gamma == normalised.algo_cfg.gamma
    assert cfg.env.normalize_reward_gamma is None


def test_a_disagreeing_run_horizon_is_refused():
    with pytest.raises(ValueError, match="They are one quantity"):
        TrainConfig(algo="ppo", total_frames=5_000,
                    algo_cfg=PPOConfig(total_frames=99))
    with pytest.raises(ValueError, match="They are one quantity"):
        TrainConfig(algo="ppo", total_frames=5_000, algo_cfg=PPOConfig(),
                    env=EnvConfig(name="atari", normalize_reward=True,
                                  normalize_reward_gamma=0.5))
    with pytest.raises(ValueError, match="no discount is available"):
        TrainConfig(algo="dqn", total_frames=5_000, algo_cfg=None,
                    env=EnvConfig(name="atari", normalize_reward=True))


def test_the_horizon_survives_an_override(tmp_path):
    path = tmp_path / "cfg.py"
    path.write_text(
        "from ttrl.train.config import TrainConfig\n"
        "from ttrl.algos.ppo.config import PPOConfig\n"
        "config = TrainConfig(algo='ppo', total_frames=100, log_dir='runs/t',\n"
        "                     algo_cfg=PPOConfig())\n"
    )
    cfg = load_config(str(path), overrides=["total_frames=400"])
    assert isinstance(cfg.algo_cfg, PPOConfig)
    assert cfg.total_frames == 400
    assert cfg.algo_cfg.total_frames == 400


def test_an_override_moves_every_value_derived_from_it(tmp_path):
    """A derived value follows its source; a supplied one is not re-derived."""

    path = tmp_path / "cfg.py"
    path.write_text(
        "from ttrl.train.config import EnvConfig, TrainConfig\n"
        "from ttrl.algos.ppo.config import PPOConfig\n"
        "config = TrainConfig(algo='ppo', total_frames=100, log_dir='runs/t',\n"
        "    seed=0, env=EnvConfig(name='atari', normalize_reward=True),\n"
        "    eval_env=EnvConfig(name='atari'), algo_cfg=PPOConfig(gamma=0.99))\n"
    )
    cfg = load_config(str(path), overrides=["seed=7"])
    assert (cfg.env.seed, cfg.eval_env.seed) == (7, 7)

    cfg = load_config(str(path), overrides=["algo_cfg.gamma=0.95"])
    assert cfg.env.normalize_reward_gamma == 0.95

    cfg = load_config(str(path), overrides=["env.seed=3"])
    assert (cfg.env.seed, cfg.eval_env.seed) == (3, 0)

    supplied = tmp_path / "supplied.py"
    supplied.write_text(
        "from ttrl.train.config import TrainConfig\n"
        "from ttrl.algos.ppo.config import PPOConfig\n"
        "config = TrainConfig(algo='ppo', total_frames=100, log_dir='runs/t',\n"
        "                     algo_cfg=PPOConfig(total_frames=100))\n"
    )
    assert load_config(str(supplied), overrides=["seed=7"]).algo_cfg.total_frames == 100
    with pytest.raises(ValueError, match="They are one quantity"):
        load_config(str(supplied), overrides=["total_frames=400"])


def test_a_derived_horizon_follows_a_replaced_total_frames(tmp_path):
    """A variant of a shipped config, written the obvious way.

    The loaded config's algo horizon was derived, so re-deriving it is what
    the user asked for; a horizon the config states for itself is still one
    the run may not silently move.
    """

    derived = tmp_path / "derived.py"
    derived.write_text(
        "from ttrl.train.config import TrainConfig\n"
        "from ttrl.algos.ppo.config import PPOConfig\n"
        "config = TrainConfig(algo='ppo', total_frames=2_000_000,\n"
        "                     log_dir='runs/t', algo_cfg=PPOConfig())\n"
    )
    variant = dataclasses.replace(load_config(str(derived)), total_frames=1_000_000)
    assert (variant.total_frames, variant.algo_cfg.total_frames) == (10**6, 10**6)

    supplied = TrainConfig(algo="ppo", total_frames=2_000_000, log_dir="runs/t",
                           algo_cfg=PPOConfig(total_frames=2_000_000))
    with pytest.raises(ValueError, match="They are one quantity"):
        dataclasses.replace(supplied, total_frames=1_000_000)


def test_a_vector_env_wider_than_the_network_batch_is_refused_early():
    """Caught before a chip is opened, by the algorithm's own refusal."""

    with pytest.raises(ValueError, match="exceeds the network batch"):
        TrainConfig(algo="dqn", env=EnvConfig(n_envs=64),
                    algo_cfg=DQNConfig(batch_size=32))


def test_an_eval_env_wider_than_the_network_batch_is_refused_early():
    """The width the training env never asks for."""

    with pytest.raises(ValueError, match="min\\(eval_episodes, eval_env.n_envs\\)=16"):
        TrainConfig(algo="dqn", env=EnvConfig(n_envs=2),
                    eval_env=EnvConfig(n_envs=16), eval_episodes=30,
                    algo_cfg=DQNConfig(batch_size=8))


def test_a_wide_eval_env_asked_for_few_episodes_is_accepted():
    cfg = TrainConfig(algo="dqn", env=EnvConfig(n_envs=2),
                      eval_env=EnvConfig(n_envs=64), eval_episodes=2,
                      algo_cfg=DQNConfig(batch_size=8))
    assert min(cfg.eval_episodes, cfg.eval_env_config.n_envs) == 2


def test_a_constant_eval_field_is_not_a_best_metric():
    """`eval/n_episodes` never changes, so selecting on it freezes the first
    checkpoint. It stays in the eval result and out of the selectable names."""

    assert "eval/n_episodes" not in eval_metric_names()
    with pytest.raises(ValueError, match="eval/mean"):
        TrainConfig(algo="dqn", total_frames=100, log_dir="runs/t",
                    best_metric="eval/n_episodes")
