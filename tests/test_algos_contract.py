"""What an algorithm refuses about its own config and its own action space.

Every assertion here is reachable without a chip: a wrong `ActionSpec` kind is
refused before any network is built, which is the point of refusing it at all.
Each case names one of the three shipped algorithms, so the table is keyed by
the registry's own names and a fourth algorithm fails this file until it is
given one.

The registry and `check_agent`, which are about an agent arriving from
outside this repo rather than about a shipped one's own settings, live in
`test_algos_extension.py`. The `Transition` and autoreset contract lives in
`test_algos_transition.py`.
"""

from __future__ import annotations

import typing

import pytest

from ttrl.algos.checkpoint import describe_config_diff
from ttrl.algos.dqn import DQN, DQNConfig
from ttrl.algos.ppo import PPO, PPOConfig
from ttrl.algos.registry import ALGOS
from ttrl.algos.sac import SAC, SACConfig
from ttrl.spaces import ActionKind, ActionSpec, ObsGroup, ObsSpec
from ttrl.train.config import EnvConfig, TrainConfig

VECTOR = ObsGroup.of(ObsSpec.vector(4))
DISCRETE = ActionSpec.discrete(3)
CONTINUOUS = ActionSpec.continuous([-1.0, -1.0], [1.0, 1.0])
MULTI_DISCRETE = ActionSpec.multi_discrete([3, 4])

#: One spec per kind `ActionSpec` declares, keyed by the kind. Derived from
#: `ActionKind`, so a fourth kind fails the guard below instead of being
#: quietly absent from every table in this file.
BY_KIND = {spec.kind: spec for spec in (DISCRETE, CONTINUOUS, MULTI_DISCRETE)}

#: Per shipped algorithm: the class, a spec it accepts, a config whose network
#: batch is 8, and the config field that batch travels in. Keyed by the
#: registry's own names, so a fourth algorithm fails the guard below rather
#: than going unexercised.
NETWORK_BATCH_CASES = {
    "dqn": (DQN, DQNConfig, DISCRETE, DQNConfig(batch_size=8), "batch_size"),
    "sac": (SAC, SACConfig, CONTINUOUS,
            SACConfig(batch_size=8, learning_starts=8), "batch_size"),
    "ppo": (PPO, PPOConfig, CONTINUOUS,
            PPOConfig(n_steps=64, minibatch_size=8), "minibatch_size"),
}


def test_every_action_kind_and_every_shipped_algorithm_has_a_case():
    """Both tables above are checked against the code that defines the set."""
    assert set(BY_KIND) == set(typing.get_args(ActionKind))
    assert set(NETWORK_BATCH_CASES) == set(ALGOS.names())

# -- refusals ---------------------------------------------------------------

def test_dqn_refuses_continuous_actions_and_names_the_alternatives():
    with pytest.raises(ValueError) as caught:
        DQN(VECTOR, CONTINUOUS, device=None)
    message = str(caught.value)
    assert "discrete actions only" in message
    assert "'sac'" in message and "'ppo'" in message


def test_sac_refuses_discrete_actions_and_names_the_alternatives():
    with pytest.raises(ValueError) as caught:
        SAC(VECTOR, DISCRETE, device=None)
    message = str(caught.value)
    assert "continuous actions only" in message
    assert "'dqn'" in message and "'ppo'" in message


def test_multi_binary_is_the_composite_spec_and_not_a_fourth_kind():
    """MultiBinary is MultiDiscrete with every factor of size 2, so it is a
    constructor over the same machinery and no algorithm branches on it."""
    assert ActionSpec.multi_binary(3) == ActionSpec.multi_discrete([2, 2, 2])
    assert ActionSpec.multi_binary(3).kind == "multi_discrete"


def test_dqn_refuses_a_composite_action_and_names_the_enumeration():
    """A factorized action set is exponential in the factors, so a Q head over
    it is not the same algorithm; the refusal says what to do instead."""
    with pytest.raises(ValueError) as caught:
        DQN(VECTOR, MULTI_DISCRETE, device=None)
    message = str(caught.value)
    assert "multi_discrete" in message
    assert "'ppo'" in message
    assert "ActionSpec.discrete(12)" in message


@pytest.mark.parametrize("kind", sorted(BY_KIND))
def test_ppo_refuses_no_action_kind_at_all(kind):
    """PPO's refusal is about the minibatch, not the action kind, and the kinds
    are every one `ActionKind` declares."""
    with pytest.raises(ValueError, match="minibatch_size"):
        PPO(VECTOR, BY_KIND[kind], device=None,
            cfg=PPOConfig(n_steps=4, minibatch_size=64), n_envs=1)


@pytest.mark.parametrize("algo", sorted(NETWORK_BATCH_CASES))
def test_an_agent_refuses_more_lanes_than_its_fixed_batch(algo):
    """A convolution fixes its batch at construction, so the acting forward
    cannot be handed more lanes than the update's batch. One shared refusal
    names the config field that carries it, which differs per algorithm."""
    cls, _config_cls, action_spec, cfg, field = NETWORK_BATCH_CASES[algo]
    with pytest.raises(ValueError) as caught:
        cls(VECTOR, action_spec, device=None, cfg=cfg, n_envs=9)
    message = str(caught.value)
    assert "exceeds the network batch of 8" in message
    assert field in message


@pytest.mark.parametrize("algo", sorted(NETWORK_BATCH_CASES))
def test_network_batch_names_the_field_the_networks_are_built_for(algo):
    """The trainer validates the pairing without knowing the algorithm, so the
    batch cannot be guessed: it is minibatch_size for PPO, batch_size for the
    off-policy pair. Every algorithm must answer it the same way."""
    _cls, config_cls, _spec, _cfg, field = NETWORK_BATCH_CASES[algo]
    assert config_cls.NETWORK_BATCH_FIELD == field
    assert config_cls(**{field: 64}).network_batch == 64


# -- per-replica budgets ----------------------------------------------------

def test_scale_for_replicas_divides_the_frame_budgets():
    cfg = DQNConfig(buffer_capacity=1_000_000, learning_starts=80_000,
                    eps_decay_frames=1_000_000, batch_size=32, train_freq=4)
    scaled = cfg.scale_for_replicas(4)
    assert scaled.buffer_capacity == 250_000
    assert scaled.learning_starts == 20_000
    assert scaled.eps_decay_frames == 250_000
    # A rate is not a budget: four replicas each taking one step per four of
    # their own frames is the configured ratio, not a quarter of it.
    assert scaled.train_freq == cfg.train_freq
    assert scaled.batch_size == cfg.batch_size
    assert SACConfig(buffer_capacity=1000, learning_starts=100,
                     batch_size=32).scale_for_replicas(2).buffer_capacity == 500
    # PPO divides only the learning-rate schedule's horizon. n_steps is the
    # GAE horizon, and dividing it would shorten the advantage estimate.
    ppo = PPOConfig(anneal_lr=True, total_frames=1000, n_steps=64,
                    minibatch_size=32).scale_for_replicas(4)
    assert (ppo.total_frames, ppo.n_steps) == (250, 64)


def test_scale_for_replicas_is_the_identity_at_one_replica():
    cfg = DQNConfig()
    assert cfg.scale_for_replicas(1) is cfg


def test_scale_for_replicas_refuses_a_budget_that_does_not_divide():
    with pytest.raises(ValueError) as caught:
        DQNConfig(buffer_capacity=1000, learning_starts=400,
                  eps_decay_frames=400).scale_for_replicas(3)
    message = str(caught.value)
    assert "does not divide by 3 replicas" in message
    assert "realised buffer_capacity" in message


def test_scale_for_replicas_leaves_an_unset_horizon_alone():
    assert PPOConfig().total_frames is None
    assert PPOConfig().scale_for_replicas(4).total_frames is None


# -- configuration validation ----------------------------------------------

@pytest.mark.parametrize("kwargs, match", [
    ({"target_sync_interval": 0}, "target_sync_interval"),
    ({"eps_start": 0.1, "eps_end": 0.5}, "epsilon schedule"),
    ({"learning_starts": 4, "batch_size": 32}, "learning_starts"),
    ({"n_step": 0}, "n_step"),
])
def test_dqn_config_refuses_values_that_would_fail_late(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DQNConfig(**kwargs)


def test_ppo_config_defers_the_anneal_horizon_to_the_run():
    # The documented path: a config file sets anneal_lr and leaves the horizon
    # to TrainConfig. Refusing at construction made that path impossible.
    cfg = PPOConfig(anneal_lr=True)
    assert cfg.total_frames is None
    with pytest.raises(ValueError, match="anneal_lr is set but total_frames"):
        cfg.require_horizon()
    assert PPOConfig(anneal_lr=True, total_frames=10_000).require_horizon() == 10_000
    assert PPOConfig().require_horizon() is None


def test_train_config_fills_the_anneal_horizon_from_the_run():
    run = TrainConfig(
        algo="ppo", total_frames=4096, log_dir="runs/t",
        env=EnvConfig(name="gym", n_envs=1),
        algo_cfg=PPOConfig(anneal_lr=True, n_steps=64, minibatch_size=32),
    )
    assert run.algo_cfg.total_frames == 4096
    assert run.algo_cfg.require_horizon() == 4096


@pytest.mark.parametrize("kwargs, match", [
    ({"n_critics": 1}, "n_critics"),
    ({"tau": 0.0}, "tau"),
    ({"init_alpha": 0.0}, "init_alpha"),
])
def test_sac_config_refuses_values_that_remove_the_algorithm(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SACConfig(**kwargs)


# -- the checkpoint config diff ---------------------------------------------

def test_config_diff_names_each_differing_field_and_both_values():
    was, now = DQNConfig(), DQNConfig(gamma=0.5, lr=1e-2)
    text = describe_config_diff(was, now)
    assert text.startswith("2 fields differ: ")
    for field in ("gamma", "lr"):
        assert f"{field}={getattr(was, field)!r}" in text
        assert f"{field}={getattr(now, field)!r}" in text
    # Only the fields that differ, so the one that matters is not buried.
    assert "n_step" not in text
    assert "..." not in text and len(text.splitlines()) == 1


def test_config_diff_says_when_a_difference_invalidates_the_parameters():
    shape = describe_config_diff(SACConfig(), SACConfig(critic_hidden=(256, 128)))
    assert "critic_hidden" in shape and "networks' shapes" in shape
    plain = describe_config_diff(SACConfig(), SACConfig(tau=0.5))
    assert "networks' shapes" not in plain


def test_config_diff_shows_every_field_when_many_differ():
    fields = {"gamma": 0.5, "lr": 1e-2, "n_step": 1, "batch_size": 64,
              "train_freq": 8, "eps_end": 0.2, "encoder": "mlp"}
    text = describe_config_diff(DQNConfig(), DQNConfig(**fields))
    assert text.startswith(f"{len(fields)} fields differ: ")
    assert all(name in text for name in fields)

