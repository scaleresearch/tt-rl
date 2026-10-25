"""Each algorithm learns, and one update moves every parameter it owns."""

from __future__ import annotations

import numpy as np
import pytest

from tests._fakes.toy_envs import REACH_GROUP, SIGN_GROUP, ReachEnv, SignEnv, run_agent
from ttrl.algos import check_agent
from ttrl.algos.base import Transition
from ttrl.algos.dqn import DQN, DQNConfig
from ttrl.algos.ppo import PPO, PPOConfig
from ttrl.algos.sac import SAC, SACConfig
from ttrl.engine import acquire
from ttrl.engine.autograd import no_grad
from ttrl.spaces import ObsGroup
from ttrl.envs.vector import SyncVectorEnv
from ttrl.train.config import EnvConfig, TrainConfig

pytestmark = pytest.mark.tt

LANES = 4


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


def _vec(env_class):
    return SyncVectorEnv([lambda i=i: env_class(seed=i) for i in range(LANES)])


def _dqn(device):
    return DQN(SIGN_GROUP, SignEnv.action_spec, device, n_envs=LANES, seed=0,
               cfg=DQNConfig(batch_size=32, learning_starts=64, train_freq=4,
                             n_step=1, gamma=0.9, lr=1e-3, target_sync_interval=20,
                             eps_decay_frames=400, eps_end=0.05,
                             buffer_capacity=4000))


def _sac(device):
    return SAC(REACH_GROUP, ReachEnv.action_spec, device, n_envs=LANES, seed=0,
               cfg=SACConfig(batch_size=64, learning_starts=128, train_freq=2,
                             lr=1e-3, gamma=0.9, tau=0.02, buffer_capacity=8000,
                             init_alpha=0.1))


def _ppo(device, env_class):
    return PPO(ObsGroup.of(env_class.obs_spec), env_class.action_spec,
               device, n_envs=LANES,
               seed=0, cfg=PPOConfig(n_steps=32, minibatch_size=32, epochs=4,
                                     lr=3e-3, gamma=0.9, entropy_coef=0.0))


def _halves(rewards: np.ndarray) -> tuple[float, float]:
    first, last = np.array_split(rewards, 4)[0], np.array_split(rewards, 4)[-1]
    return float(first.mean()), float(last.mean())


def test_dqn_learns_a_discrete_task(device):
    agent = _dqn(device)
    check_agent(agent)
    try:
        rewards, metrics = run_agent(agent, _vec(SignEnv), 2400)
        start, end = _halves(rewards)
        losses = [m["loss"] for m in metrics]
        early = float(np.mean(losses[:20]))
        late = float(np.mean(losses[-20:]))
        assert end > start + 0.1
        assert end > 0.75
        assert late < early
    finally:
        agent.close()


def test_sac_learns_a_continuous_task(device):
    agent = _sac(device)
    check_agent(agent)
    try:
        rewards, metrics = run_agent(agent, _vec(ReachEnv), 4000)
        _start, end = _halves(rewards)
        early = float(np.mean([m["critic_loss"] for m in metrics[:20]]))
        late = float(np.mean([m["critic_loss"] for m in metrics[-20:]]))
        probe = np.stack([np.random.default_rng(90 + i).normal(size=3).astype(np.float32)
                          for i in range(16)])
        distance = float(np.abs(agent.act(probe, deterministic=True)
                                - np.tanh(probe[:, :2])).mean())
        # Every bar here is absolute, against what this task's optimum is
        # worth rather than against how bad the untrained policy happened to
        # be. A reward improvement over the first quarter measures the latter:
        # four 4000-frame variants all converged to the same distance of
        # ~0.035 and were separated only by a `start` that ranged from -0.32
        # to -0.64.
        #
        # Measured, this config, 4000 frames: distance 0.041, critic_loss
        # 1.64 -> 0.0034, end -0.04. With the actor's optimizer step removed:
        # distance 0.58, 1.67 -> 1.54, end -1.51. With the critic's removed:
        # distance 0.69, 3.54 -> 3.66, end -1.34.
        assert distance < 0.1
        assert late < 0.05
        assert late < early / 10.0
        assert end > -0.5
    finally:
        agent.close()


def _critic(agent, obs) -> float:
    """The critic's mean value for `obs`, read off the acting net."""

    with no_grad():
        _policy_out, value = agent._obs(agent.obs_spec.observation(obs))
    return float(value.numpy().reshape(-1)[: len(obs)].mean())


def test_ppo_learns_a_continuous_task(device):
    env_class = ReachEnv
    agent = _ppo(device, env_class)
    check_agent(agent)
    try:
        rewards, metrics = run_agent(agent, _vec(env_class), 8000)
        _start, end = _halves(rewards)
        late = float(np.mean([m["value_loss"] for m in metrics[-20:]]))
        probe = np.stack([np.random.default_rng(90 + i).normal(size=3).astype(np.float32)
                          for i in range(16)])
        distance = float(np.abs(agent.act(probe, deterministic=True)
                                - np.tanh(probe[:, :2])).mean())
        # ReachEnv never terminates, so a state is worth its policy's reward
        # per step for an unbounded horizon: `end / (1 - gamma)` is what the
        # critic is predicting, measured on the policy being scored.
        calibration = _critic(agent, probe) / (end / (1.0 - agent.cfg.gamma))
        # Every bar here is absolute, against what this task's optimum is
        # worth rather than against how bad the untrained policy happened to
        # be. `end > start + 0.1` measured the latter and separated runs by
        # how deep their starting hole was.
        #
        # Measured, this config, 8000 frames: distance 0.126, end -0.126,
        # entropy -0.03, value_loss 32.0 -> late20 0.092, calibration 0.77.
        # With the policy and entropy terms dropped from the objective (a
        # frozen actor): distance 0.682, end -1.715, entropy 3.31, late20
        # 3.202, calibration 1.19. With the value term dropped (a frozen
        # critic): distance 0.163, end -0.140, entropy -1.02, late20 0.209,
        # calibration 0.151.
        #
        # THE LAST BAR IS THE ONLY ONE A FROZEN CRITIC FAILS, and the four
        # before it are the ones a frozen actor fails. Unlike SAC, PPO's
        # actor is not trained through its critic: it learns from GAE
        # advantages, and on a task this short a useless baseline costs it
        # nothing, so a frozen critic solves ReachEnv as well as a healthy run
        # does and no bar on the task's reward can see it. What it cannot do
        # is say what a state is worth, which is what `calibration` reads.
        assert distance < 0.3
        assert end > -0.5
        assert late < 1.0
        assert metrics[-1]["entropy"] < 1.5
        assert calibration > 0.5
    finally:
        agent.close()


def _assert_all_move(before: dict, named, what: str) -> None:
    unmoved = [name for name, param in named
               if np.abs(param.numpy() - before[name]).max() == 0.0]
    assert not unmoved, f"{what}: one update did not move {unmoved}"


def _fill_until_ready(agent, env, limit: int) -> int:
    """Collect transitions until `should_update` says yes, and stop there."""

    obs, _info = env.reset(seed=0)
    for frame in range(0, limit, env.num_envs):
        action = agent.act(obs)
        result = env.step(action)
        agent.observe(Transition.from_step(env.obs_spec, obs, action, result))
        obs = result[0]
        if agent.should_update(frame):
            return frame
    raise AssertionError(
        f"{type(agent).__name__}.should_update stayed False for {limit} frames"
    )


def test_one_update_moves_every_parameter(device):
    """Not a name check. A net whose encoder is detached from its head still
    reports a falling loss and still exposes every parameter name."""
    dqn, sac, ppo = _dqn(device), _sac(device), _ppo(device, ReachEnv)
    try:
        for agent, env_class, groups in (
            (dqn, SignEnv, [("online", lambda a: a.online)]),
            (sac, ReachEnv, [("policy", lambda a: a.nets.policy),
                             ("critics", lambda a: a.nets.critics),
                             ("targets", lambda a: a.nets.targets)]),
            (ppo, ReachEnv, [("net", lambda a: a.net)]),
        ):
            with _vec(env_class) as env:
                _fill_until_ready(agent, env, 4000)
                named = [(f"{label}.{name}", param)
                         for label, get in groups
                         for name, param in get(agent).named_parameters()]
                before = {name: param.numpy().copy() for name, param in named}
                agent.update()
                _assert_all_move(before, named, type(agent).__name__)

        # DQN's target is deliberately NOT in that list: it only moves on a
        # sync, and a target that tracked every update would remove the delay
        # the algorithm depends on.
        frozen = {name: param.numpy().copy()
                  for name, param in dqn.target.named_parameters()}
        dqn.update()
        assert dqn.update_count % dqn.cfg.target_sync_interval != 0
        moved = [name for name, param in dqn.target.named_parameters()
                 if np.abs(param.numpy() - frozen[name]).max() > 0.0]
        assert not moved, f"DQN's target moved without a sync: {moved}"
    finally:
        for agent in (dqn, sac, ppo):
            agent.close()


def test_ppo_anneals_against_the_horizon_the_run_filled_in(device):
    """The path a config file takes: anneal_lr set, horizon left to the run."""

    total = 2048
    run = TrainConfig(
        algo="ppo", total_frames=total, log_dir="runs/anneal",
        env=EnvConfig(name="gym", n_envs=LANES),
        algo_cfg=PPOConfig(anneal_lr=True, n_steps=128, minibatch_size=64,
                           epochs=1, lr=3e-4),
    )
    cfg = run.algo_cfg
    assert cfg.total_frames == total
    agent = PPO(SIGN_GROUP, SignEnv.action_spec, device, cfg=cfg,
                n_envs=LANES, seed=0)
    try:
        seen = []
        with _vec(SignEnv) as env:
            for _ in range(3):
                _fill_until_ready(agent, env, 4000)
                agent.update()
                seen.append((agent.frame_count, agent.optimizer.lr))
        for frames, lr in seen:
            want = cfg.lr * max(0.0, 1.0 - frames / total)
            assert abs(lr - want) <= 1e-12
        lrs = [lr for _frames, lr in seen]
        assert lrs[0] > lrs[-1] > 0.0
        assert lrs == sorted(lrs, reverse=True)
    finally:
        agent.close()
