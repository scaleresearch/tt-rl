"""Exact parameter averaging across replicas, driven through the Agent surface."""

from __future__ import annotations

import numpy as np
import pytest

from tests._fakes.toy_envs import REACH_GROUP, SIGN_GROUP, ReachEnv, SignEnv, run_agent
from ttrl.algos import ALGOS
from ttrl.algos.dqn import DQN, DQNConfig
from ttrl.algos.ppo import PPO, PPOConfig
from ttrl.algos.sac import SAC, SACConfig
from ttrl.engine import acquire
from ttrl.engine.mesh import ParamSync, ParamSyncError
from ttrl.envs.vector import SyncVectorEnv

pytestmark = pytest.mark.tt

LANES = 2


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


def _dqn(device, seed):
    return DQN(SIGN_GROUP, SignEnv.action_spec, device, n_envs=LANES, seed=seed,
               cfg=DQNConfig(batch_size=16, learning_starts=32, train_freq=2,
                             n_step=1, buffer_capacity=1000)), SignEnv


def _ppo(device, seed):
    return PPO(REACH_GROUP, ReachEnv.action_spec, device, n_envs=LANES,
               seed=seed, cfg=PPOConfig(n_steps=8, minibatch_size=8, epochs=1)), ReachEnv


def _sac(device, seed):
    return SAC(REACH_GROUP, ReachEnv.action_spec, device, n_envs=LANES,
               seed=seed, cfg=SACConfig(batch_size=16, learning_starts=32,
                                        train_freq=2, buffer_capacity=1000)), ReachEnv


BUILDERS = {"dqn": _dqn, "ppo": _ppo, "sac": _sac}


#: The set is derived: a newly registered algorithm has to gain a builder here
#: or collection fails. Coverage rather than equality, since a key here need
#: not be a registry name.
_unbuilt = sorted(set(ALGOS.names()) - set(BUILDERS))
assert not _unbuilt, (
    f"registered algorithm(s) {_unbuilt} have no builder in this file: add a "
    f"builder for each to BUILDERS above so they reach these gates"
)


@pytest.mark.parametrize("algo", list(BUILDERS))
def test_param_sync_averages_agents_through_the_agent_surface(algo, device):
    build = BUILDERS[algo]
    first, env_class = build(device, 0)
    second, _ = build(device, 1)
    try:
        # Different seeds, so the two replicas really differ before averaging.
        for agent, seed in ((first, 0), (second, 7)):
            with SyncVectorEnv([lambda i=i, s=seed: env_class(seed=s + i)
                                for i in range(LANES)]) as env:
                run_agent(agent, env, 200, seed=seed)

        names = [name for name, _ in first.named_parameters()]
        assert names == [name for name, _ in second.named_parameters()]
        assert len(list(first.parameters())) == len(names)
        before = [np.stack([a.numpy(), b.numpy()])
                  for (_n, a), (_m, b) in zip(first.named_parameters(),
                                              second.named_parameters())]
        spread = max(float(np.abs(pair[0] - pair[1]).max()) for pair in before)
        assert spread > 0.0, "the replicas are identical, so averaging proves nothing"

        sync = ParamSync(modules=[first, second], sync_every=1)
        assert sync.maybe_average() is True

        worst = 0.0
        for pair, (name, a), (_m, b) in zip(before, first.named_parameters(),
                                            second.named_parameters()):
            want = pair.astype(np.float64).mean(axis=0)
            worst = max(worst, float(np.abs(a.numpy() - want).max()),
                        float(np.abs(b.numpy() - want).max()))
        assert worst < 1e-6
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("algo", list(BUILDERS))
def test_broadcast_makes_two_replicas_one_model(algo, device):
    """Averaging N independent initialisations is a different distribution, so
    a run starts by broadcasting replica 0 rather than averaging."""
    build = BUILDERS[algo]
    first, _env = build(device, 0)
    second, _ = build(device, 1)
    try:
        addresses = [id(param.data) for param in second.parameters()]
        ParamSync(modules=[first, second]).broadcast(src=0)
        worst = max(float(np.abs(a.numpy() - b.numpy()).max())
                    for a, b in zip(first.parameters(), second.parameters()))
        assert worst == 0.0
        # Address stability: the write goes through the existing buffer, so a
        # captured trace keeps replaying against live weights.
        assert [id(param.data) for param in second.parameters()] == addresses
    finally:
        first.close()
        second.close()


def test_a_schema_mismatch_is_refused_rather_than_averaged_positionally(device):
    """Two same-shaped parameters in different positions would average
    unrelated tensors and diverge the run with no error anywhere."""

    dqn, _ = _dqn(device, 0)
    ppo, _ = _ppo(device, 0)
    try:
        with pytest.raises(ParamSyncError, match="schemas differ"):
            ParamSync(modules=[dqn, ppo])
    finally:
        dqn.close()
        ppo.close()
