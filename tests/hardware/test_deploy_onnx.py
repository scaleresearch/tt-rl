"""The exported ONNX graph against a policy that trained on the chip."""

from __future__ import annotations

import numpy as np
import pytest

from tests._fakes.toy_envs import SIGN_GROUP, SignEnv, run_agent
from tests.deploy import _onnx
from ttrl.algos.ppo import PPO, PPOConfig
from ttrl.deploy import export_onnx, export_policy, load_policy
from ttrl.engine import acquire
from ttrl.envs.vector import SyncVectorEnv
from ttrl.envs.vec_normalize import NormalizeObservation
from ttrl.spaces import DEFAULT_LEAF
from ttrl.train.checkpoint import L0, save_checkpoint

pytestmark = pytest.mark.tt

OBS_DIM = 4
N_ACTIONS = 2
BATCH = 64
#: `PPO` builds its acting net for `minibatch_size`, so a device forward takes
#: at most this many observations at once.
ROWS = BATCH // 2

#: The device fp32 matmul floor, relative: a logit gap below this is a
#: decision the chip and the host may legitimately make differently.
NOISE = 1.3e-3


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


@pytest.fixture(scope="module")
def trained(device, tmp_path_factory):
    """A discrete PPO actor trained on the chip, its artifact and its graph."""
    _onnx.require_onnx()
    env = NormalizeObservation(SyncVectorEnv([lambda: SignEnv(seed=4)]))
    # minibatch_size below n_steps: an episode boundary leaves a padding step
    # in the rollout, so a full rollout holds slightly fewer valid transitions
    # than it has slots.
    cfg = PPOConfig(n_steps=BATCH, minibatch_size=BATCH // 2, epochs=2,
                    total_frames=4 * BATCH)
    agent = PPO(SIGN_GROUP, SignEnv.action_spec, device, cfg=cfg, n_envs=1,
                seed=7)
    try:
        run_agent(agent, env, 4 * BATCH, seed=4)
        directory = tmp_path_factory.mktemp("deploy_onnx")
        source = str(directory / "checkpoint.npz")
        save_checkpoint(
            source, agent_state=agent.state_dict(weights_only=True),
            obs_spec=SIGN_GROUP, action_spec=SignEnv.action_spec, level=L0,
            env_policy_state=env.save_policy_state())
        artifact = export_policy(source, str(directory / "policy.npz"))
        graph = export_onnx(artifact, str(directory / "policy.onnx"))
        yield agent, load_policy(artifact), graph
    finally:
        agent.close()
        env.close()


def observations(seed: int, rows: int = ROWS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, OBS_DIM)).astype(np.float32)


def test_the_exported_policy_chooses_what_the_device_actor_chooses(trained):
    """Both sides argmax the same logits, so the decisive indices must agree."""
    agent, policy, _graph = trained
    raw = observations(21)
    on_device = agent.act(policy.normalize(raw), deterministic=True)
    decisive = _onnx.decisive(policy, raw, NOISE)
    assert decisive.mean() > 0.5, "too few decisive rows to compare on"
    np.testing.assert_array_equal(policy(raw)[decisive], on_device[decisive])


def test_the_graph_takes_the_raw_observation_the_environment_emits(trained):
    """End to end in the deployed direction: the ONNX file alone, handed what
    the environment gives a controller, with no transform applied outside it.
    """
    _agent, policy, graph = trained
    env = SignEnv(seed=9)
    raw, _info = env.reset(seed=9)
    through_graph = _onnx.run(graph, raw.reshape(1, OBS_DIM))
    assert int(through_graph[0]) == int(policy(raw))
    assert 0 <= int(through_graph[0]) < N_ACTIONS


def test_the_parameter_names_this_exporter_reads_are_the_ones_ppo_writes(trained):
    """`tests/deploy/_checkpoints.py` builds PPO-shaped state by hand, which is
    only a valid stand-in while the real agent writes these names."""
    agent, policy, _graph = trained
    names = set(agent.state_dict(weights_only=True)["net"])
    assert f"encoders.actor.towers.{DEFAULT_LEAF}.layers.0.weight" in names
    assert "heads.policy.logits.weight" in names
    assert [n for n in names if n.startswith("encoders.critic.")
            or n.startswith("heads.value.")]
    assert (policy.algo, policy.action_map) == ("ppo", "argmax")
    assert policy.n_actions == N_ACTIONS and policy.action_dim is None
