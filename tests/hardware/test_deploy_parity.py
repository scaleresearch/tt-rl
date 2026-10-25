"""The exported artifact against the actor that trained on the chip."""

from __future__ import annotations

import numpy as np
import pytest

from tests._fakes.toy_envs import run_agent
from tests.deploy import _onnx
from ttrl.algos.dqn import DQN, DQNConfig
from ttrl.algos.sac import SAC, SACConfig
from ttrl.deploy import export_onnx, export_policy, load_policy
from ttrl.engine import acquire
from ttrl.envs.vector import SyncVectorEnv
from ttrl.envs.vec_normalize import NormalizeObservation
from ttrl.spaces import DEFAULT_LEAF, ActionSpec, ObsGroup, ObsSpec
from ttrl.train.checkpoint import L0, save_checkpoint

pytestmark = pytest.mark.tt

OBS_DIM = 11
ACTION_DIM = 3
#: The reviewer's actuator. Reporting the deviation in its units is the whole
#: point of stating it here rather than on a unit cube.
TORQUE = 12.0
OBS_SPEC = ObsSpec.vector(OBS_DIM)
#: What an agent is built over: the whole observation, one leaf of it.
OBS_GROUP = ObsGroup.of(OBS_SPEC)
ACTION_SPEC = ActionSpec.continuous([-TORQUE] * ACTION_DIM, [TORQUE] * ACTION_DIM)
BATCH = 128

#: Measured on Blackhole P300, chip 3, ttnn 0.80, over 128 observations each:
#: 0.0076 Nm on the training distribution and 0.0257 Nm where the normaliser
#: clips and tanh saturates, both on a +/-12 Nm actuator. The bound is a few
#: times the worst of those, which an ordinary kernel change stays inside and
#: a wrong forward pass (a dropped ReLU, an unfolded scale, the wrong trunk)
#: misses by whole Nm.
TOLERANCE_NM = 0.1


class Joints:
    """A vector env whose entries differ in scale by two orders of magnitude."""

    obs_spec = OBS_SPEC
    action_spec = ACTION_SPEC

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def _draw(self) -> np.ndarray:
        scales = np.linspace(0.1, 10.0, OBS_DIM)
        return (self._rng.normal(size=OBS_DIM) * scales).astype(np.float32)

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        return self._draw(), {}

    def step(self, action):
        self._t += 1
        reward = float(-np.sum(np.square(np.asarray(action) / TORQUE)))
        return self._draw(), reward, False, self._t % 50 == 0, {}

    def close(self) -> None:
        pass


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


@pytest.fixture(scope="module")
def exported(device, tmp_path_factory):
    """A trained-on-device actor, its checkpoint, the artifact and its path."""
    env = NormalizeObservation(SyncVectorEnv([lambda: Joints(seed=5)]))
    cfg = SACConfig(batch_size=BATCH, learning_starts=2 * BATCH, train_freq=1,
                    critic_hidden=(256,))
    agent = SAC(OBS_GROUP, ACTION_SPEC, device, cfg=cfg, n_envs=1, seed=7)
    try:
        run_agent(agent, env, 800, seed=5)
        directory = tmp_path_factory.mktemp("deploy")
        source = str(directory / "checkpoint.npz")
        save_checkpoint(
            source,
            agent_state=agent.state_dict(weights_only=True),
            obs_spec=OBS_GROUP,
            action_spec=ACTION_SPEC,
            level=L0,
            env_policy_state=env.save_policy_state(),
        )
        artifact = export_policy(source, str(directory / "policy.npz"))
        yield agent, env, load_policy(artifact), artifact
    finally:
        agent.close()
        env.close()


def _observations(seed: int) -> np.ndarray:
    scales = np.linspace(0.1, 10.0, OBS_DIM)
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(BATCH, OBS_DIM)) * scales).astype(np.float32)


@pytest.mark.parametrize("scale", [1.0, 40.0], ids=["in_distribution", "saturated"])
def test_the_exported_policy_matches_the_device_actor(exported, scale):
    agent, _env, policy, _path = exported
    raw = _observations(11) * scale
    on_device = agent.act(policy.normalize(raw), deterministic=True)
    on_host = policy(raw)
    deviation = float(np.abs(on_host - on_device).max())
    assert deviation < TOLERANCE_NM, (
        f"exported policy deviates {deviation:.5f} Nm from the device actor on a "
        f"+/-{TORQUE} Nm actuator"
    )


def test_the_artifact_carries_the_normaliser_the_run_fitted(exported):
    """The statistics in the file are the ones the training env ended with,
    not a fresh normaliser's zeros."""
    _agent, env, policy, _path = exported
    saved = dict(env.save_policy_state()[0][1])
    assert policy.normalizer["count"] == saved["count"] > 100
    # The fitted mean whitens to zero, and a fresh normaliser's zeros would
    # whiten it to the distribution's own offset, which is tens of sigma out
    # on the velocity dimensions.
    at_the_mean = policy.normalize(saved["mean"].astype(np.float32))
    np.testing.assert_allclose(at_the_mean, 0.0, atol=1e-4)


def test_the_onnx_graph_matches_the_device_actor_too(exported):
    """The ONNX file is the whole point of exporting: it is what a deployment
    stack that is not this one reads. Measured against the device actor in the
    same Nm as the artifact, so the two numbers sit side by side, and the
    comparison that pins the graph to the last bit is the host-against-host
    one in `tests/deploy/test_onnx_parity.py`."""

    agent, _env, policy, path = exported
    _onnx.require_onnx()
    graph = export_onnx(path, path.replace("policy.npz", "policy.onnx"))
    raw = np.concatenate([_observations(31), _observations(32) * 40.0])
    on_device = agent.act(policy.normalize(_observations(31)), deterministic=True)
    through_graph = _onnx.run(graph, raw)
    deviation = float(np.abs(through_graph[:BATCH] - on_device).max())
    relative = _onnx.parity(policy, graph, raw)
    assert deviation < TOLERANCE_NM, f"{deviation:.5f} Nm against the chip"
    assert relative <= _onnx.TOLERANCE


def test_the_parameter_names_this_exporter_reads_are_the_ones_sac_writes(exported):
    """`tests/deploy` builds SAC-shaped state by hand, which is only a valid
    stand-in while the real agent writes these names."""
    agent, _env, _policy, _path = exported
    names = set(agent.state_dict(weights_only=True)["nets"])
    assert f"policy.encoder.towers.{DEFAULT_LEAF}.layers.0.weight" in names
    assert "policy.head.heads.mean.weight" in names
    assert [n for n in names if n.startswith("critics.")]


# -- a masked policy ---------------------------------------------------------

MASK_ACTIONS = 4
MASK_FEATURES = 5
MASKED_SPEC = ObsSpec.vector(MASK_FEATURES + MASK_ACTIONS,
                             action_mask=MASK_ACTIONS)
MASKED_GROUP = ObsGroup.of(MASKED_SPEC)
MASKED_ACTION = ActionSpec.discrete(MASK_ACTIONS)


class Hands:
    """Discrete with a changing legal set: the card-game shape, in miniature.

    The reward pays for the legal action nearest the first feature's sign, so
    the policy has something to learn, and the environment REFUSES an illegal
    action rather than coercing one, which is what the contract asks of a
    masked environment.
    """

    obs_spec = MASKED_SPEC
    action_spec = MASKED_ACTION

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._obs = self._draw()

    def _draw(self) -> np.ndarray:
        features = self._rng.normal(size=MASK_FEATURES)
        mask = self._rng.integers(0, 2, MASK_ACTIONS).astype(np.float64)
        if not mask.any():
            mask[self._rng.integers(0, MASK_ACTIONS)] = 1.0
        return np.concatenate([features, mask]).astype(np.float32)

    def _mask(self) -> np.ndarray:
        return self._obs[-MASK_ACTIONS:]

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._obs = self._draw()
        return self._obs.copy(), {}

    def step(self, action):
        index = int(action)
        if self._mask()[index] < 0.5:
            raise ValueError(f"action {index} is illegal in this state")
        legal = np.flatnonzero(self._mask() > 0.5)
        want = int(legal[-1] if self._obs[0] > 0.0 else legal[0])
        reward = 1.0 if index == want else 0.0
        self._t += 1
        self._obs = self._draw()
        return self._obs.copy(), reward, False, self._t % 20 == 0, {}

    def close(self) -> None:
        pass


@pytest.fixture(scope="module")
def masked(device, tmp_path_factory):
    """A masked DQN trained on the chip, its checkpoint and its artifact."""

    env = SyncVectorEnv([lambda: Hands(seed=4)])
    cfg = DQNConfig(batch_size=64, learning_starts=128, train_freq=4,
                    eval_epsilon=0.0)
    agent = DQN(MASKED_GROUP, MASKED_ACTION, device, cfg=cfg, n_envs=1, seed=3)
    try:
        run_agent(agent, env, 600, seed=4)
        directory = tmp_path_factory.mktemp("masked")
        source = str(directory / "checkpoint.npz")
        save_checkpoint(
            source,
            agent_state=agent.state_dict(weights_only=True),
            obs_spec=MASKED_GROUP,
            action_spec=MASKED_ACTION,
            level=L0,
            env_policy_state=env.save_policy_state(),
        )
        artifact = export_policy(source, str(directory / "policy.npz"))
        yield agent, load_policy(artifact)
    finally:
        agent.close()
        env.close()


def _masked_observations(rows: int = 64, seed: int = 17) -> np.ndarray:
    """Features beside every legal set with at least one action in it."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, MASK_FEATURES)) * 2.0
    sets = np.array([m for m in np.ndindex(*([2] * MASK_ACTIONS)) if any(m)],
                    np.float32)
    masks = sets[rng.integers(0, len(sets), rows)]
    return np.concatenate([features, masks], axis=1).astype(np.float32)


def test_the_exported_masked_policy_chooses_what_the_device_agent_chooses(masked):
    """The gate is the rows where the mask FORBIDS the unmasked argmax: an
    artifact that dropped the mask agrees everywhere else, so a batch without
    those rows would pass for an exporter that ignored it."""
    agent, policy = masked
    obs = _masked_observations()
    on_device = np.asarray(agent.act(obs, deterministic=True)).reshape(-1)
    on_host = policy(obs)
    legal = policy.legal_mask(obs)
    unmasked = np.argmax(policy.head(obs), axis=-1)
    forbidden = legal[np.arange(len(obs)), unmasked] < 0.5
    decisive = _decisive(policy, obs)
    assert forbidden.sum() >= 8, "too few rows here forbid the unmasked argmax"
    assert (legal[np.arange(len(obs)), on_host] > 0.5).all(), (
        "the exported policy returned an action its own mask forbade")
    agree = on_host[decisive] == on_device[decisive]
    assert agree.all(), (
        f"{int((~agree).sum())} of {int(decisive.sum())} decisive observations "
        f"chose differently from the device agent"
    )


def _decisive(policy, obs):
    """Rows whose masked top-two gap clears the device's fp32 logit noise.

    A device logit sits about 1e-3 relative from the host's, so where the top
    two legal Q values are closer than that the two sides may legitimately
    pick differently. A row with one legal action has no second value and is
    decisive by construction.
    """
    rows = np.where(policy.legal_mask(obs) > 0.5, policy.head(obs), -np.inf)
    ordered = np.sort(rows, axis=-1)
    gap = ordered[:, -1] - ordered[:, -2]
    return np.isfinite(gap) & (gap > 1e-3) | ~np.isfinite(ordered[:, -2])
