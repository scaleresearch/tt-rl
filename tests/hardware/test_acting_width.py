"""Acting runs at the lane count, through the staging the update already uses."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.algos.dqn import DQN, DQNConfig
from ttrl.engine import acquire
from ttrl.envs import ENVS, register
from ttrl.envs.wrappers import RecordEpisodeStatistics
from ttrl.engine.autograd import no_grad
from ttrl.engine.step.inputs import InputBinder
from ttrl.nets.build import build_actor_critic, build_q_net
from ttrl.nets.acting import net_at_batch
from ttrl.nets.obs import ObsBuffer, ObsViews, obs_tensor_shape
from ttrl.spaces import DEFAULT_LEAF, ActionSpec, ObsGroup, ObsSpec
from ttrl.train.config.schema import EnvConfig
from ttrl.train.evaluate import evaluate

pytestmark = pytest.mark.tt

IMAGE = ObsGroup.of(ObsSpec.image(4, 68, 68))
VECTOR = ObsGroup.of(ObsSpec.vector(11))
ACTIONS = ActionSpec.discrete(6)
WIDTH = 8
LANES = 2
#: Wide enough that the flattened-NHWC buffer clears the device-tilize gate.
GATE_WIDTH = 16
EVAL_LANES = 4
EPISODE_LEN = 3


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


def _obs(spec, lanes, seed=0):
    rng = np.random.default_rng(seed)
    leaf = spec[DEFAULT_LEAF]
    # Each leaf's own dtype: the buffers refuse a cast rather than making one.
    return rng.integers(0, 256, (lanes, *leaf.shape)).astype(leaf.numpy_dtype)


def test_the_narrow_view_reproduces_the_full_width_forward(device):
    """Bit-identical, because the only difference is how many rows are sent."""
    spec = IMAGE
    net = build_q_net(spec, ACTIONS, device, batch=WIDTH)
    obs = _obs(spec, LANES)
    wide = ObsViews(spec, device, batch=WIDTH)
    narrow = ObsViews(spec, device, batch=LANES)
    try:
        view = net_at_batch(net, LANES)
        with no_grad():
            full = net(wide.load(obs)).numpy().reshape(-1, ACTIONS.width)[:LANES]
            short = view(narrow.load(obs)).numpy().reshape(-1, ACTIONS.width)
        assert short.shape == (LANES, ACTIONS.width)
        assert np.array_equal(full, short)
    finally:
        wide.close()
        narrow.close()


def test_the_narrow_view_sees_a_parameter_update(device):
    """The views share one master weight, so writing it moves both."""
    net = build_q_net(IMAGE, ACTIONS, device, batch=WIDTH)
    view = net_at_batch(net, LANES)
    obs = _obs(IMAGE, LANES, seed=1)
    narrow = ObsViews(IMAGE, device, batch=LANES)
    try:
        with no_grad():
            before = view(narrow.load(obs)).numpy().copy()
        conv = net.encoder[DEFAULT_LEAF][0]
        conv.load_weight_numpy(conv.weight_numpy() * 0.5)
        with no_grad():
            after = view(narrow.load(obs)).numpy()
        assert not np.array_equal(before, after)
    finally:
        narrow.close()


def test_acting_buffers_are_as_wide_as_the_lanes(device):
    """The agent acts at n_envs, and batch_size sizes only the update."""
    agent = DQN(IMAGE, ACTIONS, device,
                cfg=DQNConfig(batch_size=WIDTH, learning_starts=10 ** 9),
                n_envs=LANES, seed=0)
    try:
        agent.act(_obs(IMAGE, LANES, seed=2))
        _net, views = agent._acting._view(LANES)
        leaf = views.critic.buffers[DEFAULT_LEAF]
        assert leaf.shape == obs_tensor_shape(IMAGE[DEFAULT_LEAF], LANES)
        assert leaf.shape != obs_tensor_shape(IMAGE[DEFAULT_LEAF], WIDTH)
    finally:
        agent.close()


def test_a_width_past_the_network_batch_is_refused(device):
    """Raises, naming the two knobs, rather than convolving reordered rows."""
    agent = DQN(IMAGE, ACTIONS, device,
                cfg=DQNConfig(batch_size=WIDTH, learning_starts=10 ** 9),
                n_envs=LANES, seed=0)
    try:
        with pytest.raises(ValueError, match="at most 8 observations"):
            agent.act(_obs(IMAGE, WIDTH + 1, seed=4))
    finally:
        agent.close()


def test_the_acting_upload_goes_through_the_device_tilize_gate(device, monkeypatch):
    """One staging path for acting and for learning, so the gate covers both."""
    seen = []
    real = InputBinder._device_tilize

    def record(self, name, tensor, arr):
        took = real(self, name, tensor, arr)
        seen.append((arr.nbytes, took))
        return took

    monkeypatch.setattr(InputBinder, "_device_tilize", record)
    for batch in (WIDTH, GATE_WIDTH):
        buffer = ObsBuffer(IMAGE[DEFAULT_LEAF], device, batch=batch)
        try:
            buffer.load(_obs(IMAGE, batch, seed=5))
        finally:
            buffer.close()
    assert [nbytes for nbytes, _ in seen] == [
        batch * 4 * 68 * 68 * 4 for batch in (WIDTH, GATE_WIDTH)]
    # Both sides of the gate, so a threshold change shows up here: the narrow
    # buffer is under it and the one the update's width would upload is over.
    assert [took for _, took in seen] == [False, True]


def test_a_narrow_view_keeps_what_the_net_shared(device):
    """One source module becomes one view module."""

    net = build_actor_critic(VECTOR, ACTIONS, device, batch=WIDTH)
    assert net.encoders["actor"] is net.encoders["critic"], (
        "a symmetric spec no longer shares one trunk, so this test has nothing "
        "to check; the sharing is build_actor_critic's shared-trunk case")
    view = net_at_batch(net, 2)
    assert view.encoders["actor"] is view.encoders["critic"], (
        "the narrow view split the shared trunk into two, so an acting forward "
        "runs the same columns twice and anything keyed off that identity "
        "disagrees with the net it is a view of")


class _ImageEnv:
    """Deterministic image observations, fixed-length episodes."""

    obs_spec = IMAGE
    action_spec = ACTIONS

    def __init__(self, episode_len: int = EPISODE_LEN):
        self.episode_len = episode_len
        self._t = 0
        self._seed = 0

    def _obs(self):
        return np.full(IMAGE[DEFAULT_LEAF].shape, (self._seed * 7 + self._t) % 256, dtype=np.uint8)

    def reset(self, *, seed=None):
        if seed is not None:
            self._seed = int(seed)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        return self._obs(), 1.0, self._t >= self.episode_len, False, {}

    def close(self):
        pass


def _make_image_env(**kwargs):
    return RecordEpisodeStatistics(_ImageEnv(**kwargs))


if "image_toy" not in ENVS:
    register("image_toy", _make_image_env)


@pytest.fixture
def agent(device):
    built = DQN(IMAGE, ACTIONS, device,
                cfg=DQNConfig(batch_size=WIDTH, learning_starts=10 ** 9),
                n_envs=LANES, seed=0)
    try:
        yield built
    finally:
        built.close()


def test_eval_scores_its_episode_count_on_more_lanes_than_training(agent):
    """Eval wider than training, end to end, at its configured episode count."""
    assert set(agent._acting._views) == {WIDTH, LANES}
    result = evaluate(agent, EnvConfig(name="image_toy", n_envs=EVAL_LANES),
                      n_episodes=EVAL_LANES * 2, seed=0, max_episode_steps=64)
    assert len(result["scores"]) == EVAL_LANES * 2
    assert result["n_episodes"] == EVAL_LANES * 2
    assert result["mean"] == pytest.approx(float(EPISODE_LEN))
    # The point: a width neither the agent's construction nor the training env
    # ever asked for was built, used, and kept.
    assert EVAL_LANES in agent._acting._views, sorted(agent._acting._views)


def test_fewer_episodes_than_eval_lanes_scores_exactly_that_many(agent):
    """`evaluate` never runs more lanes than episodes, which is what makes a
    wide eval env with a small episode count legal at a narrow network batch."""
    result = evaluate(agent, EnvConfig(name="image_toy", n_envs=EVAL_LANES),
                      n_episodes=2, seed=0, max_episode_steps=64)
    assert len(result["scores"]) == 2
    assert EVAL_LANES not in agent._acting._views, sorted(agent._acting._views)


