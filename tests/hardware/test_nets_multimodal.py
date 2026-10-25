"""A camera for the actor and privileged state for the critic, end to end."""

from __future__ import annotations

import numpy as np
import pytest

from tests._fakes.pixel_envs import CAMERA, STATE_SIZE, pixel_reach
from ttrl.algos.ppo import PPO, PPOConfig
from ttrl.algos.base import Transition
from ttrl.engine import acquire
from ttrl.engine.autograd import no_grad
from ttrl.engine.conv import Conv2d
from ttrl.envs import SyncVectorEnv
from ttrl.nets.build import build_actor_critic, build_sac_nets
from ttrl.nets.multi import DEFAULT_ENCODER_BY_ROLE, build_encoder
from ttrl.nets.obs import ObsViews
from ttrl.spaces import ActionSpec, ObsGroup, ObsLeaf, ObsSpec

pytestmark = pytest.mark.tt

CONTINUOUS = ActionSpec.continuous([-1.0, -1.0], [1.0, 1.0])
LANES = 2
BATCH = 16

#: The robotics group: a camera the actor reads, simulator state it must not.
ROBOT = ObsGroup((ObsLeaf("camera", CAMERA),
                  ObsLeaf("sim_state", ObsSpec.vector(STATE_SIZE), actor=False)))

#: Two vector leaves, for the order and gradient gates: a conv tower's output
#: columns are not separable by eye, and these are.
PAIR = ObsGroup((ObsLeaf("first", ObsSpec.vector(4)),
                 ObsLeaf("second", ObsSpec.vector(6))))


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


def _pair_obs(rng) -> dict:
    return {"first": rng.normal(size=(BATCH, 4)).astype(np.float32),
            "second": rng.normal(size=(BATCH, 6)).astype(np.float32)}


def _robot_obs(rng, lanes: int = LANES) -> dict:
    return {
        "camera": rng.integers(0, 256, (lanes, *CAMERA.shape), dtype=np.uint8),
        "sim_state": rng.normal(size=(lanes, STATE_SIZE)).astype(np.float32),
    }


# -- what gets built --------------------------------------------------------


def test_each_leaf_gets_the_encoder_its_own_kind_and_role_imply(device):
    """Kind maps to an architecture, per role, over a mixed group.

    Both roles in one gate because they differ on exactly one cell: a vector
    leaf of a critic defaults to 'identity', a tower of no layers, so that
    `concat(s, a)` is the critic's input. Every builder is reached through
    `build_encoder`, so what is asserted is the module tree, not the table.
    """

    kinds = {leaf: ROBOT[leaf].kind for leaf in ROBOT.names}
    for role, table in DEFAULT_ENCODER_BY_ROLE.items():
        assert set(kinds.values()) == set(table), (
            "this group does not cover every observation kind, so it cannot "
            "hold the dispatch to account")
        encoder = build_encoder(ROBOT, device, batch=LANES, role=role)
        for leaf, kind in kinds.items():
            layers = encoder[leaf].layers
            assert any(isinstance(m, Conv2d) for m in layers) == (kind == "image"), (
                role, leaf, kind)
            # 'identity' is the only tower with no layer at all, and the one
            # cell where the critic's table differs from the observation's.
            assert (len(layers) == 0) == (table[kind] == "identity"), (role, leaf)


def test_an_image_critic_still_fuses_its_action_after_the_conv_tower(device):
    """Construction alone: defaulting a pixel critic to 'identity' cannot build.

    `identity_encoder` refuses an image spec by name rather than flattening
    one, so this is the gate that the refusal stays unreachable by default. A
    mixed group puts both cells of the critic table in ONE critic: the camera
    keeps its conv tower and the vector leaf fuses at the input.
    """

    nets = build_sac_nets(ROBOT, CONTINUOUS, device, batch=LANES, n_critics=2)
    for critic in nets.critics:
        towers = critic.encoder.encoder
        assert any(isinstance(m, Conv2d) for m in towers["camera"].layers)
        assert len(towers["sim_state"].layers) == 0
        assert critic.encoder.out_features == (
            towers["camera"].out_features + ROBOT["sim_state"].size
            + CONTINUOUS.width)


# -- the concatenation ------------------------------------------------------

def test_the_features_are_concatenated_in_declaration_order(device):
    """Order is the spec's, and the wrong order is what nothing else reports."""
    encoder = build_encoder(PAIR, device, batch=BATCH, hidden=(8,))
    rng = np.random.default_rng(0)
    views = ObsViews(PAIR, device, batch=BATCH)
    try:
        tensors = views.load(PAIR.observation(_pair_obs(rng)))
        with no_grad():
            joined = encoder(tensors.critic).numpy()
            parts = [encoder[leaf](tensors.critic[leaf]).numpy()
                     for leaf in PAIR.names]
        offset = 0
        for leaf, part in zip(PAIR.names, parts):
            width = encoder[leaf].out_features
            np.testing.assert_allclose(
                joined[:, offset:offset + width], part, rtol=0, atol=0,
                err_msg=f"leaf {leaf!r} is not at column {offset}")
            offset += width
        assert offset == encoder.out_features
    finally:
        views.close()


# -- end to end -------------------------------------------------------------

def test_ppo_trains_on_a_camera_plus_privileged_state(device):
    """The robotics case, driven through the real env stack and vector env."""
    env = SyncVectorEnv([lambda: pixel_reach(seed) for seed in range(LANES)])
    group = env.obs_spec
    assert group.names == ("obs", "sim_state")
    assert group.privileged == ("sim_state",)
    cfg = PPOConfig(n_steps=8, minibatch_size=BATCH, epochs=1, lr=3e-4,
                    total_frames=10_000)
    agent = PPO(group, env.action_spec, device, cfg=cfg, n_envs=LANES, seed=0)
    try:
        obs, _info = env.reset(seed=0)
        before = {name: param.numpy().copy()
                  for name, param in agent.named_parameters()}
        for _ in range(cfg.n_steps):
            action = agent.act(obs)
            assert action.shape == (LANES, CONTINUOUS.dim)
            result = env.step(action)
            agent.observe(Transition.from_step(group, obs, action, result))
            obs = result[0]
        assert agent.should_update(cfg.n_steps * LANES)
        metrics = agent.update()
        assert np.isfinite(metrics["loss"])
        moved = [name for name, param in agent.named_parameters()
                 if np.abs(param.numpy() - before[name]).max() > 0.0]
        # Both leaves' towers are in the critic, so both must have moved.
        for leaf in group.names:
            assert any(f"towers.{leaf}." in name for name in moved), leaf
    finally:
        agent.close()
        env.close()


def test_the_policy_cannot_see_the_privileged_leaf_at_run_time(device):
    """Replace every privileged value: the value must move, the policy not."""
    net = build_actor_critic(ROBOT, CONTINUOUS, device, batch=LANES)
    views = ObsViews(ROBOT, device, batch=LANES)
    rng = np.random.default_rng(2)
    obs = _robot_obs(rng)
    other = dict(obs)
    other["sim_state"] = np.full((LANES, STATE_SIZE), 9.0, np.float32)
    try:
        with no_grad():
            (mean, log_std), value = net(views.load(ROBOT.observation(obs)))
            first = (mean.numpy().copy(), log_std.numpy().copy(),
                     value.numpy().copy())
            (mean, log_std), value = net(views.load(ROBOT.observation(other)))
        np.testing.assert_array_equal(mean.numpy(), first[0])
        np.testing.assert_array_equal(log_std.numpy(), first[1])
        assert not np.allclose(value.numpy(), first[2]), (
            "the critic's value is unchanged by replacing every privileged "
            "value, so it is not reading them and this test cannot detect a "
            "leak")
    finally:
        views.close()


