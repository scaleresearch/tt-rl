"""Spec-driven net construction on real hardware: build, forward, backward, step."""

from __future__ import annotations

import numpy as np
import pytest

from tests._fakes.nets import assert_every_parameter_trains, upload_obs
from ttrl.algos.ppo import PPO, PPOConfig
from ttrl.engine import Tensor, acquire, backward
from ttrl.engine.conv import ConvDoesNotFit
from ttrl.engine.ops import mean_all, square
from ttrl.engine.tensor import to_host
from ttrl.nets.build import build_q_net, build_sac_nets
from ttrl.nets.encoders import NATURE_CNN, ConvStage
from ttrl.nets.multi import build_encoder
from ttrl.nets.obs import ObsPair
from ttrl.spaces import DEFAULT_LEAF, ActionSpec, ObsGroup, ObsSpec

pytestmark = pytest.mark.tt

IMAGE = ObsGroup.of(ObsSpec.image(4, 84, 84))
DISCRETE = ActionSpec.discrete(6)
VECTOR = ObsGroup.of(ObsSpec.vector(17))
CONTINUOUS = ActionSpec.continuous([-1.0] * 6, [1.0] * 6)


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


# ---------------------------------------------------------------------------
# the three required cases
# ---------------------------------------------------------------------------

def test_image_obs_discrete_actions_nature_cnn_q_head(device):
    """The Atari case."""
    batch = 32
    net = build_q_net(IMAGE, DISCRETE, device, batch=batch)
    geometry = [l.out_shape for l in net.encoder[DEFAULT_LEAF].layers if hasattr(l, "out_shape")]
    assert geometry == [(32, 20, 20), (64, 9, 9), (64, 7, 7)]
    assert net.encoder.out_features == 512

    x, _ = upload_obs(IMAGE, batch, device)
    assert tuple(net(x).shape) == (batch, DISCRETE.width)
    names = assert_every_parameter_trains(net, lambda: mean_all(square(net(x))))
    assert "head.q.weight" in names


def test_64x64_needs_padding_and_is_refused_without_it(device):
    """The Nature CNN's strides require S = 4 (mod 8), and 64 is not."""
    spec = ObsGroup.of(ObsSpec.image(3, 64, 64))
    with pytest.raises(ValueError, match="ragged geometry"):
        build_q_net(spec, ActionSpec.discrete(4), device, batch=8)

    padded = (ConvStage(32, 8, 4, padding=2),) + NATURE_CNN[1:]
    net = build_q_net(spec, ActionSpec.discrete(4), device, batch=8, tower=padded)
    geometry = [l.out_shape for l in net.encoder[DEFAULT_LEAF].layers if hasattr(l, "out_shape")]
    assert geometry == [(32, 16, 16), (64, 7, 7), (64, 5, 5)]
    x, _ = upload_obs(spec, 8, device)
    assert_every_parameter_trains(net, lambda: mean_all(square(net(x))))


# ---------------------------------------------------------------------------
# actor-critic over both action kinds
# ---------------------------------------------------------------------------


def test_image_tower_builds_and_trains_above_batch_32(device):
    """The minibatch ceiling, pinned."""
    batch = 160
    net = build_q_net(IMAGE, DISCRETE, device, batch=batch)
    assert net.encoder[DEFAULT_LEAF].layers[0].plan.tspec is None, \
        "the first conv built dX machinery it will never use"
    x, _ = upload_obs(IMAGE, batch, device)
    assert tuple(net(x).shape) == (batch, DISCRETE.width)
    names = assert_every_parameter_trains(net, lambda: mean_all(square(net(x))))
    # The worry a reader will have: the first conv's WEIGHT gradient is still
    # required and still computed. Only its INPUT gradient was dropped.
    assert f"encoder.towers.{DEFAULT_LEAF}.layers.0.weight" in names


def test_first_conv_keeps_its_weight_gradient_without_an_input_gradient(device):
    """dW of the first conv is non-trivial, not merely present."""
    batch = 64
    net = build_q_net(IMAGE, DISCRETE, device, batch=batch)
    conv1 = net.encoder[DEFAULT_LEAF].layers[0]
    x, _ = upload_obs(IMAGE, batch, device)
    net.zero_grad()
    backward(mean_all(square(net(x))))
    grad = np.asarray(to_host(conv1.weight.grad))
    assert np.isfinite(grad).all()
    assert np.abs(grad).max() > 0.0
    # The observation itself carries no gradient, which is the whole premise.
    assert all(leaf.grad is None for leaf in x.actor.values())


def test_first_conv_without_dx_refuses_a_backward_that_needs_it(device):
    """No silent path: an input that does require a gradient raises."""
    batch = 8
    net = build_q_net(IMAGE, DISCRETE, device, batch=batch)
    channels, height, width = IMAGE[DEFAULT_LEAF].shape
    nhwc = np.zeros((1, 1, batch * height * width, channels), np.float32)
    # requires_grad=True stands in for an upstream layer the tower was handed.
    upstream = ObsPair.shared(
        {DEFAULT_LEAF: Tensor.from_numpy(nhwc, device, requires_grad=True)})
    net.zero_grad()
    with pytest.raises(RuntimeError, match="needs_input_grad=False"):
        backward(mean_all(square(net(upstream))))


def test_batches_whose_dx_does_not_fit_are_refused_at_construction(device):
    """conv2's dX genuinely does not fit at batch 192, and says so promptly."""
    with pytest.raises(ValueError, match="no shard placement fits for the dX"):
        build_q_net(IMAGE, DISCRETE, device, batch=192)


def test_space_to_depth_tower_trains_and_keeps_the_architecture(device):
    batch = 32
    net = build_q_net(IMAGE, DISCRETE, device, batch=batch, encoder="nature_cnn_s2d")
    conv1 = net.encoder[DEFAULT_LEAF].layers[0]
    assert conv1.geom.fold == 4
    # Same architecture: the folded convolution is a reindex, not a different
    # layer, so the output geometry is unchanged.
    assert conv1.out_shape == (32, 20, 20)
    x, _ = upload_obs(IMAGE, batch, device)
    assert_every_parameter_trains(net, lambda: mean_all(square(net(x))))


# refusals

def test_q_net_refuses_continuous_actions(device):
    with pytest.raises(ValueError, match="supports discrete actions only"):
        build_q_net(VECTOR, CONTINUOUS, device, batch=8)


def test_sac_refuses_discrete_actions(device):
    with pytest.raises(ValueError, match="supports continuous actions only"):
        build_sac_nets(VECTOR, DISCRETE, device, batch=8)


def test_sac_refuses_a_single_critic(device):
    """The twin minimum with one critic is that critic, which removes the
    overestimation correction entirely rather than weakening it."""
    with pytest.raises(ValueError, match="n_critics must be >= 2"):
        build_sac_nets(VECTOR, CONTINUOUS, device, batch=8, n_critics=1)


def test_image_encoder_refuses_a_vector_spec(device):
    with pytest.raises(ValueError, match="image encoder needs an image ObsSpec"):
        build_encoder(VECTOR, device, batch=8, name="nature_cnn")


# ---------------------------------------------------------------------------
# the registries, which need no device
# ---------------------------------------------------------------------------


def test_ppo_names_the_minibatch_when_the_network_will_not_build(device):
    """A conv's input gradient stops fitting in L1 above some batch, and the
    refusal must name `minibatch_size` at construction."""
    with pytest.raises(ValueError) as caught:
        PPO(ObsGroup.of(ObsSpec.image(4, 68, 68)), DISCRETE, device,
            cfg=PPOConfig(n_steps=4096, minibatch_size=4096), n_envs=1)
    assert "minibatch_size=4096" in str(caught.value)
    assert isinstance(caught.value.__cause__, ConvDoesNotFit)
