"""The SAC critic must be able to represent a Q that is not affine in the action."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.engine import Adam, Tensor, acquire, backward
from ttrl.engine.ops import mean_all, square, sub
from ttrl.nets.build import build_sac_nets
from ttrl.nets.fusion import ConcatEncoder
from ttrl.nets.multi import build_encoder
from ttrl.nets.heads import ValueHead
from ttrl.nets.models import ContinuousCritic
from ttrl.nets.obs import ObsPair
from ttrl.spaces import DEFAULT_LEAF, ActionSpec, ObsGroup, ObsSpec

pytestmark = pytest.mark.tt

OBS = ObsGroup.of(ObsSpec.vector(6))
ACTION = ActionSpec.continuous([-1.0] * 3, [1.0] * 3)
N_STATES = 32
ACTIONS_PER_STATE = 8
BATCH = N_STATES * ACTIONS_PER_STATE
STEPS = 400
LR = 1e-3

#: A capable critic must reach this fraction of the computed affine floor.
#: Beating the floor 3x is already a statement about what the critic can
#: REPRESENT, since the floor is a hard bound on every action-affine model.
#: Deliberately not tighter: the measured value is ~0.05 of the floor, and a
#: threshold close to the measurement turns an init that converges a little
#: slower into a failure. This test is about capability, so the margin is sized
#: for the claim and not for the best run observed.
CAPABLE_FRACTION_OF_FLOOR = 0.33

#: Weight init draws from numpy's global RNG, which every other test in a full
#: suite perturbs, so an unseeded run here is a different network each time.
#: Seeded so a failure means the critic changed rather than that the collection
#: order did. Found the hard way: an earlier 0.1 threshold passed this file
#: alone and failed inside the full hardware suite.
INIT_SEED = 0


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


def _problem(seed=0):
    """`(states, actions, targets, state_index)` for Q(s, a) = -||a - tanh(s)||^2."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(N_STATES, OBS[DEFAULT_LEAF].size)).astype(np.float32)
    states = np.repeat(base, ACTIONS_PER_STATE, axis=0)
    actions = rng.uniform(-1.0, 1.0, (BATCH, ACTION.dim)).astype(np.float32)
    best = np.tanh(states[:, : ACTION.dim])
    targets = -((actions - best) ** 2).sum(axis=1, keepdims=True).astype(np.float32)
    state_index = np.repeat(np.arange(N_STATES), ACTIONS_PER_STATE)
    return states, actions, targets, state_index


def _affine_floor(actions, targets, state_index) -> float:
    """Least-squares mse of the best action-affine model of this data."""
    design = np.zeros((targets.shape[0], ACTION.dim + N_STATES), np.float64)
    design[:, : ACTION.dim] = actions
    design[np.arange(targets.shape[0]), ACTION.dim + state_index] = 1.0
    coefficients, *_ = np.linalg.lstsq(design, targets.reshape(-1), rcond=None)
    residual = design @ coefficients - targets.reshape(-1)
    return float((residual ** 2).mean())


def _build_critic(device, **kwargs):
    """A critic from the public builder, with a reproducible init."""
    np.random.seed(INIT_SEED)
    return build_sac_nets(OBS, ACTION, device, batch=BATCH, **kwargs).critics[0]


def _regress(critic, device, seed=0):
    """Fit `critic` to the target by supervised regression; return final mse."""
    states, actions, targets, _ = _problem(seed)
    s = ObsPair.shared(
        {DEFAULT_LEAF: Tensor.from_numpy(states, device, requires_grad=False)})
    a = Tensor.from_numpy(actions, device, requires_grad=False)
    y = Tensor.from_numpy(targets, device, requires_grad=False)

    params = list(critic.parameters())
    optimizer = Adam(params, lr=LR)
    mse = float("nan")
    for _ in range(STEPS):
        critic.zero_grad()
        loss = mean_all(square(sub(critic(s, a), y)))
        backward(loss)
        optimizer.step()
        mse = float(loss.numpy().reshape(-1)[0])
    return mse, float(targets.var())


def _affine_critic(device):
    """The shipped-and-wrong arrangement: ValueHead straight after the concat."""
    np.random.seed(INIT_SEED)
    tower = build_encoder(OBS, device, batch=BATCH)
    joint = ConcatEncoder(tower, ACTION.width)
    return ContinuousCritic(joint, ValueHead(joint.out_features, device))


def test_default_sac_critic_fits_a_non_affine_q(device):
    """The capability: beat the best possible action-affine fit by at least 3x,
    measured at about 20x."""
    _, actions, targets, state_index = _problem()
    floor = _affine_floor(actions, targets, state_index)
    mse, variance = _regress(_build_critic(device), device)
    affine_mse, _ = _regress(_affine_critic(device), device)
    assert variance > 1.0, f"degenerate target, variance {variance}"
    assert floor > 0.2 * variance, (
        f"affine floor {floor:.5f} is small next to the target variance "
        f"{variance:.3f}; this problem no longer separates the two designs"
    )
    assert mse < CAPABLE_FRACTION_OF_FLOOR * floor, (
        f"default SAC critic reached mse {mse:.5f} against an action-affine "
        f"floor of {floor:.5f} (target variance {variance:.3f}); it is not "
        f"representing the action's state-dependent effect on Q"
    )
    # The control: the shipped-and-wrong critic must fail the same assertion.
    assert affine_mse > 0.5 * floor, (
        f"the action-affine critic reached mse {affine_mse:.5f} against a "
        f"computed floor of {floor:.5f}; the control no longer controls"
    )
    assert mse * 5 < affine_mse, f"capable {mse:.5f} vs affine {affine_mse:.5f}"


def test_critic_hidden_is_required_to_be_non_empty(device):
    """Asking for no hidden layer is the bug, so it is refused by name."""
    with pytest.raises(ValueError, match="affine in the action"):
        build_sac_nets(OBS, ACTION, device, batch=8, critic_hidden=())
    with pytest.raises(ValueError, match="at least one positive width"):
        build_sac_nets(OBS, ACTION, device, batch=8, critic_hidden=(0,))


