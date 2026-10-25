"""Gradient parity against torch for the vector/continuous policy.

Gated on correlation and Frobenius relative error rather than on max
elementwise relative error. The max metric divides the error on a small entry
by the tensor's largest entry, and a first-layer dW entry is a near-cancellation
of signed sums over the batch, so it reports the conditioning of the quantity
rather than the correctness of the backward.

The ReLU gate pattern comes from the device, through
`tests._fakes.torch_twin.DeviceGates`. A float64 reference cannot reproduce
which side of zero an fp32 pre-activation fell on, and one flipped gate moves
a whole `x[row] * dh[row, unit]` outer product into or out of the layer
below's gradient; that, and not the backward, used to dominate this comparison
at two hidden layers and forced a ceiling of 5e-2. With the gates agreeing the
ceiling is `GRAD_FROBENIUS_CEILING`, a few times the fp32 matmul floor.

`test_single_matmul_gradient_sits_on_the_documented_fp32_floor` is the anchor:
fp32 at HiFi4 has a measured ~1.3e-3 relative floor on this hardware, so one
matmul is the tightest a gradcheck here can be and is what the deeper cases
compound from. `test_a_backward_that_drops_the_relu_mask_fails_the_gate` and
`test_a_gate_the_reference_decides_for_itself_is_refused` are the controls
that the device's gate pattern is read as data and cannot launder a wrong
backward.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests._fakes.nets import upload_obs
from tests._fakes.torch_twin import (GRAD_FROBENIUS_CEILING, GRAD_PCC_FLOOR,
                                     DeviceGates, assert_grads_match, t64,
                                     torch_clamp, torch_stack)
from ttrl.engine import Tensor, acquire, backward
from ttrl.engine.layers import Linear
from ttrl.engine.ops import add, matmul, mean_all, square, sum_all, unary
from ttrl.engine.tensor import to_host
from ttrl.nets.build import build_sac_nets
from ttrl.spaces import DEFAULT_LEAF, ActionSpec, ObsGroup, ObsSpec

pytestmark = pytest.mark.tt

VECTOR = ObsGroup.of(ObsSpec.vector(17))
CONTINUOUS = ActionSpec.continuous([-1.0] * 6, [1.0] * 6)


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


def _policy_and_loss(hidden, device, *, batch=64, seed=3):
    """Build the policy, run the device forward, return what a twin needs."""
    policy = build_sac_nets(VECTOR, CONTINUOUS, device, batch=batch,
                            encoder_kwargs={"hidden": hidden}).policy
    x, x_np = upload_obs(VECTOR, batch, device, seed=seed)
    policy.zero_grad()
    device_mean, device_log_std = policy(x)
    loss = add(mean_all(square(device_mean)), mean_all(square(device_log_std)))
    return policy, x_np, loss


def _torch_policy(policy, x_np, gates):
    """The float64 twin of the policy, on the device's own gate pattern."""

    features, named = torch_stack(
        policy.encoder[DEFAULT_LEAF],
        torch.tensor(x_np, dtype=torch.float64), gates=gates,
        prefix=f"encoder.towers.{DEFAULT_LEAF}.layers.")
    head = policy.head
    mw, mb = t64(head.heads["mean"].weight.numpy()), t64(head.heads["mean"].bias.numpy())
    sw, sb = (t64(head.heads["log_std"].weight.numpy()),
              t64(head.heads["log_std"].bias.numpy()))
    named.update({"head.heads.mean.weight": mw, "head.heads.mean.bias": mb,
                  "head.heads.log_std.weight": sw, "head.heads.log_std.bias": sb})
    mean = features @ mw + mb
    log_std = torch_clamp(features @ sw + sb, head.log_std_min,
                          head.log_std_max, gates)
    return (mean ** 2).mean() + (log_std ** 2).mean(), named


@pytest.mark.parametrize("hidden", [(256,), (256, 256), (256, 256, 256)])
def test_gradcheck_vs_torch(hidden, device):
    """Every parameter gradient of an MLP + Gaussian policy against torch."""
    pytest.importorskip("torch")

    policy, x_np, device_loss = _policy_and_loss(hidden, device)
    gates = DeviceGates(device_loss)
    torch_loss, named = _torch_policy(policy, x_np, gates)
    gates.assert_every_site_used()
    torch_loss.backward()
    backward(device_loss)

    pairs = [(name, param, named[name])
             for name, param in policy.named_parameters()]
    assert len(pairs) == len(named)
    report = assert_grads_match(pairs)
    for name, (pcc, frobenius) in report.items():
        print(f"  {name}: pcc {pcc:.8f} frobenius {frobenius:.3e}")

    got_loss = float(device_loss.numpy().reshape(-1)[0])
    want_loss = float(torch_loss.detach())
    assert abs(got_loss - want_loss) / abs(want_loss) < 1e-2


def test_a_backward_that_drops_the_relu_mask_fails_the_gate(device, monkeypatch):
    """The control for taking the gate pattern from the device.

    A backward that fails to apply the mask is the bug class this comparison
    exists to catch, and feeding the device's own gate pattern to the
    reference must not make it passable. The reference applies the mask in its
    own forward and torch differentiates that, so the misplaced mask shows up
    in the gradient.
    """
    pytest.importorskip("torch")

    monkeypatch.setattr(unary.ttnn, "relu_bw", lambda grad, pre: [grad])
    policy, x_np, device_loss = _policy_and_loss((256, 256), device)
    torch_loss, named = _torch_policy(policy, x_np, DeviceGates(device_loss))
    torch_loss.backward()
    backward(device_loss)

    with pytest.raises(AssertionError):
        assert_grads_match([(name, param, named[name])
                            for name, param in policy.named_parameters()])


def test_a_gate_the_reference_decides_for_itself_is_refused(device):
    """The other half of the control: the device decides only the undecided.

    A pre-activation the float64 reference puts clearly on one side of zero is
    not up for negotiation, so a device pattern that disagrees there fails
    instead of being adopted. Without this, `DeviceGates` would be a knob that
    lets any gate pattern through.
    """
    pytest.importorskip("torch")

    policy, x_np, device_loss = _policy_and_loss((256,), device)
    gates = DeviceGates(device_loss)
    site = max(gates.sites, key=lambda a: a.size)
    flat = site.reshape(-1)
    # The entry just outside the band of pre-activations the reference calls
    # undecidable, which is the tightest case the guard has to catch.
    band = 1e-2 * np.sqrt(np.mean(flat ** 2))
    confident = int(np.flatnonzero(flat > band)[np.argmin(
        flat[flat > band])])
    flat[confident] = -flat[confident]

    with pytest.raises(AssertionError, match="wrong side of zero"):
        _torch_policy(policy, x_np, gates)


def test_single_matmul_gradient_sits_on_the_documented_fp32_floor(device):
    """The anchor for the compounding above: one matmul, not a composed net.

    fp32 at HiFi4 has a measured ~1.3e-3 relative floor on this hardware, so a
    single layer is the tightest a gradcheck can be and is what the deeper
    cases compound from.
    """
    torch = pytest.importorskip("torch")

    batch, in_features, out_features = 64, 17, 256
    x_np = np.random.default_rng(0).normal(size=(batch, in_features)).astype(np.float32)
    x = Tensor.from_numpy(x_np, device)
    layer = Linear(in_features, out_features, device)
    weight_np, bias_np = layer.weight.numpy().copy(), layer.bias.numpy().copy()

    backward(sum_all(square(add(matmul(x, layer.weight), layer.bias))))
    got = np.asarray(to_host(layer.weight.grad), dtype=np.float64)

    tw = torch.tensor(weight_np, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(bias_np, dtype=torch.float64, requires_grad=True)
    ((torch.tensor(x_np, dtype=torch.float64) @ tw + tb) ** 2).sum().backward()
    want = tw.grad.detach().numpy()

    relative = np.abs(got.reshape(want.shape) - want).max() / np.abs(want).max()
    assert relative < 3e-3, relative


def test_the_gate_is_tighter_than_the_kink_noise_it_replaced():
    """The constants gate on the fp32 floor and no longer on kink noise.

    One flipped relu gate was measured at 6.4e-2 Frobenius and pcc 0.99795 on
    the failing tensor, and the seed sweep in `tests._fakes.torch_twin` put a
    float64-gated reference at 1.5e-2 where the device-gated one measured
    5.4e-3 at the same depth. A ceiling above those cannot tell a flipped gate
    from a wrong backward, which is what this file is for.
    """
    assert GRAD_FROBENIUS_CEILING < 1.5e-2
    assert GRAD_PCC_FLOOR > 0.99795
