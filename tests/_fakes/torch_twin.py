"""Float64 torch twins of the engine's modules, for the algo parity tests.

Not a test module. A twin is built FROM the device module's own weights, so a
comparison measures the arithmetic rather than two different initialisations,
and it is built in float64 so the reference is not itself at the fp32 floor.

The one thing a float64 reference cannot reproduce is which side of zero a
ReLU pre-activation landed on: around 7 entries per 32768 differ between the
device's fp32 forward and the float64 forward, and one flipped gate moves a
whole rank-1 outer product into or out of the layer below's gradient. That
dominated the comparison and forced a 5e-2 Frobenius ceiling, which is wide
enough to hide a wrong backward.

So the gate pattern is taken FROM THE DEVICE, as data, through `DeviceGates`,
and the comparison then measures the backward arithmetic alone against the
fp32 matmul floor. `DeviceGates` is deliberately narrow about what the device
is allowed to decide; see its docstring.
"""

from __future__ import annotations

import numpy as np
import torch

from ttrl.engine.tensor import to_host

__all__ = ["DeviceGates", "GRAD_FROBENIUS_CEILING", "GRAD_PCC_FLOOR",
           "assert_close", "assert_grads_match", "t64", "torch_clamp",
           "torch_minimum", "torch_relu", "torch_stack"]

#: Once the gates agree, the only difference left is fp32 arithmetic, and the
#: spread across seeds collapses. Measured over 8 seeds on the policy
#: gradcheck, worst parameter of each run: 3.7e-3 to 4.7e-3 Frobenius at two
#: weight layers, 5.3e-3 to 5.8e-3 at three and 6.2e-3 to 7.8e-3 at four, with
#: pcc never below 0.999991. The same sweep with the reference recomputing its
#: own gates in float64 ran 3.9e-3 to 4.1e-2 against a 5e-2 ceiling, which is
#: the lottery these constants exist to remove.
GRAD_PCC_FLOOR = 0.9999
GRAD_FROBENIUS_CEILING = 1e-2

#: A device pre-activation is accepted as the twin's own site when it agrees
#: with the float64 pre-activation to this RMS distance, relative to the
#: tensor's own RMS. It identifies the site, so it is wide: it only has to
#: separate one site from another, and a site of a different depth or
#: different weights measures O(1) away. Measured agreement is ~1e-3.
SITE_MATCH_TOLERANCE = 5e-2

#: A gate may differ from the float64 gate only where the float64
#: pre-activation is this close to zero, relative to the tensor's own scale.
#: Below this band the sign is genuinely undecidable in fp32; above it the
#: device and the reference disagree about the function being computed, which
#: is the bug this gate exists to catch.
KINK_BAND = 1e-2


def t64(array):
    return torch.tensor(np.asarray(array, np.float64), dtype=torch.float64,
                        requires_grad=True)


class DeviceGates:
    """The ReLU gate patterns the device computed, offered to a twin as data.

    Built by walking the device tape back from a loss and reading each `relu`
    node's INPUT to host. That read is a host read, so it belongs in a test
    and never inside a captured step.

    What the device is allowed to decide is only which side of zero each
    pre-activation fell on, and only where the float64 reference cannot tell:

    * `mask_for(pre64)` identifies the site by matching the twin's own
      float64 pre-activation, so a device net with a missing, extra or
      differently shaped ReLU site has no unique match and raises.
    * the returned mask must equal the float64 gate everywhere the float64
      pre-activation is further from zero than `KINK_BAND` times the tensor's
      own scale. A device that gates a solidly positive unit off, or a mask
      taken from the wrong tensor, fails here rather than being adopted.
    * the twin, not this class, decides WHERE the mask multiplies. The mask
      comes back as a boolean array and the caller applies it at its own
      site, so a device backward that applies its mask at the wrong point,
      against the wrong operand or at the wrong layer still disagrees with the
      twin's gradient, and now disagrees against `GRAD_FROBENIUS_CEILING`
      instead of hiding under the 5e-2 the kink noise used to need.

    Gates are matched by content rather than by order so that one instance
    covers a loss built over several networks, in whatever order the twin
    happens to visit them.
    """

    def __init__(self, *roots):
        self.sites: list[np.ndarray] = []
        self.matched: set[int] = set()
        seen: set[int] = set()
        stack = list(roots)
        while stack:
            node = stack.pop()
            if isinstance(node, (tuple, list)):
                # A net whose forward returns several heads hands them over as
                # a tuple; every one of them is a root.
                stack.extend(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            if node._op == "relu":
                (source,) = node._parents
                self.sites.append(np.asarray(to_host(source.data), np.float64))
            stack.extend(node._parents)
        if not self.sites:
            raise AssertionError(
                "no relu node is reachable from the device tensors given, so "
                "there is no device gate pattern to compare against. A forward "
                "built under no_grad() records no tape."
            )

    def mask_for(self, pre64) -> np.ndarray:
        """The device's gate pattern for the float64 pre-activation `pre64`."""
        want = np.asarray(pre64.detach().numpy(), np.float64)
        scale = max(float(np.sqrt(np.mean(want ** 2))), 1e-30)
        distances = [
            (float(np.linalg.norm(site - want)) / (scale * np.sqrt(want.size)),
             index)
            for index, site in enumerate(self.sites)
            if site.shape == want.shape
        ]
        matches = [index for distance, index in distances
                   if distance <= SITE_MATCH_TOLERANCE]
        if len(matches) != 1:
            near = sorted(distance for distance, _ in distances)[:4]
            raise AssertionError(
                f"the device tape holds {len(matches)} relu pre-activations "
                f"matching a twin site of shape {want.shape} to "
                f"{SITE_MATCH_TOLERANCE}; {len(self.sites)} sites were "
                f"harvested, nearest relative distances {near}. A unique match "
                f"is required: no match means the device computed a different "
                f"set of activations than the twin, and several means the site "
                f"cannot be identified."
            )
        mask = self.sites[matches[0]] > 0.0
        decided = np.abs(want) > KINK_BAND * scale
        wrong = np.flatnonzero((decided & ((want > 0.0) != mask)).reshape(-1))
        if wrong.size:
            worst = np.abs(want.reshape(-1)[wrong]).max() / scale
            raise AssertionError(
                f"the device gated {wrong.size} of {want.size} units on the "
                f"wrong side of zero at pre-activations the float64 reference "
                f"decides unambiguously (worst {worst:.3g} of the tensor's own "
                f"scale, band {KINK_BAND}). That is a disagreement about the "
                f"function computed, not fp32 kink noise."
            )
        self.matched.add(matches[0])
        return mask

    def assert_every_site_used(self) -> int:
        """Every harvested gate site was claimed by a twin site.

        For a twin that covers the whole device graph. An unclaimed site means
        the device applied a ReLU the reference does not have, which no
        per-site comparison can see.
        """
        missing = sorted(set(range(len(self.sites))) - self.matched)
        assert not missing, (
            f"{len(missing)} of {len(self.sites)} device relu sites were never "
            f"claimed by the twin (shapes "
            f"{[self.sites[i].shape for i in missing]}); the reference is "
            f"missing an activation the device computed."
        )
        return len(self.sites)


def torch_relu(x, gates=None):
    """float64 relu, on the device's gate pattern when one is supplied."""

    if gates is None:
        return torch.relu(x)
    return x * torch.tensor(gates.mask_for(x), dtype=torch.float64)


def torch_clamp(x, lo: float, hi: float, gates=None):
    """float64 clamp composed exactly as `ttrl.engine.ops.select.clamp` is.

    The clamp is two relu identities on the device, so its two boundaries are
    two more gate sites with the same kink problem as a hidden layer's: a
    pre-activation a hair inside the band in fp32 and a hair outside it in
    float64 zeroes a whole row of the head's gradient. Composing it the same
    way here lets those two sites take the device's pattern too.
    """

    if gates is None:
        return torch.clamp(x, lo, hi)
    lifted = torch_relu(x - lo, gates) + lo
    return hi - torch_relu(hi - lifted, gates)


def torch_minimum(a, b, gates=None):
    """float64 min composed exactly as `ttrl.engine.ops.select.minimum` is.

    `a - relu(a - b)`, so the one place the gradient chooses between the two
    critics is a gate site like any other and takes the device's choice where
    the two values are within fp32 of each other.
    """

    if gates is None:
        return torch.minimum(a, b)
    return a - torch_relu(a - b, gates)


def torch_stack(module, x, *, named=None, prefix="", gates=None):
    """Run a float64 twin of an engine `Sequential`-like module over `x`.

    Returns `(output, {name: torch parameter})`, keyed by the same dotted names
    the device module reports, so a caller can look a reference gradient up by
    the name it got from `named_parameters()`.

    `gates` is a `DeviceGates` harvested from the device forward this twin is
    being compared against. Passing it makes every ReLU site take its gate
    pattern from the device; passing None recomputes the gate in float64,
    which leaves the comparison dominated by kink flips at any depth past one
    hidden layer.

    A child is classified by the parameters it owns rather than by its type: a
    child with none is an activation, a child with its own `layers` is a
    nested stack, and anything else is one affine map whose weight and bias
    may sit behind a name of their own (a `ValueHead` holds `value.weight`).
    Classifying by `hasattr(layer, "weight")` instead silently treated a
    `ValueHead` as a ReLU, which made a reference wrong rather than failing.
    """

    named = {} if named is None else named
    for index, layer in enumerate(module.layers):
        params = dict(layer.named_parameters())
        if not params:
            x = torch_relu(x, gates)
            continue
        if hasattr(layer, "layers"):
            x, named = torch_stack(layer, x, named=named, gates=gates,
                                   prefix=f"{prefix}{index}.")
            continue
        weight_name = next(k for k in params if k.endswith("weight"))
        bias_name = next(k for k in params if k.endswith("bias"))
        tw, tb = t64(params[weight_name].numpy()), t64(params[bias_name].numpy())
        named[f"{prefix}{index}.{weight_name}"] = tw
        named[f"{prefix}{index}.{bias_name}"] = tb
        # `ScaledLinear` folds ObsSpec.scale into its weight, exactly as the
        # device layer does, so the twin multiplies the same number in.
        x = x @ (tw * float(getattr(layer, "scale", 1.0))) + tb
    return x, named


def assert_close(got: float, want: float, *, tolerance: float = 2e-2, what: str = "") -> float:
    """Relative agreement of two scalars, returned so a caller can report it."""
    relative = abs(got - want) / max(abs(want), 1e-12)
    assert relative < tolerance, f"{what}: got {got!r}, want {want!r}, rel {relative}"
    return relative


def assert_grads_match(pairs, *, pcc_floor=GRAD_PCC_FLOOR,
                       frobenius_ceiling=GRAD_FROBENIUS_CEILING
                       ) -> dict[str, tuple[float, float]]:
    """`[(name, device Tensor, torch tensor)]` -> `{name: (pcc, frobenius)}`.

    Every parameter must have received a gradient: a net whose encoder is
    detached from its head still trains and still reports a falling loss, so
    the per-parameter comparison is the only thing that catches it.
    """

    report: dict[str, tuple[float, float]] = {}
    for name, param, reference in pairs:
        assert param.grad is not None, f"no gradient reached {name}"
        assert reference.grad is not None, f"the torch twin has no grad for {name}"
        got = np.asarray(to_host(param.grad), np.float64).reshape(reference.shape)
        want = reference.grad.detach().numpy()
        assert np.isfinite(got).all(), f"non-finite gradient in {name}"
        frobenius = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30)
        # Correlation is undefined for a single number, and a one-element
        # parameter (a scalar head's bias, a temperature) is exactly the case
        # where the Frobenius ratio already IS the relative error.
        pcc = (1.0 if want.size < 2
               else float(np.corrcoef(got.reshape(-1), want.reshape(-1))[0, 1]))
        assert pcc > pcc_floor, f"{name}: pcc {pcc}, frobenius {frobenius}"
        assert frobenius < frobenius_ceiling, f"{name}: frobenius {frobenius}"
        report[name] = (pcc, float(frobenius))
    return report
