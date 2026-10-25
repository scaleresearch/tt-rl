"""Selection ops built from relu identities.

The relu identity routes the gradient to the selected side on its own. They
live here rather than beside the binary or unary ops because they need both
halves.

WHY STILL COMPOSED, measured on ttnn 0.80.0 / Blackhole (p300c), 1024 fp32
elements. The bfloat16-only restriction these compositions were written to work
around is GONE: `ttnn.clip`, `ttnn.clamp`, `ttnn.clip_bw`, `ttnn.clamp_bw`,
`ttnn.minimum`, `ttnn.maximum`, `ttnn.min_bw` and `ttnn.max_bw` each accept
float32, return float32 and are exact (max abs error 0.0 against numpy, both
forward and backward), and both clip and clamp_bw round-trip through trace
capture and replay with 0.0 error. The 0.80 docstrings still list BFLOAT16 only
and are simply wrong. So the composition is no longer forced by dtype.

It is kept because the gradcheck twin, not the device, depends on its shape.
`tests/_fakes/torch_twin.py` identifies each gate site by harvesting the relu
pre-activations off the device tape, so clamp's two relu sites are how the twin
pins the band's two boundaries. Calling the native pair instead was measured:
exact agreement with this composition on random fp32 input (grad delta 0.0,
forward delta 1.19e-07) and 33.9 us against 77.1 us per traced forward+backward,
but it drops those two sites and 8 tests fail to identify gates
(test_nets_gradcheck, test_algos_ppo_parity[continuous], test_algos_sac_parity).
Taking the 2.3x means reworking the twin's gate harvesting first.

One real semantic difference, should that happen: the native kernel's gradient
is inclusive at the bound (1 at x == lo and x == hi, as torch.clamp is) where
this composition gives 0 there. Measure-zero in fp32, and the native convention
is the more standard one.
"""

from __future__ import annotations

from ttrl.engine.autograd import Tensor
from ttrl.engine.ops.elementwise import add, sub
from ttrl.engine.ops.unary import relu


def minimum(a: Tensor, b) -> Tensor:
    """min(a, b) = a - relu(a - b).

    The relu identity routes the gradient to the selected side on its own.
    `ttnn.minimum`/`min_bw` are fp32-correct on 0.80 (see the module docstring)
    but want `b` as a device tensor, while callers here pass a Python float.
    """
    return sub(a, relu(sub(a, b)))


def maximum(a: Tensor, b) -> Tensor:
    """max(a, b) = a + relu(b - a). See `minimum`."""
    return add(a, relu(sub(b, a)))


def clamp(x: Tensor, lo: float, hi: float) -> Tensor:
    """Elementwise clamp to [lo, hi], differentiable.

    min(max(x, lo), hi) via two relu identities. Gradient is 1 strictly inside
    (lo, hi) and 0 where clamped. `ttnn.clip`/`clip_bw` and `clamp_bw` serve
    fp32 correctly as of 0.80 and would be 2.3x cheaper; the module docstring
    records the measurement and why this stays composed anyway.
    """
    lifted_max = add(relu(sub(x, lo)), lo)
    return sub(hi, relu(sub(hi, lifted_max)))
