"""Adam, composed from primitive elementwise ttnn ops and updated in place.

    m <- b1*m + (1-b1)*g
    v <- b2*v + (1-b2)*g^2
    p <- p - lr * (m/(1-b1**t)) / (sqrt(v/(1-b2**t)) + eps)

There is no fused alternative to compose against. `ttnn.moreh_adam`'s
`check_tensor` default dtype list is {BFLOAT16} in 0.77, 0.80 and 0.81-rc1
(byte-identical), so it rejects the fp32 parameters this optimizer
requires. Do not retry it without first re-checking that list.
"""

from __future__ import annotations

import ttnn

from ttrl.engine.optim.base import Optimizer
from ttrl.engine.optim.checkpoint import AdamCheckpointMixin
from ttrl.engine.optim.scalars import adam_fused_scalars
from ttrl.engine.tensor.precision import MASTER_DTYPE

__all__ = ["Adam"]


class Adam(Optimizer, AdamCheckpointMixin):
    """Adam over fp32 parameters, updated in place.

    `p.data` must be `MASTER_DTYPE`, which the engine keeps every parameter
    and all optimizer state in.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        if not self.params:
            raise ValueError("Adam: params is empty, so there is nothing to optimize")
        if lr <= 0:
            raise ValueError(f"Adam: lr must be > 0, got {lr}")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError(f"Adam: betas must be in [0, 1), got {betas}")
        if eps <= 0:
            raise ValueError(f"Adam: eps must be > 0, got {eps}")
        if weight_decay < 0:
            raise ValueError(f"Adam: weight_decay must be >= 0, got {weight_decay}")
        # `zeros_like` inherits the parameter's dtype, so a bf16 parameter
        # would give bf16 moment buffers. bf16 has 8 mantissa bits, and with
        # b2 = 0.999 the v recurrence's per-step increment falls below the
        # accumulator's ulp and vanishes: training flattens with no error.
        wrong = [i for i, p in enumerate(self.params) if p.data.dtype != MASTER_DTYPE]
        if wrong:
            raise ValueError(
                f"Adam: parameters must be {MASTER_DTYPE}; parameters {wrong} have "
                f"dtype {sorted({str(self.params[i].data.dtype) for i in wrong})}. "
                "Build them without the `dtype=` override to `Tensor.from_numpy`, "
                "which already defaults to fp32."
            )
        self.lr = lr
        self.b1, self.b2 = b1, b2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        # Per-parameter update count. Bias correction divides by 1 - beta**t
        # where t counts the gradients THIS parameter's moments have seen, not
        # the number of step() calls; the two differ whenever a parameter is
        # skipped for want of a gradient. Using the global count for a
        # parameter whose first gradient arrives on call 11 would divide its
        # single observation by 1 - 0.999**11 instead of 1 - 0.999.
        self.steps = [0] * len(self.params)
        # Address stability: allocated once here and never rebound, because a
        # captured trace replays against fixed device addresses.
        self.m = [ttnn.zeros_like(p.data) for p in self.params]
        self.v = [ttnn.zeros_like(p.data) for p in self.params]
        # Scratch allocated here rather than per step: a device allocation in
        # the eager gap between two trace replays can be handed an address the
        # trace already owns. weight_decay needs a second buffer, since the
        # decayed gradient must outlive the tail that reuses the first.
        self._scratch = [ttnn.zeros_like(p.data) for p in self.params]
        self._wd_scratch = (
            [ttnn.zeros_like(p.data) for p in self.params] if weight_decay else []
        )

    # The stored accumulators are S = m/(1-b1) and V = v/(1-b2). These two
    # factors convert them back to canonical Adam m and v for checkpointing.

    @property
    def _m_scale(self) -> float:
        return 1 - self.b1

    @property
    def _v_scale(self) -> float:
        return 1 - self.b2

    def step(self) -> None:
        """One update: 7 binary ops plus one unary per parameter.

        `alpha` in the `addalpha` calls below is only ever a constant beta. A
        t-dependent scalar there would key a fresh program-cache entry on
        every step.
        """
        self.t += 1
        b1, b2 = self.b1, self.b2

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.steps[i] += 1
            alpha_t, eps_t = self._step_scalars(self.steps[i])

            g = p.grad
            if self.weight_decay != 0.0:
                g = ttnn.addalpha(
                    g, p.data, self.weight_decay, output_tensor=self._wd_scratch[i]
                )

            S, V, s = self.m[i], self.v[i], self._scratch[i]

            # V <- g^2 + b2*V ;  S <- g + b1*S, both written into the existing
            # buffer. g^2 goes through the scratch, so V must be updated before
            # the scratch is reused below.
            ttnn.mul(g, g, output_tensor=s)
            ttnn.addalpha(s, V, b2, output_tensor=V)
            ttnn.addalpha(g, S, b1, output_tensor=S)

            # s <- alpha_t * S / (sqrt(V) + eps_t), one scratch buffer reused
            # by aliasing each op's output onto one of its inputs.
            ttnn.sqrt(V, output_tensor=s)
            ttnn.add(s, eps_t, output_tensor=s)
            ttnn.div(S, s, output_tensor=s)
            ttnn.mul(s, alpha_t, output_tensor=s)

            # Address stability: p.data is written through output_tensor and
            # never rebound, because captured traces replay against fixed
            # device addresses. Rebinding here fails silently.
            ttnn.sub(p.data, s, output_tensor=p.data)

    def _step_scalars(self, t: int) -> tuple[float, float]:
        return adam_fused_scalars(self.lr, self.b1, self.b2, self.eps, t)
