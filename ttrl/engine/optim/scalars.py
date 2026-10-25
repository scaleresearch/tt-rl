"""Host-side Adam bias-correction algebra.

Device-free and pure, so the algebra is testable without a chip.
"""

from __future__ import annotations

import math

__all__ = ["adam_fused_scalars"]


def adam_fused_scalars(lr: float, b1: float, b2: float, eps: float, t: int) -> tuple[float, float]:
    """The two host scalars of the composed Adam tail, for update number `t`.

    The composed path stores S = m/(1-b1) and V = v/(1-b2) rather than
    canonical m and v, which removes the two on-device bias-correction passes:
    sqrt(vhat) = c * sqrt(V) with c = sqrt((1-b2)/(1-b2**t)), so dividing the
    numerator and the denominator of the update by c folds the whole bias
    correction into these two scalars.
    """
    if t < 1:
        raise ValueError(f"adam_fused_scalars: t must be >= 1, got {t}")
    bc1 = 1 - b1**t
    bc2 = 1 - b2**t
    c = math.sqrt((1 - b2) / bc2)
    return lr * (1 - b1) / bc1 / c, eps / c
