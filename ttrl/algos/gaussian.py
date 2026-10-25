"""The diagonal Gaussian policy, composed from primitive ops.

Both continuous algorithms need the same three pieces: a diagonal Normal
log-density, the tanh-squash correction SAC's bounded policy adds to it, and
the affine map from the squashed [-1, 1] cube onto the action bounds. None of
them is a new device op; every one composes from `sub`, `div`, `square`, `log`,
`exp`, `relu`, `abs_`, `sum_dim` and `tanh`, each of which already has a
gradchecked backward.

The host spellings next to each device one exist because acting samples on
host: the device computes the mean and log-std, the host draws the noise and
records the log-prob that a later update's importance ratio is measured
against.
"""

from __future__ import annotations

import math

import numpy as np

from ttrl.engine.autograd import Tensor
from ttrl.engine.ops import (
    abs_,
    add,
    div,
    exp,
    log,
    mul,
    neg,
    relu,
    square,
    sub,
    sum_dim,
    tanh,
)
from ttrl.spaces import ActionSpec

__all__ = [
    "ActionScale",
    "LOG_2PI",
    "gaussian_entropy",
    "log_one_minus_tanh_sq",
    "normal_logp",
    "normal_logp_numpy",
    "softplus",
    "squashed_logp",
    "squashed_logp_numpy",
]

LOG_2PI = math.log(2.0 * math.pi)
_LOG_2 = math.log(2.0)


def softplus(x: Tensor) -> Tensor:
    """log(1 + exp(x)), in the form that does not overflow.

    relu(x) + log(1 + exp(-|x|)). The naive spelling overflows for x beyond
    ~88 in fp32 and returns inf where the true value is x itself.
    """
    return add(relu(x), log(add(exp(neg(abs_(x))), 1.0)))


def normal_logp(x: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    """log N(x; mean, exp(log_std)^2), summed over the action dim, (B, 1)."""
    dim = int(x.shape[-1])
    z = div(sub(x, mean), exp(log_std))
    per_dim = sub(mul(square(z), -0.5), log_std)
    return sub(sum_dim(per_dim, dim=-1, keepdim=True), 0.5 * dim * LOG_2PI)


def log_one_minus_tanh_sq(u: Tensor) -> Tensor:
    """log(1 - tanh(u)^2), elementwise, in the form that stays finite.

    Written as 2 * (log 2 - u - softplus(-2u)), which is exact and whose
    derivative composes to -2 tanh(u). The direct spelling loses the whole
    quantity: at |u| = 10, tanh(u)^2 rounds to 1.0 in fp32 and the log is
    -inf, while the true value is about -18.6. That is the regime a converging
    SAC policy sits in, so the stable form is not a refinement.
    """
    return mul(sub(sub(_LOG_2, u), softplus(mul(u, -2.0))), 2.0)


def squashed_logp(u: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    """log density of tanh(u) under the tanh-squashed Gaussian, (B, 1)."""
    correction = sum_dim(log_one_minus_tanh_sq(u), dim=-1, keepdim=True)
    return sub(normal_logp(u, mean, log_std), correction)


def gaussian_entropy(log_std: Tensor) -> Tensor:
    """Exact differential entropy of the unsquashed diagonal Gaussian, (B, 1).

    sum(log_std) + 0.5 * dim * log(2 * pi * e). Exact rather than estimated
    from the sample, which is what PPO's entropy bonus wants on the continuous
    path; the discrete path uses `ttrl.engine.ops.entropy` on the logits.
    """
    dim = int(log_std.shape[-1])
    return add(sum_dim(log_std, dim=-1, keepdim=True),
               0.5 * dim * (LOG_2PI + 1.0))


# --------------------------------------------------------------------------
# host spellings, used by acting
# --------------------------------------------------------------------------

def normal_logp_numpy(x: np.ndarray, mean: np.ndarray, log_std: np.ndarray) -> np.ndarray:
    """`normal_logp` on host arrays, returning (B,)."""
    z = (x - mean) / np.exp(log_std)
    per_dim = -0.5 * z * z - log_std
    return (per_dim.sum(axis=-1) - 0.5 * x.shape[-1] * LOG_2PI).astype(np.float32)


def squashed_logp_numpy(u: np.ndarray, mean: np.ndarray, log_std: np.ndarray) -> np.ndarray:
    """`squashed_logp` on host arrays, returning (B,). Same stable form."""
    sp = np.maximum(-2.0 * u, 0.0) + np.log1p(np.exp(-np.abs(2.0 * u)))
    correction = 2.0 * (_LOG_2 - u - sp)
    return (normal_logp_numpy(u, mean, log_std) - correction.sum(axis=-1)).astype(np.float32)


class ActionScale:
    """The affine map from the squashed cube [-1, 1]^d onto the action bounds.

    A constant diagonal Jacobian, so it shifts every log-density by the same
    `-sum(log(half_width))` and changes no gradient. The shift is applied
    anyway, so a reported entropy or log-prob is the density of the action the
    environment actually received rather than of an intermediate.
    """

    def __init__(self, action_spec: ActionSpec, device=None):
        low = action_spec.low_array
        high = action_spec.high_array
        self.dim = int(action_spec.dim)
        self.centre = ((high + low) * 0.5).astype(np.float32)
        self.half_width = ((high - low) * 0.5).astype(np.float32)
        self.log_det = float(np.sum(np.log(self.half_width.astype(np.float64))))
        # Allocated once, because a fresh upload per call is a new device
        # address every call and a replayed trace does not see it.
        self._scale = self._centre = None
        if device is not None:
            self._scale = Tensor.from_numpy(self.half_width.reshape(1, self.dim), device)
            self._centre = Tensor.from_numpy(self.centre.reshape(1, self.dim), device)

    def to_env_numpy(self, squashed: np.ndarray) -> np.ndarray:
        return (self.centre + self.half_width * squashed).astype(np.float32)

    def to_squashed_numpy(self, env_action: np.ndarray) -> np.ndarray:
        return ((np.asarray(env_action, np.float32) - self.centre)
                / self.half_width).astype(np.float32)

    def to_env(self, squashed: Tensor) -> Tensor:
        """Device spelling, elementwise against the stored per-dim bounds.

        Broadcast from a (1, dim) row rather than a python scalar, because the
        bounds differ per action dimension in general.
        """
        if self._scale is None:
            raise ValueError(
                "ActionScale.to_env needs the device-resident bounds; construct "
                "it with ActionScale(action_spec, device)"
            )
        return add(mul(squashed, self._scale), self._centre)

    def env_logp(self, squashed_logp_value: Tensor) -> Tensor:
        return sub(squashed_logp_value, self.log_det)

    def env_logp_numpy(self, squashed_logp_value: np.ndarray) -> np.ndarray:
        return (squashed_logp_value - self.log_det).astype(np.float32)
