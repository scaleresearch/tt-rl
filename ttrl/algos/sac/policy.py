"""SAC's tanh-squashed diagonal Gaussian policy.

The reparameterised sample and its log-density, on device for the update and
on host for acting. Noise arrives as an argument rather than being drawn here,
because the device sample must read it from a bound buffer the host wrote: a
device-side draw would also make the update's randomness invisible to the
checkpointed generator state.
"""

from __future__ import annotations

import numpy as np

from ttrl.algos.gaussian import ActionScale, squashed_logp, squashed_logp_numpy
from ttrl.engine.autograd import Tensor
from ttrl.engine.ops import add, exp, mul, tanh
from ttrl.spaces import ActionSpec

__all__ = ["SquashedGaussian"]


class SquashedGaussian:
    """tanh of a diagonal Gaussian, mapped affinely onto the action bounds."""

    def __init__(self, action_spec: ActionSpec, device):
        self.action_spec = action_spec
        self.dim = int(action_spec.dim)
        self.scale = ActionScale(action_spec, device)

    def sample(self, mean: Tensor, log_std: Tensor, noise: Tensor):
        """Reparameterised sample and its log-density, both differentiable.

        -> (env_action (B, dim), logp (B, 1)). The gradient reaches `mean` and
        `log_std` through the sample itself, which is the whole point of the
        reparameterisation: SAC's actor loss differentiates the Q value of the
        sampled action with respect to the policy that sampled it.
        """
        pre_squash = add(mean, mul(exp(log_std), noise))
        logp = self.scale.env_logp(squashed_logp(pre_squash, mean, log_std))
        return self.scale.to_env(tanh(pre_squash)), logp

    def act(self, mean: np.ndarray, log_std: np.ndarray, rng, deterministic: bool):
        """Host sample. -> (env_action (B, dim), logp (B,)).

        `deterministic` takes tanh of the mean, which is the mode of the
        squashed distribution rather than its mean; the mean of a tanh-squashed
        Gaussian has no closed form and is not what an evaluation protocol
        wants anyway.
        """
        if deterministic:
            squashed = np.tanh(mean)
            logp = np.zeros(mean.shape[0], np.float32)
        else:
            pre_squash = (mean + np.exp(log_std) * rng.standard_normal(mean.shape)
                          ).astype(np.float32)
            squashed = np.tanh(pre_squash)
            logp = self.scale.env_logp_numpy(
                squashed_logp_numpy(pre_squash, mean, log_std))
        return self.scale.to_env_numpy(squashed), logp
