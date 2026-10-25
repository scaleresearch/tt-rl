"""Output heads. The head is chosen by `ActionSpec.kind` and its width comes
from `ActionSpec.width`.

Every head emits raw numbers: Q-values, logits, a mean and a log standard
deviation, a scalar value. No softmax, log-softmax or tanh squashing happens
here, because which of those an algorithm needs is the algorithm's business
(DQN's target, PPO's log-prob and entropy, SAC's reparameterised sample).
"""

from __future__ import annotations

from ttrl.engine.layers import Linear
from ttrl.engine.nn import Module, ModuleDict
from ttrl.engine.ops import clamp
from ttrl.spaces import ActionSpec

__all__ = [
    "CategoricalHead",
    "GaussianHead",
    "QHead",
    "ValueHead",
    "build_policy_head",
]

#: SAC's standard log-std bounds. Clamped rather than exponentiated raw: a
#: log_std the policy drives to +30 makes the sampled action overflow and the
#: log-prob NaN, and a clamp's gradient is exactly zero outside the range,
#: which is the correct derivative of the clamped function.
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class QHead(Module):
    """Raw action-value head, one output per discrete action."""

    def __init__(self, in_features: int, n_actions: int, device):
        self.q = Linear(in_features, n_actions, device)
        self.n_actions = int(n_actions)

    def forward(self, x):
        return self.q(x)


class CategoricalHead(Module):
    """Raw logits head, one output per discrete action."""

    def __init__(self, in_features: int, n_actions: int, device):
        self.logits = Linear(in_features, n_actions, device)
        self.n_actions = int(n_actions)

    def forward(self, x):
        return self.logits(x)


class ValueHead(Module):
    """Scalar state-value or Q(s, a) head."""

    def __init__(self, in_features: int, device):
        self.value = Linear(in_features, 1, device)

    def forward(self, x):
        return self.value(x)


class GaussianHead(Module):
    """Diagonal-Gaussian policy head: mean and log_std over a shared trunk.

    The two heads live in a `ModuleDict` so they are named `heads.mean.weight`
    and `heads.log_std.weight` and a checkpoint cannot read one back onto the
    other, which positional naming in a list would permit if the two were ever
    reordered.
    """

    def __init__(self, in_features: int, action_dim: int, device, *,
                 log_std_min: float = LOG_STD_MIN, log_std_max: float = LOG_STD_MAX):
        if log_std_min >= log_std_max:
            raise ValueError(
                f"GaussianHead: log_std_min must be < log_std_max, got "
                f"{log_std_min} and {log_std_max}"
            )
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.heads = ModuleDict({
            "mean": Linear(in_features, self.action_dim, device),
            "log_std": Linear(in_features, self.action_dim, device),
        })

    def forward(self, x):
        """`(mean, log_std)`, both `(batch, action_dim)`, log_std clamped."""
        mean = self.heads["mean"](x)
        log_std = clamp(self.heads["log_std"](x), self.log_std_min, self.log_std_max)
        return mean, log_std


def build_policy_head(action_spec: ActionSpec, in_features: int, device, **kwargs):
    """The policy head an `ActionSpec` calls for.

    Discrete gets categorical logits of width `n`; multi_discrete gets one
    `CategoricalHead` of width `sum(nvec)`, which the policy cuts into one
    segment of logits per factor; continuous gets a Gaussian of width `dim`.
    `ActionSpec.width` already is the right number for all three.

    A factorized action takes ONE head rather than one head per factor, so the
    emitted width is a build-time constant and the segment boundaries do not
    depend on anything read at step time.
    """
    if action_spec.kind in ("discrete", "multi_discrete"):
        if kwargs:
            raise ValueError(
                f"build_policy_head: a {action_spec.kind} head takes no extra "
                f"options, got {sorted(kwargs)}; log_std bounds apply to "
                f"continuous actions only"
            )
        return CategoricalHead(in_features, action_spec.width, device)
    return GaussianHead(in_features, action_spec.width, device, **kwargs)
