"""PPO's three action distributions, behind one interface.

`ActionSpec.kind` is read once, in `build_policy`, and nowhere else: the agent
holds whichever of these it was given and calls the same two methods on it.
That is what lets one PPO implementation cover a categorical policy over a
finite action set, a product of categoricals over a factorized one, and a
diagonal Gaussian over a bounded action vector.

Each distribution splits into a host half used while acting (the sample and
the log-prob recorded for the importance ratio) and a device half used while
learning (the log-prob and entropy that carry a gradient).
"""

from __future__ import annotations

import numpy as np

from ttrl.algos.gaussian import gaussian_entropy, normal_logp, normal_logp_numpy
from ttrl.algos.mask import masked_logits
from ttrl.engine.autograd import Tensor
from ttrl.engine.ops import add, entropy, log_softmax, one_hot_gather, split
from ttrl.spaces import ActionSpec

__all__ = ["CategoricalPolicy", "FactoredCategoricalPolicy", "GaussianPolicy",
           "build_policy"]


class CategoricalPolicy:
    """Softmax over `n` logits. The policy head emits the logits."""

    def __init__(self, action_spec: ActionSpec, device):
        self.action_spec = action_spec
        self.device = device
        self.n = int(action_spec.n)

    def sample(self, policy_out, rng, batch: int, deterministic: bool, mask=None):
        """-> (env_action, stored_action, logp), all host arrays of length batch.

        `mask` pushes the illegal logits below every legal one before the
        log-softmax, so the inverse-CDF draw cannot land on one and the
        recorded log-prob is the masked policy's own.

        The sampler is an explicit inverse-CDF draw against one uniform per
        lane, so the whole policy is reproducible from the stored generator
        state. `log_softmax` is computed on device and brought down, rather
        than recomputed in numpy, so the log-prob recorded here and the one
        the update measures the ratio against are the same function.
        """
        if mask is not None:
            policy_out = masked_logits(policy_out, mask)
        logp_all = log_softmax(policy_out, dim=-1).numpy()[:batch]
        if deterministic:
            actions = np.argmax(logp_all, axis=-1)
        else:
            probs = np.exp(logp_all)
            probs = probs / probs.sum(axis=-1, keepdims=True)
            cdf = np.cumsum(probs, axis=-1)
            actions = (rng.random(batch)[:, None] > cdf).sum(axis=-1)
            actions = np.clip(actions, 0, self.n - 1)
        actions = actions.astype(np.int64)
        logp = logp_all[np.arange(batch), actions].astype(np.float32)
        return actions, actions, logp

    def logp_entropy(self, policy_out, actions: np.ndarray, mask=None):
        """-> (logp (B, 1), entropy (B, 1)), both differentiable.

        The entropy is taken over the SAME masked logits as the log-prob.
        Taken over the raw head it is an entropy of a distribution the agent
        never samples, and with a positive `entropy_coef` the shipped config
        then pays the policy to put probability mass on illegal moves.
        """
        if mask is not None:
            policy_out = masked_logits(policy_out, mask)
        logp_all = log_softmax(policy_out, dim=-1)
        logp = one_hot_gather(logp_all, actions.reshape(-1).astype(np.int64), self.n)
        return logp, entropy(policy_out, dim=-1)


class FactoredCategoricalPolicy:
    """One independent categorical per factor of a composite action.

    The head emits `sum(nvec)` logits and this cuts them into one segment per
    factor. The factors are independent given the observation, so the joint
    log-prob is the sum of the per-factor log-probs and the joint entropy is
    the sum of the per-factor entropies; both are exact for a product
    distribution rather than a bound. An action is a vector of `len(nvec)`
    indices, and the stored action is the action the environment receives.

    The segment sizes come from the spec, so the cut is a build-time constant:
    no host read and no rebinding inside a step.
    """

    def __init__(self, action_spec: ActionSpec, device):
        self.action_spec = action_spec
        self.device = device
        self.nvec = tuple(int(size) for size in action_spec.nvec)

    def sample(self, policy_out, rng, batch: int, deterministic: bool, mask=None):
        """-> (env_action, stored_action, logp); actions are (batch, factors).

        `mask` is always None here: `mask_width` refuses a legal-action mask
        over a multi_discrete ActionSpec, because a mask over the factors
        cannot name an illegal COMBINATION of legal values.

        Each factor is drawn by the same explicit inverse-CDF draw the
        single-factor policy uses, one uniform per factor per lane, so the
        whole policy stays reproducible from the stored generator state.
        """
        rows = np.arange(batch)
        columns = []
        logp = np.zeros(batch, dtype=np.float32)
        for size, piece in zip(self.nvec, split(policy_out, self.nvec, dim=-1)):
            logp_all = log_softmax(piece, dim=-1).numpy()[:batch]
            if deterministic:
                picks = np.argmax(logp_all, axis=-1)
            else:
                probs = np.exp(logp_all)
                probs = probs / probs.sum(axis=-1, keepdims=True)
                cdf = np.cumsum(probs, axis=-1)
                picks = (rng.random(batch)[:, None] > cdf).sum(axis=-1)
                picks = np.clip(picks, 0, size - 1)
            picks = picks.astype(np.int64)
            columns.append(picks)
            logp = logp + logp_all[rows, picks].astype(np.float32)
        actions = np.stack(columns, axis=-1).astype(np.int64)
        return actions, actions, logp

    def logp_entropy(self, policy_out, actions: np.ndarray, mask=None):
        """-> (logp (B, 1), entropy (B, 1)), both differentiable.

        Summed over the factors, which is the joint of the product
        distribution `sample` draws from.
        """
        taken = np.asarray(actions, np.int64).reshape(-1, len(self.nvec))
        logp = None
        total_entropy = None
        pieces = split(policy_out, self.nvec, dim=-1)
        for index, (size, piece) in enumerate(zip(self.nvec, pieces)):
            factor_logp = one_hot_gather(
                log_softmax(piece, dim=-1), taken[:, index], size)
            factor_entropy = entropy(piece, dim=-1)
            logp = factor_logp if logp is None else add(logp, factor_logp)
            total_entropy = (factor_entropy if total_entropy is None
                             else add(total_entropy, factor_entropy))
        return logp, total_entropy


class GaussianPolicy:
    """Diagonal Gaussian over `dim` action scalars, unsquashed.

    The action stored for the update is the raw Gaussian sample and the action
    the environment receives is that sample clipped to the bounds. Those are
    deliberately different: the density of a clipped sample is not the
    Gaussian's, so measuring the importance ratio against the clipped value
    would make the ratio of a saturated action a function of the clip rather
    than of the policy.
    """

    def __init__(self, action_spec: ActionSpec, device):
        self.action_spec = action_spec
        self.device = device
        self.dim = int(action_spec.dim)
        self.low = action_spec.low_array
        self.high = action_spec.high_array

    def sample(self, policy_out, rng, batch: int, deterministic: bool, mask=None):
        mean_t, log_std_t = policy_out
        mean = mean_t.numpy()[:batch]
        log_std = log_std_t.numpy()[:batch]
        if deterministic:
            raw = mean
        else:
            raw = mean + np.exp(log_std) * rng.standard_normal(mean.shape)
        raw = raw.astype(np.float32)
        logp = normal_logp_numpy(raw, mean, log_std)
        env_action = np.clip(raw, self.low, self.high).astype(np.float32)
        return env_action, raw, logp

    def logp_entropy(self, policy_out, actions: np.ndarray, mask=None):
        # Always None here: `mask_width` refuses a mask over a continuous
        # ActionSpec, so a Gaussian policy never receives one.
        mean, log_std = policy_out
        taken = Tensor.from_numpy(
            np.asarray(actions, np.float32).reshape(-1, self.dim), mean.device)
        return normal_logp(taken, mean, log_std), gaussian_entropy(log_std)


def build_policy(action_spec: ActionSpec, device):
    """The distribution an `ActionSpec` calls for.

    A mapping rather than a chain of ifs, so a kind nobody wrote a
    distribution for is refused by name instead of falling through to the
    last branch and being sampled as something it is not.
    """
    policies = {
        "discrete": CategoricalPolicy,
        "multi_discrete": FactoredCategoricalPolicy,
        "continuous": GaussianPolicy,
    }
    if action_spec.kind not in policies:
        raise ValueError(
            f"ppo has no action distribution for an ActionSpec of kind "
            f"{action_spec.kind!r}; the kinds it samples are {sorted(policies)}. "
            f"See ttrl.algos.ppo.policy."
        )
    return policies[action_spec.kind](action_spec, device)
