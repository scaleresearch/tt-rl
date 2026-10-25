"""SAC's three losses, as functions of the networks and one loaded batch.

Separated from the agent because they are the algorithm and the rest is
bookkeeping: the agent owns the buffers, the optimizers and the clock, and
these own the arithmetic. Each is composed from primitive ops; no distribution
primitive was added to the engine for any of them.
"""

from __future__ import annotations

from functools import reduce

from ttrl.engine.autograd import Tensor, no_grad
from ttrl.engine.autograd import no_grad as _no_grad
from ttrl.engine.ops import (abs_, add, apply_done_mask, mean_all, minimum, mul,
                             square, sub)

__all__ = ["actor_loss", "alpha_loss", "critic_loss", "min_q", "priority_errors"]

#: `batch.obs` and `batch.next_obs` are `ObsPair`s: the policy reads the
#: actor's half and the critics read the whole observation, each because of
#: the spec it was built over rather than because of anything chosen here.


def min_q(nets, obs, action: Tensor) -> Tensor:
    """Minimum over the critics, which is the overestimation correction."""
    return reduce(minimum, [net(obs, action) for net in nets])


def priority_errors(errors) -> Tensor:
    """The per-sample TD magnitude a prioritized run stores, over the critics.

    The MEAN of the critics' absolute errors rather than one critic's: the
    priority is a statement about how surprising a transition is to the
    agent, and the agent is both critics. Detached, and returned as a device
    tensor so the host read happens with the step's other metrics.
    """
    with _no_grad():
        total = None
        for error in errors:
            magnitude = abs_(error.detach())
            total = magnitude if total is None else add(total, magnitude)
        return mul(total, 1.0 / len(errors))


def critic_loss(nets, dist, batch, alpha: Tensor, weights: Tensor | None = None):
    """Twin-critic MSE against the soft n-step target. -> (loss, target).

    `weights` is the batch's per-sample importance-sampling weights for a
    prioritized run, applied per sample BEFORE the reduction for the reason
    `ttrl.algos.dqn.losses` states.

    The target half runs under `no_grad`: the next action, its log-prob and
    the lagged critics' values are all constants of this step, and a tape
    through them would send gradient into the policy from the critic's loss.
    """
    with no_grad():
        mean, log_std = nets.policy(batch.next_obs)
        next_action, next_logp = dist.sample(mean, log_std, batch["next_noise"])
        soft_value = sub(min_q(nets.targets, batch.next_obs, next_action),
                         mul(next_logp, alpha))
        # `discounts` is gamma ** (rewards actually summed), so a window an
        # episode boundary cut short bootstraps with its own discount.
        target = add(batch["rewards"],
                     mul(apply_done_mask(soft_value, batch["dones"]),
                         batch["discounts"]))

    errors = [sub(critic(batch.obs, batch["actions"]), target)
              for critic in nets.critics]
    squared = [square(error) for error in errors]
    if weights is not None:
        squared = [mul(term, weights) for term in squared]
    # 0.5 * sum over critics, the SB3 convention: each critic sees the same
    # gradient it would have seen trained alone.
    loss = reduce(add, [mean_all(term) for term in squared])
    return mul(loss, 0.5), target, errors


def actor_loss(nets, dist, batch, alpha: Tensor):
    """E[alpha * log pi(a|s) - min_i Q_i(s, a)], a reparameterised.
    -> (loss, logp)."""
    mean, log_std = nets.policy(batch.obs)
    action, logp = dist.sample(mean, log_std, batch["noise"])
    return mean_all(sub(mul(logp, alpha), min_q(nets.critics, batch.obs, action))), logp


def alpha_loss(log_alpha: Tensor, logp: Tensor, target_entropy: float) -> Tensor:
    """-E[log alpha * (log pi + target_entropy)], the dual of the entropy
    constraint.

    `logp` is detached: the temperature responds to how much entropy the
    policy currently has, and must not push the policy.
    """
    return mul(mean_all(mul(log_alpha, add(logp.detach(), target_entropy))), -1.0)
