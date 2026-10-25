"""SAC's gradient step as two capturable regions.

Separated from the agent for the reason `ttrl.algos.sac.losses` is: the agent
owns the buffers, the clock and the optimizers, and this owns the shape of one
update. The split into two regions is forced by the algorithm rather than
chosen: the actor's loss reads the critics, so the critic optimizer has to run
between the two, and an optimizer never goes inside a trace (see
`ttrl.engine.step.TrainStep`).

Every metric leaves here as a device tensor. Nothing in a region may read a
tensor back to host, because trace capture forbids it, which is why the
gradient norms are `grad_sq_device` rather than `grad_norm`.
"""

from __future__ import annotations

import numpy as np

from ttrl.algos.grads import grad_sq_device
from ttrl.algos.sac.losses import (actor_loss, alpha_loss, critic_loss,
                                   priority_errors)
from ttrl.engine.autograd import Tensor, no_grad
from ttrl.engine.ops import exp

__all__ = ["RegionOptimizers", "actor_region", "critic_region", "summarize"]


class RegionOptimizers:
    """The optimizers one region zeroes and the ones it steps.

    They differ: the actor's loss differentiates through the critics, so its
    backward leaves gradients on critic parameters that the next critic step
    must not inherit. Every region therefore clears all three and steps only
    the parameters it trained.
    """

    def __init__(self, zeroed, stepped):
        self._zeroed = list(zeroed)
        self._stepped = list(stepped)
        self.params = [p for opt in self._stepped for p in opt.params]

    def zero_grad(self) -> None:
        for opt in self._zeroed:
            opt.zero_grad()

    def step(self) -> None:
        for opt in self._stepped:
            opt.step()


def _alpha(log_alpha: Tensor) -> Tensor:
    """The temperature, detached: it enters both losses as a constant."""
    with no_grad():
        return exp(log_alpha)


def _grad_sq(metrics: dict, key: str, params) -> None:
    total = grad_sq_device(params)
    if total is not None:
        metrics[key] = Tensor(total, requires_grad=False)


def critic_region(nets, dist, batch, log_alpha: Tensor, *,
                  prioritized: bool) -> dict:
    """Critic forward, twin-critic loss, backward. -> device metrics."""
    alpha = _alpha(log_alpha)
    weights = batch["is_weights"] if prioritized else None
    loss, target, errors = critic_loss(nets, dist, batch, alpha, weights)
    loss.backward()
    metrics = {"critic_loss": loss, "alpha": alpha, "q_target": target.detach()}
    _grad_sq(metrics, "critic_grad_sq", nets.critic_parameters())
    if prioritized:
        metrics["td_abs"] = priority_errors(errors)
    return metrics


def actor_region(nets, dist, batch, log_alpha: Tensor, target_entropy: float,
                 *, learn_alpha: bool) -> dict:
    """Actor forward, loss, backward, and the temperature's dual with it.

    The temperature shares this region because its loss reads a detached
    `logp` and puts gradient on `log_alpha` alone, so the two backwards do not
    meet and one region captures both. Running the actor's gradient norm
    before the temperature's backward or after it is the same measurement for
    that reason.
    """
    alpha = _alpha(log_alpha)
    loss, logp = actor_loss(nets, dist, batch, alpha)
    loss.backward()
    metrics = {"actor_loss": loss, "logp": logp.detach()}
    _grad_sq(metrics, "actor_grad_sq", nets.policy_parameters())
    if learn_alpha:
        dual = alpha_loss(log_alpha, logp, target_entropy)
        dual.backward()
        metrics["alpha_loss"] = dual
    return metrics


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _norm(grad_sq) -> float:
    """The gradient norm from the device-side sum of squares.

    fp32 on device where the host-read `grad_norm` accumulated in float64, so
    the same gradients log a slightly different number than they did before
    the regions became capturable. It is the only spelling a captured region
    can use, and DQN has logged its norm this way since it gained one.
    """
    total = _scalar(grad_sq) if grad_sq is not None else 0.0
    return float(np.sqrt(max(total, 0.0)))


def summarize(critic: dict, actor: dict, *, learn_alpha: bool) -> dict[str, float]:
    """One update's host-side metrics from the two regions' readbacks.

    `entropy` and `mean_q_target` are means over per-sample metrics taken on
    the host in float64, which is where they were taken before.
    """
    return {
        "critic_loss": _scalar(critic["critic_loss"]),
        "actor_loss": _scalar(actor["actor_loss"]),
        "alpha_loss": _scalar(actor["alpha_loss"]) if learn_alpha else 0.0,
        "alpha": _scalar(critic["alpha"]),
        "entropy": float(-np.mean(actor["logp"])),
        "mean_q_target": float(np.mean(critic["q_target"])),
        "critic_grad_norm": _norm(critic.get("critic_grad_sq")),
        "actor_grad_norm": _norm(actor.get("actor_grad_sq")),
    }
