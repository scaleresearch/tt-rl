"""PPO's objective, parameterised by how it reduces.

One copy of the clipped surrogate, the clipped value loss and the entropy
bonus, serving both the feedforward update and the recurrent one. They differ
in the reduction and nowhere else: a feedforward minibatch holds only valid
transitions, because the flat path selects them before it shuffles, while a
recurrent minibatch keeps its autoreset padding slots, because dropping one
would shorten the sequence and so change the graph shape. So the padding is
excluded by the reduction instead.

Two copies of this algebra would be two places for a clip bound or the 0.5 on
the value term to drift, and the recurrent copy is the one nobody would
notice, since it is the path with no torch reference.
"""

from __future__ import annotations

from typing import Callable

from ttrl.engine.autograd import Tensor
from ttrl.engine.ops import (add, clamp, div, exp, maximum, minimum, mul,
                             sub, sum_all)

__all__ = ["masked_mean", "objective"]


def masked_mean(valid: Tensor) -> Callable[[Tensor], Tensor]:
    """A reducer averaging over the rows `valid` marks, device-side.

    The denominator is `sum_all(valid)` rather than a host integer, so a
    captured trace does not freeze one minibatch's valid count and reapply it
    to every later replay.
    """
    total = sum_all(valid)

    def reduce(value: Tensor) -> Tensor:
        return div(sum_all(mul(value, valid)), total)

    return reduce


def objective(policy, policy_out, value: Tensor, batch, cfg, device, *,
              reduce: Callable[[Tensor], Tensor]):
    """`(loss, parts, new_logp)` for one minibatch.

    `parts` holds the three terms as Tensors, for the caller's metrics.
    `new_logp` is returned because the approximate KL is computed from it on
    the host after the backward, and recomputing it would be a second forward.
    """
    rows = int(batch.old_logp.shape[0])
    new_logp, entropy_per_row = policy.logp_entropy(
        policy_out, batch.actions, mask=batch.mask)

    old_logp = Tensor.from_numpy(batch.old_logp.reshape(rows, 1), device)
    advantages = Tensor.from_numpy(batch.advantages.reshape(rows, 1), device)
    returns = Tensor.from_numpy(batch.returns.reshape(rows, 1), device)
    old_value = Tensor.from_numpy(batch.values.reshape(rows, 1), device)

    ratio = exp(sub(new_logp, old_logp))
    surrogate = minimum(
        mul(ratio, advantages),
        mul(clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps), advantages),
    )
    policy_loss = mul(reduce(surrogate), -1.0)

    # PPO2 value clipping: the pessimistic (larger) of the clipped and
    # unclipped squared errors, so the gradient follows whichever term is
    # larger here rather than one picked in advance.
    clipped_value = add(old_value, clamp(sub(value, old_value),
                                         -cfg.value_clip_eps, cfg.value_clip_eps))
    unclipped_err = sub(value, returns)
    clipped_err = sub(clipped_value, returns)
    # The 0.5 is the SB3/CleanRL convention; without it value_coef=0.5 weights
    # the critic term twice as heavily as every published config.
    value_loss = mul(reduce(maximum(mul(unclipped_err, unclipped_err),
                                    mul(clipped_err, clipped_err))), 0.5)
    entropy_bonus = reduce(entropy_per_row)

    loss = add(add(policy_loss, mul(value_loss, cfg.value_coef)),
               mul(entropy_bonus, -cfg.entropy_coef))
    parts = {"policy_loss": policy_loss, "value_loss": value_loss,
             "entropy": entropy_bonus}
    return loss, parts, new_logp
