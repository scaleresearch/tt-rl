"""The legal-action mask on the agent side: one switch, two places it lands.

Where the mask lives is settled in `ttrl.spaces`: `ObsSpec.vector(dim,
action_mask=n)` declares that the final n entries of that leaf are a {0, 1}
legal-action mask, an `ObsGroup` admits the mask on one of its leaves, and
`ttrl.spaces.group_mask_rows` reads it back off whichever leaf that is. This
module is what an algorithm needs on top of that.

`mask_width` is the single switch. An agent reads it once, at construction,
and both halves of masking hang off that one integer: the action selection in
`act` and the bootstrap inside the update body. There is no second flag to
set, so an agent cannot mask its greedy choice and leave its target maxing
over moves it cannot make. That half-measure is the trap: the policy looks
fixed from the outside while the loss keeps teaching the network the value of
illegal actions.

Masking logits means pushing the illegal positions below every legal one,
which is an add of a negative constant and needs no `where`, no `softmax` and
no scatter. `PENALTY` is finite on purpose. Through `log_softmax` an illegal
entry becomes about -PENALTY, `exp` of that is exactly 0.0 in fp32, and the
entropy term `exp(logp) * logp` is therefore 0.0 * -PENALTY = 0.0. An
infinite penalty makes that same term 0 * -inf, which is NaN. 1.0e4 also
leaves the gathered log-prob of a legal action untouched, since fp32 `exp`
underflows to zero below about -87 and the Q and logit magnitudes these
networks produce are single digits.
"""

from __future__ import annotations

import numpy as np

from ttrl.engine.autograd import Tensor
from ttrl.engine.ops import add, mul, sub
from ttrl.spaces import ActionSpec, ObsGroup, draw_legal, group_mask_rows

__all__ = [
    "PENALTY",
    "epsilon_greedy_masked",
    "mask_penalty",
    "mask_width",
    "obs_mask",
    "masked_logits",
]

#: Added at the illegal positions of a row of logits or Q values.
PENALTY = 1.0e4


def mask_width(obs_spec: ObsGroup, action_spec: ActionSpec, algo: str) -> int:
    """How many actions this agent must mask, 0 for an unmasked environment.

    The one place the two specs are checked against each other, so a mask that
    does not describe the action set is refused before a chip is opened.
    """
    n = int(obs_spec.action_mask)
    if n == 0:
        return 0
    if action_spec.kind == "multi_discrete":
        joint = 1
        for size in action_spec.nvec:
            joint *= int(size)
        raise ValueError(
            f"{algo}: the ObsSpec declares a legal-action mask of width {n} but the "
            f"ActionSpec is multi_discrete (nvec={action_spec.nvec}). A mask names "
            f"which members of an action set may be played, and the members of a "
            f"factorized action set are its {joint} joint actions, not the "
            f"{int(sum(action_spec.nvec))} per-factor values: a mask over the factors "
            f"can only say that a value is illegal for EVERY setting of the other "
            f"factors, so a bet-size bucket that is legal only when the action type "
            f"is 'raise' has no spelling in it, and a factorized policy sampling each "
            f"factor independently would play the illegal pair. Masking a composite "
            f"action means enumerating it: build the environment with "
            f"ActionSpec.discrete({joint}) and action_mask={joint}, decoding the index "
            f"into factors inside the environment. Or drop action_mask from the "
            f"ObsSpec and keep the multi_discrete spec unmasked."
        )
    if action_spec.kind != "discrete":
        raise ValueError(
            f"{algo}: the ObsSpec declares a legal-action mask of width {n} but the "
            f"ActionSpec is continuous (dim={action_spec.dim}). A mask names which "
            f"members of a finite action set may be played, and a continuous action "
            f"vector has no members to name: restricting it means moving the low and "
            f"high bounds the ActionSpec already carries. Drop action_mask from the "
            f"ObsSpec, or use 'dqn' or 'ppo' over a discrete ActionSpec."
        )
    if n != int(action_spec.n):
        raise ValueError(
            f"{algo}: the ObsSpec declares a legal-action mask of width {n} but the "
            f"ActionSpec has n={action_spec.n}. The mask holds one entry per action, "
            f"so build the ObsSpec with action_mask={action_spec.n}."
        )
    return n


def epsilon_greedy_masked(q: np.ndarray, mask: np.ndarray, eps: float, rng) -> np.ndarray:
    """Epsilon-greedy over the legal columns only.

    Both halves are masked. A greedy choice taken over all columns and then
    repaired, or an exploring draw over all actions, each put an illegal action
    on the wire, which is what forces an environment to coerce.

    `draw_legal` is the same uniform-over-legal-columns draw `check_env` uses,
    so the checker and the agent explore the same action set. `obs_mask` has
    already refused a row with no legal action.
    """
    legal = np.asarray(mask) > 0.5
    greedy = np.argmax(np.where(legal, q, -np.inf), axis=-1)
    draws = rng.random(legal.shape[0])
    explore = draw_legal(rng, legal)
    return np.where(draws < eps, explore, greedy).astype(np.int64)


def mask_penalty(mask: Tensor) -> Tensor:
    """`PENALTY` at this device mask's illegal positions and 0 elsewhere.

    For a mask bound as a step input, so the whole thing stays inside a
    captured trace: two elementwise ops against an already-resident tensor.
    """
    return mul(sub(mask, 1.0), PENALTY)


def masked_logits(logits: Tensor, mask) -> Tensor:
    """`logits` with every illegal column pushed below every legal one.

    The eager spelling, for a host mask and a forward pass outside a trace.
    Rows past the end of `mask` are the acting buffer's padding and keep their
    unmasked values, which nothing reads.
    """
    rows, width = int(logits.shape[0]), int(logits.shape[-1])
    host = np.asarray(mask, np.float32)
    full = np.ones((rows, width), np.float32)
    full[: host.shape[0]] = host
    penalty = Tensor.from_numpy((full - 1.0) * PENALTY, logits.device,
                                requires_grad=False)
    return add(logits, penalty)


def obs_mask(obs, obs_spec: ObsGroup, width: int):
    """The mask of a state about to be acted on, or None when nothing is masked.

    The whole unmasked cost of masking on an acting step: one comparison of an
    integer against zero.

    The one place a state with no legal action is refused, for every algorithm
    that masks. A value-based agent and a policy-based one fail differently on
    such a row and neither failure names it: an argmax over uniformly
    penalised Q values returns column 0, and a softmax over uniformly
    penalised logits is uniform over illegal actions. Both then hand an
    illegal action to an environment, whose own correct refusal reads as a
    library bug. It is checked here rather than in `mask_rows`, because the
    successor mask a bootstrap reads is allowed to be empty: a terminal state
    has no move, and its value is zeroed by `dones` regardless.
    """
    if not width:
        return None
    mask = group_mask_rows(obs, obs_spec)
    empty = np.flatnonzero(~(mask > 0.5).any(axis=-1))
    if empty.size:
        raise ValueError(
            f"act: the observation in row {int(empty[0])} declares every action "
            f"illegal, so there is nothing to choose. An observation an agent is "
            f"asked to act on must leave at least one action legal; end the episode "
            f"instead of presenting a state with no move."
        )
    return mask
