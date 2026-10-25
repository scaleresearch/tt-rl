"""The contract's refusals: the executable half of `ttrl.algos.base`.

`base.py` declares what an agent is. This module raises when something is not
one, and every refusal here fires at CONSTRUCTION, before a chip is opened or
a network built, because each of these mistakes is otherwise discovered from
inside a running loop a thousand frames in.

Two kinds. `require_discrete` and `require_continuous` are asked by an
algorithm about the action space it was handed: the refusal names the
algorithms that do cover that space, so the reader is not left to guess.
`check_agent` is asked by `ttrl.train.builders` about the object a registered
builder returned: `Agent` is a `Protocol`, so a missing method is no longer
caught at instantiation and nothing else would notice until the trainer
called it.

Imported through `ttrl.algos.base`, which re-exports all three.
"""

from __future__ import annotations

import inspect
from typing import Any

from ttrl.spaces import ActionSpec, ObsGroup

__all__ = ["check_agent", "require_continuous", "require_discrete"]


def require_discrete(action_spec: ActionSpec, algo: str) -> None:
    if action_spec.kind == "multi_discrete":
        raise ValueError(
            f"{algo} supports plain discrete actions only, got a multi_discrete "
            f"ActionSpec with nvec={action_spec.nvec}. {algo} selects an action by "
            f"maximising one Q value per action, and a factorized action set holds "
            f"{_product(action_spec.nvec)} joint actions, which grows multiplicatively "
            f"with the factors; a per-factor argmax is a different algorithm and not "
            f"this one. Use 'ppo', whose policy factorizes over the factors, or "
            f"enumerate the joint actions yourself as "
            f"ActionSpec.discrete({_product(action_spec.nvec)})."
        )
    if action_spec.kind != "discrete":
        raise ValueError(
            f"{algo} supports discrete actions only, got a continuous ActionSpec "
            f"of dim={action_spec.dim}. {algo} selects actions by maximising over "
            f"a finite action set, which a continuous space does not have. Use "
            f"'sac' for continuous control, or 'ppo', which supports both kinds."
        )


def require_continuous(action_spec: ActionSpec, algo: str) -> None:
    if action_spec.kind != "continuous":
        described = (f"multi_discrete ActionSpec with nvec={action_spec.nvec}"
                     if action_spec.kind == "multi_discrete"
                     else f"discrete ActionSpec with n={action_spec.n}")
        raise ValueError(
            f"{algo} supports continuous actions only, got a {described}. "
            f"{algo}'s policy is a squashed Gaussian over a bounded action vector, "
            f"which has no discrete spelling. Use 'dqn' for value-based discrete "
            f"control, or 'ppo', which covers continuous, discrete and "
            f"multi_discrete actions."
        )


def _product(nvec) -> int:
    total = 1
    for size in nvec:
        total *= int(size)
    return total


def check_agent(agent: Any) -> None:
    """Assert that `agent` carries the whole `Agent` surface.

    `Protocol` checks membership at runtime but not signatures, and a trainer
    that discovers a missing `should_update` a thousand frames in has already
    thrown away the run.
    """
    for name in ("obs_spec", "action_spec"):
        if not isinstance(getattr(agent, name, None), (ObsGroup, ActionSpec)):
            raise TypeError(
                f"{type(agent).__name__} does not expose {name}; the trainer reads "
                f"both specs off the agent to size its buffers and its evaluation"
            )
    for name in ("act", "observe", "update", "should_update", "named_parameters",
                 "parameters", "state_dict", "load_state_dict", "close"):
        if not callable(getattr(agent, name, None)):
            raise TypeError(
                f"{type(agent).__name__} is missing the Agent method {name!r}; "
                f"the trainer calls every one of act, observe, update, "
                f"should_update, named_parameters, parameters, state_dict, "
                f"load_state_dict and close. named_parameters and parameters "
                f"are what exact parameter averaging across replicas walks, so "
                f"an agent without them cannot run on more than one chip."
            )
    # Last: a missing method is the more basic failure and should be the one
    # named, so the signature check runs only once every method is there.
    _check_act_signature(agent)


def _check_act_signature(agent: Any) -> None:
    """Refuse an `act` that cannot be told where episodes start.

    `episode_starts` is part of the acting contract because a policy with
    memory has no other way to learn of an episode boundary while it is being
    evaluated: it is handed observations and returns actions, and no
    transition reaches it. A feedforward agent has nothing to reset by the
    flags and needs only to accept them, which is one default argument.

    Checked here, at construction, because the alternative is a TypeError
    thrown from inside the evaluation loop a thousand frames into a run, which
    is where this was found.
    """

    act = getattr(agent, "act", None)
    if not callable(act):
        return
    try:
        params = inspect.signature(act).parameters
    except (TypeError, ValueError):
        return
    if "episode_starts" in params:
        return
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    raise TypeError(
        f"{type(agent).__name__}.act does not accept episode_starts, which "
        f"ttrl.train.evaluate and ttrl.train.record pass on every "
        f"deterministic acting call. Add `episode_starts=None` to the "
        f"signature: a policy with no memory has nothing to reset by it, and "
        f"one with memory resets its hidden state where the flags are set."
    )
