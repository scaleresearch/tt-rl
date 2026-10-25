"""How long one episode can run, asked of a constructed environment.

An environment whose episodes have no bound cannot be evaluated: evaluation
runs inline in the training loop, so an episode that does not end stalls the
whole run. `ttrl.train.evaluate` carries a step guard for that case, and a
guard is a way to notice six hours in. The question "can this environment's
episodes end on their own?" is answerable the moment the environment exists,
which is where `require_episode_bound` asks it. Both callers that need
the answer come here, `ttrl.check_env` and
`ttrl.train.builders.require_episode_bound`, so the public checker and
the run ask one question and read one message.

It is asked of the environment and not of its `EnvConfig`, because a config
carries only a builder's keywords: the bound is installed by a wrapper or by
the simulator itself, and which of those happened is a property of the stack
the builder returns.

A layer declares its bound as `episode_bound`: an upper bound on the number
of ITS OWN steps one episode can take. A bound declared below an action
repeat counts inner steps and still bounds the outer ones, so the tightest
number any layer declares is a valid bound for the whole stack, and that is
what `episode_bound` returns.
"""

from __future__ import annotations

from ttrl.envs.wrappers.core import layers

__all__ = ["EPISODE_BOUND_ATTR", "episode_bound", "require_episode_bound"]

#: The attribute a layer declares its own episode bound with.
EPISODE_BOUND_ATTR = "episode_bound"


def _declared(layer) -> int | None:
    """The bound `layer` declares, or None if it declares none."""
    value = getattr(layer, EPISODE_BOUND_ATTR, None)
    if value is None:
        # A Gymnasium environment states its limit on its own spec, and
        # `GymAdapter` over one is the layer that knows the spec is there.
        value = getattr(
            getattr(getattr(layer, "gym_env", None), "spec", None),
            "max_episode_steps",
            None,
        )
    if value is None:
        return None
    value = int(value)
    return value if value >= 1 else None


def episode_bound(env) -> int | None:
    """The tightest bound any layer of `env` declares, or None if none does.

    None means the episodes of this environment end only when the environment
    decides to end them, which no caller can put a number on.
    """
    bounds = [b for b in (_declared(layer) for layer in layers(env))
              if b is not None]
    return min(bounds) if bounds else None


def require_episode_bound(env, *, where: str) -> int:
    """`episode_bound(env)`, refusing an environment that declares none.

    THE one refusal for an unbounded environment, and the one message. It is
    asked of a constructed environment, which is what lets both the public
    checker (`ttrl.check_env`) and the run (`ttrl.train.builders`) ask the
    same question of the same thing: a checker that passed what a run refuses
    sends an author to train a configuration that cannot start.

    The message names declaring the attribute FIRST, because an author who
    wrote their own `Env` class fixes this in one line on that class, and the
    wrapper is the fix only for a stack they do not own.
    """
    bound = episode_bound(env)
    if bound is not None:
        return bound
    stack = list(layers(env))
    inner = type(stack[-1]).__name__
    chain = " -> ".join(type(layer).__name__ for layer in stack)
    raise ValueError(
        f"{where} has unbounded episodes: nothing in {chain} declares an "
        f"`{EPISODE_BOUND_ATTR}`, so nothing but the environment itself "
        f"decides when an episode ends. Evaluation runs inside the training "
        f"loop, so one episode that does not end costs the whole run, and "
        f"ttrl.run refuses such an environment before it opens a chip. If you "
        f"wrote {inner}, that is one line on it: declare "
        f"`{EPISODE_BOUND_ATTR} = <the most steps one episode can take>` as a "
        f"class attribute or a property. For a stack you do not own, wrap it: "
        f"ttrl.envs.wrappers.TimeLimit(env, max_steps=<n>), which declares the "
        f"bound for you. The Atari suite declares the ALE protocol's limit "
        f"that way, as AtariConfig.max_frames=108000 through "
        f"env.kwargs.max_frames. An environment that is deliberately endless "
        f"can only be TRAINED on: give TrainConfig.eval_env a bounded "
        f"environment to score on, and hold this one to the contract with "
        f"check_env(env, bounded=False)."
    )
