"""Wrapper base class and the explicit state-snapshot protocol.

A wrapper that holds state the underlying simulator does not (a frame stack,
a running normaliser, a life counter) must declare it by implementing
`get_state` and `set_state`. Checkpointing walks the stack and asks each
layer, so adding a wrapper never requires editing a table of known wrapper
names, and an unrecognised layer is simply a layer with no state.

Policy state. Some of what a layer holds is part of the function the weights
compute: a running observation normaliser's mean and variance are applied to
every observation the network ever saw, so weights shipped without them
receive inputs on a different scale and the policy is dead. The rest buys a
bit-exact continuation of one trajectory and nothing else. A stateful layer
says which of its own `get_state` keys are the first kind by declaring them
in a class attribute:

    class NormalizeObservation(VecWrapper):
        POLICY_STATE = ("mean", "m2", "count")

    class TimeLimit(Wrapper):
        POLICY_STATE = ()

`RUN_STATE` is the third answer, declared by the vector-level layers of
`ttrl.envs.vec_wrapper`: cross-episode statistics a resume must continue that
are NOT part of what the weights compute. A reward normaliser's running return
scale is the case it exists for. Putting it in `POLICY_STATE` would oblige
every exported policy artifact to carry a transform it never applies, since a
deployed policy consumes no reward; leaving it out of both would restart the
reward scale on resume and change the size of every gradient a run reports
itself as continuing. The checkpoint keeps it in a separate, L1-only member.

The empty tuple is the answer for a layer whose state is trajectory state,
and it is an answer rather than a default: `snapshot_policy_state` refuses a
stateful layer that declares nothing, naming the layer and both spellings, so
a wrapper nobody classified cannot be silently dropped from an exported
policy or silently carried into one. Declaring keys rather than a kind is
what lets one layer hold both kinds without being split, and it is checkable:
a key that `get_state` does not return is refused too.
"""

from __future__ import annotations

import importlib
from typing import Any, Iterator, Protocol, runtime_checkable

import numpy as np

from ttrl.envs.render import Renderable, supports_render
from ttrl.spaces import ActionSpec, ObsSpec

__all__ = [
    "Wrapper",
    "layers",
    "snapshot_policy_state",
    "restore_policy_state",
    "StatefulEnv",
    "snapshot_env",
    "restore_env",
    "require_module",
]

#: What each declaration is about, quoted back in the refusal below.
#: `POLICY_STATE` is what the weights cannot be read without and travels into
#: an exported artifact; `RUN_STATE` is cross-episode statistics a resume must
#: continue that are NOT part of what the weights compute. A reward
#: normaliser's running return scale is the second kind, and putting it in the
#: first would oblige every deployed policy artifact to carry a transform it
#: never applies.
_MEANING = {
    "POLICY_STATE": (
        "the policy, meaning the state the weights cannot be used without, "
        "such as a running observation normaliser's statistics"
    ),
    "RUN_STATE": (
        "the run, meaning cross-episode statistics a resume must continue "
        "that are not part of what the weights compute, such as a reward "
        "normaliser's running return scale"
    ),
}


def require_module(module: str, extra: str):
    """Import an optional dependency, naming the install command if absent."""

    top = module.split(".")[0]
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != top:
            raise
        raise ModuleNotFoundError(
            f"{module!r} is required here and is not installed. "
            f"Install it with: pip install 'tt-rl[{extra}]'"
        ) from exc


@runtime_checkable
class StatefulEnv(Protocol):
    """Declares that a layer owns state a checkpoint must carry."""

    def get_state(self) -> dict[str, Any]: ...

    def set_state(self, state: dict[str, Any]) -> None: ...


def declared_keys(layer, attr: str = "POLICY_STATE") -> tuple[str, ...]:
    """The `attr` keys `layer` declares, refusing an undeclared layer."""
    declared = getattr(type(layer), attr, None)
    if declared is None:
        raise TypeError(
            f"{type(layer).__name__} carries state but does not say which of "
            f"it belongs to {_MEANING[attr]}. Add {attr} naming the get_state "
            f"keys it covers, or {attr} = () if none of its state is of that "
            f"kind. See ttrl.envs.wrappers.core."
        )
    if isinstance(declared, str) or not all(isinstance(k, str) for k in declared):
        raise TypeError(
            f"{type(layer).__name__}.{attr} must be a tuple of get_state "
            f"key names, got {declared!r}"
        )
    return tuple(declared)


def declared_items(layer, attr: str = "POLICY_STATE") -> dict[str, Any]:
    keys = declared_keys(layer, attr)
    if not keys:
        return {}
    state = layer.get_state()
    missing = [k for k in keys if k not in state]
    if missing:
        raise KeyError(
            f"{type(layer).__name__}.{attr} names {missing}, which its "
            f"get_state does not return (it returns {sorted(state)}). Name the "
            f"keys get_state actually writes."
        )
    return {k: state[k] for k in keys}


def refuse_a_mismatched_stack(names: list[str], saved: list[str],
                              kind: str = "policy") -> None:
    """Refuse a snapshot whose bearers are not this stack's, naming both.

    `names` is read off the LIVE objects and `saved` off the file, so neither
    side is a literal this module keeps. Shared by the per-lane restore below
    and by `ttrl.envs.vec_normalize.VecWrapper`, which carries the same kinds
    of state one level up and must refuse them in the same words.
    """
    if names == saved:
        return
    raise ValueError(
        f"env {kind}-state restore: this stack carries {kind} state for "
        f"{names}, the checkpoint carries {saved}. The transform the weights "
        f"were trained under is not the one this stack applies, so loading "
        f"it would feed the network observations on another scale. Build the "
        f"env from the config the checkpoint stores."
    )


_policy_keys = declared_keys
_policy_items = declared_items


class Wrapper:
    """Pass-through base for wrappers over the `Env` protocol.

    Subclasses override only what they change. A subclass that changes the
    observation shape, dtype or scale must also override `obs_spec`; the
    checkers compare the declared spec against every observation returned, so
    a stale spec fails loudly rather than corrupting a replay buffer.
    """

    def __init__(self, env):
        self.env = env

    @property
    def obs_spec(self) -> ObsSpec:
        return self.env.obs_spec

    @property
    def action_spec(self) -> ActionSpec:
        return self.env.action_spec

    @property
    def unwrapped(self):
        inner = self.env
        while isinstance(inner, Wrapper):
            inner = inner.env
        return inner

    def can_render(self) -> bool:
        return supports_render(self.env)

    def render(self) -> np.ndarray:
        # Delegated rather than guarded by `supports_render`, so that an
        # environment which merely has rendering switched off explains that
        # itself instead of being described as unable to render.
        if not isinstance(self.env, Renderable):
            raise RuntimeError(
                f"{type(self).__name__} wraps {type(self.env).__name__}, which "
                f"has no render(). Ask ttrl.envs.supports_render(env) before "
                f"calling render, and see ttrl.envs.render for the capability."
            )
        return self.env.render()

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        return self.env.reset(seed=seed)

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()


def _layers(env) -> Iterator[Any]:
    """Yield the wrapper stack outermost first, then the wrapped env."""
    seen = set()
    layer = env
    while True:
        if id(layer) in seen:
            raise RuntimeError("wrapper stack contains a cycle")
        seen.add(id(layer))
        yield layer
        if not isinstance(layer, Wrapper):
            return
        layer = layer.env


#: Public name for the stack walk above. `ttrl.envs.bounds` asks every layer
#: a question of its own, and a second copy of the walk would be a second
#: answer to "what is a layer".
layers = _layers


def snapshot_policy_state(env) -> list[tuple[str, dict[str, Any]]]:
    """Collect only the state the policy's own transform needs, outermost first.

    Shaped like `snapshot_env` output and restored by `restore_policy_state`.
    Layers that declare `POLICY_STATE = ()` are left out entirely, so the
    result survives being loaded into a stack that differs in a wrapper
    holding no policy state, which is what an eval or export stack usually is.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for layer in _layers(env):
        if isinstance(layer, StatefulEnv):
            items = _policy_items(layer)
            if items:
                out.append((type(layer).__name__, items))
    return out


def restore_policy_state(env, snapshot: list[tuple[str, dict[str, Any]]]) -> None:
    """Apply `snapshot_policy_state` output onto this stack's transform layers.

    The saved keys are merged over each layer's current state, so `set_state`
    stays the one way a layer takes state even though this carries part of it.
    """
    bearers = [
        l for l in _layers(env) if isinstance(l, StatefulEnv) and _policy_keys(l)
    ]
    refuse_a_mismatched_stack([type(l).__name__ for l in bearers],
                              [name for name, _ in snapshot])
    for layer, (_, items) in zip(bearers, snapshot):
        layer.set_state({**layer.get_state(), **items})


def snapshot_env(env) -> list[tuple[str, dict[str, Any]]]:
    """Collect declared state from every layer, outermost first.

    The class name travels with each entry so a restore onto a differently
    configured stack is rejected instead of silently applying the wrong state
    to the wrong layer.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for layer in _layers(env):
        if isinstance(layer, StatefulEnv):
            state = layer.get_state()
            if not isinstance(state, dict):
                raise TypeError(
                    f"{type(layer).__name__}.get_state must return a dict, "
                    f"got {type(state).__name__}"
                )
            out.append((type(layer).__name__, state))
    return out


def restore_env(env, snapshot: list[tuple[str, dict[str, Any]]]) -> None:
    """Apply `snapshot_env` output back onto an identically shaped stack."""
    stateful = [l for l in _layers(env) if isinstance(l, StatefulEnv)]
    names = [type(l).__name__ for l in stateful]
    saved = [name for name, _ in snapshot]
    if names != saved:
        raise ValueError(
            f"env restore: this stack declares state for {names}, the "
            f"checkpoint carries {saved}. The wrapper configuration changed, "
            f"so restoring would resume a different trajectory."
        )
    for layer, (_, state) in zip(stateful, snapshot):
        layer.set_state(state)
