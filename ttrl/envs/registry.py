"""Name to environment-builder registry.

Adding an environment is registering a builder that returns something
satisfying `ttrl.envs.base.Env`, carrying its own `ObsSpec` and `ActionSpec`.
Nothing in the rest of the library has to change, and nothing branches on
which suite an env came from.

`ENVS` is the same `Registry` the encoders and the algorithms use, so the
three extension points have one shape and one set of verbs: `register`,
`names`, `resolve`, `unregister`, `in`. It is built with `allow_targets`,
which this registry alone needs: an entry may be a dotted `"module:attr"`
string, resolved on first use, so registering an environment whose
dependencies are absent costs nothing at import and fails only when someone
asks for it.

`register` is re-exported from `ttrl.envs`, which is where a user looks after
registering an encoder with `ttrl.nets.ENCODERS`.
"""

from __future__ import annotations

import pickle

from ttrl.nets.registry import Registry, import_target

__all__ = ["ENVS", "register", "make", "require_spawnable", "EnvFactory"]

#: name -> builder callable, or "module:attr" resolved on first use.
ENVS = Registry("envs", module="ttrl.envs", allow_targets=True)

#: The registration entry point, under the name `ttrl.envs.register`.
register = ENVS.register


def make(name: str, **kwargs):
    """Build the environment registered under `name`."""
    return ENVS.resolve(name)(**kwargs)


class EnvFactory:
    """A picklable zero-argument builder, for `AsyncVectorEnv` under "spawn".

    A lambda cannot cross a spawn boundary. This can, because it carries only
    the registry's entry for a name and the keyword arguments.

    It carries the ENTRY and not just the name, which is what makes a user's
    own environment work under `asynchronous=True`. The registry is
    per-process and the lanes are spawned, so a child starts with only the
    entries this module registers below; a name registered inside the
    caller's `main()` is unknown in every child. Carrying the entry means the
    child resolves what the parent resolved, with no registration of its own.
    """

    def __init__(self, name: str, **kwargs):
        # Resolved here rather than in the child, so an unknown name is
        # reported by the caller's own stack frame.
        self.target = ENVS.entry(name)
        self.name = name
        self.kwargs = kwargs

    def __call__(self):
        target = self.target
        if isinstance(target, str):
            target = import_target(target)
        return target(**self.kwargs)

    def __repr__(self) -> str:
        return f"EnvFactory({self.name!r}, **{self.kwargs!r})"


def require_spawnable(factory: EnvFactory) -> None:
    """Refuse a factory that cannot reach a spawned lane, before one is spawned.

    A builder defined inside a function pickles by reference like any other
    callable and there is no reference to it: the name does not exist at its
    module's top level. The failure would otherwise arrive as a pickling
    error from inside `multiprocessing`, naming neither the environment nor
    the asynchronous lane that asked for it.
    """
    try:
        pickle.dumps(factory)
    except Exception as exc:
        raise ValueError(
            f"env {factory.name!r} is registered as {factory.target!r}, which "
            f"cannot be sent to a subprocess, and asynchronous lanes each run "
            f"in one: {type(exc).__name__}: {exc}. Register a builder defined "
            f"at the top level of a module, or name it as a target string "
            f"(ttrl.envs.register({factory.name!r}, 'mypackage.envs:make_it')), "
            f"which the lane imports for itself. A builder defined inside a "
            f"function cannot be reached from another process; set "
            f"env.asynchronous=False to keep the lanes in this one."
        ) from exc


register("atari", "ttrl.envs.atari:make_atari")
register("mujoco", "ttrl.envs.mujoco:make_mujoco")
