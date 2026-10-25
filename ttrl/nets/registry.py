"""Register/resolve-by-name, and the one architecture seam that is live.

THE ENCODER IS THE ARCHITECTURE SEAM. `ENCODERS` maps a name to
`builder(obs_spec, device, *, batch, **kwargs) -> Encoder`, and every network
this package builds takes its tower from it: `encoder` is a field on each
algorithm's config, so a user registers a builder from their own package,
names it in a config and nothing under `ttrl/engine/` or `ttrl/train/`
changes. If adding an architecture requires editing a file here, the seam has
failed and that is a bug.

There is deliberately no registry of whole networks. What an algorithm needs
is not one module but a bundle whose shape the algorithm itself reads: DQN
takes an encoder and a Q head, PPO an actor trunk, a critic trunk and two
heads, SAC a Gaussian policy with twin critics and their targets. A name
resolving to "a network" cannot be substituted across those three, so such a
registry can only ever be an alias for the builder the algorithm already
calls. The substitutable part is the tower, and that is `ENCODERS`.

`Registry` itself is shared with `ttrl.algos`, `ttrl.envs` and
`ttrl.buffers.samplers`, so the four extension points have one shape and one
set of verbs. An entry may be a `"module:attr"` string where the registry was
built with `allow_targets=True`, which is what lets a registration name a
builder whose dependencies are absent, and what lets a registered name cross
a subprocess boundary; see `ttrl.envs.registry`.

THE BUILTIN ENTRIES BELOW ARE TARGET STRINGS, as the other three registries'
are. The alternative, a `builtins=` module list imported on first lookup,
answers `name in ENCODERS`, `ENCODERS.names()`, `len(ENCODERS)` and
`ENCODERS.entry(name)` by importing every builtin encoder, and those import
ttnn through `ttrl.engine`. Those questions are asked while validating a
config, before the process is scoped to its chip, so they must cost no
import; `resolve` stays the one query that imports, and it imports the one
encoder asked for rather than all four.
"""

from __future__ import annotations

import importlib
from typing import Callable, Iterable

__all__ = ["Registry", "ENCODERS", "import_target"]


def import_target(target: str) -> Callable:
    """The attribute a "module:attr" entry names, importing its module.

    A target that names nothing is told apart from a target whose module is
    there but whose own imports are not: the first is a typo in the
    registration, the second is a missing optional dependency, and reporting
    the second as the first would send the reader to the wrong file.
    """

    module_name, attr = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise ModuleNotFoundError(
            f"the registered target {target!r} names the module "
            f"{module_name!r}, which does not exist. Register it as "
            f"\"<importable module>:{attr}\"."
        ) from exc
    try:
        return getattr(module, attr)
    except AttributeError:
        raise AttributeError(
            f"the registered target {target!r} names {attr!r} in "
            f"{module_name!r}, which defines no such name. Register the "
            f"attribute that is there, or define {attr!r} in {module_name}."
        ) from None


class Registry:
    """Name -> builder. Duplicate names and unknown names both raise."""

    def __init__(self, what: str, builtins: Iterable[str] = (),
                 module: str = "ttrl.nets.registry",
                 allow_targets: bool = False):
        self._what = what
        # Whether an entry may be a dotted "module:attr" string, imported on
        # the first lookup. Off by default: a registry whose entries are all
        # callables cannot be handed a typo'd string that resolves to nothing
        # until someone asks for it.
        self._allow_targets = allow_targets
        self._items: dict[str, Callable] = {}
        # Where an unknown-name message tells the reader to register theirs.
        # A parameter rather than a hardcoded string, because this class is
        # reused by the algos and envs registries, and a message naming the
        # wrong module is worse than no message: it sends the reader to a file
        # that does not hold the registry they are missing.
        self._module = module
        # Modules holding the builders this package ships, imported on the first
        # lookup rather than at package import. They pull in ttnn, and
        # `import ttrl` must not, so that a launcher can still scope the visible
        # chips before ttnn initialises.
        self._builtins = tuple(builtins)
        # Kept as a separate flag rather than by clearing `_builtins`, because
        # the empty-registry message below has to name the modules that were
        # supposed to register something.
        self._loaded = not self._builtins

    def _load_builtins(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        for module in self._builtins:
            importlib.import_module(module)

    def register(self, name: str, builder: Callable | str | None = None):
        """Register `builder` under `name`, or return a decorator if omitted."""
        if not isinstance(name, str) or not name:
            raise ValueError(f"{self._what} registry: name must be a non-empty string, got {name!r}")
        if name in self._items:
            raise ValueError(
                f"{self._what} registry: {name!r} is already registered to "
                f"{self._items[name]!r}. Silently replacing it would make which "
                f"{self._what[:-1]} a config names depend on import order. Pick "
                f"another name, or unregister({name!r}) first."
            )
        if builder is None:
            return self(name)
        if isinstance(builder, str):
            self._check_target(name, builder)
        elif not callable(builder):
            raise ValueError(f"{self._what} registry: builder for {name!r} is not callable")
        self._items[name] = builder
        return builder

    def _check_target(self, name: str, target: str) -> None:
        if not self._allow_targets:
            raise TypeError(
                f"{self._what} registry: builder for {name!r} is not callable. "
                f"This registry takes callables only. To register a 'module:attr' "
                f"target, whose point is to be resolvable without importing its "
                f"dependencies, build the registry with allow_targets=True."
            )
        if target.count(":") != 1:
            raise ValueError(
                f"{self._what} registry: target {target!r} for {name!r} must "
                f"have the form 'module:attr'"
            )

    def unregister(self, name: str) -> None:
        self._load_builtins()
        if name not in self._items:
            raise KeyError(self._unknown(name))
        del self._items[name]

    def entry(self, name: str) -> Callable | str:
        """The raw entry for `name`: a callable, or its "module:attr" target.

        What `resolve` imports from. Separate because a registered name has to
        be able to travel to a subprocess that never ran the registration, and
        the target string is the only form of an entry that can.
        """
        self._load_builtins()
        found = self._items.get(name)
        if found is None:
            raise KeyError(self._empty_or_unknown(name))
        return found

    def __call__(self, name: str) -> Callable:
        """Decorator form: `@ENCODERS("my_tower")`."""
        def decorate(builder: Callable) -> Callable:
            self.register(name, builder)
            return builder
        return decorate

    def resolve(self, name: str) -> Callable:
        """The builder registered under `name`. THE ONE QUERY THAT IMPORTS.

        A `"module:attr"` entry is imported here, which for an algorithm means
        ttnn. Every other question this registry answers (`in`, `names`,
        `len`, `entry`) is answered out of the table without importing
        anything, so a caller that must not touch ttnn yet, such as one
        validating a config before the process is scoped to its chip, asks one
        of those instead. See `ttrl.train.scoping`.

        Two distinct failures, two messages. An unknown name among known ones
        is the caller's: they asked for something that does not exist. An
        EMPTY registry is never the caller's, whatever name they passed, so
        pointing at that name sends them to debug a correct argument. It means
        nothing imported the modules that do the registering, and the way that
        happens is a builtins entry naming a package whose `__init__` is lazy:
        importing the package then runs no registration at all.
        """
        found = self.entry(name)
        if isinstance(found, str):
            builder = import_target(found)
            self._items[name] = builder
            return builder
        return found

    def _empty_or_unknown(self, name: str) -> str:
        if not self._items:
            cause = (
                f"This registry imports {list(self._builtins)} on first lookup, so an entry "
                f"there naming a package whose __init__ is lazy registers nothing: "
                f"importing the package runs no decorator. Point the entry at the "
                f"module that holds the builders rather than at its package"
                if self._builtins else
                "This registry declares no builtin modules at all, so only an explicit "
                f"{self._module}.{self._what.upper()}.register(name, builder) call "
                "can populate it"
            )
            return (
                f"the {self._what} registry is empty, which is not a problem with "
                f"the name {name!r} you passed: nothing registered a builder. "
                f"{cause}."
            )
        return self._unknown(name)

    def _unknown(self, name: str) -> str:
        return (
            f"unknown {self._what[:-1]} {name!r}; registered names are "
            f"{self.names()}. Register yours with "
            f"{self._module}.{self._what.upper()}.register(name, builder)."
        )

    def names(self) -> list[str]:
        self._load_builtins()
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        self._load_builtins()
        return name in self._items

    def __len__(self) -> int:
        self._load_builtins()
        return len(self._items)


#: `builder(obs_spec, device, *, batch, **kwargs) -> Encoder`
ENCODERS = Registry("encoders", module="ttrl.nets.registry", allow_targets=True)

ENCODERS.register("nature_cnn", "ttrl.nets.encoders:image_encoder")
ENCODERS.register("nature_cnn_s2d", "ttrl.nets.encoders:image_encoder_s2d")
ENCODERS.register("mlp", "ttrl.nets.encoders:vector_encoder")
ENCODERS.register("identity", "ttrl.nets.encoders:identity_encoder")
