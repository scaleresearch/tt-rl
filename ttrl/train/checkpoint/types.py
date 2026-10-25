"""The types a checkpoint is allowed to name, beyond this library's own.

`tree.py` rebuilds a stored dataclass by its import path, and restricts that
to `ttrl.` so that loading a file never imports a module the file chose. That
restriction also locks out the one external type a run legitimately stores:
`TrainConfig.algo_cfg`, which belongs to whoever wrote the algorithm. The
advice to put plain values in the state instead cannot help there, because the
config's serialization is the framework's and not the algorithm author's.

So a package registers its own config type here, and a checkpoint may name it.
The property `TYPE_ROOT` exists for is kept exactly: nothing here imports
anything. `resolve` returns a class THIS PROCESS already registered, so a file
naming `evil.module:Thing` still cannot cause `evil.module` to be imported. A
registration is reached only by code the user already chose to import, which
is the same act of trust as `ttrl.ALGOS.register`.
"""

from __future__ import annotations

import dataclasses

__all__ = ["register_checkpoint_type", "registered_paths", "resolve", "type_path"]

#: `"module:Qualname"` -> the class, for types registered by their own package.
_REGISTERED: dict[str, type] = {}


def type_path(cls: type) -> str:
    """The `"module:Qualname"` a checkpoint names `cls` by."""
    return f"{cls.__module__}:{cls.__qualname__}"


def register_checkpoint_type(cls: type) -> type:
    """Allow a checkpoint to name `cls`. Returns `cls`, so it reads as a decorator.

    Register the config type of an algorithm registered in `ttrl.ALGOS`, from
    the module that defines it, so that importing the algorithm is what makes
    its checkpoints loadable.
    """
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise TypeError(
            f"register_checkpoint_type expects a dataclass TYPE, got "
            f"{cls!r}. A checkpoint stores a dataclass as its field values and "
            f"rebuilds it by calling the class, so decorate the class itself: "
            f"@register_checkpoint_type above @dataclasses.dataclass."
        )
    path = type_path(cls)
    seen = _REGISTERED.get(path)
    if seen is not None and seen is not cls:
        raise ValueError(
            f"register_checkpoint_type: {path!r} is already registered to a "
            f"different class. Two classes of that name in that module cannot "
            f"both be named by a checkpoint, because the file carries the path "
            f"and nothing else. Rename one of them."
        )
    _REGISTERED[path] = cls
    return cls


def resolve(path: str) -> type | None:
    """The registered class for `path`, or None. Imports nothing."""
    return _REGISTERED.get(path)


def registered_paths() -> tuple[str, ...]:
    """Every path a checkpoint may name beyond this library's own."""
    return tuple(sorted(_REGISTERED))
