"""Restoring a declared tuple a JSON round trip left as a list.

JSON has one sequence type, so `tree.split` writes a tuple field and
`tree.join` reads a list back. A config or spec whose tuple field returned as
a list is no longer `==` to the object that wrote it and is no longer
hashable, so a resume that changed nothing reports a changed field: a
`SACConfig.critic_hidden` read back as `[256]` is named by
`describe_config_diff` as a differing SHAPE field, and an `ActionSpec` whose
`low`/`high`/`nvec` came back as lists cannot be put in a set at all.
"""

from __future__ import annotations

import functools
import types
import typing
from typing import Any, Union, get_args, get_origin

__all__ = ["declared_types", "retype"]


@functools.lru_cache(maxsize=None)
def declared_types(cls) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def retype(value: Any, ftype: Any) -> Any:
    """The tuple a declared tuple field must come back as."""
    if not isinstance(value, list):
        return value
    origin = get_origin(ftype)
    if origin is Union or origin is types.UnionType:
        for arg in get_args(ftype):
            if arg is type(None):
                continue
            retyped = retype(value, arg)
            if isinstance(retyped, tuple):
                return retyped
        return value
    if origin in (tuple, list):
        args = get_args(ftype)
        if origin is tuple and args and Ellipsis not in args and len(args) == len(value):
            return tuple(retype(v, a) for v, a in zip(value, args))
        elem = args[0] if args and args[0] is not Ellipsis else Any
        retyped = [retype(v, elem) for v in value]
        return tuple(retyped) if origin is tuple else retyped
    return value
