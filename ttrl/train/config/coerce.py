"""Dataclass <-> dict conversion, with type coercion and validation.

Used both to build a :class:`~ttrl.train.config.schema.TrainConfig` from a
plain (YAML-shaped) dict and to serialize one back for override application.
"""

from __future__ import annotations

import difflib
import sys
import types
from dataclasses import fields, is_dataclass
import importlib
import pkgutil
from typing import Any, Union, get_args, get_origin

import ttrl.algos


#: Which algorithm configs declare `recurrent`, derived rather than typed:
#: a config naming a field its own algorithm does not have is an unknown key
#: here, and the generic unknown-key error would name the nearest spelling
#: instead of saying that no replay algorithm can honour the block. Walked on
#: the error path only, so nothing imports an algorithm to load a config.
def _owners_of(field: str) -> list[str]:
    owners = []
    for mod in pkgutil.walk_packages(ttrl.algos.__path__, "ttrl.algos."):
        if not mod.name.endswith(".config"):
            continue
        for obj in vars(importlib.import_module(mod.name)).values():
            if (is_dataclass(obj) and isinstance(obj, type)
                    and field in {f.name for f in fields(obj)}):
                owners.append(obj.__name__)
    return sorted(set(owners))


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass instance ``cls`` from a plain dict.

    Nested dataclass fields are built recursively; unknown keys raise (typos
    in YAML should fail loudly, not silently vanish). Every value is also
    coerced against -- and validated against -- its field's declared type
    (see :func:`_coerce_value`): a value that cannot be made to fit the
    declared type raises ``TypeError``/``ValueError`` naming the field and
    both the expected and given type/value, right here, rather than
    constructing a dataclass with a wrong-shaped field and deferring the
    failure to a confusing ``AttributeError`` much later in training.
    """
    if not is_dataclass(cls):
        return data

    kwargs: dict[str, Any] = {}
    field_map = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(field_map)
    if unknown:
        if "recurrent" in unknown and cls.__module__.startswith("ttrl.algos."):
            owners = _owners_of("recurrent")
            raise ValueError(
                f"{cls.__name__}: `recurrent` is not a field of this algorithm. "
                f"Recurrence is carried by {' and '.join(owners)}. A replay "
                f"buffer samples transitions rather than contiguous sequences, "
                f"and a stored hidden state goes stale over that buffer's "
                f"lifetime, so {cls.__name__} would need sequence-aware replay "
                f"and a burn-in that do not exist yet. Set an algorithm whose "
                f"config carries the field, or drop the `recurrent` block to "
                f"train a feedforward policy."
            )
        # `algo_cfg.lrr=1e-4` is the single most common override typo; spelling the
        # nearest real field beats making the user reread the dataclass.
        hints = []
        for key in sorted(unknown):
            close = difflib.get_close_matches(key, field_map, n=1)
            hints.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ValueError(
            f"{cls.__name__}: unknown config key(s) {', '.join(hints)}; "
            f"known keys: {sorted(field_map)}"
        )

    for name, f in field_map.items():
        if name not in data:
            continue
        value = data[name]
        ftype = f.type if not isinstance(f.type, str) else _resolve_type(cls, f.type)
        kwargs[name] = _coerce_value(cls.__name__, name, value, ftype)
    return cls(**kwargs)


def _resolve_type(cls: type, type_name: str) -> Any:
    # Field.type may be a string (postponed annotations). Resolve it against
    # the dataclass's own module globals, which is where the schema types live.
    module = sys.modules[cls.__module__]
    try:
        return eval(type_name, vars(module))  # noqa: S307 - trusted, local module
    except Exception as exc:
        # Returning `type_name` (a *string*) on failure made `_coerce_value`
        # miss every isinstance branch and pass the value through untouched, so
        # the field lost all validation -- silently, in the module whose whole
        # job is to make a bad override fail at load time.
        raise TypeError(
            f"cannot resolve annotation {type_name!r} of {cls.__name__} against "
            f"{module.__name__}: {exc}"
        ) from exc


def _is_union(ftype: Any) -> bool:
    origin = get_origin(ftype)
    return origin is Union or origin is types.UnionType


def _coerce_value(owner: str, name: str, value: Any, ftype: Any) -> Any:
    """Coerce/validate ``value`` against a field's declared type ``ftype``.

    Raises ``TypeError`` (or lets a nested ``ValueError``/``AssertionError``
    from a dataclass's own ``__post_init__`` propagate) the moment a value
    cannot be made to fit its declared type, naming the field, the expected
    type, and what was actually given -- so a typo'd override fails at load
    time, not three log lines into a run.
    """
    if is_dataclass(ftype):
        if not isinstance(value, dict):
            raise TypeError(
                f"{owner}.{name}: expected a dict to build {getattr(ftype, '__name__', ftype)}, "
                f"got {type(value).__name__} ({value!r})"
            )
        return _from_dict(ftype, value)

    if ftype is Any or ftype is None or ftype is type(None):
        return value

    if _is_union(ftype):
        args = get_args(ftype)
        allow_none = type(None) in args
        if value is None:
            if allow_none:
                return None
            raise TypeError(f"{owner}.{name}: got None, but {ftype} does not allow it")
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce_value(owner, name, value, arg)
            except (TypeError, ValueError) as e:
                errors.append(f"{arg}: {e}")
        raise TypeError(
            f"{owner}.{name}: value {value!r} ({type(value).__name__}) matches none of "
            f"the declared type {ftype}; tried: {errors}"
        )

    origin = get_origin(ftype)
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"{owner}.{name}: expected a list for {ftype}, got "
                f"{type(value).__name__} ({value!r})"
            )
        elem_args = get_args(ftype)
        elem_t = elem_args[0] if elem_args and elem_args[0] is not Ellipsis else Any
        coerced = [_coerce_value(owner, name, v, elem_t) for v in value]
        # A field declared `tuple[...]` comes back a tuple whatever the source
        # spelled it as: YAML and JSON have one sequence, and a config whose
        # tuple field loaded as a list is not `==` to the one that wrote it,
        # which makes a resume that changed nothing report a changed field.
        return tuple(coerced) if origin is tuple else coerced

    if ftype is bool:
        if isinstance(value, bool):
            return value
        raise TypeError(f"{owner}.{name}: expected bool, got {type(value).__name__} ({value!r})")

    if ftype is int:
        if isinstance(value, bool):
            raise TypeError(f"{owner}.{name}: expected int, got bool ({value!r})")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        raise TypeError(f"{owner}.{name}: expected int, got {type(value).__name__} ({value!r})")

    if ftype is float:
        if isinstance(value, bool):
            raise TypeError(f"{owner}.{name}: expected float, got bool ({value!r})")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise TypeError(f"{owner}.{name}: expected float, got {type(value).__name__} ({value!r})")

    if ftype is str:
        if isinstance(value, str):
            return value
        raise TypeError(f"{owner}.{name}: expected str, got {type(value).__name__} ({value!r})")

    if isinstance(ftype, type) and not isinstance(value, ftype):
        raise TypeError(
            f"{owner}.{name}: expected {ftype.__name__}, got {type(value).__name__} ({value!r})"
        )
    return value


def config_to_dict(obj: Any) -> Any:
    """Recursively convert a (possibly nested) dataclass to plain dicts.

    A dataclass declaring `CONFIG_VALUE = True` is kept WHOLE: it is one
    config value that happens to be written as a dataclass, so decomposing it
    would turn it into a section. `ttrl.nets.choice.UnnamedEncoder` is the
    live case: decomposed, its fields reached `coerce_encoder` as a mapping
    and were read back as a per-leaf encoder mapping, which silently replaced
    the handed-over class, and its callable reached `json.dumps` in
    `ttrl.train.provenance` and killed the run before the first frame.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        if getattr(obj, "CONFIG_VALUE", False):
            return obj
        return {f.name: config_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj
