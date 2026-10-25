"""The type-tagged value tree a checkpoint's JSON meta is made of.

A state tree holds numpy arrays, plain values, lists, dicts and `ttrl`
dataclasses. `split` writes one as JSON plus a side table of arrays; `join`
reads it back. A dataclass is tagged with its import path and its field
values, which is how a frozen algorithm config survives the round trip as
itself and therefore still compares equal to the agent's own on resume.

THE FIELD NAMES IN THE FILE ARE PART OF THE ON-DISK CONTRACT. Renaming a
config field used to make every checkpoint ever written unloadable, with a
bare `TypeError` out of a dataclass constructor naming neither the file nor
the field. So `join` checks the stored names against the class before
constructing it, and says which field and what the reader can do about it.

`UNSAVED_FIELDS` is the other half of that contract: a class lists there the
fields a checkpoint must not carry, because they describe the machine a run
happened to sit on rather than the agent it trained. Those fields are never
written, and are dropped when a file carries them anyway. A dropped field is
not restored and not defaulted individually: the object that held it is built
from this build's defaults, which for `TrainConfig.device` is the whole point,
since a stored chip id would send a scoring run at whichever chip the training
run happened to get.

THE ONE SUBSTITUTION THAT IS SILENT, and the reason `defaulted_fields` exists.
A field with a default that the file does not carry takes today's default,
because an optional setting does not change what the stored weights are. But a
resume promises to continue the checkpointed run rather
than a similar one, so `defaulted_fields` names every such field and
`ttrl.train.resume` refuses a warm resume when the list is not empty. A score
is refused for the same reason in `ttrl.train.restore`, because the config
decides what is built around the weights: the network comes from the stored
algo config and the scored environment from the stored env config.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

import numpy as np

from ttrl.train.checkpoint.retype import declared_types, retype
from ttrl.train.checkpoint.types import registered_paths
from ttrl.train.checkpoint.types import resolve as resolve_registered

__all__ = ["CheckpointFieldError", "defaulted_fields", "join", "split",
           "unsaved_fields"]

#: A dataclass in a state tree is rebuilt by importing its type. Restricted to
#: this library so a checkpoint can never name a module of its own choosing.
#: A type outside this root is reachable only by registering it from its own
#: package; see `ttrl.train.checkpoint.types`, which imports nothing.
TYPE_ROOT = "ttrl."


class CheckpointFieldError(ValueError):
    """A stored dataclass field the reading code cannot place."""


def unsaved_fields(cls) -> frozenset[str]:
    """The fields of `cls` a checkpoint neither writes nor reads back."""
    return frozenset(getattr(cls, "UNSAVED_FIELDS", ()))


def split(state: Any, prefix: str, arrays: dict[str, np.ndarray]) -> Any:
    """Separate a state tree's arrays from its scalars.

    Arrays go into `arrays` under a dotted path and leave a `{"__array__":
    path}` marker behind, so the meta blob stays tensor-free and every tensor
    is stored exactly once.
    """
    if isinstance(state, np.ndarray):
        arrays[prefix] = state
        return {"__array__": prefix}
    if isinstance(state, (bytes, bytearray)):
        # An npz member holds this losslessly as uint8. Its own marker, not
        # `__array__`: an opaque blob must come back as `bytes`, because a
        # consumer such as ALE's `restoreState` is handed it directly and a
        # uint8 array there is a silent corruption rather than an error.
        arrays[prefix] = np.frombuffer(bytes(state), dtype=np.uint8)
        return {"__bytes__": prefix}
    if isinstance(state, np.generic):
        return state.item()
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        cls = type(state)
        path = f"{cls.__module__}:{cls.__qualname__}"
        if not path.startswith(TYPE_ROOT) and resolve_registered(path) is not cls:
            raise TypeError(
                f"checkpoint state at {prefix!r} holds {cls.__name__} from "
                f"{cls.__module__}, outside {TYPE_ROOT!r} and not registered. "
                f"A checkpoint names a type by its path, and loading one must "
                f"never import a module the file chose, so a type outside this "
                f"library is written only once the process that reads the file "
                f"will already have it. Register it from the module that "
                f"defines it: `from ttrl import register_checkpoint_type` and "
                f"decorate {cls.__name__} with "
                f"@register_checkpoint_type, above the @dataclass line. Then "
                f"importing {cls.__module__.split('.')[0]} is what makes these "
                f"files loadable, as importing it is already what registers "
                f"the algorithm."
            )
        skip = unsaved_fields(cls)
        return {
            "__dataclass__": path,
            "fields": {
                f.name: split(getattr(state, f.name), f"{prefix}.{f.name}", arrays)
                for f in dataclasses.fields(state)
                if f.name not in skip
            },
        }
    if isinstance(state, dict):
        bad = [k for k in state if not isinstance(k, str)]
        if bad:
            raise TypeError(
                f"checkpoint state at {prefix!r} has non-string keys {bad}. "
                "Keys become npz member names, so they must be strings."
            )
        return {k: split(v, f"{prefix}.{k}", arrays) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return {
            "__list__": [split(v, f"{prefix}.{i}", arrays) for i, v in enumerate(state)]
        }
    if isinstance(state, (str, int, float, bool)) or state is None:
        return state
    raise TypeError(
        f"checkpoint state at {prefix!r} holds {type(state).__name__}, which has no "
        "JSON or array representation. An agent's state_dict must be built from "
        "numpy arrays, bytes, numbers, strings, bools, None, lists and dicts."
    )


def join(node: Any, arrays: dict[str, np.ndarray], *, file: str) -> Any:
    """Rebuild a tree `split` wrote. `file` only names the source in an error."""
    if isinstance(node, dict):
        if "__array__" in node or "__bytes__" in node:
            raw = "__bytes__" in node
            key = node["__bytes__"] if raw else node["__array__"]
            if key not in arrays:
                raise ValueError(
                    f"checkpoint at {file!r} references array {key!r}, which the "
                    "file does not contain. The file is truncated or was not "
                    "written by this code; rewrite it, or train afresh."
                )
            return arrays[key].tobytes() if raw else arrays[key]
        if "__list__" in node:
            return [join(v, arrays, file=file)
                    for v in node["__list__"]]
        if "__dataclass__" in node:
            return _rebuild(node["__dataclass__"], node["fields"], arrays,
                            file=file)
        return {k: join(v, arrays, file=file)
                for k, v in node.items()}
    return node


def defaulted_fields(node: Any, *, path: str = "") -> list[str]:
    """Fields a stored tree does not carry that their class would default.

    Walks the same tree `join` reads and reports `Class.field` for every
    declared, defaulted, init field absent from a stored dataclass. An empty
    list means every field of every dataclass in the file was written by the
    writer rather than invented by the reader.

    A field in `UNSAVED_FIELDS` is not reported: it is deliberately absent, and
    the object holding it is deliberately this build's default.
    """
    found: list[str] = []
    if not isinstance(node, dict):
        return found
    if "__list__" in node:
        for i, item in enumerate(node["__list__"]):
            found += defaulted_fields(item, path=f"{path}[{i}]")
        return found
    if "__dataclass__" in node:
        cls = _import_type(node["__dataclass__"])
        stored = node["fields"]
        skip = unsaved_fields(cls)
        for f in dataclasses.fields(cls):
            if f.name in stored or f.name in skip or not f.init:
                continue
            if (f.default is not dataclasses.MISSING
                    or f.default_factory is not dataclasses.MISSING):
                found.append(f"{cls.__name__}.{f.name}")
        for name, value in stored.items():
            # An unsaved field's stored value is dropped rather than read, so
            # nothing inside it is restored or defaulted.
            if name not in skip:
                found += defaulted_fields(value, path=f"{path}.{name}")
        return found
    if "__array__" in node or "__bytes__" in node:
        return found
    for name, value in node.items():
        found += defaulted_fields(value, path=f"{path}.{name}")
    return found


def _import_type(path: str):
    # A registered type is returned without importing anything, which is the
    # whole point: the file never decides what gets imported.
    registered = resolve_registered(path)
    if registered is not None:
        return registered
    if not path.startswith(TYPE_ROOT) or path.count(":") != 1:
        raise ValueError(
            f"checkpoint names the type {path!r}, which is not a 'module:Name' "
            f"path under {TYPE_ROOT!r} and is not registered. Refusing to "
            f"import it. If it is your own algorithm's config, import the "
            f"package that defines it before loading this file, and have that "
            f"package decorate the class with "
            f"ttrl.register_checkpoint_type; registered now: "
            f"{list(registered_paths())}. Otherwise rewrite the file with a "
            f"state built from plain values, or train afresh."
        )
    module_name, qualname = path.split(":")
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _rebuild(path: str, fields: dict, arrays: dict[str, np.ndarray], *, file: str):
    cls = _import_type(path)
    declared = {f.name: f for f in dataclasses.fields(cls)}
    skip = unsaved_fields(cls)

    unknown = sorted(k for k in fields if k not in declared and k not in skip)
    if unknown:
        raise CheckpointFieldError(
            f"checkpoint at {file!r} stores {cls.__name__} field(s) "
            f"{unknown}, which {cls.__module__}.{cls.__qualname__} does not "
            f"accept: the field was renamed or removed after the file was "
            f"written. The file carries this build's format marker, so it was "
            f"written by this tree before that edit, and the checkpoint format "
            f"is pre-release and carries no migrations. If the field no longer "
            f"belongs in a checkpoint at all, add it to "
            f"{cls.__name__}.UNSAVED_FIELDS and files carrying it load again. "
            f"Otherwise train afresh."
        )

    hints = declared_types(cls)
    kwargs = {k: retype(join(v, arrays, file=file), hints.get(k, Any))
              for k, v in fields.items() if k not in skip}
    # A field the file does not carry takes its default. One with no default
    # cannot be invented, so it is named instead.
    missing = sorted(
        name for name, f in declared.items()
        if name not in kwargs and f.init
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    )
    if missing:
        raise CheckpointFieldError(
            f"checkpoint at {file!r} carries no value for {cls.__name__} "
            f"field(s) {missing}, which {cls.__module__}.{cls.__qualname__} "
            f"requires and has no default for: the field was added after the "
            f"file was written, by this tree, since the file carries this "
            f"build's format marker. Give the field a default so the file "
            f"loads, or train afresh."
        )
    return cls(**{k: v for k, v in kwargs.items() if declared[k].init})
