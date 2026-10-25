"""`ModuleList` and `ModuleDict`: children held by index or by key.

`vars(self)` registration finds a child assigned to an attribute. These two
hold their children in one collection instead, and override the single
`_slot_sources` hook so discovery, naming and checkpointing all see them.
"""

from __future__ import annotations

from ttrl.engine.nn.module import Module

__all__ = ["ModuleList", "ModuleDict"]


class ModuleList(Module):
    """An ordered, mutable list of submodules or parameters.

    Exists so held-by-index children get stable dotted names (`layers.0.weight`)
    and so `append` after construction still registers.
    """

    def __init__(self, modules=()):
        self._items = list(modules)

    def _slot_sources(self):
        return [(str(index), item) for index, item in enumerate(self._items)]

    def append(self, module) -> "ModuleList":
        self._items.append(module)
        return self

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return ModuleList(self._items[index])
        return self._items[index]


class ModuleDict(Module):
    """A string-keyed mapping of submodules or parameters.

    Keys must not contain a dot, because the dotted parameter name built from
    one would be ambiguous against a nested module's own name and would read a
    checkpoint back onto the wrong tensor.
    """

    def __init__(self, modules=None):
        self._items: dict[str, object] = {}
        for key, value in dict(modules or {}).items():
            self[key] = value

    def _slot_sources(self):
        return list(self._items.items())

    def __setitem__(self, key: str, module) -> None:
        if not isinstance(key, str) or "." in key or not key:
            raise ValueError(
                f"ModuleDict keys must be non-empty strings without a dot, got "
                f"{key!r}; a dot would make the parameter name ambiguous."
            )
        self._items[key] = module

    def __getitem__(self, key: str):
        return self._items[key]

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def keys(self):
        return self._items.keys()

    def items(self):
        return self._items.items()
