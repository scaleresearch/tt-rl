"""`Module`: parameter discovery, checkpointing, and target-network sync.

Parameters are discovered implicitly by walking `vars(self)`, so a subclass
registers one by assigning it. There is no `register_parameter`, no `_modules`
bookkeeping dict, and no `.to(device)`: a module is born on the device its
constructor was handed and never moves, because every captured trace replays
against the device addresses its parameter buffers had at construction time.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ttrl.engine.autograd import Tensor
from ttrl.engine.nn.discovery import (
    named_from, serialises_own_state, slots_from,
)
from ttrl.engine.tensor import copy_tensor_in_place, restore_in_place

__all__ = ["Module"]


class Module:
    """Base class. Subclasses register Tensors and submodules as attributes.

    A subclass whose device buffer is not a portable array overrides
    `state_dict` and `load_state_dict` together; a parent then delegates that
    whole subtree to it rather than walking its parameters. See
    `serialises_own_state`.
    """

    # -- parameter discovery ----------------------------------------------

    def _slot_sources(self):
        """`(name, value)` pairs this module registers its children under.

        The single override point for a container holding children somewhere
        other than its own attributes, so no subclass reimplements discovery
        and checkpointing separately and risks the two disagreeing.
        """
        return vars(self).items()

    def _walk(self) -> Iterator[tuple[str, Tensor]]:
        """`(qualified_name, tensor)` for every parameter, de-duplicated by
        `id()` so a tied parameter bound to two attributes is yielded once.

        Names come from real attribute names, recursively, so reordering
        fields in a nested module's `__init__` cannot reinterpret an old
        checkpoint as a different parameter of the same shape.

        Self-serialising children are descended into here, unlike in
        `_state_slots`: the optimizer must see the live device buffer a child
        actually computes against, whatever portable form it checkpoints in.
        """
        seen: set[int] = set()
        for name, value in self._slot_sources():
            yield from named_from(name, value, seen)

    def named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        yield from self._walk()

    def parameters(self) -> Iterator[Tensor]:
        for _, param in self._walk():
            yield param

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    # -- checkpointing ----------------------------------------------------

    def _names_and_params(self) -> tuple[list[str], list[Tensor]]:
        named = list(self.named_parameters())
        params = list(self.parameters())
        # Identity, not just count: a subclass overriding only one of
        # `parameters()`/`named_parameters()` could yield the same tensors in
        # a different order, writing a checkpoint whose keys name the wrong
        # tensors with no shape error to catch it.
        if [id(p) for _, p in named] != [id(p) for p in params]:
            raise RuntimeError(
                f"{type(self).__name__}: named_parameters() and parameters() "
                f"disagree ({len(named)} vs {len(params)} tensors, or a "
                "different order); a subclass overrides one without the other"
            )
        return [n for n, _ in named], params

    def _state_slots(self, seen: set[int] | None = None) -> list[tuple[str, object]]:
        """`(dotted_name, target)` for everything this module checkpoints.

        A target is either a parameter Tensor, written and read generically,
        or a child module that serialises itself, delegated to whole.
        """
        seen = set() if seen is None else seen
        slots: list[tuple[str, object]] = []
        for name, value in self._slot_sources():
            slots.extend(slots_from(name, value, seen))
        return slots

    def state_dict(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, target in self._state_slots():
            if isinstance(target, Tensor):
                out[name] = target.numpy()
                continue
            for key, array in target.state_dict().items():
                out[f"{name}.{key}"] = array
        return out

    def load_state_dict(self, state: dict, device=None) -> None:
        """Restore parameters in place.

        Address stability: every write goes into a buffer that already exists,
        through `restore_in_place` here or through the child's own in-place
        write when it serialises itself. Nothing rebinds `.data`, so a trace
        captured before the restore still replays against live weights.

        Validates the entire key set and every shape before touching any
        buffer: a restore that failed halfway would leave the network a mix of
        two checkpoints with nothing to say so. `device` is accepted for the
        frozen call signature and is unused, since the destination buffers
        already carry their device.
        """
        slots = self._state_slots()
        # The expected key set and shapes come from state_dict() itself, so
        # what load accepts is by construction exactly what save produces,
        # including for a self-serialising child whose portable shape this
        # class cannot otherwise know. That costs one host read of the current
        # weights; checkpoint restore is host-side and never runs in a step.
        reference = self.state_dict()

        got = set(state.keys())
        expected = set(reference)
        if got != expected:
            raise ValueError(
                "load_state_dict: checkpoint keys do not match module "
                f"parameters (missing={sorted(expected - got)}, "
                f"extra={sorted(got - expected)})"
            )
        for name, want in reference.items():
            shape = tuple(int(d) for d in np.asarray(state[name]).shape)
            if shape != want.shape:
                raise ValueError(
                    f"load_state_dict: shape mismatch for {name!r}: "
                    f"checkpoint has {shape}, parameter expects {want.shape}"
                )

        for name, target in slots:
            if isinstance(target, Tensor):
                restore_in_place(target.data, state[name])
                continue
            prefix = f"{name}."
            target.load_state_dict(
                {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            )

    def copy_from(self, other: "Module") -> None:
        """Hard-copy another module's parameters, in place (target-network sync)."""
        dst_named = list(self.named_parameters())
        src_named = list(other.named_parameters())
        if len(dst_named) != len(src_named):
            raise ValueError(
                f"copy_from: parameter count mismatch ({len(dst_named)} vs "
                f"{len(src_named)}); target and source modules do not match"
            )
        # Count alone is not evidence the two towers are the same network: two
        # unrelated models with the same parameter count would copy across
        # positionally and silently cross-wire. Names and shapes pin it down.
        for (dst_name, dst), (src_name, src) in zip(dst_named, src_named, strict=True):
            if dst_name != src_name:
                raise ValueError(
                    f"copy_from: parameter name mismatch ({dst_name!r} vs "
                    f"{src_name!r}); the two modules have different structure"
                )
            dshape = tuple(int(d) for d in dst.shape)
            sshape = tuple(int(d) for d in src.shape)
            if dshape != sshape:
                raise ValueError(
                    f"copy_from: shape mismatch for {dst_name!r}: target "
                    f"{dshape}, source {sshape}"
                )
            copy_tensor_in_place(dst, src)

