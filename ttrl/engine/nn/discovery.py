"""Which attributes of a `Module` hold its parameters, and which its state.

Split out of `ttrl.engine.nn.module` so that the walk and the class that calls
it are each readable on their own. `named_from` is the optimizer's view
(every live parameter) and `slots_from` the checkpoint's (a self-serialising
child is delegated to whole rather than walked).
"""

from __future__ import annotations

from typing import Iterator

from ttrl.engine.autograd import Tensor

__all__ = ["named_from", "refuse_stray_parameters", "serialises_own_state",
           "slots_from"]


def _module():
    # Imported on call: `module` imports this walk at its own module
    # scope, so naming `Module` here at module scope is a cycle.
    from ttrl.engine.nn.module import Module

    return Module


def is_container(value) -> bool:
    """Duck-typed: anything that can name its own parameters is a submodule.

    Not `isinstance(value, Module)`, because a layer may hold its parameters
    behind its own `named_parameters()` without sharing this base class.
    """
    return callable(getattr(value, "named_parameters", None))


def named_from(prefix: str, value, seen: set[int]) -> Iterator[tuple[str, Tensor]]:
    """`(dotted_name, tensor)` for one attribute slot, skipping ids in `seen`.

    Plain lists, tuples and dicts are walked as well as submodules. A bare
    `self.layers = [Linear(...), ...]` whose contents were invisible here
    would train nothing in that list and raise nothing: the optimizer would
    simply never see those parameters.
    """
    if isinstance(value, Tensor):
        if value.requires_grad and id(value) not in seen:
            seen.add(id(value))
            yield prefix, value
        return
    if is_container(value):
        for sub, param in value.named_parameters():
            if id(param) not in seen:
                seen.add(id(param))
                yield f"{prefix}.{sub}", param
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from named_from(f"{prefix}.{index}", item, seen)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from named_from(f"{prefix}.{key}", item, seen)
        return
    refuse_stray_parameters(prefix, value)


def refuse_stray_parameters(prefix: str, value) -> None:
    """Raise when a slot discovery cannot reach holds trainable Tensors.

    The silent case this exists to end: a layer written without this base
    class and without `named_parameters` is reached by neither branch above,
    so its weights never reach the optimizer and no gradient ever lands on
    them. The run trains, the loss moves (the rest of the network still
    learns), and nothing says that part of the model is frozen. One attribute
    level is enough to catch it, since the object itself is what discovery
    stopped at.
    """
    held = getattr(value, "__dict__", None)
    if not isinstance(held, dict):
        return
    stray = sorted(
        name for name, item in held.items()
        if isinstance(item, Tensor) and item.requires_grad
    )
    if not stray:
        return
    raise TypeError(
        f"attribute {prefix!r} holds a {type(value).__name__} carrying "
        f"trainable tensors {stray}, and parameter discovery cannot see "
        f"them: it walks Tensors, `ttrl.nets.Module` children, lists and "
        f"dicts. Left alone those weights would get no gradient and nothing "
        f"would say so. Make {type(value).__name__} a subclass of "
        f"ttrl.nets.Module."
    )


def serialises_own_state(value) -> bool:
    """Does `value` define its own `state_dict`/`load_state_dict` pair?

    The protocol by which a layer whose device buffer is not its portable form
    takes over its own checkpointing. `Conv2d` is the live case: its master
    weight IS ttnn.conv2d's prepared matrix, whose padding depends on the shard
    and parallel config and on the ttnn build. The Nature CNN's conv1 holds
    16384 prepared elements against 2048 logical ones. Saving that buffer
    verbatim produces a checkpoint that loads only back into a layer built
    under the same config on the same wheel, and fails by silently reading the
    wrong numbers.

    Both halves are required: a class exporting a portable form while reading
    a prepared one back, or the reverse, would corrupt weights with no error.
    """
    module = _module()
    if not isinstance(value, module):
        return False
    cls = type(value)
    saves = cls.state_dict is not module.state_dict
    loads = cls.load_state_dict is not module.load_state_dict
    if saves != loads:
        missing = "load_state_dict" if saves else "state_dict"
        raise TypeError(
            f"{cls.__name__} overrides "
            f"{'state_dict' if saves else 'load_state_dict'} but not "
            f"{missing}; a module that serialises itself must do both, or a "
            "checkpoint is written in one form and read back in another."
        )
    return saves


def slots_from(prefix: str, value, seen: set[int]):
    """`(dotted_name, target)` for one slot: a parameter Tensor, or a child
    module that serialises itself and is therefore delegated to whole.

    Stops at a self-serialising child instead of walking its parameters, which
    is what keeps a prepared device layout out of the checkpoint.
    """
    if serialises_own_state(value):
        if id(value) not in seen:
            seen.add(id(value))
            yield prefix, value
        return
    if isinstance(value, Tensor):
        if value.requires_grad and id(value) not in seen:
            seen.add(id(value))
            yield prefix, value
        return
    if isinstance(value, _module()):
        for sub, target in value._state_slots(seen):
            yield f"{prefix}.{sub}", target
        return
    if is_container(value):
        # A duck-typed child with no Module base cannot take over its own
        # checkpointing, so its parameters are the portable form by default.
        for sub, param in value.named_parameters():
            if id(param) not in seen:
                seen.add(id(param))
                yield f"{prefix}.{sub}", param
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from slots_from(f"{prefix}.{index}", item, seen)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from slots_from(f"{prefix}.{key}", item, seen)
