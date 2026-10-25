"""Rendering: the optional capability of handing the caller a frame.

Watching a trained policy is a first-class use case for a library whose
targets are games and locomotion, and it must not require reaching past the
wrapper stack. An environment that can produce frames implements
`Renderable`: `render()` returns one RGB frame of the current state, and
`can_render()` says whether it will.

The capability is optional because most environments cannot render at all,
and a protocol member that most environments cannot implement is worse than
none. It carries two members because a Gymnasium environment renders only
when it was constructed with `render_mode="rgb_array"`, so a protocol whose
only member is `render()` would be satisfied by instances that then raise.
`can_render()` answers that question before any frame is wanted, and
`supports_render(env)` is the one supported way for a caller to ask: it is
False for anything that does not implement the protocol at all, so no caller
needs a `hasattr` guess.

Frames are uint8 RGB arrays of shape (height, width, 3). Returning them is
this library's job and displaying them is the caller's, so nothing here
imports a window toolkit or a video encoder.

A wrapper inherits the capability: `ttrl.envs.wrappers.Wrapper` delegates
both members down the stack, so `supports_render` on a stack of wrappers
reports what the environment underneath it can do.

`VecEnv` has no rendering. Eight lanes stepping in eight subprocesses is a
different question from one env on screen, and the answer to it is a grid
composited by whoever wants one. Build a single `Env` to watch a policy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["Renderable", "supports_render", "check_frame"]


@runtime_checkable
class Renderable(Protocol):
    """Optional capability: the current state as one RGB frame."""

    def can_render(self) -> bool:
        """Whether `render` will return a frame rather than raise."""
        ...

    def render(self) -> np.ndarray:
        """The current frame, uint8 (height, width, 3) RGB."""
        ...


def supports_render(env) -> bool:
    """Whether `env.render()` is available. The supported way to ask."""
    return isinstance(env, Renderable) and bool(env.can_render())


def check_frame(frame, *, where: str = "render") -> np.ndarray:
    """Return `frame` as a contiguous uint8 RGB array, or raise naming the fault."""
    if not isinstance(frame, np.ndarray):
        raise TypeError(
            f"{where} must return a numpy array of shape (height, width, 3), "
            f"got {type(frame).__name__}"
        )
    if frame.dtype != np.uint8:
        raise TypeError(
            f"{where} returned dtype {frame.dtype}, frames are uint8 RGB. "
            f"Scale and cast the frame before returning it."
        )
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ValueError(
            f"{where} returned shape {frame.shape}, frames are "
            f"(height, width, 3) RGB with both sizes at least 1. Stack a "
            f"greyscale frame into three channels before returning it."
        )
    return np.ascontiguousarray(frame)
