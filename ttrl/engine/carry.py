"""`ActingCarry`: the two-buffer state carry a single-step traced loop needs.

A training unroll needs no mechanism for its state. T is fixed at the shape
the caller binds, `h_0` is an ordinary bound input written in place by
`ttrl.engine.step.inputs.InputBinder`, and every timestep's state is a fresh
node with its own buffer. An ACTING step is the case that needs care: there
the state is both an input and an output of the same step, so one traced
program would read and write a single aliased buffer and the second replay
would be reading what the first replay had already overwritten.

The arrangement that keeps both engine invariants is two fixed buffers. The
captured program reads `prev` and writes a node of its own, and a
device-to-device copy after the replay returns moves the second into the
first. That copy stays OUTSIDE the captured region and reads nothing to host.

THIS IS NOT A LAYER. `ttrl.engine.recurrent` holds the differentiable
modules; this is device-memory lifecycle that happens to serve recurrence and
would serve anything else carrying a value from one step to the next, so the
two live apart.

WHERE THE EPISODE RESET GOES. Inside the captured step, not here. The step
reads `start`, a bound `(lanes, 1)` input this class also owns, and applies
`ttrl.engine.ops.apply_done_mask` to `prev` before the cell consumes it, which
is the same op in the same position as the training unroll's. So the acting
and training semantics cannot drift apart: they are one placement of one
operator, rather than two placements that agree because somebody compared
them. The alternative, masking `prev` in place here, cannot use
`apply_done_mask` at all, because that op allocates its output and assigning
it to `prev` would rebind the buffer a captured trace replays against. It
would need a second, raw-ttnn mask path, in which a one-step misplacement is
invisible.
"""

from __future__ import annotations

import numpy as np

from ttrl.engine.autograd import Tensor
from ttrl.engine.step.inputs import InputBinder
from ttrl.engine.tensor import copy_tensor_in_place, restore_in_place, to_host

__all__ = ["ActingCarry"]


class ActingCarry:
    """One fixed state buffer plus the episode-start flag that resets it.

    `prev` is the state the step reads, `(lanes, size)`, allocated once and
    zeroed. `bind` records the state the step produced and `commit` copies it
    back. `prev` and `start` keep their addresses for this object's lifetime
    and `commit` checks `prev`, because a rebind is the one failure here that
    corrupts silently: a captured trace keeps replaying against the address it
    recorded, so a carry whose `prev` was replaced advances a buffer the trace
    no longer reads. The step's OUTPUT is whatever the step produced, fixed
    under a trace and fresh per call when run eagerly, which is why only one
    side of the pair is pinned.
    """

    def __init__(self, size: int, device, *, lanes: int, name: str = "h"):
        if size < 1 or lanes < 1:
            raise ValueError(
                f"ActingCarry: size and lanes must be >= 1, got size={size}, "
                f"lanes={lanes}"
            )
        self.size = int(size)
        self.lanes = int(lanes)
        self.name = str(name)
        self.device = device
        self.prev = Tensor.from_numpy(
            np.zeros((self.lanes, self.size), dtype=np.float32), device,
            requires_grad=False)
        self.start = Tensor.from_numpy(
            np.zeros((self.lanes, 1), dtype=np.float32), device,
            requires_grad=False)
        self._binder = InputBinder(device, {"start": self.start})
        self._next: Tensor | None = None
        self._prev_id = id(self.prev.data)

    # -- the flag the step masks with ---------------------------------------

    def load_start(self, flags) -> None:
        """Write the episode-start flags for the step about to run.

        `flags` is lane-major and 1.0 where that lane's PREVIOUS step ended an
        episode, which is what makes the state entering this step a fresh one.
        It goes through the same `InputBinder` path every other bound input
        uses, so the buffer is written and never rebound.
        """
        arr = np.asarray(flags, dtype=np.float32).reshape(-1)
        if arr.shape != (self.lanes,):
            raise ValueError(
                f"ActingCarry({self.name}).load_start: expected one flag per "
                f"lane ({self.lanes},), got shape {arr.shape}"
            )
        self._binder.load("start", arr.reshape(self.lanes, 1))

    # -- the carry ----------------------------------------------------------

    @property
    def bound(self) -> bool:
        return self._next is not None

    def bind(self, h_next: Tensor) -> None:
        """Record the state the step produced, which `commit` copies back.

        Under a captured trace this is the same buffer every replay, because
        the replay writes the address it recorded. Run eagerly, each step
        allocates its own output, so this is called per step and the buffer
        legitimately differs. The asymmetry is the point: `prev` is the one
        address that must never move, and `commit` checks exactly that.
        """
        shape = tuple(int(d) for d in h_next.shape)
        if shape != (self.lanes, self.size):
            raise ValueError(
                f"ActingCarry({self.name}).bind: the step produced a state of "
                f"shape {shape}, this carry holds {(self.lanes, self.size)}"
            )
        self._next = h_next

    def commit(self) -> None:
        """Copy the step's output state into `prev`, in place.

        Device to device, outside any captured region, no host read. Runs
        after the replay has been ordered, which for a non-blocking replay is
        guaranteed by the metric readback on the same queue.
        """
        if self._next is None:
            raise RuntimeError(
                f"ActingCarry({self.name}).commit: nothing is bound. The step "
                f"must call bind() with the state it produced, or the carry "
                f"has no output to advance from."
            )
        self._check_address()
        copy_tensor_in_place(self.prev, self._next)

    def _check_address(self) -> None:
        if id(self.prev.data) != self._prev_id:
            raise RuntimeError(
                f"ActingCarry({self.name}): `prev` was rebound. A captured "
                f"trace reads the buffer address it recorded, so the step is "
                f"now reading the old buffer while this one is being copied "
                f"into. Write it in place with copy_tensor_in_place."
            )

    # -- host side, between steps -------------------------------------------

    def numpy(self) -> np.ndarray:
        """`prev` on the host, `(lanes, size)`.

        A host read, so never from inside a step. Its caller is the rollout
        storage, which records the state entering each sequence's first step,
        once every `seq_len` steps.
        """
        return to_host(self.prev.data)[: self.lanes, : self.size]

    def zero(self) -> None:
        """Clear the carry. A host write, so between rollouts and never inside
        a step."""

        restore_in_place(self.prev.data,
                         np.zeros((self.lanes, self.size), dtype=np.float32))

    def close(self) -> None:
        """Release the staging behind `load_start`. Idempotent."""
        self._binder.close()
