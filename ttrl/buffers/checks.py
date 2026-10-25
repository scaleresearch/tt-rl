"""The sampler contract's refusals: an executable test of `ReplaySampler`.

`ttrl.buffers.samplers` declares what a sampler is handed and may hand back,
and this runs one against that declaration, the way `ttrl.check_agent` runs an
agent against `Agent` and `ttrl.check_env` runs an environment against `Env`.
Call it once, before a run: every mistake it names is otherwise discovered as
a number that trains slightly wrong for several hours.

It drives the HOSTILE window on purpose. The runs it offers wrap the ring and
start away from column 0, one lane is empty, and the counts differ between
lanes, because a sampler written against `start=0` and one run per lane passes
every gentle test and then draws the write cursor's neighbourhood on the first
lap. The empty lane needs no refusal of its own: a lane offering nothing has
no position a draw can name, so `SampleWindow.locate` already refuses it.
"""

from __future__ import annotations

import sys

import numpy as np

from ttrl.buffers.window import SampleWindow

__all__ = ["check_sampler"]

#: Columns in the pretend ring the checks below draw from.
_CAPACITY = 16


def check_sampler(sampler, *, batch_size: int = 8, beta: float | None = None) -> None:
    """Hold `sampler` to the `ReplaySampler` contract, or raise saying how.

    `beta` is what a run would pass to `ReplayBuffer.sample`: leave it None
    for an unbiased sampler and pass a float for one that returns
    importance-sampling weights, because which of the two a sampler is
    changes what a correct `draw` does with it.
    """
    who = type(sampler).__name__
    for method in ("on_write", "draw"):
        if not callable(getattr(sampler, method, None)):
            raise TypeError(
                f"{who} is not a ReplaySampler: it has no callable "
                f"{method!r}. A sampler implements on_write(lane, col), called "
                f"once per stored transition, and "
                f"draw(rng, window, *, batch_size, beta) -> Draw."
            )
    window = _window()
    for lane in range(window.n_lanes):
        for col in range(_CAPACITY):
            sampler.on_write(lane, col)

    before = set(sys.modules)
    first = _drive(sampler, window, batch_size, beta)
    _no_ttnn(sampler, who, before)

    # The rng is the only source of randomness: a run's draw sequence is
    # reproducible from the buffer's seed and is restored on resume, and a
    # sampler reaching for the global `numpy.random` breaks both silently.
    second = _drive(sampler, window, batch_size, beta)
    if not (np.array_equal(first[0], second[0])
            and np.array_equal(first[1], second[1])):
        raise AssertionError(
            f"{who}.draw returned different samples for two identical windows "
            f"drawn with identically seeded generators, so it is reading "
            f"randomness from somewhere other than the `rng` it was handed. "
            f"Draw everything from `rng` (numpy.random.Generator): a run's "
            f"sampling has to be reproducible from the buffer's seed and has "
            f"to resume from the generator state the checkpoint restored."
        )

    _check_feedback(sampler, who)


def _window() -> SampleWindow:
    """A wrapped, uneven, partly empty set of runs. See the module docstring."""
    return SampleWindow(counts=np.array([11, 0, 4], dtype=np.int64),
                        start=np.array([9, 0, 13], dtype=np.int64),
                        capacity=_CAPACITY)


def _drive(sampler, window, batch_size, beta):
    draw = sampler.draw(np.random.default_rng(0), window,
                        batch_size=batch_size, beta=beta)
    # `locate` is the enforcement a real batch goes through, so the contract
    # is checked by the same code rather than by a second copy of the rules.
    return window.locate(draw, batch_size=batch_size, sampler=sampler)


def _no_ttnn(sampler, who: str, before: set) -> None:
    """Refuse a sampler that reaches for ttnn.

    A sampler is host-side numpy between device steps, and one holding device
    tensors would be holding them across a trace capture it knows nothing
    about. What is checkable in-process is the sampler's own module namespace
    and any import its first draw performed; a module-scope `import ttnn` in a
    process that already imported ttnn is not visible here, which is why
    `tests/buffers/test_host_only.py` asks the same question of this package
    in a subprocess with the import poisoned.
    """
    new = [name for name in set(sys.modules) - before
           if name == "ttnn" or name.startswith("ttnn.")]
    namespace = getattr(getattr(sampler, "draw", None), "__globals__", {})
    named = [key for key, value in namespace.items()
             if getattr(value, "__name__", "") == "ttnn"
             or str(getattr(value, "__module__", "")).startswith("ttnn")]
    if new or named:
        raise AssertionError(
            f"{who} reaches for ttnn ({new or named}), and a sampler is "
            f"host-side numpy. It runs between device steps, where it may read "
            f"the host freely, and it hands back positions rather than tensors; "
            f"a device tensor built here would be rebound per batch, which is "
            f"what a captured trace cannot see. Keep the draw in numpy."
        )


def _check_feedback(sampler, who: str) -> None:
    """A sampler that learns from TD errors implements all three of the
    feedback methods or none, so a resume cannot restore a blank table."""
    present = [name for name in ("update", "state", "load")
               if callable(getattr(sampler, name, None))]
    if present and len(present) != 3:
        raise TypeError(
            f"{who} implements {present} of the PriorityFeedback contract, "
            f"which is update(lanes, cols, priorities), state() and "
            f"load(state). A sampler that takes TD errors and cannot save its "
            f"table resumes a run from a blank one and draws from a "
            f"distribution the checkpoint never had. Implement all three, or "
            f"none to be treated as a sampler with no feedback."
        )
