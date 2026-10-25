"""The samplers themselves, and the registry a name resolves through.

The contract is declared in `ttrl.buffers.window`: a sampler is handed a
`SampleWindow` and returns a `Draw` of positions inside it, so it cannot name
a ring column outside the positional bound. `UniformSampler` lives here
because it is that contract with nothing added: every lane's run, every
position in it equally likely. Prioritized replay is in
`ttrl.buffers.sampling`, the same contract with a weight per column.

A SAMPLER IS HOST-SIDE NUMPY. It runs between device steps and never inside a
captured trace, which is what lets a user write one in plain numpy with no
regard for address stability or for host reads. Reaching for ttnn here is
therefore a design error and `ttrl.buffers.checks.check_sampler` says so.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ttrl.buffers.indexing import draw_offsets
from ttrl.buffers.window import Draw, SampleWindow
from ttrl.nets.registry import Registry

__all__ = ["PriorityFeedback", "ReplaySampler", "SAMPLERS", "UniformSampler",
           "build_sampler"]


@runtime_checkable
class ReplaySampler(Protocol):
    """Which stored transitions a batch is drawn from, and with what weight.

    The replaceable half of replay: `ReplayBuffer` owns the storage, the ring
    arithmetic and the n-step read, and a sampler owns the distribution.
    Register one with `ttrl.SAMPLERS` and name it from a config, or hand an
    instance to `ReplayBuffer(sampler=...)`.

    Host-side numpy, always. See `ttrl.buffers.checks.check_sampler`.
    """

    def on_write(self, lane: int, col: int) -> None:
        """A transition has just been written at `(lane, col)`.

        Called once per write, before the write is sampleable, so a sampler
        keeping per-column state has somewhere to initialise it. A sampler
        keeping none implements this as a no-op.
        """
        ...

    def draw(self, rng, window: SampleWindow, *, batch_size: int,
             beta: float | None) -> Draw:
        """`batch_size` samples, as positions inside `window`.

        `rng` is the buffer's own `numpy.random.Generator` and is the ONLY
        source of randomness a sampler may use: a run's draw sequence is
        reproducible from the buffer's seed and is restored on resume, and a
        sampler reaching for the global `numpy.random` breaks both.

        `beta` is whatever the caller passed to `ReplayBuffer.sample`. An
        unbiased sampler refuses a beta rather than ignoring it, and one that
        owes importance-sampling weights requires it.
        """
        ...


@runtime_checkable
class PriorityFeedback(Protocol):
    """The optional half a sampler that learns from TD errors also implements.

    Implementing all three is what makes `ReplayBuffer.update_priorities`
    legal and what gets the sampler's table into the checkpoint, so a resumed
    run draws from the distribution it stopped on. A sampler with `update` and
    no `state` would silently resume from a blank table, so the three are one
    protocol rather than three sniffs.
    """

    def update(self, lanes: np.ndarray, cols: np.ndarray,
               priorities: np.ndarray) -> None: ...

    def state(self) -> dict: ...

    def load(self, state: dict) -> None: ...


class UniformSampler:
    """Uniform over the union of every lane's sampleable run.

    The quota is proportional to each lane's sampleable count and drawn
    multinomially. Because a lane's sampleable set is exactly that contiguous
    count with no further acceptance test, this is uniform over the union
    rather than an approximation of it.
    """

    def on_write(self, lane: int, col: int) -> None:
        """A uniform sampler keeps nothing per column."""

    def draw(self, rng, window: SampleWindow, *, batch_size: int,
             beta: float | None) -> Draw:
        if beta is not None:
            raise ValueError(
                "ReplayBuffer.sample: beta was given but this buffer samples "
                "uniformly, so there is no sampling bias for an "
                "importance-sampling weight to correct. Build the buffer with "
                "priority=PriorityConfig(...), or drop the beta."
            )
        quota = rng.multinomial(batch_size, window.counts / window.total)
        lanes, offsets = draw_offsets(rng, window.counts, quota)
        return Draw(lanes, offsets, None)


#: `builder(*, lanes, columns, config) -> ReplaySampler`. `config` is the
#: sampler's own settings object, which for the prioritized sampler is the
#: `PriorityConfig` a run's config already carries, and None for a sampler
#: that has no settings of its own. A sampler needing settings of another
#: shape registers a builder that closes over them, which keeps this registry
#: free of a per-sampler keyword blob and keeps a config file's `sampler`
#: field a plain name that round-trips through a checkpoint.
SAMPLERS = Registry("samplers", module="ttrl", allow_targets=True)

#: A target string rather than a direct registration: `ttrl.buffers.sampling`
#: imports this module for the declaration, so naming it here instead of
#: importing it keeps the dependency one-way.
SAMPLERS.register("priority", "ttrl.buffers.sampling:build_priority_sampler")


@SAMPLERS("uniform")
def build_uniform_sampler(*, lanes: int, columns: int, config=None) -> UniformSampler:
    if config is not None:
        raise ValueError(
            f"the 'uniform' sampler takes no settings, and one was given: "
            f"{config!r}. A PriorityConfig selects the 'priority' sampler, so "
            f"either drop sampler='uniform' or drop the priority block."
        )
    return UniformSampler()


def build_sampler(sampler, *, lanes: int, columns: int, config=None) -> ReplaySampler:
    """The sampler a buffer runs with: an instance, or a registered name.

    An INSTANCE is taken as it is, which is the programmatic escape hatch for
    a sampler being tried before it is worth naming. A NAME is what a config
    file carries and what round-trips through a checkpoint, so a run resumed
    from a file alone draws from the distribution it was trained with. There
    is deliberately no third form that carries a class with no name: unlike an
    encoder, a sampler builder imports numpy and nothing else, so registering
    one is a single line with no dependency cost to avoid, and the instance
    form already covers the experiment.

    `None` means the declared default, which is the prioritized sampler when
    the run gave priority settings and the uniform one otherwise.
    """
    if sampler is None:
        sampler = "uniform" if config is None else "priority"
    if isinstance(sampler, str):
        built = SAMPLERS.resolve(sampler)(lanes=lanes, columns=columns,
                                          config=config)
    else:
        built = sampler
    for method in ("on_write", "draw"):
        if not callable(getattr(built, method, None)):
            raise TypeError(
                f"{type(built).__name__} is not a ReplaySampler: it has no "
                f"callable {method!r}. A sampler implements on_write(lane, col) "
                f"and draw(rng, window, *, batch_size, beta); hold yours to the "
                f"whole contract with ttrl.check_sampler(sampler) before a run."
            )
    return built
