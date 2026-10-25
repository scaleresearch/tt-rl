"""The sampler seam, exercised by a sampler this library knows nothing about.

Two claims. First, that a third sampler can be written, registered, checked
and drawn from through the public surface, which is what the seam is for.
Second, and the reason the seam hands over a window rather than the ring: that
every way of getting a draw wrong is refused by name. A sampler is where a
mistake becomes slightly wrong training data for several hours rather than a
crash, so the refusals are tested as the SET of them: the set is read off the
`raise` statements in `ttrl/buffers/window.py` and `ttrl/buffers/checks.py`
and every one of them has to be reached by a case below, which is checked by
tracing rather than by a list someone has to remember to extend.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import types

import numpy as np
import pytest

import ttrl
from ttrl.buffers.samplers import Draw, UniformSampler

OBS = ttrl.ObsGroup.of(ttrl.ObsSpec.vector(1))
ACT = ttrl.ActionSpec.discrete(2)

SEAM = [pathlib.Path(ttrl.__file__).parent / "buffers" / name
        for name in ("window.py", "checks.py")]


class RecencySampler:
    """A user's sampler: newer transitions likelier, decaying with age.

    Nothing in ttrl anticipates this one, and it needs no state of its own:
    age is a position in the window, which is the one ordering fact the seam
    promises.
    """

    def __init__(self, half_life: int = 16):
        self.decay = 0.5 ** (1.0 / float(half_life))

    def on_write(self, lane, col):
        """Recency is positional, so nothing is kept per column."""

    def _run(self, n):
        return self.decay ** np.arange(n - 1, -1, -1, dtype=np.float64)

    def draw(self, rng, window, *, batch_size, beta):
        runs = [self._run(window.count(lane)) for lane in range(window.n_lanes)]
        mass = np.array([run.sum() for run in runs])
        quota = rng.multinomial(batch_size, mass / mass.sum())
        lanes, offsets = [], []
        for lane, k in enumerate(int(v) for v in quota):
            if k == 0:
                continue
            cdf = np.cumsum(runs[lane])
            picked = np.searchsorted(cdf, rng.random(k) * cdf[-1], side="right")
            lanes.append(np.full(k, lane, dtype=np.int64))
            offsets.append(np.minimum(picked, cdf.shape[0] - 1).astype(np.int64))
        return Draw(np.concatenate(lanes), np.concatenate(offsets), None)


def _build_recency(*, lanes, columns, config=None):
    return RecencySampler()


@pytest.fixture
def registered():
    ttrl.SAMPLERS.register("test-recency", _build_recency)
    try:
        yield "test-recency"
    finally:
        ttrl.SAMPLERS.unregister("test-recency")


def _driven(sampler, writes=400, n_envs=2):
    # Imported per call, not at module scope: `test_host_only` pops
    # `ttrl.buffers.*` out of sys.modules and imports it again, which builds a
    # second SAMPLERS registry, and a ReplayBuffer bound before that would
    # resolve names against the registry this file no longer registers into.
    from ttrl.buffers import ReplayBuffer

    buf = ReplayBuffer(OBS, ACT, capacity=512, n_envs=n_envs, n_step=1,
                       gamma=0.99, seed=0, sampler=sampler)
    for t in range(writes):
        buf.add_batch(np.full((n_envs, 1), float(t), np.float32),
                      np.zeros(n_envs, np.int64), np.zeros(n_envs, np.float32),
                      np.zeros(n_envs, bool))
    return buf


def test_a_users_own_sampler_works_through_the_public_surface(registered):
    """Registered by name, held to the contract, named in a config, drawn from."""
    ttrl.check_sampler(RecencySampler())
    assert registered in ttrl.SAMPLERS

    # The name is what a run carries, so it has to survive a checkpoint.
    from ttrl.train.checkpoint.tree import join, split

    arrays: dict = {}
    cfg = ttrl.DQNConfig(sampler=registered)
    restored = join(split(cfg, "config", arrays), arrays,
                    file="<memory>")
    assert restored.sampler == registered and restored == cfg

    drawn = _driven(registered).sample(20_000).obs["obs"].reshape(-1)
    # The draw bends toward new data, and still never leaves the sampleable
    # run: the newest write has no stored successor, so it must not appear.
    newest = float(np.mean(drawn >= 360))
    oldest = float(np.mean(drawn < 200))
    assert newest > 3 * oldest, f"newest tenth {newest}, oldest half {oldest}"
    assert drawn.max() <= 398.0


# -- the refusals -----------------------------------------------------------

class _Faulty(UniformSampler):
    """Uniform, with one fault injected into what it hands back."""

    def __init__(self, fault):
        self.fault = fault

    def draw(self, rng, window, *, batch_size, beta):
        if self.fault == "its own randomness":
            return Draw(np.zeros(batch_size, np.int64),
                        np.random.randint(0, window.count(0),
                                          batch_size).astype(np.int64), None)
        # `beta` is dropped on purpose: a uniform draw refuses one, and
        # these cases are about what a draw hands back.
        draw = UniformSampler.draw(self, rng, window, batch_size=batch_size,
                                   beta=None)
        fault, weights = self.fault, None
        if fault == "an offset outside the bound":
            return Draw(draw.lanes, np.full(batch_size, window.count(0),
                                            np.int64), None)
        if fault == "too few samples":
            return Draw(draw.lanes[:-1], draw.offsets[:-1], None)
        if fault == "float offsets":
            return Draw(draw.lanes, draw.offsets.astype(np.float64), None)
        if fault == "a lane that does not exist":
            return Draw(draw.lanes + window.n_lanes, draw.offsets, None)
        if fault == "two-dimensional offsets":
            return Draw(draw.lanes, draw.offsets.reshape(-1, 1), None)
        if fault == "more lanes than offsets":
            return Draw(np.concatenate([draw.lanes, draw.lanes[:1]]),
                        draw.offsets, None)
        if fault == "integer weights":
            weights = np.ones(batch_size, np.int64)
        if fault == "too few weights":
            weights = np.ones(batch_size - 1, np.float32)
        if fault == "an infinite weight":
            weights = np.full(batch_size, np.inf, np.float32)
        if fault == "not a Draw at all":
            return draw.lanes
        return Draw(draw.lanes, draw.offsets, weights)


class _HalfFeedback(UniformSampler):
    """Takes TD errors and cannot save them, so a resume loses the table."""

    def update(self, lanes, cols, priorities):
        """Half of `PriorityFeedback`, which is the half that loses data."""


_TTNN_SAMPLER = """
import numpy as np
from ttrl.buffers.samplers import UniformSampler


class TouchesTtnn(UniformSampler):
    def draw(self, rng, window, *, batch_size, beta):
        _ = ttnn
        return UniformSampler.draw(self, rng, window, batch_size=batch_size,
                                   beta=None)
"""


def _ttnn_sampler():
    """A sampler whose module imported ttnn, built without the wheel.

    `ttnn` is not installed on a host, and the question is about the user's
    module namespace rather than about the import system, so the module is
    assembled with a stub already in its globals. The import-system half of
    the same question is `tests/buffers/test_host_only.py`.
    """
    module = types.ModuleType("fake_user_sampler")
    module.__dict__["ttnn"] = types.ModuleType("ttnn")
    exec(compile(_TTNN_SAMPLER, "<fake_user_sampler>", "exec"), module.__dict__)
    return module.TouchesTtnn()


class _NoDraw:
    def on_write(self, lane, col):
        """A sampler that keeps state and never draws is not a sampler."""


#: fault -> (the sampler carrying it, the phrase its refusal has to name).
#: Every `raise` in the seam has to be reached by one of these; see
#: `test_every_refusal_in_the_seam_is_reached`.
CASES = {
    "an offset outside the bound": "0 <= offset < window.count(lane)",
    "too few samples": "draw exactly batch_size",
    "float offsets": "must be an integer",
    "two-dimensional offsets": "must be a 1-D array",
    "more lanes than offsets": "there must be one offset per lane entry",
    "a lane that does not exist": "never an index into the ring",
    "integer weights": "an importance-sampling weight is a float",
    "too few weights": "one weight per sampled transition",
    "an infinite weight": "non-finite importance-sampling weight",
    "not a Draw at all": "must return (lanes, offsets, weights)",
    "its own randomness": "reproducible from the buffer's seed",
}

#: The same, for faults that are not a bad `draw` return.
OTHER_CASES = {
    "no draw method at all": (_NoDraw(), "is not a ReplaySampler"),
    "feedback it cannot checkpoint": (_HalfFeedback(), "PriorityFeedback"),
    "a sampler that reaches for ttnn": (_ttnn_sampler, "host-side numpy"),
}


def _sampler_for(name):
    if name in CASES:
        return _Faulty(name)
    sampler = OTHER_CASES[name][0]
    return sampler() if callable(sampler) and not hasattr(sampler, "draw") \
        else sampler


def _phrase_for(name):
    return CASES[name] if name in CASES else OTHER_CASES[name][1]


@pytest.mark.parametrize("fault", sorted(CASES) + sorted(OTHER_CASES))
def test_check_sampler_refuses_each_fault_by_name(fault):
    with pytest.raises((ValueError, TypeError, AssertionError)) as caught:
        ttrl.check_sampler(_sampler_for(fault))
    assert _phrase_for(fault) in str(caught.value)


@pytest.mark.parametrize("fault", ["an offset outside the bound",
                                   "too few samples", "float offsets"])
def test_a_live_sample_refuses_the_same_faults(fault):
    """The enforcement is in the draw path and not only in the checker, so a
    user who never ran `check_sampler` still cannot train on a column the
    write cursor has overwritten."""
    with pytest.raises((ValueError, TypeError)):
        _driven(_Faulty(fault), writes=40).sample(8)


def _raise_lines() -> set:
    """Every `raise` statement in the seam, read off the source."""
    found = set()
    for path in SEAM:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise):
                found.add((path.name, node.lineno))
    return found


def test_every_refusal_in_the_seam_is_reached():
    """The set of faults is DERIVED from the code, by tracing which refusals
    the cases above actually reach.

    A rule added to `locate` or to `check_sampler` with no case above is
    covered by nothing, which is how a contract quietly stops being one. A
    phrase list cannot see that; which `raise` ran can.
    """
    wanted = _raise_lines()
    names = {path.name for path in SEAM}
    hit = set()

    def tracer(frame, event, _arg):
        name = pathlib.Path(frame.f_code.co_filename).name
        if name not in names:
            return None
        if event == "line":
            hit.add((name, frame.f_lineno))
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        for fault in list(CASES) + list(OTHER_CASES):
            try:
                ttrl.check_sampler(_sampler_for(fault))
            except (ValueError, TypeError, AssertionError):
                pass
    finally:
        sys.settrace(old)
    assert not wanted - hit, (
        f"refusals in {[p.name for p in SEAM]} that no case above reaches: "
        f"{sorted(wanted - hit)}. Add the fault that triggers each one to "
        f"CASES; a refusal nothing exercises is a message no one has read."
    )
