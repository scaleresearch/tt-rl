"""ReplayBuffer against an INDEPENDENT oracle, not against itself."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.buffers import ReplayBuffer
from ttrl.buffers.indexing import n_step_forward, sampleable
from ttrl.buffers.replaybatch import successor_obs
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec

# 1-element observations: this file cares about WHICH observation lands where,
# never about its contents, and a 1-element observation lets the write index be
# read straight back out of the value.
OBS = ObsGroup.of(ObsSpec(shape=(1,), dtype="float32", kind="vector"))
ACT = ActionSpec.discrete(5)


def _obs(t: int) -> np.ndarray:
    return np.full((1,), float(t), dtype=np.float32)


def _decode(a) -> int:
    return int(np.asarray(a).reshape(-1)[0])


def _drive(capacity, n_step, gamma, n_writes, seed, p_end):
    """Drive one lane and record the SAME writes as a flat linear history."""
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer(OBS, ACT, capacity=capacity, n_step=n_step, gamma=gamma, seed=seed)
    hist: list[dict] = []
    while len(hist) < n_writes:
        t = len(hist)
        x = rng.random()
        done = x < 0.6 * p_end
        trunc = (not done) and x < p_end
        reward = float(np.float32(rng.normal()))
        # The episode-final observation is deliberately NOT observation t+1:
        # it is a distinct value, so a buffer that wrongly reads the
        # neighbouring slot at a boundary cannot accidentally agree.
        final = _obs(10_000 + t) if (done or trunc) else None
        buf.add(_obs(t), int(rng.integers(0, 5)), reward, done, trunc, final_obs=final)
        hist.append({"r": reward, "done": done, "end": done or trunc,
                     "final": None if final is None else _decode(final)})
    return buf, hist


def _expect(hist, capacity, n_step, gamma, t):
    """What transition `t` must contain, from the flat history alone."""
    g, disc, steps, terminated, j = 0.0, 1.0, 0, False, t
    for k in range(n_step):
        j = t + k
        g += disc * hist[j]["r"]
        disc *= gamma
        steps += 1
        if hist[j]["end"]:
            terminated = hist[j]["done"]
            break
    successor = hist[j]["final"] if hist[j]["end"] else j + 1
    return g, successor, terminated, steps


def _live_column_to_write(buf, n_writes):
    """`ring column -> flat write index`, for the writes the ring still holds."""
    per_lane = buf.per_lane_capacity
    oldest = max(0, n_writes - per_lane)
    return {t % per_lane: t for t in range(oldest, n_writes)}


@pytest.mark.parametrize("seed", range(10))
def test_n_step_return_and_successor_against_flat_history(seed):
    rng = np.random.default_rng(1000 + seed)
    n_step = int(rng.integers(1, 5))
    capacity = int(rng.integers(n_step + 3, 40))
    n_writes = int(rng.integers(1, 3 * capacity))
    gamma = 0.5
    buf, hist = _drive(capacity, n_step, gamma, n_writes, seed, p_end=0.2)

    start, count = sampleable(buf._pos, buf._full, buf.per_lane_capacity, n_step)
    if count[0] <= 0:
        pytest.skip("no sampleable transition at this size")
    cols = (start[0] + np.arange(count[0])) % buf.per_lane_capacity
    lanes = np.zeros_like(cols)
    returns, end_cols, terminated, steps = n_step_forward(
        buf.rewards, buf.ends, buf.dones, lanes, cols,
        n_step=n_step, gamma=gamma, capacity=buf.per_lane_capacity,
    )
    successors = successor_obs(buf, lanes, end_cols)
    col_to_write = _live_column_to_write(buf, n_writes)

    for b, col in enumerate(int(c) for c in cols):
        t = col_to_write[col]
        want_g, want_successor, want_done, want_steps = _expect(
            hist, buf.per_lane_capacity, n_step, gamma, t)
        # Exact, not approximate: gamma is 0.5 and the accumulation order is
        # the oracle's order, so every expectation is representable in binary.
        assert float(returns[b]) == want_g, (seed, t)
        assert bool(terminated[b]) == want_done, (seed, t)
        assert int(steps[b]) == want_steps, (seed, t)
        assert _decode(successors["obs"][b]) == want_successor, (seed, t)


@pytest.mark.parametrize("seed", range(3))
def test_public_sample_matches_flat_history(seed):
    """The same expectations through `sample()`, not through internals."""
    n_step, capacity, gamma = 3, 24, 0.5
    n_writes = 40
    buf, hist = _drive(capacity, n_step, gamma, n_writes, seed, p_end=0.25)
    col_to_write = _live_column_to_write(buf, n_writes)
    start, count = sampleable(buf._pos, buf._full, buf.per_lane_capacity, n_step)
    sampleable_cols = {int((start[0] + k) % buf.per_lane_capacity)
                       for k in range(int(count[0]))}
    expected = {col_to_write[col]: _expect(hist, capacity, n_step, gamma,
                                           col_to_write[col])
                for col in sampleable_cols}

    batch = buf.sample(600)
    seen = set()
    for b in range(len(batch)):
        t = _decode(batch.obs["obs"][b])
        assert t in expected, t
        want_g, want_successor, want_done, want_steps = expected[t]
        assert batch.rewards[b] == np.float32(want_g), t
        assert batch.dones[b] == float(want_done), t
        assert batch.discounts[b] == np.float32(gamma ** want_steps), t
        assert _decode(batch.next_obs["obs"][b]) == want_successor, t
        seen.add(t)
    assert seen, "sampled nothing"


def test_boundary_successor_is_the_recorded_final_observation():
    """A window ending on a boundary must not read the neighbouring slot."""
    buf = ReplayBuffer(OBS, ACT, capacity=16, n_step=1, gamma=0.9, seed=0)
    buf.add(_obs(0), 0, 1.0, False, False)
    buf.add(_obs(1), 0, 1.0, True, False, final_obs=_obs(999))
    buf.add(_obs(2), 0, 1.0, False, False)  # first step of the new episode
    buf.add(_obs(3), 0, 1.0, False, False)
    batch = buf.sample(400)
    by_state = {_decode(batch.obs["obs"][b]): _decode(batch.next_obs["obs"][b])
                for b in range(len(batch))}
    assert by_state[0] == 1, by_state
    assert by_state[1] == 999, by_state      # the final obs, NOT observation 2
    assert by_state[2] == 3, by_state

