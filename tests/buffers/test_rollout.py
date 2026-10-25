"""RolloutBuffer: GAE against a closed form, and continuous-action storage."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.buffers import ReplayBuffer, RolloutBuffer
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec

VEC = ObsGroup.of(ObsSpec.vector(5))
DISC = ActionSpec.discrete(4)
CONT = ActionSpec.continuous([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0])
MULTI = ActionSpec.multi_discrete([3, 5])


def _closed_form_gae(rewards, values, dones, last_values, gamma, lam):
    """A_t = sum_{l>=0} (gamma*lam)^l delta_{t+l}, with the chain cut at a done."""
    T, N = rewards.shape
    out = np.zeros((T, N), np.float64)
    for n in range(N):
        for t in range(T):
            acc, coef = 0.0, 1.0
            for l in range(t, T):
                next_value = last_values[n] if l == T - 1 else values[l + 1][n]
                not_done = 1.0 - dones[l][n]
                delta = rewards[l][n] + gamma * next_value * not_done - values[l][n]
                acc += coef * delta
                coef *= gamma * lam * not_done
                if not_done == 0.0:
                    break
            out[t, n] = acc
    return out


@pytest.mark.parametrize("seed", range(3))
def test_gae_matches_closed_form(seed):
    T, N, gamma, lam = 7, 3, 0.95, 0.9
    rng = np.random.default_rng(seed)
    rewards = rng.normal(size=(T, N)).astype(np.float32)
    values = rng.normal(size=(T, N)).astype(np.float32)
    dones = (rng.random((T, N)) < 0.2).astype(np.float32)
    last = rng.normal(size=N).astype(np.float32)

    buf = RolloutBuffer(VEC, DISC, n_envs=N, n_steps=T, seed=seed)
    for t in range(T):
        buf.add(np.zeros((N, 5), np.float32), np.zeros(N, np.int64),
                np.zeros(N, np.float32), values[t], rewards[t], dones[t])
    buf.compute_gae(gamma, lam, last)

    reference = _closed_form_gae(rewards, values, dones, last, gamma, lam)
    assert np.abs(buf.advantages - reference).max() < 1e-5
    # returns is advantages + values exactly, with no reaccumulation.
    assert np.abs(buf.returns - (buf.advantages + values)).max() == 0.0


def test_autoreset_padding_cuts_the_chain_and_carries_no_advantage():
    buf = RolloutBuffer(VEC, DISC, n_envs=1, n_steps=4, seed=0)
    for t in range(4):
        buf.add(np.zeros((1, 5), np.float32), np.zeros(1, np.int64),
                np.zeros(1, np.float32), np.zeros(1, np.float32),
                np.ones(1, np.float32), np.zeros(1, np.float32),
                valid=np.array([0.0 if t == 2 else 1.0], np.float32))
    buf.compute_gae(1.0, 1.0, np.zeros(1, np.float32))
    # gamma = lam = 1 and reward 1 everywhere, so without the cut the
    # advantages would be [4, 3, 2, 1].
    assert buf.advantages.reshape(-1).tolist() == [2.0, 1.0, 0.0, 1.0]


def test_padding_slots_never_reach_a_minibatch():
    buf = RolloutBuffer(VEC, DISC, n_envs=2, n_steps=4, seed=0)
    for t in range(4):
        obs = np.stack([np.full(5, t, np.float32), np.full(5, 100 + t, np.float32)])
        buf.add(obs, np.zeros(2, np.int64), np.zeros(2, np.float32),
                np.zeros(2, np.float32), np.ones(2, np.float32),
                np.zeros(2, np.float32),
                valid=np.array([1.0, 0.0 if t == 1 else 1.0], np.float32))
    buf.compute_gae(0.99, 0.95, np.zeros(2, np.float32))
    drawn = np.concatenate([m.obs["obs"][:, 0] for m in buf.minibatches(3, epochs=3)])
    assert 101.0 not in set(drawn.tolist())


# ---------------------------------------------------------------------------
# continuous actions
# ---------------------------------------------------------------------------

def test_continuous_replay_round_trip_is_bit_exact():
    buf = ReplayBuffer(VEC, CONT, capacity=20, n_envs=2, n_step=2, gamma=0.9, seed=4)
    assert buf.actions.shape == (2, 10, 3)
    for t in range(12):
        for e in range(2):
            stamp = float(e * 100 + t)
            buf.add(np.full(5, stamp, np.float32), np.full(3, stamp, np.float32),
                    1.0, False, False, lane=e)
    batch = buf.sample(500)
    assert batch.actions.shape == (500, 3)
    # Each action vector belongs to its own transition's observation.
    assert np.array_equal(batch.actions[:, 0], batch.obs["obs"][:, 0])
    assert np.array_equal(batch.actions[:, 2], batch.obs["obs"][:, 0])
