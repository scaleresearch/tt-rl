"""The checkers against an environment that refuses an illegal action."""

from __future__ import annotations

import numpy as np
import pytest

import ttrl
from ttrl.envs import SyncVectorEnv, check_env, check_vec_env
from ttrl.envs.vec_normalize import NormalizeObservation
from ttrl.envs.vector import SyncVectorEnv
from ttrl.envs.wrappers import ActionRepeat, FrameStack, RecordEpisodeStatistics
from ttrl.spaces import mask_rows

FOLD, CALL, BET = 0, 1, 2
N_ACTIONS = 3
OBS_DIM = 3 + 6 + N_ACTIONS

HISTORY_STATES = {
    (): 0, (CALL,): 1, (BET,): 2, (CALL, BET): 3, (CALL, CALL): 4, (BET, CALL): 5,
}


def legal_mask(history) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    if history and history[-1] == BET:
        mask[FOLD] = mask[CALL] = True
    else:
        mask[CALL] = mask[BET] = True
    return mask


def obs_for(card: int, history) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[card] = 1.0
    obs[3 + HISTORY_STATES.get(history, 0)] = 1.0
    obs[9:] = legal_mask(history)
    return obs


class KuhnPoker:
    """Kuhn poker, player 0 learning. An illegal action is a hard error."""

    #: Three actions at most in one hand, so the bound `check_env` asks for.
    episode_bound = 3

    def __init__(self, seed: int = 0):
        self.obs_spec = ttrl.ObsSpec.vector(OBS_DIM, action_mask=N_ACTIONS)
        self.action_spec = ttrl.ActionSpec.discrete(N_ACTIONS)
        self.rng = np.random.default_rng(seed)
        self.illegal_count = 0

    def reset(self, *, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        deck = self.rng.permutation(3)
        self.cards = [int(deck[0]), int(deck[1])]
        self.history = ()
        return obs_for(self.cards[0], self.history), {}

    def _payoff_p0(self) -> float:
        if self.history == (BET, FOLD):
            return 1.0
        if self.history == (CALL, BET, FOLD):
            return -1.0
        win = 1 if self.cards[0] > self.cards[1] else -1
        return float(win) * (2.0 if BET in self.history else 1.0)

    def _terminal(self) -> bool:
        h = self.history
        return (len(h) >= 2 and h[-1] == FOLD) or h in (
            (CALL, CALL), (BET, CALL), (CALL, BET, CALL)
        )

    def step(self, action):
        a = int(action)
        mask = legal_mask(self.history)
        if not mask[a]:
            self.illegal_count += 1
            raise ValueError(
                f"illegal action {a} in history {self.history}; "
                f"legal: {np.flatnonzero(mask).tolist()}"
            )
        self.history = self.history + (a,)
        if self._terminal():
            return obs_for(self.cards[0], self.history), self._payoff_p0(), True, False, {}
        opponent = int(self.rng.choice(np.flatnonzero(legal_mask(self.history))))
        self.history = self.history + (opponent,)
        if self._terminal():
            return obs_for(self.cards[0], self.history), self._payoff_p0(), True, False, {}
        return obs_for(self.cards[0], self.history), 0.0, False, False, {}

    def close(self):
        pass


def make_kuhn(**kwargs):
    return RecordEpisodeStatistics(KuhnPoker(**kwargs))


def test_a_strict_masked_env_passes_check_env_with_no_illegal_action():
    env = make_kuhn()
    check_env(env, steps=200)
    assert env.unwrapped.illegal_count == 0


def test_a_strict_masked_env_passes_check_vec_env_with_no_illegal_action():
    with SyncVectorEnv([make_kuhn] * 3) as vec:
        check_vec_env(vec, steps=200)
        assert [e.unwrapped.illegal_count for e in vec.envs] == [0, 0, 0]


class DeadEnd(KuhnPoker):
    """Presents a state with no legal action, which is not a state at all."""

    def reset(self, *, seed=None):
        obs, info = super().reset(seed=seed)
        obs[9:] = 0.0
        return obs, info


def test_an_observation_with_no_legal_action_is_refused():
    """`draw_legal` has no answer for an all-illegal row, so the checker names
    the fault rather than drawing action 0 and blaming the environment."""
    with pytest.raises(ValueError, match="every action illegal"):
        check_env(DeadEnd(), steps=4)


def test_action_repeat_refuses_a_masked_env():
    """Holding one action over several states replays it into states that may
    forbid it, which is the one way an illegal action could still reach a
    masked environment from inside the library."""

    with pytest.raises(ValueError, match="legal-action mask"):
        ActionRepeat(KuhnPoker(), 4)


def test_normalize_observation_refuses_a_masked_env():
    """Whitening the observation would rewrite the mask indicators."""

    with pytest.raises(ValueError, match="whiten the legal-action mask"):
        NormalizeObservation(SyncVectorEnv([KuhnPoker]))


def test_frame_stack_keeps_the_newest_mask_in_the_tail():
    """Stacking a masked vector is safe: the newest observation is last, so
    the final entries of the stack are still its mask."""

    env = FrameStack(KuhnPoker(), 2)
    obs, _ = env.reset(seed=0)
    assert env.obs_spec.shape == (2 * OBS_DIM,)
    assert np.array_equal(
        mask_rows(obs[None], env.obs_spec)[0], legal_mask(()).astype(np.float32)
    )
