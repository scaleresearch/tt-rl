"""`SyncVectorEnv`'s autoreset contract, seeding, refusals and state."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.envs import FINAL_OBS_KEY, check_vec_env
from ttrl.envs.vector import SyncVectorEnv
from ttrl.spaces import ActionSpec, ObsSpec

SPEC = ObsSpec.vector(2)
ACTIONS = ActionSpec.discrete(3)


class StepCounter:
    """Ends an episode every `period` steps. Observation encodes (episode, t)."""

    def __init__(self, period: int = 3, *, spec: ObsSpec = SPEC,
                 action_spec: ActionSpec = ACTIONS, terminate: bool = True):
        self.obs_spec = spec
        self.action_spec = action_spec
        self.period = period
        self.terminate = terminate
        self.episode = 0
        self.t = 0
        self.closed = False
        self.seeds: list[int | None] = []

    def _obs(self):
        return np.array([self.episode, self.t], dtype=np.float32)

    def reset(self, *, seed=None):
        self.seeds.append(seed)
        if seed is not None:
            self.episode = seed  # a legible stand-in for a seeded stream
        else:
            self.episode += 1
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        done = self.t % self.period == 0
        obs = self._obs()
        # The contract requires episode statistics on the terminal step; one
        # unit of reward per step makes the return the episode length.
        info = {"episode": {"r": float(self.period), "l": self.period}} if done else {}
        return obs, 1.0, done and self.terminate, done and not self.terminate, info

    def close(self):
        self.closed = True


def _fns(n, **kwargs):
    return [lambda kw=kwargs: StepCounter(**kw) for _ in range(n)]


def test_it_honours_the_vector_contract():
    """`check_vec_env` is what gives the contract teeth: it drives enough
    steps to see an episode end and compares the final observation against
    the one returned, which is what separates NEXT_STEP from SAME_STEP."""
    with SyncVectorEnv(_fns(3)) as env:
        check_vec_env(env, steps=12)


def test_reset_seeds_each_lane_differently():
    """One seed for every lane makes the lanes replicas of each other, which
    shows up only as worse learning."""
    with SyncVectorEnv(_fns(4)) as env:
        env.reset(seed=10)
        assert [e.seeds[-1] for e in env.envs] == [10, 11, 12, 13]


def test_the_terminal_step_returns_the_final_observation_and_the_next_resets():
    """The whole NEXT_STEP contract, read off one boundary."""
    with SyncVectorEnv(_fns(1, period=2)) as env:
        env.reset(seed=0)
        obs, reward, term, trunc, info = env.step([0])
        assert not term[0] and not trunc[0]
        assert info[FINAL_OBS_KEY] == [None]

        obs, reward, term, trunc, info = env.step([0])
        assert term[0] and not trunc[0]
        assert reward[0] == 1.0
        final = info[FINAL_OBS_KEY][0]
        assert np.array_equal(final["obs"], obs["obs"][0])
        assert np.array_equal(final["obs"], [0, 2])  # episode 0, step 2

        # The reset step: the action is ignored, the reward is zero, neither
        # flag is set, and the observation is the new episode's first.
        obs, reward, term, trunc, info = env.step([0])
        assert not term[0] and not trunc[0]
        assert reward[0] == 0.0
        assert np.array_equal(obs["obs"][0], [1, 0])  # episode 1, step 0
        assert info[FINAL_OBS_KEY] == [None]


def test_truncation_is_reported_separately_from_termination():
    """They differ in the bootstrap mask and agree in everything else, which
    is what makes them the pair most easily conflated."""
    with SyncVectorEnv(_fns(1, period=2, terminate=False)) as env:
        env.reset(seed=0)
        env.step([0])
        _obs, _r, term, trunc, info = env.step([0])
        assert trunc[0] and not term[0]
        assert info[FINAL_OBS_KEY][0] is not None


def test_state_round_trip_carries_the_autoreset_phase():
    """A resume that loses `pending_reset` steps a lane that owed a reset,
    continuing an episode the checkpointed run had already ended."""
    with SyncVectorEnv(_fns(2, period=2)) as env:
        env.reset(seed=0)
        env.step([0, 0])
        env.step([0, 0])
        saved = env.save_state()
        assert saved["pending_reset"].tolist() == [True, True]

        env.step([0, 0])
        assert not env.pending_reset.any()

        env.load_state(saved)
        assert env.pending_reset.tolist() == [True, True]


def test_it_refuses_lanes_whose_specs_disagree():
    """Lane 0's spec is the one the networks and the replay buffer are built
    from, so a lane that quietly returns a different shape would be written
    into a buffer shaped for the other."""
    fns = [lambda: StepCounter(), lambda: StepCounter(spec=ObsSpec.vector(5))]
    with pytest.raises(ValueError, match="lane 1 declares obs_spec"):
        SyncVectorEnv(fns)

    fns = [lambda: StepCounter(), lambda: StepCounter(action_spec=ActionSpec.discrete(7))]
    with pytest.raises(ValueError, match="lane 1 declares action_spec"):
        SyncVectorEnv(fns)

