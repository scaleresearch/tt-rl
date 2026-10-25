"""The ALE-only wrappers, against a scripted env and a fake emulator."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.envs.atari import (
    AtariConfig,
    EpisodicLife,
    FireReset,
    MaxAndSkip,
    NoopReset,
)
from ttrl.spaces import ActionSpec, ObsSpec

FRAME_SHAPE = (3, 2, 2)


class ScriptedEnv:
    """An `Env` whose step results are fully scripted in advance."""

    def __init__(self, script, lives_script=None, action_meanings=("NOOP", "FIRE", "RIGHT")):
        self.script = list(script)
        self.lives_script = list(lives_script or [3] * (len(script) + 1))
        self.idx = 0
        self.reset_count = 0
        self.reset_kwargs = []
        self.obs_spec = ObsSpec.image(*FRAME_SHAPE)
        self.action_spec = ActionSpec.discrete(len(action_meanings))
        self._meanings = list(action_meanings)

    # -- the emulator surface the ALE wrappers take explicitly -------------
    @property
    def ale(self):
        return self

    def lives(self):
        # idx counts completed steps; lives_script has one entry per state
        # (index 0 is lives right after reset).
        return self.lives_script[min(self.idx, len(self.lives_script) - 1)]

    def action_meanings(self):
        return list(self._meanings)

    # -- Env ---------------------------------------------------------------
    def reset(self, *, seed=None):
        self.reset_count += 1
        self.reset_kwargs.append(seed)
        return np.zeros(FRAME_SHAPE, dtype=np.uint8), {}

    def step(self, action):
        item = self.script[self.idx]
        self.idx += 1
        obs = np.full(FRAME_SHAPE, item["obs"], dtype=np.uint8)
        return (
            obs,
            item.get("reward", 0.0),
            item.get("terminated", False),
            item.get("truncated", False),
            {},
        )

    def close(self):
        pass


# -- MaxAndSkip: no stale frame may leak across step() calls ----------------


def test_max_and_skip_maxes_the_last_two_frames():
    env = MaxAndSkip(ScriptedEnv([{"obs": v} for v in (10, 20, 30, 40)]), skip=4)
    env.reset()
    obs, _r, terminated, truncated, _i = env.step(0)
    assert not terminated and not truncated
    assert np.all(obs == 40)


def test_max_and_skip_does_not_pool_against_the_previous_calls_frame():
    """A step that terminates on its first sub-step has no "last two" to pool."""
    script = [{"obs": v} for v in (10, 20, 30, 40)]
    script.append({"obs": 99, "terminated": True})
    env = MaxAndSkip(ScriptedEnv(script), skip=4)
    env.reset()
    env.step(0)  # leaves the buffer holding 30 and 40

    obs, _r, terminated, _tr, _i = env.step(0)
    assert terminated
    assert np.all(obs == 99), f"got {np.unique(obs)}"


def test_max_and_skip_terminating_immediately_after_reset_leaks_no_zeros():
    env = MaxAndSkip(ScriptedEnv([{"obs": 77, "terminated": True}]), skip=4)
    env.reset()
    obs, _r, terminated, _tr, _i = env.step(0)
    assert terminated
    assert np.all(obs == 77)


# -- NoopReset --------------------------------------------------------------


def test_noop_reset_recovers_from_a_termination_during_the_noops():
    inner = ScriptedEnv([{"obs": 1, "terminated": True}, {"obs": 2}])
    obs, _info = NoopReset(inner, noop_max=1).reset(seed=0)
    assert obs.shape == FRAME_SHAPE
    assert inner.reset_count == 2


def test_noop_reset_recovery_does_not_replay_the_seed():
    """Re-seeding the recovery reset would rewind the RNG to exactly the state
    that just produced the failed start, so the recovery reproduces it."""
    inner = ScriptedEnv([{"obs": 1, "terminated": True}, {"obs": 2}])
    NoopReset(inner, noop_max=1).reset(seed=7)
    assert inner.reset_kwargs == [7, None]


# -- EpisodicLife -----------------------------------------------------------


def test_episodic_life_resets_the_game_only_on_a_real_done():
    """Resetting the ALE on a life loss throws away the rest of the game,
    which is the whole point of the wrapper."""
    inner = ScriptedEnv([{"obs": 1}, {"obs": 2}], lives_script=[3, 2, 2])
    env = EpisodicLife(inner, inner)
    env.reset()
    assert inner.reset_count == 1
    env.step(0)  # life loss
    env.reset()
    assert inner.reset_count == 1  # stepped through, did not reset


# -- FireReset --------------------------------------------------------------


def test_fire_reset_returns_a_started_game_after_a_failed_fire():
    """The observation handed back must come from after a successful FIRE, not
    from the bare recovery reset, which would leave the game unstarted."""
    inner = ScriptedEnv([{"obs": 1, "terminated": True}, {"obs": 2}])
    obs, _info = FireReset(inner, inner).reset()
    assert inner.reset_count == 2
    assert np.all(obs == 2)


# -- AtariConfig ------------------------------------------------------------


def test_the_evaluation_protocol_refuses_episodic_life():
    """Life-loss termination moves the episode boundaries, so the reported
    score would not be a game score at all."""
    with pytest.raises(ValueError, match="episodic_life"):
        AtariConfig.evaluation("ALE/Pong-v5", episodic_life=True)

