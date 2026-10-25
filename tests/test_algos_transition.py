"""The `Transition` record and the autoreset phase. No hardware.

Both are pure host bookkeeping: the lane-major shape contract, and the mask
that drops the padding step a NEXT_STEP vector environment emits after a lane
ends. Neither needs a chip, so neither is allowed to want one.
"""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.algos.base import AutoresetTracker, Transition
from ttrl.spaces import ObsGroup, ObsSpec

GROUP = ObsGroup.of(ObsSpec.vector(4))

def _transition(n: int = 2, *, terminated=None, truncated=None) -> Transition:
    obs = GROUP.observation(np.zeros((n, 4), np.float32))
    return Transition(
        obs=obs, action=np.zeros(n, np.int64), reward=np.zeros(n, np.float32),
        terminated=np.zeros(n, bool) if terminated is None else np.asarray(terminated),
        truncated=np.zeros(n, bool) if truncated is None else np.asarray(truncated),
        next_obs=GROUP.observation(obs["obs"] + 1.0),
    )


# -- the transition ---------------------------------------------------------

def test_transition_requires_every_field_to_be_lane_major():
    obs = GROUP.observation(np.zeros((3, 4), np.float32))
    with pytest.raises(ValueError, match="lane-major"):
        Transition(obs=obs, action=np.zeros(2), reward=np.zeros(3),
                   terminated=np.zeros(3, bool), truncated=np.zeros(3, bool),
                   next_obs=obs)
    wider = ObsGroup.of(ObsSpec.vector(5)).observation(
        np.zeros((3, 5), np.float32))
    with pytest.raises(ValueError, match="next_obs shape"):
        Transition(obs=obs, action=np.zeros(3), reward=np.zeros(3),
                   terminated=np.zeros(3, bool), truncated=np.zeros(3, bool),
                   next_obs=wider)


def test_transition_from_step_reads_a_vec_env_return():
    obs = np.zeros((2, 4), np.float32)
    step = (obs + 1.0, [0.5, -0.5], [True, False], [False, False],
            {"final_obs": [obs[0] + 1.0, None]})
    transition = Transition.from_step(GROUP, obs, np.zeros(2), step)
    assert transition.n_envs == 2
    assert transition.done.tolist() == [True, False]
    # The final observation is next_obs for the lane that ended: that is the
    # NEXT_STEP contract, and it is why Transition carries no separate field.
    assert np.array_equal(transition.next_obs["obs"][0],
                          step[4]["final_obs"][0])


# -- autoreset phase --------------------------------------------------------

def test_autoreset_tracker_drops_the_step_after_a_lane_ends():
    tracker = AutoresetTracker(3)
    assert tracker.valid(_transition(3)).tolist() == [True, True, True]
    assert tracker.valid(
        _transition(3, terminated=[False, True, False])).tolist() == [True] * 3
    # Lane 1's next step is the autoreset: its action was ignored.
    assert tracker.valid(_transition(3)).tolist() == [True, False, True]
    assert tracker.valid(_transition(3)).tolist() == [True, True, True]


def test_autoreset_tracker_refuses_a_same_step_autoreset():
    tracker = AutoresetTracker(2)
    tracker.valid(_transition(2, terminated=[True, False]))
    with pytest.raises(ValueError, match="NEXT_STEP"):
        tracker.valid(_transition(2, truncated=[True, False]))


def test_a_restored_tracker_starts_its_phase_FRESH():
    """A checkpoint is loaded into a run whose envs are about to be reset.

    So no lane owes an autoreset step, whatever the writing run owed.
    Restoring the stored flags instead discarded the first REAL transition
    after every resume, for each lane that had ended its episode on the
    checkpointed step: with an episode length that divides the checkpoint
    interval that is every lane, every time. Measured on a 2-lane toy DQN run
    checkpointed on an episode boundary: the first post-resume round added 0
    transitions to the replay buffer instead of 2.
    """
    tracker = AutoresetTracker(2)
    tracker.valid(_transition(2, terminated=[True, False]))
    assert tracker.state_dict()["pending"].tolist() == [True, False], (
        "the phase is still written: it describes the run that wrote it")
    restored = AutoresetTracker(2)
    restored.load_state_dict(tracker.state_dict())
    assert restored.valid(_transition(2)).tolist() == [True, True]
    # The shape is still checked: a file whose lane count is not this run's
    # is the wrong file.
    with pytest.raises(ValueError, match="shape"):
        AutoresetTracker(3).load_state_dict(tracker.state_dict())
