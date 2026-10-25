"""The division of one observation between the actor and the critic.

An asymmetric actor-critic is the general case: the critic conditions on
whatever the simulator knows and the actor on what the real system can sense.
These tests pin the spec half of that, which is where the division is declared
and where every refusal has to fire, because by the time a network exists the
only evidence of a leak is a policy that works in simulation and not on the
robot.
"""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.envs.wrappers import FrameStack, PrivilegedObservation
from ttrl.spaces import ObsField, ObsSpec

SENSED, PRIV = 4, 3
SPLIT = ObsSpec.split_vector(ObsField("sensed", SENSED),
                             ObsField("sim_state", PRIV, actor=False))


def test_the_actor_gets_its_own_columns_and_nothing_else():
    obs = np.arange(2 * (SENSED + PRIV), dtype=np.float32).reshape(2, -1)
    np.testing.assert_array_equal(SPLIT.actor_obs(obs), obs[:, :SENSED])
    assert SPLIT.actor_spans == ((0, SENSED),)


def test_non_contiguous_actor_fields_are_gathered_in_declaration_order():
    spec = ObsSpec.split_vector(
        ObsField("a", 2), ObsField("hidden", 1, actor=False), ObsField("b", 2))
    obs = np.arange(5, dtype=np.float32).reshape(1, 5)
    assert spec.actor_spans == ((0, 2), (3, 5))
    np.testing.assert_array_equal(spec.actor_obs(obs), [[0.0, 1.0, 3.0, 4.0]])


def test_cutting_an_already_narrow_observation_is_refused_by_name():
    """The deployment failure: sensor-only observations handed to an agent
    built for the simulator's wider ones. Reading the first columns of those
    as the actor's would be wrong and silent."""
    with pytest.raises(ValueError, match="sim_state"):
        SPLIT.actor_obs(np.zeros((2, SENSED), np.float32))


def test_a_single_net_algorithm_is_refused_and_told_what_to_use():
    with pytest.raises(ValueError, match="sim_state"):
        SPLIT.require_actor("dqn")
    assert ObsSpec.vector(4).require_actor("dqn") == ObsSpec.vector(4)


# ---------------------------------------------------------------------------
# what the spec refuses to describe
# ---------------------------------------------------------------------------

def test_fields_must_cover_the_observation_exactly():
    with pytest.raises(ValueError, match="cover 5 entries"):
        ObsSpec.vector(7, fields=(ObsField("a", 2), ObsField("b", 3)))


def test_a_spec_with_no_actor_field_is_refused():
    with pytest.raises(ValueError, match="no input at all"):
        ObsSpec.split_vector(ObsField("a", 2, actor=False))


def test_the_legal_action_mask_may_not_fall_in_a_critic_only_field():
    """The mask is the final entries of the observation and the actor is what
    masks, so privileged state has to sit ahead of it."""
    with pytest.raises(ValueError, match="legal-action mask"):
        ObsSpec.split_vector(ObsField("sensed", 4),
                             ObsField("sim_state", 2, actor=False),
                             action_mask=2)
    ok = ObsSpec.split_vector(ObsField("sim_state", 2, actor=False),
                              ObsField("sensed", 4),
                              ObsField("mask", 2),
                              action_mask=2)
    assert ok.actor.action_mask == 2
    assert ok.actor.size == 6


# ---------------------------------------------------------------------------
# the environment half
# ---------------------------------------------------------------------------

class _Toy:
    """Three sensed features and a two-entry simulator state to append."""

    obs_spec = ObsSpec.vector(3)
    action_spec = None

    def __init__(self):
        self.hidden = np.array([7.0, 8.0], np.float32)

    def reset(self, *, seed=None):
        return np.array([1.0, 2.0, 3.0], np.float32), {"from": "reset"}

    def step(self, action):
        return np.array([4.0, 5.0, 6.0], np.float32), 1.0, False, False, {}

    def close(self) -> None:
        pass


def _privileged_toy():
    inner = _Toy()
    return PrivilegedObservation(inner, lambda env: env.hidden,
                                 name="sim_state", size=2)


def test_the_wrapper_appends_the_field_and_declares_it_critic_only():
    env = _privileged_toy()
    assert env.obs_spec.size == 5
    assert env.obs_spec.privileged == ("sim_state",)
    assert env.obs_spec.actor.size == 3

    obs, info = env.reset()
    np.testing.assert_array_equal(obs, [1.0, 2.0, 3.0, 7.0, 8.0])
    assert info == {"from": "reset"}
    obs, *_ = env.step(None)
    np.testing.assert_array_equal(obs, [4.0, 5.0, 6.0, 7.0, 8.0])
    np.testing.assert_array_equal(env.obs_spec.actor_obs(obs.reshape(1, -1)),
                                  [[4.0, 5.0, 6.0]])


def test_the_wrapper_refuses_an_extractor_of_the_wrong_width():
    env = PrivilegedObservation(_Toy(), lambda e: np.zeros(3, np.float32),
                                name="sim_state", size=2)
    with pytest.raises(ValueError, match="returned 3 values"):
        env.reset()


def test_stacking_repeats_the_division_frame_by_frame():
    env = FrameStack(_privileged_toy(), 2)
    spec = env.obs_spec
    assert spec.size == 10
    assert spec.privileged == ("sim_state/0", "sim_state/1")
    assert spec.actor.size == 6
    assert spec.actor_spans == ((0, 3), (5, 8))

    obs, _ = env.reset()
    np.testing.assert_array_equal(
        spec.actor_obs(obs.reshape(1, -1)), [[1.0, 2.0, 3.0] * 2])
