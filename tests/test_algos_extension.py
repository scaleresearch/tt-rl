"""The extension seam: a name resolves to a builder, and what it builds is checked.

Registering an algorithm is the one way to add one, so this file drives that
path with an agent written here, outside `ttrl.algos`, and never with a
shipped algorithm. The refusals an algorithm raises about its own config and
its own action space are `tests/test_algos_contract.py`; this is about
whatever arrives through the registry.

`check_agent` belongs here for the same reason. `Agent` is a `Protocol`, so
nothing fails at instantiation and the trainer is the next thing to find out:
the check is what turns the contract into a refusal at construction, and the
agent it is asked about is by definition one this repo did not write.

No hardware: nothing here builds a network.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttrl
from ttrl.algos import ALGOS, check_agent, resolve_and_build
from ttrl.algos.base import Transition
from ttrl.algos.dqn import DQN
from ttrl.algos.ppo import PPO
from ttrl.algos.ppo.boundaries import TransitionBoundaries
from ttrl.algos.sac import SAC
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec

VECTOR = ObsGroup.of(ObsSpec.vector(4))
DISCRETE = ActionSpec.discrete(3)


def test_registry_resolves_the_three_shipped_algorithms():
    assert ALGOS.names() == ["dqn", "ppo", "sac"]
    assert ALGOS.resolve("dqn") is DQN
    assert ALGOS.resolve("ppo") is PPO
    assert ALGOS.resolve("sac") is SAC


def test_registry_refuses_an_unknown_name_and_lists_what_it_has():
    with pytest.raises(KeyError) as caught:
        ALGOS.resolve("rainbow")
    message = str(caught.value)
    assert "rainbow" in message
    assert "'dqn', 'ppo', 'sac'" in message
    assert "ALGOS.register" in message


def test_registry_refuses_a_duplicate_registration():
    with pytest.raises(ValueError, match="already registered"):
        ALGOS.register("dqn", DQN)


def test_a_third_party_algorithm_needs_no_core_edit():
    class Mine:
        def __init__(self, obs_spec, action_spec, device, **kwargs):
            self.obs_spec, self.action_spec = obs_spec, action_spec

        def act(self, obs, *, deterministic=False, episode_starts=None):
            return np.zeros(len(obs), np.int64)

        def observe(self, transition): pass

        def update(self): return {}

        def should_update(self, frame): return False

        def named_parameters(self): return iter(())

        def parameters(self): return iter(())

        def state_dict(self, *, weights_only=False): return {}

        def load_state_dict(self, state, *, weights_only=False): pass

        def close(self): pass

    ALGOS.register("mine-under-test", Mine)
    try:
        agent = resolve_and_build("mine-under-test", VECTOR, DISCRETE, device=None)
        assert isinstance(agent, Mine)
        check_agent(agent)
    finally:
        del ALGOS._items["mine-under-test"]


def test_check_agent_names_the_missing_method():
    class Half:
        obs_spec = VECTOR
        action_spec = DISCRETE

        def act(self, obs, *, deterministic=False): return None

        def observe(self, transition): pass

        def update(self): return {}

    with pytest.raises(TypeError, match="should_update"):
        check_agent(Half())

    class NoAveraging(Half):
        def should_update(self, frame): return False

        def state_dict(self, *, weights_only=False): return {}

        def load_state_dict(self, state, *, weights_only=False): pass

        def close(self): pass

    with pytest.raises(TypeError) as caught:
        check_agent(NoAveraging())
    assert "named_parameters" in str(caught.value)
    assert "more than one chip" in str(caught.value)

    class NoSpecs:
        pass

    with pytest.raises(TypeError, match="obs_spec"):
        check_agent(NoSpecs())


def test_check_agent_names_an_act_that_cannot_be_told_where_episodes_start():
    """`episode_starts` is part of the acting contract, so an agent written
    without it is refused at construction rather than mid-evaluation.

    A policy with memory has no other channel for an episode boundary while it
    is being evaluated: evaluation hands it observations and takes actions,
    and `test_evaluate_never_sees_an_agents_update_surface` keeps `observe`
    out of that loop deliberately. So the flags ride on `act`, every
    implementer accepts them, and one with no memory ignores them.
    """
    class Deaf:
        obs_spec = VECTOR
        action_spec = DISCRETE

        def act(self, obs, *, deterministic=False):
            return np.zeros(len(obs), np.int64)

        def observe(self, transition): pass

        def update(self): return {}

        def should_update(self, frame): return False

        def named_parameters(self): return iter(())

        def parameters(self): return iter(())

        def state_dict(self, *, weights_only=False): return {}

        def load_state_dict(self, state, *, weights_only=False): pass

        def close(self): pass

    with pytest.raises(TypeError, match="episode_starts"):
        check_agent(Deaf())

    class Kwargs(Deaf):
        def act(self, obs, **kwargs):
            return np.zeros(len(obs), np.int64)

    # A **kwargs act accepts them, so it satisfies the contract: the check is
    # about whether the flags can be delivered, not about how they are spelled.
    check_agent(Kwargs())


def test_check_boundaries_holds_an_episode_boundary_hook_to_its_mechanics():
    """The training half of the same seam: where a recurrent policy resets.

    `episode_starts` above is the acting channel, and it was the only public
    one. During training PPO derives the flags itself, so the memory-defeated
    arm of a memory ablation -- every step begins an episode, so no hidden
    state survives one -- was reachable only through private attributes.
    `EpisodeBoundaries` is that answer's declared source, and this is the
    contract test a user runs against their own, so the mechanics it refuses
    are pinned here.

    It judges the mechanics and not the policy: ones is an ablation and
    `terminated` alone is a silent bug, and both are legal hooks.
    """

    class DefeatedMemory:
        def boundaries(self, transition):
            return np.ones(transition.n_envs, np.float32)

    ttrl.check_boundaries(DefeatedMemory())
    assert isinstance(DefeatedMemory(), ttrl.EpisodeBoundaries)

    class WrongWidth:
        def boundaries(self, transition):
            return np.ones(1, np.float32)

    with pytest.raises(ValueError, match="fixed when the acting step is"):
        ttrl.check_boundaries(WrongWidth(), n_envs=3)

    class Unbounded:
        def boundaries(self, transition):
            return np.full(transition.n_envs, -1.0, np.float32)

    with pytest.raises(ValueError, match=r"outside \[0.0, 1.0\]"):
        ttrl.check_boundaries(Unbounded())

    class NotFinite:
        def boundaries(self, transition):
            return np.full(transition.n_envs, np.nan, np.float32)

    with pytest.raises(ValueError, match="non-finite"):
        ttrl.check_boundaries(NotFinite())

    class Writes:
        def boundaries(self, transition):
            transition.truncated[:] = True
            return np.asarray(transition.done, np.float32)

    with pytest.raises(AssertionError, match="wrote through the transition"):
        ttrl.check_boundaries(Writes())

    with pytest.raises(TypeError, match="boundaries"):
        ttrl.check_boundaries(object())

    # The default, and the one answer that is not negotiable about it: a
    # truncation ends an episode for the memory even though it bootstraps.
    obs = VECTOR.observation({"obs": np.zeros((2, 4), np.float32)})
    truncating = Transition(
        obs=obs, action=np.zeros(2, np.int64), reward=np.zeros(2, np.float32),
        terminated=np.zeros(2, bool), truncated=np.array([False, True]),
        next_obs=obs)
    assert TransitionBoundaries().boundaries(truncating).tolist() == [0.0, 1.0]
