"""Checkpoint round-trip and resume refusals, for all three algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from tests._fakes.toy_envs import REACH_GROUP, SIGN_GROUP, ReachEnv, SignEnv, run_agent
from ttrl.algos import ALGOS
from ttrl.algos.base import Transition
from ttrl.algos.dqn import DQN, DQNConfig
from ttrl.algos.ppo import PPO, PPOConfig
from ttrl.algos.ppo.config import RecurrentConfig
from ttrl.algos.ppo.partial import STEP_FIELDS
from ttrl.algos.sac import SAC, SACConfig
from ttrl.engine import acquire
from ttrl.envs.vector import SyncVectorEnv

pytestmark = pytest.mark.tt

LANES = 2


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


def _dqn(device, **over):
    cfg = DQNConfig(batch_size=16, learning_starts=32, train_freq=2, n_step=1,
                    buffer_capacity=1000, **over)
    return DQN(SIGN_GROUP, SignEnv.action_spec, device, cfg=cfg,
               n_envs=LANES, seed=0), SignEnv


def _sac(device, **over):
    cfg = SACConfig(batch_size=16, learning_starts=32, train_freq=2,
                    buffer_capacity=1000, **over)
    return SAC(REACH_GROUP, ReachEnv.action_spec, device, cfg=cfg,
               n_envs=LANES, seed=0), ReachEnv


def _ppo(device, **over):
    cfg = PPOConfig(n_steps=8, minibatch_size=8, epochs=2, **over)
    return PPO(REACH_GROUP, ReachEnv.action_spec, device, cfg=cfg,
               n_envs=LANES, seed=0), ReachEnv


def _ppo_recurrent(device, **over):
    """PPO with memory. Its rollout carries the hidden state entering each
    sequence plus the live acting carry, and those are the members with no
    second path to notice a bad round trip.

    Shaped so the 200 frames below leave the buffer part way through a
    SEQUENCE as well as part way through a rollout: 100 steps is 4 into a
    6-step rollout, which is one step into the second of its 3-step
    sequences, so the stored sequence-start states have to round up.
    """

    cfg = PPOConfig(n_steps=6, minibatch_size=6, epochs=2,
                    recurrent=RecurrentConfig(hidden_size=16, seq_len=3),
                    **over)
    return PPO(REACH_GROUP, ReachEnv.action_spec, device, cfg=cfg,
               n_envs=LANES, seed=0), ReachEnv


BUILDERS = {"dqn": _dqn, "sac": _sac, "ppo": _ppo,
            "ppo_recurrent": _ppo_recurrent}


#: The set is derived: a newly registered algorithm has to gain a builder here
#: or collection fails. `ppo_recurrent` is an extra parametrisation rather than
#: a registry name, so this is coverage of the registry, not equality with it.
_unbuilt = sorted(set(ALGOS.names()) - set(BUILDERS))
assert not _unbuilt, (
    f"registered algorithm(s) {_unbuilt} have no builder in this file: add a "
    f"builder for each to BUILDERS above so they reach these gates"
)


def _all_params(agent):
    if hasattr(agent, "online"):
        return [("online." + n, p) for n, p in agent.online.named_parameters()] + \
               [("target." + n, p) for n, p in agent.target.named_parameters()]
    if hasattr(agent, "nets"):
        return [(f"{label}.{n}", p)
                for label, module in (("policy", agent.nets.policy),
                                      ("critics", agent.nets.critics),
                                      ("targets", agent.nets.targets))
                for n, p in module.named_parameters()] + \
               [("log_alpha", agent.log_alpha)]
    return list(agent.net.named_parameters())


@pytest.mark.parametrize("algo", list(BUILDERS))
def test_state_dict_round_trips_into_a_fresh_agent(algo, device):
    trained, env_class = BUILDERS[algo](device)
    fresh, _ = BUILDERS[algo](device)
    weights, _ = BUILDERS[algo](device)
    try:
        with SyncVectorEnv([lambda i=i: env_class(seed=i) for i in range(LANES)]) as env:
            run_agent(trained, env, 200)
        state = trained.state_dict()

        differing = [name for (name, param), (_, other)
                     in zip(_all_params(trained), _all_params(fresh))
                     if np.abs(param.numpy() - other.numpy()).max() > 0.0]
        assert differing, "the two agents start identical, so the load proves nothing"

        addresses = {name: id(param.data) for name, param in _all_params(fresh)}
        fresh.load_state_dict(state)

        worst = 0.0
        for (name, param), (_, other) in zip(_all_params(trained), _all_params(fresh)):
            worst = max(worst, float(np.abs(param.numpy() - other.numpy()).max()))
            # Address stability: a restore writes into the existing buffer, so
            # a trace captured before it still replays against live weights.
            assert id(other.data) == addresses[name], f"load rebound {name}"
        assert worst == 0.0, f"worst parameter difference after load {worst}"
        assert fresh.frame_count == trained.frame_count
        assert fresh.update_count == trained.update_count
        assert (fresh._rng.bit_generator.state
                == trained._rng.bit_generator.state)

        # A weights-only load restores parameters and leaves the run half alone.
        snapshot = trained.state_dict(weights_only=True)
        assert snapshot["weights_only"] is True
        assert not [key for key in trained._RUN_KEYS if key in snapshot]
        weights.load_state_dict(snapshot, weights_only=True)
        for (_n, a), (_m, b) in zip(_all_params(trained), _all_params(weights)):
            assert np.abs(a.numpy() - b.numpy()).max() == 0.0
        assert weights.frame_count == 0
        assert weights.update_count == 0
    finally:
        trained.close()
        fresh.close()
        weights.close()


def test_resume_refuses_a_different_configuration(device):
    algo, over = "dqn", {"gamma": 0.5}
    saved, _ = BUILDERS[algo](device)
    other, _ = BUILDERS[algo](device, **over)
    field, value = next(iter(over.items()))
    try:
        with pytest.raises(ValueError, match="differs from") as caught:
            other.load_state_dict(saved.state_dict())
        # Naming the field is the point: a reader resuming with one changed
        # value must not have to diff two config reprs by eye.
        message = str(caught.value)
        assert f"{field}=" in message
        assert f"{field}={value!r}" in message
        assert repr(getattr(saved.cfg, field)) in message
        assert "field differs" in message
        # Not the whole config printed twice, which is what this replaced: the
        # repr of a dataclass leads with its class name.
        assert type(saved.cfg).__name__ not in message
    finally:
        saved.close()
        other.close()


def test_resume_refuses_a_checkpoint_missing_state(device):
    agent, _ = BUILDERS["dqn"](device)
    try:
        state = agent.state_dict()
        for key in list(state):
            partial = {k: v for k, v in state.items() if k != key}
            with pytest.raises(ValueError, match="missing"):
                agent.load_state_dict(partial)
    finally:
        agent.close()


def test_a_weights_only_state_is_refused_where_moments_are_expected(device):
    agent, _ = BUILDERS["dqn"](device)
    try:
        with pytest.raises(ValueError) as caught:
            agent.load_state_dict(agent.state_dict(weights_only=True))
        message = str(caught.value)
        assert "parameters only" in message
        assert "zero" in message
        # The other direction is allowed and useful: an evaluation loads the
        # parameters out of a resumable file without inheriting its optimizer.
        agent.load_state_dict(agent.state_dict(), weights_only=True)
    finally:
        agent.close()


def test_ppo_resumes_the_rollout_it_was_saved_part_way_through(device):
    """A checkpoint off a rollout boundary must carry the steps in the buffer.

    They used to be dropped, which cost the resumed run one gradient step and
    said nothing: 1024 frames at a 64-frame rollout took 16 updates
    uninterrupted and 16 across a split at 512, and 15 across a split at 544,
    where 8 steps were sitting in the buffer. The invariant gated here is the
    consequence rather than the cursor: fed the same transitions, a resumed
    agent reaches its next gradient step on the same step as the one it
    resumed.
    """

    trained, env_class = _ppo(device)
    fresh, _ = _ppo(device)
    try:
        with SyncVectorEnv([lambda i=i: env_class(seed=i)
                            for i in range(LANES)]) as env:
            run_agent(trained, env, 200)
            part = trained.buffer._t
            assert 0 < part < trained.cfg.n_steps, (
                f"this save must land part way through a rollout or the test "
                f"proves nothing; the buffer holds {part} steps"
            )
            fresh.load_state_dict(trained.state_dict())
            assert fresh.buffer._t == part
            for name in STEP_FIELDS:
                stored = getattr(fresh.buffer, name)[:part]
                if name == "valid":
                    # The seam. The step after the last stored one is the
                    # first step of a new episode, because the resume resets
                    # the env, so the last stored step has no successor in
                    # this rollout and becomes a padding slot: no gradient,
                    # and `compute_gae` zeroes the chain at it rather than
                    # bootstrapping the whole rollout off a new episode's
                    # value. `ttrl.algos.ppo.partial` argues the cost.
                    assert np.array_equal(stored[:part - 1],
                                          trained.buffer.valid[:part - 1])
                    assert not stored[part - 1].any(), "the seam is not closed"
                    continue
                assert np.array_equal(
                    stored, getattr(trained.buffer, name)[:part]), name
            assert (fresh.buffer._carried_boundary == 1.0).all(), (
                "every lane's next step must be recorded as beginning an "
                "episode, which is what resets the recurrent memory")
            for name, array in trained.buffer.obs.items():
                assert np.array_equal(fresh.buffer.obs[name][:part],
                                      array[:part]), name

            # Both agents, the same transitions, from here on.
            obs, _info = env.reset(seed=1)
            fired: dict[str, int] = {}
            for step in range(trained.cfg.n_steps + 1):
                action = trained.act(obs)
                fresh.act(obs)
                result = env.step(action)
                transition = Transition.from_step(
                    env.obs_spec, obs, action, result)
                for agent, label in ((trained, "trained"), (fresh, "fresh")):
                    agent.observe(transition)
                    if agent.should_update(step):
                        fired[label] = step
                obs = result[0]
                # A full buffer refuses the next observe, so the first update
                # either agent asks for ends the comparison.
                if fired:
                    break
            assert fired.get("fresh") == fired.get("trained") is not None, (
                f"the resumed agent's next update must fall on the step the "
                f"checkpointed one's does, got {fired}"
            )
    finally:
        trained.close()
        fresh.close()
