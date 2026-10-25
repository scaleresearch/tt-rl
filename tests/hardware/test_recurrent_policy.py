"""A recurrent policy end to end: the two paths agree, and PPO drives both."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.algos.base import Transition
from ttrl.algos.ppo import PPO, PPOConfig
from ttrl.algos.ppo.config import RecurrentConfig
from ttrl.engine import acquire
from ttrl.engine.autograd import Tensor, no_grad
from ttrl.engine.carry import ActingCarry
from ttrl.nets.acting import ActingNet
from ttrl.nets.recurrent import ACTOR, CRITIC, build_recurrent_actor_critic
from ttrl.spaces import ActionSpec, ObsGroup, ObsLeaf, ObsSpec

pytestmark = pytest.mark.tt

SEQ_LEN = 4
N_ENVS = 3
HIDDEN = 7
OBS_DIM = 5
N_ACTIONS = 6
#: One episode start strictly inside the sequence, so the reset is exercised
#: in both paths rather than only at the sequence edge.
RESET_AT = 2

SYMMETRIC = ObsGroup.of(ObsSpec.vector(OBS_DIM))
ASYMMETRIC = ObsGroup((ObsLeaf("sensed", ObsSpec.vector(OBS_DIM)),
                       ObsLeaf("sim_state", ObsSpec.vector(OBS_DIM + 1),
                               actor=False)))
DISCRETE = ActionSpec.discrete(N_ACTIONS)


@pytest.fixture(scope="module")
def dev():
    with acquire(0) as (handle, _profile):
        yield handle


def test_the_acting_carry_and_the_training_unroll_agree(dev):
    """One lane stepped `SEQ_LEN` times, then the same steps unrolled."""
    rng = np.random.default_rng(17)
    net = build_recurrent_actor_critic(
        SYMMETRIC, DISCRETE, dev, batch=SEQ_LEN, hidden_size=HIDDEN)
    # One view per width, so acting at one lane and unrolling SEQ_LEN rows
    # share the same parameters.
    acting = ActingNet(net, SYMMETRIC, dev, width=SEQ_LEN, lanes=1)
    rows = rng.normal(size=(SEQ_LEN, OBS_DIM)).astype(np.float32)
    starts = np.zeros(SEQ_LEN, dtype=np.float32)
    starts[RESET_AT] = 1.0

    carry = ActingCarry(HIDDEN, dev, lanes=1)
    # Something non-zero to carry, so the reset at RESET_AT has an effect and
    # the agreement is not between two zero states.
    carry.bind(Tensor.from_numpy(
        rng.normal(size=(1, HIDDEN)).astype(np.float32), dev))
    carry.commit()
    h0 = carry.numpy().copy()

    acted = []
    for t in range(SEQ_LEN):
        carry.load_start(starts[t:t + 1])
        with no_grad():
            logits, value, h_next = acting.step(
                {"obs": rows[t:t + 1]}, {ACTOR: carry.prev},
                {ACTOR: carry.start})
        acted.append((logits.numpy()[0].copy(), float(value.numpy().reshape(-1)[0])))
        carry.bind(h_next[ACTOR])
        carry.commit()

    with no_grad():
        logits, value = acting(
            {"obs": rows},
            {ACTOR: Tensor.from_numpy(h0, dev)},
            Tensor.from_numpy(starts.reshape(SEQ_LEN, 1, 1), dev),
            seq_len=SEQ_LEN,
        )
    unrolled_logits = logits.numpy()
    unrolled_value = value.numpy().reshape(-1)

    spread = float(np.abs(np.diff([row for row, _v in acted], axis=0)).max())
    assert spread > 1e-4, (
        "the acting outputs are identical across timesteps, so an agreement "
        "between the two paths would be vacuous")
    for t, (logit_row, value_row) in enumerate(acted):
        assert np.allclose(logit_row, unrolled_logits[t], atol=2e-3, rtol=2e-3), (
            f"timestep {t}: acting logits {logit_row} against unrolled "
            f"{unrolled_logits[t]}. The two paths compute different "
            f"recurrences."
        )
        assert unrolled_value[t] == pytest.approx(value_row, abs=2e-3)


def test_an_asymmetric_observation_gives_the_critic_its_own_memory(dev):
    """A single GRU over both trunks would carry the critic's privileged
    features into the state the actor reads, which trains fine and leaves the
    policy unusable on the deployed system. So the memories are separate, and
    no parameter of the actor's path lives under the critic's."""
    net = build_recurrent_actor_critic(
        ASYMMETRIC, DISCRETE, dev, batch=SEQ_LEN, hidden_size=HIDDEN)
    assert not net.shared
    assert net.memories[ACTOR] is not net.memories[CRITIC]
    assert set(net.hidden_sizes) == {ACTOR, CRITIC}

    names = [name for name, _p in net.named_parameters()]
    actor_side = [n for n in names if n.startswith(("encoders.actor",
                                                    "memories.actor"))]
    critic_memory = [n for n in names if n.startswith("memories.critic")]
    assert actor_side and critic_memory, (
        f"the parameter names do not separate the two memories: {names}")
    assert not set(actor_side) & set(critic_memory)
    # And the privileged leaf has no tower on the actor's side at all, which
    # is the property the asymmetric spec exists for.
    assert not [n for n in names if "actor" in n and "sim_state" in n], (
        f"a sim_state tower reached the actor: {names}")


def _cfg(**kwargs) -> PPOConfig:
    return PPOConfig(
        n_steps=SEQ_LEN * 3, minibatch_size=SEQ_LEN * 2, epochs=1,
        recurrent=RecurrentConfig(hidden_size=HIDDEN, seq_len=SEQ_LEN),
        **kwargs)


def _feed(agent, rng, steps: int, *, boundary_at=()) -> None:
    group = agent.obs_spec
    for t in range(steps):
        obs = {name: rng.normal(size=(N_ENVS, *group[name].shape)
                                ).astype(np.float32)
               for name in group.names}
        agent.act(obs)
        terminated = np.zeros(N_ENVS, bool)
        if t in boundary_at:
            terminated[0] = True
        agent.observe(Transition(
            obs=group.observation(obs),
            action=rng.integers(0, N_ACTIONS, N_ENVS),
            reward=rng.normal(size=N_ENVS).astype(np.float32),
            terminated=terminated, truncated=np.zeros(N_ENVS, bool),
            next_obs=group.observation(obs),
        ))


def test_ppo_collects_a_recurrent_rollout_and_updates(dev):
    """One config field away, which is the surface this exists to provide."""
    cfg = _cfg()
    agent = PPO(SYMMETRIC, DISCRETE, dev, cfg=cfg, n_envs=N_ENVS, seed=0)
    try:
        assert agent.buffer.recurrent and agent.buffer.seq_len == SEQ_LEN
        _feed(agent, np.random.default_rng(19), cfg.n_steps, boundary_at=(2,))
        assert agent.should_update(cfg.n_steps * N_ENVS)
        stored = agent.buffer.hidden[ACTOR]
        assert float(np.abs(stored[1:]).max()) > 0.0, (
            "every sequence after the first starts from a zero state, so the "
            "acting carry is not reaching the rollout storage")
        metrics = agent.update()
        for key in ("loss", "policy_loss", "value_loss", "entropy"):
            assert np.isfinite(metrics[key]), f"{key} is {metrics[key]}"
        assert metrics["minibatches"] == float(
            (cfg.n_steps // SEQ_LEN) * N_ENVS // (cfg.minibatch_size // SEQ_LEN))
    finally:
        agent.close()


def test_a_deterministic_act_without_episode_starts_is_refused(dev):
    """Evaluation feeds no transitions back, so a recurrent policy cannot
    derive its own resets there. Refused rather than scored with a hidden
    state carried across episode boundaries."""
    agent = PPO(SYMMETRIC, DISCRETE, dev, cfg=_cfg(), n_envs=N_ENVS, seed=0)
    try:
        obs = {"obs": np.zeros((N_ENVS, OBS_DIM), np.float32)}
        with pytest.raises(ValueError, match="episode_starts"):
            agent.act(obs, deterministic=True)
        agent.act(obs, deterministic=True,
                  episode_starts=np.ones(N_ENVS, np.float32))
    finally:
        agent.close()


def test_an_evaluation_act_does_not_disturb_the_training_carry(dev):
    """A deterministic rollout run between two training steps must not move
    the state the training rollout is in the middle of. Sharing one carry
    leaves every shape intact and resumes the next training step from an
    evaluation episode's memory."""
    cfg = _cfg()
    rng = np.random.default_rng(23)
    obs = [{"obs": rng.normal(size=(N_ENVS, OBS_DIM)).astype(np.float32)}
           for _ in range(2)]

    def run(interleave_eval: bool) -> np.ndarray:
        agent = PPO(SYMMETRIC, DISCRETE, dev, cfg=cfg, n_envs=N_ENVS, seed=0)
        try:
            _feed(agent, np.random.default_rng(29), SEQ_LEN)
            if interleave_eval:
                for step in obs:
                    agent.act(step, deterministic=True,
                              episode_starts=np.ones(N_ENVS, np.float32))
            carries = agent._recurrent.carries(N_ENVS, deterministic=False)
            return carries[ACTOR].numpy().copy()
        finally:
            agent.close()

    clean = run(False)
    interleaved = run(True)
    assert float(np.abs(clean).max()) > 0.0, (
        "the training carry is zero after a rollout, so this comparison is "
        "vacuous")
    assert np.allclose(clean, interleaved, atol=1e-6), (
        "an evaluation rollout moved the training carry, so the next training "
        "step would resume from an evaluation episode's hidden state")
