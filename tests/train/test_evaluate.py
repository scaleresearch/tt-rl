"""The fixed eval protocol."""

from __future__ import annotations

import dataclasses

import pytest

from tests._fakes.train_layer import ScriptedAgent, ToyEnv
from ttrl.envs.registry import ENVS, register
from ttrl.envs.wrappers import RecordEpisodeStatistics, ScaleReward
from ttrl.train.config import EnvConfig
from ttrl.train.evaluate import evaluate

ENV = EnvConfig(name="toy", n_envs=4, kwargs={"episode_len": 3})


def test_exactly_the_requested_episodes_are_scored():
    result = evaluate(ScriptedAgent(), ENV, n_episodes=7)
    assert result["n_episodes"] == 7
    assert len(result["scores"]) == 7


def test_the_summary_is_the_environments_own_returns():
    result = evaluate(ScriptedAgent(), ENV, n_episodes=4)
    assert result["scores"] == [3.0, 3.0, 3.0, 3.0]
    assert (result["mean"], result["median"], result["min"], result["max"]) == (
        3.0, 3.0, 3.0, 3.0,
    )


@pytest.mark.parametrize("env_name", ["toy", "bare_toy", "lane_scaled_toy"])
def test_a_scaled_training_reward_never_reaches_the_score(env_name):
    """The easiest thing to get wrong about reward normalisation."""
    register("bare_toy", lambda **kwargs: ToyEnv(**kwargs))
    register("lane_scaled_toy",
             lambda **kwargs: ScaleReward(
                 RecordEpisodeStatistics(ToyEnv(**kwargs)), 10.0))
    try:
        scaled = dataclasses.replace(
            ENV, name=env_name, normalize_reward=True,
            normalize_reward_gamma=0.9, normalize_obs=True)
        result = evaluate(ScriptedAgent(), scaled, n_episodes=4)
    finally:
        ENVS.unregister("bare_toy")
        ENVS.unregister("lane_scaled_toy")
    assert result["scores"] == [3.0, 3.0, 3.0, 3.0], (
        "the eval score is the environment's own return, not the scaled one"
    )


def test_the_agent_is_asked_for_deterministic_actions():
    agent = ScriptedAgent()
    seen = []
    real = agent.act

    def act(obs, *, deterministic=False, episode_starts=None):
        seen.append(deterministic)
        return real(obs, deterministic=deterministic)

    agent.act = act
    evaluate(agent, ENV, n_episodes=4)
    assert seen and all(seen)


def test_an_episode_that_never_ends_is_a_refusal_not_a_stall():
    never = dataclasses.replace(ENV, n_envs=1, kwargs={"episode_len": 50})
    with pytest.raises(RuntimeError, match="without the episode ending"):
        evaluate(ScriptedAgent(), never, n_episodes=1, max_episode_steps=5)



class SeedScoredEnv(ToyEnv):
    """A toy whose episode return IS the reset seed it started from.

    Every score therefore names the start it came from, which is what lets a
    test say that an eval's episodes began at different places.
    """

    def step(self, action):
        obs, _reward, terminated, truncated, info = super().step(action)
        return obs, float(self._seed) if terminated else 0.0, terminated, truncated, info


@pytest.mark.parametrize("n_envs", [1, 4])
def test_each_episode_starts_from_its_own_seed_and_the_eval_repeats(n_envs):
    """Different starts within an eval, same starts between two evals.

    An eval used to seed one reset and take its remaining episodes off the
    chained RNG, so on an environment with a small reset perturbation the
    episodes after the first were near-repeats and `eval/mean` was one draw
    reported as a mean of ten. Both halves matter: without distinct starts
    the extra episodes buy no variance reduction, and without reproducible
    starts two checkpoints are no longer scored on the same ground.
    """
    register("seed_scored_toy",
             lambda **kwargs: RecordEpisodeStatistics(SeedScoredEnv(**kwargs)))
    try:
        env = dataclasses.replace(ENV, name="seed_scored_toy", n_envs=n_envs)
        scores = evaluate(ScriptedAgent(), env, n_episodes=6, seed=100)["scores"]
        again = evaluate(ScriptedAgent(), env, n_episodes=6, seed=100)["scores"]
    finally:
        ENVS.unregister("seed_scored_toy")
    assert sorted(scores) == [100.0, 101.0, 102.0, 103.0, 104.0, 105.0], (
        "episode e must start from seed + e, with no start used twice"
    )
    assert again == scores, "one seed must score one policy the same way twice"
