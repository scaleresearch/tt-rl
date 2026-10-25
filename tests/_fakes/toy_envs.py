"""Two trivial learnable environments, and the loop that drives an Agent.

Not a test module. Both envs are solvable by one linear layer, so a few
hundred steps is enough to tell learning from non-learning, and neither
imports gymnasium, ttnn or any env suite.
"""

from __future__ import annotations

import numpy as np

from ttrl.algos.base import Transition
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec

__all__ = ["REACH_GROUP", "SIGN_GROUP", "SignEnv", "ReachEnv", "run_agent"]

EPISODE_LENGTH = 10


class SignEnv:
    """Discrete: pick the action matching the sign of the first feature.

    Reward 1 for the right action and 0 for the wrong one, so the mean
    episode return of a random policy is EPISODE_LENGTH / 2 and of the optimal
    policy is EPISODE_LENGTH.
    """

    obs_spec = ObsSpec.vector(4)
    action_spec = ActionSpec.discrete(2)

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._obs = self._draw()

    def _draw(self) -> np.ndarray:
        return self._rng.normal(size=(4,)).astype(np.float32)

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._obs = self._draw()
        return self._obs.copy(), {}

    def step(self, action):
        want = int(self._obs[0] > 0.0)
        reward = 1.0 if int(action) == want else 0.0
        self._t += 1
        self._obs = self._draw()
        truncated = self._t >= EPISODE_LENGTH
        return self._obs.copy(), reward, False, truncated, {}

    def close(self) -> None:
        pass


class ReachEnv:
    """Continuous: emit the action nearest tanh of the first two features.

    Reward is the negated squared distance, so a random policy scores about
    -1 per step per dimension and the optimum is 0.
    """

    obs_spec = ObsSpec.vector(3)
    action_spec = ActionSpec.continuous([-1.0, -1.0], [1.0, 1.0])

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._obs = self._draw()

    def _draw(self) -> np.ndarray:
        return self._rng.normal(size=(3,)).astype(np.float32)

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._obs = self._draw()
        return self._obs.copy(), {}

    def step(self, action):
        target = np.tanh(self._obs[:2])
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        reward = float(-np.sum((action - target) ** 2))
        self._t += 1
        self._obs = self._draw()
        truncated = self._t >= EPISODE_LENGTH
        return self._obs.copy(), reward, False, truncated, {}

    def close(self) -> None:
        pass


def run_agent(agent, env, frames: int, *, seed: int = 0):
    """Drive one agent through one vector env, with no algorithm branches.

    This is deliberately the whole trainer: act, step, observe, and update
    when the agent says so. If an algorithm needed anything else from its
    caller, it would show up here as a branch, and there is none.
    """
    obs, _info = env.reset(seed=seed)
    rewards: list[float] = []
    metrics: list[dict] = []
    for frame in range(0, frames, env.num_envs):
        action = agent.act(obs)
        result = env.step(action)
        agent.observe(Transition.from_step(env.obs_spec, obs, action, result))
        rewards.append(float(np.mean(result[1])))
        if agent.should_update(frame):
            metrics.append(agent.update())
        obs = result[0]
    return np.asarray(rewards, np.float64), metrics


#: The observation groups an agent is built from. A single environment
#: declares the leaf spec it emits and the vector env promotes it; an agent
#: built straight off one env's spec promotes it the same way.
SIGN_GROUP = ObsGroup.of(SignEnv.obs_spec)
REACH_GROUP = ObsGroup.of(ReachEnv.obs_spec)
