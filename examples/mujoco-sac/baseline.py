"""Score a uniform-random policy on the same eval protocol `train.py` uses.

No chip is opened: a random policy needs no network, and `ttrl.evaluate` only
asks an agent for `act(obs, deterministic=True)`. This is the number a trained
policy has to beat.
"""

from __future__ import annotations

import argparse

import numpy as np

import ttrl
import ttrl.envs


class RandomAgent:
    def __init__(self, action_spec, seed=0):
        self.spec = action_spec
        self.rng = np.random.default_rng(seed)

    def act(self, obs, deterministic=False, episode_starts=None):
        lanes = obs.lanes
        low = np.asarray(self.spec.low, dtype=np.float32)
        high = np.asarray(self.spec.high, dtype=np.float32)
        return self.rng.uniform(low, high, size=(lanes, self.spec.dim)).astype("float32")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="Hopper-v5")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    probe = ttrl.envs.make_mujoco(task=args.task)
    action_spec = probe.action_spec
    probe.close()

    result = ttrl.evaluate(
        RandomAgent(action_spec, seed=args.seed),
        ttrl.EnvConfig(name="mujoco", n_envs=1, kwargs={"task": args.task}),
        n_episodes=args.episodes,
        seed=args.seed,
    )
    print(
        f"random policy on {args.task}: mean {result['mean']:.1f} "
        f"median {result['median']:.1f} min {result['min']:.1f} max {result['max']:.1f}"
    )


if __name__ == "__main__":
    main()
