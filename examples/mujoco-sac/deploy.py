"""Turn a checkpoint into a deployable policy, and run it with numpy alone.

    uv run python deploy.py runs/hopper_sac/checkpoint.npz

Writes `<checkpoint>.policy.npz` and scores it on the task. Opens no chip:
everything after the export is numpy, which is the point. The torques this
prints are the torques a robot's host would command from the same file.

`--against-chip N` additionally rebuilds the actor on a Tenstorrent chip and
reports the largest disagreement between the two, in the task's own action
units. That is the number to quote when someone asks whether the exported
file is the policy that trained.
"""

from __future__ import annotations

import argparse

import numpy as np

import ttrl


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="a checkpoint written by train.py")
    p.add_argument("--out", default="", help="artifact path (default: alongside)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--task", default="Hopper-v5")
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--against-chip", type=int, default=None, metavar="N",
                   help="also compare against the actor on this chip")
    return p.parse_args()


def rollout(policy, task, *, episodes, seed):
    """Episodes driven by the artifact, on the unwrapped task.

    No `NormalizeObservation` in this stack: the artifact carries the
    transform the weights were trained under, so a deployment applies it
    inside the policy and the simulator hands over raw observations exactly
    as a robot's sensors would.
    """
    env = ttrl.envs.make_mujoco(task)
    returns, actions = [], []
    try:
        for episode in range(episodes):
            obs, _info = env.reset(seed=seed + episode)
            total, done = 0.0, False
            while not done:
                action = policy(obs)
                actions.append(action)
                obs, reward, terminated, truncated, _info = env.step(action)
                total += reward
                done = terminated or truncated
            returns.append(total)
    finally:
        env.close()
    return np.asarray(returns), np.asarray(actions)


def disagreement(checkpoint, policy, observations, chip):
    """The largest gap between the artifact and the actor on the chip."""
    with ttrl.device_for(ttrl.DeviceConfig(chip=chip)) as (device, _profile):
        agent, _cfg, _loaded = ttrl.agent_from_checkpoint(
            checkpoint, device, weights_only=True)
        try:
            on_chip = agent.act(policy.normalize(observations), deterministic=True)
        finally:
            agent.close()
    return float(np.abs(policy(observations) - on_chip).max())


def main():
    args = parse_args()
    out = args.out or args.checkpoint.replace(".npz", "") + ".policy.npz"
    artifact = ttrl.export_policy(args.checkpoint, out)
    policy = ttrl.load_policy(artifact)
    print(f"wrote {artifact}: {policy}")

    returns, actions = rollout(policy, args.task, episodes=args.episodes,
                               seed=args.seed)
    print(f"{args.episodes} episodes on {args.task}: mean return "
          f"{returns.mean():.1f}, scores {np.round(returns, 1).tolist()}")
    print(f"commanded torque range {actions.min():.3f} to {actions.max():.3f} "
          f"within bounds {policy.action_low} to {policy.action_high}")

    if args.against_chip is not None:
        batch = actions.shape[0]
        # One act() call, so no more observations than the actor's batch.
        sample = _observations(args.task, min(batch, 128), args.seed)
        gap = disagreement(args.checkpoint, policy, sample, args.against_chip)
        print(f"largest disagreement with the actor on chip "
              f"{args.against_chip}: {gap:.5f} action units")


def _observations(task, count, seed):
    """Raw observations from the task, for the comparison batch."""
    env = ttrl.envs.make_mujoco(task)
    try:
        obs, _info = env.reset(seed=seed)
        rows = [obs]
        rng = np.random.default_rng(seed)
        while len(rows) < count:
            obs, _r, terminated, truncated, _info = env.step(
                rng.uniform(env.action_spec.low_array, env.action_spec.high_array))
            rows.append(obs)
            if terminated or truncated:
                obs, _info = env.reset(seed=seed + len(rows))
        return np.asarray(rows[:count], np.float32)
    finally:
        env.close()


if __name__ == "__main__":
    main()
