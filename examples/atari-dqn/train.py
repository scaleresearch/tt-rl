"""Train Double DQN on an Atari game on one Tenstorrent chip.

Run `python train.py --help` for the knobs. Everything below uses only
public `ttrl.*` names.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import ttrl
from ttrl.train.checkpoint import BEST, best_path, describe
from ttrl.train.logging import build_logger


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chip", type=int, default=0,
                   help="physical chip id, as ttrl doctor prints it")
    p.add_argument("--frames", type=int, default=10_000, help="environment steps")
    p.add_argument("--game", default="ALE/Breakout-v5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default="runs/breakout_dqn")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-starts", type=int, default=2_000)
    p.add_argument("--resume", default="")
    p.add_argument(
        "--checkpoint-replay", action="store_true",
        help="also store the replay buffer in each checkpoint, which is what "
             "makes --resume continue this run rather than be refused. Off by "
             "default: the transitions are the bulk of the file",
    )
    p.add_argument("--buffer", type=int, default=0,
                   help="replay capacity (default: min(frames, 100k))")
    p.add_argument("--eps-decay", type=int, default=0,
                   help="frames to decay epsilon over (default: frames/2)")
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--target-sync", type=int, default=1_000,
                   help="updates between target network syncs")
    p.add_argument("--eval-interval", type=int, default=0,
                   help="frames between deterministic evals (default: ten "
                        "over the run)")
    p.add_argument("--eval-episodes", type=int, default=2)
    p.add_argument("--checkpoint-interval", type=int, default=0,
                   help="frames between checkpoints (default: halfway and at the end)")
    p.add_argument("--checkpoint-keep", type=int, default=10,
                   help="checkpoints of the run's history to retain")
    return p.parse_args()


def build_config(args):
    return ttrl.TrainConfig(
        algo="dqn",
        total_frames=args.frames,
        # Ten evals over the run, because one eval at the end leaves the
        # best-checkpoint selection with a single candidate and the best file
        # is then just a copy of the final policy.
        eval_interval=args.eval_interval or max(args.frames // 10, 1),
        eval_episodes=args.eval_episodes,
        checkpoint_interval=args.checkpoint_interval or max(args.frames // 2, 1),
        checkpoint_keep=args.checkpoint_keep,
        checkpoint_replay=args.checkpoint_replay,
        log_dir=args.log_dir,
        seed=args.seed,
        device=ttrl.DeviceConfig(chips=1, chip=args.chip),
        env=ttrl.EnvConfig(
            name="atari",
            n_envs=1,
            kwargs={"game": args.game, "sticky": 0.25, "episodic_life": True},
        ),
        # The published Atari number comes off whole-game episodes with
        # untransformed rewards, so the scored environment undoes both of
        # training's shaping choices.
        eval_env=ttrl.EnvConfig(
            name="atari",
            n_envs=1,
            kwargs={
                "game": args.game,
                "sticky": 0.25,
                "episodic_life": False,
                "reward": "none",
            },
        ),
        algo_cfg=ttrl.DQNConfig(
            double=True,
            n_step=args.n_step,
            lr=1e-4,
            train_freq=4,
            target_sync_interval=args.target_sync,
            buffer_capacity=args.buffer or min(args.frames, 100_000),
            eps_start=1.0,
            eps_end=0.01,
            eps_decay_frames=args.eps_decay or max(args.frames // 2, 1),
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
        ),
    )


def report_best(config, result):
    """Name the best-scoring checkpoint the run kept, and what it scored.

    Every run writes one, on `TrainConfig.best_metric`. The retention bound
    prunes the frame-keyed history oldest first, so this file is the only
    place a policy from the middle of a long run survives, and a user who is
    never told the name reaches for `checkpoint.npz` instead.
    """
    best = result["best"]
    if best["best_score"] is None:
        return
    print(f"{describe(best)}: {best_path(config.log_dir)}")
    print(f"    resume or score it by name, as {BEST}")


def main():
    args = parse_args()
    np.random.seed(args.seed)  # weight init draws from numpy's global RNG
    config = build_config(args)
    logger = build_logger(config.log_dir, config.total_frames)
    result = ttrl.run(config, resume=args.resume or None, logger=logger)
    print(json.dumps(result, indent=2))
    print(f"checkpoint: {config.log_dir}/checkpoint.npz")
    report_best(config, result)


if __name__ == "__main__":
    main()
