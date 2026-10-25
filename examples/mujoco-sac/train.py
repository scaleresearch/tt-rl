"""Train SAC on MuJoCo Hopper-v5 on one Tenstorrent chip.

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
    p.add_argument("--frames", type=int, default=100_000, help="environment steps")
    p.add_argument("--task", default="Hopper-v5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default="runs/hopper_sac")
    p.add_argument("--eval-interval", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--resume", default="")
    p.add_argument("--critic-hidden", default="256,256",
                   help="comma-separated widths between the state-action "
                        "concatenation and Q")
    p.add_argument("--critic-encoder", default="auto",
                   help="tower each critic runs before its action is "
                        "concatenated on, so this is where the action enters "
                        "Q. 'auto' on a vector observation is 'identity', no "
                        "tower at all, which fuses at the critic's input so "
                        "every width of --critic-hidden depends on the "
                        "action; that is the published SAC critic. 'mlp' "
                        "fuses after the observation's own hidden layers "
                        "instead, which is what a pixel observation has to do")
    p.add_argument("--privileged", action="store_true",
                   help="give the critic the simulator's full qpos+qvel as a "
                        "critic-only observation field; the actor's input is "
                        "unchanged")
    p.add_argument(
        "--checkpoint-replay", action="store_true",
        help="also store the replay buffer in each checkpoint, which is what "
             "makes --resume continue this run rather than be refused. Off by "
             "default: the transitions are the bulk of the file",
    )
    p.add_argument("--randomize", type=float, default=1.0, metavar="RANGE",
                   help="train across randomised physics: every body mass, "
                        "joint damping, friction and actuator gear is drawn "
                        "log-uniformly in [x/RANGE, x*RANGE] each episode. "
                        "1.0 (the default) uses the nominal model. See "
                        "randomize.py")
    return p.parse_args()


def build_config(args):
    kwargs = {"task": args.task, "privileged": args.privileged}
    name = "mujoco"
    if args.randomize > 1.0:
        # Deferred: randomize imports mujoco, which only this flag needs.
        import randomize  # registers the builder under this name

        name = "mujoco-randomized"
        kwargs.update(mass=args.randomize, damping=args.randomize,
                      friction=args.randomize, gear=args.randomize)
    env = ttrl.EnvConfig(name=name, n_envs=1, kwargs=kwargs, normalize_obs=True)
    return ttrl.TrainConfig(
        algo="sac",
        total_frames=args.frames,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_interval=max(args.frames // 4, 1),
        checkpoint_replay=args.checkpoint_replay,
        log_dir=args.log_dir,
        seed=args.seed,
        device=ttrl.DeviceConfig(chips=1, chip=args.chip),
        env=env,
        # Scored on the same environment: Hopper's reward needs no shaping, so
        # there is nothing for a separate eval env to undo.
        algo_cfg=ttrl.SACConfig(
            lr=3e-4,
            batch_size=args.batch_size,
            buffer_capacity=min(args.frames, 200_000),
            learning_starts=max(5_000, args.batch_size),
            train_freq=1,
            tau=0.005,
            critic_hidden=tuple(int(w) for w in args.critic_hidden.split(",")),
            critic_encoder=args.critic_encoder,
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
    report_best(config, result)
    # Score the saved checkpoint on the protocol and seed the run's own last
    # eval used, so the two numbers are directly comparable: this is the
    # checkpoint round trip, not just a second eval.
    checkpoint = f"{config.log_dir}/checkpoint.npz"
    score = ttrl.evaluate_checkpoint(
        checkpoint,
        episodes=config.eval_episodes,
        seed=config.seed + 10_000,
        chip=args.chip,
    )
    print(
        f"reloaded {checkpoint}: mean {score['mean']:.2f} "
        f"median {score['median']:.2f} scores {score['scores']}"
    )
    print(
        "To deploy this policy off the chip:\n"
        f"    uv run python deploy.py {checkpoint} --against-chip {args.chip}"
    )


if __name__ == "__main__":
    main()
