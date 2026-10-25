"""`ttrl train <config.py> [dotted.key=value ...]`: start a training run.

Two public calls and nothing else: ttrl.load_config builds the config and
ttrl.run acquires what it asks for, trains, and releases it. An override is a
dotted assignment applied on top of the config file, the same syntax
load_config(..., overrides=[...]) takes. A field of the config is named by
its path (`env.n_envs`), and an environment builder's own keyword lives under
`env.kwargs`:

    ttrl train configs/pong_dqn.py algo_cfg.lr=5e-5 env.kwargs.game=ALE/Breakout-v5

`--resume best` continues from the best-scoring checkpoint the run directory
holds rather than its newest one, which is the file a user actually wants: a
return curve is not monotonic, so the newest checkpoint is routinely the
worse policy.

`--stop-reward` and `--stop-patience` build the stop conditions of
`ttrl.train.stop` into a `TrainHooks`, and the closing line reports whether
the run completed its budget or stopped early and why. A run that can end
itself has to say that it did.

PROGRESS IS ON BY DEFAULT, and there is no flag to turn it off. A command
line is a person watching, and the library's own default sink (a JSONL file)
leaves that person with nothing to read for hours. The line is throttled by
wall clock to one every five seconds, so it costs nothing on a run measured
in days and is still there to be tailed. `metrics.jsonl` is written either
way, so nothing about the record depends on who is watching.
"""

from __future__ import annotations

import argparse
import os

import ttrl
from ttrl.cli.logs import LEVELS, quiet_stack_logs
from ttrl.train.checkpoint import BEST, best_path, describe, read_meta
from ttrl.train.config import config_to_dict


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", help="a Python config module exporting `config = TrainConfig(...)`"
    )
    parser.add_argument(
        "overrides", nargs="*", metavar="dotted.key=value",
        help="assignments applied on top of the config, e.g. algo_cfg.lr=5e-5",
    )
    parser.add_argument(
        "--resume", default=None, metavar="CHECKPOINT",
        help="continue from a checkpoint instead of starting fresh; "
             "'best' means the run directory's own checkpoint-best.npz",
    )
    parser.add_argument(
        "--wandb", default=None, metavar="PROJECT",
        help="also log to this Weights & Biases project "
             "(needs pip install 'tt-rl[tracker]')",
    )
    parser.add_argument(
        "--wandb-name", default=None, metavar="NAME",
        help="the W&B run name (default: whatever wandb picks)",
    )
    parser.add_argument(
        "--stop-reward", type=float, default=None, metavar="SCORE",
        help="stop once the run's best_metric passes this score",
    )
    parser.add_argument(
        "--stop-patience", type=int, default=None, metavar="EVALS",
        help="stop once this many evals in a row fail to improve",
    )
    parser.add_argument(
        "--tensorboard", action="store_true",
        help="also write TensorBoard event files under <log_dir>/tb "
             "(needs pip install 'tt-rl[logging]')",
    )
    parser.add_argument(
        "--ttnn-logs", default="error", choices=LEVELS, metavar="LEVEL",
        help="how much of ttnn's own log stream to keep: "
             + ", ".join(LEVELS) + " (default: %(default)s). Raise it to "
             "diagnose hardware.",
    )


def run(args: argparse.Namespace) -> int:
    # Before ttrl, because ttrl imports ttnn on the way to a device and both
    # loggers are read once.
    quiet_stack_logs(args.ttnn_logs)

    bad = [tok for tok in args.overrides if "=" not in tok]
    if bad:
        # Silently ignoring these is how a run quietly uses the wrong lr.
        raise SystemExit(
            f"not an override: {bad}. Every positional after the config path "
            "must be a dotted assignment such as algo_cfg.lr=5e-5."
        )
    config = ttrl.load_config(args.config, overrides=list(args.overrides))
    logger = build_logger(config, tensorboard=args.tensorboard,
                          wandb_project=args.wandb, wandb_name=args.wandb_name)
    result = ttrl.run(
        config,
        resume=resolve_resume(args.resume, config),
        logger=logger,
        hooks=build_hooks(stop_reward=args.stop_reward,
                          stop_patience=args.stop_patience),
    )
    print(f"training finished at frame {result['frame_count']} "
          f"after {result['updates']} update(s) on {result['replicas']} chip(s)")
    report_outcome(result)
    return 0


def resolve_resume(resume: str | None, config) -> str | None:
    """`--resume best` as a path, or the path the user gave, or None.

    The run directory is `config.log_dir`, so the best file of the run being
    continued is the one named. A missing file is refused here with the path
    it looked for: the alternative is `ttrl.run` starting a fresh run and the
    user discovering at the end that their resume was silently ignored.
    """
    if resume != "best":
        return resume

    path = best_path(config.log_dir)
    if not os.path.isfile(path):
        raise SystemExit(
            f"--resume best found no {path}. A run writes that file once an "
            f"evaluation improves on its best score, so a run that has not "
            f"evaluated yet has none. Resume from a specific checkpoint, or "
            f"drop --resume to start fresh."
        )
    print(f"resuming from {path}: {_best_line(path)}")
    return path


def _best_line(path: str) -> str:
    """What the best file says it scored, for the user who is about to ship it.

    Best is the highest-scoring eval, so a run whose policy moves between
    checkpoints has a best file that is partly a lucky draw, and this is the
    moment the user is choosing it. A file this build cannot read is not a
    reason to refuse the resume, which reads it properly a moment later.
    """

    try:
        _arrays, meta = read_meta(path)
        return describe((meta.get("extra") or {}))
    except Exception:
        return "it carries no readable best score"


def build_hooks(*, stop_reward: float | None, stop_patience: int | None):
    """A `TrainHooks` carrying the stop conditions the flags asked for.

    None when neither flag was given, which is what passing no hooks means.
    Both flags together is `AnyOf`, so the reason recorded names the one
    condition that ended the run.
    """

    conditions = []
    if stop_reward is not None:
        conditions.append(ttrl.StopOnRewardThreshold(stop_reward))
    if stop_patience is not None:
        conditions.append(ttrl.StopOnNoImprovement(stop_patience))
    if not conditions:
        return None
    stop = conditions[0] if len(conditions) == 1 else ttrl.AnyOf(*conditions)
    return ttrl.TrainHooks(stop=stop)


def report_outcome(result: dict) -> None:
    """Say whether the run finished its budget or ended itself, and why.

    Early stop is new, so a terminal that reports a stopped run exactly as it
    reports a completed one leaves the user to notice a missing million
    frames on their own.
    """
    # The name the file is written under, so the line and the file cannot
    # disagree.

    if result.get("status") == "stopped":
        print(f"stopped early: {result.get('stop_reason')}")
    # `BestScore.stamp()`, which is what the best checkpoint's own `extra`
    # carries, so the line and the file say the same thing.
    best = result.get("best") or {}
    if best.get("best_score") is not None:
        print(f"{describe(best)}, saved as {BEST}")


def build_logger(config, *, tensorboard: bool = False,
                 wandb_project: str | None = None,
                 wandb_name: str | None = None):
    """The sinks a command-line run gets: the file, the screen, and the trackers.

    `JsonlLogger` is named explicitly because passing any logger replaces the
    Trainer's default one, and the metrics file is the run's evidence.
    """

    sinks = [
        ttrl.JsonlLogger(config.log_dir),
        ttrl.StdoutLogger(total_frames=config.total_frames),
    ]
    if tensorboard:
        sinks.append(ttrl.TensorBoardLogger(config.log_dir))
    if wandb_project:
        # The tracker shows the config beside the curves, and it is handed the
        # config rather than reaching for one: a sink that read a TrainConfig
        # would be a logging module that knows the config schema.
        sinks.append(ttrl.WandbLogger(
            wandb_project, name=wandb_name, config=config_to_dict(config)))
    return ttrl.MultiLogger(*sinks)
