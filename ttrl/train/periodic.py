"""The two periodic jobs a run does between rounds: score, and checkpoint.

Split out of `ttrl.train.trainer`, which owns a run's resources, its loop and
WHEN these happen. Each is a function of the replica set and the config, so
each is written here once and `Trainer.evaluate` and `Trainer.save_checkpoint`
are the names a caller reaches them by.

Both average the replicas first. Replicas drift apart between syncs, so
replica 0 alone is not the model being trained: averaging first is what makes
the number reported and the weights a checkpoint ships the same thing.
"""

from __future__ import annotations

from ttrl.train.checkpoint import L1, history_path, point_at_latest, prune
from ttrl.train.config import TrainConfig
from ttrl.train.evaluate import evaluate
from ttrl.train.logging import EVAL_PREFIX
from ttrl.train.snapshot import write_checkpoint

__all__ = ["run_eval", "write_run_checkpoint"]


def run_eval(replicas, config: TrainConfig) -> tuple[dict, dict]:
    """Score the lead replica's averaged model. Returns the result and its metrics.

    The eval env starts from the training source's transform state, so a
    normalised policy is scored on the observation scale it was trained on
    rather than on statistics restarted at count 0 (measured at 4.2% of the
    score on a 250-step Hopper episode, and worse where the first steps decide
    the episode). This holds even when `eval_env` declares a different
    protocol: a policy snapshot names only transform-bearing layers, so it
    survives a stack differing in a frame stack or a time limit, and an Atari
    stack carries none at all. A `StepSource` that is not an environment
    declares the same member and answers `[]`, which is why this needs no
    branch; see `ttrl.train.loop.StepSource`.
    """

    if replicas.param_sync is not None:
        replicas.param_sync.average()
    result = evaluate(
        replicas.lead.agent,
        config.eval_env_config,
        n_episodes=config.eval_episodes,
        seed=config.seed + 10_000,
        env_policy_state=replicas.lead.env.save_policy_state(),
    )
    scored = {f"{EVAL_PREFIX}{k}": v for k, v in result.items() if k != "scores"}
    return result, scored


def write_run_checkpoint(replicas, config: TrainConfig, *, frame: int,
                         path: str | None = None, level: str = L1,
                         extra: dict | None = None) -> str:
    """Write a checkpoint of the averaged model and return its path.

    `ttrl.train.snapshot` owns what goes in one and which replica each member
    is read off.

    `path` writes that one file and nothing else, which is how a snapshot is
    published to a chosen name. With no `path` the checkpoint joins the run's
    history: a frame-keyed file in `log_dir`, `checkpoint.npz` pointed at it,
    and anything past `checkpoint_keep` pruned. The cadence in `Trainer.run`
    goes this way, so a long run keeps a progression.
    """
    if replicas.param_sync is not None:
        replicas.param_sync.average()
    target = path if path is not None else history_path(config.log_dir, frame)
    write_checkpoint(replicas, config, target, level=level, frame=frame,
                     extra=extra)
    if path is None:
        point_at_latest(config.log_dir, target)
        prune(config.log_dir, config.checkpoint_keep)
    return target
