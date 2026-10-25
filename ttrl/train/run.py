"""`run(config)`: acquire what the config asks for, train, and hand back the result.

The one-call entry point. The scope-acquire-Trainer sequence below is the only
copy of it: a command-line front end, a launcher and a user's own script all
call this rather than writing it again.
"""

from __future__ import annotations

from typing import Any

from ttrl.algos import ALGOS
from ttrl.engine import acquire
from ttrl.train.config import TrainConfig
from ttrl.train.provenance import write_run_metadata
from ttrl.train.scoping import scope_for_run
from ttrl.train.trainer import Trainer

__all__ = ["run"]


def run(config: TrainConfig, *, resume: str | None = None, logger=None,
        hooks=None, cadence=None, source=None,
        agent=None) -> dict[str, Any]:
    """Train `config` to completion and return the Trainer's result dict.

    `hooks` is a `ttrl.train.hooks.TrainHooks`, passed through to the
    `Trainer`. It is how curriculum, an external override, a tracker of the
    caller's own and an early stop reach a run that goes through this entry
    point rather than constructing a `Trainer` by hand.

    `cadence`, `source` and `agent` are the `Trainer`'s stage factories,
    passed straight through: this entry point is how a third-party update
    cadence, step source or agent adaptation -- a `ttrl.EpisodeBoundaries`,
    say -- reaches a run that is driven from a config file. See
    `ttrl.train.trainer.Trainer`.

    Scopes the process to its chip first, which has to happen before ttnn is
    imported and therefore cannot happen inside `acquire`. See
    `ttrl.train.scoping`.

    At `config.device.chips == 1` this opens the one chip the config names and
    releases it on the way out. Otherwise the Trainer opens one independent
    device per resolved chip itself, so acquiring a handle here would hold a
    chip the replicas are about to open.
    """
    resume_path = resume or config.resume_path or None
    # Before the device and before the first frame: a run killed early, which
    # on a contended box means killed while waiting for a chip, still leaves
    # the config and the code state that produced its curve.
    write_run_metadata(config.log_dir, config)
    scope_for_run(config.device)
    # The record is rewritten with the resolved chip by `Trainer.__init__`,
    # which every path below goes through, so a run composed without this
    # entry point records the same thing.
    _check_algo_name(config.algo)
    if config.device.chips != 1:
        with Trainer(config, None, None, resume_path=resume_path, logger=logger,
                     hooks=hooks, cadence=cadence, source=source,
                     agent=agent) as trainer:
            return trainer.run()

    # No chip id: after scoping the visible set is this run's one chip, and a
    # local index into it would be the second notion of chip identity that
    # `DeviceConfig` exists to avoid. `acquire_timeout` is how long this waits
    # for a free chip; the wait is a number the caller sets and never a retry
    # loop, because a caller on a contended box owns that policy.
    with acquire(
        cfg=config.device.engine_config(),
        timeout=config.device.acquire_timeout,
    ) as (device, profile):
        with Trainer(
            config, device, profile, resume_path=resume_path, logger=logger,
            hooks=hooks, cadence=cadence, source=source, agent=agent,
        ) as trainer:
            return trainer.run()


def _check_algo_name(algo: str) -> None:
    """Refuse an unregistered algorithm before a chip is acquired.

    `in` costs no import: both registries hold `"module:attr"` targets and
    answer which names exist out of their table, so this asks the same
    question of `ALGOS` that `ttrl.train.config.loader` asks of `ENVS`, and
    neither reaches ttnn. It stays here rather than moving beside that one
    because a `TrainConfig` built by hand never passes through
    `ttrl.load_config`, and this is the last point every run goes through
    before a device is opened.
    """

    if algo not in ALGOS:
        raise ValueError(
            f"algo {algo!r} is not a registered algorithm; registered names "
            f"are {ALGOS.names()}. Fix the name, or register yours with "
            "ttrl.algos.registry.ALGOS.register(name, builder)."
        )
