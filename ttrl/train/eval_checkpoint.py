"""Scoring a checkpoint file: load it, resolve its config, open one chip.

`ttrl.train.evaluate` scores an agent on an environment it is handed and
knows nothing about files or devices. This is the entry point that produces
both from a checkpoint path and then delegates to it, together with the two
helpers that bend the checkpoint's own `EnvConfig` towards another level or
game on the way.
"""

from __future__ import annotations

import dataclasses
import json as json_mod
from typing import Any

from ttrl.train.config import load_config
from ttrl.train.config.schema import DeviceConfig, EnvConfig
from ttrl.train.device import device_for
from ttrl.train.restore import agent_from_checkpoint

__all__ = ["evaluate_checkpoint", "with_env_kwargs", "with_game"]


def evaluate_checkpoint(
    checkpoint: str,
    *,
    config_path: str | None = None,
    overrides: list[str] | None = None,
    env_kwargs: dict[str, Any] | None = None,
    n_episodes: int = 30,
    seed: int | None = None,
    chip: int | None = None,
    json_path: str | None = None,
    logger=None,
) -> dict[str, Any]:
    """Score a checkpoint, opening and closing its own device.

    An L1 checkpoint carries its config and its specs, so the path alone is
    enough and no environment is constructed to rebuild the agent, and the
    weights are taken without the optimizer: a score is not a resume.
    `config_path` scores it against a different config than it trained on.

    `env_kwargs` overrides the eval environment builder's own keywords, which
    is how one checkpoint is scored on another level or game: the Atari
    builder calls its id `game`, as `{"game": "ALE/Breakout-v5"}`.

    `chip` is a PHYSICAL chip id, the number `ttrl doctor` prints, and the
    process is scoped to it before ttnn loads.

    `logger` is passed through to `evaluate`, which reports each finished
    episode to it. Nothing is logged if none is given.
    """

    # Imported here because `ttrl.train.evaluate` re-exports this module's
    # names, so importing it at module level would close a cycle.
    from ttrl.train.evaluate import evaluate

    # No video recording. Rendering is the optional `Renderable` protocol
    # (ttrl.envs.render) and `VecEnv` has none, so the env this scores cannot
    # be asked for a frame; compositing N lanes in N subprocesses belongs to
    # whoever wants it.
    config = None
    if config_path is not None:
        config = load_config(config_path, overrides=overrides)

    # A checkpoint stores no DeviceConfig (TrainConfig.UNSAVED_FIELDS): which
    # chip a training run got says nothing about its weights, and scoring on
    # it would send every score at one chip whoever trained the file happened
    # to be given. With no config to read, the defaults plus `chip` decide.
    device_config = config.device if config is not None else DeviceConfig()
    if chip is not None:
        device_config = dataclasses.replace(device_config, chip=chip)
    # `device_for` scopes before ttnn loads and then opens the chip, in that
    # order: scoring a checkpoint must not take the whole board either. One
    # chip is all a score needs, and a multi-chip config is refused there.
    with device_for(device_config) as (device, profile):
        agent, cfg, loaded = agent_from_checkpoint(
            checkpoint, device, config=config, profile=profile, weights_only=True
        )
        env_config = cfg.eval_env_config
        if env_kwargs:
            env_config = with_env_kwargs(env_config, env_kwargs)
        try:
            result = evaluate(
                agent, env_config, n_episodes=n_episodes,
                seed=cfg.seed if seed is None else seed,
                env_policy_state=loaded["env_policy_state"], logger=logger)
        finally:
            agent.close()

    result = {**result, "checkpoint": checkpoint, "frame": loaded.get("frame", 0)}
    if json_path:
        with open(json_path, "w") as fh:
            json_mod.dump(result, fh, indent=2)
    return result


def with_env_kwargs(env_config: EnvConfig, kwargs: dict[str, Any]) -> EnvConfig:
    """`env_config` with `kwargs` over its builder's own keywords.

    Every key must be one the builder already takes, so a typo is named here
    rather than reaching the builder as a keyword it ignores or dies on.
    """
    unknown = sorted(set(kwargs) - set(env_config.kwargs))
    if unknown:
        raise ValueError(
            f"this checkpoint's env builder {env_config.name!r} takes no "
            f"{unknown} keyword(s) (it takes {sorted(env_config.kwargs)}). "
            f"Score it against a different environment with --config instead."
        )
    return dataclasses.replace(
        env_config, kwargs={**env_config.kwargs, **kwargs})


def with_game(env_config: EnvConfig, game: str) -> EnvConfig:
    """`env_config` with its builder's env id replaced, for `ttrl.train.record`.

    `game` is the Atari builder's name for that keyword.
    """
    return with_env_kwargs(env_config, {"game": game})
