"""Rebuilding a runnable agent from a checkpoint file alone.

What a WARM RESUME additionally demands of a file, and the refusals that go
with it, are `ttrl.train.resume`, which reads the agent state through this
module. The two are separate subjects: this one is asked by a score and by a
deployment export as well, and it refuses almost nothing a resume refuses.
The one thing it does refuse is a config this reader had to complete from
today's defaults, because those defaults build the network and a score's
environment; `_refuse_defaulted_config` is where that is argued.

An L1 checkpoint carries its `TrainConfig`, its `ObsSpec` and its
`ActionSpec`, so everything the agent's builder needs is in the file. No
environment is constructed, which means scoring a checkpoint needs neither
the environment suite nor its assets installed.

The one part of the config that is NOT in the file is `device`: the returned
`TrainConfig` carries a default `DeviceConfig`, because which chip the
training run sat on is not a property of the agent. See
`TrainConfig.UNSAVED_FIELDS`.
"""

from __future__ import annotations

from typing import Any

from ttrl.train import builders
from ttrl.train.checkpoint import L0, L1, load_checkpoint
from ttrl.train.config import TrainConfig

__all__ = ["agent_from_checkpoint", "config_from_checkpoint"]


def config_from_checkpoint(path: str) -> TrainConfig:
    """The `TrainConfig` a checkpoint stores. L1 only."""
    loaded = load_checkpoint(path)
    if loaded["level"] != L1:
        raise ValueError(
            f"{path!r} is a {loaded['level']} checkpoint and stores no config. Only "
            "L1 checkpoints are self-describing; pass an explicit config to score it."
        )
    return loaded["config"]


def agent_from_checkpoint(
    path: str,
    device,
    *,
    config: TrainConfig | None = None,
    profile=None,
    weights_only: bool = False,
) -> tuple[Any, TrainConfig, dict]:
    """Build the agent a checkpoint describes and load its state.

    `config` scores a checkpoint against a different run config than it
    trained on, which is a deliberate act. The specs always come from the
    file: they describe the weights, and a mismatch must fail in
    `load_state_dict` rather than train silently wrong.

    `weights_only=True` takes the weights and leaves the optimizer behind,
    which is what scoring wants: a resumable checkpoint can be read without
    inheriting moments the scored agent will never use. An L0 file holds
    nothing else, so it is read that way regardless; the reverse, a plain load
    of a weights-only state, raises in the agent rather than quietly
    restarting the moments at zero.

    Returns `(agent, config, loaded)` where `loaded` is `load_checkpoint`'s
    dict.
    """

    loaded = load_checkpoint(path)
    if config is None:
        _refuse_defaulted_config(loaded, path)
    cfg = config if config is not None else config_from_checkpoint(path)
    agent = builders.build_agent(
        cfg,
        obs_spec=loaded["obs_spec"],
        action_spec=loaded["action_spec"],
        device=device,
        profile=profile,
    )
    only_weights = weights_only or loaded["level"] == L0
    agent.load_state_dict(loaded["agent_state"], weights_only=only_weights)
    return agent, cfg, loaded


def _refuse_defaulted_config(loaded: dict, path: str) -> None:
    """A file whose config this reader had to complete, with no config given.

    A WEIGHTS-ONLY READ IS NOT CONFIG-FREE. `ttrl.train.checkpoint.tree` lets
    a field the file does not carry take today's default, on the ground that
    an optional setting cannot change the stored weight arrays. True, and not
    the question: the config decides what is BUILT AROUND those arrays. The
    network is assembled from `algo_cfg`, so an encoder or a memory the file
    never recorded is chosen here; the env a score then runs comes from
    `cfg.eval_env_config`, so `normalize_obs`, its clip and the whole eval
    protocol are chosen here too. Any of them differing from the training
    run's scores a different function and reports the number as this
    checkpoint's.

    Which is why `config` is the fix and not merely an alternative: given one,
    nothing above is read off the file and the caller owns every such choice.
    """
    absent = loaded.get("defaulted_fields") or []
    if absent:
        raise ValueError(
            f"{path!r} carries no value for {absent}, so this reader would "
            f"supply them from today's defaults. Those fields are not part of "
            f"the stored weights, but they decide what is built around them: "
            f"the network comes from the stored algo config and a score's "
            f"environment from the stored env config, so a default standing "
            f"in for what the run trained under measures a different "
            f"function and labels the number this checkpoint's. The "
            f"checkpoint format is pre-release and carries no migrations, so "
            f"nothing can add those values to this file. Pass the config the "
            f"run was trained from, which replaces the stored one entirely, "
            f"or score a checkpoint written by this build."
        )
