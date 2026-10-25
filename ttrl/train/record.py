"""Watching a trained policy: one environment, one agent, frames to an mp4.

ONE LANE, DELIBERATELY. `ttrl.train.evaluate` scores on a `VecEnv` and a
`VecEnv` has no rendering, so this builds a single `Env` with
`render_mode="rgb_array"` and leaves the N-lane question unsolved. The
returns reported are the recorded episodes' own; a score is
`ttrl.evaluate`'s answer over its own episode count.

A RECORDING IS A DEPLOYMENT, not a shortened run. The observation transform a
run fitted (`ttrl.envs.vec_normalize.NormalizeObservation`) is one estimator
above the whole vector environment, so there is no vector env here for it to
sit on; but a finished run's statistics are a fixed affine map and a clip,
which is exactly how `ttrl.deploy` reads them. So the transform is applied
here in its frozen form, from the checkpoint's own `env_policy_state` rather
than from a config flag, and nothing about rendering reaches the vector
layer. The alternative, a one-lane `SyncVectorEnv` under the live normaliser,
would need rendering on a `VecEnv` (which `AsyncVectorEnv` cannot honestly
offer) and would keep fitting the estimator on the clip it is filming.

The frame filmed is the emulator's native colour frame, unrelated to the
observation the policy trains on: `ttrl.train.film` composites it, and the
observation pipeline is read, never altered. The encoder is OpenCV, imported
when a file is actually written, so nothing on the training path depends on
it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ttrl.envs.render import check_frame
from ttrl.envs.vec_normalize import FrozenNormalization, NormalizeObservation
from ttrl.envs.wrappers.core import snapshot_policy_state
from ttrl.train.config import load_config
from ttrl.train.config.schema import DeviceConfig, EnvConfig
from ttrl.train.device import device_for
from ttrl.train.evaluate import with_game
from ttrl.train.film import ContinuousFilm, Film
from ttrl.train.record_env import render_env
from ttrl.train.restore import agent_from_checkpoint
from ttrl.train.video import Mp4Writer

__all__ = ["record_video"]

#: Steps one recorded episode may take before giving up: a near-greedy policy
#: on some games survives indefinitely, and an unbounded clip fills a disk.
MAX_EPISODE_STEPS = 10_000

#: The keyword each env builder names its task with, for the panel's title.
#: `EnvConfig.name` is the builder ("mujoco"), which is not what is on screen.
TITLE_KEYS = ("game", "task")


def record_video(
    checkpoint: str,
    video_path: str,
    *,
    episodes: int = 1,
    config_path: str | None = None,
    overrides: list[str] | None = None,
    game: str | None = None,
    seed: int | None = None,
    chip: int | None = None,
    fps: float = 15.0,
    max_episode_steps: int = MAX_EPISODE_STEPS,
) -> dict[str, Any]:
    """Record `episodes` of `checkpoint` playing, as an mp4 at `video_path`.

    The checkpoint carries its own config and specs, so the path alone is
    enough; `config_path` records against a different config, as in
    `ttrl.evaluate_checkpoint`, and `game` replaces the builder's env id. Returns `{"video", "episodes", "frames", "returns", "checkpoint", "frame"}`.
    """
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    config = None
    if config_path is not None:
        config = load_config(config_path, overrides=overrides)

    # A checkpoint stores no DeviceConfig: which chip a run trained on says
    # nothing about its weights.
    device_config = config.device if config is not None else DeviceConfig()
    if chip is not None:
        device_config = dataclasses.replace(device_config, chip=chip)

    with device_for(device_config) as (device, profile):
        agent, cfg, loaded = agent_from_checkpoint(
            checkpoint, device, config=config, profile=profile, weights_only=True)
        env_config = cfg.eval_env_config
        if game is not None:
            env_config = with_game(env_config, game)
        env = render_env(env_config)
        try:
            whiten = _frozen_transform(env, loaded["env_policy_state"], checkpoint)
            returns, frames = _roll(
                agent, env, video_path, whiten=whiten, episodes=episodes,
                film=_film(env, env_config, loaded.get("frame", 0)),
                seed=cfg.seed if seed is None else seed, fps=fps,
                max_episode_steps=max_episode_steps,
            )
        finally:
            env.close()
            agent.close()

    return {
        "video": video_path,
        "episodes": len(returns),
        "frames": frames,
        "returns": returns,
        "checkpoint": checkpoint,
        "frame": loaded.get("frame", 0),
    }


def _film(env, env_config: EnvConfig, frame: int):
    """The compositor for this environment's action space.

    The panels are a mapping keyed by `ActionSpec.kind`, so the kinds the
    refusal names are the ones that are actually dispatched on, and a kind
    nobody drew a block for is refused rather than filmed with a blank one.
    """

    title = next((str(env_config.kwargs[k]) for k in TITLE_KEYS
                  if k in env_config.kwargs), env_config.name)
    shared = {"title": title, "subtitle": f"trained {frame:,} frames"}
    spec = env.action_spec
    panels = {
        "discrete": lambda: Film(**shared),
        "continuous": lambda: ContinuousFilm(
            bounds=np.maximum(np.abs(spec.low_array), np.abs(spec.high_array)),
            **shared),
    }
    if spec.kind not in panels:
        raise ValueError(
            f"no video panel draws an action of kind {spec.kind!r}; the ones "
            f"drawn are {sorted(panels)}. See ttrl.train.film.")
    return panels[spec.kind]()


def _frozen_transform(env, policy_state, checkpoint: str):
    """This checkpoint's observation transform, as a fixed function or None.

    The authority is the FILE, not the config: `env_policy_state` says what
    the weights were trained under, and a recorder that read a config flag
    instead would film a renamed or hand-edited config's idea of the stack.

    Refused by name, rather than left to a stack-mismatch message that cannot
    say why this stack has no such layer:
      * a transform this module cannot reimplement frozen, because dropping it
        would film a policy fed observations on a scale it was never fitted on;
      * a single-env stack that carries transform state of its own, because
        the recorder would then apply the whitening twice;
      * a file too old to carry the member at all, which says nothing either
        way about what was applied.
    """

    carried = [name for name, _ in snapshot_policy_state(env)]
    if carried:
        raise ValueError(
            f"cannot record {checkpoint!r}: the recorded environment's own "
            f"stack carries policy state for {carried}, which applies a "
            f"transform this recorder also applies from the checkpoint, so "
            f"the observations would be transformed twice. Restore that "
            f"layer with ttrl.envs.wrappers.restore_policy_state instead.")
    if policy_state is None:
        raise ValueError(
            f"cannot record {checkpoint!r}: it carries no 'env_policy_state', "
            f"so it does not say what transform its observations went "
            f"through. Every run under this version of ttrl stores that member, "
            f"and the format is pre-release and carries no migrations, so "
            f"nothing can add it to this file. An EMPTY one is an answer "
            f"and records; absent is not, and filming a whitened policy with "
            f"no whitening would show a policy fed observations on a scale it "
            f"was never fitted on. Re-save the checkpoint from a run under "
            f"this version of ttrl.")
    if not policy_state:
        return None
    known = NormalizeObservation.__name__
    names = [name for name, _ in policy_state]
    if names != [known]:
        raise ValueError(
            f"cannot record {checkpoint!r}: it was trained under the "
            f"transform-bearing wrapper(s) {names}, and the only one this "
            f"recorder applies in its frozen form is {known!r}. Dropping the "
            f"rest would film a policy fed observations on a scale it was "
            f"never fitted on. Score it with ttrl.evaluate_checkpoint, which "
            f"builds the vector stack the transforms live on, or add them to "
            f"ttrl.envs.vec_normalize.FrozenNormalization.")
    return FrozenNormalization(dict(policy_state[0][1]))


def _rest_action(spec):
    """What the panel reads before the first action of an episode exists."""
    if spec.kind == "discrete":
        return 0
    return np.zeros(spec.dim, dtype=np.float32)


def _agent_view(obs) -> np.ndarray | None:
    """The newest frame of a stacked image observation, or nothing to show."""
    ok = isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.dtype == np.uint8
    return obs[-1] if ok else None


def _roll(agent, env, video_path: str, *, episodes: int, seed: int | None,
          fps: float, max_episode_steps: int, film, whiten,
          ) -> tuple[list[float], int]:
    """Drive `episodes` greedy episodes, writing every frame as it is produced."""

    returns: list[float] = []
    frames = 0
    writer = Mp4Writer(video_path, fps=fps)
    try:
        for episode in range(episodes):
            obs, _info = env.reset(seed=None if seed is None else seed + episode)
            total, steps = 0.0, 0
            action = _rest_action(env.action_spec)
            # The first observation of an episode begins one, where a policy
            # with memory resets its hidden state.
            start = np.ones(1, dtype=np.float32)
            while True:
                writer.write(film.frame(
                    check_frame(env.render()), ret=total, step=steps,
                    action=action, n_actions=env.action_spec.width,
                    agent_view=_agent_view(obs)))
                frames += 1
                # `act` takes a leading lane axis, so one env arrives batched,
                # and the transform is fitted on lane-major rows too.
                rows = obs[None]
                action = agent.act(rows if whiten is None else whiten.apply(rows),
                                   deterministic=True, episode_starts=start)[0]
                start = np.zeros(1, dtype=np.float32)
                obs, reward, terminated, truncated, _info = env.step(action)
                total += float(reward)
                steps += 1
                if terminated or truncated:
                    break
                if steps >= max_episode_steps:
                    raise RuntimeError(
                        f"recorded episode {episode} ran {max_episode_steps} "
                        f"steps without ending. Add a TimeLimit to the "
                        f"environment's config, or raise max_episode_steps.")
            returns.append(total)
    finally:
        writer.close()
    return returns, frames
