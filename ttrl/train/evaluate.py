"""Scoring an agent, parameterised by the environment config and nothing else.

The protocol is fixed: a separate environment built from the run's own
`EnvConfig`, `act(obs, deterministic=True)`, a fixed number of episodes, and
the summary in `ttrl.train.eval_metrics` of the undiscounted returns the
environment itself reports. What `deterministic=True` means is the agent's
business, so nothing here knows which algorithm it is driving.

Episodes are stepped concurrently, up to `n_envs` at a time and never more
than `n_episodes`. A lane that has finished the episode it was given keeps
being stepped, because a vector environment steps every lane or none, and its
later terminals are ignored: scoring is driven by `done & active`, so exactly
`n_episodes` are counted.

EVERY EPISODE GETS ITS OWN RESET SEED. Episodes run in waves of `lanes`, each
wave reseeding the whole environment, so episode `e` of an eval seeded `s`
starts from `s + e` whatever the lane count and whatever the policy did.
Before this, one `reset(seed=s)` was followed by unseeded autoresets, whose
starts come off the environment's continuing generator and so depended on
how much randomness the earlier episodes consumed: on an environment that
draws during a step, two checkpoints scored under one seed met DIFFERENT
starts from the third episode on, and `n_envs=1` and `n_envs=4` scored one
policy differently. Those starts were already independent draws from the
reset distribution, so this buys comparability rather than spread: over 200
Hopper episodes the two schemes measure the same spread, std 3.30 and 3.32.

Waves cost throughput at `n_envs > 1`. A lane that finishes early is stepped
and ignored until the wave's slowest episode ends, where the previous scheme
repacked a fresh episode onto the freed lane as soon as it came free, so an
eval of uneven episodes now runs at the pace of the longest episode in each
wave. Anyone benchmarking evaluation should expect that rather than measure
it as a regression.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ttrl.envs.base import EPISODE_KEY, LANE_KEY
from ttrl.envs.vec_normalize import freeze_observation_transform
from ttrl.train.builders import build_env
from ttrl.train.config.schema import EnvConfig
from ttrl.train.eval_checkpoint import evaluate_checkpoint, with_env_kwargs, with_game
from ttrl.train.eval_metrics import EVAL_SUMMARY, EVAL_UNRANKABLE, eval_metric_names

__all__ = ["EVAL_SUMMARY", "EVAL_UNRANKABLE", "eval_metric_names", "evaluate",
           "evaluate_checkpoint", "with_env_kwargs", "with_game"]

#: Agent steps a single eval episode may take before the run refuses to wait.
#: Eval runs inline in the training loop, so an episode that never terminates
#: stalls the whole run, and a near-greedy policy on some games produces
#: exactly that. A LAST RESORT and not the detection: `ttrl.train.builders`
#: `require_episode_bound` refuses an eval environment whose episodes are not
#: bounded at run start, before a chip is opened, so what reaches this guard
#: declared a bound and then did not honour it.
#:
#: It is therefore at least the largest bound any shipped environment
#: declares, the ALE protocol's 108,000 emulator frames
#: (`AtariConfig.max_frames`), which at frameskip 4 is 27,000 agent steps, so
#: the number here is loose by design. A guard BELOW the configured limit
#: fires on correct behaviour: at 10,000 it killed a Breakout run six hours
#: in, on a long episode the environment would have truncated and scored
#: by itself.
#: `tests/hardware/test_configs_smoke.py` derives the shipped bounds from the
#: configs and holds this number above every one of them.
MAX_EPISODE_STEPS = 108_000


def _lane_score(info: dict[str, Any], lane: int) -> float:
    """The return the environment reports for a lane that just ended.

    `build_env` refuses a stack that declares no episode statistics, so this
    is reached only by one that declares them and then does not deliver them
    on a terminal step. It says so, rather than raising a bare KeyError.
    """
    try:
        return float(info[LANE_KEY][lane][EPISODE_KEY]["r"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"eval lane {lane} ended its episode but no score is under "
            f"info[{LANE_KEY!r}][{lane}][{EPISODE_KEY!r}]['r'] ({exc!r}). That is "
            f"the one place a score is read from; see ttrl.envs.base and drive "
            f"the environment through ttrl.envs.check_vec_env, which asserts "
            f"it on every terminal step."
        ) from exc


def evaluate(
    agent,
    env_config: EnvConfig,
    *,
    n_episodes: int = 30,
    seed: int | None = None,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    env_policy_state: list | None = None,
    logger=None,
) -> dict[str, Any]:
    """Run `n_episodes` and return their score summary.

    The eval environment is built from `env_config` with its training-only
    shaping left to the config: the caller passes the env it wants scored.

    The observation transform is FROZEN for the duration: see the call to
    `freeze_observation_transform` below for why a score may not be taken
    through a transform that is still being fitted.

    `env_policy_state` is an `Env.save_policy_state()` to start the eval env
    from, which is how a normalised policy is scored on the observation scale
    it was trained on. Without it `NormalizeObservation` starts at count 0 and
    the policy is fed inputs it never saw: 4.2% of the score over 250-step
    Hopper episodes, and larger wherever the opening steps decide the episode.
    It names only transform-bearing layers, so it survives an eval stack
    differing in a frame stack, a time limit or its lane count;
    `load_policy_state` refuses a stack whose transform layers are not the
    checkpoint's.

    `seed` is the eval's base start: episode `e` starts from `seed + e`, so
    two checkpoints scored under one `seed` meet the same starts and one
    checkpoint scored twice gives the same numbers. It does not make the
    result stable: the spread `min`, `max` and `std` report is the policy's
    own behaviour over its reset distribution, and a run whose score swings
    between evals is swinging between checkpoints, which no sample size here
    damps.

    `logger` is any `Logger`, and gets `{"episode_return": r}` per finished
    episode stepped by the episode count, which a `StdoutLogger` built with
    `unit="episode"` reports. Thirty long episodes is a long silence
    otherwise, and the summary arrives only at the end.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
    if max_episode_steps < 1:
        raise ValueError(f"max_episode_steps must be >= 1, got {max_episode_steps}")

    # Never more lanes than episodes: a lane's first terminal is a score, so
    # eight lanes asked for two episodes would return eight of them.
    lanes = min(n_episodes, env_config.n_envs)
    env = build_env(dataclasses.replace(env_config, n_envs=lanes))
    scores: list[float] = []
    try:
        if env_policy_state is not None:
            env.load_policy_state(env_policy_state)
        # SCORING NEVER FITS. A running observation normaliser updates its
        # statistics on every reset and step, so an evaluation that left it
        # fitting would score a transform that moves between its first
        # episode and its last, and report a number for a function nothing
        # else applies: `ttrl.train.record` and `ttrl.deploy` both drive the
        # frozen form, and fitting further on near-deterministic trajectories
        # is what makes the two differ.
        freeze_observation_transform(env)
        while len(scores) < n_episodes:
            _score_one_wave(agent, env, lanes, n_episodes, scores,
                            seed=seed, max_episode_steps=max_episode_steps,
                            logger=logger)
    finally:
        env.close()

    return {"scores": scores,
            **{name: field(scores, n_episodes)
               for name, field in EVAL_SUMMARY.items()}}


def _score_one_wave(agent, env, lanes: int, n_episodes: int, scores: list[float],
                    *, seed: int | None, max_episode_steps: int, logger) -> None:
    """Reseed every lane and score one episode on each, up to the total asked.

    The base seed is `seed` plus the episodes already scored, and the vector
    layer seeds lane `i` from `base + i`, so an eval's starts are `seed + e`,
    one per episode, and rerunning it reproduces them. A final wave with
    fewer episodes left than lanes scores only the lanes it needs; the rest
    are stepped and ignored, as a lane that has already scored is.
    """
    wanted = min(lanes, n_episodes - len(scores))
    base = None if seed is None else int(seed) + len(scores)
    obs, _info = env.reset(seed=base)
    active = np.arange(lanes) < wanted
    starts = np.ones(lanes, dtype=bool)
    steps = np.zeros(lanes, dtype=np.int64)
    while active.any():
        # The previous step's boundaries, where a policy with memory resets
        # its hidden state. Evaluation feeds no transitions back.
        actions = agent.act(obs, deterministic=True,
                            episode_starts=starts.astype(np.float32))
        obs, _rewards, terminated, truncated, info = env.step(actions)
        steps += 1
        done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        starts = done
        overrun = np.flatnonzero((steps >= max_episode_steps) & ~done & active)
        if overrun.size:
            raise RuntimeError(
                f"eval lane(s) {overrun.tolist()} ran {max_episode_steps} steps "
                "without the episode ending. Eval blocks the training loop, so "
                "this is a stall rather than a slow episode. The episode "
                "bound this environment declares is not one it honours: see "
                "ttrl.envs.bounds and ttrl.train.builders."
                "require_episode_bound, which refuses an unbounded eval "
                "environment at run start."
            )
        for lane in np.flatnonzero(done & active):
            lane = int(lane)
            scores.append(_lane_score(info, lane))
            if logger is not None:
                logger.log({"episode_return": scores[-1]}, len(scores))
            steps[lane] = 0
            active[lane] = False
