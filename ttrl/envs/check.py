"""The contract of `ttrl.envs.base`, as an executable test.

`check_env` and `check_vec_env` drive a real environment and raise on the
first violation with a message naming what was expected. Everything they
assert is stated in `ttrl.envs.base` and asserted by one clause each in
`ttrl.envs.check_clauses`; this module is only the driving, which is why it is
separate from both.

Actions are sampled from the `ActionSpec`, and for an environment that
declares a legal-action mask they are sampled from the mask that the
observation in hand carries. An environment that refuses an illegal action is
therefore checkable, which it was not while the sampler drew uniformly over
the whole action set.
"""

from __future__ import annotations

import numpy as np

from ttrl.envs import bounds
from ttrl.envs import check_clauses as clauses
from ttrl.envs.base import EPISODE_KEY, Env, VecEnv
from ttrl.envs.obs_batch import observation_group
from ttrl.envs.render import check_frame, supports_render
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec

__all__ = ["check_env", "check_env_contract", "check_vec_env"]


def check_env_contract(env: Env, *, where: str = "the environment") -> None:
    """Assert the part of the `Env` contract that costs no step.

    What `check_env` and `check_vec_env` assert needs a driven environment and
    an episode that ends, which an environment with long episodes does not
    deliver within any step budget a builder could afford. This is the subset
    a constructed environment can be held to immediately, and it is run in
    three places: `ttrl.train.builders.build_env` on every run, and both
    public checkers before either of them steps anything. That is what keeps
    the checkers a strict SUPERSET of what a run enforces, so a clause added
    here cannot pass `check_env` and then fail a build.
    `tests/envs/test_check_superset.py` pins the relationship.
    """
    for name, kind in (("obs_spec", (ObsSpec, ObsGroup)), ("action_spec", ActionSpec)):
        spec = getattr(env, name, None)
        if not isinstance(spec, kind):
            raise TypeError(
                f"{where} declares {name} {spec!r}, which is not a "
                f"{kind}. Every environment carries both specs; see "
                f"ttrl.envs.base.Env."
            )


def check_env(env: Env, *, steps: int = 20, seed: int = 0,
              bounded: bool = True) -> None:
    """Assert that `env` honours the single-environment contract.

    Runs `check_env_contract` first and then drives, so everything a run
    enforces is enforced here too and this is a strict superset of it.
    Raises on the first violation with a message naming what was expected.

    REFUSES an environment whose episodes are not bounded, which every run
    refuses at `Trainer.__init__`: a checker that passes what the trainer
    rejects sends its reader off to train a configuration that cannot start.
    `ttrl.envs.bounds` owns the question and the message. It is asked here
    and not in `check_env_contract`, which `build_env` runs on the TRAINING
    environment too, nor in `check_vec_env`, whose lanes are not layers of
    its stack and so cannot answer. `bounded=False` is the author declaring
    an endless environment they only train on and score through a separate
    `TrainConfig.eval_env`; the refusal names it.

    WARNS, rather than raising, when every episode in the rollout ended as a
    truncation before the declared bound, which is what a `step` returning its
    terminated and truncated flags in the other order looks like from out here.
    `ttrl.envs.check_clauses.warn_if_swapped` says why that one is a warning.
    """
    check_env_contract(env)
    bound = None
    if bounded:
        # Through the module, so one patch of the question covers this
        # caller and the run's; `tests/envs/test_check_superset.py`.
        bound = bounds.require_episode_bound(env, where="the environment")
    spec, aspec = observation_group(env.obs_spec), env.action_spec
    rng = np.random.default_rng(seed)

    obs, info = env.reset(seed=seed)
    obs = clauses.check_obs(obs, spec, lanes=None, where="reset")
    if not isinstance(info, dict):
        raise TypeError(f"reset must return a dict info, got {type(info).__name__}")
    # Rendering is optional, so an environment that does not offer it passes.
    # One that does is held to the frame format from its first frame on.
    if supports_render(env):
        check_frame(env.render(), where="render after reset")

    ends: list[tuple[bool, int]] = []
    episode_steps = 0
    for i in range(steps):
        action = clauses.sample_action(aspec, rng, None, spec, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_steps += 1
        obs = clauses.check_obs(obs, spec, lanes=None, where=f"step {i}")
        if not np.isfinite(reward):
            raise ValueError(f"step {i} returned non-finite reward {reward}")
        if not isinstance(terminated, (bool, np.bool_)):
            raise TypeError(f"step {i} terminated must be bool, got {type(terminated).__name__}")
        if not isinstance(truncated, (bool, np.bool_)):
            raise TypeError(f"step {i} truncated must be bool, got {type(truncated).__name__}")
        if terminated or truncated:
            ends.append((bool(terminated), episode_steps))
            episode_steps = 0
            # Not required of a single `Env`: the vector env accumulates the
            # return and length itself (`ttrl.envs.episodes`). One that
            # supplies the key anyway, which is what
            # `RecordEpisodeStatistics` below a reward transform is for, is
            # held to its shape.
            if isinstance(info, dict) and EPISODE_KEY in info:
                clauses.check_episode(info, where=f"step {i}")
            obs, info = env.reset()
            obs = clauses.check_obs(obs, spec, lanes=None,
                                    where="reset after episode end")
    clauses.warn_if_swapped(ends, bound)


def check_vec_env(env: VecEnv, *, steps: int = 40, seed: int = 0) -> None:
    """Assert that `env` honours the vector contract, including NEXT_STEP autoreset.

    Runs `check_env_contract` first and then drives, so everything a run
    enforces is enforced here too and this is a strict superset of it.
    Raises on the first violation with a message naming what was expected.
    """
    check_env_contract(env)
    n, spec, aspec = env.num_envs, observation_group(env.obs_spec), env.action_spec
    if n < 1:
        raise ValueError(f"num_envs must be at least 1, got {n}")
    rng = np.random.default_rng(seed)

    obs, info = env.reset(seed=seed)
    obs = clauses.check_obs(obs, spec, lanes=n, where="reset")

    saw_episode_end = False
    pending_autoreset = np.zeros(n, dtype=bool)
    for i in range(steps):
        actions = clauses.sample_action(aspec, rng, n, spec, obs)
        obs, rewards, terminated, truncated, info = env.step(actions)
        obs = clauses.check_obs(obs, spec, lanes=n, where=f"step {i}")
        for name, arr in (("rewards", rewards), ("terminated", terminated), ("truncated", truncated)):
            arr = np.asarray(arr)
            if arr.shape != (n,):
                raise ValueError(f"step {i} {name} has shape {arr.shape}, expected {(n,)}")
        if not np.isfinite(np.asarray(rewards)).all():
            raise ValueError(f"step {i} returned a non-finite reward")

        done = np.asarray(terminated) | np.asarray(truncated)
        if done.any():
            clauses.check_final_obs(info, obs, done, spec, where=f"step {i}")
            clauses.check_lane_episodes(info, done, where=f"step {i}")
        # NEXT_STEP autoreset: a lane that ended last step must not report ending
        # again on the step that delivers its new episode's first observation.
        both = pending_autoreset & done
        if both.any():
            raise ValueError(
                f"step {i} reports lanes {np.flatnonzero(both).tolist()} as done "
                f"immediately after their autoreset. The contract is NEXT_STEP: the "
                f"step following a termination delivers the new episode's first "
                f"observation and must report done False for that lane."
            )
        pending_autoreset = done
        saw_episode_end |= bool(done.any())

    if not saw_episode_end:
        raise ValueError(
            f"no lane ended an episode in {steps} steps, so autoreset went untested. "
            f"Raise steps, or use an environment with shorter episodes."
        )
