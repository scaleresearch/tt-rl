"""The two places this layer reaches into `ttrl.envs` and `ttrl.algos`.

Both go through a registry, so adding an environment or an algorithm never
means editing anything here. Keeping them in one module is also what lets a
host-only test drive the whole trainer without a device: there are exactly
two constructors to substitute.
"""

from __future__ import annotations

from ttrl.algos.base import check_agent
from ttrl.algos.registry import resolve_and_build
from ttrl.envs.registry import EnvFactory, require_spawnable
from ttrl.envs.vec_normalize import NormalizeObservation, NormalizeReward
from ttrl.envs.vector import SyncVectorEnv
from ttrl.envs.vector_async import AsyncVectorEnv
from ttrl.train.config import TrainConfig
from ttrl.train.config.schema import EnvConfig

__all__ = ["build_agent", "build_env", "require_episode_bound"]


def require_episode_bound(cfg: EnvConfig, *, where: str) -> int:
    """The bound on one episode of `cfg`, refusing an environment that has none.

    Asked at run start, by name, of the environment a run will score on. The
    question and the refusal both live in `ttrl.envs.bounds`, which is also
    where `ttrl.check_env` asks it, so an environment that passes the public
    checker cannot be refused here.

    Costs one environment constructed and closed, with no step taken, and no
    device: this runs before a chip is opened.
    """
    # Late-bound: tests/envs/test_check_superset.py patches the origin once to
    # cover every caller.
    from ttrl.envs.bounds import require_episode_bound as require

    env = EnvFactory(cfg.name, **cfg.kwargs)()
    try:
        return require(env, where=f"{where} {cfg.name!r} ({cfg.kwargs!r})")
    finally:
        env.close()


def build_env(cfg: EnvConfig):
    """Build the run's vectorised environment from its registered name.

    `EnvFactory` carries the registry's entry for the name and the keyword
    arguments, so the same factory works for the in-process and the
    subprocess vector env.

    The contract is checked on one environment built from the factory, for the
    reason `build_agent` checks the agent it built: a failure the construction
    of the stack already determined should be named here rather than arriving
    a whole `eval_interval` later. What is checked is the construction-time
    subset (`check_env_contract`) and not `check_vec_env`: driving the
    contract needs an episode to end, which an environment with long episodes
    does not deliver in any step budget a builder can spend, so a full check
    here would refuse honest environments. `ttrl.envs.check_env` and
    `check_vec_env` run this same function first, which is what makes them a
    superset of what this enforces. The cost is one environment constructed
    and closed, with no step taken.

    Public: this is the one path from an `EnvConfig` to a stepping vector env,
    reachable as `ttrl.build_env`.
    """
    # Late-bound for the same reason as in `require_episode_bound`.
    from ttrl.envs.check import check_env_contract

    factory = EnvFactory(cfg.name, **cfg.kwargs)
    if cfg.asynchronous:
        # Before any lane is spawned: a factory that cannot cross the boundary
        # must be refused here rather than inside multiprocessing.
        require_spawnable(factory)
    probe = factory()
    try:
        check_env_contract(probe, where=f"env {cfg.name!r}")
    finally:
        probe.close()
    if cfg.asynchronous:
        vec = AsyncVectorEnv([factory] * cfg.n_envs)
    else:
        vec = SyncVectorEnv([factory] * cfg.n_envs)
    return _normalized(vec, cfg)


def _normalized(vec, cfg: EnvConfig):
    """The vector env under the normalisers `cfg` asks for.

    THE one place they are applied, so the training stack and the eval stack
    are wrapped identically from one statement. They go here rather than in a
    lane's builder because each holds ONE estimator fitted on every lane's
    rows; a per-lane copy sees 1/N of the data and, under `AsyncVectorEnv`,
    cannot be shared at all.

    Observation whitening goes on first, so reward scaling wraps it and the
    order of the two is a property of this function rather than of a config.
    """

    if cfg.normalize_obs:
        vec = NormalizeObservation(vec, clip=cfg.normalize_obs_clip)
    if cfg.normalize_reward:
        # `TrainConfig.__post_init__` resolves the discount from the
        # algorithm's own and refuses a disagreement, so an unset one here is
        # an `EnvConfig` that never passed through it.
        if cfg.normalize_reward_gamma is None:
            raise ValueError(
                "env.normalize_reward is on and env.normalize_reward_gamma is "
                "unset. It is normally filled from the algorithm's gamma by "
                "TrainConfig; an EnvConfig built on its own has to state it."
            )
        vec = NormalizeReward(vec, gamma=cfg.normalize_reward_gamma,
                              clip=cfg.normalize_reward_clip)
    return vec


def build_agent(
    config: TrainConfig,
    *,
    obs_spec,
    action_spec,
    device,
    profile=None,
    algo_cfg=None,
):
    """Build the run's agent from its registered algorithm name.

    The agent owns its own experience storage, so the lane count goes in here
    rather than being something the loop keeps on its behalf. `algo_cfg`
    overrides the config's own, which is how a replica gets its divided share
    of the run's frame budgets; left None, `config.algo_cfg` is used, and if
    that is None too the keyword is omitted so the algorithm's own defaults
    apply rather than this layer inventing a stand-in for them.

    `check_agent` runs here because `Agent` is a Protocol: a missing method is
    no longer caught at instantiation, and discovering it at the first
    `observe` has already thrown away the run.
    """

    kwargs = {"profile": profile, "n_envs": config.env.n_envs, "seed": config.seed}
    cfg = config.algo_cfg if algo_cfg is None else algo_cfg
    if cfg is not None:
        kwargs["cfg"] = cfg
    agent = resolve_and_build(config.algo, obs_spec, action_spec, device, **kwargs)
    check_agent(agent)
    return agent
