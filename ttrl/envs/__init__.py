"""Environment adapters.

This is the whole public surface for writing an environment: the `Env` and
`VecEnv` contracts of `ttrl.envs.base`, the two checkers of
`ttrl.envs.check`, the `info` keys they require, the
optional `Renderable` capability and the two vector implementations, the
`ENVS` registry and its `register`, the `registry` and `wrappers`
submodules, the second of which `RecordEpisodeStatistics` lives in, and the
vector-level `NormalizeObservation` and `NormalizeReward`, which hold one
shared `RunningMeanStd` over every lane rather than one per lane, and
`episode_bound`, which answers whether an environment's episodes end on their
own (`ttrl.envs.bounds`).

`make_atari` and `make_mujoco` are resolved lazily because each pulls in an
optional dependency (ale-py and opencv for Atari, mujoco for MuJoCo).
Importing `ttrl.envs` must stay free of both so that a core install, and
`import ttrl` itself, work without any environment suite present.
"""

import importlib

from ttrl.envs.base import EPISODE_KEY, FINAL_OBS_KEY, LANE_KEY, Env, VecEnv
from ttrl.envs.bounds import episode_bound
from ttrl.envs.check import check_env, check_env_contract, check_vec_env
from ttrl.envs.registry import ENVS, register
from ttrl.envs.render import Renderable, supports_render

__all__ = [
    "Env",
    "VecEnv",
    "Renderable",
    "AsyncVectorEnv",
    "SyncVectorEnv",
    "check_env",
    "check_env_contract",
    "check_vec_env",
    "episode_bound",
    "ENVS",
    "register",
    "supports_render",
    "EPISODE_KEY",
    "FINAL_OBS_KEY",
    "LANE_KEY",
    "make_atari",
    "make_mujoco",
    "registry",
    "wrappers",
    "NormalizeObservation",
    "NormalizeReward",
    "RunningMeanStd",
    "VecWrapper",
]

#: Submodules reachable as `ttrl.envs.<name>`, imported on first access.
_SUBMODULES = ("registry", "wrappers")

#: name -> (module, attribute, extra that provides it)
_LAZY = {
    "AsyncVectorEnv": ("ttrl.envs.vector_async", "AsyncVectorEnv", None),
    "SyncVectorEnv": ("ttrl.envs.vector", "SyncVectorEnv", None),
    "NormalizeObservation": ("ttrl.envs.vec_normalize", "NormalizeObservation", None),
    "NormalizeReward": ("ttrl.envs.vec_normalize", "NormalizeReward", None),
    "RunningMeanStd": ("ttrl.envs.running", "RunningMeanStd", None),
    "VecWrapper": ("ttrl.envs.vec_wrapper", "VecWrapper", None),
    "make_atari": ("ttrl.envs.atari", "make_atari", "atari"),
    "make_mujoco": ("ttrl.envs.mujoco", "make_mujoco", "mujoco"),
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f"ttrl.envs.{name}")
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr, extra = _LAZY[name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{name} needs an optional dependency that is not installed "
            f"({exc.name}). Install it with: pip install 'tt-rl[{extra}]'"
        ) from exc
    return getattr(module, attr)


def __dir__() -> list[str]:
    return sorted(__all__)
