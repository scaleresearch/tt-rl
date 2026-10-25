"""Generic, spec-driven environment wrappers.

Every wrapper here works on any environment satisfying `ttrl.envs.base.Env`,
takes its sizes as parameters, and updates the `ObsSpec` or `ActionSpec` it
exposes. Nothing here knows about any particular benchmark.
"""

from ttrl.envs.wrappers.control import (
    ActionRepeat,
    RecordEpisodeStatistics,
    TimeLimit,
)
from ttrl.envs.wrappers.core import (
    StatefulEnv,
    Wrapper,
    require_module,
    restore_env,
    restore_policy_state,
    snapshot_env,
    snapshot_policy_state,
)
from ttrl.envs.wrappers.leaf import PrivilegedLeaf
from ttrl.envs.wrappers.privileged import PrivilegedObservation
from ttrl.envs.wrappers.observation import FrameStack, ResizeObservation
from ttrl.envs.wrappers.reward import ClipReward, ScaleReward, SignReward

__all__ = [
    "ActionRepeat",
    "ClipReward",
    "FrameStack",
    "PrivilegedLeaf",
    "PrivilegedObservation",
    "RecordEpisodeStatistics",
    "ResizeObservation",
    "ScaleReward",
    "SignReward",
    "StatefulEnv",
    "TimeLimit",
    "Wrapper",
    "require_module",
    "restore_env",
    "restore_policy_state",
    "snapshot_env",
    "snapshot_policy_state",
]
