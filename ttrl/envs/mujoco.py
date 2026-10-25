"""MuJoCo continuous control, the robotics validation target.

Locomotion tasks are the counterexample to every Atari assumption: the
observation is a short float32 vector rather than a stack of frames, the
action is a bounded torque vector rather than one of eighteen buttons, the
reward has meaningful magnitude, and the episode ends on a time limit the
simulator owns. Nothing here is shared with `ttrl.envs.atari`; both reach the
rest of the library only through `ObsSpec` and `ActionSpec`.

Hopper-v5 is the default because it is the cheapest of the standard
locomotion suite (3 actuators, 11 observations) while still requiring gait,
so a full training run fits on one chip.

Gymnasium reports MuJoCo observations as float64. `ObsSpec` admits only
uint8 and float32, and the device pipeline these observations feed is fp32
throughout, so float32 is the precision the numbers were going to be carried
at anyway and the adapter casts on the way in. Anyone comparing against a
float64 reference implementation should expect a difference here and should
not read it as the adapter being wrong.

Privileged state. `MujocoConfig.privileged` appends the simulator's full
generalised position and velocity to every observation as an `ObsSpec` field
marked critic-only, which is how an asymmetric actor-critic is expressed here:
the critic conditions on simulator state, the policy keeps the task's own
observation, and no network over the privileged columns is ever built for the
actor.

Camera. A locomotion policy's whole point is that it travels, so a camera
bolted to the origin loses the body within a few seconds of gait and the rest
of a clip is empty floor. Every locomotion model ships a camera named "track"
that follows the body, and `MujocoEnv` takes `camera_name`, so
`MujocoConfig.track_camera` selects it through the env's own constructor and
nothing reaches into the renderer. It is read only when `render_mode` is set,
and the observation is untouched either way: rendering and the observation
pipeline share no state.

`gymnasium` and `mujoco` are imported at call time.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import numpy as np

from ttrl.envs.gym_adapter import GymAdapter
from ttrl.envs.wrappers.core import require_module
from ttrl.envs.wrappers import (
    PrivilegedObservation,
    RecordEpisodeStatistics,
    TimeLimit,
)

__all__ = ["MujocoConfig", "MujocoSim", "make_mujoco", "sim_state",
           "LOCOMOTION_TASKS", "MOVED_TO_ENV_CONFIG", "TRACKING_CAMERA"]

#: Keywords this builder used to take that now belong to `EnvConfig`, because
#: what they switched on is one estimator over every lane and a lane's builder
#: can only make a per-lane one. Named rather than removed silently: a config
#: carrying them would otherwise fail with a bare dataclass TypeError.
MOVED_TO_ENV_CONFIG = {
    "normalize_obs": "env.normalize_obs",
    "normalize_clip": "env.normalize_obs_clip",
}

#: Standard locomotion tasks, cheapest first. Not a closed set: any
#: Gymnasium MuJoCo id with a 1-D Box observation works.
LOCOMOTION_TASKS = ("Hopper-v5", "Walker2d-v5", "HalfCheetah-v5", "Ant-v5")

#: The camera every Gymnasium locomotion model declares to follow its body.
#: Gymnasium leaves it unselected for backward compatibility, so a recording
#: has to name it.
TRACKING_CAMERA = "track"


def _has_tracking_camera(env) -> bool:
    # Optional dependency; make_mujoco refuses with the extra's name before this runs.
    import mujoco

    model = env.gym_env.unwrapped.model
    return mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, TRACKING_CAMERA) >= 0


class MujocoSim(GymAdapter):
    """One MuJoCo task, declaring the simulator state a checkpoint must carry.

    Generalised position and velocity are the whole dynamical state, so
    together with Gymnasium's generator (which draws the reset perturbation)
    they are enough to resume a trajectory exactly.
    """

    POLICY_STATE = ()

    @property
    def _sim(self):
        return self.gym_env.unwrapped

    def get_state(self) -> dict[str, Any]:
        data = self._sim.data
        return {
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "np_random": self._sim.np_random.bit_generator.state,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._sim.set_state(state["qpos"], state["qvel"])
        self._sim.np_random.bit_generator.state = state["np_random"]


def sim_state(env) -> "np.ndarray":
    """The simulator's full generalised position and velocity.

    This is privileged state: a locomotion task's observation omits the root
    x coordinate, and no real robot measures its own generalised velocities
    directly. It is what an asymmetric critic is given while the policy keeps
    only the task's own observation.
    """
    data = env.gym_env.unwrapped.data
    return np.concatenate([data.qpos, data.qvel]).astype(np.float32)


@dataclass(frozen=True)
class MujocoConfig:
    task: str = "Hopper-v5"
    #: Append `sim_state` to every observation as a critic-only ObsSpec
    #: field. The actor's half of the observation is unchanged, so a policy
    #: trained this way runs against the unprivileged environment.
    privileged: bool = False
    #: Report undiscounted return and length in info["episode"] at every
    #: episode end, below normalisation so the reward is the task's own.
    record_statistics: bool = True
    #: Override the task's own time limit. None keeps the registered one.
    max_steps: int | None = None
    render_mode: str | None = None
    #: Render from the model's own "track" camera, which follows the body, so
    #: a travelling body stays in frame. Read only when `render_mode` is set.
    #: None, the declared default, tracks when the model declares that camera
    #: and keeps the task's static camera when it does not, which is the right
    #: answer for a fixed-base task like Reacher. True demands tracking and
    #: refuses a model that declares no such camera rather than handing back a
    #: camera that never moves. False keeps the task's own camera.
    track_camera: bool | None = None

    def __post_init__(self) -> None:
        if not self.task:
            raise ValueError(
                "task is empty and must be a Gymnasium MuJoCo env id, such as "
                "'Hopper-v5'. Pass it through env.kwargs: "
                "EnvConfig(name='mujoco', kwargs={'task': 'Hopper-v5'})."
            )
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1 or None, got {self.max_steps}")


def make_mujoco(
    task: str | None = None,
    *,
    config: MujocoConfig | None = None,
    seed: int | None = None,
    **kwargs,
):
    """Build a MuJoCo locomotion env behind the `Env` contract."""
    moved = sorted(k for k in kwargs if k in MOVED_TO_ENV_CONFIG)
    if moved:
        raise ValueError(
            f"make_mujoco no longer takes {moved}: observation whitening is "
            f"one running estimator shared by every lane, so it is applied "
            f"above the vector environment and switched on by "
            f"{[MOVED_TO_ENV_CONFIG[k] for k in moved]} instead. A per-lane "
            f"copy of it saw 1/N of the data and whitened each lane against "
            f"its own statistics."
        )
    if config is not None:
        if task is not None or kwargs:
            raise ValueError(
                "make_mujoco takes either config= or field overrides, not both"
            )
    else:
        if task is not None:
            kwargs["task"] = task
        config = MujocoConfig(**kwargs)

    gym = require_module("gymnasium", "mujoco")
    require_module("mujoco", "mujoco")
    make_kwargs: dict[str, Any] = {"render_mode": config.render_mode}
    tracking = config.render_mode is not None and config.track_camera is not False
    if tracking:
        make_kwargs["camera_name"] = TRACKING_CAMERA
    env = MujocoSim(gym.make(config.task, **make_kwargs))
    if tracking and not _has_tracking_camera(env):
        if config.track_camera:
            env.close()
            raise ValueError(
                f"{config.task} declares no camera named "
                f"{TRACKING_CAMERA!r}, so track_camera=True would render from "
                f"a camera that never follows anything. Pass "
                f"track_camera=False to record it with the task's own camera."
            )
    if env.obs_spec.kind != "vector":
        raise ValueError(
            f"{config.task} has observation spec {env.obs_spec}, which is not a "
            f"vector. This adapter is for state-based continuous control."
        )
    if env.action_spec.kind != "continuous":
        raise ValueError(
            f"{config.task} has action spec {env.action_spec}, which is not "
            f"continuous. This adapter is for torque control."
        )
    if config.privileged:
        sim = env.gym_env.unwrapped
        env = PrivilegedObservation(
            env, sim_state, name="sim_state",
            size=int(sim.model.nq) + int(sim.model.nv),
        )
    # TimeLimit goes below the statistics wrapper so a truncation it raises is
    # counted as an episode end. Observation whitening is not in this stack:
    # it is one estimator over every lane and lives above the vector env, so
    # `EnvConfig.normalize_obs` switches it on.
    if config.max_steps is not None:
        env = TimeLimit(env, config.max_steps)
    if config.record_statistics:
        env = RecordEpisodeStatistics(env)
    if seed is not None:
        env.reset(seed=seed)
    return env
