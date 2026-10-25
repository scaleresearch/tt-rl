"""Domain randomisation for the MuJoCo locomotion tasks, in user code.

A policy trained on one simulator fits that simulator's masses, frictions and
gear ratios. The robot has different ones, so the usual fix is to resample
them every episode and train a policy that works across the range.

This lives in the example rather than in `ttrl` on purpose. Everything it
touches is `mujoco`'s own model struct: `body_mass`, `dof_damping`,
`geom_friction`, `actuator_gear`. A library helper would either take those
field names as strings, which is this file with extra indirection, or grow a
table of per-suite field names, which is the branch on which environment
suite produced the spec that the whole `ttrl.envs` contract exists to avoid.

What makes it fully a first-class environment rather than a script is the
registry: `ttrl.envs.register` gives the builder a name, and
`ttrl.EnvConfig(name=..., kwargs=...)` carries only plain numbers, so the run
config stays checkpointable and `--resume` reproduces the same randomisation
ranges. That seam is the library's contribution; the field names are the
user's.

Ranges are multiplicative and log-uniform, which is the convention the
sim-to-real literature uses: a mass range of 1.5 means every body mass is
drawn in [m/1.5, m*1.5] with no bias toward heavier or lighter.
"""

from __future__ import annotations

import numpy as np

import ttrl
from ttrl.envs.wrappers import Wrapper

#: Model fields this randomises, and the keyword naming each range. Every one
#: is a positive quantity, so one multiplicative rule covers them all.
FIELDS = {
    "body_mass": "mass",
    "dof_damping": "damping",
    "geom_friction": "friction",
    "actuator_gear": "gear",
}


class DomainRandomization(Wrapper):
    """Resample the simulator's physical parameters at every reset.

    The nominal values are copied once at construction and every episode's
    values are drawn from those, so the ranges do not compound over a run.
    """

    #: Nothing here transforms the observation, so none of its state travels
    #: with the weights. See POLICY_STATE in ttrl.envs.wrappers.
    POLICY_STATE = ()

    def __init__(self, env, *, seed=None, **ranges):
        super().__init__(env)
        unknown = sorted(set(ranges) - set(FIELDS.values()))
        if unknown:
            raise ValueError(
                f"unknown randomisation range(s) {unknown}; this wrapper "
                f"randomises {sorted(FIELDS.values())}"
            )
        if any(r < 1.0 for r in ranges.values()):
            raise ValueError(
                f"a multiplicative range is >= 1.0 (1.0 randomises nothing), "
                f"got {ranges}"
            )
        self._model = self.unwrapped.gym_env.unwrapped.model
        self._ranges = {field: float(ranges.get(key, 1.0))
                        for field, key in FIELDS.items()}
        self._nominal = {f: getattr(self._model, f).copy() for f in self._ranges}
        self._rng = np.random.default_rng(seed)

    def sample(self) -> None:
        """Draw one episode's physics into the live model."""
        for field, span in self._ranges.items():
            nominal = self._nominal[field]
            if span == 1.0:
                getattr(self._model, field)[...] = nominal
                continue
            log = np.log(span)
            factor = np.exp(self._rng.uniform(-log, log, size=nominal.shape))
            getattr(self._model, field)[...] = nominal * factor

    def reset(self, *, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Before the reset, so the first observation of the episode is already
        # the observation of the randomised robot.
        self.sample()
        return self.env.reset(seed=seed)


def make_randomized_mujoco(*, seed=None, mass=1.0, damping=1.0, friction=1.0,
                           gear=1.0, **mujoco_kwargs):
    """A `ttrl` MuJoCo env whose physics is resampled every episode.

    Every keyword is a plain number, which is what keeps a `TrainConfig`
    naming this builder checkpointable and resumable.
    """
    env = ttrl.envs.make_mujoco(seed=seed, **mujoco_kwargs)
    return DomainRandomization(env, seed=seed, mass=mass, damping=damping,
                               friction=friction, gear=gear)


#: Registered by import path rather than by object, so a spawned vector-env
#: worker resolves it by importing this module instead of pickling a closure.
ttrl.envs.register("mujoco-randomized", "randomize:make_randomized_mujoco")
