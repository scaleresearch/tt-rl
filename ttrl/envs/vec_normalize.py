"""Normalisation fitted on ONE estimator shared by every lane.

Both normalisers here sit above the vector environment and hold a
`ttrl.envs.running.RunningMeanStd` that is updated with the whole lane batch
on every step. That placement is the point of the module: a per-lane wrapper
is built inside each lane's factory, so an estimator living there sees 1/N of
the data, describes a distribution the other lanes do not draw from, and
under `AsyncVectorEnv` cannot be shared at all because the lanes are separate
processes.

Observation whitening and reward scaling are one idea used twice, so there is
one estimator and one declaration mechanism, and two wrappers because the
caller names the transform it wants (as with `ScaleReward`, `ClipReward` and
`SignReward`, which are also three wrappers rather than one with a mode).

Host-side numpy. Neither device invariant binds here: nothing on this path
holds a `ttnn` tensor, so there is no `.data` to rebind and no fixed device
address a captured trace replays against, and every number is computed before
`ttrl.train.loop` writes an observation into a device buffer.

`ttrl.train.builders.build_env` is the one place these are applied, from
`EnvConfig`, so the training stack and the eval stack are wrapped the same way
from one statement.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ttrl.envs.base import FINAL_OBS_KEY
from ttrl.envs.running import RunningMeanStd
from ttrl.envs.vec_wrapper import VecWrapper

__all__ = ["FrozenNormalization", "NormalizeObservation", "NormalizeReward",
           "freeze_observation_transform"]


class FrozenNormalization:
    """The whitening a run fitted, as a fixed function of one observation.

    Here rather than in a caller because this is `NormalizeObservation`'s
    transform with the fitting taken out, and the two must not drift: the
    statistics come back through the same `RunningMeanStd`, so the variance
    placeholder for `count <= 1` and the epsilon under the square root are the
    ones that were applied while the weights were trained.

    Frozen is the correct reading of a finished run's statistics, and the only
    available one off a device: `ttrl.train.record` and `ttrl.deploy` drive a
    policy whose weights no longer move, and fitting the estimator further on
    one near-deterministic episode would make the transform a different
    function at the end of a clip than at its start.

    `rows` is lane-major, as everywhere else in this module.
    """

    def __init__(self, items: dict[str, Any]):
        rms = RunningMeanStd(np.shape(items["mean"]))
        rms.set_state(items)
        self.mean = rms.mean
        self.std = rms.std(float(items["epsilon"]))
        self.clip = float(items["clip"])
        self.count = rms.count

    def apply(self, rows) -> np.ndarray:
        raw = np.asarray(rows, dtype=np.float64)
        if raw.shape[1:] != self.mean.shape:
            raise ValueError(
                f"this run's observation transform was fitted on rows of "
                f"shape {self.mean.shape} and was given a batch of shape "
                f"{raw.shape}; a batch is (lanes, *{self.mean.shape})"
            )
        z = (raw - self.mean) / self.std
        return np.clip(z, -self.clip, self.clip).astype(np.float32)


class NormalizeObservation(VecWrapper):
    """Whiten a vector observation with ONE running mean and variance.

    Continuous-control observations mix joint angles with velocities that are
    orders of magnitude larger, so an unnormalised MLP spends its first
    thousands of updates undoing the scale. Image pipelines do not need this:
    `ObsSpec.scale` already folds the 1/255 into the first layer.

    The statistics are one estimator over every lane. They used to be one
    estimator per lane, which made each of them an estimate from 1/N of the
    data, made the scale the policy saw differ lane to lane, and made the
    transform an export carried whichever lane the lead replica held.
    """

    #: The whole transform, not only its statistics: `clip` and `epsilon`
    #: decide what the network's input is just as much as the running mean
    #: does, and both are per-run, so a reader that assumed the defaults would
    #: feed exported weights a different function than they were trained under.
    POLICY_STATE = ("mean", "m2", "count", "clip", "epsilon")
    RUN_STATE = ()

    def __init__(self, env, *, clip: float = 10.0, epsilon: float = 1e-8):
        super().__init__(env)
        group = env.obs_spec
        if len(group) != 1:
            raise ValueError(
                f"NormalizeObservation whitens one observation vector and this "
                f"environment declares {len(group)} leaves {list(group.names)}. "
                f"Leaves differ in kind and scale, so one running estimator "
                f"cannot describe them; normalise inside the environment."
            )
        self._leaf = group.names[0]
        spec = group[self._leaf]
        if spec.kind != "vector":
            raise ValueError(
                f"NormalizeObservation needs a vector observation, got kind "
                f"{spec.kind!r} with shape {spec.shape}"
            )
        if spec.dtype != "float32":
            raise ValueError(
                f"NormalizeObservation needs float32 observations, got "
                f"{spec.dtype}"
            )
        # Whitening the whole vector would rewrite the mask indicators into
        # whatever the running statistics say, and the mask must stay {0, 1}.
        if spec.action_mask:
            raise ValueError(
                f"NormalizeObservation would whiten the legal-action mask in "
                f"the final {spec.action_mask} entries of this observation, "
                f"which must stay 0.0 or 1.0. Normalise the features inside "
                f"the environment and append the mask afterwards."
            )
        if spec.scale != 1.0:
            raise ValueError(
                f"NormalizeObservation was given an observation declaring "
                f"scale {spec.scale}, which the network folds into its first "
                f"weight. Whitening already sets the input scale, so the two "
                f"would compose into neither one. Declare scale 1.0 on a "
                f"normalised observation."
            )
        if not clip > 0.0 or not epsilon > 0.0:
            raise ValueError(
                f"clip and epsilon must be positive, got {clip} and {epsilon}")
        self.clip, self.epsilon = float(clip), float(epsilon)
        self._group = group
        self._rms = RunningMeanStd(spec.shape)
        self._fitting = True

    def freeze(self) -> None:
        """Stop fitting. The statistics as they stand become the transform.

        Scoring and deploying both need this. Fitting continues to change the
        function the weights were trained under, so an evaluation that keeps
        fitting scores a transform that moves between its first episode and
        its last, and reports a number for a function no deployment applies.
        `FrozenNormalization` is the same transform for a caller that has no
        vector stack to hold a wrapper; this is it for a caller that does.

        `set_state` still applies: the statistics a checkpoint carries can be
        loaded either side of this, and only the updating is given up.
        """
        self._fitting = False

    def reset(self, *, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        return self._whiten(obs), info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        whitened = self._whiten(obs)
        # A terminal lane's final observation is the same array this step
        # returned, and a replay buffer stores it as the transition's
        # next_obs. Left raw it would be the one observation in the buffer on
        # a scale the network never sees.
        finals = info[FINAL_OBS_KEY]
        info = {**info, FINAL_OBS_KEY: [
            None if final is None else whitened.select(lane)
            for lane, final in enumerate(finals)
        ]}
        return whitened, rewards, terminated, truncated, info

    def _whiten(self, obs):
        raw = np.asarray(obs[self._leaf], dtype=np.float64)
        if self._fitting:
            self._rms.update(raw)
        z = (raw - self._rms.mean) / self._rms.std(self.epsilon)
        return self._group.observation(
            {self._leaf: np.clip(z, -self.clip, self.clip).astype(np.float32)})

    def get_state(self) -> dict[str, Any]:
        return {**self._rms.get_state(),
                "clip": self.clip, "epsilon": self.epsilon}

    def set_state(self, state: dict[str, Any]) -> None:
        self._rms.set_state(state)
        self.clip = float(state["clip"])
        self.epsilon = float(state["epsilon"])


def freeze_observation_transform(env) -> list[str]:
    """Freeze every observation normaliser above this vector env's lanes.

    Returns the layers frozen, so a caller can assert on a stack rather than
    hope. The walk is over the live `VecWrapper` chain, which is where these
    estimators live: a per-lane layer could not hold one (see this module's
    docstring), so there is nothing below to reach.
    """
    frozen = []
    layer = env
    while isinstance(layer, VecWrapper):
        if isinstance(layer, NormalizeObservation):
            layer.freeze()
            frozen.append(type(layer).__name__)
        layer = layer.env
    return frozen


class NormalizeReward(VecWrapper):
    """Divide reward by the running standard deviation of the return.

    PPO's advantage scale is the reward scale, and a loss whose advantages are
    two orders of magnitude from its entropy bonus is tuned by the reward
    units rather than by its hyperparameters. The standard treatment divides
    every reward by the running standard deviation of the DISCOUNTED return,
    with the discount being the algorithm's own gamma, so the quantity
    normalised is the one the critic regresses on.

    The mean is deliberately not subtracted. Centering a reward changes which
    trajectories are better: a survival bonus of +1 per step becomes 0, and
    an agent paid nothing to survive stops surviving. Only the scale is taken.

    `clip` bounds the result. On a run whose early returns are nearly constant
    the running deviation is tiny, and one step can otherwise arrive at the
    loss thousands of times the size of every other.

    The environment's OWN undiscounted return is untouched by this: it is
    accumulated by `RecordEpisodeStatistics` inside each lane, below this
    wrapper, and reported through `info`, which this never rewrites. That is
    what `ttrl.train.evaluate` reads a score from.
    """

    POLICY_STATE = ()
    #: Run state, not policy state: the weights compute the same function
    #: whatever the reward was divided by, so an exported policy artifact
    #: carries none of this, while a resume must continue it or the loss
    #: changes scale under a run that reports itself as continuing.
    RUN_STATE = ("mean", "m2", "count", "gamma", "clip", "epsilon")

    def __init__(self, env, *, gamma: float, clip: float,
                 epsilon: float = 1e-8):
        super().__init__(env)
        if not 0.0 < gamma <= 1.0:
            raise ValueError(
                f"reward normalisation discounts the return with the "
                f"algorithm's gamma, which must be in (0, 1], got {gamma}")
        if not clip > 0.0 or not epsilon > 0.0:
            raise ValueError(
                f"clip and epsilon must be positive, got {clip} and {epsilon}")
        self.gamma, self.clip = float(gamma), float(clip)
        self.epsilon = float(epsilon)
        self._rms = RunningMeanStd(())
        self._returns = np.zeros(env.num_envs, dtype=np.float64)
        #: Lanes that ended on the PREVIOUS step. Under NEXT_STEP autoreset
        #: those lanes are being reset on this one, so their discounted return
        #: starts again here rather than carrying across the boundary.
        self._ended = np.zeros(env.num_envs, dtype=bool)

    def reset(self, *, seed: int | None = None):
        self._returns[:] = 0.0
        self._ended[:] = False
        return self.env.reset(seed=seed)

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        raw = np.asarray(rewards, dtype=np.float64)
        self._returns = self._returns * self.gamma * (~self._ended) + raw
        self._rms.update(self._returns)
        scaled = np.clip(raw / self._rms.std(self.epsilon),
                         -self.clip, self.clip).astype(np.float32)
        self._ended = (np.asarray(terminated, dtype=bool)
                       | np.asarray(truncated, dtype=bool))
        return obs, scaled, terminated, truncated, info

    def get_state(self) -> dict[str, Any]:
        return {**self._rms.get_state(), "gamma": self.gamma,
                "clip": self.clip, "epsilon": self.epsilon,
                "returns": self._returns.copy(), "ended": self._ended.copy()}

    def set_state(self, state: dict[str, Any]) -> None:
        self._rms.set_state(state)
        self.gamma = float(state["gamma"])
        self.clip = float(state["clip"])
        self.epsilon = float(state["epsilon"])
        self._returns[:] = state["returns"]
        self._ended[:] = state["ended"]
